/**
 * Care-home acoustic monitor — on-device inference over WiFi
 * XIAO ESP32S3 Plus + Adafruit SPH0645LM4H
 *
 * Wiring: DOUT->D6(GPIO43)  BCLK->D4(GPIO5)  LRCL->D5(GPIO6)  3V->3V3  GND->GND  SEL->GND
 *
 * Pipeline:
 *   I2S → RMS gate → log-mel spectrogram → TFLite INT8 CNN → HTTP POST
 *
 * Run before building:
 *   python convert_tflite.py   (generates firmware/include/model_data.h + mel_filterbank.h)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <driver/i2s.h>
#include <esp_dsp.h>
#include <math.h>

// TFLite Micro runtime (from tflm_esp32 library)
// tflm_esp32.h pulls in micro_interpreter.h, micro_mutable_op_resolver.h,
// system_setup.h, and schema_generated.h
#include <esp_heap_caps.h>
#include <tflm_esp32.h>

#include "wifi_config.h"
#include "model_data.h"
#include "mel_filterbank.h"

// ── I2S pins ─────────────────────────────────────────────────────────────────
#ifndef PIN_I2S_SD
#define PIN_I2S_SD  1
#endif
#ifndef PIN_I2S_SCK
#define PIN_I2S_SCK 2
#endif
#ifndef PIN_I2S_WS
#define PIN_I2S_WS  3
#endif
#define PCM_SHIFT 15

// ── Audio / mel parameters ────────────────────────────────────────────────────
static const int kSampleRate   = 16000;
static const int kAudioSamples = MEL_N_SAMPLES;  // 16384 (1.024s)
static const int kNFft         = MEL_N_FFT;
static const int kFftBins      = MEL_FFT_BINS;
static const int kNMels        = MEL_N_MELS;
static const int kNFrames      = MEL_N_FRAMES;
static const int kHop          = MEL_HOP;
static const int kDmaChunk     = 320;
static const float kRmsThreshold = 0.025f;

// ── Detection thresholds ──────────────────────────────────────────────────────
static const int   kNClasses = 4;
static const char* kClassNames[kNClasses] = {"fall", "cough", "normal", "other"};
static const float kThresholds[kNClasses] = {0.88f, 0.85f, 0.0f, 0.0f};
static const float kNonAlertMaxCombined   = 0.40f;
static const float kNonAlertMargin        = 0.35f;

// ── TFLite Micro (PSRAM arena) ────────────────────────────────────────────────
// Arena in PSRAM: avoids DRAM pressure; INT8 model needs ~218 KB peak.
// resolver is a static local in setup() to avoid global-constructor crashes.
static const int kArenaSize = 1 * 1024 * 1024;  // 1 MB in PSRAM (§5.1)
static uint8_t*                  tensor_arena  = nullptr;
static tflite::MicroInterpreter* interpreter   = nullptr;
static TfLiteTensor*             input_tensor  = nullptr;
static TfLiteTensor*             output_tensor = nullptr;

// ── Audio buffers (PSRAM) ─────────────────────────────────────────────────────
static int32_t* dma_buf   = nullptr;
static float*   audio_buf = nullptr;
static float*   fft_buf   = nullptr;
static float*   mel_buf   = nullptr;
static float*   hann_win  = nullptr;

static unsigned long last_heartbeat_ms = 0;
static const unsigned long kHeartbeatInterval = 30000UL;  // §6.3: 30-second heartbeat

// ─────────────────────────────────────────────────────────────────────────────
// WiFi helpers
// ─────────────────────────────────────────────────────────────────────────────

static void wifi_connect() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.print("WiFi connecting");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 30) {
    delay(500); Serial.print('.'); retries++;
  }
  if (WiFi.status() == WL_CONNECTED)
    Serial.printf("\nWiFi OK  IP=%s\n", WiFi.localIP().toString().c_str());
  else
    Serial.println("\nWiFi FAILED — will retry");
}

static void http_post(const char* path, const char* json_body) {
  if (WiFi.status() != WL_CONNECTED) {
    wifi_connect();
    if (WiFi.status() != WL_CONNECTED) return;
  }
  char url[64];
  snprintf(url, sizeof(url), "http://" SERVER_IP ":%d%s", SERVER_PORT, path);
  HTTPClient http;
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(json_body);
  if (code < 0) Serial.printf("HTTP POST %s failed: %d\n", path, code);
  http.end();
}

static void post_alert(const char* cls, float conf) {
  char body[192];
  snprintf(body, sizeof(body),
    "{\"device_id\":\"" DEVICE_ID "\","
    "\"event_type\":\"%s\","
    "\"confidence\":%.3f,"
    "\"uptime_ms\":%lu,"
    "\"rssi_dbm\":%d}",
    cls, conf, millis(), (int)WiFi.RSSI());
  Serial.printf("[ALERT] %s  %.1f%%\n", cls, conf * 100.0f);
  http_post("/api/alert", body);
}

static void post_heartbeat() {
  char body[96];
  snprintf(body, sizeof(body),
    "{\"device_id\":\"" DEVICE_ID "\","
    "\"status\":\"online\","
    "\"rssi_dbm\":%d}",
    (int)WiFi.RSSI());
  http_post("/api/heartbeat", body);
}

static void post_telemetry(float rms, bool ran_inference,
                           const float* probs, unsigned long infer_ms,
                           const char* alert_cls) {
  char body[256];
  if (ran_inference && probs) {
    if (alert_cls) {
      snprintf(body, sizeof(body),
        "{\"device_id\":\"" DEVICE_ID "\",\"rms\":%.4f"
        ",\"probs\":[%.3f,%.3f,%.3f,%.3f]"
        ",\"inference_ms\":%lu,\"alert\":\"%s\"}",
        rms, probs[0], probs[1], probs[2], probs[3], infer_ms, alert_cls);
    } else {
      snprintf(body, sizeof(body),
        "{\"device_id\":\"" DEVICE_ID "\",\"rms\":%.4f"
        ",\"probs\":[%.3f,%.3f,%.3f,%.3f]"
        ",\"inference_ms\":%lu,\"alert\":null}",
        rms, probs[0], probs[1], probs[2], probs[3], infer_ms);
    }
  } else {
    snprintf(body, sizeof(body),
      "{\"device_id\":\"" DEVICE_ID "\",\"rms\":%.4f}", rms);
  }
  http_post("/telemetry", body);
}

// ─────────────────────────────────────────────────────────────────────────────
// I2S
// ─────────────────────────────────────────────────────────────────────────────

static bool init_i2s() {
  i2s_config_t cfg = {
    .mode              = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate       = (uint32_t)kSampleRate,
    .bits_per_sample   = I2S_BITS_PER_SAMPLE_32BIT,
    // Stereo RIGHT_LEFT: ESP32-S3 old API doesn't fill DMA in ONLY_LEFT/mono mode.
    // With RIGHT_LEFT, DMA is [R0, L0, R1, L1, ...].
    // SPH0645 SEL=GND → outputs during WS=LOW (LEFT I2S channel) → ODD indices.
    .channel_format    = I2S_CHANNEL_FMT_RIGHT_LEFT,
#if defined(I2S_COMM_FORMAT_I2S_MSB)
    .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
#else
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
#endif
    .intr_alloc_flags  = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count     = 8,
    .dma_buf_len       = 256,
    .use_apll          = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk        = 0,
  };
  i2s_pin_config_t pins = {
    .mck_io_num   = I2S_PIN_NO_CHANGE,
    .bck_io_num   = PIN_I2S_SCK,
    .ws_io_num    = PIN_I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = PIN_I2S_SD,
  };
  if (i2s_driver_install(I2S_NUM_0, &cfg, 0, nullptr) != ESP_OK) return false;
  if (i2s_set_pin(I2S_NUM_0, &pins)                   != ESP_OK) return false;
  if (i2s_set_clk(I2S_NUM_0, kSampleRate, I2S_BITS_PER_SAMPLE_32BIT,
                  I2S_CHANNEL_STEREO)                  != ESP_OK) return false;
  i2s_zero_dma_buffer(I2S_NUM_0);
  return true;
}

static void detect_channel() {
  // Mono mode — no interleave detection needed.
  // Warm up DMA with a few reads and print a sample raw value for debugging.
  const int read_bytes = kDmaChunk * sizeof(int32_t);
  size_t bytes_read = 0;
  for (int round = 0; round < 4; round++)
    i2s_read(I2S_NUM_0, dma_buf, read_bytes, &bytes_read, pdMS_TO_TICKS(200));
  Serial.printf("I2S mono warm-up done, sample raw[0]=0x%08lx\n", (unsigned long)dma_buf[0]);
}

static bool collect_audio() {
  int collected = 0;
  // Stereo: read kDmaChunk pairs (R, L interleaved).
  // SPH0645 with this wiring outputs on EVEN indices (R hardware channel).
  const int read_bytes = kDmaChunk * 2 * sizeof(int32_t);
  while (collected < kAudioSamples) {
    size_t bytes_read = 0;
    i2s_read(I2S_NUM_0, dma_buf, read_bytes, &bytes_read, pdMS_TO_TICKS(500));
    int n_pairs = (int)(bytes_read / sizeof(int32_t)) / 2;
    int to_copy = min(n_pairs, kAudioSamples - collected);
    for (int i = 0; i < to_copy; i++) {
      int32_t raw    = dma_buf[i * 2];  // even = R hardware channel = mic signal
      int16_t sample = (int16_t)(raw >> PCM_SHIFT);
      audio_buf[collected + i] = sample / 32768.0f;
    }
    collected += to_copy;
  }
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// RMS
// ─────────────────────────────────────────────────────────────────────────────

static float compute_rms(const float* buf, int n) {
  float dc = 0.0f;
  for (int i = 0; i < n; i++) dc += buf[i];
  dc /= n;
  float sum = 0.0f;
  for (int i = 0; i < n; i++) { float s = buf[i] - dc; sum += s * s; }
  return sqrtf(sum / n);
}

// ─────────────────────────────────────────────────────────────────────────────
// Log-mel spectrogram
// ─────────────────────────────────────────────────────────────────────────────

static void compute_log_mel() {
  float max_power = 1e-10f;
  for (int frame = 0; frame < kNFrames; frame++) {
    int start = frame * kHop;
    for (int k = 0; k < kNFft; k++) {
      int src = start + k;
      float s = (src < kAudioSamples) ? audio_buf[src] : 0.0f;
      fft_buf[2 * k]     = s * hann_win[k];
      fft_buf[2 * k + 1] = 0.0f;
    }
    dsps_fft2r_fc32(fft_buf, kNFft);
    dsps_bit_rev_fc32(fft_buf, kNFft);

    float power[MEL_FFT_BINS];
    for (int k = 0; k < kFftBins; k++) {
      float re = fft_buf[2 * k], im = fft_buf[2 * k + 1];
      power[k] = re * re + im * im;
    }
    float* row = mel_buf + frame * kNMels;
    for (int m = 0; m < kNMels; m++) {
      float val = 0.0f;
      for (int k = 0; k < kFftBins; k++) val += mel_filterbank[m][k] * power[k];
      row[m] = val;
      if (val > max_power) max_power = val;
    }
  }
  const float ref_db = 10.0f * log10f(max_power);
  for (int i = 0; i < kNFrames * kNMels; i++) {
    float db = 10.0f * log10f(mel_buf[i] + 1e-10f) - ref_db;
    if (db < -80.0f) db = -80.0f;
    mel_buf[i] = (db - NORM_MEAN) / NORM_STD;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inference
// ─────────────────────────────────────────────────────────────────────────────

static int run_inference(float probs_out[kNClasses]) {
  // Copy mel into input tensor — handles both float32 and INT8 IO models
  if (input_tensor->type == kTfLiteFloat32) {
    memcpy(input_tensor->data.f, mel_buf, kNFrames * kNMels * sizeof(float));
  } else {
    // INT8 input: quantize float → int8 using tensor's scale/zero_point
    float   scale = input_tensor->params.scale;
    int32_t zp    = input_tensor->params.zero_point;
    for (int i = 0; i < kNFrames * kNMels; i++) {
      int32_t v = (int32_t)roundf(mel_buf[i] / scale) + zp;
      if (v < -128) v = -128;
      if (v >  127) v =  127;
      input_tensor->data.int8[i] = (int8_t)v;
    }
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke FAILED");
    return -1;
  }

  // Read output probabilities — handles both float32 and INT8 IO models
  if (output_tensor->type == kTfLiteFloat32) {
    for (int i = 0; i < kNClasses; i++) probs_out[i] = output_tensor->data.f[i];
  } else {
    float   scale = output_tensor->params.scale;
    int32_t zp    = output_tensor->params.zero_point;
    for (int i = 0; i < kNClasses; i++)
      probs_out[i] = (output_tensor->data.int8[i] - zp) * scale;
  }

  float non_alert  = probs_out[2] + probs_out[3];
  int   best_cls   = -1;
  float best_score = 0.0f;
  for (int i = 0; i < kNClasses; i++) {
    if (i == 2 || i == 3) continue;
    float score = probs_out[i];
    if (score >= kThresholds[i]
     && score >= non_alert + kNonAlertMargin
     && non_alert <= kNonAlertMaxCombined
     && score > best_score) {
      best_score = score; best_cls = i;
    }
  }
  return best_cls;
}

// ─────────────────────────────────────────────────────────────────────────────
// setup / loop
// ─────────────────────────────────────────────────────────────────────────────

#define DIAG(msg) do { Serial.println(msg); Serial.flush(); } while(0)
#define DIAGF(fmt, ...) do { Serial.printf(fmt, ##__VA_ARGS__); Serial.flush(); } while(0)

void setup() {
  Serial.begin(115200);
  delay(3000);  // plain delay — while(!Serial) can deadlock on ESP32-S3 CDC
  Serial.println("\n\n=== BOOT ==="); Serial.flush();

  DIAGF("Chip: %s  Rev: %d  Cores: %d  Freq: %d MHz\n",
        ESP.getChipModel(), ESP.getChipRevision(),
        ESP.getChipCores(), ESP.getCpuFreqMHz());
  DIAGF("Flash: %d KB  PSRAM: %d KB  Heap: %d KB\n",
        (int)(ESP.getFlashChipSize() / 1024),
        (int)(ESP.getPsramSize()     / 1024),
        (int)(ESP.getFreeHeap()      / 1024));
  DIAGF("model_data_len=%u bytes  arena=%d KB\n",
        model_data_len, kArenaSize / 1024);

  DIAG("Step 1: InitializeTarget");
  tflite::InitializeTarget();

  DIAG("Step 2: allocate buffers");
  auto xmalloc = [](size_t n) -> void* {
    void* p = ps_malloc(n); if (!p) p = malloc(n); return p;
  };

  dma_buf      = (int32_t*)xmalloc(kDmaChunk * 2 * sizeof(int32_t));
  audio_buf    = (float*)  xmalloc(kAudioSamples * sizeof(float));
  fft_buf      = (float*)  xmalloc(kNFft * 2 * sizeof(float));
  mel_buf      = (float*)  xmalloc(kNFrames * kNMels * sizeof(float));
  hann_win     = (float*)  xmalloc(kNFft * sizeof(float));
  DIAGF("  dma_buf=%p  audio_buf=%p  fft_buf=%p  mel_buf=%p  hann_win=%p\n",
        dma_buf, audio_buf, fft_buf, mel_buf, hann_win);

  DIAG("Step 3: allocate tensor arena (PSRAM)");
  tensor_arena = (uint8_t*)heap_caps_malloc(kArenaSize,
                              MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  DIAGF("  tensor_arena=%p  PSRAM free after: %d KB\n",
        tensor_arena, (int)(ESP.getFreePsram() / 1024));

  if (!dma_buf || !audio_buf || !fft_buf || !mel_buf || !hann_win || !tensor_arena) {
    DIAG("ERROR: memory allocation failed — halting");
    while (true) delay(1000);
  }

  DIAG("Step 4: Hann window");
  for (int i = 0; i < kNFft; i++)
    hann_win[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (kNFft - 1)));

  DIAG("Step 5: ESP-DSP FFT init");
  dsps_fft2r_init_fc32(hann_win, kNFft);
  for (int i = 0; i < kNFft; i++)
    hann_win[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (kNFft - 1)));

  DIAG("Step 6: GetModel");
  const tflite::Model* model = tflite::GetModel(model_data);
  DIAGF("  model=%p  version=%d  expected=%d\n",
        model, model ? (int)model->version() : -1, TFLITE_SCHEMA_VERSION);
  if (!model || model->version() != TFLITE_SCHEMA_VERSION) {
    DIAG("ERROR: bad model schema — halting");
    while (true) delay(1000);
  }

  DIAG("Step 7: op resolver");
  static tflite::MicroMutableOpResolver<30> static_resolver;
  static_resolver.AddConv2D();
  static_resolver.AddReshape();
  static_resolver.AddAdd();
  static_resolver.AddFullyConnected();
  static_resolver.AddBatchMatMul();
  static_resolver.AddSoftmax();
  static_resolver.AddTranspose();
  static_resolver.AddMul();
  static_resolver.AddMean();
  static_resolver.AddRsqrt();
  static_resolver.AddSub();
  static_resolver.AddQuantize();
  static_resolver.AddDequantize();
  static_resolver.AddStridedSlice();
  static_resolver.AddExpandDims();
  static_resolver.AddRelu6();
  static_resolver.AddShape();
  static_resolver.AddPack();
  static_resolver.AddSplit();
  static_resolver.AddConcatenation();
  static_resolver.AddGather();
  static_resolver.AddUnpack();
  static_resolver.AddNeg();
  static_resolver.AddSquaredDifference();
  DIAG("  ops registered");

  DIAG("Step 8: MicroInterpreter");
  static tflite::MicroInterpreter static_interpreter(
    model, static_resolver, tensor_arena, kArenaSize);
  interpreter = &static_interpreter;
  DIAG("  interpreter created");

  DIAG("Step 9: AllocateTensors");
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  DIAGF("  arena used: %d / %d bytes  status=%d\n",
        (int)interpreter->arena_used_bytes(), kArenaSize, (int)alloc_status);
  if (alloc_status != kTfLiteOk) {
    DIAG("ERROR: AllocateTensors failed — halting");
    while (true) delay(1000);
  }
  DIAG("  TFLite ready");

  input_tensor  = interpreter->input(0);
  output_tensor = interpreter->output(0);
  DIAGF("  Input  type=%d  shape=[%d,%d,%d,%d]\n",
        input_tensor->type,
        input_tensor->dims->data[0], input_tensor->dims->data[1],
        input_tensor->dims->data[2], input_tensor->dims->data[3]);
  DIAGF("  Output type=%d  shape=[%d,%d]\n",
        output_tensor->type,
        output_tensor->dims->data[0], output_tensor->dims->data[1]);

  DIAG("Step 10: I2S init");
  if (!init_i2s()) {
    DIAG("ERROR: I2S init failed — halting");
    while (true) delay(1000);
  }
  DIAG("  I2S ready");
  detect_channel();

  DIAG("Step 11: WiFi");
  WiFi.mode(WIFI_STA);
  wifi_connect();

  DIAG("=== READY — listening ===");
}

void loop() {
  unsigned long now = millis();
  if (now - last_heartbeat_ms >= kHeartbeatInterval) {
    post_heartbeat();
    last_heartbeat_ms = now;
  }

  collect_audio();

  // Diagnostic: print raw DMA for both channels every 3 s
  // dma_buf is [R0, L0, R1, L1, ...] — R=even, L=odd (SPH0645 with SEL=GND → L)
  static unsigned long last_diag = 0;
  if (millis() - last_diag > 3000) {
    Serial.printf("R(even): %08lx %08lx  L(odd): %08lx %08lx  audio[L]: %.5f %.5f\n",
      (unsigned long)dma_buf[0], (unsigned long)dma_buf[2],
      (unsigned long)dma_buf[1], (unsigned long)dma_buf[3],
      audio_buf[0], audio_buf[1]);
    last_diag = millis();
  }

  float rms = compute_rms(audio_buf, kAudioSamples);
  Serial.printf("RMS=%.5f\n", rms);
  if (rms < kRmsThreshold) {
    post_telemetry(rms, false, nullptr, 0, nullptr);
    return;
  }

  compute_log_mel();

  unsigned long t_infer = millis();
  float probs[kNClasses];
  int alert_cls = run_inference(probs);
  unsigned long infer_ms = millis() - t_infer;

  const char* alert_name = (alert_cls >= 0) ? kClassNames[alert_cls] : nullptr;
  Serial.printf("Inference: %lu ms  fall=%.2f cough=%.2f norm=%.2f other=%.2f",
    infer_ms, probs[0], probs[1], probs[2], probs[3]);

  post_telemetry(rms, true, probs, infer_ms, alert_name);

  if (alert_cls >= 0) {
    Serial.printf("  → ALERT: %s", kClassNames[alert_cls]);
    post_alert(kClassNames[alert_cls], probs[alert_cls]);
  }
  Serial.println();
}
