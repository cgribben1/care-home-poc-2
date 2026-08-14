/**
 * Care-home acoustic monitor — on-device inference over WiFi
 * XIAO ESP32S3 Plus + Adafruit SPH0645LM4H
 *
 * Wiring: DOUT->D0  BCLK->D1  LRCL->D2  3V->3V3  GND->GND  SEL->GND
 *
 * Pipeline:
 *   I2S → RMS gate → log-mel spectrogram → CNN (manual C++ forward pass) → HTTP POST
 *
 * Run before building:
 *   python generate_model_header.py   (generates firmware/include/model_weights.h)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <driver/i2s.h>
#include <esp_dsp.h>
#include <math.h>

#include "wifi_config.h"
#include "mel_filterbank.h"
#include "model_weights.h"   // weights + cnn_infer() — no TFLite dependency

// ── I2S pins ────────────────────────────────────────────────────────────────
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

// ── Audio / mel parameters — must match train_cnn.py ────────────────────────
static const int kSampleRate    = 16000;
static const int kAudioSamples  = 16000;
static const int kNFft          = MEL_N_FFT;
static const int kFftBins       = MEL_FFT_BINS;
static const int kNMels         = MEL_N_MELS;
static const int kNFrames       = MEL_N_FRAMES;
static const int kHop           = MEL_HOP;
static const int kDmaChunk      = 320;

static const float kRmsThreshold = 0.025f;

// ── Detection thresholds ─────────────────────────────────────────────────────
static const int   kNClasses = 4;
static const char* kClassNames[kNClasses]  = {"fall", "cough", "normal", "other"};
static const float kThresholds[kNClasses]  = {0.88f, 0.85f, 0.0f, 0.0f};
static const float kNonAlertMaxCombined    = 0.40f;
static const float kNonAlertMargin         = 0.35f;

// ── CNN scratch buffers (PSRAM) ──────────────────────────────────────────────
// buf_a needs ≥ 198,656 floats (Conv0 output: 97×64×32)
// buf_b needs ≥  49,152 floats (MaxPool0 output: 48×32×32)
static float* cnn_buf_a = nullptr;
static float* cnn_buf_b = nullptr;

// ── Audio buffers (PSRAM) ────────────────────────────────────────────────────
static int32_t* dma_buf   = nullptr;
static float*   audio_buf = nullptr;
static float*   fft_buf   = nullptr;
static float*   mel_buf   = nullptr;
static float*   hann_win  = nullptr;

// ── Timing ───────────────────────────────────────────────────────────────────
static unsigned long last_heartbeat_ms = 0;

// ─────────────────────────────────────────────────────────────────────────────
// WiFi helpers
// ─────────────────────────────────────────────────────────────────────────────

static void wifi_connect() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.print("WiFi connecting");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 30) {
    delay(500);
    Serial.print('.');
    retries++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nWiFi OK  IP=%s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\nWiFi FAILED — will retry");
  }
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
  char body[128];
  snprintf(body, sizeof(body),
    "{\"class\":\"%s\",\"confidence\":%.3f,\"device_id\":\"" DEVICE_ID "\"}",
    cls, conf);
  Serial.printf("[ALERT] %s  %.1f%%\n", cls, conf * 100.0f);
  http_post("/event", body);
}

static void post_heartbeat() {
  char body[64];
  snprintf(body, sizeof(body), "{\"device_id\":\"" DEVICE_ID "\"}");
  http_post("/heartbeat", body);
}

// ─────────────────────────────────────────────────────────────────────────────
// I2S
// ─────────────────────────────────────────────────────────────────────────────

static bool init_i2s() {
  i2s_config_t cfg = {
    .mode              = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate       = (uint32_t)kSampleRate,
    .bits_per_sample   = I2S_BITS_PER_SAMPLE_32BIT,
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

static bool collect_audio() {
  int collected = 0;
  const int read_bytes = kDmaChunk * 2 * sizeof(int32_t);
  while (collected < kAudioSamples) {
    size_t bytes_read = 0;
    i2s_read(I2S_NUM_0, dma_buf, read_bytes, &bytes_read, pdMS_TO_TICKS(500));
    int pairs    = (int)(bytes_read / (2 * sizeof(int32_t)));
    int to_copy  = min(pairs, kAudioSamples - collected);
    for (int i = 0; i < to_copy; i++) {
      int32_t raw    = dma_buf[i * 2];
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
      int   src = start + k;
      float s   = (src < kAudioSamples) ? audio_buf[src] : 0.0f;
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
// CNN inference (manual forward pass — no TFLite)
// ─────────────────────────────────────────────────────────────────────────────

static int run_inference(float probs_out[kNClasses]) {
  cnn_infer(mel_buf, cnn_buf_a, cnn_buf_b, probs_out);

  float non_alert_sum = probs_out[2] + probs_out[3];

  int   best_cls   = -1;
  float best_score = 0.0f;
  for (int i = 0; i < kNClasses; i++) {
    if (i == 2 || i == 3) continue;
    float score = probs_out[i];
    if (score >= kThresholds[i]
     && score >= non_alert_sum + kNonAlertMargin
     && non_alert_sum <= kNonAlertMaxCombined
     && score > best_score) {
      best_score = score;
      best_cls   = i;
    }
  }
  return best_cls;
}

// ─────────────────────────────────────────────────────────────────────────────
// setup / loop
// ─────────────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 8000) { delay(10); }
  Serial.println("Care-home monitor booting...");

  auto xmalloc = [](size_t n) -> void* {
    void* p = ps_malloc(n); if (!p) p = malloc(n); return p;
  };

  // CNN scratch buffers in PSRAM
  cnn_buf_a = (float*)ps_malloc(200000 * sizeof(float));  // 775 KB
  cnn_buf_b = (float*)ps_malloc( 50000 * sizeof(float));  // 192 KB

  // Audio buffers
  dma_buf   = (int32_t*)xmalloc(kDmaChunk * 2 * sizeof(int32_t));
  audio_buf = (float*)  xmalloc(kAudioSamples * sizeof(float));
  fft_buf   = (float*)  xmalloc(kNFft * 2 * sizeof(float));
  mel_buf   = (float*)  xmalloc(kNFrames * kNMels * sizeof(float));
  hann_win  = (float*)  xmalloc(kNFft * sizeof(float));

  Serial.printf("PSRAM free: %d bytes\n", (int)ESP.getFreePsram());
  if (!cnn_buf_a || !cnn_buf_b || !dma_buf || !audio_buf || !fft_buf || !mel_buf || !hann_win) {
    Serial.println("ERROR: PSRAM allocation failed");
    while (true) delay(1000);
  }

  // Hann window
  for (int i = 0; i < kNFft; i++)
    hann_win[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (kNFft - 1)));

  // ESP-DSP FFT tables (overwrites hann_win as scratch)
  Serial.println("Init: ESP-DSP FFT...");
  dsps_fft2r_init_fc32(hann_win, kNFft);
  for (int i = 0; i < kNFft; i++)
    hann_win[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (kNFft - 1)));

  Serial.println("Init: CNN model (no TFLite)...");
  Serial.printf("  cnn_buf_a=%p  cnn_buf_b=%p\n", cnn_buf_a, cnn_buf_b);
  Serial.println("  Weights in flash — no init needed");

  if (!init_i2s()) {
    Serial.println("ERROR: I2S init failed");
    while (true) delay(1000);
  }
  Serial.println("I2S ready");

  WiFi.mode(WIFI_STA);
  wifi_connect();

  Serial.println("Ready — listening...");
}

void loop() {
  unsigned long now = millis();
  if (now - last_heartbeat_ms >= HEARTBEAT_MS) {
    post_heartbeat();
    last_heartbeat_ms = now;
  }

  collect_audio();

  float rms = compute_rms(audio_buf, kAudioSamples);
  if (rms < kRmsThreshold) return;

  compute_log_mel();

  unsigned long t_infer = millis();
  float probs[kNClasses];
  int alert_cls = run_inference(probs);
  Serial.printf("Inference: %lu ms  RMS=%.4f  fall=%.2f cough=%.2f norm=%.2f other=%.2f",
    millis() - t_infer, rms, probs[0], probs[1], probs[2], probs[3]);

  if (alert_cls >= 0) {
    Serial.printf("  → ALERT: %s", kClassNames[alert_cls]);
    post_alert(kClassNames[alert_cls], probs[alert_cls]);
  }
  Serial.println();
}
