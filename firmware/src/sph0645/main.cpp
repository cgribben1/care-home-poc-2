/**
 * XIAO ESP32S3 Plus + Adafruit SPH0645LM4H I2S MEMS mic
 *
 * Wiring (Adafruit breakout -> XIAO):
 *   3V   -> 3V3
 *   GND  -> GND
 *   SEL  -> GND (left channel) or 3V3 (right channel)
 * Wiring option A (default env sph0645):
 *   DOUT -> D0 | BCLK -> D1 | LRCL -> D2
 *
 * Wiring option B (env sph0645_d456 — common Gemini suggestion):
 *   BCLK -> D4 | LRCL -> D5 | DOUT -> D6
 *
 * Streams 16 kHz mono PCM16 over USB serial in framed packets.
 * Python: serial_bridge.py --port COM3
 */

#include <Arduino.h>
#include <driver/i2s.h>

#ifndef PIN_I2S_SD
#define PIN_I2S_SD 1  // D0 — DOUT
#endif
#ifndef PIN_I2S_SCK
#define PIN_I2S_SCK 2  // D1 — BCLK
#endif
#ifndef PIN_I2S_WS
#define PIN_I2S_WS 3  // D2 — LRCL
#endif

static constexpr int kPinI2sSd = PIN_I2S_SD;
static constexpr int kPinI2sSck = PIN_I2S_SCK;
static constexpr int kPinI2sWs = PIN_I2S_WS;

static constexpr uint32_t SAMPLE_RATE = 16000;
static constexpr size_t SAMPLES_PER_FRAME = 320;
static constexpr uint8_t FRAME_MAGIC_0 = 0xA5;
static constexpr uint8_t FRAME_MAGIC_1 = 0x5A;

static int32_t sample_buffer[SAMPLES_PER_FRAME * 2];
static bool use_right_channel = false;
static int pcm_shift = 15;  // SPH0645: often 1 bit left of INMP441 alignment

static i2s_comm_format_t comm_format() {
#if defined(I2S_COMM_FORMAT_I2S_MSB)
  return (i2s_comm_format_t)(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB);
#else
  return I2S_COMM_FORMAT_STAND_I2S;
#endif
}

static bool init_i2s(bool swap_clock_pins) {
  i2s_config_t config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
      .communication_format = comm_format(),
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 256,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0,
  };

  const int bck = swap_clock_pins ? kPinI2sWs : kPinI2sSck;
  const int ws = swap_clock_pins ? kPinI2sSck : kPinI2sWs;

  i2s_pin_config_t pins = {
      .mck_io_num = I2S_PIN_NO_CHANGE,
      .bck_io_num = bck,
      .ws_io_num = ws,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = kPinI2sSd,
  };

  if (i2s_driver_install(I2S_NUM_0, &config, 0, nullptr) != ESP_OK) {
    return false;
  }
  if (i2s_set_pin(I2S_NUM_0, &pins) != ESP_OK) {
    return false;
  }
  if (i2s_set_clk(I2S_NUM_0, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT,
                  I2S_CHANNEL_STEREO) != ESP_OK) {
    return false;
  }
  i2s_zero_dma_buffer(I2S_NUM_0);
  return true;
}

static void shutdown_i2s() {
  i2s_driver_uninstall(I2S_NUM_0);
}

static int32_t peak_abs(const int32_t *raw, size_t pairs, bool right, int shift) {
  int32_t peak = 0;
  for (size_t i = 0; i < pairs; ++i) {
    const int32_t v = right ? raw[i * 2 + 1] : raw[i * 2];
    const int32_t scaled = v >> shift;
    const int32_t a = scaled < 0 ? -scaled : scaled;
    if (a > peak) {
      peak = a;
    }
  }
  return peak;
}

static int32_t diagnose_peak(bool swap_clock_pins) {
  if (!init_i2s(swap_clock_pins)) {
    return -1;
  }

  size_t dummy_bytes = 0;
  for (int i = 0; i < 6; ++i) {
    i2s_read(I2S_NUM_0, sample_buffer, sizeof(sample_buffer), &dummy_bytes,
             pdMS_TO_TICKS(100));
  }

  int32_t max_l = 0;
  int32_t max_r = 0;
  for (int i = 0; i < 20; ++i) {
    size_t bytes_read = 0;
    i2s_read(I2S_NUM_0, sample_buffer, sizeof(sample_buffer), &bytes_read,
             pdMS_TO_TICKS(200));
    const size_t pairs = bytes_read / (2 * sizeof(int32_t));
    for (int shift = 14; shift <= 17; ++shift) {
      max_l = max(max_l, peak_abs(sample_buffer, pairs, false, shift));
      max_r = max(max_r, peak_abs(sample_buffer, pairs, true, shift));
    }
  }

  shutdown_i2s();

  Serial.printf("DIAG swap=%d maxL=%ld maxR=%ld first=%ld/%ld\n", swap_clock_pins ? 1 : 0,
                static_cast<long>(max_l), static_cast<long>(max_r),
                static_cast<long>(sample_buffer[0]),
                static_cast<long>(sample_buffer[1]));
  Serial.flush();

  return max_l > max_r ? max_l : max_r;
}

static void send_pcm_frame(const int16_t *samples, size_t count) {
  const uint16_t sample_count = static_cast<uint16_t>(count);
  const uint8_t header[4] = {
      FRAME_MAGIC_0,
      FRAME_MAGIC_1,
      static_cast<uint8_t>(sample_count & 0xFF),
      static_cast<uint8_t>((sample_count >> 8) & 0xFF),
  };

  Serial.write(header, sizeof(header));
  Serial.write(reinterpret_cast<const uint8_t *>(samples), count * sizeof(int16_t));
}

static void send_i2s_frame(const int32_t *raw, size_t pairs) {
  int16_t pcm[SAMPLES_PER_FRAME];
  const size_t n = pairs < SAMPLES_PER_FRAME ? pairs : SAMPLES_PER_FRAME;
  for (size_t i = 0; i < n; ++i) {
    const int32_t sample = use_right_channel ? raw[i * 2 + 1] : raw[i * 2];
    pcm[i] = static_cast<int16_t>(sample >> pcm_shift);
  }
  send_pcm_frame(pcm, n);
}

static int pick_best_shift(bool swap_clock) {
  if (!init_i2s(swap_clock)) {
    return 15;
  }

  int32_t max_l = 0;
  int32_t max_r = 0;
  for (int i = 0; i < 20; ++i) {
    size_t bytes_read = 0;
    i2s_read(I2S_NUM_0, sample_buffer, sizeof(sample_buffer), &bytes_read,
             pdMS_TO_TICKS(200));
    const size_t pairs = bytes_read / (2 * sizeof(int32_t));
    for (int shift = 14; shift <= 17; ++shift) {
      const int32_t pl = peak_abs(sample_buffer, pairs, false, shift);
      const int32_t pr = peak_abs(sample_buffer, pairs, true, shift);
      if (pl > max_l) {
        max_l = pl;
      }
      if (pr > max_r) {
        max_r = pr;
      }
    }
  }

  use_right_channel = max_r > max_l;
  const int32_t chosen = use_right_channel ? max_r : max_l;

  int best_shift = 15;
  int32_t best_peak = 0;
  for (int shift = 14; shift <= 17; ++shift) {
    int32_t peak = 0;
    for (int i = 0; i < 10; ++i) {
      size_t bytes_read = 0;
      i2s_read(I2S_NUM_0, sample_buffer, sizeof(sample_buffer), &bytes_read,
               pdMS_TO_TICKS(200));
      const size_t pairs = bytes_read / (2 * sizeof(int32_t));
      peak = max(peak, peak_abs(sample_buffer, pairs, use_right_channel, shift));
    }
    if (peak > best_peak) {
      best_peak = peak;
      best_shift = shift;
    }
  }

  pcm_shift = best_shift;
  Serial.printf("DIAG using %s channel shift=%d peak=%ld swap=%d\n",
                use_right_channel ? "RIGHT" : "LEFT", pcm_shift,
                static_cast<long>(chosen), swap_clock ? 1 : 0);
  Serial.flush();
  return best_shift;
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("BOOT");
  Serial.println("CARE_AUDIO sph0645 16000Hz mono");
  Serial.println("DIAG: clap or talk during the next 2 seconds...");

  const int32_t peak_normal = diagnose_peak(false);
  const int32_t peak_swapped = diagnose_peak(true);

  bool swap_clock = false;
  int32_t best_peak = peak_normal;
  if (peak_swapped > best_peak) {
    best_peak = peak_swapped;
    swap_clock = true;
  }

  if (best_peak < 0) {
    Serial.println("ERROR: I2S init failed — check SPH0645 wiring on D0/D1/D2");
    while (true) {
      delay(1000);
    }
  }

  if (!init_i2s(swap_clock)) {
    Serial.println("ERROR: I2S init failed on second pass");
    while (true) {
      delay(1000);
    }
  }

  pick_best_shift(swap_clock);

  if (best_peak == 0) {
    Serial.println("WARN: mic still silent — check:");
    Serial.println("  3V->3V3  GND->GND  SEL->GND or 3V3");
    Serial.println("  DOUT->D0  BCLK->D1  LRCL->D2  (try swapping BCLK & LRCL)");
  }

  Serial.println("STREAM_BEGIN");
}

void loop() {
  size_t bytes_read = 0;
  esp_err_t err = i2s_read(I2S_NUM_0, sample_buffer, sizeof(sample_buffer),
                           &bytes_read, pdMS_TO_TICKS(250));

  if (err != ESP_OK || bytes_read == 0) {
    return;
  }

  const size_t pairs = bytes_read / (2 * sizeof(int32_t));
  send_i2s_frame(sample_buffer, pairs);
}
