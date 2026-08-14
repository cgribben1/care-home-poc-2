/**
 * XIAO ESP32S3 Sense + Dollatek INMP441
 *
 * Wiring (INMP441 -> XIAO):
 *   VDD -> 3V3 | GND -> GND | L/R -> GND (left) or 3V3 (right)
 *   SD  -> D0 (GPIO1) | SCK -> D1 (GPIO2) | WS -> D2 (GPIO3)
 *
 * Streams 16 kHz mono PCM16 over USB serial in framed packets.
 * Python: ../serial_bridge.py --port COM3
 */

#include <Arduino.h>
#include <driver/i2s.h>

static constexpr int PIN_I2S_SD = 1;   // D0
static constexpr int PIN_I2S_SCK = 2;  // D1
static constexpr int PIN_I2S_WS = 3;   // D2

static constexpr uint32_t SAMPLE_RATE = 16000;
static constexpr size_t SAMPLES_PER_FRAME = 320;  // 20 ms @ 16 kHz mono out
static constexpr uint8_t FRAME_MAGIC_0 = 0xA5;
static constexpr uint8_t FRAME_MAGIC_1 = 0x5A;

// Stereo read: 320 mono frames = 320 L/R pairs.
static int32_t sample_buffer[SAMPLES_PER_FRAME * 2];
static bool use_right_channel = false;
static int pcm_shift = 16;  // INMP441: 24-bit data left-aligned in 32-bit slot

static bool init_i2s(bool swap_clock_pins) {
  i2s_config_t config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 256,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0,
  };

  const int bck = swap_clock_pins ? PIN_I2S_WS : PIN_I2S_SCK;
  const int ws = swap_clock_pins ? PIN_I2S_SCK : PIN_I2S_WS;

  i2s_pin_config_t pins = {
      .mck_io_num = I2S_PIN_NO_CHANGE,
      .bck_io_num = bck,
      .ws_io_num = ws,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = PIN_I2S_SD,
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

static int32_t peak_abs(const int32_t *raw, size_t pairs, bool right) {
  int32_t peak = 0;
  for (size_t i = 0; i < pairs; ++i) {
    const int32_t v = right ? raw[i * 2 + 1] : raw[i * 2];
    const int32_t a = v < 0 ? -v : v;
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
    max_l = max(max_l, peak_abs(sample_buffer, pairs, false));
    max_r = max(max_r, peak_abs(sample_buffer, pairs, true));
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

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("BOOT");
  Serial.println("CARE_AUDIO inmp441 16000Hz mono");
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
    Serial.println("ERROR: I2S init failed — check INMP441 wiring on D0/D1/D2");
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

  // Re-measure on the chosen wiring to pick L/R channel and bit shift.
  int32_t max_l = 0;
  int32_t max_r = 0;
  for (int i = 0; i < 20; ++i) {
    size_t bytes_read = 0;
    i2s_read(I2S_NUM_0, sample_buffer, sizeof(sample_buffer), &bytes_read,
             pdMS_TO_TICKS(200));
    const size_t pairs = bytes_read / (2 * sizeof(int32_t));
    max_l = max(max_l, peak_abs(sample_buffer, pairs, false));
    max_r = max(max_r, peak_abs(sample_buffer, pairs, true));
  }

  use_right_channel = max_r > max_l;
  const int32_t chosen_peak = use_right_channel ? max_r : max_l;

  // Pick shift: >>16 for full-scale 24-bit alignment, >>14 if signal is weak.
  pcm_shift = (chosen_peak > 0 && chosen_peak < (1 << 18)) ? 14 : 16;

  Serial.printf("DIAG using %s channel shift=%d peak=%ld swap=%d\n",
                use_right_channel ? "RIGHT" : "LEFT", pcm_shift,
                static_cast<long>(chosen_peak), swap_clock ? 1 : 0);
  Serial.flush();

  if (chosen_peak == 0) {
    Serial.println("WARN: mic still silent — check:");
    Serial.println("  VDD->3V3  GND->GND  L/R->GND or 3V3");
    Serial.println("  SD->D0  SCK->D1  WS->D2  (try swapping SCK & WS)");
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
