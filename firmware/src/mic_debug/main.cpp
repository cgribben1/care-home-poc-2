/**
 * Text-only INMP441 wiring test — no binary stream.
 *
 * Upload:  pio run -e mic_debug -t upload
 * Monitor: pio device monitor -e mic_debug
 *
 * Clap when prompted. You should see non-zero peak values for one config.
 */

#include <Arduino.h>
#include <driver/i2s.h>

struct PinSet {
  const char *name;
  int sd;
  int sck;
  int ws;
};

static constexpr PinSet PIN_SETS[] = {
    {"D0/D1/D2 (SD/SCK/WS)", 1, 2, 3},
    {"D0/D2/D1 (SD/WS/SCK swapped)", 1, 3, 2},
    {"D3/D4/D5", 4, 5, 6},
};

static int32_t buffer[320 * 2];

static int32_t peak_stereo(const int32_t *raw, size_t pairs, bool right) {
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

static bool run_test(i2s_port_t port, const PinSet &pins, bool swap_clk) {
  i2s_driver_uninstall(port);

  i2s_config_t config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = 16000,
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

  const int bck = swap_clk ? pins.ws : pins.sck;
  const int ws = swap_clk ? pins.sck : pins.ws;

  i2s_pin_config_t pin_cfg = {
      .mck_io_num = I2S_PIN_NO_CHANGE,
      .bck_io_num = bck,
      .ws_io_num = ws,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = pins.sd,
  };

  if (i2s_driver_install(port, &config, 0, nullptr) != ESP_OK) {
    Serial.printf("  FAIL install port=%d\n", port);
    return false;
  }
  if (i2s_set_pin(port, &pin_cfg) != ESP_OK) {
    Serial.printf("  FAIL pins port=%d\n", port);
    i2s_driver_uninstall(port);
    return false;
  }
  i2s_set_clk(port, 16000, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_STEREO);
  i2s_zero_dma_buffer(port);

  size_t dummy = 0;
  for (int i = 0; i < 8; ++i) {
    i2s_read(port, buffer, sizeof(buffer), &dummy, pdMS_TO_TICKS(100));
  }

  int32_t max_l = 0;
  int32_t max_r = 0;
  for (int i = 0; i < 15; ++i) {
    size_t bytes = 0;
    i2s_read(port, buffer, sizeof(buffer), &bytes, pdMS_TO_TICKS(200));
    const size_t pairs = bytes / (2 * sizeof(int32_t));
    max_l = max(max_l, peak_stereo(buffer, pairs, false));
    max_r = max(max_r, peak_stereo(buffer, pairs, true));
  }

  Serial.printf("  port=%d swap=%d SD=%d SCK=%d WS=%d -> maxL=%ld maxR=%ld first=%ld/%ld\n",
                port, swap_clk ? 1 : 0, pins.sd, pins.sck, pins.ws,
                static_cast<long>(max_l), static_cast<long>(max_r),
                static_cast<long>(buffer[0]), static_cast<long>(buffer[1]));

  return max_l > 1000 || max_r > 1000;
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println();
  Serial.println("=== INMP441 WIRING DEBUG (text only) ===");
  Serial.println("Clap or talk now — testing pin configs...");
  Serial.println("Expected wiring: VDD->3V3 GND->GND SD->D0 SCK->D1 WS->D2");
  Serial.println();

  bool any = false;
  for (const PinSet &pins : PIN_SETS) {
    Serial.printf("Config: %s\n", pins.name);
    any |= run_test(I2S_NUM_0, pins, false);
    any |= run_test(I2S_NUM_0, pins, true);
    any |= run_test(I2S_NUM_1, pins, false);
    Serial.println();
  }

  if (any) {
    Serial.println("SUCCESS: at least one config saw audio. Note the line above with high maxL/maxR.");
  } else {
    Serial.println("ALL ZERO — hardware issue. Check:");
    Serial.println("  1. VDD on 3V3 (measure ~3.3V with multimeter)");
    Serial.println("  2. Module pin order (VDD/GND/SD/SCK/WS/L-R vary by board)");
    Serial.println("  3. L/R pin -> try 3V3 instead of GND");
    Serial.println("  4. Swap SCK and WS wires");
    Serial.println("  5. Try built-in mic: pio run -e builtin_stream -t upload");
  }

  Serial.println();
  Serial.println("Repeating best-effort live RMS every 2s (port 0, D0/D1/D2)...");
  run_test(I2S_NUM_0, PIN_SETS[0], false);
}

void loop() {
  size_t bytes = 0;
  i2s_read(I2S_NUM_0, buffer, sizeof(buffer), &bytes, pdMS_TO_TICKS(500));
  const size_t pairs = bytes / (2 * sizeof(int32_t));
  const int32_t peak_l = peak_stereo(buffer, pairs, false);
  const int32_t peak_r = peak_stereo(buffer, pairs, true);
  const int32_t peak = peak_l > peak_r ? peak_l : peak_r;
  const float rms = peak / 65536.0f;
  Serial.printf("live peak=%ld (~rms %0.4f)  L=%ld R=%ld\n",
                static_cast<long>(peak), rms,
                static_cast<long>(peak_l), static_cast<long>(peak_r));
  delay(2000);
}
