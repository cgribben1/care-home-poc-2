/**
 * Mic capture test — records 3s of stereo I2S audio and streams raw int32
 * values over serial for offline analysis. No TFLite, no WiFi.
 *
 * Wiring: DOUT->D6(GPIO43)  BCLK->D4(GPIO5)  LRCL->D5(GPIO6)
 *
 * Usage:
 *   1. Flash this env: pio run -e mic_test --target upload
 *   2. Close serial monitor
 *   3. Run: python tools/mic_capture.py
 *   4. Make noise when board says "MAKE NOISE NOW"
 *   5. Listen to R_channel.wav and L_channel.wav
 */

#include <Arduino.h>
#include <driver/i2s.h>

#define SAMPLE_RATE 16000
#define N_SECONDS   3
#define N_PAIRS     (SAMPLE_RATE * N_SECONDS)   // stereo pairs

static int32_t* raw_buf  = nullptr;
static int32_t  dma_tmp[512];

void setup() {
  Serial.begin(921600);
  delay(3000);
  Serial.println("\n=== MIC CAPTURE TEST ===");

  raw_buf = (int32_t*)ps_malloc(N_PAIRS * 2 * sizeof(int32_t));
  if (!raw_buf) {
    Serial.println("ERROR: ps_malloc failed");
    for (;;) delay(1000);
  }

  i2s_config_t cfg = {
    .mode              = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate       = SAMPLE_RATE,
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
    .bck_io_num   = 5,
    .ws_io_num    = 6,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = 43,
  };

  if (i2s_driver_install(I2S_NUM_0, &cfg, 0, nullptr) != ESP_OK) {
    Serial.println("ERROR: i2s_driver_install failed"); for (;;) delay(1000);
  }
  if (i2s_set_pin(I2S_NUM_0, &pins) != ESP_OK) {
    Serial.println("ERROR: i2s_set_pin failed"); for (;;) delay(1000);
  }
  i2s_zero_dma_buffer(I2S_NUM_0);
  Serial.println("I2S ready. Starting record loop...\n");
}

void loop() {
  Serial.println(">>> MAKE NOISE NOW — recording 3 seconds...");

  int collected = 0;
  const int total_i32 = N_PAIRS * 2;
  while (collected < total_i32) {
    size_t br = 0;
    i2s_read(I2S_NUM_0, dma_tmp, sizeof(dma_tmp), &br, pdMS_TO_TICKS(1000));
    int n  = (int)(br / sizeof(int32_t));
    int cp = min(n, total_i32 - collected);
    memcpy(raw_buf + collected, dma_tmp, cp * sizeof(int32_t));
    collected += cp;
  }

  Serial.println("Recording done. Streaming raw int32 data...");

  // Magic header so the Python script can sync
  const uint8_t magic[4] = {0xCA, 0xFE, 0xBA, 0xBE};
  uint32_t n32 = (uint32_t)total_i32;
  Serial.write(magic, 4);
  Serial.write((const uint8_t*)&n32, 4);
  Serial.write((const uint8_t*)raw_buf, total_i32 * sizeof(int32_t));
  Serial.flush();

  Serial.println("\nStream complete. Waiting 5s before next recording...\n");
  delay(5000);
}
