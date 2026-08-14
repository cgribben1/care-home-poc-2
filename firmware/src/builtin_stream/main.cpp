/**
 * Built-in PDM mic stream — XIAO ESP32S3 Sense (expansion board attached).
 * No external INMP441 wiring needed.
 *
 * Upload env: builtin_stream
 * Python:     python serial_bridge.py --port COM3
 */

#include <Arduino.h>
#include <I2S.h>

static constexpr size_t SAMPLES_PER_FRAME = 320;
static constexpr uint8_t FRAME_MAGIC_0 = 0xA5;
static constexpr uint8_t FRAME_MAGIC_1 = 0x5A;

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

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("BOOT");
  Serial.println("CARE_AUDIO builtin_pdm_stream 16000Hz mono");

  // Seeed wiki: WS=42, DATA=41 for built-in PDM mic.
  I2S.setAllPins(-1, 42, 41, -1, -1);
  if (!I2S.begin(PDM_MONO_MODE, 16000, 16)) {
    Serial.println("ERROR: built-in PDM mic init failed");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("STREAM_BEGIN");
}

void loop() {
  int16_t pcm[SAMPLES_PER_FRAME];
  for (size_t i = 0; i < SAMPLES_PER_FRAME; ++i) {
    int sample = 0;
    while (sample == 0 || sample == -1 || sample == 1) {
      sample = I2S.read();
      if (sample == 0 && !I2S.available()) {
        delay(1);
      }
    }
    pcm[i] = static_cast<int16_t>(sample);
  }
  send_pcm_frame(pcm, SAMPLES_PER_FRAME);
}
