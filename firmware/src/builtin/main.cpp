/**
 * Built-in PDM microphone test — XIAO ESP32S3 Sense (no external wiring).
 *
 * Upload env: builtin_mic
 * Monitor @ 115200 — clap/talk and RMS should jump.
 */

#include <Arduino.h>
#include <I2S.h>

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("BOOT");
  Serial.println("CARE_AUDIO builtin_pdm_test 16000Hz");

  I2S.setAllPins(-1, 42, 41, -1, -1);
  if (!I2S.begin(PDM_MONO_MODE, 16000, 16)) {
    Serial.println("ERROR: built-in PDM mic init failed");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("Built-in mic ready. RMS printed once per second.");
}

void loop() {
  static uint32_t last_print = 0;
  static double sum_sq = 0.0;
  static uint32_t n = 0;

  const int sample = I2S.read();
  if (sample == -1 || sample == 0 || sample == 1) {
    return;
  }

  sum_sq += static_cast<double>(sample) * sample;
  n++;

  const uint32_t now = millis();
  if (now - last_print >= 1000 && n > 0) {
    const float rms = static_cast<float>(sqrt(sum_sq / n));
    Serial.print("RMS ");
    Serial.println(rms, 1);
    sum_sq = 0.0;
    n = 0;
    last_print = now;
  }
}
