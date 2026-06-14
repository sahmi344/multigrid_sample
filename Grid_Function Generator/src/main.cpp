#include <Arduino.h>
#include <SPI.h>

void setFrequency(float f);
void writeControlWord(uint16_t word);
void writeFreqWord(uint16_t word);
void writeFrame(uint16_t word);



const uint8_t PIN_FSYNC = 5;  // FSYNC (active LOW)
const uint8_t PIN_SCLK  = 18;  // SCLK
const uint8_t PIN_SDIO  = 23;  // SDATA

const unsigned long MCLK_HZ = 25000000UL; // 25 MHz master clock

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  pinMode(PIN_FSYNC, OUTPUT);
  pinMode(PIN_SCLK, OUTPUT);
  pinMode(PIN_SDIO, OUTPUT);

  // Idle states required by AD9833
  digitalWrite(PIN_FSYNC, HIGH);
  digitalWrite(PIN_SCLK, HIGH);
  digitalWrite(PIN_SDIO, LOW);

  delay(10); // allow module to settle

  // Start with a default frequency (e.g., 400 Hz from the example)
  setFrequency(100000.0f);
  Serial.println("Ready. Type frequency in Hz and press Enter (e.g., 1000).");
}

void loop() {
  // Read serial input (simple line-based)
  if (Serial.available()) {
    String s = Serial.readStringUntil('\n');
    s.trim();
    if (s.length() > 0) {
      float f = s.toFloat();
      if (f > 0.0f && f < (MCLK_HZ/2.0f)) {
        setFrequency(f);
      } else {
        Serial.println("Invalid frequency. Must be >0 and less than MCLK/2.");
      }
    }
  }
}

// setFrequency: compute 28-bit freqReg and update AD9833 FREQ0
void setFrequency(float freqHz) {
  // 28-bit frequency register calculation
  // freqReg = round(freqHz * 2^28 / MCLK)
  const double FRAC = (double(1UL << 28) / double(MCLK_HZ));
  uint32_t freqReg = (uint32_t)round(freqHz * FRAC);

  // Sanity: limit to 28 bits
  freqReg &= 0x0FFFFFFFUL;

  // Split into two 14-bit words
  uint16_t lsb14 = freqReg & 0x3FFF;           // lower 14 bits
  uint16_t msb14 = (freqReg >> 14) & 0x3FFF;   // upper 14 bits

  // Form the AD9833 16-bit words (0x4000 OR data for FREQ registers)
  uint16_t freq0_lsb_word = 0x4000 | lsb14;
  uint16_t freq0_msb_word = 0x4000 | msb14;

  Serial.print("Setting freq: ");
  Serial.print(freqHz, 6);
  Serial.print(" Hz -> freqReg = 0x");
  Serial.print(freqReg, HEX);
  Serial.print(" (");
  Serial.print(freqReg);
  Serial.println(" decimal)");

  Serial.print("Freq0 LSB word: 0x");
  Serial.println(freq0_lsb_word, HEX);
  Serial.print("Freq0 MSB word: 0x");
  Serial.println(freq0_msb_word, HEX);

  // Sequence: 1) Write control word B28=1, RESET=1 (hold reset while writing)
  writeControlWord(0x2100); // B28=1, RESET=1

  delayMicroseconds(5);

  // 2) Write the two frequency words (LSB then MSB)
  writeFreqWord(freq0_lsb_word);
  writeFreqWord(freq0_msb_word);

  delayMicroseconds(5);

  // 3) Clear RESET to allow DAC to use new value:
  // Use a control word with RESET=0. Keep other bits minimal (example uses 0x2000).
  writeControlWord(0x2000);

  // Small pause to allow DAC to start updating (a few MCLK cycles)
  delay(1);

  Serial.println("Frequency updated.");
}

// Write a control word (16-bit) to AD9833. This is a single-frame write.
void writeControlWord(uint16_t word) {
  writeFrame(word);
}

// Write a FREQ register word (assumed to already have 0x4000 bit set)
void writeFreqWord(uint16_t word) {
  writeFrame(word);
}

// Core frame write: FSYNC low, shift 16 bits MSB-first, FSYNC high.
// Data is clocked on the falling edge of SCLK (we generate the falling edge).
void writeFrame(uint16_t word) {
  // Ensure SCLK high before FSYNC low (datasheet requirement).
  digitalWrite(PIN_SCLK, HIGH);
  delayMicroseconds(1);

  // Assert FSYNC (active low) to start a frame
  digitalWrite(PIN_FSYNC, LOW);
  delayMicroseconds(1); // t7 setup

  for (int bit = 15; bit >= 0; --bit) {
    // Put bit on SDATA
    digitalWrite(PIN_SDIO, ((word >> bit) & 1) ? HIGH : LOW);
    delayMicroseconds(1); // small settle

    // Falling edge to latch data
    digitalWrite(PIN_SCLK, LOW);
    delayMicroseconds(1); // t10 hold

    // Rising edge prepare for next bit
    digitalWrite(PIN_SCLK, HIGH);
    delayMicroseconds(1);
  }

  // End frame
  digitalWrite(PIN_FSYNC, HIGH);
  delayMicroseconds(1); // t8 post-frame
}