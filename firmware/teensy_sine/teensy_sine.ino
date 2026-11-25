/*
  teensy_sine.ino - Teensy 4.0 + AD9833 (TeensyTRX board): continuous SINE output (2026-11-23)

  Stand-alone: power it from a battery and the sine at DEFAULT_HZ starts by itself ~5 s after power-up (AUTO_START)
  and runs until the power is removed. No computer needed. With USB connected you can still type 'off' / 'on' / 'f <hz>'.
  Battery: a USB power bank on the Teensy's USB port, or 3.6-5.5 V into VIN (e.g. 1-cell LiPo 3.7 V, 3 x AA).
  The AD9833 sine has a FIXED amplitude of ~0.6 Vpp at VOUT (no amplitude control on this chip). That is ~17 dB
  below the square wave the 1 mA network was set up with, so through the same network the sine current is roughly
  0.15-0.2 mA. The current is set by the resistor network, not by this sketch: check it with the scope before a
  person is connected.

  Wiring = TeensyTRX: FSYNC 10, SDATA 11, SCLK 13 (bit-banged; hardware SPI did not work on this board), pin 2 HIGH
  while transmitting (TX/RX switch), MCLK 25 MHz.

  Serial (USB, any baud, line ending Newline):
    on            start the sine
    off           stop (RESET + sleep: no output)
    f <hz>        set the frequency (1 Hz - 500 kHz), keeps on/off state
    wave sine|square   waveform (square = ~17 dB more carrier, the level of the 1 mA BER runs)
    ?             status
  Board: Teensy 4.0, USB Type: Serial.
*/

#if !defined(__IMXRT1062__)
#error "Wrong board: select Tools > Board > Teensy 4.0 (and USB Type: Serial). digitalWriteFast is Teensy-only."
#endif

const uint8_t  PIN_FSYNC = 10;
const uint8_t  PIN_SDATA = 11;
const uint8_t  PIN_SCLK  = 13;
const uint8_t  PIN_TR    = 2;
const double   MCLK       = 25000000.0;
const double   DEFAULT_HZ = 250.0;              // frequency sent at power-up (change here for battery use; 159585 for the tablet)
const bool     START_SQUARE = true;             // false = sine (~0.6 Vpp at VOUT); true = square (full 3.3 V swing,
                                                // = the level of the 1 mA through-body BER runs, ~17 dB above the sine)
// Below 1 kHz the IEC 60601-1 type BF limit is 0.1 mA (e.g. 50 Hz); at 159.585 kHz it is 10 mA. The current is set by
// the resistor network, not by this sketch.
const bool     AUTO_START = true;               // true = sine starts by itself START_DELAY_MS after power-up (battery, no computer)
const uint32_t START_DELAY_MS = 3000;
const double   MAX_HZ     = 500000.0;

const uint16_t B28 = 1 << 13, RST = 1 << 8, SL1 = 1 << 7, SL12 = 1 << 6;
const uint16_t SINE_ON = B28;                       // sine, RESET cleared
const uint16_t OPBITEN = 1 << 5, DIV2 = 1 << 3;
const uint16_t SQUARE_ON = B28 | OPBITEN | DIV2;    // square = DAC off, comparator MSB at full VDD swing
uint16_t runw = SINE_ON;
const uint16_t OFFW    = B28 | RST | SL1 | SL12;    // RESET + MCLK stopped + DAC powered down: no output

double hz_now = DEFAULT_HZ;
bool   on_now = false;

static void adw(uint16_t v) {                       // 16 bits, MSB first; SCLK idles HIGH, AD9833 reads on falling edge
  digitalWriteFast(PIN_FSYNC, LOW);
  delayNanoseconds(50);
  for (int i = 15; i >= 0; i--) {
    digitalWriteFast(PIN_SDATA, (v >> i) & 1);
    delayNanoseconds(100);
    digitalWriteFast(PIN_SCLK, LOW);
    delayNanoseconds(100);
    digitalWriteFast(PIN_SCLK, HIGH);
  }
  delayNanoseconds(50);
  digitalWriteFast(PIN_FSYNC, HIGH);
  delayNanoseconds(500);
}

static uint32_t fword(double hz) { return (uint32_t)(hz * 268435456.0 / MCLK + 0.5) & 0x0FFFFFFF; }

static void sineOff() {
  adw(OFFW);
  digitalWriteFast(PIN_TR, LOW);
  on_now = false;
}

static void sineOn(double hz) {
  uint32_t w = fword(hz);
  adw(OFFW);                                        // registers are written only while the output is off
  adw(0x4000 | (w & 0x3FFF));                       // FREQ0 LSB
  adw(0x4000 | ((w >> 14) & 0x3FFF));               // FREQ0 MSB
  adw(0xC000);                                      // PHASE0 = 0
  digitalWriteFast(PIN_TR, HIGH);                   // switch closed before the output starts
  adw(runw);
  hz_now = w * MCLK / 268435456.0;
  on_now = true;
}

static void status() {
  Serial.print(F("# sine "));
  Serial.print(on_now ? F("ON ") : F("off "));
  Serial.print(hz_now, 3);
  Serial.println(runw == SQUARE_ON ? F(" Hz square (~3.3 V swing at VOUT)") : F(" Hz sine (~0.6 Vpp at VOUT)"));
}

static void handle(char *s) {
  while (*s == ' ') s++;
  if (!strcmp(s, "on"))  { sineOn(hz_now); status(); return; }
  if (!strcmp(s, "off")) { sineOff(); status(); return; }
  if (s[0] == 'f' && (s[1] == ' ' || s[1] == 0)) {
    double hz = atof(s + 1);
    if (hz < 1 || hz > MAX_HZ) { Serial.println(F("# f: refused (1 Hz - 500 kHz)")); return; }
    if (on_now) sineOn(hz); else hz_now = fword(hz) * MCLK / 268435456.0;
    status(); return;
  }
  if (!strncmp(s, "wave", 4)) {
    runw = strstr(s, "sq") ? SQUARE_ON : SINE_ON;
    if (on_now) sineOn(hz_now);
    status(); return;
  }
  if (s[0] == '?') { Serial.println(F("teensy_sine - commands: on | off | f <hz> | wave sine|square | ?")); status(); return; }
  Serial.println(F("# unknown command (on | off | f <hz> | ?)"));
}

void setup() {
  pinMode(PIN_FSYNC, OUTPUT); digitalWriteFast(PIN_FSYNC, HIGH);
  pinMode(PIN_TR, OUTPUT);    digitalWriteFast(PIN_TR, LOW);
  pinMode(PIN_SCLK, OUTPUT);  digitalWriteFast(PIN_SCLK, HIGH);
  pinMode(PIN_SDATA, OUTPUT); digitalWriteFast(PIN_SDATA, LOW);
  // weakest drive on the three AD9833 lines (as TeensyTRX: strong edges ring on breadboard jumpers)
  for (uint8_t pin : {PIN_FSYNC, PIN_SDATA, PIN_SCLK})
    *(digital_pin_to_info_PGM[pin].pad) = IOMUXC_PAD_DSE(1) | IOMUXC_PAD_SPEED(0);
  sineOff();
  Serial.begin(0);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 2000) {}      // on a battery there is no USB host: this just times out
  runw = START_SQUARE ? SQUARE_ON : SINE_ON;
  if (AUTO_START) {
    delay(START_DELAY_MS);
    sineOn(DEFAULT_HZ);
  }
  Serial.println(F("teensy_sine - commands: on | off | f <hz> | wave sine|square | ?"));
  status();
}

void loop() {
  static char line[64]; static uint8_t n = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') { line[n] = 0; if (n) handle(line); n = 0; }
    else if (n < sizeof(line) - 1) line[n++] = c;
  }
}
