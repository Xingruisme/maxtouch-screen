// stent_tx.ino - stent transmitter for the maXTouch tests: Teensy 4.0 + AD9833 DDS (the TeensyTRX hardware).
// Written 2026-10-20 because the original TeensyTRX firmware / transceiver code is not available here.
// Build: Arduino + Teensyduino, Board "Teensy 4.0", USB Type "Serial". Host driver: ../../stent_tx.py
//
// A transmission is a list of segments (frequency [Hz], duration [us]); frequency 0 = carrier off (frozen DDS
// phase = DC, the same 'zero word' OOK the TeensyTRX used). OOK, FSK and the probe sweeps are all segment lists.
// Glitch-free timing: the AD9833 has two frequency registers. While one segment plays from one register, the next
// segment's frequency is already in the other; at the boundary an IntervalTimer interrupt (crystal timed) only
// flips FSELECT with one SPI word, then preloads the following segment into the register just freed.
//
// Text commands over USB serial (one per line, replies start with '#'):
//   id                       -> # stenttx 1 mode=dds mclk=25000000 fsync=10 wave=square outpin=2 nseg=0
//   mclk <Hz>                AD9833 master clock (default 25 MHz, the usual module crystal)
//   fsync <pin>              AD9833 FSYNC (chip select) pin (default 10; SCLK=13, SDATA=MOSI=11 are the SPI pins)
//   wave square|sine|tri     AD9833 output shape (default square, as in the CapTIvate tests)
//   out dds | out pin <n>    output through the AD9833 (default) or a plain 50 % square wave on Teensy pin n
//                            (fallback if the AD9833 does not respond; pin mode frequency steps are ~0.1-0.3 %)
//   clear                    empty the segment list
//   s <hz>:<us> <hz>:<us> .. append segments (many per line, lines up to 2000 chars)
//   sum                      -> # sum nseg=<n> total_us=<sum of durations> hz_x10=<sum of round(hz*10)>
//   go <delay_ms>            -> # armed in <delay_ms> ms nseg=<n> total_us=<t>   ... later: # done
//   tone <hz>                continuous carrier (stops a running list)   -> # tone <hz>
//   off                      stop and turn the carrier off               -> # off
//   status                   -> # status running=<0|1> seg=<k> nseg=<n>
//   adc <n>                  self-test: min/max/std of n fast reads on each analog pin A0-A13 (input only)
//                            -> # adc A0:<min>/<max>/<std> A1:...   compare with 'tone' on and off

#include <SPI.h>
#include <IntervalTimer.h>

struct Seg { float hz; uint32_t us; };
static const int MAX_SEG = 40000;
DMAMEM Seg segs[MAX_SEG];                       // 320 KB in OCRAM
volatile int nseg = 0;

double mclk = 25000000.0;
int fsync_pin = 10;
enum Wave { SQUARE, SINE, TRI } wave = SQUARE;
bool pin_mode = false;
int out_pin = 2;

IntervalTimer timer;
volatile bool running = false, finished = false;
volatile int k_cur = -1;                        // segment now playing (-1 = start delay)

SPISettings spiset(8000000, MSBFIRST, SPI_MODE2);   // AD9833: data clocked on the falling edge, SCLK idles high

// ------------------------------------------------------------------ AD9833
void ad_write(uint16_t w) {
  SPI.beginTransaction(spiset);
  digitalWriteFast(fsync_pin, LOW);
  SPI.transfer16(w);
  digitalWriteFast(fsync_pin, HIGH);
  SPI.endTransaction();
}

uint16_t ctrl_base() {                          // B28 = 1: 28-bit frequency words written as two 14-bit halves
  uint16_t c = 0x2000;
  if (wave == SQUARE) c |= 0x0028;              // OPBITEN + DIV2: MSB of the phase accumulator = full-rate square
  else if (wave == TRI) c |= 0x0002;            // MODE
  return c;
}

uint32_t fword(float hz) {
  if (hz <= 0) return 0;
  double w = (double)hz * 268435456.0 / mclk + 0.5;
  return w > 268435455.0 ? 268435455u : (uint32_t)w;
}

void ad_load(int reg, float hz) {
  uint32_t w = fword(hz);
  uint16_t addr = reg ? 0x8000 : 0x4000;
  ad_write(addr | (w & 0x3FFF));
  ad_write(addr | ((w >> 14) & 0x3FFF));
}

void ad_select(int reg) { ad_write(ctrl_base() | (reg ? 0x0800 : 0)); }

void ad_init() {
  pinMode(fsync_pin, OUTPUT);
  digitalWriteFast(fsync_pin, HIGH);
  ad_write(ctrl_base() | 0x0100);               // RESET while loading
  ad_load(0, 0); ad_load(1, 0);
  ad_write(0xC000); ad_write(0xE000);           // phase registers 0
  ad_select(0);                                 // clear RESET
}

// ------------------------------------------------------------------ output (DDS or plain pin)
volatile float pin_hz[2] = {0, 0};             // pin mode: the two 'registers' are just remembered frequencies
void out_preload(int reg, float hz) { if (pin_mode) pin_hz[reg] = hz; else ad_load(reg, hz); }
void out_select(int reg) {
  if (!pin_mode) { ad_select(reg); return; }
  float hz = pin_hz[reg];
  if (hz > 0) { analogWriteFrequency(out_pin, hz); analogWrite(out_pin, 128); }
  else analogWrite(out_pin, 0);
}
void out_off() {
  if (pin_mode) { analogWrite(out_pin, 0); pin_hz[0] = pin_hz[1] = 0; }
  else { ad_load(0, 0); ad_load(1, 0); ad_select(0); }
}

// ------------------------------------------------------------------ playback
// segment k plays from register reg(k) = (k + 1) & 1, so the start delay (k = -1, carrier off) uses register 0.
inline int reg_of(int k) { return (k + 1) & 1; }

void step() {                                   // IntervalTimer: one boundary, entering segment k_cur + 1
  int k = k_cur + 1;
  if (k >= nseg) {
    timer.end();
    out_off();
    running = false; finished = true; k_cur = nseg;
    return;
  }
  out_select(reg_of(k));                        // already loaded: the switch is one SPI word
  k_cur = k;
  if (k + 1 < nseg) {
    out_preload(reg_of(k + 1), segs[k + 1].hz);
    timer.update(segs[k + 1].us);               // period after this one
  } else {
    out_preload(reg_of(k + 1), 0);
    timer.update(1000);                         // the 'end' boundary comes segs[k].us after this one anyway
  }
}

void go(uint32_t delay_ms) {
  timer.end();
  running = false; finished = false;
  if (nseg == 0) { Serial.println("# error no segments"); return; }
  out_preload(0, 0);                            // start delay: carrier off
  out_preload(1, segs[0].hz);
  out_select(0);
  k_cur = -1;
  uint64_t total = 0;
  for (int i = 0; i < nseg; i++) total += segs[i].us;
  running = true;
  timer.begin(step, delay_ms > 0 ? delay_ms * 1000.0 : 10.0);
  timer.update(segs[0].us);
  Serial.printf("# armed in %lu ms nseg=%d total_us=%llu\n", (unsigned long)delay_ms, nseg, total);
}

void stop_all() { timer.end(); running = false; out_off(); }

// ------------------------------------------------------------------ commands
char line[2100];
int llen = 0;

void cmd_id() {
  Serial.printf("# stenttx 1 mode=%s mclk=%.0f fsync=%d wave=%s outpin=%d nseg=%d\n", pin_mode ? "pin" : "dds", mclk,
                fsync_pin, wave == SQUARE ? "square" : wave == SINE ? "sine" : "tri", out_pin, nseg);
}

void handle(char *s) {
  while (*s == ' ') s++;
  if (!*s) return;
  char *arg = strchr(s, ' ');
  if (arg) { *arg++ = 0; while (*arg == ' ') arg++; } else arg = s + strlen(s);

  if (!strcmp(s, "id")) cmd_id();
  else if (!strcmp(s, "mclk")) { mclk = atof(arg); Serial.printf("# mclk %.0f\n", mclk); }
  else if (!strcmp(s, "fsync")) { stop_all(); fsync_pin = atoi(arg); ad_init(); Serial.printf("# fsync %d\n", fsync_pin); }
  else if (!strcmp(s, "wave")) {
    wave = !strcmp(arg, "sine") ? SINE : !strcmp(arg, "tri") ? TRI : SQUARE;
    if (!running) ad_select(0);
    Serial.printf("# wave %s\n", arg);
  }
  else if (!strcmp(s, "out")) {
    stop_all();
    if (!strncmp(arg, "pin", 3)) { pin_mode = true; out_pin = atoi(arg + 3); pinMode(out_pin, OUTPUT); analogWriteResolution(8); analogWrite(out_pin, 0); }
    else { pin_mode = false; ad_init(); }
    cmd_id();
  }
  else if (!strcmp(s, "clear")) { stop_all(); nseg = 0; Serial.println("# clear"); }
  else if (!strcmp(s, "s")) {
    if (running) { Serial.println("# error running"); return; }
    char *p = arg;
    while (*p) {
      char *colon = strchr(p, ':');
      if (!colon) break;
      float hz = atof(p);
      uint32_t us = strtoul(colon + 1, &p, 10);
      if (nseg >= MAX_SEG) { Serial.println("# error too many segments"); return; }
      if (us < 20) us = 20;
      segs[nseg].hz = hz; segs[nseg].us = us; nseg++;
      while (*p == ' ') p++;
    }
  }
  else if (!strcmp(s, "sum")) {
    uint64_t t = 0, h = 0;
    for (int i = 0; i < nseg; i++) { t += segs[i].us; h += (uint64_t)(segs[i].hz * 10.0f + 0.5f); }
    Serial.printf("# sum nseg=%d total_us=%llu hz_x10=%llu\n", nseg, t, h);
  }
  else if (!strcmp(s, "go")) go(strtoul(arg, 0, 10));
  else if (!strcmp(s, "tone")) {
    stop_all();
    float hz = atof(arg);
    out_preload(0, hz); out_select(0);
    Serial.printf("# tone %.3f word=%lu\n", hz, (unsigned long)fword(hz));
  }
  else if (!strcmp(s, "off")) { stop_all(); Serial.println("# off"); }
  else if (!strcmp(s, "status")) Serial.printf("# status running=%d seg=%d nseg=%d\n", running ? 1 : 0, k_cur, nseg);
  else if (!strcmp(s, "adc")) {
    // self-test: read every analog pin (A0-A13 = pins 14-27, input only) n times as fast as possible and report
    // min/max/std, so 'tone' on vs off shows which pin senses E+ (the TeensyTRX recorded its own E+ with the ADC).
    // A pin that is really an output elsewhere stays an input afterwards; 'fsync <pin>' re-inits the AD9833 pin.
    int n = constrain(atoi(arg), 16, 4000);
    analogReadResolution(12);
    analogReadAveraging(1);
    Serial.print("# adc");
    for (int p = 0; p < 14; p++) {
      int mn = 4095, mx = 0; double s1 = 0, s2 = 0;
      for (int i = 0; i < n; i++) {
        int v = analogRead(14 + p);
        mn = min(mn, v); mx = max(mx, v); s1 += v; s2 += (double)v * v;
      }
      double m = s1 / n;
      Serial.printf(" A%d:%d/%d/%.1f", p, mn, mx, sqrt(fmax(s2 / n - m * m, 0)));
    }
    Serial.println();
  }
  else Serial.printf("# error unknown command '%s'\n", s);
}

void setup() {
  Serial.begin(921600);                         // USB serial: the baud rate is ignored, kept for the old scripts
  SPI.begin();
  ad_init();
  // no LED: on the Teensy 4.0 LED_BUILTIN is pin 13 = SPI SCK to the AD9833. Driving it as an LED took the
  // clock away from the SPI port and the first AD9833 write hung the firmware (2026-10-20).
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') { line[llen] = 0; if (llen) handle(line); llen = 0; }
    else if (llen < (int)sizeof(line) - 1) line[llen++] = c;
  }
  if (finished) { finished = false; Serial.println("# done"); }
}
