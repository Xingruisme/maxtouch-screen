/*
  TeensyTRX ask2 (2026-11-24) - TeensyTRX ask1 (firmware/TeensyTRX_ask, 2026-11-23) moved to the ADG729 circuit of
  "ASK Transceiver Wiring" (2026-11-24): the per-step AMPLITUDE for 2-ASK / 4-ASK is now one I2C byte to an ADG729
  (Wire: SDA 18, SCL 19, 400 kHz, address 0x44-0x47 found at boot). Pins 3 / 4 (the 74HC4066 attenuator of ask1) are
  no longer used.
    Hardware: AD9833 OUT -> Ra 1 k -> 3 x 330 Ohm ladder -> VREF (1.65 V); the four ladder nodes go to ADG729 1A-4A and
    DA goes through the 10 uF to E+.
      level index 0 -> byte 0x01 (1A) 100 %      index 2 -> byte 0x04 (3A) 33 %
      level index 1 -> byte 0x02 (2A)  67 %      index 3 -> byte 0x08 (4A)  0 % (DA on VREF, no carrier at E+)
      idle / off / boot = byte 0x00 (every switch open, the transmit network is off E+);  idle load -> 0x01 instead
    Schedule words (same format as ask1): bits 0-27 = AD9833 frequency word, bits 28-29 = level index of that step. Old
    tools send words < 2^28 -> index 0 -> full level. A step with frequency word 0 ('carrier off', OOK) = byte 0x08.
    The carrier is no longer stopped between steps: sgo puts the switch on 0x08 and starts the AD9833 on the first
    frequency of the schedule right away (it runs through the sgo delay, nothing reaches E+), and a step whose
    frequency is the one already running only writes the ADG729 byte (one I2C write, ~60 us, done in the step
    interrupt, only when the byte changes). So 2-ASK / 4-ASK / OOK symbols have a continuous carrier and every node
    stays on 1.65 V DC. A step with a NEW frequency (FSK) still does off -> level -> FREQ0 -> on as before. At the end:
    0x08 -> AD9833 off -> idle byte. DBPSK (psk cont) and ftx frames hold 0x01 for the whole transmission.
    New commands:
      amp <0-3>                        level index for cw (default 0)
      ask2 <hz> <baud> <hex|prbs> [n]  2-ASK now: bit 1 = 100 %, bit 0 = level index asklo (default 2 = 33 %)
      ask4 <hz> <baud> <hex|prbs> [n]  4-ASK now: 2 bits per symbol, (1,1) 100 % (1,0) 67 % (0,0) 33 % (0,1) 0 %
                                       hex = data bits, MSB first; prbs = PRBS9 (seed all ones); n = number of bits
                                       (default: all hex digits / 511). Replaces the schedule, starts after 1 s.
      asklo <1-3>                      2-ASK low level (3 = 0 % = OOK)
      idle open|load                   switch byte while nothing is sent: 0x00 (default) or 0x01 (old receive loading)
      adg [hex byte]                   ADG729 status + read-back / write one byte (bring-up)
      i2c                              scan the I2C bus
      dry 0|1                          1 = rehearsal: the AD9833 is never written (stays asleep) and levels 0-2 become
                                       0x00, so a schedule runs with its real timing and I2C traffic but nothing
                                       reaches E+
    The wave is SQUARE at boot (ask1 and older: sine): the ladder ends on VREF = 1.65 V, the mean of the square wave;
    a sine (mean ~0.33 V) would put a DC step on every level change. wave sine still selects it.
    sstat adds amp= adg= adg_err= (failed I2C writes since sgo) dry= fw=ask2. sgo / cw refuse when the ADG729 does not
    answer (with every switch open nothing would be sent).

  rev 2 (2026-11-24 night) - fail-safe. Reason: twice in one evening the board stopped answering after an A0 stream
  (stent_tx.py selftest: mono / x, then the next command got no reply), once with the carrier still on. The hung
  board did not answer USB control requests either (port open took the 5 s control time-out), so not only loop() but
  also the USB interrupt was dead, and it did not reboot (a fault reboots after 8 s). What is in here:
    1) Watchdog (WDOG1, 2 s). loop() feeds it. If loop() stops for 2 s - whatever the cause - the processor resets;
       setup() then writes the AD9833 off and opens every ADG729 switch, and the board is back on USB by itself (no
       replug). The carrier can outlive a hang by at most ~2 s + the boot time (~1 s).
       0.5 s before the reset a watchdog interrupt (priority 0, above everything) stores where the processor was
       stuck: the stacked PC and the active exception number. fs shows them after the reboot.
    2) Sample-clock stop with the PIT interrupt held off (stopSampling). IntervalTimer::end() removes its handler
       BEFORE it stops the PIT channel; a tick in between leaves the PIT interrupt asserted with nobody to clear its
       flag, and pit_isr is entered again and again at priority 32: loop() and the USB interrupt (112) never run
       again. That is the one interrupt storm this firmware can produce and it matches the symptoms above.
    3) cw hold: a continuous carrier (cw) is switched off when no byte has arrived from the host for cw_hold seconds
       (default 60; fs hold <s>, 0 = never). A schedule (sgo / ask2 / ask4) or a frame (ftx) ends by itself and is
       not cut. Any byte counts as 'host alive'; an unknown line such as "ka" is ignored and is a fine keep-alive.
    4) Stage marks: loop() / the stream stop / the output switching write a stage number that survives a reset.
    New commands:
      fs               "# fs rev=2 wdt_ms=2000 cw_hold_s=60 trips=<cw switched off by 3> boot=<power|wdog|...>
                        wdog_resets=<since power-up> last_stage=<hex, see stage()> last_cmd=<first 4 chars>
                        stuck_pc=<hex> stuck_exc=<exception number, 0 = loop(), 138 = PIT, 129 = USB>
                        mono=<streams started>:<samples>/<slips>/<bytes the USB core dropped> of the last stopped stream"
                       (last_* / stuck_* describe the run BEFORE the last reset)
      fs hold <s>      cw hold time (0 = never switch a cw off)
      fs hang [irq]    TEST: stop loop() on purpose (irq = with interrupts disabled) - the watchdog must reset
    sstat ends with rev=2. The first line of the help still starts with "TeensyTRX ask2" (the tools look for that).

  TeensyTRX - Teensy 4.0 + AD9833 갈바닉 트랜시버. 전극쌍 하나를 송신과 수신이 같이 쓴다.
  결선도: https://claude.ai/artifact/PS91BH4sDXn1jEkRHCZx21

  수신: TeensyLink.ino 와 같은 mono 스트림 (pin 14 = A0, 2바이트/샘플, 14비트, 3.3 V 풀스케일).
        receiver_gui.py, combo_sweep_rx.py 가 그대로 받는다. 표본화 방식(ADC 연속변환 + 타이머 읽기)도
        TeensyLink 와 똑같이 두어 지금까지의 측정과 같은 조건이다.
  송신: AD9833 — FSYNC 10, SDATA 11, SCLK 13, VCC 3.3 V. 13번은 보드 LED 핀이라 LED 는 쓰지 않는다.
        AD9833 쓰기는 핀을 직접 토글한다(한 단어 ~2.5 us). 2026-10-14 실제 보드에서 하드웨어 SPI(SPI 라이브러리)로는
        AD9833 가 응답하지 않았고 같은 핀 직접 토글은 됐다 - 원인 미확인이라 검증된 쪽을 쓴다.
        핀 2 는 출력이 켜져 있는 동안 HIGH (선택 사항인 74HC4066 송수신 스위치 제어).

  스윕 시각은 ADC 표본 시계에 묶여 있다. 표본 ISR(PIT, 우선순위 32)이 표본 수를 세다가 다음 전환 시각이
  되면 낮은 우선순위의 소프트웨어 인터럽트(IRQ_SOFTWARE)를 걸고, 거기서 AD9833 에 쓴다. 그래서
    1) AD9833 쓰기(수 us)가 표본을 빼먹게 하지 않는다 (PIT 채널은 인터럽트 하나를 같이 써서 다른 타이머로는 안 됨)
    2) 전환 시각이 Teensy 크리스털 정밀도로 정확하다 (Pico 의 늦음·밀림이 없다)
    3) 자기 캡처 안에서 송신 시작 표본 번호를 안다 (sstat 의 t0)
  주파수 전환은 사건마다 '끔 -> FREQ0 쓰기 -> 켬' 이다. Pico 처럼 쉬는 레지스터에 미리 써 두고 FSELECT 만 바꾸는
  방식(위상 연속)은 이 보드에서 출력 중 레지스터 쓰기가 깨져 쓰지 않는다 (2026-10-14). 단계마다 ~20 us 출력이 빈다.

  명령 (USB 시리얼, 줄 단위). 스트리밍 중에는 아무것도 출력하지 않는다 (바이너리 스트림이 깨지므로).
    mono [sps]        A0 스트림 시작 (기본 500000, 최대 1000000). 스윕 중이면 표본 시계는 그대로 둔다
    x                 스트림 중지 -> "# mono stopped <sps> S/s  slips N  total M"
    wave sine|square  파형 (사각 = OPBITEN|DIV2, VDD 풀스윙)
    cw <hz>           연속파 (스윕은 멈춤)
    off               출력 끔 (RESET + MCLK 정지 + DAC 차단). 수신할 때는 반드시 이 상태
    sclr              스윕 시간표 지우기
    sseg <step_us> <reps> <n>   구간 시작: 주파수 워드 n 개를 step_us 마다, 처음부터 reps 번
    sw <w1> <w2> ...  현재 구간에 주파수 워드(28비트 정수, f*2^28/25 MHz) 이어 붙이기
    sinfo             시간표 요약과 체크섬
    sgo [delay_ms] [sps]  delay 뒤 스윕 시작. 표본 시계가 꺼져 있으면 sps(기본 1 MS/s)로 켠다 - 뒤이어 보낼
                      mono 와 같은 값을 줄 것 (스윕 중에는 표본률을 바꾸지 않는다). 기본 1000 ms
    sstop             스윕 중지 + 출력 끔
    sstat             스윕 상태. t0 = 첫 출력 표본의 스트림 번호(마지막 mono 시작 기준, slips 반영)
    psk <sym_us> [차수] [cont|reset]  DBPSK 겹치기: 스윕 단계 안을 sym_us 마다 나눠 PRBS(기본 9 차, 시드 전부 1) 비트가
                      1 이면 위상을 180° 뒤집는다 (차동 부호). 단계 step_us 는 sym_us 의 배수여야 한다. psk 0 = 끔 (기본)
                      cont (기본, 2026-10-15 저녁): 시작 전에(꺼진 상태) FREQ0/FREQ1 = 시간표의 두 주파수, PHASE0/PHASE1 = 0/180°
                        를 넣고, 심볼마다 제어 워드 하나로 FSEL/PSEL 만 바꾼다 -> 위상 연속, 출력 공백 없음.
                        시간표에 0 이 아닌 주파수가 셋 이상이거나 0 워드(OOK)가 있으면 reset 으로 저절로 바뀐다 (sstat psk_mode).
                      reset (예전): 심볼마다 '끔 -> PHASE0 -> FREQ0 -> 켬'. 켜는 순간이 표본 틱(1 us)에 묶여 가끔 1 us 늦고,
                        168 kHz 에서 1 us = 60° 라 이웃 심볼이 2 us 어긋나면 판정이 뒤집혔다 (송신 루프백 BER ~1e-3, 2026-10-15)
    ?                 도움말

  DBPSK 비트열 (2026-10-27, trx_psk_link_v2.py 의 FEC 실측): psk 의 반전 비트를 PRBS 대신 호스트가 올린 비트열에서 꺼낸다.
  그 밖의 송신 (시간표·cont 방식·sgo·표본 시계) 은 psk 와 같다. 예전 원본 firmware/TeensyTRX_backup_20261027/.
    pbclr             비트열 지우기 (psk 는 PRBS 로 돌아감)
    pb <hex>          비트열 뒤에 이어 붙이기. 16 진 한 글자 = 4 비트, 최상위 먼저 (한 줄 ≤ ~500 글자), 최대 524288 비트
                      -> "# pb n=<지금 비트 수>" (넘치거나 16 진이 아닌 글자면 거기서 멈추고 " refused")
    pbinfo            "# pb n=<비트 수> ones=<1 의 수> checksum=<1 인 자리 (i + 1) 의 합, 2^32 순환> src=prbs|buf"
    psk <sym_us> buf [cont|reset]   반전 비트를 비트열에서: 심볼 k 의 반전 = 비트 k (sgo 마다 0 번부터, 끝나면 처음부터).
                      buf 없는 psk (예전 도구 모두) 는 PRBS 로 돌린다. sstat 끝에 psk_src=prbs|buf

  프레임 송신 (2026-10-22, 스텐트끼리 프로토콜 stent_node.py): DBPSK + 직접 확산(DSSS). 칩마다 위상 = 비트 위상 XOR 코드 칩,
  psk cont 와 같이 꺼진 상태에서 FREQ0 = 반송파, PHASE0/1 = 0/180° 를 넣고 칩마다 제어 워드(PSEL)만 바꾼다 (위상 연속).
  칩 시각은 스윕처럼 표본 시계에 묶인다: 비트 i 의 칩 j 시작 = i x bit_us + round(j x bit_us / n) us (2026-10-23: 칩 길이 대신
  비트 길이를 받아 속도를 1000/500/250/100/50 bps 처럼 딱 떨어지게). 스트리밍 중에도 받는다 (대답은 없음 - 스트림이 깨지므로).
    carrier <hz>      반송파 (기본 167840 Hz)
    code <slot> <n> <hex>   코드 칸 slot(0–5)에 칩 n 개(≤ 1023), 16진 MSB 먼저 (첫 칩 = 첫 글자의 최상위 비트). n = 1, hex 0 = 확산 없음
    cinfo             코드 칸 길이·체크섬
    ftx <slot> <bit_us> <delay_us> <nbits> <hex>   delay 뒤 프레임 송신. 비트 위상(이미 차동 부호화된 0/1, MSB 먼저) nbits 개(≤ 512)
                      x 코드 칸 slot 의 칩 n 개. 길이 = nbits x bit_us (≤ 20 s), 칩 ≥ 2 us. 끝나면 출력 끔
    fstat             프레임 송신 상태 (보낸 수, 거절 수, 늦음)
    mark 0|1          1 이면 출력이 켜져 있는 표본의 14비트 값 최하위 비트를 1 로 (12비트 ADC 라 원래 늘 0) — 호스트가
                      자기 송신 구간을 표본 단위로 안다. 기본 0 (예전 도구와 같은 스트림)

  필요 라이브러리: ADC (pedvide/ADC) — Teensyduino 에 들어 있다.
*/
#include <ADC.h>
#include <IntervalTimer.h>
#include <Wire.h>

const uint8_t  PIN_RX    = A0;          // = 14
const uint8_t  PIN_FSYNC = 10;
const uint8_t  PIN_SDATA = 11;
const uint8_t  PIN_SCLK  = 13;
const uint8_t  PIN_TR    = 2;
const uint8_t  ADG_OPEN  = 0x00;        // ask2: every switch open (transmit network off E+)
const uint8_t  ADG_L0    = 0x08;        // ask2: 4A = DA on VREF, no carrier at E+
const uint8_t  ADG_LOAD  = 0x01;        // ask2: 1A = full level / the old receive loading
const uint8_t  ADG_LVL[4] = {0x01, 0x02, 0x04, 0x08};   // ask2: level index 0-3 -> 1A 100 %, 2A 67 %, 3A 33 %, 4A 0 %
const uint32_t ADG_TIMEOUT_US = 300;    // ask2: one I2C write takes ~60 us at 400 kHz
const uint32_t WMASK     = 0x0FFFFFFF;  // ask1: frequency bits of a schedule word
const uint32_t AMPMASK   = 0x30000000;  // ask1: level bits of a schedule word
const uint32_t RING      = 1u << 16;    // 65536 샘플 (131 kB)
const uint32_t SPS_MAX   = 1000000;
const uint32_t SPS_MIN   = 1000;
const uint32_t SPS_SWEEP = 1000000;     // 스트림 없이 스윕만 할 때의 표본 시계
const uint32_t PIT_HZ    = 24000000;    // IntervalTimer(PIT) 클럭
const uint16_t MAXW      = 4096;        // 주파수 워드 수
const uint8_t  MAXSEG    = 16;

// AD9833 제어 비트
const uint16_t B28 = 1 << 13, RST = 1 << 8, SL1 = 1 << 7, SL12 = 1 << 6, FSEL = 1 << 11, PSEL = 1 << 10;
const uint16_t OPBITEN = 1 << 5, DIV2 = 1 << 3;
const uint16_t OFFW = B28 | RST | SL1 | SL12;   // MCLK 정지 + DAC 차단: 사인·사각 모두 출력 없음
const double   MCLK = 25000000.0;

ADC           *adc = new ADC();
IntervalTimer  tmr;

// ---- 스트림 (TeensyLink 와 같음)
volatile uint16_t ring[RING];
volatile uint32_t wr = 0, rd = 0, slips = 0;
volatile bool     streaming = false;

// ---- 표본 시계
volatile uint32_t samp = 0;             // 표본 ISR 마다 +1 (32비트 순환, 비교는 순환에 안전하게)
bool     sampling = false;
uint32_t cur_sps = 0, tps = 24;         // tps = 표본당 PIT 틱
uint32_t stream0 = 0;                   // 마지막 mono 시작 때의 samp

// ---- 스윕
struct Seg { uint16_t i0, n, want; uint32_t step_us, reps; };
Seg      segs[MAXSEG];
uint8_t  nseg = 0;
uint32_t fw[MAXW];
uint16_t nw = 0;
enum { S_IDLE, S_ARMED, S_RUN, S_DONE, S_STOP };
volatile uint8_t  sw_state = S_IDLE;
volatile bool     armed = false;        // 표본 ISR 이 다음 사건을 기다리는 중
volatile uint32_t ev_samp = 0;          // 다음 사건의 표본 시각
uint32_t t0_samp = 0;                   // 첫 출력의 표본 시각
volatile uint32_t t0_slips = 0;
volatile bool     t0_seen = false;
uint8_t  es; uint32_t er; uint16_t ei;  // 다음에 실행할 사건 (구간, 반복, 단계)
bool     eend;                          // 다음 사건이 '출력 끔'
uint64_t et_us;                         // 다음 사건 시각 [us, 스윕 T0 기준]
uint8_t  cur_reg = 1;                   // 지금 출력 중인 FREQ 레지스터
volatile uint32_t ev_done = 0, late_max = 0, late_n = 0;
uint16_t runw = B28 | OPBITEN | DIV2;   // ask2: square at boot (the ladder ends on VREF = 1.65 V = the square's mean); wave sine -> B28
// ---- DBPSK 겹치기 (2026-10-15): 단계(주파수 = FSK 비트) 안을 psk_us 마다 나눠 위상을 PRBS 로 차동 반전
uint32_t psk_us = 0;                    // 0 = 끔
uint8_t  psk_order = 9;
uint32_t psk_lfsr = 0;
uint8_t  psk_ph = 0;                    // 지금 위상 (1 = 180°)
uint32_t sub_i = 0;                     // 단계 안 심볼 번호
uint32_t cur_w = 0;                     // 지금 단계의 주파수 워드
volatile uint32_t psk_n = 0;            // 보낸 심볼 수
bool     psk_cont_req = true;           // psk 명령이 요청한 방식 (cont / reset)
bool     psk_cont = false;              // 이번 스윕에서 실제로 쓰는 방식 (sweepGo 가 정한다)
uint32_t psk_wA = 0, psk_wB = 0;        // cont: FREQ0 / FREQ1 에 넣은 두 워드
uint16_t psk_ctrl = 0;                  // cont: 마지막으로 쓴 제어 워드
// ---- DBPSK 비트열 (2026-10-27, trx_psk_link_v2): 반전 비트를 PRBS 대신 호스트가 올린 비트열에서 (FEC 부호 프레임을 실제로 보내기)
const uint32_t MAXPB = 1u << 19;        // 524288 비트 = 64 kB
DMAMEM uint32_t pbuf[MAXPB / 32];       // 비트 i = pbuf[i / 32] 의 (31 − i % 32) 번 비트 (RAM2)
uint32_t pb_n = 0;                      // 올린 비트 수
uint32_t pb_i = 0;                      // 다음에 꺼낼 자리 (sgo 마다 0 부터)
bool     psk_buf = false;               // true = 반전 비트를 pbuf 에서 (psk <sym_us> buf). psk 를 buf 없이 주면 false
bool     cw_on = false;
double   cw_hz = 0;
// ---- 프레임 송신 (2026-10-22): 코드 칩 x 프레임 비트 위상을 칩마다 PSEL 로 (DBPSK + 직접 확산)
const uint16_t MAXCHIP = 1023;          // 코드 길이 최대 (2^10 - 1)
const uint8_t  NSLOT   = 6;             // 코드 칸 (확산 배수마다 하나; 2026-10-23 칩 32 us·SF 5 단계로 4 -> 6)
const uint16_t MAXFBIT = 512;           // 프레임 비트 최대
const uint64_t FR_MAX_US = 20000000ull; // 프레임 하나의 최대 송신 시간 (안전 상수)
uint8_t  cslot[NSLOT][MAXCHIP];         // 칩 0/1
uint16_t clen[NSLOT] = {0, 0, 0, 0, 0, 0};
uint8_t  fbits[MAXFBIT];                // 비트마다 위상 0/1 (= 0°/180°, 호스트가 차동 부호화해서 보낸다)
uint16_t fnbits = 0, fbit_i = 0, fchip_i = 0;
uint8_t  fslot = 0;
uint32_t fbit_us = 0;
uint32_t fr_t0 = 0;                     // 첫 칩의 표본 시각
uint32_t fr_w = 0;                      // 반송파 워드 (carrier, setup 에서 167840 Hz)
uint16_t fr_ctrl = 0;                   // 마지막으로 쓴 제어 워드
enum { F_IDLE, F_ARMED, F_RUN };
volatile uint8_t  fr_state = F_IDLE;
volatile uint32_t fr_count = 0, fr_refused = 0;
volatile bool     tx_on = false;        // AD9833 출력이 켜져 있는 동안 (mark 1 이면 스트림에 싣는다)
bool     mark_en = false;

// ------------------------------------------------------------------ rev 2: fail-safe (watchdog, stage marks, cw hold)
const uint16_t WDT_WT      = 3;         // WDOG1 time-out = (WT + 1) x 0.5 s = 2 s
const uint32_t WDT_FEED_MS = 50;        // loop() feeds at most this often
const uint32_t FS_MAGIC    = 0x46533032;   // "FS02": the marks below are ours and valid
uint32_t cw_hold_ms = 60000;            // cw is switched off after this long without a byte from the host (0 = never)
uint32_t host_ms = 0;                   // millis() of the last byte from the host
uint32_t fs_trips = 0;                  // cw switched off by the hold since boot
uint32_t wdt_fed = 0;
uint32_t boot_srsr = 0, boot_stage = 0, boot_cmd = 0, boot_pc = 0, boot_exc = 0, wdog_resets = 0;
uint32_t cur_stage = 0;
uint32_t usb_short = 0, mono_n = 0, last_wr = 0, last_slips = 0, last_short = 0;   // stream bookkeeping for 'fs' (mono=)
// The marks live in CrashReport's breadcrumb row (0x2027FFC0, top of RAM2): it is not cleared at boot and
// CrashReport.breadcrumb() flushes the cache row, so a value written just before a reset is still there after it.
//   slot 1 stage | state bits   2 first 4 chars of the command   3 FS_MAGIC   4 watchdog resets since power-up
//   slot 5 stacked PC, 6 stacked xPSR of whatever the watchdog interrupt interrupted
typedef struct crashreport_breadcrumbs_struct FsCrumbs;   // the core's layout: bitmask, value[6], checksum
static volatile FsCrumbs *const FSBC = (volatile FsCrumbs *)0x2027FFC0;
// stage numbers: 0x01 setup  0x10 loop idle  0x11 / 0x12 stream write before / after  0x20 in a command
//   0x30 stopMono  0x31 its delay  0x32 its reply  0x38 stopSampling before / 0x39 after the timer stop / 0x3A ADC stopped
//   0x40 outputOff  0x41 switch to VREF  0x42 AD9833 off  0x43 idle byte  0x44 sample clock
//   0x50 startSampling  0x51 ADC set up  0x52 timer started   0x60 startCW   0x70 cw hold trip   0x7F fs hang
// state bits: 0x10000 cw on, 0x20000 streaming, 0x40000 sample clock on, 0x80000 output on (tx_on)
static inline void stage(uint32_t s) {
  s |= (cw_on ? 0x10000u : 0) | (streaming ? 0x20000u : 0) | (sampling ? 0x40000u : 0) | (tx_on ? 0x80000u : 0);
  if (s == cur_stage) return;
  cur_stage = s;
  CrashReport.breadcrumb(1, s);
}
static inline void wdtFeed() { WDOG1_WSR = 0x5555; WDOG1_WSR = 0xAAAA; }
extern "C" void wdog_capture(uint32_t *frame) {     // frame = r0 r1 r2 r3 r12 lr pc xpsr of the interrupted code
  FSBC->value[4] = frame[6];
  FSBC->value[5] = frame[7];
  FSBC->bitmask |= 0x30;
  arm_dcache_flush((void *)FSBC, sizeof(FsCrumbs));
  WDOG1_WICR = (uint16_t)(1 << 14);                 // clear the interrupt (WTIS), leave it disabled: the reset follows
  NVIC_DISABLE_IRQ(IRQ_WDOG1);
}
__attribute__((naked)) static void wdog_isr() { __asm__ volatile("mrs r0, msp\n" "b wdog_capture\n"); }
static void wdtBegin() {
  CCM_CCGR3 |= CCM_CCGR3_WDOG1(CCM_CCGR_ON);
  attachInterruptVector(IRQ_WDOG1, wdog_isr);
  NVIC_SET_PRIORITY(IRQ_WDOG1, 0);                  // above the sample interrupt (32): it must get through a storm
  NVIC_ENABLE_IRQ(IRQ_WDOG1);
  WDOG1_WMCR = 0;                                   // power-down counter off
  WDOG1_WICR = (uint16_t)((1 << 15) | (1 << 14) | 1);   // WIE, clear WTIS, interrupt 0.5 s before the time-out
  // SRS = 1 and WDA = 1 are 'no software reset / do not pull WDOG_B now'; WDBG stops the count under a debugger
  // (the bootloader chip halts the processor over JTAG when it programs it). WDE cannot be cleared again.
  WDOG1_WCR = WDOG_WCR_WT(WDT_WT) | WDOG_WCR_SRS | WDOG_WCR_WDA | WDOG_WCR_WDBG;
  WDOG1_WCR |= WDOG_WCR_WDE;
  wdtFeed();
}
static void bootCause() {                            // first thing in setup(): why did we boot, where was the last run
  boot_srsr = SRC_SRSR;
  SRC_SRSR = boot_srsr;                              // write-1-to-clear: the next boot shows its own cause only
  bool valid = FSBC->value[2] == FS_MAGIC;
  if (valid) { boot_stage = FSBC->value[0]; boot_cmd = FSBC->value[1]; wdog_resets = FSBC->value[3];
               boot_pc = FSBC->value[4]; boot_exc = FSBC->value[5] & 0x1FF; }
  bool wd = boot_srsr & (SRC_SRSR_WDOG_RST_B | SRC_SRSR_WDOG3_RST_B);
  if (!wd) { boot_pc = 0; boot_exc = 0; }
  if (boot_srsr & SRC_SRSR_IPP_RESET_B) wdog_resets = 0;          // power-up
  if (wd) wdog_resets++;
  CrashReport.breadcrumb(3, FS_MAGIC);
  CrashReport.breadcrumb(4, wdog_resets);
  CrashReport.breadcrumb(5, 0);
  CrashReport.breadcrumb(6, 0);
  CrashReport.breadcrumb(2, 0);
}
static const char *bootName() {
  if (boot_srsr & (SRC_SRSR_WDOG_RST_B | SRC_SRSR_WDOG3_RST_B)) return "wdog";
  if (boot_srsr & SRC_SRSR_IPP_RESET_B) return "power";
  if (boot_srsr & SRC_SRSR_LOCKUP_SYSRESETREQ) return "soft";    // software reset / crash reboot / lockup
  if (boot_srsr & (SRC_SRSR_JTAG_RST_B | SRC_SRSR_JTAG_SW_RST)) return "jtag";
  if (boot_srsr & SRC_SRSR_IPP_USER_RESET_B) return "user";
  return "other";
}

// ------------------------------------------------------------------ AD9833
volatile uint16_t dbgw[96]; volatile uint8_t ndbg = 0;   // 디버그: 최근 sgo 이후 쓴 단어
bool dry = false;                       // ask2: rehearsal - the AD9833 is never written, levels 0-2 become 0x00
static inline void adw(uint16_t v) {
  if (ndbg < 96) dbgw[ndbg++] = v;      // 16비트, MSB 먼저. SCLK 는 HIGH 에서 쉬고 AD9833 는 하강 에지에서 읽는다
  if (dry) return;
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

static inline void loadReg(uint8_t reg, uint32_t w) {     // reg 0 = FREQ0, 1 = FREQ1
  uint16_t base = reg ? 0x8000 : 0x4000;
  adw(base | (w & 0x3FFF));
  adw(base | ((w >> 14) & 0x3FFF));
}

// ---- ask2: per-step level = one byte to the ADG729 (I2C, LPI2C1 = Wire on pins 18 / 19)
uint8_t  cw_amp = 0;                    // level index for cw (amp command)
uint8_t  ask_lo = 2;                    // 2-ASK low level index (asklo command)
uint8_t  adg_addr = 0x44;
bool     adg_found = false;
uint8_t  adg_idle = ADG_OPEN;           // switch byte while nothing is sent (idle command)
volatile uint8_t  cur_adg = 0xFF;       // last byte the ADG729 acknowledged (0xFF = none yet)
volatile uint32_t adg_err = 0, adg_n = 0;
uint32_t run_f = 0;                     // frequency word the AD9833 is running on (0 = not running)

// One write, polled (no Wire calls: this also runs in the step interrupt, where Wire's 50 ms timeouts and yield() do not
// belong). Wire.begin() / setClock() in setup() configure the pins and the 400 kHz timing.
static bool adgWrite(uint8_t req) {
  uint8_t b = (dry && req != ADG_L0) ? ADG_OPEN : req;                   // rehearsal: no ladder tap ever reaches DA
  IMXRT_LPI2C_t *p = &IMXRT_LPI2C1;
  uint32_t t0 = micros();
  while ((p->MSR & LPI2C_MSR_BBF) && !(p->MSR & LPI2C_MSR_MBF))          // bus still busy with the previous STOP
    if (micros() - t0 > ADG_TIMEOUT_US) { adg_err++; return false; }
  p->MSR = 0x00007F00;                                                   // clear old flags
  p->MTDR = LPI2C_MTDR_CMD_START | (uint32_t)(adg_addr << 1);
  p->MTDR = LPI2C_MTDR_CMD_TRANSMIT | b;
  p->MTDR = LPI2C_MTDR_CMD_STOP;
  for (;;) {
    uint32_t s = p->MSR;
    if (s & (LPI2C_MSR_ALF | LPI2C_MSR_FEF | LPI2C_MSR_NDF | LPI2C_MSR_PLTF)) {
      p->MCR |= LPI2C_MCR_RTF | LPI2C_MCR_RRF;                           // drop what is left in the FIFOs
      if (s & (LPI2C_MSR_NDF | LPI2C_MSR_PLTF)) p->MTDR = LPI2C_MTDR_CMD_STOP;
      break;
    }
    if ((s & LPI2C_MSR_SDF) && !(p->MFSR & 0x07)) { cur_adg = req; adg_n++; return true; }
    if (micros() - t0 > ADG_TIMEOUT_US) { p->MCR |= LPI2C_MCR_RTF | LPI2C_MCR_RRF; break; }
  }
  adg_err++;
  return false;
}
static inline void adgSet(uint8_t b) { if (b != cur_adg) adgWrite(b); }   // step interrupt: only when the byte changes
static bool adgMain(uint8_t b) {                                         // main context: keep the step interrupt out
  NVIC_DISABLE_IRQ(IRQ_SOFTWARE);
  bool ok = adgWrite(b);
  NVIC_ENABLE_IRQ(IRQ_SOFTWARE);
  return ok;
}

static inline uint32_t us2samp(uint64_t us) {             // 스윕 시각 [us] -> 표본 수
  return (uint32_t)((us * (PIT_HZ / 1000000)) / tps);
}

// ------------------------------------------------------------------ 인터럽트
static void sample_isr() {
  uint16_t v = (uint16_t)adc->adc0->analogReadContinuous();
  uint32_t s = samp++;
  if (streaming) {
    if (wr - rd >= RING) { slips++; }
    else { ring[wr & (RING - 1)] = v | ((mark_en && tx_on) ? 0x8000 : 0); wr++; }   // 비트 15 = 송신 중 (loop 에서 최하위로)
  }
  if (armed && (int32_t)(s - ev_samp) >= 0) {
    armed = false;
    NVIC_SET_PENDING(IRQ_SOFTWARE);
  }
}

// AD9833 는 출력이 돌고 있을 때 주파수 레지스터를 쓰면(쉬는 레지스터라도) 출력이 꺼지거나 엉뚱한 값이 됐다.
// 끈 상태(RESET+슬립)에서 쓰면 항상 정확했다 (2026-10-14 실측, 원인 미확인). 그래서 사건마다
// '끔 -> FREQ0 쓰기 -> 켬' 을 한다. 대가: 단계마다 ~20 us 출력이 비고 위상이 0 에서 다시 시작한다.
static inline uint8_t psk_next() {     // 호스트 ook_rx.prbs() 와 같은 LFSR (시드 전부 1)
  if (psk_buf) {                        // 올린 비트열 (2026-10-27): 심볼 k 의 반전 = 비트 k, 끝나면 처음부터
    uint8_t bit = (pbuf[pb_i >> 5] >> (31 - (pb_i & 31))) & 1;
    if (++pb_i >= pb_n) pb_i = 0;
    return bit;
  }
  uint8_t a, b;
  switch (psk_order) { case 7: a = 6; b = 5; break; case 11: a = 10; b = 8; break; case 15: a = 14; b = 13; break;
                       default: a = 8; b = 4; }
  uint32_t mask = (1u << psk_order) - 1;
  uint8_t bit = ((psk_lfsr >> a) ^ (psk_lfsr >> b)) & 1;
  psk_lfsr = ((psk_lfsr << 1) | bit) & mask;
  return bit;
}

// 프레임 송신의 칩 사건 하나. 첫 사건은 RESET·슬립 해제(= 출력 시작), 그다음은 위상이 바뀔 때만 제어 워드를 쓴다.
// 마지막 칩 다음 사건에서 출력을 끈다. 칩 시각 = fr_t0 + (비트 시작 + 반올림한 칩 자리) — 늦음이 쌓이지 않는다
static void frame_step() {
  uint32_t lt = samp - ev_samp;
  if (lt > late_max) late_max = lt;
  if (lt > 3) late_n++;
  if (fbit_i >= fnbits) {
    adgSet(ADG_L0);                                         // ask2: DA on VREF, then stop the carrier, then idle
    adw(OFFW); fr_ctrl = OFFW;
    tx_on = false;
    digitalWriteFast(PIN_TR, LOW);
    fr_state = F_IDLE; fr_count++;
    adgSet(adg_idle);
    return;
  }
  uint8_t ph = fbits[fbit_i] ^ cslot[fslot][fchip_i];
  uint16_t c = runw | (ph ? PSEL : 0);
  if (fr_state == F_ARMED) { adw(c); fr_ctrl = c; fr_state = F_RUN; tx_on = true; }
  else if (c != fr_ctrl) { adw(c); fr_ctrl = c; }
  if (++fchip_i >= clen[fslot]) { fchip_i = 0; fbit_i++; }
  uint64_t n = clen[fslot];                                 // 다음 칩 시작 [us] = 비트 시작 + round(칩 x bit_us / n) (끝이면 프레임 끝)
  uint64_t t_us = (uint64_t)fbit_i * fbit_us + ((uint64_t)fchip_i * fbit_us * 2 + n) / (2 * n);
  ev_samp = fr_t0 + us2samp(t_us);                          // 순서: 시각 먼저, 그다음 armed
  armed = true;
}

static void step_isr() {                                   // 낮은 우선순위: 표본 ISR 이 끼어들 수 있다
  if (fr_state != F_IDLE) { frame_step(); return; }
  if (sw_state != S_ARMED && sw_state != S_RUN) return;
  uint32_t lt = samp - ev_samp;                             // 늦음 [표본]
  if (lt > late_max) late_max = lt;
  if (lt > 3) late_n++;
  if (!eend) {
    if (sub_i == 0) cur_w = fw[segs[es].i0 + ei];           // 새 단계: 주파수 워드
    if (psk_us && psk_cont) {                               // DBPSK 위상 연속: 제어 워드 하나 (바뀔 때만)
      psk_ph ^= psk_next();
      uint16_t c = runw | (cur_w == psk_wB ? FSEL : 0) | (psk_ph ? PSEL : 0);
      if (c != psk_ctrl || sw_state == S_ARMED) { adw(c); psk_ctrl = c; }   // 첫 사건은 RESET·슬립 해제 = 출력 시작
      psk_n++;
    } else {
      uint32_t f = cur_w & WMASK;
      uint8_t lvl = f ? ADG_LVL[(cur_w >> 28) & 3] : ADG_L0;   // ask2: frequency word 0 = no carrier at E+
      if (psk_us) {                                         // DBPSK 리셋 방식: PRBS 비트가 1 이면 위상 반전
        adw(OFFW);
        psk_ph ^= psk_next();
        adw(0xC000 | (psk_ph ? 2048 : 0));                  // PHASE0
        psk_n++;
        adgSet(lvl);                                        // level, while the output is off
        loadReg(0, f);
        adw(runw);                                          // FREQ0, RESET·슬립 해제 -> 출력
        run_f = f;
      } else if (f == 0 || f == run_f) {
        adgSet(lvl);                                        // ask2: same carrier (or none) - only the level switch
      } else {                                              // new frequency: off -> level -> FREQ0 -> on, as before
        adw(OFFW);
        adgSet(lvl);
        loadReg(0, f);
        adw(runw);
        run_f = f;
      }
    }
    if (sw_state == S_ARMED) { sw_state = S_RUN; t0_slips = slips; t0_seen = true; tx_on = true; }
    ev_done++;
    bool adv = true;                                        // 다음 사건으로
    if (psk_us) {
      et_us += psk_us;
      if (++sub_i < segs[es].step_us / psk_us) adv = false; else sub_i = 0;
    } else {
      et_us += segs[es].step_us;
    }
    if (adv && ++ei >= segs[es].n) { ei = 0; if (++er >= segs[es].reps) { er = 0; if (++es >= nseg) eend = true; } }
    ev_samp = t0_samp + us2samp(et_us);                     // 순서: 시각 먼저, 그다음 armed
    armed = true;
  } else {
    adgSet(ADG_L0);                                         // ask2: DA on VREF, then stop the carrier, then idle
    adw(OFFW);
    run_f = 0;
    sw_state = S_DONE;
    adgSet(adg_idle);
    tx_on = false;
    digitalWriteFast(PIN_TR, LOW);
  }
}

// ------------------------------------------------------------------ 표본 시계
static void startSampling(uint32_t sps) {
  if (sps < SPS_MIN) sps = SPS_MIN;
  if (sps > SPS_MAX) sps = SPS_MAX;
  tps = (PIT_HZ + sps / 2) / sps;
  stage(0x50);
  adc->adc0->setAveraging(1);
  adc->adc0->setResolution(12);
  adc->adc0->setConversionSpeed(ADC_CONVERSION_SPEED::VERY_HIGH_SPEED);
  adc->adc0->setSamplingSpeed(ADC_SAMPLING_SPEED::VERY_HIGH_SPEED);
  adc->adc0->startContinuous(PIN_RX);
  delayMicroseconds(200);
  stage(0x51);
  cur_sps = PIT_HZ / tps;
  sampling = true;
  tmr.begin(sample_isr, (float)tps / (float)(PIT_HZ / 1000000));   // 정확히 tps 틱
  tmr.priority(32);
  stage(0x52);
}

// rev 2: IntervalTimer::end() (Teensyduino 1.62 cores/teensy4/IntervalTimer.cpp) empties its handler slot and only
// THEN stops the PIT channel (TCTRL = 0, TFLG = 1). A PIT tick between the two - the timer runs at up to 1 MS/s -
// enters pit_isr with an empty slot: it returns without clearing TFLG, the interrupt stays asserted and is entered
// again at once, for ever, at priority 32. loop() never gets another instruction and the USB interrupt (112) never
// runs: the board is silent and whatever the AD9833 was doing goes on. So the PIT interrupt is held off at the NVIC
// while the timer stops; by the time it is let in again the channel is stopped and its flag cleared.
// Measured 2026-11-25 (trx_switch_test.py --dry, 1 MS/s): the old stop hung 8 times in ~900 stream stops, every time
// caught by the watchdog at stage 0x38 with the PC inside IntervalTimer::end() / pit_isr; this stop: 0 in 5700.
static void stopSampling() {
  if (!sampling) return;
  stage(0x38);
#ifndef FS_TEST_UNGUARDED_STOP                   // (test builds only: the stop as it was before rev 2)
  NVIC_DISABLE_IRQ(IRQ_PIT);
  __asm__ volatile("dsb\n" "isb\n");
#endif
  tmr.end();
#ifndef FS_TEST_UNGUARDED_STOP
  NVIC_CLEAR_PENDING(IRQ_PIT);
  NVIC_ENABLE_IRQ(IRQ_PIT);
#endif
  sampling = false;
  stage(0x39);
  adc->adc0->stopContinuous();
  stage(0x3A);
}

static bool sweepActive() { return sw_state == S_ARMED || sw_state == S_RUN; }
static bool clockNeeded() { return sweepActive() || fr_state != F_IDLE; }   // 표본 시계를 끄면 안 되는 송신

// ------------------------------------------------------------------ 출력
static void outputOff() {
  uint32_t back = cur_stage & 0xFFFF;
  stage(0x40);
  noInterrupts();
  armed = false;
  if (sweepActive()) sw_state = S_STOP;   // 걸려 있던 step_isr 은 이 상태를 보고 그냥 나간다
  fr_state = F_IDLE;
  interrupts();
  stage(0x41);
  if (cw_on || tx_on || run_f) adgMain(ADG_L0);   // ask2: a running carrier leaves E+ before the AD9833 stops
  stage(0x42);
  adw(OFFW);
  run_f = 0;
  stage(0x43);
  adgMain(adg_idle);
  cw_on = false;
  tx_on = false;
  digitalWriteFast(PIN_TR, LOW);
  stage(0x44);
  if (!streaming) stopSampling();
  stage(back);
}

static uint32_t fword(double hz) {
  return (uint32_t)(hz * 268435456.0 / MCLK + 0.5) & 0x0FFFFFFF;
}

static bool startCW(double hz) {
  outputOff();
  stage(0x60);
  uint32_t w = fword(hz);
  cw_hz = w * MCLK / 268435456.0;
  adw(OFFW);
  adw(0xC000); adw(0xE000);               // PHASE0 = PHASE1 = 0
  loadReg(0, w);
  if (!adgMain(ADG_L0)) return false;     // ask2: DA on VREF while the AD9833 starts
  digitalWriteFast(PIN_TR, HIGH);
  adw(runw);                              // FREQ0, RESET·슬립 해제 -> 출력
  run_f = w;
  adgMain(ADG_LVL[cw_amp]);
  cw_on = true;
  tx_on = true;
  cur_reg = 0;
  return true;
}

// ------------------------------------------------------------------ 스윕
static uint32_t schedSum(uint64_t *total_us, uint32_t *events) {
  uint32_t cs = 0; uint64_t t = 0; uint32_t ev = 0;
  for (uint8_t k = 0; k < nseg; k++) {
    cs += segs[k].step_us ^ segs[k].reps ^ segs[k].n;
    t  += (uint64_t)segs[k].step_us * segs[k].n * segs[k].reps;
    ev += segs[k].n * segs[k].reps * (psk_us ? segs[k].step_us / psk_us : 1);
  }
  for (uint16_t i = 0; i < nw; i++) cs += fw[i];
  if (total_us) *total_us = t;
  if (events) *events = ev;
  return cs;
}

static const char *sweepGo(uint32_t delay_ms, uint32_t sps) {
  if (nseg == 0) return "no schedule";
  for (uint8_t k = 0; k < nseg; k++) if (segs[k].n == 0 || segs[k].n != segs[k].want || segs[k].reps == 0) return "incomplete segment";
  if (psk_us) for (uint8_t k = 0; k < nseg; k++) if (segs[k].step_us % psk_us) return "psk: step_us not a multiple of sym_us";
  psk_lfsr = (1u << psk_order) - 1; psk_ph = 0; sub_i = 0; psk_n = 0; pb_i = 0;
  // 위상 연속 방식이 되는지: 시간표의 주파수가 0 이 아닌 두 개 이하일 때만 (0 워드 = OOK 는 리셋 방식이어야 0 V 에 선다)
  psk_cont = psk_us && psk_cont_req;
  psk_wA = nw ? fw[0] : 0; psk_wB = psk_wA;
  for (uint16_t i = 0; i < nw && psk_cont; i++) {
    if (fw[i] == 0 || (fw[i] & AMPMASK)) psk_cont = false;   // ask1: levels need the reset path
    else if (fw[i] != psk_wA) { if (psk_wB == psk_wA) psk_wB = fw[i]; else if (fw[i] != psk_wB) psk_cont = false; }
  }
  outputOff();                            // 스트림이 없으면 표본 시계도 여기서 꺼진다
  if (!sampling) startSampling(sps ? sps : SPS_SWEEP);   // 다음 mono 와 같은 표본률이어야 스트림 중에 바꾸지 않는다
  adw(OFFW);
  adg_err = 0;
  if (psk_cont) {                         // 꺼진 상태에서 레지스터를 다 넣어 둔다 (출력 중 레지스터 쓰기는 이 보드에서 깨짐)
    loadReg(0, psk_wA & WMASK); loadReg(1, psk_wB & WMASK);
    adw(0xC000); adw(0xE000 | 2048);      // PHASE0 = 0°, PHASE1 = 180°
    psk_ctrl = OFFW;
    if (!adgMain(ADG_LVL[0])) return "ADG729 not answering (level switch)";   // DBPSK: full level throughout
  } else {
    adw(0xC000); adw(0xE000);             // PHASE0 = PHASE1 = 0
    if (!adgMain(ADG_L0)) return "ADG729 not answering (level switch)";       // ask2: DA on VREF, nothing at E+ yet
    if (!psk_us) {                        // ask2: start the carrier now, on the schedule's first frequency
      uint32_t f0 = 0;
      for (uint16_t i = 0; i < nw && !f0; i++) f0 = fw[i] & WMASK;
      if (f0) { loadReg(0, f0); adw(runw); run_f = f0; }
    }
  }
  cur_reg = 0; es = 0; er = 0; ei = 0; eend = false; et_us = 0;
  ev_done = 0; late_max = 0; late_n = 0; t0_seen = false; t0_slips = 0;
  ndbg = 0;
  digitalWriteFast(PIN_TR, HIGH);         // 스위치를 출력보다 먼저 닫는다
  if (delay_ms < 5) delay_ms = 5;
  noInterrupts();
  t0_samp = samp + us2samp((uint64_t)delay_ms * 1000);
  ev_samp = t0_samp;
  sw_state = S_ARMED;
  armed = true;
  interrupts();
  return nullptr;
}

// ------------------------------------------------------------------ 프레임 송신
static int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// ------------------------------------------------------------------ ask2: 2-ASK / 4-ASK straight from one command
// ask2 / ask4 <hz> <baud> <hex|prbs> [nbits]: writes the schedule (one step per symbol, level index in bits 28-29) and
// starts it after 1 s, exactly as sclr / sseg / sw / sgo 1000 would. 4-ASK pairs are Gray-mapped like decode_trx_mfsk:
// (1,1) index 0, (1,0) index 1, (0,0) index 2, (0,1) index 3, first bit first.
uint32_t ask_step = 0; uint16_t ask_nsym = 0; double ask_hz = 0;
static const char *askGo(int M, char *p) {
  if (clockNeeded()) return "busy (sweep or frame running)";
  double hz = strtod(p, &p), baud = strtod(p, &p);
  while (*p == ' ') p++;
  if (hz <= 0 || hz > 12.5e6 || !fword(hz)) return "bad frequency";
  if (baud < 1 || baud > 5000) return "bad baud (1-5000)";
  bool prbs = !strncmp(p, "prbs", 4);
  char *hex = p;
  while (*p && *p != ' ') p++;
  long hexbits = prbs ? 0 : 4 * (long)(p - hex);
  long n = strtol(p, nullptr, 10);
  if (n <= 0) n = prbs ? 511 : hexbits;
  if (!prbs && (hexbits == 0 || n > hexbits)) return "fewer hex digits than bits";
  if (!prbs) for (long i = 0; i < (n + 3) / 4; i++) if (hexval(hex[i]) < 0) return "bad hex";
  if (M == 4 && (n & 1)) n++;                               // 4-ASK: an odd PRBS length gets one more bit
  if (M == 4 && !prbs && n > hexbits) return "4-ASK needs an even number of bits";
  long nsym = M == 4 ? n / 2 : n;
  if (nsym < 1 || nsym > MAXW) return "too many symbols (max 4096)";
  uint32_t f = fword(hz), lfsr = 0x1FF;
  long bi = 0;
  nseg = 0; nw = 0;
  for (long s = 0; s < nsym; s++) {
    uint8_t b[2] = {0, 0};
    for (int k = 0; k < (M == 4 ? 2 : 1); k++, bi++) {
      if (prbs) { uint8_t x = ((lfsr >> 8) ^ (lfsr >> 4)) & 1; lfsr = ((lfsr << 1) | x) & 0x1FF; b[k] = x; }   // = psk order 9
      else b[k] = (hexval(hex[bi >> 2]) >> (3 - (bi & 3))) & 1;
    }
    uint8_t idx = M == 4 ? (b[0] ? (b[1] ? 0 : 1) : (b[1] ? 3 : 2)) : (b[0] ? 0 : ask_lo);
    fw[nw++] = ((uint32_t)idx << 28) | f;
  }
  ask_step = (uint32_t)(1e6 / baud + 0.5); ask_nsym = (uint16_t)nsym; ask_hz = f * MCLK / 268435456.0;
  segs[0] = Seg{0, (uint16_t)nsym, (uint16_t)nsym, ask_step, 1};
  nseg = 1;
  psk_us = 0; psk_buf = false;
  return sweepGo(1000, 0);
}

static void adgStatus() {                                   // main context, nothing running
  int rb = -1;
  if (Wire.requestFrom(adg_addr, (uint8_t)1) == 1) rb = Wire.read();
  Serial.printf("# adg addr=0x%02x found=%u last=0x%02x readback=", adg_addr, adg_found ? 1 : 0, (unsigned)cur_adg);
  if (rb < 0) Serial.print(F("none")); else Serial.printf("0x%02x", rb);
  Serial.printf(" idle=0x%02x asklo=%u writes=%lu adg_err=%lu dry=%u\n", adg_idle, ask_lo, (unsigned long)adg_n,
                (unsigned long)adg_err, dry ? 1 : 0);
}

static void adgFind() {                                     // ADG729 = 0x44 + A1 A0
  adg_found = false;
  for (uint8_t a = 0x44; a <= 0x47 && !adg_found; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) { adg_addr = a; adg_found = true; }
  }
}

static uint32_t codeSum(uint8_t s) {                       // cinfo 체크섬: 칩이 1 인 자리 (i + 1) 의 합
  uint32_t cs = 0;
  for (uint16_t i = 0; i < clen[s]; i++) if (cslot[s][i]) cs += i + 1;
  return cs;
}

static const char *frameGo(long slot, long bit_us, long delay_us, long nbits, const char *hex) {
  if (sweepActive()) return "busy (sweep running)";
  if (fr_state != F_IDLE) return "busy (frame running)";
  if (slot < 0 || slot >= NSLOT || clen[slot] == 0) return "empty code slot";
  if (bit_us < 2L * clen[slot] || bit_us > 1000000) return "bad bit_us (chip < 2 us or bit > 1 s)";
  if (nbits < 1 || nbits > MAXFBIT) return "bad nbits";
  if ((uint64_t)nbits * (uint64_t)bit_us > FR_MAX_US) return "frame too long";
  if ((long)strlen(hex) < (nbits + 3) / 4) return "short hex";
  for (long i = 0; i < nbits; i++) {
    int d = hexval(hex[i >> 2]);
    if (d < 0) return "bad hex";
    fbits[i] = (d >> (3 - (i & 3))) & 1;
  }
  if (cw_on) outputOff();
  if (!sampling) startSampling(SPS_SWEEP);                  // 스트림이 없으면 1 MS/s 표본 시계로
  adw(OFFW);                                                // 꺼진 상태에서 레지스터를 다 넣는다 (출력 중 쓰기는 이 보드에서 깨짐)
  loadReg(0, fr_w);
  adw(0xC000); adw(0xE000 | 2048);                          // PHASE0 = 0°, PHASE1 = 180°
  fr_ctrl = OFFW;
  if (!adgMain(ADG_LVL[0])) return "ADG729 not answering (level switch)";   // frames: full level throughout
  fslot = (uint8_t)slot; fbit_us = (uint32_t)bit_us; fnbits = (uint16_t)nbits; fbit_i = 0; fchip_i = 0;
  late_max = 0; late_n = 0;
  digitalWriteFast(PIN_TR, HIGH);                           // 스위치를 출력보다 먼저 닫는다
  if (delay_us < 200) delay_us = 200;
  noInterrupts();
  fr_t0 = samp + us2samp((uint64_t)delay_us);
  ev_samp = fr_t0;
  fr_state = F_ARMED;
  armed = true;
  interrupts();
  return nullptr;
}

// ------------------------------------------------------------------ 스트림
static void startMono(uint32_t sps) {
  noInterrupts(); streaming = false; interrupts();
  if (sampling && sps != cur_sps && !clockNeeded()) stopSampling();
  if (!sampling) startSampling(sps);       // 스윕 중이면 표본 시계를 그대로 둔다 (시각이 표본 수로 정해지므로)
  noInterrupts();
  wr = rd = slips = 0;
  usb_short = 0; mono_n++;
  stream0 = samp;
  streaming = true;
  interrupts();
}

static void stopMono() {
  if (!streaming) return;
  noInterrupts(); streaming = false; interrupts();
  stage(0x30);
  if (!clockNeeded()) stopSampling();
  stage(0x31);
  delay(20);
  stage(0x32);
  Serial.print(F("# mono stopped  "));
  Serial.print(cur_sps); Serial.print(F(" S/s  slips "));
  Serial.print(slips);   Serial.print(F("  total "));
  Serial.println(wr);
  last_wr = wr; last_slips = slips; last_short = usb_short;
  stage(0x20);
}

// ------------------------------------------------------------------ 명령
static void help() {
  Serial.printf("TeensyTRX ask2 - galvanic transceiver (Teensy 4.0, RX pin 14 = A0, AD9833 FSYNC 10 / SDATA 11 / SCLK 13, "
                "ADG729 level switch SDA 18 / SCL 19: %s 0x%02x) rev 2: watchdog 2 s, cw hold %lu s, last boot %s\n",
                adg_found ? "found at" : "NOT ANSWERING, expected", adg_addr, (unsigned long)(cw_hold_ms / 1000), bootName());
  Serial.println(F("  fs   fs hold <s> (cw goes off after <s> s without a byte from the host; 0 = never)   fs hang [irq] (test)"));
  Serial.println(F("  mono [sps]   stream A0, 2 bytes/sample (TeensyLink format; default 500000, max 1000000)"));
  Serial.println(F("  x            stop stream"));
  Serial.println(F("  wave sine|square (square at boot)   cw <hz>   off   amp <0-3> (cw level; schedule words carry it in bits 28-29)"));
  Serial.println(F("  levels: 0 = 100 % (0x01)  1 = 67 % (0x02)  2 = 33 % (0x04)  3 = 0 % (0x08);  nothing sent = idle byte"));
  Serial.println(F("  ask2 <hz> <baud> <hex|prbs> [nbits]   2-ASK now: bit 1 = 100 %, bit 0 = level asklo (default 2 = 33 %)"));
  Serial.println(F("  ask4 <hz> <baud> <hex|prbs> [nbits]   4-ASK now: bit pairs 11 / 10 / 00 / 01 = 100 / 67 / 33 / 0 %"));
  Serial.println(F("  asklo <1-3>   idle open|load   adg [hex byte]   i2c   dry 0|1 (rehearsal, nothing reaches E+)"));
  Serial.println(F("  sclr   sseg <step_us> <reps> <n>   sw <w1> <w2> ...   sinfo"));
  Serial.println(F("  sgo [delay_ms] [sps]   sstop   sstat"));
  Serial.println(F("  psk <sym_us> [order] [cont|reset]   DBPSK overlay inside sweep steps (0 = off, default cont)"));
  Serial.println(F("  pbclr   pb <hex>   pbinfo   psk <sym_us> buf [cont|reset]   DBPSK flips from an uploaded bit string"));
  Serial.println(F("  carrier <hz>   code <slot> <n> <hex>   cinfo   ftx <slot> <bit_us> <delay_us> <nbits> <hex>   fstat   mark 0|1"));
}

static void fstat() {
  static const char *names[] = {"idle", "armed", "run"};
  Serial.printf("# frame state=%s sent=%lu refused=%lu bit_us=%lu nbits=%u slot=%u late_max=%lu late_n=%lu mark=%u carrier=",
                names[fr_state], (unsigned long)fr_count, (unsigned long)fr_refused, (unsigned long)fbit_us, fnbits, fslot,
                (unsigned long)late_max, (unsigned long)late_n, mark_en ? 1 : 0);
  Serial.println(fr_w*MCLK/268435456.0, 3);                 // (cw 처럼 print 로 — printf 의 %f 에 기대지 않는다)
}

static void sinfo() {
  uint64_t tot; uint32_t ev;
  uint32_t cs = schedSum(&tot, &ev);
  Serial.printf("# sched segs=%u words=%u events=%lu total_us=%lu checksum=%lu\n",
                nseg, nw, (unsigned long)ev, (unsigned long)(tot > 0xFFFFFFFFull ? 0xFFFFFFFFull : tot),
                (unsigned long)cs);
  for (uint8_t k = 0; k < nseg; k++)
    Serial.printf("# seg %u step_us=%lu reps=%lu n=%u/%u\n", k, (unsigned long)segs[k].step_us,
                  (unsigned long)segs[k].reps, segs[k].n, segs[k].want);
}

static void pbinfo() {                                     // 호스트가 올린 비트열과 맞는지 (trx_psk_link_v2.pb_checksum 과 같은 계산)
  uint32_t cs = 0, ones = 0;
  for (uint32_t i = 0; i < pb_n; i++)
    if ((pbuf[i >> 5] >> (31 - (i & 31))) & 1) { cs += i + 1; ones++; }
  Serial.printf("# pb n=%lu ones=%lu checksum=%lu src=%s\n", (unsigned long)pb_n, (unsigned long)ones, (unsigned long)cs,
                psk_buf ? "buf" : "prbs");
}

static void sstat() {
  static const char *names[] = {"idle", "armed", "run", "done", "stop"};
  uint32_t ev; schedSum(nullptr, &ev);
  long t0 = t0_seen ? (long)(t0_samp - stream0) - (long)t0_slips : -1;
  Serial.printf("# sweep state=%s t0=%ld events=%lu/%lu late_max=%lu late_n=%lu sps=%lu wave=%s psk_us=%lu psk_n=%lu psk_mode=%s psk_src=%s amp=%u adg=0x%02x adg_err=%lu dry=%u fw=ask2 rev=2\n",
                names[sw_state], t0, (unsigned long)ev_done, (unsigned long)ev,
                (unsigned long)late_max, (unsigned long)late_n, (unsigned long)cur_sps,
                (runw & OPBITEN) ? "square" : "sine", (unsigned long)psk_us, (unsigned long)psk_n,
                !psk_us ? "off" : psk_cont ? "cont" : (psk_cont_req ? "reset(auto)" : "reset"), psk_buf ? "buf" : "prbs",
                (unsigned)cw_amp, (unsigned)cur_adg, (unsigned long)adg_err, dry ? 1 : 0);
}

static bool quiet() { return streaming; }      // 스트리밍 중에는 출력 금지
static char fsChar(uint32_t c) { c &= 0xFF; return (c >= 0x20 && c < 0x7F) ? (char)c : '.'; }

static void handleLine(char *line) {
  while (*line == ' ') line++;
  char *arg = line;
  while (*arg && *arg != ' ') arg++;
  while (*arg == ' ') arg++;

  if (!strcmp(line, "x") || !strcmp(line, "X")) { stopMono(); return; }
  if (line[0] == '?') { if (!quiet()) help(); return; }
  if (!strncmp(line, "fs", 2) && (line[2] == ' ' || line[2] == 0)) {     // rev 2: fail-safe status / hold time / test
    if (!strncmp(arg, "hold", 4)) {
      char *q; long v = strtol(arg + 4, &q, 10);
      if (q != arg + 4 && v >= 0 && v <= 86400) cw_hold_ms = (uint32_t)v * 1000;
    } else if (!strncmp(arg, "hang", 4)) {                                // test: stop loop() - the watchdog must reset
      if (!quiet()) { Serial.println(F("# fs hang: loop() stops now, the watchdog resets in 2 s")); Serial.flush(); }
      stage(0x7F);
      if (strstr(arg, "irq")) noInterrupts();
      for (;;) {}
    }
    if (!quiet())
      Serial.printf("# fs rev=2 wdt_ms=%lu cw_hold_s=%lu trips=%lu boot=%s wdog_resets=%lu last_stage=0x%lx last_cmd=%c%c%c%c "
                    "stuck_pc=0x%lx stuck_exc=%lu mono=%lu:%lu/%lu/%lu\n", (unsigned long)((WDT_WT + 1) * 500), (unsigned long)(cw_hold_ms / 1000),
                    (unsigned long)fs_trips, bootName(), (unsigned long)wdog_resets, (unsigned long)boot_stage,
                    fsChar(boot_cmd >> 24), fsChar(boot_cmd >> 16), fsChar(boot_cmd >> 8), fsChar(boot_cmd),
                    (unsigned long)boot_pc, (unsigned long)boot_exc, (unsigned long)mono_n,
                    (unsigned long)last_wr, (unsigned long)last_slips, (unsigned long)last_short);
    return;
  }
  if (!strncmp(line, "mono", 4) || !strncmp(line, "listen", 6)) {   // R4 GUI 가 listen 을 보내도 mono 로
    long v = atol(arg);
    stopMono();
    startMono(v > 0 ? (uint32_t)v : 500000);
    return;
  }
  if (!strncmp(line, "amp", 3) && (line[3] == ' ' || line[3] == 0)) {   // ask1
    if (clockNeeded()) { if (!quiet()) Serial.println(F("# busy (sweep or frame running)")); return; }
    char *q; long v = strtol(arg, &q, 10);
    if (q == arg || v < 0 || v > 3) { if (!quiet()) Serial.println(F("# amp: refused (0-3)")); return; }
    cw_amp = (uint8_t)v;
    if (cw_on) adgMain(ADG_LVL[cw_amp]);
    if (!quiet()) Serial.printf("# amp %u\n", (unsigned)cw_amp);
    return;
  }
  if ((!strncmp(line, "ask2", 4) || !strncmp(line, "ask4", 4)) && line[4] == ' ') {   // ask2: 2-ASK / 4-ASK now
    int M = line[3] == '4' ? 4 : 2;
    const char *err = askGo(M, arg);
    if (!quiet()) {
      if (err) Serial.printf("# ask%d: refused (%s)\n", M, err);
      else {
        Serial.printf("# ask%d armed: %u symbols x %lu us (%lu ms), low level %u, starts in 1000 ms, carrier ", M, ask_nsym,
                      (unsigned long)ask_step, (unsigned long)((uint64_t)ask_nsym * ask_step / 1000), M == 4 ? 3 : ask_lo);
        Serial.println(ask_hz, 3);
      }
    }
    return;
  }
  if (!strncmp(line, "asklo", 5)) {
    long v = atol(arg);
    if (v >= 1 && v <= 3) ask_lo = (uint8_t)v;
    if (!quiet()) Serial.printf("# asklo %u\n", ask_lo);
    return;
  }
  if (!strncmp(line, "idle", 4) && (line[4] == ' ' || line[4] == 0)) {
    if (clockNeeded() || cw_on) { if (!quiet()) Serial.println(F("# busy (transmitting)")); return; }
    if (!strncmp(arg, "load", 4)) adg_idle = ADG_LOAD; else if (!strncmp(arg, "open", 4)) adg_idle = ADG_OPEN;
    bool ok = adgMain(adg_idle);
    if (!quiet()) Serial.printf("# idle %s (0x%02x)%s\n", adg_idle == ADG_LOAD ? "load" : "open", adg_idle, ok ? "" : " - ADG729 not answering");
    return;
  }
  if (!strncmp(line, "adg", 3) && (line[3] == ' ' || line[3] == 0)) {
    if (quiet()) return;
    if (clockNeeded() || cw_on) { Serial.println(F("# busy (transmitting)")); return; }
    if (!adg_found) adgFind();
    char *q; long v = strtol(arg, &q, 16);
    if (q != arg) {
      if (v < 0 || v > 0xFF) { Serial.println(F("# adg: refused (00-ff)")); return; }
      if (!adgMain((uint8_t)v)) Serial.println(F("# adg: write not acknowledged"));
    }
    adgStatus();
    return;
  }
  if (!strcmp(line, "i2c")) {
    if (quiet()) return;
    if (clockNeeded() || cw_on) { Serial.println(F("# busy (transmitting)")); return; }
    Serial.print(F("# i2c:"));
    for (uint8_t a = 0x08; a < 0x78; a++) {
      wdtFeed();                             // a stuck bus makes every address wait for the Wire time-out
      Wire.beginTransmission(a);
      if (Wire.endTransmission() == 0) Serial.printf(" 0x%02x", a);
    }
    Serial.println();
    adgFind();
    return;
  }
  if (!strncmp(line, "dry", 3) && (line[3] == ' ' || line[3] == 0)) {
    if (*arg) {
      dry = false;                           // let outputOff really stop the AD9833 first
      outputOff();
      dry = atol(arg) != 0;
      adgMain(adg_idle);
    }
    if (!quiet()) Serial.printf("# dry %u\n", dry ? 1 : 0);
    return;
  }
  if (!strncmp(line, "wave", 4)) {
    if (clockNeeded()) { if (!quiet()) Serial.println(F("# busy (sweep or frame running)")); return; }
    runw = (!strncmp(arg, "sq", 2)) ? (B28 | OPBITEN | DIV2) : B28;
    if (cw_on) adw(runw);
    if (!quiet()) Serial.printf("# wave %s\n", (runw & OPBITEN) ? "square" : "sine");
    return;
  }
  if (!strncmp(line, "cw", 2)) {
    double hz = atof(arg);
    if (hz <= 0 || hz > 12.5e6) { if (!quiet()) Serial.println(F("# cw: bad frequency")); return; }
    if (!startCW(hz)) {
      outputOff();
      if (!quiet()) Serial.println(F("# cw: bad level switch (ADG729 not answering) - nothing sent"));
      return;
    }
    if (!quiet()) {
      Serial.print(F("# cw ")); Serial.print(cw_hz, 3);
      Serial.println((runw & OPBITEN) ? F(" Hz square") : F(" Hz sine"));
    }
    return;
  }
  if (!strcmp(line, "off") || !strcmp(line, "sstop")) {
    outputOff();
    if (!quiet()) Serial.println(F("# off"));
    return;
  }
  if (!strcmp(line, "sclr")) {
    outputOff();
    nseg = 0; nw = 0; sw_state = S_IDLE;
    if (!quiet()) Serial.println(F("# sched cleared"));
    return;
  }
  if (!strncmp(line, "sseg", 4)) {
    char *p = arg;
    uint32_t step_us = strtoul(p, &p, 10), reps = strtoul(p, &p, 10), n = strtoul(p, &p, 10);
    if (sweepActive() || nseg >= MAXSEG || step_us == 0 || reps == 0 || n == 0 || nw + n > MAXW) {
      if (!quiet()) Serial.println(F("# sseg: refused"));
      return;
    }
    segs[nseg] = Seg{nw, 0, (uint16_t)n, step_us, reps};
    nseg++;
    return;
  }
  if (!strncmp(line, "sw", 2) && (line[2] == ' ' || line[2] == 0)) {
    if (nseg == 0 || sweepActive()) return;
    Seg &g = segs[nseg - 1];
    char *p = arg;
    while (*p) {
      char *q;
      uint32_t w = strtoul(p, &q, 10);
      if (q == p) break;
      p = q;
      if (g.n < g.want && nw < MAXW) { fw[nw++] = w & (WMASK | AMPMASK); g.n++; }
    }
    return;
  }
  if (!strcmp(line, "sinfo")) { if (!quiet()) sinfo(); return; }
  if (!strncmp(line, "sgo", 3)) {
    char *p = arg;
    long d = strtol(p, &p, 10), sp = strtol(p, &p, 10);
    const char *err = sweepGo(d > 0 ? (uint32_t)d : 1000, sp > 0 ? (uint32_t)sp : 0);
    if (!quiet()) {
      if (err) Serial.printf("# sgo: %s\n", err);
      else Serial.printf("# sweep armed in %ld ms\n", d > 0 ? d : 1000);
    }
    return;
  }
  if (!strcmp(line, "sstat")) { if (!quiet()) sstat(); return; }
  if (!strncmp(line, "psk", 3) && (line[3] == ' ' || line[3] == 0)) {
    if (sweepActive()) { if (!quiet()) Serial.println(F("# busy (sweep running)")); return; }
    char *p = arg;
    long us = strtol(p, &p, 10), ord = 0;
    while (*p == ' ') p++;
    bool buf = !strncmp(p, "buf", 3);                       // 2026-10-27: 차수 자리에 buf = 올린 비트열
    if (buf) p += 3; else ord = strtol(p, &p, 10);
    while (*p == ' ') p++;
    if (buf && pb_n == 0) { if (!quiet()) Serial.println(F("# psk: buffer empty (pb)")); return; }
    psk_us = us > 0 ? (uint32_t)us : 0;
    psk_buf = buf;                                          // buf 없는 psk (예전 도구) = PRBS
    if (ord == 7 || ord == 9 || ord == 11 || ord == 15) psk_order = (uint8_t)ord;
    psk_cont_req = strncmp(p, "reset", 5) != 0;             // 기본 cont
    if (!quiet()) Serial.printf("# psk sym_us=%lu order=%u mode=%s src=%s n=%lu\n", (unsigned long)psk_us, psk_order,
                                psk_cont_req ? "cont" : "reset", psk_buf ? "buf" : "prbs", (unsigned long)pb_n);
    return;
  }
  if (!strcmp(line, "pbclr")) {
    if (sweepActive()) { if (!quiet()) Serial.println(F("# busy (sweep running)")); return; }
    pb_n = 0; pb_i = 0; psk_buf = false;
    if (!quiet()) Serial.println(F("# pb cleared"));
    return;
  }
  if (!strcmp(line, "pbinfo")) { if (!quiet()) pbinfo(); return; }
  if (!strncmp(line, "pb", 2) && line[2] == ' ') {          // 비트열 이어 붙이기: 16 진 한 글자 = 4 비트, 최상위 먼저
    if (sweepActive()) { if (!quiet()) Serial.println(F("# busy (sweep running)")); return; }
    bool ok = true;
    for (char *p = arg; *p && ok; p++) {
      int d = hexval(*p);
      if (d < 0 || pb_n + 4 > MAXPB) { ok = false; break; }
      for (int b = 3; b >= 0; b--) {
        uint32_t w = pb_n >> 5, m = 1u << (31 - (pb_n & 31));
        if ((d >> b) & 1) pbuf[w] |= m; else pbuf[w] &= ~m;
        pb_n++;
      }
    }
    if (!quiet()) Serial.printf("# pb n=%lu%s\n", (unsigned long)pb_n, ok ? "" : " refused");
    return;
  }
  if (!strncmp(line, "carrier", 7)) {
    double hz = atof(arg);
    if (hz <= 0 || hz > 12.5e6) { if (!quiet()) Serial.println(F("# carrier: bad frequency")); return; }
    if (fr_state != F_IDLE) { if (!quiet()) Serial.println(F("# busy (frame running)")); return; }
    fr_w = fword(hz);
    if (!quiet()) { Serial.print(F("# carrier ")); Serial.println(fr_w*MCLK/268435456.0, 3); }
    return;
  }
  if (!strncmp(line, "code", 4) && line[4] == ' ') {
    char *p = arg;
    long s = strtol(p, &p, 10), n = strtol(p, &p, 10);
    while (*p == ' ') p++;
    bool ok = s >= 0 && s < NSLOT && n >= 1 && n <= MAXCHIP && (long)strlen(p) >= (n + 3) / 4 &&
              !(fr_state != F_IDLE && fslot == s);
    for (long i = 0; ok && i < n; i++) {
      int d = hexval(p[i >> 2]);
      if (d < 0) ok = false; else cslot[s][i] = (d >> (3 - (i & 3))) & 1;
    }
    if (ok) clen[s] = (uint16_t)n;
    if (!quiet()) {
      if (ok) Serial.printf("# code slot=%ld n=%ld checksum=%lu\n", s, n, (unsigned long)codeSum((uint8_t)s));
      else Serial.println(F("# code: refused"));
    }
    return;
  }
  if (!strcmp(line, "cinfo")) {
    if (!quiet()) for (uint8_t s = 0; s < NSLOT; s++)
      Serial.printf("# cinfo slot=%u n=%u checksum=%lu\n", s, clen[s], (unsigned long)codeSum(s));
    return;
  }
  if (!strncmp(line, "ftx", 3) && line[3] == ' ') {
    char *p = arg;
    long s = strtol(p, &p, 10), cu = strtol(p, &p, 10), dl = strtol(p, &p, 10), nb = strtol(p, &p, 10);
    while (*p == ' ') p++;
    const char *err = frameGo(s, cu, dl, nb, p);
    if (err) fr_refused++;
    if (!quiet()) {
      if (err) Serial.printf("# ftx: %s\n", err);
      else Serial.printf("# ftx armed slot=%ld bit_us=%ld nbits=%ld in %ld us\n", s, cu, nb, dl);
    }
    return;
  }
  if (!strcmp(line, "fstat")) { if (!quiet()) fstat(); return; }
  if (!strncmp(line, "mark", 4)) {
    mark_en = atol(arg) != 0;
    if (!quiet()) Serial.printf("# mark %u\n", mark_en ? 1 : 0);
    return;
  }
  if (!strcmp(line, "sdump")) {
    if (!quiet()) { Serial.printf("# sdump %u:", ndbg); for (uint8_t i = 0; i < ndbg; i++) Serial.printf(" %04x", dbgw[i]); Serial.println(); }
    return;
  }
  if (!strncmp(line, "raw", 3)) {             // 디버그: 16비트 단어(16진)를 연달아 그대로 쓴다. 스윕은 멈춘다
    if (sweepActive()) outputOff();
    char *p = arg; int k = 0;
    while (*p) { char *q; uint32_t w = strtoul(p, &q, 16); if (q == p) break; p = q; adw((uint16_t)w); k++; }
    if (!quiet()) Serial.printf("# raw %d words\n", k);
    return;
  }
}

void setup() {
  bootCause();                             // rev 2: reset cause and the marks of the run before, then output off below
  stage(0x01);
  pinMode(PIN_FSYNC, OUTPUT); digitalWriteFast(PIN_FSYNC, HIGH);
  pinMode(PIN_TR, OUTPUT);    digitalWriteFast(PIN_TR, LOW);
  pinMode(PIN_RX, INPUT_DISABLE);          // 디지털 입력 버퍼 끔 (아날로그 핀 잡음 감소)
  pinMode(PIN_SCLK, OUTPUT);  digitalWriteFast(PIN_SCLK, HIGH);    // SCLK 는 HIGH 에서 쉰다
  pinMode(PIN_SDATA, OUTPUT); digitalWriteFast(PIN_SDATA, LOW);
  // AD9833 선 세 가닥은 구동 세기를 가장 약하게(DSE 1, 약 150 Ω) - pinMode 기본(DSE 7, 가장 셈)은 브레드보드 점퍼에서
  // 에지가 울려 AD9833 가 클럭을 더 세는 것으로 보였다 (2026-10-14: 연속파는 되는데 스윕에서 단어가 깨짐)
  for (uint8_t pin : {PIN_FSYNC, PIN_SDATA, PIN_SCLK})
    *(digital_pin_to_info_PGM[pin].pad) = IOMUXC_PAD_DSE(1) | IOMUXC_PAD_SPEED(0);
  adw(OFFW);
  Wire.begin();                            // ask2: ADG729 on pins 18 (SDA) / 19 (SCL)
  Wire.setClock(400000);
  adgFind();
  adgWrite(adg_idle);                      // every switch open (also the ADG729 power-on state)
  adg_err = 0;
  fr_w = fword(167840.0);
  attachInterruptVector(IRQ_SOFTWARE, step_isr);
  NVIC_SET_PRIORITY(IRQ_SOFTWARE, 64);     // 표본 PIT(32) 보다 낮고, USB(112) 보다 높게
  NVIC_ENABLE_IRQ(IRQ_SOFTWARE);
  Serial.begin(0);                         // USB CDC: baud 무의미
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 2000) {}
  help();
  host_ms = wdt_fed = millis();
  wdtBegin();                              // rev 2: from here on loop() has to come round at least every 2 s
  stage(0x10);
}

void loop() {
  static char line[512]; static uint16_t n = 0;
  uint32_t now = millis();
  if (now - wdt_fed >= WDT_FEED_MS) { wdtFeed(); wdt_fed = now; }
  while (Serial.available()) {
    char c = Serial.read();
    host_ms = now;
    if (c == '\n' || c == '\r') {
      line[n] = 0;
      if (n) {
        CrashReport.breadcrumb(2, ((uint32_t)(uint8_t)line[0] << 24) | ((uint32_t)(uint8_t)line[n > 1 ? 1 : 0] << 16) |
                                  ((uint32_t)(uint8_t)line[n > 2 ? 2 : 0] << 8) | (uint8_t)line[n > 3 ? 3 : 0]);
        stage(0x20);
        handleLine(line);
        stage(0x10);
      }
      n = 0;
    }
    else if (n < sizeof(line) - 1) line[n++] = c;
  }
  if (cw_on && cw_hold_ms && millis() - host_ms > cw_hold_ms) {   // rev 2: nobody has talked to us - carrier off
    stage(0x70);
    outputOff();
    fs_trips++;
    if (!quiet()) Serial.printf("# failsafe: cw off, no byte from the host for %lu s\n", (unsigned long)(cw_hold_ms / 1000));
    stage(0x10);
  }
  if (sampling && !streaming && !clockNeeded()) stopSampling();   // 스트림 없이 돌던 스윕·프레임이 끝나면
  if (!streaming) { stage(0x10); return; }

  // 링버퍼 -> USB. 한 번에 최대 1024 샘플 (2 kB). 형식은 TeensyLink mono 와 같다
  static uint8_t out[2048];
  uint32_t avail = wr - rd;
  if (avail == 0) return;
  if (avail > 1024) avail = 1024;
  for (uint32_t i = 0; i < avail; i++) {
    uint16_t r = ring[(rd + i) & (RING - 1)];
    uint16_t v14 = (uint16_t)(((r & 0x0FFF) << 2) | (r >> 15));   // 최하위 비트 = 송신 중 (mark 1 일 때만)
    out[2*i]     = 0x80 | (uint8_t)(v14 >> 7);
    out[2*i + 1] = (uint8_t)(v14 & 0x7F);
  }
  rd += avail;
  stage(0x11);
  size_t w = Serial.write(out, 2 * avail);
  if (w != 2 * avail) usb_short += 2 * avail - w;   // the core gave up on the host (120 ms without a free buffer)
  stage(0x12);
}
