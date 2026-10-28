# Stent-to-maXTouch Link: Frequency Sweeps and BER Tests, 22 October 2026

All numbers below come from the saved run folders in `data_maxtouch/` (`meta.json`, `capture.npz`, `result*.json`,
`decoded*.json`, `onoff*.csv`). `make_daily_summary.py` regenerates every table and figure. Figures are numbered as in
`README.md`; the workbook `Experimental_Summary_20261022.xlsx` holds the full tables.

## 1. Methods

**Hardware.**
- **Transmitter:** Teensy 4.0 with an AD9833 DDS, running the original TeensyTRX firmware (reflashed and verified at 10:38).
  The output is a square wave on the E+ wire, clipped to a metal stent that lies on the maXTouch EVK glass at the top
  edge (X line 0).
- **Receiver:** a Microchip maXTouch ATMXT2952T EVK (41 × 71 node matrix) read over USB HID.
  - Each sample is the T37 mutual-capacitance delta of one or more nodes.
  - Every read triggers a fresh acquisition.
  - Touch reporting and T72 noise suppression are switched off while recording; both are restored afterwards.

**Frequency sweeps** (`maxtouch_stent_rx.py probe --tx`):
- The transmitter steps through a list of carriers. Each is switched on and off in blocks (coarse: 1 s × 3
  repetitions; fine: 0.5 s × 2), and 64 nodes (Y0–Y63 on X0) are read.
- **Response metric:** energy of sample-to-sample changes while the carrier is on, against while it's off, in dB, per node and carrier.
- **Response threshold:** max(6 dB, median + 6 robust σ) over all node × carrier pairs of a run.
- The node × carrier matrix isn't stored in `result.json`, so it was recomputed from `capture.npz` with the current
  analysis code, with the start fixed to the transmitter's known start. For the two stent sweeps, the recomputed best
  carrier and level match the saved results exactly (163.0 kHz, +55.7 dB; 155.692 kHz, +57.3 dB).

**ON/OFF tests** (`maxtouch_stent_rx.py onoff`): 10 s off, 10 s continuous carrier, 10 s off. Statistics leave out
0.5 s after each switch.

**BER tests** (`maxtouch_stent_rx.py ber --tx --frames 24 --y0 0 --ny 28`):
- **Keying:** on/off (OOK) at the stated physical bit rate.
- **Frame:** 170 bits = 16-bit preamble + 11-bit sync word (SFD) + **127-bit PRBS7 payload** + 16 gap bits.
- **Per run:** 24 frames = 3,048 payload bits.
- **Read window:** 28 nodes (Y0–Y27), i.e. one HID transfer and one fresh sample per scan, 117–125 samples/s.

**Decoders.** All were applied to the same raw recordings:

| Decoder | What it is | When fixed |
|---|---|---|
| **Original** | `analyse_ber` at recording time: interpolated time grid, per-node feature chosen after decoding | before the runs |
| **v3** | `decode_v3.py`: same grid, max of z-scored feature over the top 3 nodes; logenergy at ≥ 4 samples/bit, absdev below | frozen 19:01, sha `5cb4cecf1de6` |
| **v4** | `decode_v4.py`: **raw-timestamp decoding**, no interpolation; samples within ±2 ms of a bit edge skipped; distance from the OFF baseline; max over the top 3 nodes (Y2–Y4 in every valid run); frame timing and clock drift from the headers | frozen 19:16, sha `59cb5e989bc9` |

**Run roles** (relative to v4):
- **Exploratory:** the 11 BER runs recorded before v4 was written. v4 was developed on them, so zero errors there are optimistic.
- **Validation:** the 13 BER runs recorded after the freeze and decoded only after it.

They are *decoder-independent* validation: all runs share one session, one stent position and one panel, so they aren't
physically independent setups.

**Statistics.**
- BER = errors / payload bits **of the decoded frames**; frame success is reported separately.
- A conservative BER counts every undecoded frame's 127 payload bits as 50% errors.
- **95% bounds:** 0 errors → one-sided upper bound 1 − 0.05^(1/n); otherwise two-sided Clopper–Pearson.
- **Assumption:** bit errors are independent (Bernoulli). The errors seen at 90–100 bps cluster within frames, so the true uncertainty is larger than these bounds suggest.

**Exclusions.** Three BER runs failed the v4 health check: the top node must be ≥ 10 dB above the lead level in ≥ 95%
of 2 s windows. Their signal was present 0%, 13% and 11% of the time, caused by a transmitter wiring fault. They are listed
in the inventory and excluded from all performance comparisons.

## 2. Experimental Results

### 2.1 Inventory

| Type | Runs | Notes |
|---|---|---|
| Frequency sweep | 4 | 10:49 stent on glass coarse, 10:55 stent on glass fine, 15:53 hand electrode + finger (fine), 20:33 stent on skin + finger (fine) |
| ON/OFF | 4 | carrier detected at 10:47 (163 kHz, 64 nodes) and 17:49 (155.692 kHz, 28 nodes); none at 16:10 and 16:58 (AD9833 not responding to commands, wiring fault) |
| BER | 24 | 3 signal lost (excluded), 8 exploratory with signal, 13 validation |

### 2.2 Frequency response (Figures 1–2)

| Run | Configuration | Analysis rate (samples/s) | Threshold (dB) | Strongest carriers (best node, dB) | Responding nodes at best carrier |
|---|---|---|---|---|---|
| 10:49 coarse | stent, 14 carriers 0.1 kHz–4 MHz | 63.5 | 10.8 | 163.0 kHz (Y2) +55.7; 50 kHz +36.5; 2.0/4.0 kHz +23 | Y1–Y4 |
| 10:55 fine | stent, 50–500 kHz in 2.5% steps | 63.7 | 16.8 | **155.692 kHz (Y2) +57.3**; 163.6 kHz +53.7; 159.6 kHz +53.4; 151.9 kHz +47.3 | Y2–Y4 |
| 15:53 fine | electrode on hand, finger on top edge | 62.5 | 13.9 | 176.2 kHz +8.0 (below threshold) | none |
| 20:33 fine | **stent on skin, finger on top edge (body-coupled)** | 63.0 | 18.3 | **163.6 kHz (Y1) +56.7**; 159.6 kHz +55.9; 155.7 kHz +55.5; 52.5 kHz (Y0) +54.8 | Y0–Y2 |

- **Where:** with the stent on glass, the response is confined to **Y0–Y8 of X line 0 and is strongest at Y2–Y4**, under the stent. With the stent on skin, it sits at **Y0–Y2**, next to the finger: in the lead, before transmission, the panel reads +412 counts on Y2.
- **Which frequency:** the band **151–168 kHz** gives the strongest response, with secondary maxima near 52 kHz and 473 kHz, and narrow notches in between (Figure 1a).
  - The body-coupled run (20:33) follows the same frequency pattern, notches included, at a similar level: 55.5–56.7 dB at 155.7–163.6 kHz, against 53.4–57.3 dB on glass.
  - Its 52 kHz region is as strong as the main band (+54.8 dB), against +47.0 dB on glass.
  - Since the pattern persists across coupling paths, it seems to come from the panel's sensing rather than from the coupling path. This is an interpretation from two runs.
- **Hand electrode (15:53):** no node × carrier pair exceeded the threshold.
  - The saved `result.json` of that run, analysed with the earlier alignment, reports 176.2 kHz +6.5 dB. Both versions agree that there was no response.
  - The 20:33 run shows that a body-coupled path can work. So the 15:53 null result is more likely due to that setup (dry hand electrode) or to the intermittent transmitter fault of that afternoon (Section 4) than to the body path itself. It wasn't a matched comparison: different electrode, different time.

### 2.3 ON/OFF detection (Figure 3), 155.692 kHz precheck at 17:49:54

- **Spread of Y3:** 4.1 counts (first OFF), **2,510.9 counts (ON)**, 4.0 counts (second OFF). That's a ratio of 624 to the mean OFF level.
- **Other responding nodes:** Y2 ×618, Y4 ×301, Y1 ×188, Y0 ×103, Y5 ×100.
- **Sample rate:** 125.2 / 121.1 / 123.5 per second in the three phases.
- **Magnitudes don't overlap:** OFF samples stay below about 15 counts; ON samples are mostly 300–4,000 counts.
- **10:47 run:** the 163 kHz test with 64 nodes showed the same picture (Y3 ×532).

### 2.4 BER vs bit rate, 155.7 kHz (Figure 4)

Errors / payload bits, combined blind decision, 24/24 frames decoded unless stated otherwise:

| Bit rate | Samples/bit | Role (v4) | Original | v3 | **v4** | v4 error-free frames |
|---|---|---|---|---|---|---|
| 20 | 6.2 | exploratory | 20/3048 | 2/3048 | 0/3048 | 24/24 |
| 34 | 3.6 | exploratory | 16/3048 | 7/3048 | 0/3048 | 24/24 |
| 38 | 3.3 | exploratory | 9/3048 | 7/3048 | 0/3048 | 24/24 |
| 38 | 3.3 | validation | 18/3048 | 11/3048 | **0/3048** | 24/24 |
| 40 | 3.1 | exploratory | 20/3048 | 3/3048 | 0/3048 | 24/24 |
| 40 | 2.9 | validation | 65/3048 | 47/3048 | **2/3048 (0.066%)** | 22/24 |
| 45 | 2.7 | exploratory | 86/3048 | 67/3048 | 0/3048 | 24/24 |
| 45 | 2.7 | validation | 82/3048 | 60/3048 | **0/3048** | 24/24 |
| 68 | 1.8 | exploratory | 248/3048 | 209/3048 | 0/3048 | 24/24 |
| 68 | 1.8 | validation | 239/3048 | 201/3048 | **0/3048** | 24/24 |
| 80 | 1.55 | validation ×3 | 388, 407, 387 /3048 | 350, 379, 339 /3048 | **0, 0, 0 /3048** | 24/24 each |
| 90 | 1.37 | validation | 458/3048 | 421/3048 | **4/3048 (0.13%)** | 20/24 |
| 90 | 1.37 | validation | 437/2921 (23/24 fr) | 405/2921 (23/24 fr) | **15/2921 (0.51%), 23/24 frames** | 14/24 |
| 100 | 1.24 | validation | 520/3048 | 487/3048 | **32/3048 (1.05%)** | 4/24 |

**Pooled v4 validation results, 155.7 kHz:**

| Bit rate | Errors / bits | BER (95% bounds) |
|---|---|---|
| 80 bps | 0 / 9,144 | **< 0.033%** |
| 90 bps | 19 / 5,969 | 0.32% (0.19–0.50%); 1 frame of 48 not decoded, conservative BER of the second run 2.6% |
| 100 bps | 32 / 3,048 | 1.05% (0.72–1.48%) |
| All rates ≤ 80 bps | 2 / 21,336 | 0.0094% (0.0011–0.034%) |

### 2.5 Decoder comparison on identical recordings (Figure 5)

| Recording | Original | v3 | v4 single (top-ranked node) | **v4 combined** |
|---|---|---|---|---|
| 68 bps, exploratory | 248 | 209 | 17 | **0** |
| 68 bps, validation | 239 | 201 | 24 | **0** |
| 80 bps, validation 19:27 | 388 | 350 | 21 | **0** |
| 80 bps, validation 19:50 | 407 | 379 | 37 | **0** |
| 80 bps, validation 19:51 | 387 | 339 | 35 | **0** |
| 80 bps at 163.6 kHz, validation | 364 | 324 | 33 | **3** |

### 2.6 Carrier comparison, v4 (Figure 6)

Only the tested combinations are shown; 159.6 kHz was **not** tested at 80 bps.

| Carrier | 20 bps | 34 bps | 80 bps |
|---|---|---|---|
| 155.7 kHz | 0/3048 (exploratory) | 0/3048 (exploratory) | 0/9144 over 3 runs (validation) |
| 159.6 kHz | 0/3048 (validation) | 0/3048 (exploratory) | not tested |
| 163.6 kHz | 0/3048 (validation) | 0/3048 (exploratory) | **3/3048 = 0.10% (0.020–0.29%), 21/24 frames error-free** (validation) |

- At 20 and 34 bps, the three carriers can't be told apart: all have 0 errors.
- At 80 bps, 163.6 kHz had 3 errors, all "sent 1 read as 0", while 155.7 kHz had none in 9,144 bits. That rests on a single 163.6 kHz run, so it is suggestive, not conclusive.

### 2.7 Multi-node combining (Figure 7), v4 validation pooled per rate, 155.7 kHz

| Bit rate | Single nodes Y2 / Y3 / Y4 | Combined Y2–Y4 |
|---|---|---|
| 80 bps (9,144 bits) | 96 / 93 / 122 errors, i.e. 1.0–1.3% | **0** |
| 68 bps | 24 / 24 / 22 errors | **0** |

All three nodes see the same carrier. Taking the strongest node at each sample removes the samples that are weak on one node only.

### 2.8 Timing and decoder analysis (Figure 8)

- **Step response (b), averaged over 216 off-edges and 192 on-edges of the 34 bps run:**
  - The node response steps from 0 to 100% within about 2–3 ms.
  - It arrives a constant **10.75–12.50 ms** (all valid runs) after the transmitted edge: sensing plus USB delay.
  - It drops back to 0 within one bin, so **there is no filtering tail over several scans**.
- **Interpolation and window misalignment (a)** produced v3's "spill-over" errors:
  - In the v3 error shown (34 bps, a sent 0 after a 1), every raw sample inside the bit is ≤ 3 counts.
  - The linearly interpolated grid places values of about 1,500 counts inside the 0-bit window. These are mixtures of the neighbouring ON (about 3,000) and OFF samples. They don't exist in the raw data.
- **Change-based measure (c, d)** produced v3's "fades":
  - In the v3 error shown (20 bps, 159.6 kHz validation run, a sent 1 read as 0), the raw values stay at −2,300 to −3,300 counts throughout the bit. They vary only slowly.
  - logenergy measures sample-to-sample *change*, so it falls towards the OFF level. absdev measures *distance from the OFF baseline*, so it stays high.
  - All of v3's "1 read as 0" errors at 20 bps in these recordings showed this pattern.

**Decoder summary (Figure 10).**
- (a) Side by side: the original decoder and v3 interpolate onto a uniform grid and use fractional bit windows; v3 also uses a change-based measure at ≥ 4 samples/bit. v4 uses raw timestamps, a ±2 ms edge guard and distance from the OFF baseline.
- (b) One transmitted frame of the 19:27 80 bps validation run: Original 17/127 errors, v3 16/127, v4 0/127. In bits 40–79, the original decoder and v3 err mostly on a sent 0 right after a 1.
- (c) The pooled validation BER per rate for all three decoders.

### 2.9 Frame success and payload throughput (Figure 9), v4 at 155.7 kHz

Payload throughput = physical rate × 127/170 × frames decoded / frames sent. Error-free-frame throughput counts only
frames with zero payload errors.

| Physical bit rate | Frames decoded | Frames error-free | Payload BER | Payload throughput | Error-free-frame throughput |
|---|---|---|---|---|---|
| 68 bps (validation) | 24/24 | 24/24 | 0/3048 | 50.8 bit/s | 50.8 bit/s |
| **80 bps** (validation, 3 runs) | 72/72 | 72/72 | 0/9144 | **59.8 bit/s** | **59.8 bit/s** |
| 90 bps (validation, 2 runs) | 47/48 | 34/48 | 19/5969 | 67.2 and 64.4 bit/s (raw) | 56.0 and 39.2 bit/s |
| 100 bps (validation) | 24/24 | 4/24 | 32/3048 | 74.7 bit/s (raw) | 12.5 bit/s |

## 3. Main Findings

1. **The stent couples strongly into the maXTouch at 151–168 kHz.**
   - The peak is 155.7 kHz at +57.3 dB, about 40 dB above the response threshold.
   - The response is localised to nodes **Y2–Y4** of X line 0, under the stent.
   - A single sample separates ON from OFF: spread about 2,500 counts against about 4 counts.
2. **The earlier "fading" and "inter-symbol interference" were mostly decoder artifacts.**
   - The raw response is a sharp step after a fixed delay of about 11–12 ms, with no multi-scan tail.
   - Interpolation and bit-window misalignment created the spill-over errors.
   - A change-based measure missed strong but slowly varying responses.
3. **The frozen raw-timestamp decoder v4 is far better than the original and v3 on identical recordings.** At 80 bps: 0 errors against 387–407 (original) and 339–379 (v3).
4. **80 bps is the highest repeatedly verified zero-observed-error rate under this setup at 155.7 kHz**:
   - 3 validation runs, 0 errors in 9,144 bits (BER < 0.033%, 95% one-sided, independence assumed);
   - all 72 frames decoded and error-free;
   - payload throughput **59.8 bit/s** at 80 bps physical.

   Errors appear at 90 bps (0.32% pooled, 1 frame lost) and at 100 bps (1.05%; only 4 of 24 frames error-free).
5. **Combining Y2–Y4 is essential at high rates.** At 80 bps the single nodes had 1.0–1.3% BER, and the combined decision had none.
6. **A body-coupled path also works.** With the stent on skin and a finger on the panel (20:33), the response reached +56.7 dB at 163.6 kHz on nodes Y0–Y2, about as strong as with the stent on glass. No BER test has been run in this configuration yet.

## 4. Limitations

- **One session, one stent position, one panel.** Validation is decoder-independent, not a physically independent
  replication. Repeats in another session or position, or on another EVK, are needed before the numbers can be generalised.
- **80 bps is not a hardware limit.** It's the highest rate tested repeatedly without errors with this transmitter, stent
  position, 28-node read window and decoder. Throughput is bounded by about one fresh sample every 8.0–8.2 ms. Faster
  panel scanning, other node windows, coding or modulation could change it.
- **The 95% bounds assume independent bit errors.** The errors at 90–100 bps cluster within frames, so the real uncertainty is larger.
- **Carrier comparison:** 159.6 kHz wasn't tested at 80 bps. The 163.6 kHz result at 80 bps (3 errors) comes from one run.
- **The hand-electrode sweep (15:53) happened in the same afternoon as an intermittent transmitter fault.** The AD9833 was
  found unresponsive at 16:10–17:00, and the Teensy dropped off USB at 15:44 and 17:54. The transmitter's output during
  that sweep wasn't verified independently, so its null result shouldn't be over-interpreted.
- **The body-coupled result rests on one sweep (20:33).** There's no ON/OFF test or BER test in that configuration yet. The
  finger position on the panel (around Y2) wasn't controlled precisely.
- **Slow-scan episodes:** some runs contain 10–20 s periods in which the panel scans slower (10–14 ms) and all nodes
  shift by +26–35 counts. The cause is unknown. v4 decoded through them without errors, but they reduce the sample rate.
- **The sweep matrices are recomputed** from `capture.npz` with the current analysis code; the hand run's saved `result.json` differs slightly (see 2.2).

## 5. Next Steps

1. **Replication:** repeat 80 and 90 bps at 155.7 kHz in a new session, after re-placing the stent. Ideally also at a second stent position.
2. **Carrier robustness at the operating rate:** 80 bps at 159.6 kHz, and repeats at 163.6 kHz.
3. **More bits per point** (several 24-frame runs) to tighten the BER bounds at 80–90 bps.
4. **Body coupling:** for the stent-on-skin path, run an ON/OFF test and a BER ladder (20 → 80 bps) at 163.6 kHz with the finger held over Y0–Y2. Also repeat the hand-electrode test with a verified transmitter (`stent_tx.py selftest` before and after) and a gel electrode.
5. **Going beyond about 80 bps:** chip configuration for faster scans, and longer payloads per frame to reduce the 43-bit frame overhead.
6. **Health monitoring during runs:** an A0 self-test before and after each run, so transmitter faults are caught at once.
