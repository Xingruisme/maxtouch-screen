# Falsification audit of the 80 bps zero-error result (frozen decoder v4)

**Date:** 2026-10-23.

**Subject:** the reported result of 0 errors in 9,144 payload bits. It comes from three validation runs at 155.7 kHz and 80 bps, decoded by `decode_v4.py` (frozen 2026-10-22 19:16, sha256 prefix `59cb5e989bc9`):
- `stent_ber_val4_80bps_20261022_192702`
- `stent_ber_val4b_80bps_20261022_195008`
- `stent_ber_val4b_80bps_20261022_195109`

**Approach:** try to break the result rather than confirm it. Every claim below is backed by a script in this folder and a number in its `*_results.json`. `run_all.sh` reproduces everything in ~2 minutes.

**Integrity:** none of the 254 input files was modified. `sha256_before.txt` was written before the audit and re-checked after it:
- the decoders (v2, v3, v4), `stent_frames.py`, `stent_tx.py`, `maxtouch_stent_rx.py`, `cap_ook_rx.py` and `TeensyTRX.ino`;
- every `stent_ber_*` folder, its capture and every result and decoded file;
- the two ON/OFF control recordings.

The frozen decoder was imported for the leakage test. Its `decode_run` (which writes files) was never called.

---

## 0. Verdict in brief

| | |
|---|---|
| **Independently verified** | <ul><li>**Correct count:** 0 errors in 9,144 payload bits, all 72 transmitted frames recovered, no bit skipped, no false frame.</li><li>**Truly blind:** reproduced by an independent decoder that never had access to the payload, with frame acceptance based on the header only.</li><li>**Payload can't change the bits:** passing a random, inverted or all-zero payload to the frozen decoder leaves every decoded bit, threshold and timing value unchanged.</li><li>**The bits carry the real payload:** every other alignment of the PRBS7 scores exactly 1,536/3,048 errors, and 10,000 random sequences score at least 1,008.</li><li>**The scorer can find errors:** the same pipeline finds exactly the errors v4 reported at 40, 90 and 100 bps and at 163.6 kHz.</li></ul> |
| **Real weaknesses found** | <ul><li>**Payload oracle (1):** v4's frame acceptance uses the payload (`decode_v4.py:160`). It had no effect here.</li><li>**Weak header rule (2):** acceptance by header correlation alone (r ≥ 0.3) passes 10–15 % of noise slots, 93 % of time-reversed frames and 92 % of frames misaligned by ±2 bits. It caused no false frame here only because the global timing peak was unambiguous.</li><li>**Narrow working point (3):** the zero sits in a narrow window. Bit timing must be within about ±1 ms, the guard must be 2–3 ms, and all three nodes must be max-combined. Any single node gives about 1 % BER.</li><li>**Stale samples (4):** the decoder takes host arrival time as acquisition time. After a stalled USB transfer, a sample can carry the previous bit's value with a late timestamp. This is the cause of both 40 bps errors, 6 of 15 errors in one 90 bps run and 5 of 32 at 100 bps. It also caused the closest 80 bps near-miss on a 0 that follows a 1.</li><li>**Same 127 bits every frame (5):** every frame carries the same 127-bit PRBS7, and the decoder rules were developed on runs with that same payload.</li></ul> |
| **Uncertain** | <ul><li>**The true BER at 80 bps.** The 95 % upper bound of 0.033 % assumes independent errors, and that assumption fails at run level. The per-sample error mechanisms seen in other runs occur at 1–10 × 10⁻⁴ per sample, and at 80 bps one sample decides 93 % of bits. A BER of order 10⁻⁴ is therefore plausible and would show up in a longer recording.</li></ul> |
| **Conclusion** | 0/9,144 is a **genuine, correctly scored, payload-blind** result for these three recordings. It is **not** evidence that 80 bps is error-free in general. Report it as "0 errors observed in 9,144 bits (3 runs, one session, one payload)". The experiments in §12 are needed before a stronger claim. |

---

## 1. What was run

| Step | Script | What it does |
|---|---|---|
| 1 | `blind_decoder.py`, `run_blind.py` | **New decoder written from v4's rules text, not from its code.** Only numpy is imported, and it refuses to run if any payload-carrying module is loaded (`assert_blind`). It knows only the frame *structure*. It reads `t`, `X`, `args.bps` and `t_prompt`, and never `t_first` or any result file. **Acceptance: header only.** Two modes: **v4-rules blind** (lead = samples before `t_prompt`) and **strict blind** (lead = first 3 s of the recording, no metadata timing at all). |
| 2 | `score.py` | **The only place the payload exists.** PRBS7 is rebuilt from the recurrence o[n+7] = o[n] ⊕ o[n+1], cross-checked against an LFSR and against the payload v4 saved. Frames are matched to transmitted frames by time. Lost frames, false positives, misaligned frames and duplicates are counted separately. The same scorer also scores the frozen v4's saved output. |
| 3 | `leakage_test.py` | Static trace of every payload / timing / BER use in `decode_v4.py`. The frozen `decode_frames` is called with true, random, inverted and all-zero payloads. |
| 4 | `controls.py` | Negative controls (OFF-only recordings, surrogates, time reversal), forced timing errors, wrong payloads. |
| 5 | `mechanism.py` | Sample → bit statistics, per payload bit. |
| 6 | `sensitivity.py` | **Informational only:** one frozen rule changed at a time; sample-level error rate. Nothing reported elsewhere is replaced by these numbers. |
| 7 | `stalls.py` | Samples that follow a stalled HID transfer. |
| | `figures.py` | Figures A–E. |

Positive controls: the same pipeline was also run on the eight other validation runs at 155.7 kHz (38–100 bps) and on the 163.6 kHz 80 bps run, where v4 *does* make errors.

---

## 2. The pipeline, line by line

### 2.1 Transmitter (what was actually sent)

| Where | What |
|---|---|
| `stent_frames.py:18-39` | Frame = `[1,0]×8` preamble + Barker-11 SFD `11100010010` + payload + 16 zeros = 170 bits. Payload = `np.roll(prbs7(), 120)`, the same 127 bits in every frame. |
| `stent_tx.py:51-63` `ber_segments` | `frame_bits × 24` → run-length segments, 1 = carrier (155,692 Hz), 0 = 0 Hz; `bit_us = round(1e6/80) = 12,500` exactly. |
| `stent_tx.py:75-84` `to_schedule` | One `sseg` on the greatest common divisor of the segment lengths: 12,500 µs steps, 4,080 frequency words (≤ the firmware's 4,096). |
| `stent_tx.py:156-173` `upload` | `sinfo` checksum compared with the host's. The audit of 2026-10-22 recomputed it from an independently built bit stream: **3089373205**, the value the Teensy reported. |
| `TeensyTRX.ino:281-315` `sweepGo` | Arms the schedule `delay_ms` (500) ahead: `t0_samp = samp + us2samp(delay)`. |
| `TeensyTRX.ino:171-210` `step_isr` | Every step at `ev_samp = t0_samp + us2samp(et_us)` (line 203). Timing comes from the Teensy's own sample clock and is **absolute, not accumulated**, so there is no drift within a run apart from the crystal offset. Each step does `off → FREQ0 → on` (184, 190), giving a ~20 µs gap at every bit edge, including inside runs of 1s. That's far shorter than a panel scan and not visible in the data. |
| `stent_tx.py:175-179` `go` | `t_first = t_cmd + 0.5 s + 0.5 ms`: the host's estimate of the first step, used only by the known-timing path and by scoring. |

### 2.2 Receiver (what was recorded)

`maxtouch_stent_rx.py:214-267` `record`:
- One T37 delta read of X0 Y0–Y27 per loop.
- Each sample is stamped with the **host arrival time** (`perf_counter`), not the panel's acquisition time.
- Measured at 80 bps: median interval 8.000 ms (123.6 samples/s); 94.8–95.7 % of intervals within ±0.2 ms of 8 ms; 2.2 % > 10 ms (stalled transfers).

### 2.3 Decoder v4 (frozen), as executed for the headline number

| Line(s) | Operation | Inputs |
|---|---|---|
| 179-191 | Load capture, meta; `frame_bits, truth = TX.frame_bits(...)` (**payload loaded here**); `Tb = 1/80`, `nb = 170` | meta, TX |
| 192-195 | Node ranking: 10·log10 of mean squared successive difference after/before `t_prompt`; top 3 (Y2, Y3, Y4 in all three runs) | X, t_prompt |
| 196 + 70-82 | Per node: `absdev = log10(1 + (x − b)² / σ²)`. σ = 1.4826·MAD of the lead (4.4 counts). b = block medians (50-sample chunks, interpolated) of the quiet samples only (\|x − lead median\| < 20σ). Then `R.zscore` (`maxtouch_stent_rx.py:390`: (v − median)/IQR over the whole record) | X, lead mask |
| 197 | `comb` = per-sample **max** of the three z-scored features | |
| 199-201 | `all_frames`: every frame slot whose start lies inside the recording | t, τ, ε |
| 204 + 112-125 | `search`: τ over one frame period at ε = 0 (4 ms steps), then τ ± 6 ms (0.25 ms) × ε ∈ [−200, 200] ppm (10 ppm), maximising… | comb, t |
| 94-109 | …`header_score`: Pearson r between the 27 header bit means and the known header, summed over all slots | comb, t, HDR |
| 205 + 128-161 | `decode_frames` per slot. <ul><li>**Bit means:** with `bit_index` (85-91), the mean of `comb` over samples more than 2 ms from both bit edges (fallback: all samples of the bit) (143-147).</li><li>**Empty bits:** a slot with an empty header or payload bit is skipped (148).</li><li>**Threshold:** midway between the means of the known 1s (13 header ones) and known 0s (14 header zeros + 16 gap zeros) (150-152).</li><li>**Decision:** `got = v > thr` (153).</li><li>**Scoring:** `err = sum(got != truth)` (154); header r (155-156).</li><li>**Acceptance:** `valid = r ≥ 0.3 and ber < 0.35` (160).</li></ul> | comb, t, τ, ε, known bits; truth only at 154, 157, 160 |
| 206 | `V2.frames_summary` (`decode_v2.py:54-61`): bits and errors over `valid` frames; `missing = max(sent − valid, 0)` | |

---

## 3. How a raw sample becomes a bit (Figure A)

**Bit assignment.** For sample i with host timestamp tᵢ (decode_v4 `bit_index`, lines 85-91):

  uᵢ = (tᵢ − τ) / (T_b (1 + ε)),  frame mᵢ = ⌊uᵢ / 170⌋,  bit jᵢ = ⌊uᵢ − 170 mᵢ⌋,  position pᵢ = uᵢ − 170 mᵢ − jᵢ ∈ [0, 1)

with T_b = 12.5 ms.

- **τ:** the start of frame 0 on the host clock. It contains every delay between the transmitter's step and the moment the sample reaches Python: the host's `sgo` latency, the panel's sensing, the T37 read and USB. Measured from the blind τ against the host's `t_first`, the delay is **11.22 / 11.44 / 11.26 ms**, constant within each run to 0.01 ms. It agrees with the known-timing search (11.25 / 11.50 / 11.25 ms on its 0.25 ms grid). Bit edges are therefore placed ~0.9 bit after the transmitter's edges. That is why a decoder that ignores the delay fails.
- **ε:** the Teensy crystal versus the host clock, −20 / −10 / −10 ppm. Over the 51 s transmission that's −1.02 / −0.51 / −0.51 ms. Removing it changes nothing at 80 bps: 0 errors (Figure E). At 90 and 100 bps it matters: without it, the two 90 bps runs have 10 and 13 errors instead of 4 and 15, and the 100 bps run 55 instead of 32.
- **Guard:** a sample counts as "inner" if g < pᵢ < 1 − g, with g = 2 ms / 12.5 ms = 0.16. This means 2.0–10.5 ms after the inferred edge, an 8.5 ms window.

**Decision.**

  v_b = mean{ cᵢ : sample i in bit b, inner } (fallback: all samples of b)
  cᵢ = maxₙ zₙ(i),  zₙ = (aₙ − med aₙ)/IQR(aₙ),  aₙ(i) = log₁₀(1 + (xₙ(i) − bₙ(i))² / σₙ²)
  bit = 1 ⇔ v_b > θ_m,  θ_m = ½ (mean v over the 13 known 1s + mean v over the 30 known 0s of frame m)

**Why 1.55 samples per bit is enough.** Sample intervals are 8.000 ms and bits are 12.5 ms, so each bit contains one or two samples:

| Samples per bit (9,144 bits) | 1 | 2 | 0 |
|---|---|---|---|
| All samples in the bit | 4,178 | 4,966 | 0 |
| Inner samples (outside the guard) | **8,533 (93.3 %)** | 526 | **85** (fallback to edge samples) |

- **At least one inner sample, almost always.** The inner window is 8.5 ms, longer than the 8 ms interval, so every bit contains at least one inner sample unless an interval is stretched (the 2.2 % of intervals > 10 ms). That is exactly where the 85 fallback bits come from.
- **Sampling phases are diverse.** gcd(8, 12.5) = 0.5 ms, so sample phases fall on a 0.5 ms lattice that covers the whole bit. Each payload position is seen at 15–20 distinct phases across the 72 frames.
- **So, almost always, one raw reading decides each bit.** This works because the panel's response to a carrier edge is a sharp step, 2–4 ms wide, with no multi-scan tail. Each scan is an almost instantaneous reading of "carrier on" (|delta| of 10²–3·10³ counts, random sign) or "off" (σ = 4.4 counts).
- **The separation is large.** The combined decision value of every sent 1 lies above its frame threshold, and that of every sent 0 below it (Figure A d). The smallest margins are +0.007 (a sent 1) and +0.028 (a sent 0), on a scale where the typical level is ±0.5.

---

## 4. Ground-truth leakage (item 1)

### 4.1 Every use of payload, expected timing and BER in `decode_v4.py`

The static trace is in `leakage_results.json → static_trace`.

| Line | Use | Affects | Severity |
|---|---|---|---|
| 187 | `truth = TX.frame_bits(...)` | loaded | – |
| 131-132 | `len(truth)` | payload *length* (slice positions) only | none |
| 151 | `fb == 1` / `fb == 0` at `known_mask` positions | threshold. The mask excludes the payload, so only header and gap values are used. Tested in §4.2: thresholds identical under any payload | none |
| 154, 157 | `err`, `ber` | computed after the decision (153) | none for the bits |
| **160** | **`valid = r ≥ 0.3 and ber < 0.35`** | **frame acceptance, hence the BER denominator** | **high in principle**: an oracle that drops badly decoded frames |
| 206 / `decode_v2.py:58` | `missing = max(sent − valid, 0)` | a spurious accepted frame would offset a missing one | low–medium |
| 211, 223 | `error_types(…, truth)` | reporting | none |
| 221 | best-of-3 single node chosen by BER | reporting only, labelled "optimistic"; not the headline | low |
| 225-231 | known-timing path: τ search limited to `t_first` + 0–25 ms, every frame `valid` | a separate result, not the headline | low |
| 164-175, 232 | `health_raw` windows start at `t_first` | flags signal-lost runs only | low |
| 192, 196 | `t_prompt` for the lead | the receiver's own "START TX" prompt, not transmitter timing; removed in strict mode with identical results (§6) | low |
| 112-125, 94-109, 199-201 | timing search, header score, frame slots | **only `comb`, `t` and the 27-bit header** | none |

### 4.2 Code-path test on the frozen functions

v4's own `search` gives the same τ and ε as the blind decoder (difference 0.0 ms, 0 ppm).

v4's own `decode_frames` was then called four times on the same feature and timing, with only the payload argument changed (and the payload part of `frame_bits` changed with it):

| Payload passed in | `got` | `thr` | `norm` | `r` | `s0` | Frames `valid` (each run) |
|---|---|---|---|---|---|---|
| true | – | – | – | – | – | 24 / 24 / 24 |
| random | identical | identical | identical | identical | identical | **0 / 0 / 0** |
| inverted | identical | identical | identical | identical | identical | **0 / 0 / 0** |
| all zeros | identical | identical | identical | identical | identical | **0 / 0 / 0** |

**Conclusions:**
- **No payload reaches a bit decision.** The payload can't influence synchronisation, node choice, the threshold or any decoded bit.
- **Acceptance does depend on it.** Line 160 decides which frames are counted: with a wrong payload, every frame is rejected.
- **In these runs that rule rejected nothing.** No frame was rejected only by the payload rule. The two rejected candidates per run lie outside the transmission and already have r < 0.3. Removing the rule gives identical results (§6).

---

## 5. Multi-node combination, baseline, threshold: is anything biased toward the expected bits? (item 4)

**Combination.** Per sample, the maximum of the three z-scored features. It is computed before any frame timing and has no access to bit values.
- **The max is not dominated by the strongest node.** For sent-1 bits, Y2 and Y4 supply it most often; Y3 rarely does in two of the runs: Y2/Y3/Y4 = 36/0/64 %, 33/0/67 % and 71/19/9 % (Figure A f).
- **Why Y3 rarely wins:** its absdev distribution has the widest IQR, so its z-values are compressed. This is a property of the z-scoring, not of the bits.
- **The max raises the OFF level too.** The maximum of three noise values is higher than one, so OFF decision values rise slightly (median −0.014). The per-frame threshold absorbs this, and it is symmetric with respect to the truth.

**OFF baseline.** Per node, from the transmitter-off lead (median and MAD) and from quiet samples during the run (|x − lead median| < 20σ), as block medians.
- **Selection is by amplitude.** Quiet samples are chosen by how far they sit from the lead level, not by bit value or timing.
- **Docstring mismatch (cosmetic):** the docstring (line 71) says "running median (50 samples)". The code (`cap_ook_rx.py:202`) uses medians of consecutive 50-sample blocks, interpolated. That is a documentation error with no effect on results.

**Threshold.**
- **It uses only known frame structure:** the midpoint between the means of the 13 header ones and the 30 known zeros of that frame.
- **The payload has no effect:** the threshold is identical under any substituted payload (§4.2).
- **Possible bias:** the preamble ones are isolated 1s (1010…), which could pull the "1" mean down and the threshold with it, favouring 1s. The data show no such asymmetry at 80 bps: of the 82 bits with margin < 0.1, 81 are sent 1s (weak ON readings), the opposite of what a low threshold would produce.

**Bias toward expected values:** none found. No quantity before the decision depends on payload values.

---

## 6. Original versus strictly payload-blind decoding (item 2)

All rows were scored by the same independent scorer (`score.py`).

**Columns:**
- **acc** = accepted frames;
- **true** = matched one-to-one to a transmitted frame with residual 0–25 ms;
- **FP** = accepted outside the transmission;
- **mis** = inside the transmission but off the bit grid;
- **bits same** = decoded bits identical to the frozen v4 output in all candidate slots.

| Run | Decoder | acc | true | missing | FP | mis | Errors / bits | Bits same as frozen v4 |
|---|---|---|---|---|---|---|---|---|
| 192702 | frozen v4 (saved `decoded_v4.json`) | 24 | 24 | 0 | 0 | 0 | 0 / 3,048 | – |
| | frozen v4 output, payload rule removed | 24 | 24 | 0 | 0 | 0 | 0 / 3,048 | – |
| | **blind, v4 rules, header-only acceptance** | 24 | 24 | 0 | 0 | 0 | **0 / 3,048** | 3,302 / 3,302 |
| | **strict blind (no metadata timing)** | 24 | 24 | 0 | 0 | 0 | **0 / 3,048** | 3,300 / 3,302 ¹ |
| | blind, header-Hamming acceptance (≤ 2 of 27 wrong) | 24 | 24 | 0 | 0 | 0 | 0 / 3,048 | |
| 195008 | frozen v4 / payload rule removed / blind / strict / Hamming | 24 | 24 | 0 | 0 | 0 | **0 / 3,048** (all) | 3,302 / 3,302 (blind and strict) |
| 195109 | frozen v4 / payload rule removed / blind / strict / Hamming | 24 | 24 | 0 | 0 | 0 | **0 / 3,048** (all) | 3,302 / 3,302 (blind and strict) |
| **Pooled** | **strictly payload-blind** | **72** | **72** | **0** | **0** | **0** | **0 / 9,144** | |

¹ The 2 differing bits are in slot 26, a noise slot after the transmission that every decoder rejects (r = 0.09). The strict lead changes its baseline slightly.

**Timing of the true frames:**
- **Residuals:** 11.22 / 11.44 / 11.26 ms (constant within each run to 0.01 ms).
- **Header r:** at least 0.946, median 0.98–0.99.
- **Header bit errors:** 0 in all 72 frames.

**Positive controls.** Same pipeline, other validation runs. The audit decoder must reproduce errors where they exist:

| Run | v4 (saved) | Blind (header-only) | Bits identical |
|---|---|---|---|
| 38 bps 192113 | 0 / 3,048 | 0 / 3,048 | 3,048 / 3,048 |
| 40 bps 192306 | 2 / 3,048 | 2 / 3,048 | 3,048 / 3,048 |
| 45 bps 192448 | 0 / 3,048 | 0 / 3,048 | 3,048 / 3,048 |
| 68 bps 192559 | 0 / 3,048 | 0 / 3,048 | 3,048 / 3,048 |
| 90 bps 195206 | 4 / 3,048 | 4 / 3,048 | 3,302 / 3,302 |
| 90 bps 195302 | 15 / 2,921 (23/24 frames) | 15 / 2,921 (23/24) ² | 3,175 / 3,175 |
| 100 bps 192753 | 32 / 3,048 | 32 / 3,048 | 3,302 / 3,302 |
| 163.6 kHz, 80 bps 200327 | 3 / 3,048 | 3 / 3,048 | 3,302 / 3,302 |

² Transmitted frame 2 is missing in both decoders. A 14 ms arrival interval (T_b = 11.1 ms) left payload bit 117 without any sample, so the slot is skipped (`decode_v4.py:148`) and counted as missing. The payload rule played no part.

---

## 7. Repeated-payload risk (item 5)

**The payload is identical in every frame.** All 24 frames of every run carry the same 127 bits (`stent_tx.py:53`: `frame_bits × frames`). Every saved `sent_payload` equals the independently regenerated PRBS7 shifted by 120.

**Could the decoder exploit the repetition directly?** No. Quantities shared across frames:
- τ and ε: header only.
- Node ranking: energy.
- z-score statistics: the whole record's distribution.
- Baseline: time-local.

Bit decisions use only the samples of their own frame and bit (`decode_frames` loops per slot, 137-160). No step averages payload bits across frames.

**Could it benefit indirectly?**
- **Development used the same sequence.** The v4 rules (absdev, 2 ms guard, max-combining) were developed on runs with this sequence, to remove "spill-over" errors at specific bit contexts. The validation runs are independent recordings (new noise, new sample phases) but **not** a new payload.
- **No sign of pattern-specific tuning at 80 bps.** If the rules had been tuned to particular positions, low-margin bits would recur at the same payload positions. They don't:
  - Spearman correlation of the per-position mean margin between the three runs: −0.18, +0.01, −0.01.
  - Low-margin events (< 0.1) fall on 15, 26 and 31 distinct positions, at most twice per position.
  - Each position was sampled at 15–20 different phases.
- **At 90 and 100 bps errors *are* pattern-dependent:** they concentrate on 0s after a 1 (§10). A PRBS7 contains every 7-bit context exactly once per frame, so the 72 frames contain only 127 distinct contexts, repeated with different noise. That is fine for measuring noise, but limits coverage of pattern effects.

**Effect on the confidence bound.** The 72 frames are not 72 independent pattern samples, and error mechanisms cluster in time (§10). The 95 % bound of 1 − 0.05^(1/9144) = 0.033 % assumes independent bits, so it is optimistic.

---

## 8. Negative controls (item 6; Figures C and B)

All were decoded with the blind decoder at the 80 bps frame structure. Every accepted frame below is a false detection. The table counts frame slots and how many each acceptance rule passes.

| Control | Slots | r ≥ 0.3 (v4 header rule) | r ≥ 0.7 | ≤ 2 header-bit errors |
|---|---|---|---|---|
| A1 TX dead, 210 s (`stent_ber_c156_20bps_20261022_160629`: 0 % signal windows), own nodes / Y2-Y4 | 98 / 97 | 9 / 14 | 0 / 0 | 0 / 0 |
| A2 no carrier at panel, clean, 30 s (`onoff_precheck_…_161009`) | 11 / 11 | 2 / 3 | 0 / 0 | 0 / 0 |
| A3 no carrier, ±100-180-count disturbances on Y2-Y4, 30 s (`…_165856`) | 12 / 11 | 1 / 1 | 0 / 0 | 0 / 0 |
| A4 transmitter-off leads of all 24 BER runs, concatenated (96 s) | 37 / 41 | 5 / 7 | 0 / 0 | 0 / 0 |
| B1 OFF bootstrap on real timestamps (lead resampled in 25-sample blocks), 20 × 3 runs | 1,506 | 192 (12.7 %) | 1 | 0 |
| B3 sample-permuted captures (timing destroyed, amplitudes kept), 20 × 3 | 1,511 | 221 (14.6 %) | 1 | 0 |
| B2 time-reversed captures, 3 | 75 | **70** | **66** | 0 |
| C τ shifted by ±2 bits on the real captures | 156 | **144** (BER ≈ 0.50) | – | 0 |
| **True frames, 80 bps** | **72** | **72** | **72** | **72** |

**Findings:**
- **The v4 header rule alone is not a safe detector.**
  - **Noise:** pure noise passes r ≥ 0.3 in 10–15 % of slots. The best-τ search is run on the noise itself, so it picks alignments that maximise correlation.
  - **±2-bit misalignment:** a frame shifted by ±2 bits keeps a correlated preamble (r ≈ 0.43) while its payload is random.
  - **Time reversal:** reversed frames correlate at r ≈ 0.75.
  - **Why the real runs were unaffected:** in a payload-blind receiver, all of these would be delivered as data. In the real runs the global timing peak was unambiguous (header score 22.9 / 22.9 / 22.7 against 14.2 / 14.8 / 14.4 for the best alignment more than one bit away, the ±2-bit preamble sidelobes; Figure B b). Both out-of-transmission slots scored r ≤ 0.09.
- **A payload-blind rule separates cleanly:** ≤ 2 of 27 decoded header bits wrong.
  - **On controls:** 0 false detections in 3,566 control slots (95 % upper bound 0.084 % per slot).
  - **On true frames:** it accepts all 72 at 80 bps (all had 0 header-bit errors) and all frames at 90 and 100 bps (at most 1 and 2 header-bit errors).
  - **Caveat:** it was chosen after seeing these controls, so it is a recommendation for v5, not part of v4.

**Timing falsification** (Figure B a). All 24 frames were scored whether accepted or not:

| Offset from found τ | −6 ms | −4 | −3 | −2 | −1 | **0** | +1 | +2 | +3 | +4 | +6 ms | ±1, ±2, ±3 bits | ε = 0 | ε ± 100 ppm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Errors (3 runs pooled, /9,144) | 2,177 | 1,064 | 493 | 132 | 15 | **0** | 1 | 103 | 478 | 1,001 | 2,156 | 4,463–4,607 (≈ 0.5) | **0** | 152–264 per run |

The decoder fails exactly as physics predicts when timing is wrong. It is error-free only within about ±1 ms of the header-derived timing, inside an eye of roughly 4 ms in a 12.5 ms bit.

**Wrong payloads.** The 72 decoded payloads were scored against every other cyclic shift of the PRBS7: each scores exactly 1,536 / 3,048 errors per run. That's the m-sequence property, 64 of 127 bits differ. Against 10,000 random sequences the minimum is 1,008 and the mean 1,524. The decoded bits carry the transmitted sequence and no other.

---

## 9. Independent scoring (item 8)

| Check | Result |
|---|---|
| **Payload regeneration** | Recurrence and LFSR give the same sequence (64 ones, period 127, no smaller period). Shifted by 120 it equals `decoded_v4.json → sent_payload` in all 11 runs scored. |
| **What was transmitted** | Checksum 3089373205 recomputed from these bits equals the Teensy's upload report (audit of 2026-10-22). This proves the uploaded schedule, not the emitted signal (§11, L15). |
| **Frame matching** | k = round((s0 − t_first)/P′); all 72 accepted frames map to k = 0…23 once each per run; residuals 11.22 / 11.44 / 11.26 ms; no duplicates, no out-of-transmission acceptances. |
| **BER denominator** | 127 × matched frames = 3,048 per run, 9,144 pooled. Equal to `result_v4.json` (`bits`, `errors`, `frames_valid`). |
| **Skipped bits** | 0. Every payload bit of every true frame has ≥ 1 sample: 85 decided from edge samples (fallback), 0 without a sample. |
| **Lost-frame handling** | Missing frames are counted separately. The conservative BER counts each as 50 % errors (0 here); false-positive and misaligned frames would be counted with their errors. |

---

## 10. Trying to make the zero fail: physical error mechanisms

The zero was attacked with all the data, not only the three runs.

### 10.1 Bit-level inspection (item 7; Figure D)

Eight bits were plotted from raw samples with:
- the inferred bit edges and the ±2 ms guard;
- the OFF baseline and ±3σ noise band;
- the per-node z-features and their max;
- the frame threshold, the per-bit decision value, and sent and decoded bits.

| Panel | Bit (run, frame, payload bit) | Margin | What the raw data show |
|---|---|---|---|
| random correct 1 | 195109, 19, 126 | +0.53 | One inner sample at ~7 ms: \|Δ\| ≈ 10³ on all three nodes |
| random correct 0 | 195008, 21, 0 | +0.48 | One inner sample within ±3σ of baseline on all nodes |
| lowest-margin sent 1 | 195109, 18, 49 | **+0.007** | A *weak* ON reading: 50–100 counts on the three nodes at 8.8 ms, mid-bit. A fade (the carrier-induced delta happened to be small), not an edge effect |
| 2nd lowest sent 1 | 195008, 12, 27 | +0.013 | Same: weak ON reading (~100 counts) at 9.4 ms |
| lowest-margin 1 → 0 | 195109, 9, 104 | **+0.028** | Two inner samples. The first (2.0 ms, just past the guard) still reads ON (≈ 900 counts): **a stale sample, arriving 10.0 ms after its predecessor instead of 8 ms**. The second (10 ms) is clean OFF. Their mean falls just below the threshold |
| 2nd lowest 1 → 0 | 192702, 19, 21 | +0.30 | Clean |
| random 1 → 0 | 195109, 1, 43 | +0.48 | Clean OFF samples after a strong ON bit; no tail |
| edge-only (fallback) | 195008, 21, 75 | +0.23 | No inner sample; decided from its two edge samples (0.7 ms and ~10.9 ms), both reading ON |

**Where the near-misses sit.** Of the 25 bits with margin < 0.05, 24 are sent 1s read weakly mid-bit (3–10 ms) and only one is a 0 after a 1. At 80 bps the residual risk is ON fades and stale samples, not filtering or inter-symbol interference.

### 10.2 Stale samples after stalled transfers (`stalls.py`)

**The 40 bps errors.** Both of the 40 bps run's errors are full-amplitude ON values (−3,166 and −3,214 counts) inside a sent 0. Each arrived 14 and 16 ms after the previous sample instead of 8 ms, so its host timestamp is 6–8 ms later than its scan.

| Run | Samples in payload | After a stall (> 10 ms) | Stale value on wrong side | …and used in the decision | Error bits with a stall sample / all errors |
|---|---|---|---|---|---|
| 40 bps 192306 | 8,882 | **1,162 (13 %)** | 19 | 5 | **2 / 2** |
| 38, 45, 68 bps | 9,896 / 8,346 / 5,518 | 237 / 235 / 176 | 10 / 7 / 5 | 0 / 2 / 0 | 0 / 0 / 0 |
| **80 bps ×3** | 14,110 | **317 (2.2 %)** | 18 | **1 (outvoted)** | **0 / 0** |
| 90 bps 195206 / 195302 | 4,179 / 4,003 | 96 / 103 | 7 / 11 | 2 / 5 | 0 / 4, **6 / 15** |
| 100 bps 192753 | 3,761 | 95 | 6 | 6 | **5 / 32** |

**Implications:**
- **The decoder can't tell a stale sample apart.** It has no way to recognise one, because the panel's acquisition time is not recorded.
- **Stall rate varies by recording.** The 80 bps recordings had a normal rate (~2 %). The 40 bps recording had six times that.
- **The 80 bps near-miss survived by luck of sampling.** Of 317 stall samples, the only stale one used in a decision was outvoted by a clean second sample in the same bit. At 80 bps 93 % of bits have only one inner sample, so a stale sample landing at a bit transition will produce an error.

### 10.3 Sample-level error rate (more samples than 9,144 bits; `sensitivity.py` S2)

In all ten 155.7 kHz validation runs, every guard-passing sample was compared with its frame's threshold. Only "deep" bits count: bits whose neighbours carry the same value, so no edge or stale effect is possible.

| | Samples | On the wrong side |
|---|---|---|
| Deep ON (sent 1 between 1s) | 5,447 | 1: a fade, 141/86/−59 counts (40 bps run, t = 7.59 s) |
| Deep OFF (sent 0 between 0s) | 5,205 | 3: two isolated +250 to +500-count glitches on Y2/Y4 (40 bps run, t = 13.67 s and 22.17 s) |

**Pooled per-sample error rate.** It is 4 / 10,652 = 3.8 × 10⁻⁴ (95 % CI 1.0–9.6 × 10⁻⁴), with all four events in one recording.
- **At 40 bps** (~3 samples per bit) the other samples of those bits outvoted them.
- **At 80 bps** one sample decides 93 % of bits, so each such event would be a bit error.
- **Interpretation:** the 80 bps upper bound of 3.3 × 10⁻⁴ and this per-sample rate are consistent, and together they say the true 80 bps BER is plausibly ~10⁻⁴. The three 80 bps recordings had no glitch, fade or decisive stale event. Their zero reflects that, not an error-free mechanism.

### 10.4 How narrow is the working point (Figure E; informational)

Errors in 9,144 bits when one frozen rule is changed:

| Change | Errors |
|---|---|
| none (frozen v4) | **0** |
| acceptance: header errors ≤ 2 | 0 |
| no clock correction | 0 |
| nearest-to-centre sample only | 1 |
| guard 3 ms | 3 |
| guard 1 ms | 11 |
| guard 4 ms | 50 |
| mean instead of max of 3 nodes | 80 |
| top 2 nodes | 84 |
| Y3 only / Y2 only / Y4 only | 93 / 96 / 122 |
| guard 0 ms | 166 |

**Each node alone runs at ~1 % BER.** The zero comes from max-combining three nodes, whose fades are nearly independent, together with a 2–3 ms guard.
- **These are not post-hoc choices.** The rules were frozen at 19:16 on 34–68 bps development runs, and the 80 bps runs were recorded at 19:27, 19:50 and 19:51.
- **Still, the working point is narrow.** The guard optimum depends on bit rate: at 100 bps a 1 ms guard gives 0 errors and the frozen 2 ms gives 32. The result should not be presented as robust to decoder details.

---

## 11. Failure and leakage points

| # | Point | Where | Severity | Evidence | Fix |
|---|---|---|---|---|---|
| L1 | Frame acceptance uses the payload (`ber < 0.35`) | `decode_v4.py:157,160` (inherited from `cap_ook_rx`) | **High in principle, no effect here** | Wrong payload → 0/24 frames valid (§4.2); true payload: 0 frames rejected only by this rule; removing it changes nothing (§6) | Payload-blind acceptance; report BER only with it |
| L2 | Header r ≥ 0.3 alone is a weak detector | `decode_v4.py:155-156,160`; `RX.R_LOCAL` | **Medium** | 10–15 % of noise slots, 93 % of time-reversed, 92 % of ±2-bit-shifted frames pass (§8); no false frame here because the timing peak was unambiguous | ≤ 2 of 27 header bits wrong: 0 false in 3,566 control slots, 72/72 true |
| L3 | Spurious frames can offset missing ones | `decode_v2.py:58` | Low–medium | Not triggered (0 false positives) | Match frames to transmitted frames by time (`score.py`) |
| L4 | Host arrival time used as acquisition time | `maxtouch_stent_rx.py:250`; decoder `bit_index` | **Medium–high for generalisation** | Both 40 bps errors, 6/15 (90 bps), 5/32 (100 bps), and the closest 80 bps 1 → 0 near-miss are stale samples (§10.2) | Flag or drop the first sample after an interval > 1.25 × median (a pre-registered v5 rule); log stall rate per run; investigate host load |
| L5 | One sample decides 93 % of bits at 80 bps | sampling (8 ms) vs bit (12.5 ms) | **Medium** | Per-sample error mechanisms at ~10⁻⁴ (§10.3) translate directly into BER | Longer runs; more nodes; or accept that 80 bps is near the one-sample limit |
| L6 | Narrow working point (timing ±1 ms, guard 2–3 ms, 3-node max) | rules 2–4 of v4 | Medium | Figure B a, Figure E | Report alongside the BER; keep rules frozen; validate on new sessions |
| L7 | Same 127-bit payload in every frame and in development | `stent_tx.py:53`, `stent_frames.py:38` | Medium (indirect) | No position dependence at 80 bps (Spearman −0.18…0.01), but only 127 contexts tested | New payload per run (seed withheld until scoring), or random per frame |
| L8 | Independence assumed in the 95 % bound | report convention | Medium | Error mechanisms cluster by recording (stall rate 2 % vs 13 %; all deep-sample events in one run) | Report per-run results; collect runs across sessions; state the bound as conditional |
| L9 | Validation shares session and setup with development | protocol | Medium | All runs 2026-10-22 17:55–20:03, same placement | Repeat on another day with re-placement |
| L10 | Known-timing path marks every frame valid | `decode_v4.py:160` (`known_timing or …`), 225-231 | Low | Separate result, not the headline | Keep it labelled as a diagnostic |
| L11 | Best-of-3 node chosen after decoding | `decode_v4.py:221` | Low | Labelled "optimistic", not used | – |
| L12 | Health check windows from `t_first` | `decode_v4.py:164-175` | Low | Flag only | – |
| L13 | Docstring says running median; code uses block medians | `decode_v4.py:71` vs `cap_ook_rx.py:202` | Cosmetic | – | Fix the text in v5 |
| L14 | `ber_upper95` = 3/bits (rule of three) in `frames_summary` | `decode_v2.py:60` | Low | The summary uses 1 − 0.05^(1/n) instead; both are reported clearly | – |
| L15 | Emitted signal not recorded | protocol | Low–medium | Only the uploaded schedule (checksum) is verified; firmware `sstat` (late steps) not logged | Log `sstat` after each run; optionally record A0 or a scope trace |

---

## 12. Verdict

**Independently verified (with code paths and numbers above):**
1. **The count:** 0 errors in 9,144 payload bits in the three runs. All 72 transmitted frames were recovered, with no false, misaligned, duplicate or partial frames and no skipped bits.
2. **Payload-blind:** the result holds without the payload. It's reproduced by an independent decoder that never had access to the payload, with header-only acceptance and with no metadata timing at all. It is bit-identical to the frozen decoder.
3. **No leakage into the bits:** no payload or BER information enters synchronisation, node choice, baseline, threshold or any bit decision. The only payload dependence (frame acceptance) rejected nothing in these runs.
4. **The scored bits are the transmitted bits:** independent PRBS regeneration, the matching transmitter checksum, a unique match against every alternative alignment, and time-matched frames.
5. **The pipeline is capable of reporting errors:** the same scorer finds exactly v4's errors at 40, 90, 100 bps and at 163.6 kHz. Mistimed decoding produces errors as expected.

**Still uncertain:**
1. **The actual BER at 80 bps.** 0/9,144 bounds it below 3.3 × 10⁻⁴ only if errors are independent. They aren't: they cluster by recording. The per-sample error rate seen elsewhere (~4 × 10⁻⁴) suggests a true 80 bps BER of order 10⁻⁴, which these three runs were too short and too quiet to show.
2. **Reproducibility across sessions, placements and payloads.** All evidence comes from one session, one stent placement and one 127-bit sequence.
3. **Robustness to host timing.** The result depends on few USB stalls, and nothing in the decoder detects a stale sample.

**Necessary further experiments** (pre-register the decoder before recording):
1. **New payloads, kept from the analyst:** a different sequence per run, e.g. PRBS9/PRBS11 segments or random bits from a seed revealed only after decoding is frozen. The 4,096-word schedule allows 24 frames of 170 bits at 80 bps per upload. Repeat the uploads, or use `sseg` reps for longer runs of the same sequence.
2. **Enough bits for a meaningful bound:** ≥ 30,000 bits with 0 errors for BER < 10⁻⁴ (95 %), i.e. about 10 runs of 24 frames, spread over at least two sessions with the stent re-placed. Report per-run counts, not only the pooled number.
3. **Payload-blind acceptance in the frozen rules:** ≤ 2 header-bit errors, with the payload-BER condition removed. A rule that flags or drops the first sample after a stalled transfer, frozen before the new data.
4. **Logging to settle the open mechanisms:** the stall rate per run; the Teensy's `sstat` (late steps) after each run; optionally an A0 or scope record of the emitted OOK.
5. **Rate bracket with the same protocol:** 68, 80 and 90 bps, to state the highest rate meeting a target BER with a bound, rather than a single zero-error rate.

**Bottom line:** the 0/9,144 result is genuine. It is not an artefact of ground-truth leakage, frame selection, scoring or timing assumptions. It is a small-sample result from a favourable session, at a bit rate where one sample decides nearly every bit, and it should be reported that way.

---

## 13. Files

Folder: `data_maxtouch/audit_v4_80bps_falsification_20261023/`. Reproduce with `./run_all.sh` (it checks input hashes before and after).

| File | Contents |
|---|---|
| `blind_decoder.py`, `run_blind.py`, `blind_out/` | Payload-blind decoder and its outputs (both modes, 11 runs) |
| `score.py`, `score_results.json` | Independent scoring; frozen v4 vs blind; bit agreement |
| `leakage_test.py`, `leakage_results.json` | Static trace and code-path test on the frozen functions |
| `controls.py`, `controls_results.json`, `controls_log.txt` | Negative controls, timing falsification, wrong payloads |
| `mechanism.py`, `mechanism_results.json`, `bits_table.npz` | Sample → bit statistics (one row per payload bit) |
| `sensitivity.py`, `sensitivity_results.json`, `sensitivity_log.txt` | One-rule-at-a-time variants; sample-level error rate |
| `stalls.py`, `stalls_results.json` | Stalled transfers and stale samples |
| `figures.py`, `figA_mechanism`, `figB_timing`, `figC_controls`, `figD_bits`, `figE_sensitivity` (.png/.pdf), `figures_index.json` | Figures and the bits shown in Figure D |
| `sha256_before.txt` | Hashes of the 254 input files |
