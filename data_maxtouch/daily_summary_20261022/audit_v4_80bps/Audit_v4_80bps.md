# Audit: frozen v4 decoder, 155.7 kHz / 80 bps validation runs

**Scope:**
- **Decoder:** `decode_v4.py`, sha256 prefix `59cb5e989bc9`. It is read-only and byte-identical to `data_maxtouch/decode_v4_frozen_59cb5e989bc9.py`.
- **Runs:**
  - `stent_ber_val4_80bps_20261022_192702`
  - `stent_ber_val4b_80bps_20261022_195008`
  - `stent_ber_val4b_80bps_20261022_195109`

All three are validation runs: recorded after the freeze, and all four of their `result_v4.json` files carry that sha.

**Nothing was modified.** The decoder, `capture.npz`, `meta.json` and every result and decoded file are unchanged. The audit code (`audit_v4_80bps.py`) and its output (`audit_results.json`) are new files in this folder. The audit is not a re-optimisation: nothing was re-decoded with new rules.

## 1. Does the blind decoder use the known payload?

Code review of `decode_v4.py` (line numbers refer to the frozen file):

| Stage | Lines | What it uses | Uses the payload |
|---|---|---|---|
| Node choice (top 3) | 192–195 | Energy change of raw samples before vs after the START-TX prompt time | No |
| Feature (absdev) | 70–82 | OFF baseline and noise from the lead, before transmission | No |
| Timing (offset τ, clock ε) | 112–125, 94–109, 204 | Summed preamble + SFD (27-bit header) correlation over all frame slots in the recording | No |
| Threshold | 150–152 | Mean of the known header bits and the 16 trailing gap zeros (fixed frame structure, not payload) | No |
| Bit decision | 153 | `v > thr` | No |
| Edge guard | 134–147 | Samples more than 2 ms from both bit edges; a bit without such a sample falls back to all its samples | No |
| Frame dropped | 148–149 | A frame with a header or payload bit containing *no* sample is skipped and counts as missing | No |
| **Frame acceptance** | **160** | **`valid = r ≥ 0.3 and frame BER < 0.35`. The second condition uses the known payload** | **Yes** |
| Best-of-3 single node | 221 | Chosen after decoding, by BER; reported as "optimistic" only, not used for the combined result | Yes, reporting only |
| Accounting | `decode_v2.frames_summary` | Counts accepted frames without capping at the number sent; `missing = max(sent − accepted, 0)` | n/a |

**The finding:** timing, thresholds, node choice and bit decisions use no payload information. **Frame acceptance does.**
A frame whose header correlates (r ≥ 0.3) but whose BER is ≥ 35% would be left out of the BER denominator and counted
as missing. The accounting would also let a spurious extra frame add bits. Section 2 checks whether either happened in these three runs.

## 2. Independent recount

**The transmitted payload, regenerated from its specification** (PRBS7 x⁷ + x⁶ + 1, Fibonacci LFSR, seed all ones;
64 ones in 127 bits, i.e. an m-sequence):
- In all three runs, the saved `sent_payload` equals this sequence shifted cyclically by 120, the documented start offset.
- The frame was rebuilt from its specification (16-bit preamble, 11-bit Barker SFD, payload, 16 gap zeros) × 24 frames at 80 bps.
- From those rebuilt frames, the transmitter's schedule checksum comes out as **3089373205**. That equals the checksum the Teensy reported when the schedule was uploaded (run-time log of 195008 and 195109; the 192702 upload line wasn't captured, but its schedule is identical). **The bits compared are the bits that were transmitted.**

| | 192702 | 195008 | 195109 | Pooled |
|---|---|---|---|---|
| Frames transmitted | 24 | 24 | 24 | 72 |
| Candidate frame slots in the recording | 26 | 26 | 26 | |
| Frames accepted | 24 | 24 | 24 | 72 |
| Accepted frames ↔ transmitted frames | 0–23, each exactly once | same | same | |
| Max timing offset from expected frame start | 0.03 ms | 0.06 ms | 0.01 ms | |
| Rejected candidates | 2, both outside the transmission (before and after), header r < 0.3 | same | same | 6 |
| Rejected **only** by the payload-BER rule | 0 | 0 | 0 | **0** |
| **Payload bits compared** | 3,048 | 3,048 | 3,048 | **9,144** |
| **Errors, recounted** (`decoded_v4.json` vs regenerated PRBS) | **0** | **0** | **0** | **0** |
| Errors in `result_v4.json` | 0 / 3,048 | 0 / 3,048 | 0 / 3,048 | match |
| Excluded or skipped payload bits | 0 | 0 | 0 | 0 |

- **No transmitted frame was lost or rejected.** The two rejected candidates per run lie before the first and after the last transmitted frame, where no signal exists (BER 0.47–0.58), and the header criterion alone rejects them. **The payload-dependent rule rejected nothing** in these runs. There were no spurious extra frames.
- **Result:** 0 errors in 9,144 payload bits. The one-sided 95% upper bound is 1 − 0.05^(1/9144) = **0.033%**, assuming independent bit errors.

## 3. Edge guard: does it hide difficult bits?

Each payload bit was re-evaluated from the raw sample timestamps, using the decoder's own frame timing:

| | 192702 | 195008 | 195109 |
|---|---|---|---|
| Bits with ≥ 1 sample outside the ±2 ms guard | 3,012 | 3,024 | 3,023 |
| Bits decided from edge-only samples (fallback) | 36 | 24 | 25 |
| Bits with no sample at all | 0 | 0 | 0 |
| Errors among fallback bits | 0 | 0 | 0 |

**The guard removes no bit from the denominator.** At 80 bps (12.5 ms bits, ~8 ms between samples), 85 of 9,144 bits
(0.9%) had only samples within 2 ms of an edge. They were decided from those samples and counted. No frame was dropped for missing samples.

## 4. Robustness of the zero-error result

**Raw check, independent of the decoder.** For each bit, I took the largest |delta − lead median| over Y2–Y4 among the samples outside the guard, relative to the lead noise (σ = 4.4 counts):

| | Sent 1: minimum | Sent 0: maximum | Sent-1 bits below 10σ | Sent-0 bits above 10σ |
|---|---|---|---|---|
| 192702 | 24.1σ | 3.8σ | 0 | 0 |
| 195008 | 21.8σ | 6.3σ | 0 | 0 |
| 195109 | 20.5σ | 208σ | 0 | **1** |

The single exception is in 195109, frame 11, payload bit 104: a sent 0 that follows a 1. Its first sample lies just past the 2 ms guard (at 0.16 of the bit) and still reads +928 counts, a leaked ON reading from the previous bit. The decoder averaged it with a clean OFF sample and decided 0, correctly, **but with a margin of only −0.028**.

**Decision margins** (normalised decision value: threshold 0, typical levels ±0.5):

| Margin on the correct side | 192702 | 195008 | 195109 | Pooled |
|---|---|---|---|---|
| < 0 (an error) | 0 | 0 | 0 | 0 |
| < 0.05 | 2 | 9 | 14 | 25 |
| < 0.10 | 18 | 30 | 34 | 82 (0.9%) |
| < 0.25 | 85 | 106 | 114 | 305 (3.3%) |
| Median | 0.52 | 0.52 | 0.51 | |

## 5. Conclusions

1. **The recount confirms the result:** 0 errors in 9,144 compared payload bits; all 72 transmitted frames recovered, matched one to one in time; no bit excluded or skipped. The bits compared are the transmitted bits (independent PRBS, matching transmitter checksum).
2. **Blind synchronisation, thresholds, node choice and decisions use only the header, the gap and lead-period data.**
3. **One payload dependence exists in the frozen decoder:** frame acceptance (`frame BER < 0.35`), inherited from the original receiver. It **had no effect on these three runs**, since no transmitted frame reached it. For future reporting, frame acceptance should use the header alone, or the rule should be stated next to every BER figure.
4. **The ±2 ms guard doesn't discard bits.** 85 edge-only bits were decided and counted.
5. **Caveat for reporting:** the zero-error result has thin margins on a small fraction of bits (25 bits below 0.05). The failure mechanism is visible: a sample right next to the guard still carrying the neighbouring bit's level. This fits errors appearing at 90 bps. The 95% bound (0.033%) assumes independent errors. Near-misses that cluster at edges argue for stating 80 bps as the highest verified zero-observed-error rate, not as a rate with comfortable margin.
