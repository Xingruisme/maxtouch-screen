# Daily summary, 22 October 2026: stent-to-maXTouch frequency sweeps and BER tests

Everything in this folder is generated from the saved run folders in `../` by one script, and nothing in those folders
is modified:

```
cd ~/xingru/maxtouch-screen-main/data_maxtouch/daily_summary_20261022
~/.venvs/galvanic/bin/python make_daily_summary.py      # needs numpy, scipy, matplotlib, openpyxl
```

| File | Contents |
|---|---|
| `Experimental_Report_20261022.md` | Methods, results, main findings, limitations, next steps |
| `Experimental_Summary_20261022.xlsx` | Sheets: Inventory, Frequency sweeps, ON-OFF tests, BER results, Decoder comparison, Carrier comparison, Run metadata. Derived columns (BER, 95% bounds, frame success, throughput) are Excel formulas over the raw counts; the workbook recalculates when opened |
| `summary_numbers.json` | Every number used in the figures, with the source run of each |
| `make_daily_summary.py` | Regenerates all of the above |
| `fig*.png` / `fig*.pdf` | Figures, 300 dpi PNG and vector PDF |

## Figures and their source runs

All run folders below are in `data_maxtouch/`.

1. **fig1_carrier_frequency_response:** best-node energy increase (dB) vs carrier.
   - (a) Fine sweeps 50–500 kHz: stent on glass, `stent_probe_trx_fine_20261022_105543`; hand electrode + finger, `stent_probe_trx_fine_20261022_155334`; stent on skin + finger, `stent_probe_phantom_sweep_20261022_203302`.
   - (b) Coarse 14-carrier sweep, `stent_probe_trx_wide_20261022_104923`.
   - Dashed lines: each run's response threshold. Values are recomputed from `capture.npz`; the stent runs match their saved `result.json`.
2. **fig2_spatial_response_heatmap:** node (Y0–Y15) × carrier heat map for the three fine sweeps, with each run's responding nodes boxed (glass Y2–Y4, skin Y0–Y2, hand electrode none). Same runs as Figure 1a.
3. **fig3_onoff_detection:** `stent_onoff_precheck_155.692kHz_20261022_174954`.
   - (a) Raw delta of Y3 over OFF / ON / OFF.
   - (b) Magnitude distributions per phase.
   - (c) Per-node spread from `onoff_stats.csv`.
4. **fig4_ber_vs_bitrate:** combined blind BER vs physical bit rate at 155.7 kHz for the original decoder, v3 and v4.
   - Data: `result.json`, `result_v3.json` and `result_v4.json` of all valid `stent_ber_*` runs at 155.692 kHz.
   - Markers: filled = validation, open = exploratory, triangles = 95% upper bound for zero-error runs.
5. **fig5_decoder_comparison:** error counts of the original decoder, v3, v4 single node and v4 combined on the same recordings:
   - `c156_68bps_..._181556`, `val4_68bps_..._192559`, `val4_80bps_..._192702`, `val4b_80bps_..._195008`, `val4b_80bps_..._195109`, `val4b_80bps_..._200327` (163.6 kHz).
6. **fig6_carrier_comparison:** v4 BER at 155.7 / 159.6 / 163.6 kHz for 20, 34 and 80 bps. Untested combinations are marked "not tested".
7. **fig7_multinode_combining:** v4 single node (Y2, Y3, Y4) vs combined. Validation runs at 155.7 kHz only, pooled per bit rate (`single_all` in `result_v4.json`).
8. **fig8_timing_and_decoder_analysis:**
   - (a) Raw samples vs the v3 interpolated grid around a v3 error: `stent_ber_c156_34bps_20261022_180957`, frame 9, payload bit 19, node Y3.
   - (b) Edge-averaged step response of the same run.
   - (c, d) A strong but slowly varying 1 bit that v3's change-based measure missed: `stent_ber_c160_20bps_20261022_190624`, frame 0, payload bit 44, node Y3.
9. **fig9_effective_throughput:** v4 payload throughput (correct bits/s, and payload in error-free frames only) vs physical bit rate at 155.7 kHz (`result_v4.json`, `decoded_v4.json`).
10. **fig10_decoder_methods:** summary of the decoders.
    - (a) Design choices of the original decoder, v3 and v4 (orange = sources of artificial errors, green = what removed them).
    - (b) One transmitted frame (frame 4 of 24, payload bits 40–79) of `stent_ber_val4_80bps_20261022_192702` through each decoder, from `decoded.json`, `decoded_v3.json` and `decoded_v4.json`. Each decoder's frames are matched to the transmitted frame by time. Errors in that frame: Original 17/127, v3 16/127, v4 0/127.
    - (c) Pooled BER vs bit rate at 155.7 kHz, validation runs only.

## Conventions

- **Roles (relative to frozen v4):** "exploratory" = the BER runs v4 was developed on (recorded before 19:16); "validation" = recorded after the freeze. Validation is independent of the decoder, not of the setup.
- **Signal-lost runs** (`stent_ber_c156_20bps_20261022_160629`, `_175518`, `_180016`) are in the inventory but excluded from every performance figure and comparison.
- **BER** is computed on decoded payload. Frame success and the conservative BER (undecoded frames counted as 50% errors) are reported separately; a missing frame is never counted as zero errors.
- **95% bounds:** zero errors → 1 − 0.05^(1/n); otherwise Clopper–Pearson. Both assume independent bit errors.
