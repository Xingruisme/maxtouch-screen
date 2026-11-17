#!/usr/bin/env python3
"""Through-body carrier sweep on the Pixel 6: TeensyTRX (AD9833) -> body -> finger -> Pixel 6 touch panel.

Reader: pixel6_capacitance.MatrixStream (stm_fts_cmd command 13 via Magisk su, 16 x 34, ~3.1 frames/s, PROBE.md): 3 s
blocks by default so each block still has ~8 frames after the edge guards.
Shared run / schedule / current check / analysis: screen-bloodflow-core-main/phone_body_sweep.py (the OnePlus Pad
sweep's schedule and analysis, unchanged). Keep the screen ON for the whole run.

Usage (galvanic venv):
    python pixel6_body_sweep.py plan [--current-ma 1]
    python pixel6_body_sweep.py run --label ctl --no-tx                                         # control: finger on, nothing sent
    python pixel6_body_sweep.py run --label hand_lo --site hand2finger --carriers log:100:10e3:10% --current-ma 0.1
    python pixel6_body_sweep.py run --label hand_hi --site hand2finger --carriers log:10.1e3:500e3:10% --current-ma 1
    python pixel6_body_sweep.py response data/body_sweep/<run> --control data/body_sweep/<control run>
Default carriers log:100:500e3:10%; a person-connected run needs --current-ma and is refused where it exceeds the
IEC 60601-1 type-BF patient-current limit (0.1 mA up to 1 kHz, x f/1 kHz above, 10 mA from 100 kHz).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pixel6_capacitance as DEV                                        # noqa: E402  (adds screen-bloodflow-core-main to sys.path)
sys.path.append(str(HERE.parent / "screen-bloodflow-core-main"))
import phone_body_sweep as PS                           # noqa: E402

PHONE = PS.Phone(
    name="Pixel 6", panel="STM FTM5, 16 x 34", rows=DEV.ROWS, cols=DEV.COLS,
    stream=lambda adb: DEV.MatrixStream(adb, "delta"),
    read_once=lambda adb: DEV.read_matrix(adb, "delta"),
    resolve_serial=DEV.resolve_adb_serial,
    model_ok=lambda s: any(k in s.upper() for k in ("PIXEL 6", "ORIOLE")),
    out_dir=HERE / "data/body_sweep", fps_note="~3.1 frames/s (stm_fts_cmd dump cost, PROBE.md)", default_block=3.0, default_min_blob=2,
    min_blob_note=": phone maps idle at a few counts on every node; 1 node gave false carriers in the 8 Pro simulation",
    to_screen=lambda r, c: (c, r, DEV.COLS, DEV.ROWS),   # pixel6-cap-monitor FtsReader.mapForPortraitDisplay: plain transpose
    baseline_frames=8,                                   # stm_fts_cmd 13 is raw: subtract a finger-off baseline
    script=__file__)

if __name__ == "__main__":
    sys.exit(PS.main(PHONE))
