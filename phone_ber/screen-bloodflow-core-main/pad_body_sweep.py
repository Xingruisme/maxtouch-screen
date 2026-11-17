#!/usr/bin/env python3
"""Through-body carrier sweep on the OnePlus Pad via the shared phone sweep (phone_body_sweep.py), so the Pad gets the
same --quick, --current-ma (IEC 60601-1 BF check), model check and `response` figure as the 8 Pro / Pixel 4 / Pixel 6.

Reader: oneplus_monitor.MatrixStream on /proc/touchpanel/debug_info/delta (Novatek NT36532, 38 x 54, ~6 frames/s,
root via Magisk) - the same reader as oneplus_body_sweep.py. The delta map is already baseline-subtracted by the
controller (idle ~0), so no finger-off baseline is taken. Keep the screen ON for the whole run.

Usage (galvanic venv):
    python pad_body_sweep.py plan --quick --carriers log:1e3:500e3:10% --current-ma 0.1
    python pad_body_sweep.py run --quick --label ctl --no-tx --carriers log:1e3:500e3:10%
    python pad_body_sweep.py run --quick --label hand_1k_500k --site hand2finger --carriers log:1e3:500e3:10% --current-ma 0.1
    python pad_body_sweep.py response data_oneplus/body_sweep/<run> --control data_oneplus/body_sweep/<control run>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from oneplus_capacitance import COLS, ROWS, resolve_adb_serial   # noqa: E402
import oneplus_monitor as MON                                      # noqa: E402
import phone_body_sweep as PS                                      # noqa: E402

PHONE = PS.Phone(
    name="OnePlus Pad", panel="Novatek NT36532, 38 x 54", rows=ROWS, cols=COLS,
    stream=lambda adb: MON.MatrixStream(adb, "delta"),
    read_once=lambda adb: MON.read_matrix(adb, "delta"),
    resolve_serial=resolve_adb_serial,
    model_ok=lambda s: any(k in s.upper() for k in ("OPD2203", "OP59BCL1", "ONEPLUS PAD")),
    out_dir=HERE / "data_oneplus/body_sweep", fps_note="~6 frames/s measured 2026-11-13", default_block=2.0,
    default_min_blob=2, min_blob_note=": same as the phones (the Pad's own script used 1; idle delta ~0)",
    script=__file__)

if __name__ == "__main__":
    sys.exit(PS.main(PHONE))
