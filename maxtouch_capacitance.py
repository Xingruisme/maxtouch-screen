"""Paths and constants for maXTouch T37 capacitance session logs.

Sessions are recorded by ``maxtouch_cap_multifinger.py`` (capacitance only) or
``maxtouch_ppg_multifinger.py`` (cap + optional PPG) into ``data_maxtouch/cap_log/``.
Each folder contains long-format ``capacitance.csv`` (frame/row/col/delta),
optional ``ppg_green.csv``, and a ``{session}_regions.npz`` file from
``measure_maxtouch.py``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_MAXTOUCH = REPO_ROOT / "data_maxtouch"
CAP_LOG_DIR = DATA_MAXTOUCH / "cap_log"
