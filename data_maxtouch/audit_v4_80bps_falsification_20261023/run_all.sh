#!/usr/bin/env bash
# Reproduces every number and figure of Audit_v4_80bps_falsification.md (~2 min). Read-only on all run folders.
set -euo pipefail
cd "$(dirname "$0")"
PY=~/.venvs/galvanic/bin/python
ROOT=../..
( cd "$ROOT" && sha256sum -c --quiet data_maxtouch/audit_v4_80bps_falsification_20261023/sha256_before.txt ) && echo "inputs unchanged"
$PY -I run_blind.py --all      # 1. payload-blind decoding (own process, no payload-carrying module loaded)
$PY -I score.py                # 2. independent scoring, frozen v4 output scored the same way
$PY leakage_test.py            # 3. leakage tests on the frozen decode_v4 functions (imports decode_v4; writes nothing in run folders)
$PY -I controls.py > controls_log.txt   # 4. negative controls, timing falsification, wrong payloads
$PY -I mechanism.py > /dev/null         # 5. sample -> bit mechanism
$PY -I sensitivity.py > sensitivity_log.txt  # 6. one-rule-at-a-time sensitivity (informational)
$PY -I stalls.py               # 7. late-stamped samples after stalled transfers
$PY -I figures.py              # figures A-E
( cd "$ROOT" && sha256sum -c --quiet data_maxtouch/audit_v4_80bps_falsification_20261023/sha256_before.txt ) && echo "inputs still unchanged"
