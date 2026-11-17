#!/usr/bin/env python3
"""
Decoder for trx-frame BER runs through the body to a phone whose touch panel answers a carrier with a PANEL-WIDE LEVEL
SHIFT (pixel6_trx_ber.py; Pixel 6 sweep 2026-11-14: at 54.8 / 276.8 kHz the median of all non-finger nodes steps from
~+4 to ~-4 counts within one read of the carrier edge and back within one read after it; at 15.9 / 37.4 / 88.2 kHz it
rises +5..+13 over ~1 s; no-TX control: no shift at any carrier). Written 2026-11-15.

Rules (fixed 2026-11-15 before any recording with this decoder):
  L1. Finger columns: column of the largest |median map| over the lead reads (t < t_first, transmitter off); that
      column +-4 is left out (the Pixel 6 finger shows as a streak along its column, sweep 2026-11-14).
  L2. P_k = median of the remaining nodes of read k (frames.npz F = raw - finger-off baseline).
  L3. Feature y_k = |P_k - median(P over the lead reads)|: a step down (54.8 kHz) and a rise (37.4 kHz) both give a
      positive y, so the OOK/FSK sync reference (state 1 above state 0) holds for either sign.
  Then decode_trx_pad rules 2-8 unchanged (decode_trx_pad.decode_arrays: known-timing search tau = t_first +
  0..600 ms and +-200 ppm on the Pearson correlation with the sent states; mean y per symbol; per-frame sync reference;
  decode_trx_tablet soft values, every trx decoder, CRC-16; before / after FEC accounting). OOK and FSK only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode_trx_pad as DP                     # noqa: E402  (rules 2-8; it calls the frozen decode_trx_tablet)

VERSION = "trx-level-1"
FROZEN_AT = "2026-11-15 01:00"
RULES_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]
PAD_RULES_SHA = DP.RULES_SHA
TABLET_SHA = DP.TABLET_SHA
DT = DP.DT
FINGER_HALF = 4


def finger_columns(Fm, lead):
    """L1"""
    med = np.median(np.asarray(Fm, float)[lead], axis=0) if lead.any() else np.zeros(Fm.shape[1:])
    c = int(np.unravel_index(int(np.argmax(np.abs(med))), med.shape)[1])
    return max(0, c - FINGER_HALF), min(Fm.shape[2], c + FINGER_HALF + 1)


def feature(Fm, t, t_first):
    """L1-L3 -> (y, P, (c0, c1))"""
    Fm = np.asarray(Fm, float)
    lead = t < t_first
    c0, c1 = finger_columns(Fm, lead)
    keep = np.ones(Fm.shape[1:], bool)
    keep[:, c0:c1] = False
    P = np.median(Fm[:, keep], axis=1)
    ref = float(np.median(P[lead])) if lead.any() else float(np.median(P))
    return np.abs(P - ref), P, (c0, c1)


def decode_run(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    if meta.get("modulation", "ook") == "dbpsk":
        raise SystemExit("decode_trx_level: OOK / FSK only")
    z = np.load(os.path.join(d, "frames.npz"))
    sq = np.load(os.path.join(d, "tx_seq.npz"))
    t = z["t"].astype(float)
    y, P, cols = feature(z["F"], t, meta["t_first"])
    res = DP.decode_arrays(t, y, meta, sq)
    res.update(decoder=VERSION, rules_sha=RULES_SHA, frozen_at=FROZEN_AT, pad_rules_sha=PAD_RULES_SHA,
               nodes=[f"median of non-finger nodes (columns {cols[0]}-{cols[1] - 1} left out)"],
               run=os.path.basename(os.path.normpath(d)))
    json.dump(res, open(os.path.join(d, "result_trx.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = decode_run(p)
        print(p, f"delay {r['delay_ms']:.0f} ms", r["state_errors"], "/", r["state_symbols"], len(r["frames"]), "frames")
