#!/usr/bin/env python3
"""
Per-run (errors, bits) behind the pooled points of the paper figures, for their error bars: each bar spans the lowest
to the highest per-run BER of the runs pooled into the point (the convention of the slides: 'bars: lowest to highest
session'). Read from what the analyses already saved; nothing re-decoded.

  stent_modcomp runs  one row per recording in the session's summary.json (errors / bits, frozen decode_v4, known
                      timing) - per recording
  stent_trx_ber runs  one row per frame in the summary folder's frames.csv - summed per recording chunk (run)

A point backed by a single run has no spread: spread() returns None and no bar is drawn.
"""
from __future__ import annotations

import collections
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_maxtouch")


def _rows(session):
    return json.load(open(os.path.join(D, session, "summary.json")))["rows"]


def conditions_row(row):
    """[(errors, bits)] per recording of one row of a stent_modcomp conditions.csv"""
    want = row["run"].split("+")
    found = {r["run"]: (r["errors"], r["bits"]) for s in row["session"].split("+") for r in _rows(s)
             if r["run"] in want and r.get("bits")}
    assert len(found) == len(want), (row["condition"], row["modulation"], row["bps"], len(found), len(want))
    return [found[r] for r in want]


def sessions(names, mod, bps):
    """[(errors, bits)] per recording of one modulation and rate over stent_modcomp session folders"""
    return [(r["errors"], r["bits"]) for s in names for r in _rows(s)
            if r["modulation"] == mod and round(r["bps"]) == bps and r.get("bits")]


def trx(folder, decoder, mod, baud, after_fec=False, base=D):
    """[(errors, bits)] per recording chunk of a stent_trx_ber summary folder (frames.csv).
    before FEC: pre_err / code_bits (for 'none_full' the frame's own bits); after FEC: info_err / info_bits"""
    ke, kn = ("info_err", "info_bits") if after_fec else ("pre_err", "code_bits")
    acc = collections.OrderedDict()
    for r in csv.DictReader(open(os.path.join(base, folder, "frames.csv"))):
        if r["decoder"] == decoder and r["modulation"] == mod and round(float(r["baud"])) == baud:
            e, n = acc.get(r["run"], (0, 0))
            acc[r["run"]] = (e + int(r[ke]), n + int(r[kn]))
    assert acc, (folder, decoder, mod, baud)
    return list(acc.values())


def trx_frames(folder, decoder, mod, baud, after_fec=False, base=D):
    """[(errors, bits)] per frame of a stent_trx_ber summary folder (frames.csv, one row per frame; 96 info bits or 192
    code bits each) - the finest resolution the coded sessions have, used where a point is a single recording"""
    ke, kn = ("info_err", "info_bits") if after_fec else ("pre_err", "code_bits")
    out = [(int(r[ke]), int(r[kn])) for r in csv.DictReader(open(os.path.join(base, folder, "frames.csv")))
           if r["decoder"] == decoder and r["modulation"] == mod and round(float(r["baud"])) == baud]
    assert out, (folder, decoder, mod, baud)
    return out


def spread(runs, scale=1.0):
    """(lowest, highest) per-run BER x scale, or None for a single run"""
    if len(runs) < 2:
        return None
    b = [scale * e / n for e, n in runs]
    return min(b), max(b)


def yerr(y, spreads):
    """matplotlib yerr ([below], [above]) from pooled values and per-point spreads (None -> no bar)"""
    lo = [0.0 if s is None else max(yi - s[0], 0.0) for yi, s in zip(y, spreads)]
    hi = [0.0 if s is None else max(s[1] - yi, 0.0) for yi, s in zip(y, spreads)]
    return [lo, hi]


# the same error-bar look in every figure (print size)
EB = dict(elinewidth=0.7, capsize=1.8, capthick=0.7)
