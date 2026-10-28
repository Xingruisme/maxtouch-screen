#!/usr/bin/env python3
"""
FSK-A decoder (decode_fska.py, rules sha in the output) against frozen v4 on every existing OOK / FSK recording, from
the same raw captures, plus an ablation of where the difference comes from. Neither decoder is changed here.

Per run: known-timing and blind errors / bits, blind frame recovery, 0->1 and 1->0 errors (erased bits apart), for v4
and for the FSK-A decoder, and the run's role:
  fa_dev       used to develop the FSK-A decoder (ankle2finger 10-26 and 10-27 13:11-13:51 runs)
  v4_dev       v4's own development runs (decode_v4.DEVELOPMENT_RUNS)
  existing     every other existing run: not used to tune either decoder, but analysed after the fact, so exploratory
               evidence, not independent validation (that needs recordings made after both decoders are frozen)
  degraded     flag: transmitter ~40 dB low (10-23 17:30 - 10-26 11:40) or the carrier dropping out (10-27 14:45 session)

Ablation (known timing; every factor swapped on its own, nodes are the same top 3 for both decoders):
  sync        v4's timing (its delay / clock search on its feature)  |  FSK-A's (pilot-correlation grid on v)
  feature     v4: per-node log10(1 + ((x - quiet-sample baseline)/lead sigma)^2), z-scored, max over the 3 nodes
              FSK-A: RMS over the 3 nodes of (x - 0.5 s running median) / node rms
  calibration per frame from known bits: 'mid' = v4's threshold halfway between the known-1 and known-0 bit means
              (header + gap); 'scale' = FSK-A's u = feature / median of the frame's pilot-1 scans ('affine' = (f - pilot-0
              median) / (pilot-1 - pilot-0 median), for v4's log feature, which is not zero for 'off'); 'none' = one
              run-wide scale
  decision    'mean' = mean feature of the bit's guarded scans vs the threshold (v4)  |  'llr' = summed per-scan log
              p1/p0 with the run's pooled pilot densities (FSK-A)
Bit accounting is the FSK-A decoder's for every ablation cell (a bit with no guarded scan uses all its scans, none ->
erasure counted as an error; every frame scored), so the cells differ only in the factor swapped. Two cells are checked
against the real decoders: v4 timing + v4 feature + mid + mean with v4's frame skipping reproduces v4's known-timing
errors, and FSK-A timing + feature + scale + llr reproduces decode_fska.

   python decoder_ablation.py [--out data_maxtouch/decoder_comparison]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import glob
import io
import json
import os

import numpy as np

import decode_fska as FA
import decode_v4 as V4

R = V4.R
FRAME, TRUTH, NP = FA.FRAME, FA.TRUTH, FA.NP
NB = len(FRAME)
KNOWN = np.zeros(NB, bool); KNOWN[:NP] = True; KNOWN[NP + len(TRUTH):] = True     # v4: header + trailing gap
TX_FAULT = (datetime.datetime(2026, 10, 23, 17, 30), datetime.datetime(2026, 10, 26, 11, 40))
DROPOUT_SESSIONS = ("stent_mod_ankle2finger_fska_still_",)
FA_DEV = ("stent_mod_ankle2finger_1mA_", "stent_mod_ankle2finger_still_6rates_")
CELLS = [  # (sync, feature, calibration, decision)
    ("v4", "v4", "mid", "mean"), ("fa", "v4", "mid", "mean"), ("v4", "fa", "mid", "mean"), ("fa", "fa", "mid", "mean"),
    ("v4", "v4", "affine", "llr"), ("fa", "v4", "affine", "llr"), ("v4", "fa", "scale", "llr"), ("fa", "fa", "scale", "llr"),
    ("fa", "fa", "none", "llr"), ("fa", "fa", "affine", "llr"), ("fa", "v4", "none", "llr"),
]


def runs():
    out = []
    for d in sorted(glob.glob(os.path.join(R.DATADIR, "stent_*"))):
        mf, cf = os.path.join(d, "meta.json"), os.path.join(d, "capture.npz")
        if not (os.path.exists(mf) and os.path.exists(cf)):
            continue
        m = json.load(open(mf))
        if m.get("mode") != "ber" or m.get("modulation", "ook") not in ("ook", "fsk"):
            continue
        out.append((d, m))
    return out


def role(name, meta):
    r = ("fa_dev" if name.startswith(FA_DEV) else "v4_dev" if name in V4.DEVELOPMENT_RUNS else "existing")
    when = datetime.datetime.fromtimestamp(meta.get("t_first") or meta.get("t_prompt") or 0)
    degraded = TX_FAULT[0] <= when <= TX_FAULT[1] or name.startswith(DROPOUT_SESSIONS)
    return r, degraded


# ---------------------------------------------------------------- ablation pieces
def feat_v4(X, lead, cols):
    return np.max([R.zscore(V4.absdev(X[:, i], lead)) for i in cols], axis=0)


def bit_values(f, t, starts, tb):
    """per frame x bit: mean feature (guarded, else whole bit, else nan) and the scan ranges"""
    lo, hi = FA.bit_ranges(t, starts, np.arange(NB), tb)
    c = np.concatenate([[0.0], np.cumsum(f)])
    n = hi - lo
    return np.where(n > 0, (c[hi] - c[lo]) / np.maximum(n, 1), np.nan), lo, hi


def decide(f, t, starts, tb, cal, dec):
    """got (frames x bits) and erased mask, with the given calibration and decision"""
    v, lo, hi = bit_values(f, t, starts, tb)
    erased = hi <= lo
    if dec == "mean":
        got = np.zeros_like(erased, int)
        for k in range(len(starts)):
            kn = KNOWN & ~np.isnan(v[k])
            h1, h0 = v[k, kn & (FRAME == 1)].mean(), v[k, kn & (FRAME == 0)].mean()
            if cal == "mid":
                got[k] = (np.nan_to_num(v[k], nan=-np.inf) > (h1 + h0) / 2).astype(int)
        return got, erased
    # llr: per-scan normalisation, then pooled pilot densities (decode_fska.kde) and summed log-ratios
    u = np.full(len(f), np.nan)
    pil1 = lambda k: np.concatenate([np.arange(lo[k, j], hi[k, j]) for j in range(NP) if FRAME[j] == 1] or [np.array([], int)])
    pil0 = lambda k: np.concatenate([np.arange(lo[k, j], hi[k, j]) for j in range(NP) if FRAME[j] == 0] or [np.array([], int)])
    ok = np.zeros(len(starts), bool)
    run_scale = np.median(np.concatenate([f[pil1(k)] for k in range(len(starts))] or [np.array([1.0])]))
    for k in range(len(starts)):
        a, b = lo[k, 0], hi[k, NB - 1]
        p1, p0 = pil1(k), pil0(k)
        if not len(p1):
            continue
        if cal == "scale":
            s = np.median(f[p1])
            if s > 0:
                u[a:b] = f[a:b] / s; ok[k] = True
        elif cal == "affine":
            m1, m0 = np.median(f[p1]), np.median(f[p0]) if len(p0) else 0.0
            if m1 > m0:
                u[a:b] = (f[a:b] - m0) / (m1 - m0); ok[k] = True
        else:
            u[a:b] = f[a:b] / run_scale; ok[k] = True
    pil = {s: np.concatenate([(pil1 if s else pil0)(k) for k in range(len(starts)) if ok[k]] or [np.array([], int)])
           for s in (0, 1)}
    lo_g = min(np.nanmin(u[pil[0]]) if len(pil[0]) else 0, np.nanmin(u[pil[1]]) if len(pil[1]) else 0, 0) - 0.5
    grid = np.linspace(lo_g, 4, 1001)
    llr_grid = np.log(FA.kde(u[pil[1]], grid)) - np.log(FA.kde(u[pil[0]], grid))
    llr = np.interp(np.nan_to_num(u, nan=0.0), grid, llr_grid)
    cl = np.concatenate([[0.0], np.cumsum(llr)])
    got = ((cl[hi] - cl[lo]) > 0).astype(int)
    got[~ok] = 1 - np.tile(FRAME, (int((~ok).sum()), 1))      # a frame that could not be calibrated: all wrong
    return got, erased


def count(got, erased, frames=None):
    pl = slice(NP, NP + len(TRUTH))
    rows = range(len(got)) if frames is None else frames
    g = np.array([got[k, pl] for k in rows]); z = np.array([erased[k, pl] for k in rows])
    if not len(g):
        return dict(errors=0, bits=0, sent0_read1=0, sent1_read0=0, erasures=0)
    e01 = int(((g == 1) & (TRUTH == 0) & ~z).sum()); e10 = int(((g == 0) & (TRUTH == 1) & ~z).sum())
    return dict(errors=e01 + e10 + int(z.sum()), bits=int(g.size), sent0_read1=e01, sent1_read0=e10, erasures=int(z.sum()))


def frame_starts(tau, tb, n, t):
    s = tau + np.arange(n) * NB * tb
    return s[s + NB * tb <= t[-1]]


# ---------------------------------------------------------------- per run
def analyse(d, meta):
    name = os.path.basename(d)
    z = np.load(os.path.join(d, "capture.npz"))
    t, X = z["t"].astype(float), z["X"].astype(float)
    with contextlib.redirect_stdout(io.StringIO()):
        v4 = V4.decode_run(d)
    fa, (fa_starts, fa_tb) = FA.decode(z["X"], t, meta) if meta.get("t_first") else (None, (np.array([]), None))
    if fa is None:                              # no transmitter start time: blind only
        cols = FA.nodes(X, t, meta["t_prompt"])
        v = FA.feature(X, t, meta["t_prompt"], cols)
        Tb = meta["plan"]["bit_us"] / 1e6
        fa = dict(nodes=cols, known=None,
                  blind=FA.blind(v, t, Tb, meta.get("tx_frames") or 0, float(np.median(v[t < meta["t_prompt"]]))))
    rl, degraded = role(name, meta)
    cb, kt = v4["combined_blind"], v4.get("combined_known_timing")
    n_sent = meta.get("tx_frames") or cb["frames_sent"]
    row = dict(run=name, role=rl, degraded=degraded, modulation=meta.get("modulation", "ook"),
               bps=round(meta["plan"]["bps"], 3) if meta.get("plan") else None, frames_sent=n_sent,
               same_nodes=sorted(fa["nodes"]) == sorted(R.node_names(meta).index(n) for n in v4["nodes"]),
               v4_known_errors=kt["errors"] if kt else None, v4_known_bits=kt["bits"] if kt else None,
               v4_known_0to1=kt["errors_by_type"]["sent0_read1"] if kt else None,
               v4_known_1to0=kt["errors_by_type"]["sent1_read0"] if kt else None,
               v4_blind_errors=cb["errors"], v4_blind_bits=cb["bits"], v4_frames=cb["frames_valid"],
               v4_blind_0to1=cb["errors_by_type"]["sent0_read1"], v4_blind_1to0=cb["errors_by_type"]["sent1_read0"])
    fk, fb = fa["known"], fa["blind"]
    row.update(fa_known_errors=fk["errors"] if fk else None, fa_known_bits=fk["bits"] if fk else None,
               fa_known_0to1=fk["sent0_read1"] if fk else None, fa_known_1to0=fk["sent1_read0"] if fk else None,
               fa_known_erased=fk["erasures"] if fk else None,
               fa_blind_errors=fb["errors"], fa_blind_bits=fb["bits"], fa_frames=fb["frames_valid"],
               fa_blind_0to1=fb["sent0_read1"], fa_blind_1to0=fb["sent1_read0"], fa_blind_erased=fb["erasures"])
    pb = len(TRUTH)
    for p in ("v4", "fa"):                      # conservative blind BER: a missing frame counts as half wrong
        e, n, fr = row[f"{p}_blind_errors"], row[f"{p}_blind_bits"], row[f"{p}_frames"]
        miss = max(n_sent - fr, 0) * pb
        row[f"{p}_blind_ber_cons"] = (e + 0.5 * miss) / (n + miss) if n + miss else None
    abl = []
    if kt and fk:
        lead = t < meta["t_prompt"]
        Tb = meta["plan"]["bit_us"] / 1e6
        feats = dict(v4=feat_v4(X, lead, fa["nodes"]), fa=FA.feature(X, t, meta["t_prompt"], fa["nodes"]))
        eps4 = kt["clock_ppm"] / 1e6
        sync = dict(v4=(frame_starts(meta["t_first"] + kt["delay_ms"] / 1e3, Tb * (1 + eps4), n_sent, t), Tb * (1 + eps4)),
                    fa=(fa_starts, fa_tb))
        for s, f, c, dcs in CELLS:
            st, tb = sync[s]
            got, er = decide(feats[f], t, st, tb, c, dcs)
            abl.append(dict(run=name, role=rl, degraded=degraded, sync=s, feature=f, calibration=c, decision=dcs,
                            **count(got, er)))
        # checks against the real decoders
        st, tb = sync["v4"]
        got, er = decide(feats["v4"], t, st, tb, "mid", "mean")
        keep = [k for k in range(len(st)) if not er[k, :NP + len(TRUTH)].any()]   # v4 skips frames with an empty bit
        row["check_v4_reimpl"] = count(got, er, keep)["errors"] == kt["errors"]
        row["check_fa_reimpl"] = abl[7]["errors"] == fk["errors"]
    return row, abl


# ---------------------------------------------------------------- report
def summarise(rows, abl, out):
    os.makedirs(out, exist_ok=True)
    for fn, data in (("per_run.csv", rows), ("ablation.csv", abl)):
        with open(os.path.join(out, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
            w.writeheader(); w.writerows(data)
    groups = [("all", lambda r: True), ("fa_dev", lambda r: r["role"] == "fa_dev"),
              ("existing, clean", lambda r: r["role"] != "fa_dev" and not r["degraded"]),
              ("  of which v4_dev", lambda r: r["role"] == "v4_dev" and not r["degraded"]),
              ("existing, degraded", lambda r: r["role"] != "fa_dev" and r["degraded"])]
    S = lambda rr, k: sum(r[k] or 0 for r in rr)
    lines = [f"FSK-A decoder {FA.RULES_SHA} vs v4 {V4.RULES_SHA}; {len(rows)} runs",
             f"checks: v4 re-implementation reproduces v4 known timing in "
             f"{sum(1 for r in rows if r.get('check_v4_reimpl'))}/{sum(1 for r in rows if 'check_v4_reimpl' in r)} runs, "
             f"FSK-A re-implementation in {sum(1 for r in rows if r.get('check_fa_reimpl'))}/"
             f"{sum(1 for r in rows if 'check_fa_reimpl' in r)}; same 3 nodes in "
             f"{sum(1 for r in rows if r['same_nodes'])}/{len(rows)}", "",
             f"{'group':20s} {'runs':>4}  {'known v4':>17} {'known FSK-A':>17}  {'blind v4':>17} {'frames':>8}  "
             f"{'blind FSK-A':>17} {'frames':>8}"]
    for g, sel in groups:
        rr = [r for r in rows if sel(r)]
        if not rr:
            continue
        kk = [r for r in rr if r["v4_known_bits"] and r["fa_known_bits"]]
        fmt = lambda e, n: f"{e}/{n} {100 * e / n:5.2f}%" if n else "-"
        lines.append(f"{g:20s} {len(rr):4d}  {fmt(S(kk, 'v4_known_errors'), S(kk, 'v4_known_bits')):>17} "
                     f"{fmt(S(kk, 'fa_known_errors'), S(kk, 'fa_known_bits')):>17}  "
                     f"{fmt(S(rr, 'v4_blind_errors'), S(rr, 'v4_blind_bits')):>17} {S(rr, 'v4_frames'):>3}/{S(rr, 'frames_sent'):<4} "
                     f"{fmt(S(rr, 'fa_blind_errors'), S(rr, 'fa_blind_bits')):>17} {S(rr, 'fa_frames'):>3}/{S(rr, 'frames_sent'):<4}")
    lines += ["", "error types, clean existing runs (known timing): v4 0->1 / 1->0 = "
              f"{S([r for r in rows if r['role'] != 'fa_dev' and not r['degraded']], 'v4_known_0to1')} / "
              f"{S([r for r in rows if r['role'] != 'fa_dev' and not r['degraded']], 'v4_known_1to0')}, FSK-A = "
              f"{S([r for r in rows if r['role'] != 'fa_dev' and not r['degraded']], 'fa_known_0to1')} / "
              f"{S([r for r in rows if r['role'] != 'fa_dev' and not r['degraded']], 'fa_known_1to0')} "
              f"(+ {S([r for r in rows if r['role'] != 'fa_dev' and not r['degraded']], 'fa_known_erased')} erased)"]
    # worse runs: by the conservative blind BER (realistic receiver) or by known timing, beyond counting noise
    lines += ["", "runs where FSK-A is worse (difference > 2 sqrt(sum) and >= 5 errors), blind conservative or known:"]
    for r in rows:
        why = []
        for p, e4, n4, ef, nf in (("known", "v4_known_errors", "v4_known_bits", "fa_known_errors", "fa_known_bits"),):
            if r[n4] and r[nf]:
                a, b = r[e4] / r[n4] * r[nf], r[ef]           # v4 errors scaled to FSK-A's bit count
                if b - a >= 5 and b - a > 2 * np.sqrt(a + b):
                    why.append(f"known {r[e4]}/{r[n4]} -> {r[ef]}/{r[nf]}")
        c4, cf = r["v4_blind_ber_cons"], r["fa_blind_ber_cons"]
        nb = r["frames_sent"] * len(TRUTH)
        if c4 is not None and cf is not None and (cf - c4) * nb >= 5 and (cf - c4) * nb > 2 * np.sqrt((c4 + cf) * nb):
            why.append(f"blind {r['v4_blind_errors']}/{r['v4_blind_bits']} ({r['v4_frames']} fr) -> "
                       f"{r['fa_blind_errors']}/{r['fa_blind_bits']} ({r['fa_frames']} fr)")
        if why:
            lines.append(f"  {r['run']:62s} [{r['role']}{', degraded' if r['degraded'] else ''}] " + "; ".join(why))
    lines += ["", "ablation, known timing, total errors / bits (FSK-A bit accounting) by run group:",
              f"{'sync':5s} {'feature':7s} {'calib':7s} {'decide':6s}  {'fa_dev':>16} {'existing clean':>16} {'degraded':>16}"]
    for s, f, c, dcs in CELLS:
        cells = []
        for sel in (lambda a: a["role"] == "fa_dev", lambda a: a["role"] != "fa_dev" and not a["degraded"],
                    lambda a: a["role"] != "fa_dev" and a["degraded"]):
            aa = [a for a in abl if (a["sync"], a["feature"], a["calibration"], a["decision"]) == (s, f, c, dcs) and sel(a)]
            e, n = S(aa, "errors"), S(aa, "bits")
            cells.append(f"{e}/{n} {100 * e / n:5.2f}%" if n else "-")
        lines.append(f"{s:5s} {f:7s} {c:7s} {dcs:6s}  " + " ".join(f"{x:>16}" for x in cells))
    text = "\n".join(lines)
    open(os.path.join(out, "summary.txt"), "w").write(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description="FSK-A decoder vs frozen v4 on all existing OOK/FSK runs + ablation")
    ap.add_argument("--out", default=os.path.join(R.DATADIR, "decoder_comparison"))
    a = ap.parse_args()
    rows, abl = [], []
    todo = runs()
    for i, (d, meta) in enumerate(todo):
        print(f"[{i + 1}/{len(todo)}] {os.path.basename(d)}", flush=True)
        r, ab = analyse(d, meta)
        rows.append(r); abl += ab
    summarise(rows, abl, a.out)


if __name__ == "__main__":
    main()
