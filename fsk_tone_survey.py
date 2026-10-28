#!/usr/bin/env python3
"""
Can the panel tell two transmitted tones apart by amplitude alone? Survey of the existing frequency sweeps
(stent_probe_*, nothing transmitted) for an amplitude-discriminated 2-FSK: bit 1 = 159.585 kHz, bit 0 = a second tone
the panel does see (candidate 53.513 kHz) versus the baseline bit 0 = 180.556 kHz (a panel null).

Per sweep, per node Y0..Y{nodes-1}, per tone: the per-scan amplitude |x - b| in the tone's ON blocks, b = median of the
same carrier slot's OFF blocks (the v4 'absdev' feature; the carrier's phase is random on every scan, so one scan of a
tone of rms A is |A sqrt2 cos phi + noise|, not A). Reported:
  rms per tone and OFF noise, tone-0 / tone-1 amplitude ratio, its spread over sweeps,
  the classification error between the two tones with an energy statistic (mean of |x - b|^2 over k scans, the best
  single threshold, equal priors), k = 1, 2, 3, 6, 12 scans per bit (~125, 63, 42, 21, 10 bps at ~125 scans/s),
  from resampling the recorded ON scans (independent draws: consecutive scans have uncorrelated carrier phase).
Sweeps recorded while the transmitter output was ~40 dB low (10-23 17:30 - 10-26 11:40) are left out.
The 94-carrier sweeps of 10-24 / 10-26 have no 53.513 kHz; their nearest carrier 53.844 kHz (+0.6 %) stands in, and
the neighbours 52.531 / 55.191 kHz show how flat the panel response is there.

   python fsk_tone_survey.py [--nodes 4] [--out data_maxtouch/fsk_tone_survey]
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

import maxtouch_stent_rx as R

ONE = 159585.0
ZEROS = {"180.6k (baseline)": 180556.0, "53.5k (candidate)": 53513.0}
NEIGHBOURS = (52531.0, 55191.0)
K = (1, 2, 3, 6, 12)
# transmitter output ~40 dB low (selftests 10-24 / 10-26 morning): these sweeps do not measure the panel response
TX_FAULT = (datetime.datetime(2026, 10, 23, 17, 30), datetime.datetime(2026, 10, 26, 11, 40))
GROUPS = (("body", ("stent_probe_body_", "stent_probe_frequency_sweep_new_")),
          ("phantom", ("stent_probe_phantom_sweep_",)), ("glass", ("stent_probe_glass_wide", "stent_probe_trx_fine_")))


def group(name):
    return next((g for g, pre in GROUPS if any(name.startswith(p) for p in pre)), None)


def load(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "capture.npz"))
    with contextlib.redirect_stdout(io.StringIO()):
        res, aux = R.analyse_probe(z["t"], z["X"], meta)
    return meta, res, aux


def slot(aux, meta, j):
    """ON and OFF scan indices of carrier slot j (edges dropped), as body_current_sweeps.block_rms"""
    T, tau, a = aux["T"], aux["tau"], meta["args"]
    b = int(round(a["block"] / T))
    s0 = tau + int(round((2 * 3.0 + j * 2 * a["reps"] * a["block"]) / T))
    on = np.concatenate([np.arange(s0 + 2 * k * b + 3, s0 + (2 * k + 1) * b - 1) for k in range(a["reps"])])
    off = np.concatenate([np.arange(s0 + (2 * k + 1) * b + 3, s0 + (2 * k + 2) * b - 1) for k in range(a["reps"])])
    G = aux["G"]
    return on[on < len(G)], off[off < len(G)]


def amplitudes(aux, meta, f, node):
    car = np.array(aux["car"])
    j = int(np.argmin(np.abs(car - f)))
    on, off = slot(aux, meta, j)
    x = aux["G"][:, node]
    b = np.median(x[off])
    return float(car[j]), np.abs(x[on] - b), np.abs(x[off] - b)


def k_error(a1, a0, k, rng, n=20000):
    """min over thresholds of the equal-prior error of mean(|x|^2 over k scans): tone 1 vs tone 0"""
    s1 = (rng.choice(a1, (n, k)) ** 2).mean(axis=1)
    s0 = (rng.choice(a0, (n, k)) ** 2).mean(axis=1)
    thr = np.unique(np.quantile(np.concatenate([s1, s0]), np.linspace(0, 1, 801)))
    e = [0.5 * (np.mean(s1 < t) + np.mean(s0 >= t)) for t in thr]
    i = int(np.argmin(e))
    return float(e[i]), float(np.sqrt(thr[i]))


def main():
    ap = argparse.ArgumentParser(description="amplitude survey of the 2-FSK tone pairs in the existing sweeps")
    ap.add_argument("--nodes", type=int, default=4, help="nodes Y0.. searched for the best node (finger at Y0-Y3)")
    ap.add_argument("--out", default=os.path.join(R.DATADIR, "fsk_tone_survey"))
    a = ap.parse_args()
    rng = np.random.default_rng(1)
    rows = []
    for d in sorted(glob.glob(os.path.join(R.DATADIR, "stent_probe_*"))):
        name = os.path.basename(d)
        g = group(name)
        if g is None or not os.path.exists(os.path.join(d, "capture.npz")):
            continue
        m0 = json.load(open(os.path.join(d, "meta.json")))
        when = datetime.datetime.fromtimestamp(m0.get("t_first") or m0.get("t_prompt") or 0)
        if TX_FAULT[0] <= when <= TX_FAULT[1]:
            print(f"{name:52s} skipped: recorded while the transmitter output was ~40 dB low")
            continue
        meta, res, aux = load(d)
        if aux is None:
            continue
        car = np.array(aux["car"])
        if np.min(np.abs(car - ONE)) / ONE > 0.003 and g != "glass":
            continue
        # node: the strongest bit-1 response among Y0..; a sweep without a bit-1 carrier (ON < 3x OFF) is left out
        cands = []
        for n in range(min(a.nodes, aux["G"].shape[1])):
            f1, a1, off = amplitudes(aux, meta, ONE, n)
            cands.append((np.sqrt(np.mean(a1 ** 2)), n, f1, a1, off))
        rms1, node, f1, a1, off1 = max(cands, key=lambda c: c[0])
        noise = float(np.sqrt(np.mean(off1 ** 2)))
        if abs(f1 - ONE) / ONE > 0.003 or rms1 < 3 * noise:
            continue
        for label, f0 in ZEROS.items():
            fz, a0, off0 = amplitudes(aux, meta, f0, node)
            if abs(fz - f0) / f0 > 0.01:
                continue
            rms0 = float(np.sqrt(np.mean(a0 ** 2)))
            row = dict(group=g, sweep=name, node=f"Y{meta.get('y0', 0) + node}", pair=label, f1_hz=f1, f0_hz=fz,
                       rms1=float(rms1), rms0=rms0, noise_rms=noise, ratio_db=20 * np.log10(max(rms0, 1e-9) / rms1),
                       n_scans1=len(a1), n_scans0=len(a0))
            if label.startswith("53.5"):
                row.update({f"rms_{nb / 1e3:.1f}k": float(np.sqrt(np.mean(amplitudes(aux, meta, nb, node)[1] ** 2)))
                            for nb in NEIGHBOURS})
            for k in K:
                row[f"err_k{k}"], row[f"thr_k{k}"] = k_error(a1, a0, k, rng)
            row["_a1"], row["_a0"] = a1, a0
            rows.append(row)
        print(f"{name:52s} {g:7s} Y{node}  1: {rms1:6.0f}  noise {noise:5.1f}  "
              + "  ".join(f"{r['pair'][:5]} {r['rms0']:6.0f} ({r['ratio_db']:+5.1f} dB) err k1/k3/k6 "
                          f"{r['err_k1']:.3f}/{r['err_k3']:.4f}/{r['err_k6']:.4f}" for r in rows if r["sweep"] == name))
    os.makedirs(a.out, exist_ok=True)
    keys = [k for k in rows[0] if not k.startswith("_")] + [f"rms_{nb / 1e3:.1f}k" for nb in NEIGHBOURS]
    keys = list(dict.fromkeys(keys))
    with open(os.path.join(a.out, "survey.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print("\nper group and pair (median over sweeps, [min - max]):")
    for g, _ in GROUPS:
        for label in ZEROS:
            rr = [r for r in rows if r["group"] == g and r["pair"] == label]
            if not rr:
                continue
            q = lambda key, fmt: (f"{np.median([r[key] for r in rr]):{fmt}} [{min(r[key] for r in rr):{fmt}} - "
                                  f"{max(r[key] for r in rr):{fmt}}]")
            print(f"  {g:7s} {label:18s} n={len(rr):2d}  tone0/tone1 {q('ratio_db', '+.1f')} dB   "
                  + "  ".join(f"err k{k} {q(f'err_k{k}', '.4f')}" for k in (1, 3, 6)))
    plot(rows, os.path.join(a.out, "survey"))
    print(f"\nSaved {os.path.relpath(a.out)}/ (survey.csv, survey.png/.pdf)")


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink2, grid = "#52514e", "#e4e3df"
    col = {"180.6k (baseline)": "#2a78d6", "53.5k (candidate)": "#eb6834"}
    body = [r for r in rows if r["group"] == "body"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.0))
    # a: per-scan amplitude distributions, all body sweeps pooled after normalising by the sweep's bit-1 rms
    bins = np.linspace(0, 2.2, 89)
    a1 = np.concatenate([r["_a1"] / r["rms1"] for r in body if r["pair"].startswith("180")])
    ax[0].hist(a1, bins, density=True, histtype="stepfilled", color="#1baf7a", alpha=0.35, label="bit 1: 159.6 kHz")
    for label, c in col.items():
        a0 = np.concatenate([r["_a0"] / r["rms1"] for r in body if r["pair"] == label])
        ax[0].hist(a0, bins, density=True, histtype="step", lw=2, color=c, label=f"bit 0: {label}")
    ax[0].set_yscale("log"); ax[0].set_xlabel("per-scan amplitude / sweep's 159.6 kHz rms")
    ax[0].set_ylabel("density (all body sweeps)")
    ax[0].set_title("One scan: amplitude distributions", fontsize=10, loc="left")
    # b: tone-0 / tone-1 ratio per sweep
    for label, c in col.items():
        rr = sorted([r for r in rows if r["pair"] == label], key=lambda r: (r["group"], r["sweep"]))
        x = np.arange(len(rr))
        ax[1].plot(x, [r["ratio_db"] for r in rr], "o", color=c, ms=7, label=f"bit 0 = {label}")
    ax[1].set_xticks([]); ax[1].set_xlabel("sweep (body, then glass / phantom)")
    ax[1].set_ylabel("bit-0 rms / bit-1 rms (dB)")
    ax[1].set_title("Bit-0 level relative to bit 1, per sweep", fontsize=10, loc="left")
    # c: error vs scans per bit (median and range over body sweeps)
    for label, c in col.items():
        rr = [r for r in body if r["pair"] == label]
        e = np.array([[max(r[f"err_k{k}"], 1e-5) for k in K] for r in rr])
        ax[2].fill_between(K, e.min(axis=0), e.max(axis=0), color=c, alpha=0.18, lw=0)
        ax[2].semilogy(K, np.median(e, axis=0), "-o", color=c, lw=2, ms=7, label=f"bit 0 = {label}")
    ax[2].set_xscale("log"); ax[2].set_xticks(K); ax[2].set_xticklabels([str(k) for k in K]); ax[2].minorticks_off()
    ax[2].set_ylim(1e-5, 0.5); ax[2].set_xlabel("scans per bit (125 scans/s: 125, 63, 42, 21, 10 bps)")
    ax[2].set_ylabel("bit error, tone 1 vs tone 0 (best threshold)")
    ax[2].set_title("Body sweeps: median and range (floor 1e-5)", fontsize=10, loc="left")
    for a_, letter in zip(ax, "abc"):
        a_.grid(color=grid, lw=0.6, which="both"); a_.set_axisbelow(True)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
        a_.text(-0.12, 1.04, letter, transform=a_.transAxes, fontsize=14, fontweight="bold")
        a_.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=150); fig.savefig(path + ".pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
