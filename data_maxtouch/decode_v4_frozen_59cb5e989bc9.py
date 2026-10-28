#!/usr/bin/env python3
"""
Decoder v4 for the stent -> maXTouch BER runs: decodes from the RAW samples by their own timestamps.
Writes result_v4.json / decoded_v4.json; the original decoder (result.json), v2 and v3 are not touched.

   python decode_v4.py data_maxtouch/stent_ber_<label>_<bps>bps_<time>    # one run
   python decode_v4.py all                                                # every run + original / v3 / v4 table

Why (2026-10-22, development runs only): the 34-45 bps 'spill-over' errors (a 0 after a 1 read as 1) are made by
the decoding, not by the touchscreen. Averaged over thousands of bit edges, a node's |delta| steps from 0 to 100 %
within 2-4 ms, ~11-13 ms after the transmitter edge (sensing + USB delay; no multi-scan tail, so no filtering), and
each sample is a near-instant reading of the node. The raw samples inside every v3 error bit were clean (0-8 counts
in a sent 0, thousands in a sent 1). The errors came from (a) linear interpolation of the samples onto a uniform
grid, which mixes ON and OFF values next to an edge, (b) bit windows in fractional grid samples, which let a value
of the neighbouring bit in, and (c) a heavy-tailed feature, so one leaked ON value lifts a 0 bit over threshold.

Rules (fixed before any run recorded after 2026-10-22 19:02 is decoded):
  1. Nodes: the 3 with the largest energy increase after START TX vs the lead (raw samples, before decoding).
  2. Feature per raw sample and node: absdev = log10(1 + (x - OFF baseline)^2 / OFF noise^2), the OFF baseline a
     running median (50 samples) of the quiet samples only (within 20 sigma of the lead level; sigma = lead MAD);
     z-scored; combined = per-sample max over the 3 nodes (as v3).
  3. Timing from raw timestamps: bit edge j of frame m at tau + (m * frame_bits + j) * Tb * (1 + eps). Blind: tau
     over the whole frame period (4 ms steps, then +-6 ms in 0.25 ms) and eps in +-200 ppm (10 ppm) maximising the
     summed preamble+SFD correlation of all frames (the Teensy repeats frames on its crystal). Known timing: the
     same search with tau limited to the known first-bit time + 0..25 ms.
  4. Bit value = mean feature of the samples more than GUARD_MS = 2 ms from both bit edges (the measured step
     width incl. the 1 ms USB arrival steps); a bit with no such sample uses all its samples.
  5. Decision per frame: threshold halfway between the means of the known 1 and 0 bits (header + trailing gap).
  6. Frame validity, accounting, health and error types exactly as v2/v3 (header r >= 0.3, frame BER < 0.35;
     conservative BER; signal present in >= 95 % of 2 s windows; sent1_read0 / sent0_read1 / bursts).
Run roles: 'exploratory' = runs in DEVELOPMENT_RUNS (recorded and looked at before v4 was written); every other run
is 'validation'.
"""
from __future__ import annotations

import contextlib
import csv
import glob
import hashlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode_v2 as V2                          # frames_summary, fmt, baseline_row
import decode_v3 as V3                          # error_types

R, RX, TX = V2.R, V2.RX, V2.TX
VERSION = "v4"
N_NODES = 3
GUARD_MS = 2.0
EPS_PPM = np.arange(-200, 201, 10)
DEVELOPMENT_RUNS = {                            # every BER run recorded before v4 was written (19:02)
    "stent_ber_c156_20bps_20261022_160629", "stent_ber_c156_20bps_20261022_175518",
    "stent_ber_c156_20bps_20261022_180016", "stent_ber_c156_20bps_20261022_180358",
    "stent_ber_c156_34bps_20261022_180957", "stent_ber_c156_38bps_20261022_182051",
    "stent_ber_c156_40bps_20261022_182245", "stent_ber_c156_45bps_20261022_181139",
    "stent_ber_c156_68bps_20261022_181556", "stent_ber_c160_34bps_20261022_182456",
    "stent_ber_c164~_34bps_20261022_190157",
}
FROZEN_AT = "2026-10-22 19:16"                  # rules fixed; no change after this
RULES_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]
HDR = np.array(RX.HDR, float)


def absdev(x, lead):
    """log10(1 + (x - OFF baseline)^2 / OFF noise^2). The baseline is a running median of the quiet samples only
    (within 20 sigma of the lead level): a median over all samples was pulled tens of counts off the OFF level by
    the +-2000-count swings of long 1 runs at 20 bps, and clean OFF samples then read as 1 (development run
    c156_20bps_180358: 8 errors, all sent 0 with raw values -14..+3)."""
    med = float(np.median(x[lead]))
    sig = max(1.4826 * float(np.median(np.abs(x[lead] - med))), 1.0)
    q = np.where(np.abs(x - med) < 20 * sig)[0]
    if len(q) < 10:
        b = np.full(len(x), med)
    else:
        b = np.interp(np.arange(len(x)), q, RX.baseline(x[q], 50))
    return np.log10(1 + (x - b) ** 2 / sig ** 2)


def bit_index(t, tau, Tb, eps, nb_frame):
    """(frame, bit, position 0..1 inside the bit) of each sample"""
    u = (t - tau) / (Tb * (1 + eps))
    m = np.floor(u / nb_frame).astype(int)
    jb = u - m * nb_frame
    j = np.floor(jb).astype(int)
    return m, j, jb - j


def header_score(f, t, tau, Tb, eps, nb_frame, frames):
    """sum over frames of the preamble+SFD Pearson correlation (positive = ON raises the feature)"""
    m, j, _ = bit_index(t, tau, Tb, eps, nb_frame)
    sel = (j < len(HDR)) & np.isin(m, frames)
    if not sel.any():
        return -np.inf
    idx = (np.searchsorted(frames, m[sel])) * len(HDR) + j[sel]
    n = len(frames) * len(HDR)
    s = np.bincount(idx, f[sel], n); c = np.bincount(idx, minlength=n)
    H = np.where(c > 0, s / np.maximum(c, 1), np.nan).reshape(len(frames), len(HDR))
    ok = ~np.isnan(H).any(axis=1)
    if not ok.any():
        return -np.inf
    Hc = H[ok] - H[ok].mean(axis=1, keepdims=True); tc = HDR - HDR.mean()
    den = np.linalg.norm(Hc, axis=1) * np.linalg.norm(tc)
    return float(np.sum(np.where(den > 0, Hc @ tc / np.where(den > 0, den, 1), 0)))


def search(f, t, Tb, nb_frame, tau_lo, tau_hi, frames_of):
    """best (tau, eps): coarse tau at eps = 0, then fine tau x eps"""
    best = (-np.inf, tau_lo, 0.0)
    for tau in np.arange(tau_lo, tau_hi, 0.004):
        s = header_score(f, t, tau, Tb, 0.0, nb_frame, frames_of(tau, 0.0))
        if s > best[0]:
            best = (s, tau, 0.0)
    tau_c = best[1]
    for eps in EPS_PPM * 1e-6:
        for tau in np.arange(tau_c - 0.006, tau_c + 0.006 + 1e-9, 0.00025):
            s = header_score(f, t, tau, Tb, eps, nb_frame, frames_of(tau, eps))
            if s > best[0]:
                best = (s, tau, eps)
    return best[1], best[2]


def decode_frames(f, t, tau, Tb, eps, frame_bits, truth, frames, known_timing):
    nb = len(frame_bits)
    fb = np.array(frame_bits)
    known_mask = np.zeros(nb, bool); known_mask[:len(HDR)] = True; known_mask[len(HDR) + len(truth):] = True
    pl = slice(len(HDR), len(HDR) + len(truth))
    m, j, pos = bit_index(t, tau, Tb, eps, nb)
    g = GUARD_MS / 1e3 / (Tb * (1 + eps))
    inner = (pos > g) & (pos < 1 - g)
    out = []
    for mm in frames:
        sel = m == mm
        if not sel.any():
            continue
        jj, ff, inn = j[sel], f[sel], inner[sel]
        v = np.full(nb, np.nan)
        for b in range(nb):
            w = jj == b
            if w.any():
                wi = w & inn
                v[b] = ff[wi].mean() if wi.any() else ff[w].mean()
        if np.isnan(v[:len(HDR)]).any() or np.isnan(v[pl]).any():
            continue
        kn = known_mask & ~np.isnan(v)
        hi, lo = v[kn & (fb == 1)].mean(), v[kn & (fb == 0)].mean()
        thr = (hi + lo) / 2
        got = (v[pl] > thr).astype(int)
        err = int(np.sum(got != np.array(truth)))
        hc = v[:len(HDR)] - v[:len(HDR)].mean(); tc = HDR - HDR.mean()
        r = float(hc @ tc / (np.linalg.norm(hc) * np.linalg.norm(tc) + 1e-12))
        ber = err / len(truth)
        out.append(dict(m=int(mm), s0=float(tau + mm * nb * Tb * (1 + eps)), r=r, bits=len(truth), errors=err, ber=ber,
                        thr=float(thr), got=got.tolist(), norm=((v[pl] - thr) / max(hi - lo, 1e-9)).tolist(),
                        valid=bool(known_timing or (r >= RX.R_LOCAL and ber < RX.BER_INVALID))))
    return out


def health_raw(y, t, meta, total_s):
    e = np.concatenate([[0.0], np.diff(y) ** 2])
    lead = e[t < meta["t_prompt"]].mean() + 1e-9
    w0, ok, n = meta["t_first"], 0, 0
    while w0 + V2.HEALTH_WIN_S <= meta["t_first"] + total_s:
        w = (t >= w0) & (t < w0 + V2.HEALTH_WIN_S)
        if w.any():
            n += 1
            ok += 10 * np.log10(e[w].mean() / lead + 1e-12) >= V2.HEALTH_DB
        w0 += V2.HEALTH_WIN_S
    frac = ok / n if n else 0.0
    return dict(windows=n, signal_fraction=frac, signal_lost=bool(frac < V2.HEALTH_MIN))


def decode_run(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    if meta.get("mode") != "ber":
        return None
    z = np.load(os.path.join(d, "capture.npz"))
    t, X = z["t"].astype(float), z["X"].astype(float)
    a = R.argparse.Namespace(**meta["args"])
    link = R.link_from(a, False)
    _, plan = TX.build(link)
    frame_bits, truth = TX.frame_bits(link.payload, link.prbs)
    Tb, nb = 1.0 / plan["bps"], len(frame_bits)
    P = nb * Tb
    names = R.node_names(meta)
    n_sent = meta.get("tx_frames", 0) or 0
    pre, post = t < meta["t_prompt"], t >= meta["t_prompt"]
    e = np.diff(X, axis=0) ** 2
    score = 10 * np.log10((e[post[1:]].mean(axis=0) + 1e-9) / (e[pre[1:]].mean(axis=0) + 1e-9))
    top = [int(i) for i in np.argsort(-score)[:N_NODES]]
    zf = {i: R.zscore(absdev(X[:, i], pre)) for i in top}
    comb = np.max([zf[i] for i in top], axis=0)

    def all_frames(tau, eps):                    # frames whose header lies inside the recording
        last = int((t[-1] - tau) / (P * (1 + eps)))
        return np.arange(max(0, int(np.ceil((t[0] - tau) / (P * (1 + eps))))), last + 1)

    def blind(f):
        tau, eps = search(f, t, Tb, nb, t[0], t[0] + P, all_frames)
        fr = decode_frames(f, t, tau, Tb, eps, frame_bits, truth, all_frames(tau, eps), False)
        return V2.frames_summary(fr, n_sent, len(truth)), fr, tau, eps

    single = []
    for i in top:
        s, fr, _, _ = blind(zf[i])
        single.append(dict(node=names[i], score_db=float(score[i]), **s, errors_by_type=V3.error_types(fr, truth)))
    cs, cfr, tau_b, eps_b = blind(comb)
    run = os.path.basename(os.path.normpath(d))
    res = dict(decoder=VERSION, frozen_at=FROZEN_AT, rules_sha=RULES_SHA,
               role="exploratory" if run in DEVELOPMENT_RUNS else "validation",
               rules=f"raw-timestamp decoding, top {N_NODES} nodes, absdev, max of z, guard {GUARD_MS:g} ms, "
                     "header-searched offset and clock drift",
               run=run, bps=plan["bps"], carrier_hz=a.carrier,
               samples_per_bit=float(Tb * (len(t) - 1) / (t[-1] - t[0])), nodes=[names[i] for i in top],
               single_top_ranked=single[0],
               single_best_of_3_after_decoding=min(single, key=lambda s: (s["ber"] is None, s["ber"] or 0)),
               single_all=single,
               combined_blind=dict(cs, errors_by_type=V3.error_types(cfr, truth), clock_ppm=eps_b * 1e6))
    kfr = []
    if meta.get("t_first") and n_sent:
        tau_k, eps_k = search(comb, t, Tb, nb, meta["t_first"], meta["t_first"] + 0.025,
                              lambda tau, eps: np.arange(n_sent))
        kfr = decode_frames(comb, t, tau_k, Tb, eps_k, frame_bits, truth, np.arange(n_sent), True)
        ks = V2.frames_summary(kfr, n_sent, len(truth))
        res["combined_known_timing"] = dict(ks, delay_ms=(tau_k - meta["t_first"]) * 1e3, clock_ppm=eps_k * 1e6,
                                            errors_by_type=V3.error_types(kfr, truth))
        res["health"] = health_raw(X[:, top[0]], t, meta, n_sent * plan["frame_s"])
    json.dump(res, open(os.path.join(d, "result_v4.json"), "w"), indent=1)
    keep = ("m", "s0", "r", "valid", "bits", "errors", "ber", "thr", "got", "norm")
    pack = lambda frs: [{kk: v for kk, v in fr.items() if kk in keep} for fr in frs]
    json.dump(dict(decoder=VERSION, rules_sha=RULES_SHA, sent_payload=[int(b) for b in truth], combined_nodes=res["nodes"],
                   blind_frames=pack(cfr), known_timing_frames=pack(kfr)), open(os.path.join(d, "decoded_v4.json"), "w"))
    return res


def report(res):
    h = res.get("health", {})
    print(f"\n[v4 {res['role'].upper()}] {res['run']}: {res['bps']:g} bps, {res['carrier_hz'] or 0:g} Hz, "
          f"{res['samples_per_bit']:.1f} samples/bit, nodes {', '.join(res['nodes'])}"
          + (f", signal present {h['signal_fraction']:.0%}" if h else "")
          + ("   ** SIGNAL LOST - not a link measurement **" if h.get("signal_lost") else ""))
    print(f"  single, top-ranked {res['single_top_ranked']['node']}:  {V2.fmt(res['single_top_ranked'])}")
    b = res["single_best_of_3_after_decoding"]
    print(f"  single, best of 3 (after decoding, optimistic) {b['node']}:  {V2.fmt(b)}")
    c = res["combined_blind"]
    print(f"  combined blind:        {V2.fmt(c)}   clock {c['clock_ppm']:+.0f} ppm   errors {c['errors_by_type']}")
    kt = res.get("combined_known_timing")
    if kt:
        print(f"  combined known timing: {V2.fmt(kt)}   delay {kt['delay_ms']:.2f} ms, clock {kt['clock_ppm']:+.0f} ppm")


def compare_all(only=None):
    rows = []
    for d in sorted(glob.glob(os.path.join(HERE, "data_maxtouch", "stent_ber_*"))):
        run = os.path.basename(d)
        if only is not None and run not in only:
            continue
        if not os.path.exists(os.path.join(d, "capture.npz")):
            continue
        res = decode_run(d)
        if res is None:
            continue
        report(res)
        b = V2.baseline_row(d)
        p3 = os.path.join(d, "result_v3.json")       # v3 as saved by the frozen decode_v3.py, not recomputed
        v3 = json.load(open(p3)) if os.path.exists(p3) else None
        h = res.get("health") or {}
        rows.append(dict(run=run, role=res["role"], carrier_hz=res["carrier_hz"], bps=res["bps"],
                         samples_per_bit=round(res["samples_per_bit"], 2), signal_present=h.get("signal_fraction"),
                         signal_lost=h.get("signal_lost"), **b,
                         v3_comb_blind=V2.fmt(v3["combined_blind"]) if v3 else "not run",
                         v3_comb_blind_ber=v3["combined_blind"]["ber"] if v3 else None,
                         v4_single_top=V2.fmt(res["single_top_ranked"]), v4_single_top_ber=res["single_top_ranked"]["ber"],
                         v4_comb_blind=V2.fmt(res["combined_blind"]), v4_comb_blind_ber=res["combined_blind"]["ber"],
                         v4_comb_blind_conservative=res["combined_blind"]["ber_conservative"],
                         v4_comb_known=V2.fmt(res.get("combined_known_timing")),
                         v4_sent1_read0=res["combined_blind"]["errors_by_type"]["sent1_read0"],
                         v4_sent0_read1=res["combined_blind"]["errors_by_type"]["sent0_read1"],
                         rules_sha=res["rules_sha"]))
    out = os.path.join(HERE, "data_maxtouch", "ber_compare_v4.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nv4 rules sha {RULES_SHA}, frozen {FROZEN_AT}")
    print(f"{'run':40s} {'role':>11} {'sig':>5} | {'ORIGINAL comb':>13} | {'v3 comb blind':>24} | {'v4 top node':>24} {'v4 comb blind':>24}")
    for r in rows:
        sig = "LOST" if r["signal_lost"] else f"{r['signal_present']:.0%}" if r["signal_present"] is not None else "-"
        print(f"{r['run'][10:50]:40s} {r['role']:>11} {sig:>5} | {r.get('base_comb', '-'):>13} | {r['v3_comb_blind']:>24} | "
              f"{r['v4_single_top']:>24} {r['v4_comb_blind']:>24}")
    print(f"\ntable: {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "all":
        compare_all()
    elif sys.argv[1] == "dev":                  # development runs only (before freezing)
        compare_all(DEVELOPMENT_RUNS)
    else:
        report(decode_run(sys.argv[1]))
