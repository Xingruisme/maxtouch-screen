#!/usr/bin/env python3
"""
Step 4 of the audit: negative controls and timing falsification, all decoded with the payload-blind decoder at the
80 bps frame structure. Kept separate from the BER measurement (writes controls_results.json only).

A. OFF-only recordings (nothing at 80 bps was transmitted; every accepted frame is a false detection):
   A1 stent_ber_c156_20bps_20261022_160629   210 s, transmitter dead (0 % signal windows, best node +0.4 dB)
   A2 stent_onoff_precheck_..._161009         30 s, carrier commanded but not reaching the panel (clean noise)
   A3 stent_onoff_precheck_..._165856         30 s, no carrier response but +-100-180 count disturbances on Y2-Y4
   A4 the transmitter-off leads of all 24 BER runs, concatenated (~95 s)
   each with the decoder's own node choice and with Y2-Y4 forced.
B. Surrogates of the three 80 bps captures (Y2-Y4 forced, the run's own lead):
   B1 OFF bootstrap: every sample replaced by lead samples (blocks of 25) on the real timestamps, 20 per run
   B2 time reversal of the whole capture
   B3 sample permutation after t_prompt (amplitudes kept, timing destroyed), 20 per run
C. Timing falsification on the real 80 bps captures: decode with tau shifted by whole bits and by fractions of a
   bit, and with the clock correction removed; header-score landscape over one frame period.
D. Wrong payloads: decoded bits scored against all 126 other cyclic shifts of the PRBS7 and 10,000 random sequences.
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import blind_decoder as BD
import score as SC

TARGET = ["stent_ber_val4_80bps_20261022_192702", "stent_ber_val4b_80bps_20261022_195008",
          "stent_ber_val4b_80bps_20261022_195109"]
BPS = 80.0
Y234 = [2, 3, 4]
R_LEVELS = (0.3, 0.5, 0.7)
rng = np.random.default_rng(20261023)


def load_ber(run):
    d = os.path.join(DATA, run)
    m = json.load(open(os.path.join(d, "meta.json"))); z = np.load(os.path.join(d, "capture.npz"))
    return z["t"].astype(float), z["X"].astype(float), m


def load_onoff(name):
    a = np.genfromtxt(os.path.join(DATA, name, "onoff.csv"), delimiter=",", skip_header=1)
    return a[:, 1], a[:, 4:4 + 28]


def fp_stats(res):
    fr = res["frames"]
    r = np.array([f["r"] for f in fr]) if fr else np.zeros(0)
    he = np.array([f["header_errors"] for f in fr]) if fr else np.zeros(0)
    return dict(slots=len(fr), nodes=[f"Y{i}" for i in res["nodes"]], header_score=res["header_score"],
                r_max=float(r.max()) if len(r) else None, r_mean=float(r.mean()) if len(r) else None,
                **{f"accepted_r{x:g}": int((r >= x).sum()) for x in R_LEVELS},
                accepted_hdr_errors_le2=int((he <= 2).sum()), r_values=np.round(r, 3).tolist())


def off_controls():
    out = {}
    t, X, m = load_ber("stent_ber_c156_20bps_20261022_160629")
    sets = {"A1_tx_dead_210s": (t, X)}
    sets["A2_onoff_clean_30s"] = load_onoff("stent_onoff_precheck_155.692kHz_20261022_161009")
    sets["A3_onoff_disturbed_30s"] = load_onoff("stent_onoff_precheck_155.692kHz_20261022_165856")
    ts, xs, t0 = [], [], 0.0
    for d in sorted(os.listdir(DATA)):
        if d.startswith("stent_ber_") and os.path.exists(os.path.join(DATA, d, "capture.npz")):
            tt, XX, mm = load_ber(d)
            k = tt < mm["t_prompt"]
            ts.append(tt[k] - tt[k][0] + t0); xs.append(XX[k]); t0 = ts[-1][-1] + 0.008
    sets["A4_concatenated_leads"] = (np.concatenate(ts), np.concatenate(xs))
    for name, (t, X) in sets.items():
        lead = t < t[0] + 3.0
        e = dict(seconds=float(t[-1] - t[0]), samples=len(t))
        e["auto_nodes"] = fp_stats(BD.decode(t, X, BPS, lead))
        e["Y2_Y4_forced"] = fp_stats(BD.decode(t, X, BPS, lead, nodes=Y234))
        out[name] = e
        print(name, {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "r_values"})
                     for k, v in e.items()}, flush=True)
    return out


def surrogate_controls():
    out = {}
    for run in TARGET:
        t, X, m = load_ber(run)
        lead = t < m["t_prompt"]
        L = np.where(lead)[0]
        e = {"B1_off_bootstrap": [], "B3_permuted": []}
        for _ in range(20):
            idx = []
            while len(idx) < len(t):
                s = rng.integers(L[0], L[-1] - 25); idx.extend(range(s, s + 25))
            Xs = X[np.array(idx[:len(t)])]
            e["B1_off_bootstrap"].append(fp_stats(BD.decode(t, Xs, BPS, lead, nodes=Y234)))
            Xp = X.copy(); post = np.where(~lead)[0]; Xp[post] = X[rng.permutation(post)]
            e["B3_permuted"].append(fp_stats(BD.decode(t, Xp, BPS, lead, nodes=Y234)))
        tr = t[0] + (t[-1] - t[::-1])
        e["B2_time_reversed"] = fp_stats(BD.decode(tr, X[::-1], BPS, lead[::-1], nodes=Y234))
        for k in ("B1_off_bootstrap", "B3_permuted"):
            L_ = e[k]
            e[k] = dict(realisations=len(L_), slots=int(sum(x["slots"] for x in L_)),
                        **{f"accepted_r{x:g}": int(sum(y[f"accepted_r{x:g}"] for y in L_)) for x in R_LEVELS},
                        accepted_hdr_errors_le2=int(sum(y["accepted_hdr_errors_le2"] for y in L_)),
                        r_max=float(max(y["r_max"] for y in L_)),
                        header_score_max=float(max(y["header_score"] for y in L_)),
                        r_values=sum((y["r_values"] for y in L_), []))
        out[run] = e
        print(run, {k: {kk: vv for kk, vv in v.items() if kk != "r_values"} for k, v in e.items()}, flush=True)
    return out


def timing_controls(pl):
    out = {}
    for run in TARGET:
        t, X, m = load_ber(run)
        lead = t < m["t_prompt"]
        base = BD.decode(t, X, BPS, lead)
        tau, eps, Tb = base["tau"], base["eps"], base["Tb"]
        sent = int(m["tx_frames"])

        def ev(tau_, eps_, label):
            r = BD.decode(t, X, BPS, lead, nodes=base["nodes"], timing=(tau_, eps_))
            fr = r["frames"]
            # score every slot that lies in the transmission, accepted or not, against the payload
            allv = [dict(f, valid=True) for f in fr]
            s = SC.score(allv, m["t_first"], sent, Tb, eps_, pl)
            tf = [row for row in s["rows"] if 0 <= row["k"] < sent]
            errs = sum(row["errors"] for row in tf); bits = sum(row["bits"] for row in tf)
            return dict(label=label, mean_header_r=float(np.mean([f["r"] for f in fr])),
                        accepted_r03=int(sum(f["r"] >= 0.3 for f in fr)),
                        accepted_hdr_errors_le2=int(sum(f["header_errors"] <= 2 for f in fr)),
                        header_score=r["header_score"], payload_errors_in_tx_slots=errs, payload_bits=bits,
                        ber=errs / bits if bits else None)

        rows = [ev(tau, eps, "found (tau, eps)")]
        for k in (-3, -2, -1, 1, 2, 3):
            rows.append(ev(tau + k * Tb * (1 + eps), eps, f"tau {k:+d} bit"))
        for ms in (-6, -4, -3, -2, -1, 1, 2, 3, 4, 6):
            rows.append(ev(tau + ms / 1e3, eps, f"tau {ms:+d} ms"))
        rows.append(ev(tau, 0.0, "clock correction removed (eps = 0)"))
        for ppm in (-100, 100):
            rows.append(ev(tau, eps + ppm * 1e-6, f"eps {ppm:+d} ppm"))
        # landscape: header score over one frame period at the found eps
        f = base["f"]; P = BD.NB * Tb
        grid = np.arange(t[0], t[0] + P, 0.001)
        sc = np.array([BD.header_score(f, t, x, Tb, eps, BD.slots(t, x, eps, P)) for x in grid])
        i0 = int(np.argmax(sc))
        far = np.abs(((grid - grid[i0]) + P / 2) % P - P / 2) > Tb   # more than one bit away from the peak
        out[run] = dict(rows=rows, landscape=dict(step_ms=1.0, peak=float(sc[i0]), second_peak_beyond_1bit=float(sc[far].max()),
                                                  peak_ratio=float(sc[i0] / max(sc[far].max(), 1e-9)),
                                                  peak_tau_minus_found_ms=float((grid[i0] - tau) * 1e3),
                                                  tau_s=(grid - t[0]).tolist(), score=sc.tolist()))
        print(run, "landscape peak", out[run]["landscape"]["peak"], "2nd", out[run]["landscape"]["second_peak_beyond_1bit"])
        for r in rows:
            print("   ", r)
    return out


def wrong_payload(pl):
    out = {}
    for run in TARGET:
        b = json.load(open(os.path.join(HERE, "blind_out", f"{run}.v4rules_blind.json")))
        G = np.array([f["got"] for f in b["frames"] if f["valid"]])
        shifts = [int(np.sum(G != np.roll(pl, k))) for k in range(127)]
        rnd = [int(np.sum(G != rng.integers(0, 2, 127))) for _ in range(10000)]
        out[run] = dict(bits=int(G.size), errors_true=shifts[0], other_shifts_min=min(shifts[1:]), other_shifts_max=max(shifts[1:]),
                        random_min=min(rnd), random_mean=float(np.mean(rnd)))
        print(run, out[run])
    return out


def main():
    pl = SC.payload()
    out = dict(A_off_only=off_controls(), B_surrogates=surrogate_controls(), C_timing=timing_controls(pl),
               D_wrong_payload=wrong_payload(pl))
    json.dump(out, open(os.path.join(HERE, "controls_results.json"), "w"))


if __name__ == "__main__":
    main()
