#!/usr/bin/env python3
"""
Step 5 of the audit: how raw samples become bits, measured on the payload-blind decoding of the saved captures.
After decoding (blind_out/*.v4rules_blind.json, timing tau/eps from the blind search) and scoring (score.py), per
payload bit of every true frame: number of samples and of guard-passing ('inner') samples, their positions in the
bit, the decision value, the node that supplied the max, the margin, the previous bit. Writes mechanism_results.json
and bits_table.npz (one row per payload bit, used by figures.py).
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import blind_decoder as BD
import score as SC

RUNS = {"80a": "stent_ber_val4_80bps_20261022_192702", "80b": "stent_ber_val4b_80bps_20261022_195008",
        "80c": "stent_ber_val4b_80bps_20261022_195109", "68": "stent_ber_val4_68bps_20261022_192559",
        "90a": "stent_ber_val4b_90bps_20261022_195206", "90b": "stent_ber_val4b_90bps_20261022_195302",
        "100": "stent_ber_val4_100bps_20261022_192753"}


def per_bit(run, pl):
    d = os.path.join(DATA, run)
    meta = json.load(open(os.path.join(d, "meta.json"))); z = np.load(os.path.join(d, "capture.npz"))
    t, X = z["t"].astype(float), z["X"].astype(float)
    bps = float(meta["args"]["bps"]); lead = t < meta["t_prompt"]
    res = BD.decode(t, X, bps, lead)                       # identical to blind_out (deterministic)
    b = json.load(open(os.path.join(HERE, "blind_out", f"{run}.v4rules_blind.json")))
    assert abs(res["tau"] - b["tau"]) < 1e-12 and [f["got"] for f in res["frames"]] == [f["got"] for f in b["frames"]]
    Tb, eps, tau = res["Tb"], res["eps"], res["tau"]
    s = SC.score(b["frames"], meta["t_first"], int(meta["tx_frames"]), Tb, eps, pl)
    true = {r["m"]: r["k"] for r in s["rows"] if r["kind"] == "true"}
    m, j, pos = BD.bit_index(t, tau, Tb, eps)
    g = 2.0 / 1e3 / (Tb * (1 + eps)); inner = (pos > g) & (pos < 1 - g)
    Z = np.array([res["z"][i] for i in res["nodes"]])          # nodes x samples
    full = np.concatenate([BD.HDR, pl, np.zeros(16)])
    rows = []
    for f in res["frames"]:
        if f["m"] not in true:
            continue
        for bb in range(127):
            jb = 27 + bb
            w = np.where((m == f["m"]) & (j == jb))[0]
            wi = w[inner[w]]
            use = wi if len(wi) else w
            src = Z[:, use].argmax(axis=0)                     # which node supplied the max, per used sample
            rows.append(dict(frame_k=true[f["m"]], m=f["m"], bit=bb, sent=int(pl[bb]), prev=int(full[jb - 1]),
                             got=f["got"][bb], margin=(f["norm"][bb] if pl[bb] else -f["norm"][bb]),
                             v=f["v"][jb], thr=f["thr"], n=len(w), n_inner=len(wi),
                             pos_ms=[float(p * Tb * 1e3) for p in pos[w]],
                             used_pos_ms=[float(p * Tb * 1e3) for p in pos[use]],
                             node_of_max=[res["nodes"][k] for k in src]))
    dt = np.diff(t)
    return rows, dict(run=run, bps=bps, nodes=[f"Y{i}" for i in res["nodes"]], tau=tau, eps_ppm=eps * 1e6,
                      delay_ms=s["residual_ms"], dt_median_ms=float(np.median(dt) * 1e3),
                      dt_p5_p95_ms=[float(np.percentile(dt, 5) * 1e3), float(np.percentile(dt, 95) * 1e3)],
                      dt_within_0p2ms_of_8ms=float(np.mean(np.abs(dt - 0.008) < 0.0002)),
                      gaps_over_20ms=[float(x * 1e3) for x in dt[dt > 0.02]],
                      gap_times_s=[float(t[i] - meta["t_first"]) for i in np.where(dt > 0.02)[0]],
                      samples_per_bit=float(Tb * (len(t) - 1) / (t[-1] - t[0])),
                      drift_over_tx_ms=float(eps * int(meta["tx_frames"]) * 170 * Tb * 1e3))


def summarise(rows):
    n = np.array([r["n"] for r in rows]); ni = np.array([r["n_inner"] for r in rows])
    mg = np.array([r["margin"] for r in rows]); sent = np.array([r["sent"] for r in rows])
    prev = np.array([r["prev"] for r in rows]); v = np.array([r["v"] for r in rows])
    used1 = np.array([len(r["used_pos_ms"]) == 1 for r in rows])
    nodes = np.array([r["node_of_max"][0] for r in rows])
    out = dict(bits=len(rows), errors=int(np.sum(mg < 0)),
               samples_per_bit_hist={str(k): int(np.sum(n == k)) for k in range(0, 4)},
               inner_samples_hist={str(k): int(np.sum(ni == k)) for k in range(0, 3)},
               fallback_bits=int(np.sum(ni == 0)), decided_by_one_sample=int(np.sum(used1)),
               v_sent1_min=float(v[sent == 1].min()), v_sent1_median=float(np.median(v[sent == 1])),
               v_sent0_max=float(v[sent == 0].max()), v_sent0_median=float(np.median(v[sent == 0])),
               margin_median=float(np.median(mg)),
               margin_below={str(x): int(np.sum(mg < x)) for x in (0, 0.05, 0.1, 0.25)},
               margin_below_by_class={c: {str(x): int(np.sum((mg < x) & k)) for x in (0.05, 0.1)} | {"bits": int(k.sum())}
                                      for c, k in (("sent1", sent == 1), ("sent0_after_0", (sent == 0) & (prev == 0)),
                                                   ("sent0_after_1", (sent == 0) & (prev == 1)))},
               max_node_share_sent1={f"Y{i}": float(np.mean(nodes[sent == 1] == i)) for i in sorted(set(nodes))})
    return out


def repeatability(rows_by_run, keys):
    """is the margin a property of the payload POSITION (pattern-dependent, repeats across frames and runs)?"""
    M = {}
    for k in keys:
        a = np.full((127,), np.nan); rows = rows_by_run[k]
        for b in range(127):
            a[b] = np.mean([r["margin"] for r in rows if r["bit"] == b])
        M[k] = a
    rk = lambda x: np.argsort(np.argsort(x))
    pairs = {f"{a}~{b}": float(np.corrcoef(rk(M[a]), rk(M[b]))[0, 1]) for i, a in enumerate(keys) for b in keys[i + 1:]}
    # within one run: positions' low-margin events across frames, and the sample-phase diversity per position
    low = {}
    for k in keys:
        rows = rows_by_run[k]
        lowpos = [r["bit"] for r in rows if r["margin"] < 0.1]
        low[k] = dict(low_margin_bits=len(lowpos), distinct_positions=len(set(lowpos)),
                      max_events_at_one_position=max([lowpos.count(p) for p in set(lowpos)] or [0]))
    phases = []
    for b in range(127):
        ph = [round(r["used_pos_ms"][0] * 2) / 2 for k in keys for r in rows_by_run[k] if r["bit"] == b]
        phases.append(len(set(ph)))
    return dict(spearman_mean_margin_by_position=pairs, low_margin=low,
                distinct_sample_phases_per_position=dict(min=int(min(phases)), median=float(np.median(phases)),
                                                         max=int(max(phases))))


def main():
    pl = SC.payload()
    out, rows_by_run = dict(runs={}), {}
    for k, run in RUNS.items():
        rows, info = per_bit(run, pl)
        rows_by_run[k] = rows
        out["runs"][k] = dict(info, **summarise(rows))
        print(k, json.dumps({kk: vv for kk, vv in out["runs"][k].items() if kk not in ("delay_ms",)})[:1500], flush=True)
    out["pooled_80"] = summarise(sum((rows_by_run[k] for k in ("80a", "80b", "80c")), []))
    out["repeatability_80"] = repeatability(rows_by_run, ["80a", "80b", "80c"])
    out["low_margin_bits_80"] = sorted(([dict(run=k, **{kk: r[kk] for kk in ("frame_k", "bit", "sent", "prev", "margin", "n",
                                                                           "n_inner", "used_pos_ms", "node_of_max")})
                                         for k in ("80a", "80b", "80c") for r in rows_by_run[k] if r["margin"] < 0.05]),
                                       key=lambda x: x["margin"])
    print("pooled 80", out["pooled_80"]); print("repeatability", out["repeatability_80"])
    json.dump(out, open(os.path.join(HERE, "mechanism_results.json"), "w"), indent=1)
    flat = [dict(run=k, **{kk: r[kk] for kk in ("frame_k", "m", "bit", "sent", "prev", "got", "margin", "v", "thr", "n", "n_inner")},
                 pos0=r["used_pos_ms"][0]) for k in RUNS for r in rows_by_run[k]]
    np.savez(os.path.join(HERE, "bits_table.npz"), **{c: np.array([x[c] for x in flat]) for c in flat[0]})


if __name__ == "__main__":
    main()
