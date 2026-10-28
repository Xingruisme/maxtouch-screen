#!/usr/bin/env python3
"""
Step 6 of the audit (informational, NOT a re-optimisation; no reported number is replaced by these):
  S1  how fragile is 0 errors at 80 bps? the blind decoder with single frozen parameters changed one at a time
      (guard 0/1/3/4 ms, single nodes, mean instead of max, nearest-to-centre sample, no clock correction,
      the stricter header-Hamming acceptance), scored independently. 90 and 100 bps decoded the same way for context.
  S2  sample-level error probability from many more samples than 9,144 bits: in every 155.7 kHz validation run,
      every guard-passing sample inside a payload bit whose neighbours carry the same value (deep ON / deep OFF,
      no edge nearby) is compared with its frame's threshold. If bit errors at 80 bps come from single weak ON
      readings (fades), this per-sample rate bounds them.
Writes sensitivity_results.json.
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import blind_decoder as BD
import score as SC

R80 = ["stent_ber_val4_80bps_20261022_192702", "stent_ber_val4b_80bps_20261022_195008",
       "stent_ber_val4b_80bps_20261022_195109"]
CONTEXT = ["stent_ber_val4b_90bps_20261022_195206", "stent_ber_val4b_90bps_20261022_195302",
           "stent_ber_val4_100bps_20261022_192753"]
VAL156 = ["stent_ber_val4_38bps_20261022_192113", "stent_ber_val4_40bps_20261022_192306",
          "stent_ber_val4_45bps_20261022_192448", "stent_ber_val4_68bps_20261022_192559"] + R80 + CONTEXT


def load(run):
    d = os.path.join(DATA, run)
    m = json.load(open(os.path.join(d, "meta.json"))); z = np.load(os.path.join(d, "capture.npz"))
    return z["t"].astype(float), z["X"].astype(float), m


VARIANTS = [("frozen v4 rules", {}), ("guard 0 ms", dict(guard_ms=0.0)), ("guard 1 ms", dict(guard_ms=1.0)),
            ("guard 3 ms", dict(guard_ms=3.0)), ("guard 4 ms", dict(guard_ms=4.0)),
            ("mean of 3 nodes", dict(combine="mean")), ("nearest-to-centre sample", dict(decision="nearest_center")),
            ("no clock correction (eps = 0)", dict(eps_ppm=[0])),
            ("Y2 only", dict(nodes=[2])), ("Y3 only", dict(nodes=[3])), ("Y4 only", dict(nodes=[4])),
            ("2 nodes (top 2)", dict(n_nodes=2))]


def s1(pl):
    out = {}
    for run in R80 + CONTEXT:
        t, X, m = load(run); lead = t < m["t_prompt"]; bps = float(m["args"]["bps"])
        rows = []
        for name, kw in VARIANTS:
            r = BD.decode(t, X, bps, lead, **kw)
            s = SC.score(r["frames"], m["t_first"], int(m["tx_frames"]), r["Tb"], r["eps"], pl)
            rows.append(dict(variant=name, true_frames=s["true_frames"], missing=s["missing_frames"],
                             false_positive=s["false_positive"], misaligned=s["misaligned"],
                             errors=s["errors"], bits=s["payload_bits"], ber_conservative=s["ber_conservative"]))
            if name == "frozen v4 rules":       # stricter payload-blind acceptance: <= 2 of 27 header bits wrong
                fr = [dict(f, valid=bool(f["header_errors"] <= 2)) for f in r["frames"]]
                s = SC.score(fr, m["t_first"], int(m["tx_frames"]), r["Tb"], r["eps"], pl)
                rows.append(dict(variant="frozen rules, acceptance: header errors <= 2", true_frames=s["true_frames"],
                                 missing=s["missing_frames"], false_positive=s["false_positive"],
                                 misaligned=s["misaligned"], errors=s["errors"], bits=s["payload_bits"],
                                 ber_conservative=s["ber_conservative"]))
        out[run] = rows
        print(run); [print("   ", x) for x in rows]
    return out


def s2(pl):
    full = np.concatenate([BD.HDR, pl, np.zeros(16)])
    out = {}
    for run in VAL156:
        t, X, m = load(run); lead = t < m["t_prompt"]; bps = float(m["args"]["bps"])
        r = BD.decode(t, X, bps, lead)
        s = SC.score(r["frames"], m["t_first"], int(m["tx_frames"]), r["Tb"], r["eps"], pl)
        true = {x["m"] for x in s["rows"] if x["kind"] == "true"}
        thr = {f["m"]: f["thr"] for f in r["frames"]}
        mm, j, pos = BD.bit_index(t, r["tau"], r["Tb"], r["eps"])
        g = 2.0 / 1e3 / (r["Tb"] * (1 + r["eps"]))
        ok = np.isin(mm, list(true)) & (j >= 28) & (j < 27 + 127 - 1) & (pos > g) & (pos < 1 - g)
        jj = np.clip(j, 1, 168)
        deep_on = ok & (full[jj] == 1) & (full[jj - 1] == 1) & (full[jj + 1] == 1)
        deep_off = ok & (full[jj] == 0) & (full[jj - 1] == 0) & (full[jj + 1] == 0)
        th = np.array([thr.get(x, np.nan) for x in mm])
        f = r["f"]
        out[run] = dict(bps=bps, deep_on_samples=int(deep_on.sum()), deep_on_below_thr=int(np.sum(f[deep_on] < th[deep_on])),
                        deep_off_samples=int(deep_off.sum()), deep_off_above_thr=int(np.sum(f[deep_off] > th[deep_off])),
                        on_margin_q001=float(np.quantile((f[deep_on] - th[deep_on]), 0.001)),
                        off_margin_q999=float(np.quantile((f[deep_off] - th[deep_off]), 0.999)),
                        on_values=np.round(f[deep_on] - th[deep_on], 4).tolist(),
                        off_values=np.round(f[deep_off] - th[deep_off], 4).tolist())
        print(run, {k: v for k, v in out[run].items() if not k.endswith("values")})
    tot = lambda k: sum(v[k] for v in out.values())
    n_on, e_on, n_off, e_off = tot("deep_on_samples"), tot("deep_on_below_thr"), tot("deep_off_samples"), tot("deep_off_above_thr")
    pooled = dict(deep_on_samples=n_on, deep_on_below_thr=e_on, deep_off_samples=n_off, deep_off_above_thr=e_off,
                  p_on_upper95=(1 - 0.05 ** (1 / n_on)) if e_on == 0 else None,
                  p_off_upper95=(1 - 0.05 ** (1 / n_off)) if e_off == 0 else None)
    print("pooled", pooled)
    return dict(runs=out, pooled=pooled)


def main():
    pl = SC.payload()
    out = dict(S1_variants=s1(pl), S2_sample_level=s2(pl))
    json.dump(out, open(os.path.join(HERE, "sensitivity_results.json"), "w"))


if __name__ == "__main__":
    main()
