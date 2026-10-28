#!/usr/bin/env python3
"""
Step 7 of the audit: late-stamped samples. The decoder takes the host arrival time as the acquisition time. After a
stalled HID transfer (interval since the previous arrival > 10 ms instead of 8 ms) the sample can carry an older
scan's value with a late timestamp; both 40 bps errors are exactly this. For every run: stalls during the
transmission, where the stalled sample landed, whether it decided its bit, and whether its bit differs from the
previous one (only then can a stale value flip the decision). Writes stalls_results.json.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import blind_decoder as BD, score as SC

RUNS = ["stent_ber_val4_38bps_20261022_192113", "stent_ber_val4_40bps_20261022_192306",
        "stent_ber_val4_45bps_20261022_192448", "stent_ber_val4_68bps_20261022_192559",
        "stent_ber_val4_80bps_20261022_192702", "stent_ber_val4b_80bps_20261022_195008",
        "stent_ber_val4b_80bps_20261022_195109", "stent_ber_val4b_90bps_20261022_195206",
        "stent_ber_val4b_90bps_20261022_195302", "stent_ber_val4_100bps_20261022_192753"]


def main():
    pl = SC.payload(); full = np.concatenate([BD.HDR, pl, np.zeros(16)])
    out = {}
    for run in RUNS:
        m = json.load(open(os.path.join(DATA, run, "meta.json"))); z = np.load(os.path.join(DATA, run, "capture.npz"))
        t, X = z["t"].astype(float), z["X"].astype(float); bps = float(m["args"]["bps"])
        r = BD.decode(t, X, bps, t < m["t_prompt"])
        s = SC.score(r["frames"], m["t_first"], int(m["tx_frames"]), r["Tb"], r["eps"], pl)
        true = {x["m"] for x in s["rows"] if x["kind"] == "true"}
        fr = {f["m"]: f for f in r["frames"]}
        mm, j, pos = BD.bit_index(t, r["tau"], r["Tb"], r["eps"])
        g = 2e-3 / (r["Tb"] * (1 + r["eps"])); inner = (pos > g) & (pos < 1 - g)
        dt = np.concatenate([[0.008], np.diff(t)])
        ev = []
        for i in np.where((dt > 0.010) & np.isin(mm, list(true)) & (j >= 27) & (j < 154))[0]:
            b = j[i] - 27
            w = np.where((mm == mm[i]) & (j == j[i]))[0]
            used = (inner[i] or not inner[w].any())
            ev.append(dict(m=int(mm[i]), t_s=float(t[i] - m["t_first"]), dt_ms=float(dt[i] * 1e3), bit=int(b), pos_ms=float(pos[i] * r["Tb"] * 1e3),
                           used_for_decision=bool(used), bit_differs_from_previous=bool(full[j[i]] != full[j[i] - 1]),
                           sample_on_wrong_side=bool((r["f"][i] > fr[mm[i]]["thr"]) != bool(full[j[i]])),
                           bit_error=bool(fr[mm[i]]["got"][b] != pl[b])))
        dangerous = [e for e in ev if e["used_for_decision"] and e["bit_differs_from_previous"]]
        out[run] = dict(bps=bps, payload_samples=int(np.sum(np.isin(mm, list(true)) & (j >= 27) & (j < 154))),
                        stalls_in_payload=len(ev), stalls_deciding_a_bit_after_a_transition=len(dangerous),
                        stale_value_on_wrong_side=sum(e["sample_on_wrong_side"] for e in ev),
                        stale_wrong_side_and_used=sum(e["sample_on_wrong_side"] and e["used_for_decision"] for e in ev),
                        stall_bit_errors=len({(e["m"], e["bit"]) for e in ev if e["bit_error"] and e["used_for_decision"]}),
                        bit_errors_total=s["errors"], events=ev)
        print(f"{run:42s} {bps:5.0f} bps  stalls in payload {len(ev):3d}  deciding a bit after a transition {len(dangerous):3d}  "
              f"stale wrong-side {out[run]['stale_value_on_wrong_side']} (used {out[run]['stale_wrong_side_and_used']})  error bits with a stall sample {out[run]['stall_bit_errors']} / all errors {s['errors']}")
    json.dump(out, open(os.path.join(HERE, "stalls_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
