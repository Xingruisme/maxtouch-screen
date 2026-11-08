#!/usr/bin/env python3
"""
v5 (frozen) vs v6 (experimental, finger-localised nodes) on the same runs: per run and per group.
Both are decoded by decode_v6.decode_run (pass 1 = v5 exactly, pass 2 = v6), so they share the recording.

   python v6_compare.py GROUP=<session or run dir>+... ... --label <name> [--rates 20,40,50,60]

Output data_maxtouch/v6_compare_<label>/: per_run.csv, summary.csv, tables.html.
Per group and decoder: known-timing errors / bits and error types; blind frames recovered and blind BER; errors in
frames where the located finger moved (centre shifted > 0.5 node between 0.25 s windows) vs frames where it did not;
how often the finger was visible and how often v6 used a node outside v4/v5's three.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode_v6 as V6
from v5_compare import expand, pct

DATADIR = os.path.join(HERE, "data_maxtouch")


def row_of(group, d):
    r6 = V6.decode_run(d)
    row = dict(group=group, run=r6["run"], modulation=r6["modulation"], bps=round(r6["bps"]),
               frames_sent=r6["v5_known"]["frames_sent"], finger_visible=r6["finger_visible"],
               outside_v4=r6["window_sets_outside_v4"], v4_nodes=" ".join(r6["v4_nodes"]),
               v6_nodes=" ".join(f"{k} {v:.0%}" for k, v in list(r6["nodes_used"].items())[:5]))
    mv = r6["frame_moving"]
    for dec in ("v5", "v6"):
        k, b = r6[f"{dec}_known"], r6[f"{dec}_blind"]
        fe = k["frame_errors"]
        row.update({f"{dec}_errors": k["errors"], f"{dec}_bits": k["bits"],
                    f"{dec}_sent1_read0": k["errors_by_type"]["sent1_read0"], f"{dec}_sent0_read1": k["errors_by_type"]["sent0_read1"],
                    f"{dec}_in_bursts": k["errors_by_type"]["in_bursts"],
                    f"{dec}_err_moving": sum(e for m, e in fe.items() if mv.get(m)),
                    f"{dec}_err_steady": sum(e for m, e in fe.items() if not mv.get(m)),
                    f"{dec}_blind_frames": b["frames_valid"], f"{dec}_blind_errors": b["errors"], f"{dec}_blind_bits": b["bits"]})
    row.update(frames_moving=sum(mv.values()), bits_moving=127 * sum(1 for m, v in mv.items() if v and m in r6["v5_known"]["frame_errors"]),
               bits_steady=127 * sum(1 for m, v in mv.items() if not v and m in r6["v5_known"]["frame_errors"]))
    print(f"  {row['run'][10:66]:56s} v5 {row['v5_errors']:4d} v6 {row['v6_errors']:4d} /{row['v5_bits']:<5d} blind "
          f"{row['v5_blind_frames']}/{row['v6_blind_frames']} of {row['frames_sent']}  finger visible {row['finger_visible']:.0%}  "
          f"outside v4 {row['outside_v4']:.0%}  moving frames {row['frames_moving']}", flush=True)
    return row


def summarise(rows):
    out = []
    for g in dict.fromkeys(r["group"] for r in rows):
        rr = [r for r in rows if r["group"] == g]
        sent = sum(r["frames_sent"] for r in rr)
        for dec in ("v5", "v6"):
            s = lambda k: sum(r[f"{dec}_{k}"] for r in rr)
            bm, bs = sum(r["bits_moving"] for r in rr), sum(r["bits_steady"] for r in rr)
            out.append(dict(group=g, decoder=dec, runs=len(rr), errors=s("errors"), bits=s("bits"), ber=s("errors") / s("bits"),
                            sent1_read0=s("sent1_read0"), sent0_read1=s("sent0_read1"), in_bursts=s("in_bursts"),
                            ber_moving_frames=s("err_moving") / bm if bm else None, bits_moving=bm,
                            ber_steady_frames=s("err_steady") / bs if bs else None, bits_steady=bs,
                            blind_frames=s("blind_frames"), frames_sent=sent, frame_recovery=s("blind_frames") / sent,
                            blind_ber=s("blind_errors") / s("blind_bits") if s("blind_bits") else None,
                            runs_better=sum(r[f"{dec}_errors"] < r[f"{'v6' if dec == 'v5' else 'v5'}_errors"] for r in rr),
                            finger_visible=float(np.mean([r["finger_visible"] for r in rr])),
                            windows_outside_v4=float(np.mean([r["outside_v4"] for r in rr]))))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("groups", nargs="+")
    p.add_argument("--label", required=True)
    p.add_argument("--rates")
    a = p.parse_args()
    rates = {int(v) for v in a.rates.split(",")} if a.rates else None
    rows = []
    for g in a.groups:
        name, spec = g.split("=", 1)
        for d in expand(spec):
            if rates and round(json.load(open(os.path.join(d, "meta.json")))["plan"]["bps"]) not in rates:
                continue
            rows.append(row_of(name, d))
    summ = summarise(rows)
    out = os.path.join(DATADIR, f"v6_compare_{a.label}")
    os.makedirs(out, exist_ok=True)
    for fn, data in (("per_run.csv", rows), ("summary.csv", summ)):
        with open(os.path.join(out, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    th = lambda c: "<tr>" + "".join(f"<th>{x}</th>" for x in c) + "</tr>"
    td = lambda c: "<tr>" + "".join(f"<td>{x}</td>" for x in c) + "</tr>"
    h = [f"<h3>v5 (sha {V6.V5.RULES_SHA}) vs v6 experimental - {a.label}</h3><table border=1 cellspacing=0 cellpadding=4>",
         th(["Group", "Decoder", "Runs", "Errors / bits", "BER", "1->0", "0->1", "In bursts", "BER, frames with finger moving",
             "BER, steady frames", "Blind frames", "Blind BER", "Runs better"])]
    for q in summ:
        h.append(td([q["group"], q["decoder"], q["runs"], f"{q['errors']}/{q['bits']}", pct(q["ber"]), q["sent1_read0"],
                     q["sent0_read1"], q["in_bursts"], f"{pct(q['ber_moving_frames'])} of {q['bits_moving']}",
                     f"{pct(q['ber_steady_frames'])} of {q['bits_steady']}",
                     f"{q['blind_frames']}/{q['frames_sent']}", pct(q["blind_ber"]), q["runs_better"]]))
    h += ["</table><h3>Per run</h3><table border=1 cellspacing=0 cellpadding=4>",
          th(["Group", "Run", "v5 errors", "v6 errors", "v5 / v6 blind frames", "Finger visible", "v6 windows outside v4 nodes",
              "v4 nodes", "v6 nodes used"])]
    for r in rows:
        h.append(td([r["group"], r["run"][10:], f"{r['v5_errors']}/{r['v5_bits']}", f"{r['v6_errors']}/{r['v6_bits']}",
                     f"{r['v5_blind_frames']} / {r['v6_blind_frames']} of {r['frames_sent']}", f"{r['finger_visible']:.0%}",
                     f"{r['outside_v4']:.0%}", r["v4_nodes"], r["v6_nodes"]]))
    open(os.path.join(out, "tables.html"), "w").write("\n".join(h + ["</table>"]))
    print(f"\n{'group':18s} dec {'errors/bits':>13s} {'BER':>8s} {'1->0':>5s} {'0->1':>5s} {'moving fr BER':>14s} "
          f"{'steady fr BER':>14s} {'blind fr':>9s} {'blind BER':>9s} better  finger vis  outside v4")
    for q in summ:
        print(f"{q['group']:18s} {q['decoder']}  {q['errors']:6d}/{q['bits']:<6d} {pct(q['ber']):>8s} {q['sent1_read0']:5d} "
              f"{q['sent0_read1']:5d} {pct(q['ber_moving_frames']):>14s} {pct(q['ber_steady_frames']):>14s} "
              f"{q['blind_frames']:4d}/{q['frames_sent']:<4d} {pct(q['blind_ber']):>9s} {q['runs_better']:5d}  "
              f"{q['finger_visible']:9.0%}  {q['windows_outside_v4']:9.0%}")
    print(f"\nSaved {os.path.relpath(out, HERE)}/  (all runs here are development data for v6)")


if __name__ == "__main__":
    main()
