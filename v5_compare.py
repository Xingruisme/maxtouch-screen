#!/usr/bin/env python3
"""
v4 vs v5 on the same recordings, per condition and per run. Reads v4 from the saved result_v4.json (frozen decoder,
not re-run) and decodes v5 with the frozen decode_v5.py. Nothing in the run folders is changed except result_v5.json /
decoded_v5.json, which are written only for validation runs (recorded after v5's FROZEN_AT).

   python v5_compare.py still=<dir>+<dir>... moving=<dir>+<dir>... --label <name> [--rates 20,40,50,60] [--include-dev]
   python v5_compare.py --realtime-notes

A <dir> is a session folder (data_maxtouch/modcomp_<label>_<time>: its OOK and FSK runs) or a single run folder.
Development runs (recorded before FROZEN_AT) are refused unless --include-dev, and are then marked in every table.

Output data_maxtouch/v5_compare_<label>/: per_run.csv, summary.csv, tables.html (summary + per-run, paste into
Slides), per_run.png (v4 vs v5 errors of every run).

Per condition and decoder:
  known-timing BER (errors / bits, 95 % Clopper-Pearson), errors by type (sent 1 read 0, sent 0 read 1, errors within
  2 bits of another error), frames decoded with known timing (v4/v5 skip a frame with an empty bit)
  blind frame recovery (valid frames found without the start time / frames sent) and blind BER on those frames
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
import decode_v5 as V5

DATADIR = os.path.join(HERE, "data_maxtouch")
REALTIME_NOTES = """\
Is v5 offline?  Yes, and so is v4; the +-0.25 s baseline window is only one of the reasons.
  - Baseline (v5 only): a centred window needs 0.25 s of samples after the one being decoded. On its own this is a
    fixed 0.25 s latency, not a barrier: a live decoder can buffer 0.25 s.
  - Z-scoring (v4 and v5): median and IQR of the feature over the whole run.
  - Timing (v4 and v5): known timing searches delay and clock drift over all frames; blind decoding searches the
    whole recording for the frame grid.
  - Threshold (v4 and v5): per frame, halfway between the known 1 and 0 bits of the header AND the trailing gap after
    the payload, so a frame is decided only after it has ended.
What a real-time version would need (each change is a new decoder to validate against this offline reference):
  1. Baseline: keep the centred +-0.25 s median with a 0.25 s output delay (same numbers as offline), or a causal
     trailing median (no delay, but it lags a moving finger by ~half its window).
  2. Feature scale: z-score from the lead (before START TX) and a running estimate, not from the whole run.
  3. Timing: per-frame preamble/SFD correlation as samples arrive, with the clock drift tracked from frame to frame.
  4. Threshold: from the header (preamble + SFD) only, before the payload, optionally updated decision by decision.
  5. Cost is small: the median over ~63 samples (0.5 s at 125 scans/s) x 3 nodes per scan.
"""


def chance_ci(e, n):
    from scipy.stats import beta
    lo = beta.ppf(0.025, e, n - e + 1) if e > 0 else 0.0
    hi = beta.ppf(0.975, e + 1, n - e) if e < n else 1.0
    return lo, hi


def expand(spec):
    out = []
    for d in spec.split("+"):
        d = d if os.path.isabs(d) or os.path.exists(d) else os.path.join(DATADIR, d)
        if os.path.exists(os.path.join(d, "summary.json")):
            out += [os.path.join(DATADIR, r["run"]) for r in json.load(open(os.path.join(d, "summary.json")))["rows"]
                    if r["modulation"] in ("ook", "fsk")]
        elif os.path.exists(os.path.join(d, "meta.json")):
            out.append(d)
        else:
            sys.exit(f"not a session or run folder: {d}")
    return out


def row_of(cond, d, include_dev):
    meta = json.load(open(os.path.join(d, "meta.json")))
    role = V5.role(d, meta)
    if role != "validation" and not include_dev:
        sys.exit(f"{os.path.basename(d)} was recorded before v5 was frozen ({V5.FROZEN_AT}); use --include-dev to "
                 "show development runs (marked as such)")
    v4 = json.load(open(os.path.join(d, "result_v4.json")))
    v5 = V5.decode_run(d, write=role == "validation")
    r = dict(condition=cond, run=os.path.basename(d), role=role, modulation=meta["modulation"],
             bps=round(meta["plan"]["bps"]), frames_sent=v5["combined_known_timing"]["frames_sent"])
    for name, res in (("v4", v4), ("v5", v5)):
        k, b = res["combined_known_timing"], res["combined_blind"]
        r.update({f"{name}_errors": k["errors"], f"{name}_bits": k["bits"], f"{name}_frames_known": k["frames_valid"],
                  f"{name}_sent1_read0": k["errors_by_type"]["sent1_read0"],
                  f"{name}_sent0_read1": k["errors_by_type"]["sent0_read1"],
                  f"{name}_in_bursts": k["errors_by_type"]["in_bursts"],
                  f"{name}_blind_frames": b["frames_valid"], f"{name}_blind_errors": b["errors"], f"{name}_blind_bits": b["bits"]})
    r.update(v4_sha=v4.get("rules_sha"), v5_sha=v5["rules_sha"])
    print(f"  {r['run'][10:70]:60s} {r['role'][:3]}  v4 {r['v4_errors']:4d}/{r['v4_bits']:<5d} "
          f"v5 {r['v5_errors']:4d}/{r['v5_bits']:<5d}  blind frames v4 {r['v4_blind_frames']}/{r['frames_sent']} "
          f"v5 {r['v5_blind_frames']}/{r['frames_sent']}", flush=True)
    return r


def summarise(rows):
    out = []
    for cond in dict.fromkeys(r["condition"] for r in rows):
        rr = [r for r in rows if r["condition"] == cond]
        for name in ("v4", "v5"):
            s = {k: sum(r[f"{name}_{k}"] for r in rr) for k in
                 ("errors", "bits", "frames_known", "sent1_read0", "sent0_read1", "in_bursts", "blind_frames", "blind_errors", "blind_bits")}
            sent = sum(r["frames_sent"] for r in rr)
            lo, hi = chance_ci(s["errors"], s["bits"])
            out.append(dict(condition=cond, decoder=name, runs=len(rr), sessions_dev=sum(r["role"] != "validation" for r in rr),
                            errors=s["errors"], bits=s["bits"], ber=s["errors"] / s["bits"], ber_lo95=lo, ber_hi95=hi,
                            sent1_read0=s["sent1_read0"], sent0_read1=s["sent0_read1"], in_bursts=s["in_bursts"],
                            frames_known=s["frames_known"], frames_sent=sent, blind_frames=s["blind_frames"],
                            frame_recovery=s["blind_frames"] / sent if sent else None,
                            blind_ber=s["blind_errors"] / s["blind_bits"] if s["blind_bits"] else None,
                            better_runs=sum(r[f"{name}_errors"] < r[f"{'v5' if name == 'v4' else 'v4'}_errors"] for r in rr)))
    return out


def pct(x):
    return "-" if x is None else f"{100 * x:.2f} %"


def html_tables(summ, rows, label):
    th = lambda cells: "<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>"
    td = lambda cells: "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
    s = [f"<h3>v4 vs v5 ({label}) - v5 sha {V5.RULES_SHA}, frozen {V5.FROZEN_AT}; v4 sha {V5.V4_SHA}</h3>",
         "<table border=1 cellspacing=0 cellpadding=4>",
         th(["Condition", "Decoder", "Runs", "Errors / bits", "BER (95 % CI)", "Sent 1 read 0", "Sent 0 read 1",
             "In bursts", "Frames recovered blind", "Blind BER", "Runs better than other decoder"])]
    for q in summ:
        s.append(td([q["condition"], q["decoder"], q["runs"], f"{q['errors']}/{q['bits']}",
                     f"{pct(q['ber'])} ({pct(q['ber_lo95'])}-{pct(q['ber_hi95'])})", q["sent1_read0"], q["sent0_read1"],
                     q["in_bursts"], f"{q['blind_frames']}/{q['frames_sent']} ({pct(q['frame_recovery'])})",
                     pct(q["blind_ber"]), q["better_runs"]]))
    s += ["</table>", "<h3>Per run (known-timing errors / bits; blind frames recovered)</h3>",
          "<table border=1 cellspacing=0 cellpadding=4>",
          th(["Condition", "Run", "Role", "Mod", "bps", "v4 errors", "v5 errors", "v4 1->0 / 0->1", "v5 1->0 / 0->1",
              "v4 blind frames", "v5 blind frames"])]
    for r in rows:
        s.append(td([r["condition"], r["run"][10:], r["role"], r["modulation"].upper(), r["bps"],
                     f"{r['v4_errors']}/{r['v4_bits']}", f"{r['v5_errors']}/{r['v5_bits']}",
                     f"{r['v4_sent1_read0']} / {r['v4_sent0_read1']}", f"{r['v5_sent1_read0']} / {r['v5_sent0_read1']}",
                     f"{r['v4_blind_frames']}/{r['frames_sent']}", f"{r['v5_blind_frames']}/{r['frames_sent']}"]))
    return "\n".join(s + ["</table>"])


def figure(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    fig, ax = plt.subplots(1, len(conds), figsize=(5.2 * len(conds), 4.8), squeeze=False)
    for a, cond in zip(ax[0], conds):
        rr = [r for r in rows if r["condition"] == cond]
        x = np.array([r["v4_errors"] for r in rr]); y = np.array([r["v5_errors"] for r in rr])
        top = max(10, x.max(), y.max()) * 1.15
        a.plot([0, top], [0, top], color="#888", lw=1, ls="--")
        for mod, mk, c in (("ook", "s", "#ff7f0e"), ("fsk", "^", "#2ca02c")):
            k = [r["modulation"] == mod for r in rr]
            a.scatter(x[k], y[k], marker=mk, s=46, color=c, edgecolor="white", linewidth=0.8, label=mod.upper(), zorder=3)
        a.set_xlim(-0.03 * top, top); a.set_ylim(-0.03 * top, top)
        a.set_xlabel("v4 errors per run (known timing)"); a.set_ylabel("v5 errors per run (known timing)")
        a.set_title(f"{cond}: v5 better in {int(np.sum(y < x))}, worse in {int(np.sum(y > x))}, same {int(np.sum(y == x))}",
                    fontsize=10)
        a.text(0.97, 0.03, "below the line = v5 fewer errors", transform=a.transAxes, ha="right", fontsize=8, color="#555")
        a.legend(frameon=False, loc="upper left"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("groups", nargs="*", help="CONDITION=dir+dir...")
    p.add_argument("--label")
    p.add_argument("--rates", help="keep only these bit rates, e.g. 20,40,50,60")
    p.add_argument("--include-dev", action="store_true", help="allow runs recorded before v5 was frozen (marked)")
    p.add_argument("--realtime-notes", action="store_true")
    a = p.parse_args()
    if a.realtime_notes:
        print(REALTIME_NOTES); return
    if not a.groups or not a.label:
        p.error("give CONDITION=dirs groups and --label")
    rates = {int(v) for v in a.rates.split(",")} if a.rates else None
    print(f"v5 sha {V5.RULES_SHA} (frozen {V5.FROZEN_AT}), v4 sha {V5.V4_SHA}")
    rows = []
    for g in a.groups:
        cond, spec = g.split("=", 1)
        for d in expand(spec):
            meta = json.load(open(os.path.join(d, "meta.json")))
            if rates and round(meta["plan"]["bps"]) not in rates:
                continue
            rows.append(row_of(cond, d, a.include_dev))
    summ = summarise(rows)
    out = os.path.join(DATADIR, f"v5_compare_{a.label}")
    os.makedirs(out, exist_ok=True)
    for name, data in (("per_run.csv", rows), ("summary.csv", summ)):
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    open(os.path.join(out, "tables.html"), "w").write(html_tables(summ, rows, a.label))
    figure(rows, os.path.join(out, "per_run.png"))
    print(f"\n{'condition':18s} {'dec':3s} {'errors/bits':>13s} {'BER':>8s} {'1->0':>5s} {'0->1':>5s} {'burst':>5s} "
          f"{'blind frames':>14s} {'blind BER':>9s}")
    for q in summ:
        print(f"{q['condition']:18s} {q['decoder']:3s} {q['errors']:6d}/{q['bits']:<6d} {pct(q['ber']):>8s} "
              f"{q['sent1_read0']:5d} {q['sent0_read1']:5d} {q['in_bursts']:5d} "
              f"{q['blind_frames']:5d}/{q['frames_sent']:<4d}({pct(q['frame_recovery'])}) {pct(q['blind_ber']):>9s}")
    if any(r["role"] != "validation" for r in rows):
        print("\nNOTE: includes development runs (recorded before v5 was frozen) - not a validation result")
    print(f"\nSaved {os.path.relpath(out, HERE)}/ (per_run.csv, summary.csv, tables.html, per_run.png)")


if __name__ == "__main__":
    main()
