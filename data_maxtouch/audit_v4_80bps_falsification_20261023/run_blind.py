#!/usr/bin/env python3
"""
Step 1 of the audit: payload-blind decoding of the saved captures. Runs in its own process with only numpy, json and
blind_decoder imported (blind_decoder refuses to run next to any module that carries the payload).

Per run it reads capture.npz (t, X) and from meta.json ONLY args.bps and t_prompt (the end of the transmitter-off
lead). Two modes:
  v4rules_blind : lead = samples before t_prompt (as v4)
  strict_blind  : lead = the first 3.0 s of the recording; no meta timing at all
Writes blind_out/<run>.<mode>.json. Nothing in the run folders is written.
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import blind_decoder as BD

TARGET = ["stent_ber_val4_80bps_20261022_192702", "stent_ber_val4b_80bps_20261022_195008",
          "stent_ber_val4b_80bps_20261022_195109"]
POSITIVE = ["stent_ber_val4_38bps_20261022_192113", "stent_ber_val4_40bps_20261022_192306",
            "stent_ber_val4_45bps_20261022_192448", "stent_ber_val4_68bps_20261022_192559",
            "stent_ber_val4b_90bps_20261022_195206", "stent_ber_val4b_90bps_20261022_195302",
            "stent_ber_val4_100bps_20261022_192753", "stent_ber_val4b_80bps_20261022_200327"]
OUT = os.path.join(HERE, "blind_out")


def receiver_inputs(run):
    d = os.path.join(DATA, run)
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "capture.npz"))
    return z["t"].astype(float), z["X"].astype(float), float(meta["args"]["bps"]), float(meta["t_prompt"])


def pack(res, run, mode, lead_desc):
    return dict(run=run, mode=mode, lead=lead_desc, nodes=[f"Y{i}" for i in res["nodes"]], tau=res["tau"],
                eps_ppm=res["eps"] * 1e6, header_score=res["header_score"], bps=1 / res["Tb"],
                frames=res["frames"])


def main(runs):
    os.makedirs(OUT, exist_ok=True)
    for run in runs:
        t, X, bps, t_prompt = receiver_inputs(run)
        for mode, lead, desc in (("v4rules_blind", t < t_prompt, "t < t_prompt"),
                                 ("strict_blind", t < t[0] + 3.0, "first 3.0 s")):
            res = BD.decode(t, X, bps, lead)
            json.dump(pack(res, run, mode, desc), open(os.path.join(OUT, f"{run}.{mode}.json"), "w"))
            acc = sum(f["valid"] for f in res["frames"])
            print(f"{run:45s} {mode:14s} nodes {res['nodes']} eps {res['eps'] * 1e6:+.0f} ppm  "
                  f"slots {len(res['frames'])}  accepted (header r >= 0.3) {acc}", flush=True)


if __name__ == "__main__":
    main(TARGET + POSITIVE if "--all" in sys.argv else TARGET)
