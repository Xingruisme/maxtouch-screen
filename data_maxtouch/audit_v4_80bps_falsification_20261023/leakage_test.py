#!/usr/bin/env python3
"""
Step 3 of the audit: ground-truth leakage in the FROZEN decode_v4.py, tested on its own functions.

decode_v4 is imported read-only; decode_run (which writes result_v4.json / decoded_v4.json) is never called. The
combined feature is rebuilt with v4's own functions exactly as decode_run builds it (decode_v4.py lines 192-197).
  1. static trace: every line of decode_v4.py that touches the payload, frame bits, expected timing or BER
  2. v4.search() never receives the payload: its (tau, eps) is compared with the blind decoder's
  3. v4.decode_frames() is called with the true payload, a random payload, the inverted payload and all zeros
     (frame_bits' payload part replaced consistently): which outputs change?
Writes leakage_results.json.
"""
import json, os, re, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
ROOT = os.path.dirname(DATA)
sys.path.insert(0, ROOT)
import decode_v4 as V4                          # frozen; only its functions are used

RUNS = ["stent_ber_val4_80bps_20261022_192702", "stent_ber_val4b_80bps_20261022_195008",
        "stent_ber_val4b_80bps_20261022_195109"]
TOKENS = r"\b(truth|frame_bits|fb|payload|prbs|t_first|ber|err|errors|known_timing|known_mask|tx_frames|n_sent)\b"


def static_trace():
    src = open(V4.__file__).read().splitlines()
    # code lines only: line 34 on (after the module docstring), not comments
    return [dict(line=i + 1, code=s.strip()) for i, s in enumerate(src)
            if i + 1 >= 34 and re.search(TOKENS, s) and not s.lstrip().startswith("#")]


def main():
    out = dict(decoder_sha=V4.RULES_SHA, static_trace=static_trace(), runs={})
    for run in RUNS:
        d = os.path.join(DATA, run)
        meta = json.load(open(os.path.join(d, "meta.json")))
        z = np.load(os.path.join(d, "capture.npz")); t, X = z["t"].astype(float), z["X"].astype(float)
        a = V4.R.argparse.Namespace(**meta["args"]); link = V4.R.link_from(a, False)
        _, plan = V4.TX.build(link); frame_bits, truth = V4.TX.frame_bits(link.payload, link.prbs)
        Tb, nb = 1.0 / plan["bps"], len(frame_bits); P = nb * Tb
        pre, post = t < meta["t_prompt"], t >= meta["t_prompt"]
        e = np.diff(X, axis=0) ** 2
        score = 10 * np.log10((e[post[1:]].mean(axis=0) + 1e-9) / (e[pre[1:]].mean(axis=0) + 1e-9))
        top = [int(i) for i in np.argsort(-score)[:V4.N_NODES]]
        comb = np.max([V4.R.zscore(V4.absdev(X[:, i], pre)) for i in top], axis=0)

        def all_frames(tau, eps):
            last = int((t[-1] - tau) / (P * (1 + eps)))
            return np.arange(max(0, int(np.ceil((t[0] - tau) / (P * (1 + eps))))), last + 1)

        tau, eps = V4.search(comb, t, Tb, nb, t[0], t[0] + P, all_frames)
        blind = json.load(open(os.path.join(HERE, "blind_out", f"{run}.v4rules_blind.json")))
        rng = np.random.default_rng(20261023)
        variants = {"true_payload": list(truth), "random_payload": rng.integers(0, 2, 127).tolist(),
                    "inverted_payload": [1 - b for b in truth], "all_zero_payload": [0] * 127}
        res = {}
        for name, p in variants.items():
            fb = list(frame_bits[:27]) + list(p) + list(frame_bits[27 + 127:])
            fr = V4.decode_frames(comb, t, tau, Tb, eps, fb, p, all_frames(tau, eps), False)
            res[name] = fr
        ref = res["true_payload"]
        cmp = {}
        for name, fr in res.items():
            same = lambda k: all(np.allclose(np.array(x[k], float), np.array(y[k], float)) for x, y in zip(fr, ref))
            cmp[name] = dict(frames=len(fr), identical_got=same("got"), identical_thr=same("thr"),
                             identical_norm=same("norm"), identical_r=same("r"), identical_s0=same("s0"),
                             accepted=int(sum(f["valid"] for f in fr)), errors_in_accepted=int(sum(f["errors"] for f in fr if f["valid"])),
                             mean_frame_ber=float(np.mean([f["ber"] for f in fr])))
        out["runs"][run] = dict(nodes=[f"Y{i}" for i in top], v4_search_tau_minus_blind_ms=(tau - blind["tau"]) * 1e3,
                                v4_search_eps_ppm=eps * 1e6, blind_eps_ppm=blind["eps_ppm"], decode_frames_by_payload=cmp)
        print(run, out["runs"][run]["v4_search_tau_minus_blind_ms"], eps * 1e6)
        for k, v in cmp.items():
            print(f"   {k:18s} {v}")
    json.dump(out, open(os.path.join(HERE, "leakage_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
