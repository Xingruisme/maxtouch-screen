#!/usr/bin/env python3
"""
Slide figure: BER (%) per decoding method at each bit rate, 155.7 kHz, final reported result per bit rate.
Final result = the validation runs at that rate, pooled (errors / bits summed) when there are several; the
exploratory run only where no validation run exists (20 and 34 bps). Signal-lost runs are never used.
Reads result.json / result_v3.json / result_v4.json; writes fig_ber_by_method.png/.pdf and fig_ber_by_method.csv.
"""
import csv, json, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
RUNS = {
    20: ["stent_ber_c156_20bps_20261022_180358"],
    34: ["stent_ber_c156_34bps_20261022_180957"],
    40: ["stent_ber_val4_40bps_20261022_192306"],
    45: ["stent_ber_val4_45bps_20261022_192448"],
    68: ["stent_ber_val4_68bps_20261022_192559"],
    80: ["stent_ber_val4_80bps_20261022_192702", "stent_ber_val4b_80bps_20261022_195008",
         "stent_ber_val4b_80bps_20261022_195109"],
    90: ["stent_ber_val4b_90bps_20261022_195206", "stent_ber_val4b_90bps_20261022_195302"],
    100: ["stent_ber_val4_100bps_20261022_192753"],
}
METHODS = [("Original", "#8f8e8a"), ("Grid-based", "#eda100"), ("Timestamp-based, 1 node", "#86b6ef"),
           ("Timestamp-based, 3 nodes", "#2a78d6")]   # = original, v3, v4 single (top-ranked node), v4 combined


def counts(run):
    d = os.path.join(DATA, run)
    j = lambda f: json.load(open(os.path.join(d, f)))
    v4 = j("result_v4.json")
    assert v4["carrier_hz"] == 155692.0 and not v4["health"]["signal_lost"]
    pick = [j("result.json")["combined"], j("result_v3.json")["combined_blind"],
            v4["single_top_ranked"], v4["combined_blind"]]       # single node = top-ranked, chosen before decoding
    return [(p["errors"], p["bits"]) for p in pick]


def label(v):
    return "0" if v == 0 else f"{v:.2f}" if v < 1 else f"{v:.1f}"


def main():
    bps = list(RUNS)
    ber = np.zeros((len(bps), len(METHODS)))
    rows = []
    for i, b in enumerate(bps):
        c = np.array([counts(r) for r in RUNS[b]]).sum(axis=0)      # pooled errors, bits per method
        ber[i] = 100 * c[:, 0] / c[:, 1]
        rows.append([b] + [f"{e}/{n}" for e, n in c] + [f"{x:.4f}" for x in ber[i]])
    with open(os.path.join(HERE, "fig_ber_by_method.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bps"] + [f"{m} errors/bits" for m, _ in METHODS] + [f"{m} BER %" for m, _ in METHODS])
        w.writerows(rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor="white")
    x = np.arange(len(bps)); w = 0.2
    for k, (name, col) in enumerate(METHODS):
        xs = x + (k - 1.5) * w
        ax.bar(xs, ber[:, k], w, color=col, edgecolor="white", linewidth=1.5, label=name, zorder=2)
        for xx, v in zip(xs, ber[:, k]):
            ax.text(xx, v + 0.25, label(v), ha="center", va="bottom", rotation=90, fontsize=8.5, color="#2b2b2b")
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in bps])
    ax.set_xlabel("Bit rate (bps)"); ax.set_ylabel("BER (%)")
    ax.set_ylim(0, 20); ax.set_xlim(-0.6, len(bps) - 0.4)
    ax.yaxis.grid(True, color="#e4e3df", linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#52514e")
    ax.tick_params(colors="#52514e")
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=10, handlelength=1.2, columnspacing=1.6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"fig_ber_by_method.{ext}"), dpi=300, facecolor="white")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
