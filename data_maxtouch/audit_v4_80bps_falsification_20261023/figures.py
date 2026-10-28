#!/usr/bin/env python3
"""
Figures of the audit (PNG 200 dpi + PDF), from the saved captures, blind_out/, and the *_results.json of the other
steps. Read-only on all run folders.
  figA_mechanism        how samples become bits at 80 bps
  figB_timing           timing falsification: BER vs offset, header-score landscape
  figC_controls         negative controls: header r / header-bit errors of true frames vs OFF-only and surrogates
  figD_bits             bit-level diagnostics of 8 selected bits
  figE_sensitivity      errors at 80 bps when one frozen rule is changed
"""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import blind_decoder as BD
import score as SC

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
NODE_C = {2: "#2a78d6", 3: "#eb6834", 4: "#1baf7a"}
RUN_C = {"80a": "#2a78d6", "80b": "#eb6834", "80c": "#1baf7a"}
RUNS80 = {"80a": "stent_ber_val4_80bps_20261022_192702", "80b": "stent_ber_val4b_80bps_20261022_195008",
          "80c": "stent_ber_val4b_80bps_20261022_195109"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})
J = lambda f: json.load(open(os.path.join(HERE, f)))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_run(run):
    m = json.load(open(os.path.join(DATA, run, "meta.json"))); z = np.load(os.path.join(DATA, run, "capture.npz"))
    t, X = z["t"].astype(float), z["X"].astype(float)
    lead = t < m["t_prompt"]
    return t, X, m, lead, BD.decode(t, X, float(m["args"]["bps"]), lead)


# ------------------------------------------------------------------ A
def fig_mechanism(T):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.2))
    ax = ax.ravel()
    sel = np.isin(T["run"], ["80a", "80b", "80c"])
    # (a) arrival intervals
    dts = []
    for k, run in RUNS80.items():
        t, X, m, lead, _ = load_run(run)
        w = (t > m["t_first"]) & (t < m["t_first"] + 51)
        dts.append(np.diff(t[w]) * 1e3)
    d = np.concatenate(dts)
    ax[0].hist(d, bins=np.arange(5, 20.25, 0.25), color="#2a78d6", edgecolor="white", linewidth=0.3)
    ax[0].set_yscale("log"); ax[0].axvline(12.5, color=INK2, ls="--", lw=1)
    ax[0].text(12.6, ax[0].get_ylim()[1] * 0.3, "bit = 12.5 ms", color=INK2)
    ax[0].set_xlabel("interval between sample arrivals [ms]"); ax[0].set_ylabel("count (3 runs, during transmission)")
    ax[0].set_title(f"(a) arrivals: median {np.median(d):.2f} ms, {np.mean(d > 10) * 100:.1f} % > 10 ms (stalled transfer)",
                    loc="left", fontsize=9)
    # (b) position of the deciding sample(s) inside the bit
    p = T["pos0"][sel]
    ax[1].hist(p, bins=np.arange(0, 12.75, 0.25), color="#2a78d6", edgecolor="white", linewidth=0.3)
    for x0, x1 in ((0, 2), (10.5, 12.5)):
        ax[1].axvspan(x0, x1, color=GRID, alpha=0.6, lw=0)
    ax[1].text(0.1, ax[1].get_ylim()[1] * 0.9, "guard", color=INK2); ax[1].text(10.6, ax[1].get_ylim()[1] * 0.9, "guard", color=INK2)
    ax[1].set_xlabel("position of the first deciding sample in its bit [ms]  (bit edge = tau, incl. ~11.3 ms delay)")
    ax[1].set_ylabel("payload bits"); ax[1].set_title("(b) where the deciding sample sits (samples on a 0.5 ms lattice)", loc="left", fontsize=9)
    # (c) samples per bit
    n, ni = T["n"][sel], T["n_inner"][sel]
    xs = np.arange(4); w = 0.38
    ax[2].bar(xs - w / 2, [np.sum(n == k) for k in xs], w, color="#86b6ef", label="all samples in the bit")
    ax[2].bar(xs + w / 2, [np.sum(ni == k) for k in xs], w, color="#2a78d6", label="samples outside the +-2 ms guard")
    for x, v in zip(xs + w / 2, [np.sum(ni == k) for k in xs]):
        ax[2].text(x, v, f"{v}", ha="center", va="bottom", fontsize=8, color=INK)
    ax[2].set_xticks(xs); ax[2].set_xlabel("samples per payload bit"); ax[2].set_ylabel("payload bits (9,144)")
    ax[2].set_ylim(0, 11500); ax[2].legend(loc="upper right"); ax[2].set_title("(c) 1.55 samples/bit: 93 % of bits decided by ONE sample", loc="left", fontsize=9)
    # (d) decision values relative to the frame threshold
    rel = (T["v"] - T["thr"])[sel]; s = T["sent"][sel]; pv = T["prev"][sel]
    bins = np.arange(-1.0, 1.0, 0.02)
    ax[3].hist(rel[s == 0], bins=bins, color="#1baf7a", alpha=0.85, label="sent 0")
    ax[3].hist(rel[s == 1], bins=bins, color="#eb6834", alpha=0.85, label="sent 1")
    ax[3].hist(rel[(s == 0) & (pv == 1)], bins=bins, histtype="step", color=INK, lw=1, label="sent 0 after a 1")
    ax[3].axvline(0, color=INK, lw=1); ax[3].set_yscale("log")
    ax[3].set_xlabel("decision value - frame threshold  [z units of the combined feature]"); ax[3].set_ylabel("payload bits")
    ax[3].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3); ax[3].set_title("(d) no bit crosses; closest margins +0.007 (1), +0.028 (0)", loc="left", fontsize=9)
    # (e) margin CDF lower tail by bit rate
    labs = {"68": "68 bps (0 err)", "80a": "80 bps x3 (0 err)", "90a": "90 bps x2 (19 err)", "100": "100 bps (32 err)"}
    cols = {"68": "#6250d6", "80a": "#2a78d6", "90a": "#eb6834", "100": "#e34948"}
    groups = {"68": ["68"], "80a": ["80a", "80b", "80c"], "90a": ["90a", "90b"], "100": ["100"]}
    for k, g in groups.items():
        mg = np.sort(T["margin"][np.isin(T["run"], g)])
        ax[4].step(mg, np.arange(1, len(mg) + 1) / len(mg), where="post", color=cols[k], label=labs[k], lw=1.6)
    ax[4].set_yscale("log"); ax[4].set_xlim(-0.6, 0.35); ax[4].set_ylim(5e-5, 0.2); ax[4].axvline(0, color=INK, lw=1)
    ax[4].set_xlabel("decision margin (normalised; < 0 = bit error)"); ax[4].set_ylabel("fraction of bits with margin <= x")
    ax[4].legend(loc="lower right"); ax[4].set_title("(e) the 80 bps tail ends just above 0; 90/100 bps cross it", loc="left", fontsize=9)
    # (f) which node supplies the max for sent-1 bits
    M = J("mechanism_results.json")["runs"]
    ks = ["80a", "80b", "80c"]
    bottom = np.zeros(3)
    for node in ("Y2", "Y3", "Y4"):
        v = np.array([M[k]["max_node_share_sent1"].get(node, 0) for k in ks])
        ax[5].bar(ks, v, bottom=bottom, color=NODE_C[int(node[1])], label=node, edgecolor="white", linewidth=1)
        bottom += v
    ax[5].set_ylabel("share of sent-1 bits"); ax[5].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    ax[5].set_xticks(range(3)); ax[5].set_xticklabels(["192702", "195008", "195109"])
    ax[5].set_title("(f) node that supplied the per-sample max (sent-1 bits)", loc="left", fontsize=9)
    fig.suptitle("A. Detection mechanism at 155.7 kHz / 80 bps (three validation runs, payload-blind decoding)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "figA_mechanism")


# ------------------------------------------------------------------ B
def fig_timing():
    C = J("controls_results.json")["C_timing"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1.1, 1]))
    for k, run in RUNS80.items():
        rows = C[run]["rows"]
        ms = [(0.0, rows[0])] + [(float(r["label"].split()[1]), r) for r in rows if r["label"].endswith(" ms")]
        ms.sort()
        x = [a for a, _ in ms]; y = [r["payload_errors_in_tx_slots"] for _, r in ms]
        ax[0].plot(x, np.maximum(y, 0.5), "o-", color=RUN_C[k], ms=4, lw=1.4, label=run[-15:])
    ax[0].set_yscale("log"); ax[0].axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax[0].text(-6, 0.55, "0 errors", color=INK2, fontsize=8)
    ax[0].set_xlabel("bit grid shifted from the blind estimate [ms]  (bit = 12.5 ms)")
    ax[0].set_ylabel("payload errors / 3,048 (all 24 frames scored)")
    ax[0].legend(loc="upper center"); ax[0].set_title("(a) error-free only within about +-1 ms of the header-found timing", loc="left", fontsize=9)
    L = C[RUNS80["80a"]]["landscape"]
    tt = np.array(L["tau_s"]) * 1e3; sc = np.array(L["score"])
    ax[1].plot(tt, sc, color="#2a78d6", lw=0.8)
    i0 = int(np.argmax(sc))
    ax[1].annotate(f"peak {L['peak']:.1f}\n(found timing)", (tt[i0], sc[i0]), xytext=(tt[i0] + 250, sc[i0] - 2), fontsize=8,
                   arrowprops=dict(arrowstyle="-", color=INK2))
    ax[1].axhline(L["second_peak_beyond_1bit"], color=INK2, ls="--", lw=0.8)
    ax[1].text(tt[0], L["second_peak_beyond_1bit"] + 0.6, f"best elsewhere (> 1 bit away): {L['second_peak_beyond_1bit']:.1f}",
               fontsize=8, color=INK2)
    ax[1].set_xlabel("frame offset tau - first sample [ms]  (one frame = 2,125 ms)")
    ax[1].set_ylabel("summed header correlation (24 frames)")
    ax[1].set_title("(b) header-score landscape, run 192702: one unambiguous peak (+-2-bit sidelobes = preamble)", loc="left", fontsize=9)
    fig.tight_layout()
    save(fig, "figB_timing")


# ------------------------------------------------------------------ C
def fig_controls():
    C = J("controls_results.json")
    true_r, true_he = [], []
    for k, run in RUNS80.items():
        b = json.load(open(os.path.join(HERE, "blind_out", f"{run}.v4rules_blind.json")))
        S = J("score_results.json")["runs"][run]["v4rules_blind"]
        tm = {r["m"] for r in S["rows"] if r["kind"] == "true"}
        true_r += [f["r"] for f in b["frames"] if f["m"] in tm]; true_he += [f["header_errors"] for f in b["frames"] if f["m"] in tm]
    groups = [("true frames, 80 bps (72)", true_r, "#2a78d6")]
    off = []
    for name in ("A1_tx_dead_210s", "A2_onoff_clean_30s", "A3_onoff_disturbed_30s", "A4_concatenated_leads"):
        off += C["A_off_only"][name]["Y2_Y4_forced"]["r_values"] + C["A_off_only"][name]["auto_nodes"]["r_values"]
    groups.append((f"OFF-only recordings ({len(off)} slots)", off, "#1baf7a"))
    boot = sum((C["B_surrogates"][r]["B1_off_bootstrap"]["r_values"] for r in RUNS80.values()), [])
    perm = sum((C["B_surrogates"][r]["B3_permuted"]["r_values"] for r in RUNS80.values()), [])
    rev = sum((C["B_surrogates"][r]["B2_time_reversed"]["r_values"] for r in RUNS80.values()), [])
    groups += [(f"OFF bootstrap surrogates ({len(boot)})", boot, "#eda100"), (f"sample-permuted ({len(perm)})", perm, "#e87ba4"),
               (f"time-reversed captures ({len(rev)})", rev, "#6250d6")]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    bins = np.arange(-1, 1.02, 0.04)
    for name, v, c in groups:
        ax[0].hist(v, bins=bins, histtype="step", lw=1.6, color=c, label=name, density=True)
    ax[0].axvline(0.3, color=INK, ls="--", lw=1); ax[0].text(0.31, ax[0].get_ylim()[1] * 0.85, "v4 header rule r >= 0.3", fontsize=8)
    ax[0].set_xlabel("per-frame header correlation r"); ax[0].set_ylabel("density"); ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title("(a) r >= 0.3 accepts 10-15 % of noise slots and most time-reversed frames", loc="left", fontsize=9)
    rows = [("OFF-only (A1-A4), all node choices", sum(C["A_off_only"][n][m]["slots"] for n in C["A_off_only"] for m in ("auto_nodes", "Y2_Y4_forced")),
             sum(C["A_off_only"][n][m]["accepted_r0.3"] for n in C["A_off_only"] for m in ("auto_nodes", "Y2_Y4_forced")),
             sum(C["A_off_only"][n][m]["accepted_r0.7"] for n in C["A_off_only"] for m in ("auto_nodes", "Y2_Y4_forced")),
             sum(C["A_off_only"][n][m]["accepted_hdr_errors_le2"] for n in C["A_off_only"] for m in ("auto_nodes", "Y2_Y4_forced")))]
    for key, lab in (("B1_off_bootstrap", "OFF bootstrap (60 realisations)"), ("B3_permuted", "sample-permuted (60)")):
        rows.append((lab, sum(C["B_surrogates"][r][key]["slots"] for r in RUNS80.values()),
                     sum(C["B_surrogates"][r][key]["accepted_r0.3"] for r in RUNS80.values()),
                     sum(C["B_surrogates"][r][key]["accepted_r0.7"] for r in RUNS80.values()),
                     sum(C["B_surrogates"][r][key]["accepted_hdr_errors_le2"] for r in RUNS80.values())))
    rows.append(("time-reversed (3)", sum(C["B_surrogates"][r]["B2_time_reversed"]["slots"] for r in RUNS80.values()),
                 sum(C["B_surrogates"][r]["B2_time_reversed"]["accepted_r0.3"] for r in RUNS80.values()),
                 sum(C["B_surrogates"][r]["B2_time_reversed"]["accepted_r0.7"] for r in RUNS80.values()),
                 sum(C["B_surrogates"][r]["B2_time_reversed"]["accepted_hdr_errors_le2"] for r in RUNS80.values())))
    sh = [r for run in RUNS80.values() for r in C["C_timing"][run]["rows"] if r["label"] in ("tau -2 bit", "tau +2 bit")]
    rows.append(("tau shifted +-2 bits (real captures)", 26 * len(sh), sum(r["accepted_r03"] for r in sh), None,
                 sum(r["accepted_hdr_errors_le2"] for r in sh)))
    rows.append(("TRUE frames, 80 bps", 72, sum(r >= 0.3 for r in true_r), sum(r >= 0.7 for r in true_r), sum(h <= 2 for h in true_he)))
    ax[1].axis("off")
    cell = [[a, f"{b}", f"{c}", "-" if d is None else f"{d}", f"{e}"] for a, b, c, d, e in rows]
    tb = ax[1].table(cellText=cell, colLabels=["data", "slots", "r >= 0.3", "r >= 0.7", "hdr errors <= 2"],
                     loc="center", cellLoc="right", colLoc="right", colWidths=[0.40, 0.10, 0.13, 0.13, 0.20])
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.5)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(GRID)
        if i == 0:
            c.set_text_props(weight="bold")
        if i == len(cell):
            c.set_text_props(weight="bold")
    ax[1].set_title("(b) accepted frames per acceptance rule (everything above the last row is a false detection)",
                    loc="left", fontsize=9)
    fig.tight_layout()
    save(fig, "figC_controls")
    return rows


# ------------------------------------------------------------------ D
def pick_bits(T, rng):
    k80 = np.isin(T["run"], ["80a", "80b", "80c"])
    idx = np.arange(len(T["run"]))
    pick = []
    c1 = idx[k80 & (T["sent"] == 1) & (T["margin"] > 0.3)]; pick.append(("random correct 1", rng.choice(c1)))
    c0 = idx[k80 & (T["sent"] == 0) & (T["prev"] == 0) & (T["margin"] > 0.3)]; pick.append(("random correct 0", rng.choice(c0)))
    s1 = idx[k80 & (T["sent"] == 1)]; o = s1[np.argsort(T["margin"][s1])]
    pick += [("lowest-margin sent 1", o[0]), ("2nd lowest-margin sent 1", o[1])]
    t10 = idx[k80 & (T["sent"] == 0) & (T["prev"] == 1)]; o = t10[np.argsort(T["margin"][t10])]
    pick += [("lowest-margin 1 -> 0", o[0]), ("2nd lowest-margin 1 -> 0", o[1])]
    pick.append(("random 1 -> 0", rng.choice(t10)))
    fb = idx[k80 & (T["n_inner"] == 0)]; o = fb[np.argsort(T["margin"][fb])]
    pick.append(("lowest-margin edge-only (fallback) bit", o[0]))
    return pick


def fig_bits(T):
    rng = np.random.default_rng(7)
    pl = SC.payload(); full = np.concatenate([BD.HDR, pl, np.zeros(16)])
    picks = pick_bits(T, rng)
    cache = {}
    fig = plt.figure(figsize=(16, 22))
    gs = fig.add_gridspec(8, 2, height_ratios=[1.25, 1] * 4, hspace=0.55, wspace=0.12)
    for n, (label, i) in enumerate(picks):
        key = str(T["run"][i]); run = RUNS80[key]
        if run not in cache:
            t, X, m, lead, res = load_run(run)
            base = {nd: BD.off_baseline(X[:, nd], lead) for nd in res["nodes"]}
            cache[run] = (t, X, m, lead, res, base)
        t, X, m, lead, res, base = cache[run]
        mm, bit = int(T["m"][i]), int(T["bit"][i]); jb = 27 + bit
        Tb_ = res["Tb"] * (1 + res["eps"])
        t_b = res["tau"] + (mm * 170 + jb) * Tb_
        fr = next(f for f in res["frames"] if f["m"] == mm)
        r0, c0 = (n // 2) * 2, n % 2
        a1 = fig.add_subplot(gs[r0, c0]); a2 = fig.add_subplot(gs[r0 + 1, c0], sharex=a1)
        lo, hi = t_b - 2 * Tb_, t_b + 3 * Tb_
        w = (t >= lo - 0.004) & (t <= hi + 0.004)
        x_ms = (t[w] - t_b) * 1e3
        for a in (a1, a2):
            for kk in range(-2, 4):
                e = kk * Tb_ * 1e3
                a.axvline(e, color=INK2, lw=0.8)
                a.axvspan(e - 2, e + 2, color=GRID, alpha=0.55, lw=0)
            a.set_xlim(-2 * Tb_ * 1e3 - 1, 3 * Tb_ * 1e3 + 1)
        for nd in res["nodes"]:
            b = base[nd][0]
            a1.plot(x_ms, (X[w, nd] - b[w]), "o-", ms=4, lw=0.7, color=NODE_C[nd], label=f"Y{nd} - OFF baseline")
            a2.plot(x_ms, res["z"][nd][w], "o", ms=3, color=NODE_C[nd], alpha=0.8)
        a1.set_yscale("symlog", linthresh=20); a1.axhline(0, color=INK, lw=0.6)
        sig = BD.off_baseline(X[:, res["nodes"][0]], lead)[2]
        a1.axhspan(-3 * sig, 3 * sig, color="#1baf7a", alpha=0.12, lw=0)
        a2.plot(x_ms, res["f"][w], "s", ms=5, mfc="none", mec=INK, label="combined = max of z")
        a2.axhline(fr["thr"], color=INK, ls="--", lw=1, label=f"frame threshold {fr['thr']:.2f}")
        for kk in range(-2, 3):
            jj = jb + kk
            if 0 <= jj < 170 and not np.isnan(fr["v"][jj]):
                xs = np.array([kk, kk + 1]) * Tb_ * 1e3
                a2.plot(xs, [fr["v"][jj]] * 2, color=INK, lw=2.2, alpha=0.6)
                dec = int(fr["v"][jj] > fr["thr"])
                ok = dec == int(full[jj])
                a1.text((kk + 0.5) * Tb_ * 1e3, 1.02, f"sent {int(full[jj])}", transform=a1.get_xaxis_transform(), ha="center",
                        fontsize=8, color=INK, weight="bold" if kk == 0 else "normal")
                a2.text((kk + 0.5) * Tb_ * 1e3, 0.92, f"read {dec}" + ("" if ok else " x"), transform=a2.get_xaxis_transform(),
                        ha="center", fontsize=8, color=INK if ok else "#e34948", weight="bold" if kk == 0 else "normal")
        mg = T["margin"][i]
        a1.set_title(f"{label}: run {run[-6:]}, frame {T['frame_k'][i]}, payload bit {bit}  (margin {mg:+.3f}, "
                     f"{T['n_inner'][i]} guard-passing sample{'s' if T['n_inner'][i] != 1 else ''})", loc="left", fontsize=9, pad=16)
        a1.set_ylabel("raw delta - baseline\n[counts, symlog]"); a2.set_ylabel("z-scored absdev")
        a2.set_xlabel("time from the bit's edge (host time of the sample - inferred bit start) [ms]")
        if n == 0:
            a1.legend(loc="lower left", fontsize=7, ncol=3); a2.legend(loc="lower left", fontsize=7)
    fig.suptitle("D. Bit-level diagnostics (80 bps validation runs). Grey bands: +-2 ms guard around each inferred bit edge. "
                 "Green band: +-3 sigma OFF noise. Thick grey bars: per-bit decision value.", x=0.01, ha="left", y=0.905, fontsize=10)
    save(fig, "figD_bits")
    return [(lab, str(T["run"][i]), int(T["frame_k"][i]), int(T["bit"][i]), float(T["margin"][i])) for lab, i in picks]


# ------------------------------------------------------------------ E
def fig_sensitivity():
    S = J("sensitivity_results.json")["S1_variants"]
    names = [r["variant"] for r in S[RUNS80["80a"]]]
    err = {nm: sum(next(r for r in S[run] if r["variant"] == nm)["errors"] for run in RUNS80.values()) for nm in names}
    order = names
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(order))[::-1]
    v = [err[nm] for nm in order]
    cols = ["#2a78d6" if nm.startswith("frozen") else "#86b6ef" for nm in order]
    ax.barh(y, np.maximum(v, 0.0), color=cols)
    for yy, vv in zip(y, v):
        ax.text(vv + 1, yy, f"{vv}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("payload bit errors in 9,144 bits (three 80 bps runs pooled)")
    ax.set_title("E. Sensitivity (informational, not used for any reported number): one frozen rule changed at a time",
                 loc="left", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "figE_sensitivity")


def main():
    T = dict(np.load(os.path.join(HERE, "bits_table.npz")))
    fig_mechanism(T); fig_timing(); rows = fig_controls(); picks = fig_bits(T); fig_sensitivity()
    json.dump(dict(control_table=rows, figD_bits=picks), open(os.path.join(HERE, "figures_index.json"), "w"), indent=1)
    print("figD bits:"); [print("  ", p) for p in picks]


if __name__ == "__main__":
    main()
