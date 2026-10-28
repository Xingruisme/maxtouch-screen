#!/usr/bin/env python3
"""
Regenerates every table and figure of the 2026-10-22 stent -> maXTouch daily summary from the saved run folders.
Read-only on the run folders: nothing in data_maxtouch/<run>/ is written; all output goes next to this script.

   ~/.venvs/galvanic/bin/python make_daily_summary.py

Sources per run folder (data_maxtouch/stent_*_20261022_*):
  BER runs   meta.json, capture.npz, result.json (original decoder), result_v3.json, result_v4.json, decoded_v3.json
  ON/OFF     meta.json, onoff.csv, onoff_stats.csv
  Probes     meta.json, capture.npz, result.json. The node x carrier response matrix is not stored in result.json, so
             it is recomputed here from capture.npz with maxtouch_stent_rx.analyse_probe (current code: start time
             fixed to the transmitter's known start, +-3 samples). The saved result.json is reported next to it.
Statistics: BER = errors / payload bits of the decoded frames. 95 % bounds assume independent bit errors
(Bernoulli): zero errors -> one-sided upper bound 1 - 0.05^(1/n); otherwise two-sided Clopper-Pearson.
"""
from __future__ import annotations

import contextlib
import csv
import glob
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
ROOT = os.path.dirname(DATA)
sys.path.insert(0, ROOT)
import maxtouch_stent_rx as R                   # noqa: E402  (also loads cap_ook_rx as R.RX, stent_frames as R.TX)

RX, TX = R.RX, R.TX
DAY = "20261022"
FRAME_BITS, PAYLOAD_BITS = 170, 127
HDR_BITS = 27
V4_FROZEN, V3_FROZEN = "2026-10-22 19:16", "2026-10-22 19:01"

import matplotlib                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402
from matplotlib.lines import Line2D             # noqa: E402
from scipy import stats                         # noqa: E402

# ---------------------------------------------------------------- style
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", violet="#6250d6")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "lines.linewidth": 1.6, "legend.frameon": False,
})
DEC = {"Original": C["blue"], "v3": C["orange"], "v4": C["aqua"]}
CAR = {155692.0: C["blue"], 159600.0: C["orange"], 163600.0: C["aqua"]}


def save(fig, name):
    for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)


def ci(err, n):
    """(BER, lower, upper) with the 95 % rule in the module docstring"""
    if not n:
        return None, None, None
    if err == 0:
        return 0.0, 0.0, 1 - 0.05 ** (1 / n)
    lo = stats.beta.ppf(0.025, err, n - err + 1)
    hi = stats.beta.ppf(0.975, err + 1, n - err)
    return err / n, float(lo), float(hi)


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def stamp(run):
    s = run.split("_")[-1]
    return f"{s[:2]}:{s[2:4]}:{s[4:6]}"


# ---------------------------------------------------------------- BER runs
def ber_rows():
    rows = []
    for d in sorted(glob.glob(os.path.join(DATA, f"stent_ber_*_{DAY}_*")), key=lambda p: p.split("_")[-1]):
        run = os.path.basename(d)
        meta = jload(os.path.join(d, "meta.json"))
        o, v3, v4 = (jload(os.path.join(d, f)) for f in ("result.json", "result_v3.json", "result_v4.json"))
        if not (meta and o):
            continue
        a = meta["args"]
        samp = o.get("sampling")
        if not samp:
            z = np.load(os.path.join(d, "capture.npz"))
            samp = dict(arrival_rate_hz=float((len(z["t"]) - 1) / (z["t"][-1] - z["t"][0])), scan_period_ms=None)
        h = (v4 or {}).get("health", {})
        sent = meta.get("tx_frames") or 0
        r = dict(run=run, time=stamp(run), carrier_hz=float(a["carrier"]), bps=float(a["bps"]), frames_sent=sent,
                 payload_bits_sent=sent * PAYLOAD_BITS, fresh_rate_hz=samp["arrival_rate_hz"],
                 scan_period_ms=samp.get("scan_period_ms"),
                 samples_per_bit=(v4 or {}).get("samples_per_bit", samp["arrival_rate_hz"] / float(a["bps"])),
                 signal_fraction=h.get("signal_fraction"), signal_lost=bool(h.get("signal_lost")),
                 role_v4=(v4 or {}).get("role"), role_v3=(v3 or {}).get("role"),
                 orig_err=o["combined"]["errors"], orig_bits=o["combined"]["bits"],
                 orig_frames=o["combined"].get("frames_valid"),
                 orig_known_err=(o.get("known_timing") or {}).get("errors"), orig_known_bits=(o.get("known_timing") or {}).get("bits"))
        for tag, res in (("v3", v3), ("v4", v4)):
            if not res:
                continue
            c, s, k = res["combined_blind"], res["single_top_ranked"], res.get("combined_known_timing") or {}
            r.update({f"{tag}_err": c["errors"], f"{tag}_bits": c["bits"], f"{tag}_frames": c["frames_valid"],
                      f"{tag}_single_node": s["node"], f"{tag}_single_err": s["errors"], f"{tag}_single_bits": s["bits"],
                      f"{tag}_known_err": k.get("errors"), f"{tag}_known_bits": k.get("bits"),
                      f"{tag}_sent1_read0": c["errors_by_type"]["sent1_read0"],
                      f"{tag}_sent0_read1": c["errors_by_type"]["sent0_read1"], f"{tag}_nodes": ",".join(res["nodes"])})
        if v4:
            D4 = jload(os.path.join(d, "decoded_v4.json"))
            r["v4_frames_errorfree"] = sum(1 for f in D4["blind_frames"] if f["valid"] and f["errors"] == 0) if D4 else None
            r["v4_singles"] = [(s["node"], s["errors"], s["bits"]) for s in v4["single_all"]]
            r["v4_delay_ms"] = (v4.get("combined_known_timing") or {}).get("delay_ms")
            r["v4_clock_ppm"] = v4["combined_blind"].get("clock_ppm")
        rows.append(r)
    return rows


# ---------------------------------------------------------------- probes and ON/OFF
PROBE_CONFIG = {
    "stent_probe_trx_wide_20261022_104923": "Stent on top edge, 14-carrier coarse sweep (1 s blocks x 3)",
    "stent_probe_trx_fine_20261022_105543": "Stent on top edge, 50-500 kHz fine sweep (2.5 % steps, 0.5 s blocks x 2)",
    "stent_probe_trx_fine_20261022_155334": "Electrode on hand, finger on top edge, same fine sweep",
    "stent_probe_phantom_sweep_20261022_203302": "Stent on skin, finger on top edge (body-coupled), same fine sweep",
}
SWEEP_STYLE = {                                 # colour = configuration, same in every sweep panel
    "stent_probe_trx_wide_20261022_104923": ("Stent on glass", C["blue"]),
    "stent_probe_trx_fine_20261022_105543": ("Stent on glass", C["blue"]),
    "stent_probe_trx_fine_20261022_155334": ("Electrode on hand, finger on panel (15:53)", C["aqua"]),
    "stent_probe_phantom_sweep_20261022_203302": ("Stent on skin, finger on panel (20:33)", C["orange"]),
}


def probe_data():
    out = []
    for run, cfg in PROBE_CONFIG.items():
        d = os.path.join(DATA, run)
        meta, saved = jload(os.path.join(d, "meta.json")), jload(os.path.join(d, "result.json"))
        z = np.load(os.path.join(d, "capture.npz"))
        res, aux = quiet(R.analyse_probe, z["t"], z["X"], meta)
        out.append(dict(run=run, config=cfg, meta=meta, saved=saved, res=res, aux=aux))
    return out


def onoff_data():
    out = []
    for d in sorted(glob.glob(os.path.join(DATA, f"stent_onoff_*_{DAY}_*")), key=lambda p: p.split("_")[-1]):
        meta = jload(os.path.join(d, "meta.json"))
        st = list(csv.DictReader(open(os.path.join(d, "onoff_stats.csv"))))
        rows = list(csv.reader(open(os.path.join(d, "onoff.csv"))))
        head, data = rows[0], rows[1:]
        t = np.array([float(x[0]) for x in data]); ph = np.array([x[2] for x in data])
        rates = {p: float((ph == p).sum() / meta["phase_s"]) for p in ("off1", "on", "off2")}
        best = max(st, key=lambda s: float(s["std_ratio_on_off"]))
        out.append(dict(run=os.path.basename(d), meta=meta, stats=st, head=head, data=data, t=t, phase=ph, rates=rates,
                        best=best, detected=float(best["std_ratio_on_off"]) >= 10))
    return out


# ---------------------------------------------------------------- figures, part 1
def fig_sweep(probes):
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 7.4), gridspec_kw=dict(height_ratios=[1.4, 1], hspace=0.85))
    ax = axs[0]
    for p in probes[1:]:
        l, c = SWEEP_STYLE[p["run"]]
        car = np.array(p["aux"]["car"]) / 1e3
        E = p["aux"]["E"].max(axis=0)
        thr = p["res"]["noise_floor_db"]["threshold"]
        ax.semilogx(car, E, "o-", ms=3.0, lw=1.2, color=c, label=f"{l} (threshold {thr:.1f} dB, dashed)")
        ax.axhline(thr, color=c, lw=0.9, ls="--")
    lines = []                                  # peak values as one text block (point labels collided)
    for p in probes[1:]:
        l, c = SWEEP_STYLE[p["run"]]
        cr = np.array(p["aux"]["car"]); Ep = p["aux"]["E"].max(axis=0)
        if not p["res"]["significant"]:
            lines.append((f"{l}: no carrier above threshold (best {Ep.max():+.1f} dB)", c)); continue
        picks = [int(np.argmin(np.abs(cr - f0))) for f0 in (155692.5, 159600, 163600)]
        for j in picks:
            ax.plot(cr[j] / 1e3, Ep[j], "o", ms=7, mfc="none", mec=c, mew=1.1)
        lines.append((f"{l}: " + ", ".join(f"{cr[j] / 1e3:.1f} kHz {Ep[j]:+.1f} dB" for j in picks)
                      + f"; peak {cr[int(np.argmax(Ep))] / 1e3:.1f} kHz {Ep.max():+.1f} dB", c))
    for k, (txt, c) in enumerate(lines):
        ax.text(0.01, 0.98 - k * 0.075, txt, transform=ax.transAxes, fontsize=7, color=c, va="top")
    ax.set_xlim(48, 520); ax.set_ylim(-2, 82)
    ax.set_xticks([50, 100, 155.7, 200, 300, 500]); ax.set_xticklabels(["50", "100", "155.7", "200", "300", "500"])
    ax.set_xlabel("Carrier frequency (kHz)"); ax.set_ylabel("Energy increase, best node (dB)")
    ax.set_title("(a) Fine sweep 50-500 kHz, 2.5 % steps, 64 nodes (Y0-Y63 on X0); circles = 155.7 / 159.6 / 163.6 kHz", loc="left", fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=1, fontsize=7.5)
    ax = axs[1]
    p = probes[0]
    car = np.array(p["aux"]["car"]) / 1e3; E = p["aux"]["E"].max(axis=0); thr = p["res"]["noise_floor_db"]["threshold"]
    ax.semilogx(car, E, "o", ms=5, color=C["blue"], label=f"Stent on glass, 14 discrete carriers (threshold {thr:.1f} dB, dashed)")
    ax.axhline(thr, color=C["blue"], lw=0.9, ls="--")
    for x, y in zip(car, E):
        if y > 30:
            ax.annotate(f"{x:g} kHz", (x, y), xytext=(6, -3), textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xlabel("Carrier frequency (kHz)"); ax.set_ylabel("Energy increase, best node (dB)")
    ax.set_title("(b) Coarse sweep 0.1 kHz - 4 MHz (markers only: carriers are not contiguous)", loc="left")
    ax.legend(loc="upper left", fontsize=7.5); ax.set_ylim(-2, 66)
    save(fig, "fig1_carrier_frequency_response")


def fig_heatmap(probes):
    NY = 16                                     # rows shown; the rest is summarised in the title
    fine = probes[1:]
    fig, axs = plt.subplots(1, len(fine), figsize=(3.7 * len(fine), 3.9), sharey=True, gridspec_kw=dict(wspace=0.14))
    vmax = max(p["aux"]["E"].max() for p in fine)
    rest = max(float(p["aux"]["E"][NY:].max()) for p in fine)
    for ax, p, ttl in zip(axs, fine, [f"({'abc'[k]}) {SWEEP_STYLE[p['run']][0]}" for k, p in enumerate(fine)]):
        E, car = p["aux"]["E"][:NY], np.array(p["aux"]["car"]) / 1e3
        im = ax.imshow(np.clip(E, 0, None), aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=vmax,
                       extent=(np.log10(car[0]), np.log10(car[-1]), -0.5, NY - 0.5))
        ticks = [50, 100, 155.7, 200, 300, 500]
        ax.set_xticks(np.log10(ticks)); ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_xlabel("Carrier frequency (kHz)"); ax.set_title(ttl, loc="left", fontsize=8.5); ax.grid(False)
        resp = [int(n[1:]) for n in p["res"]["best"]["responding_nodes"]]
        if resp:                                # this run's responding nodes at its best carrier
            ax.axhspan(min(resp) - 0.5, max(resp) + 0.5, fill=False, ec=C["orange"], lw=1.4)
            ax.text(np.log10(52), max(resp) + 0.9, f"Y{min(resp)}-Y{max(resp)}", color=C["orange"], fontsize=8, va="bottom")
    axs[0].set_ylabel("Sensor node (X line 0)")
    axs[0].set_yticks(range(NY)); axs[0].set_yticklabels([f"Y{v}" for v in range(NY)], fontsize=7)
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label("Energy increase (dB)")
    fig.suptitle(f"Spatial response per node and carrier, Y0-Y{NY - 1} shown (orange box = that run's responding nodes "
                 f"at its best carrier; Y{NY}-Y63 at most {rest:.1f} dB in any run)", x=0.07, ha="left", fontsize=9.5)
    save(fig, "fig2_spatial_response_heatmap")


def fig_onoff(oo):
    o = next(x for x in oo if x["run"].endswith("174954"))
    head, data, t, ph = o["head"], o["data"], o["t"], o["phase"]
    X = np.array([[float(v) for v in r[4:]] for r in data]); names = head[4:]
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 7.6), gridspec_kw=dict(height_ratios=[1.2, 0.9, 1], hspace=0.6))
    ax = axs[0]
    t_on, t_off = t[ph == "on"][0], t[ph == "off2"][0]
    ax.axvspan(t_on, t_off, color="#f0efec", zorder=0)
    i3 = names.index("Y3")
    ax.plot(t, X[:, i3], lw=0.5, color=C["blue"])
    ax.set_yscale("symlog", linthresh=20)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Delta, node Y3 (counts, symlog)")
    ax.set_title("(a) Raw delta of node Y3, 155.692 kHz precheck 17:49:54 (shaded = carrier ON)", loc="left")
    for x0, txt in ((t[0] + 0.3, "OFF"), (t_on + 0.3, "ON"), (t_off + 0.3, "OFF")):
        ax.text(x0, 3500, txt, fontsize=8, color=INK2, va="bottom")
    ax = axs[1]
    bins = np.logspace(0, 4, 49)
    for key, lab, c in (("off1", "OFF (first 10 s)", C["blue"]), ("on", "ON", C["orange"]), ("off2", "OFF (last 10 s)", C["aqua"])):
        v = X[(ph == key) & (t > t[ph == key][0] + 0.5), i3]
        ax.hist(1 + np.abs(v), bins=bins, histtype="step", lw=1.4, color=c, label=f"{lab}: std {v.std():.1f} counts, n = {len(v)}")
    ax.set_xscale("log")
    ax.set_xlabel("1 + |delta|, node Y3 (counts, log)"); ax.set_ylabel("Samples per bin")
    ax.set_title("(b) Distribution of Y3 magnitudes per phase (0.5 s after each switch excluded)", loc="left")
    ax.legend(loc="upper center", fontsize=7.5)
    ax = axs[2]
    st = {s["node"]: s for s in o["stats"]}
    xs = np.arange(len(names))
    for k, (key, lab, c) in enumerate((("std_off1", "OFF (first 10 s)", C["blue"]), ("std_on", "ON", C["orange"]),
                                       ("std_off2", "OFF (last 10 s)", C["aqua"]))):
        ax.bar(xs + (k - 1) * 0.27, [float(st[n][key]) for n in names], width=0.25, color=c, label=lab)
    ax.set_yscale("log"); ax.set_xticks(xs); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("Std of delta (counts, log)"); ax.set_title("(c) Per-node spread in each phase (onoff_stats.csv)", loc="left")
    ax.legend(ncol=3, loc="upper right")
    save(fig, "fig3_onoff_detection")


# ---------------------------------------------------------------- figures, part 2
def valid_rows(rows):
    return [r for r in rows if not r["signal_lost"] and r.get("v4_bits")]


def plot_ber_point(ax, x, err, n, color, marker, filled, dx=0.0):
    b, lo, hi = ci(err, n)
    if b is None:
        return
    mfc = color if filled else "white"
    if err == 0:
        ax.plot(x + dx, hi * 100, marker="v", ms=6.5, color=color, mfc=mfc, mec=color, ls="none")
    else:
        ax.errorbar(x + dx, b * 100, yerr=[[(b - lo) * 100], [(hi - b) * 100]], fmt=marker, ms=5.5, color=color,
                    mfc=mfc, mec=color, elinewidth=0.8, capsize=2)


def fig_ber_vs_rate(rows):
    vr = [r for r in valid_rows(rows) if r["carrier_hz"] == 155692.0]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    rep = {}                                    # repeats of the same bit rate are spread sideways
    for r in vr:
        rep[r["run"]] = sum(1 for q in vr if q["bps"] == r["bps"] and q["run"] < r["run"])
    for dec, dx in (("Original", -1.6), ("v3", 0.0), ("v4", 1.6)):
        for r in vr:
            key = {"Original": "orig", "v3": "v3", "v4": "v4"}[dec]
            role = "validation" if dec == "Original" else r.get(f"role_{dec}")
            plot_ber_point(ax, r["bps"], r[f"{key}_err"], r[f"{key}_bits"], DEC[dec], "o", role == "validation", dx + 0.45 * rep[r["run"]])
    ax.set_yscale("log"); ax.set_ylim(5e-3, 40)
    ax.set_xlabel("Physical bit rate (bps)"); ax.set_ylabel("Payload BER, combined blind (%)")
    ax.set_title("BER vs bit rate at 155.7 kHz (Y0-Y27 read; 3,048 payload bits per run)", loc="left")
    h = [Line2D([], [], color=DEC[d], marker="o", ls="none", label=d) for d in DEC]
    h += [Line2D([], [], color=INK2, marker="o", ls="none", label="validation (post-freeze)"),
          Line2D([], [], color=INK2, marker="o", mfc="white", ls="none", label="exploratory (development)"),
          Line2D([], [], color=INK2, marker="v", ls="none", label="0 errors: 95 % upper bound")]
    ax.legend(handles=h, loc="upper left", ncol=2)
    ax.text(0.99, 0.02, "Error bars: Clopper-Pearson 95 %. Original decoder has no tuning, shown as filled.",
            transform=ax.transAxes, ha="right", fontsize=7, color=INK2)
    save(fig, "fig4_ber_vs_bitrate")


def fig_decoders(rows):
    pick = ["stent_ber_c156_68bps_20261022_181556", "stent_ber_val4_68bps_20261022_192559",
            "stent_ber_val4_80bps_20261022_192702", "stent_ber_val4b_80bps_20261022_195008",
            "stent_ber_val4b_80bps_20261022_195109", "stent_ber_val4b_80bps_20261022_200327"]
    rr = [next(r for r in rows if r["run"] == p) for p in pick]
    series = [("Original combined", "orig_err", C["blue"]), ("v3 combined", "v3_err", C["orange"]),
              ("v4 single node (top-ranked)", "v4_single_err", C["yellow"]), ("v4 combined", "v4_err", C["aqua"])]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    xs = np.arange(len(rr))
    for k, (lab, key, c) in enumerate(series):
        vals = [r[key] for r in rr]
        ax.bar(xs + (k - 1.5) * 0.2, vals, width=0.19, color=c, label=lab)
        for x, v in zip(xs + (k - 1.5) * 0.2, vals):
            ax.text(x, max(v, 0.6) * 1.12, f"{v}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_yscale("symlog", linthresh=1); ax.set_ylim(0, 2000)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['bps']:g} bps\n{r['carrier_hz']/1e3:.1f} kHz\n{r['time']}\n{r['role_v4']}" for r in rr], fontsize=7)
    ax.set_ylabel("Payload bit errors (of 3,048)")
    ax.set_title("Decoder comparison on identical recordings (68 and 80 bps)", loc="left")
    ax.legend(ncol=2, loc="upper left")
    save(fig, "fig5_decoder_comparison")


def fig_carriers(rows):
    vr = valid_rows(rows)
    rates = [20, 34, 80]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, b in enumerate(rates):
        for j, (car, c) in enumerate(CAR.items()):
            rs = [r for r in vr if r["bps"] == b and r["carrier_hz"] == car]
            x = i + (j - 1) * 0.22
            if not rs:
                ax.text(x, 0.02, "not\ntested", ha="center", fontsize=6.5, color=INK2)
                continue
            for k, r in enumerate(rs):
                plot_ber_point(ax, x, r["v4_err"], r["v4_bits"], c, "o", r["role_v4"] == "validation", (k - (len(rs) - 1) / 2) * 0.06)
    ax.set_yscale("log"); ax.set_ylim(1e-2, 2)
    ax.set_xticks(range(len(rates))); ax.set_xticklabels([f"{b} bps" for b in rates])
    ax.set_ylabel("v4 payload BER, combined blind (%)")
    ax.set_title("Carrier comparison, frozen v4 (one marker per run)", loc="left")
    h = [Line2D([], [], color=c, marker="o", ls="none", label=f"{k/1e3:.1f} kHz") for k, c in CAR.items()]
    h += [Line2D([], [], color=INK2, marker="o", mfc="white", ls="none", label="exploratory"),
          Line2D([], [], color=INK2, marker="o", ls="none", label="validation"),
          Line2D([], [], color=INK2, marker="v", ls="none", label="0 errors: 95 % upper bound")]
    ax.legend(handles=h, loc="upper left", ncol=2, fontsize=7)
    save(fig, "fig6_carrier_comparison")


def fig_combining(rows):
    vr = [r for r in valid_rows(rows) if r["carrier_hz"] == 155692.0 and r["role_v4"] == "validation"]
    rates = sorted({r["bps"] for r in vr})
    cols = {"Y2": C["blue"], "Y3": C["orange"], "Y4": C["violet"], "combined": C["aqua"]}
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, b in enumerate(rates):
        rs = [r for r in vr if r["bps"] == b]
        pool = {k: [0, 0] for k in cols}
        for r in rs:
            for nd, e, n in r["v4_singles"]:
                if nd in pool:
                    pool[nd][0] += e; pool[nd][1] += n
            pool["combined"][0] += r["v4_err"]; pool["combined"][1] += r["v4_bits"]
        for k, (key, c) in enumerate(cols.items()):
            e, n = pool[key]
            plot_ber_point(ax, i, e, n, c, "s" if key != "combined" else "o", True, (k - 1.5) * 0.17)
        ax.text(i, 3.2, f"{len(rs)} run{'s' if len(rs) > 1 else ''}\n{pool['combined'][1]:,} bits", ha="center", fontsize=6.5, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(1e-2, 6)
    ax.set_xticks(range(len(rates))); ax.set_xticklabels([f"{b:g} bps" for b in rates])
    ax.set_ylabel("v4 payload BER (%)")
    ax.set_title("Single node vs combined (per-sample max over Y2-Y4), frozen v4, 155.7 kHz, validation runs pooled per rate",
                 loc="left", fontsize=9)
    h = [Line2D([], [], color=c, marker="s" if k != "combined" else "o", ls="none",
                label=f"single node {k}" if k != "combined" else "combined Y2-Y4") for k, c in cols.items()]
    h += [Line2D([], [], color=INK2, marker="v", ls="none", label="0 errors: 95 % upper bound")]
    ax.legend(handles=h, loc="lower right", ncol=3, fontsize=7)
    save(fig, "fig7_multinode_combining")


def fig_timing(rows):
    """(a) raw samples vs the v3 interpolated grid around a v3 error, (b) edge-averaged step response,
    (c)/(d) a strong but slowly varying 1 bit that the change-based feature missed"""
    fig = plt.figure(figsize=(7.4, 8.6))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 0.9, 0.8, 0.8], hspace=0.62)
    # (a) development run 34 bps: first v3 combined error
    run = "stent_ber_c156_34bps_20261022_180957"
    d = os.path.join(DATA, run)
    meta = jload(os.path.join(d, "meta.json")); z = np.load(os.path.join(d, "capture.npz")); t, X = z["t"], z["X"].astype(float)
    v4 = jload(os.path.join(d, "result_v4.json")); D3 = jload(os.path.join(d, "decoded_v3.json"))
    node = int(v4["nodes"][0][1:])
    G, T, t0, _, _ = quiet(R.regular, t, z["X"])
    bps = float(meta["args"]["bps"]); Tb = 1 / bps; k = Tb / T
    truth = np.array(D3["sent_payload"])
    fr = next(f for f in D3["blind_frames"] if f["valid"] and np.any(np.array(f["got"]) != truth))
    b = int(np.where(np.array(fr["got"]) != truth)[0][0])
    delay = v4["combined_known_timing"]["delay_ms"] / 1e3; ppm = v4["combined_known_timing"]["clock_ppm"] * 1e-6
    full = np.array(TX.frame_bits(PAYLOAD_BITS, 7)[0] * meta["tx_frames"])
    jg = fr["m"] * FRAME_BITS + HDR_BITS + b
    edge = lambda j: meta["t_first"] + delay + j * Tb * (1 + ppm)
    tc = edge(jg)
    ax = fig.add_subplot(gs[0])
    for j in range(jg - 3, jg + 4):
        if full[j]:
            ax.axvspan((edge(j) - tc) * 1e3, (edge(j + 1) - tc) * 1e3, color="#f0efec", zorder=0)
    w = (t > tc - 4 * Tb) & (t < tc + 4 * Tb)
    ax.plot((t[w] - tc) * 1e3, np.abs(X[w, node]), "o", ms=4.5, color=C["blue"], label="raw samples (own timestamp)")
    tg = t0 + np.arange(len(G)) * T
    wg = (tg > tc - 4 * Tb) & (tg < tc + 4 * Tb)
    ax.plot((tg[wg] - tc) * 1e3, np.abs(G[wg, node]), "x", ms=5, color=C["orange"], label="v3 interpolated grid values")
    s0 = fr["s0"]; j0 = HDR_BITS + b
    wins = []
    for jj in (j0, j0 + 1):
        a_ = t0 + (s0 + jj * k) * T
        wins.append(a_)
        ax.axvline((a_ - tc) * 1e3, color=C["orange"], lw=1.0, ls="--")
    inw = (tg >= wins[0]) & (tg < wins[1])
    if inw.any():                               # the leaked (interpolated) values inside the 0-bit window
        jmax = np.where(inw)[0][int(np.argmax(np.abs(G[inw, node])))]
        ax.annotate("interpolated ON/OFF mixture\ninside the sent-0 window", ((tg[jmax] - tc) * 1e3, abs(G[jmax, node])),
                    xytext=(18, -46), textcoords="offset points", fontsize=7, color=INK,
                    arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    ax.set_yscale("symlog", linthresh=10); ax.set_ylim(0, 1e4)
    ax.set_xlabel("Time relative to the error bit's true start (ms; transmitter edge + measured 11-13 ms delay)")
    ax.set_ylabel("|delta| node Y%d (counts)" % node)
    ax.set_title(f"(a) v3 error at 34 bps (sent 0 after a 1; dev run {stamp(run)}): shaded = sent 1, "
                 "dashed = v3 bit window", loc="left", fontsize=8.5)
    ax.legend(loc="upper right", fontsize=7)
    # (b) edge-averaged step response, all edges of the same run
    ax = fig.add_subplot(gs[1])
    y = np.abs(X[:, node])
    for kind, c in (("off", C["blue"]), ("on", C["orange"])):
        js = [j for j in range(2, len(full) - 3)
              if (kind == "off" and full[j - 1] == 1 and full[j - 2] == 1 and full[j] == 0 and full[j + 1] == 0)
              or (kind == "on" and full[j - 1] == 0 and full[j - 2] == 0 and full[j] == 1 and full[j + 1] == 1)]
        ed = meta["t_first"] + np.array(js) * Tb * (1 + ppm)
        rel = (t[None, :] - ed[:, None]) * 1e3
        bins = np.arange(-20, 41, 1.0); m_ = []
        for b0 in bins:
            mm = (rel >= b0) & (rel < b0 + 1)
            m_.append(y[np.where(mm)[1]].mean() if mm.any() else np.nan)
        m_ = np.array(m_); lvl = np.nanmedian(m_[bins > 20]) if kind == "on" else np.nanmedian(m_[bins < -5])
        ax.plot(bins + 0.5, 100 * m_ / lvl, color=c, label=f"carrier {kind.upper()} edge ({len(js)} edges)")
    ax.axvspan(delay * 1e3 - 1, delay * 1e3 + 1, color=GRID, zorder=0)
    ax.set_xlabel("Time since transmitted edge, host arrival time (ms)"); ax.set_ylabel("Mean |delta| (% of ON level)")
    ax.set_title("(b) Edge-averaged response: sharp step after a fixed delay, no multi-scan tail", loc="left", fontsize=8.5)
    ax.legend(loc="center right", fontsize=7)
    # (c)/(d) validation run 20 bps at 159.6 kHz: v3 'fade' error (sent 1 read 0)
    run2 = "stent_ber_c160_20bps_20261022_190624"
    d2 = os.path.join(DATA, run2)
    m2 = jload(os.path.join(d2, "meta.json")); z2 = np.load(os.path.join(d2, "capture.npz")); t2, X2 = z2["t"], z2["X"].astype(float)
    v42 = jload(os.path.join(d2, "result_v4.json")); D32 = jload(os.path.join(d2, "decoded_v3.json"))
    n2 = int(v42["nodes"][0][1:])
    G2, T2, t02, _, _ = quiet(R.regular, t2, z2["X"])
    tr2 = np.array(D32["sent_payload"])
    f2 = next(f for f in D32["blind_frames"] if f["valid"] and np.any((np.array(f["got"]) != tr2) & (tr2 == 1)))
    b2 = int(np.where((np.array(f2["got"]) != tr2) & (tr2 == 1))[0][0])
    Tb2 = 1 / float(m2["args"]["bps"]); dl2 = v42["combined_known_timing"]["delay_ms"] / 1e3
    pp2 = v42["combined_known_timing"]["clock_ppm"] * 1e-6
    full2 = np.array(TX.frame_bits(PAYLOAD_BITS, 7)[0] * m2["tx_frames"])
    jg2 = f2["m"] * FRAME_BITS + HDR_BITS + b2
    e2 = lambda j: m2["t_first"] + dl2 + j * Tb2 * (1 + pp2)
    tc2 = e2(jg2)
    tg2 = t02 + np.arange(len(G2)) * T2
    le = RX.features(G2[:, n2])["logenergy"]; ad = RX.features(G2[:, n2])["absdev"]
    for row, (yy, lab) in enumerate(((None, "raw"), ("feat", "features"))):
        ax = fig.add_subplot(gs[2 + row])
        for j in range(jg2 - 3, jg2 + 4):
            if full2[j]:
                ax.axvspan((e2(j) - tc2) * 1e3, (e2(j + 1) - tc2) * 1e3, color="#f0efec", zorder=0)
        ax.axvspan(0, Tb2 * 1e3, fill=False, ec=C["magenta"], lw=1.2)
        if yy is None:
            w2 = (t2 > tc2 - 3 * Tb2) & (t2 < tc2 + 4 * Tb2)
            ax.plot((t2[w2] - tc2) * 1e3, X2[w2, n2], "o-", ms=3.5, lw=0.8, color=C["blue"])
            ax.set_ylabel(f"Raw delta Y{n2} (counts)")
            ax.set_title(f"(c) v3 'fade', 20 bps, 159.6 kHz (validation run {stamp(run2)}): magenta box = sent 1 read as 0",
                         loc="left", fontsize=8.5)
        else:
            wg2 = (tg2 > tc2 - 3 * Tb2) & (tg2 < tc2 + 4 * Tb2)
            ax.plot((tg2[wg2] - tc2) * 1e3, R.zscore(le)[wg2], "o-", ms=3, lw=0.8, color=C["orange"], label="logenergy (change-based, v3 at >= 4 samples/bit)")
            ax.plot((tg2[wg2] - tc2) * 1e3, R.zscore(ad)[wg2], "s-", ms=3, lw=0.8, color=C["aqua"], label="absdev (distance from baseline)")
            ax.set_ylabel("Feature (z-score)")
            ax.set_xlabel("Time relative to the bit's true start (ms)")
            ax.set_title("(d) The change-based feature drops inside the strong bit; the level-based one does not",
                         loc="left", fontsize=8.5, pad=20)
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=7)
    save(fig, "fig8_timing_and_decoder_analysis")
    return dict(a_run=run, a_frame=fr["m"], a_bit=b, a_node=node, c_run=run2, c_frame=f2["m"], c_bit=b2, c_node=n2)


def fig_throughput(rows):
    vr = [r for r in valid_rows(rows) if r["carrier_hz"] == 155692.0]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs = np.array([20, 100])
    ax.plot(xs, xs * PAYLOAD_BITS / FRAME_BITS, color=INK2, lw=0.9, ls="--", label="ideal payload rate (127/170 of physical)")
    for r in vr:
        fs = r["v4_frames"] / r["frames_sent"]
        thr = r["bps"] * PAYLOAD_BITS / FRAME_BITS * fs
        good = thr * (1 - r["v4_err"] / r["v4_bits"])
        ef = r["bps"] * PAYLOAD_BITS / FRAME_BITS * r["v4_frames_errorfree"] / r["frames_sent"]
        filled = r["role_v4"] == "validation"
        ax.plot(r["bps"], good, "o", ms=5.5, color=C["aqua"], mfc=C["aqua"] if filled else "white")
        ax.plot(r["bps"] + 0.8, ef, "s", ms=5, color=C["orange"], mfc=C["orange"] if filled else "white")
    ax.set_xlabel("Physical bit rate (bps)"); ax.set_ylabel("Payload throughput (bit/s)")
    ax.set_title("Effective payload throughput, frozen v4, 155.7 kHz (missing frames count as zero)", loc="left")
    ax.legend(handles=[Line2D([], [], color=INK2, ls="--", label="ideal payload rate (127/170 x physical)"),
                       Line2D([], [], color=C["aqua"], marker="o", ls="none", label="correct payload bits/s (bit errors allowed)"),
                       Line2D([], [], color=C["orange"], marker="s", ls="none", label="payload in error-free frames only"),
                       Line2D([], [], color=INK2, marker="o", mfc="white", ls="none", label="open marker = exploratory run")],
              loc="upper left", fontsize=7.5)
    save(fig, "fig9_effective_throughput")


def fig_methods(rows):
    """Figure 10: how the decoders differ (a), the same transmitted frame through each (b), BER vs rate (c)."""
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(7.4, 10.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.35, 1.0], hspace=0.42)
    # (a) the design choices, cells that caused artificial errors in orange, v4's changes in aqua
    ax = fig.add_subplot(gs[0]); ax.axis("off")
    cols = ["Sample timing", "Bit window", "Feature per sample", "Node combining", "Rules fixed"]
    cells = [["Uniform grid,\nlinear interpolation", "Fractional grid\nsamples", "Chosen per node after\ndecoding; combined:\nabsdev", "Weighted mean of\nz-scores, 3 nodes", "Before the runs\n(recording code)"],
             ["Uniform grid,\nlinear interpolation", "Fractional grid\nsamples", "logenergy (change-\nbased) if >= 4 samples\nper bit, else absdev", "Max of z-scores\nper sample, 3 nodes", "Frozen 19:01"],
             ["Raw samples at their\nown timestamps", "Bit time, samples\n< 2 ms from an edge\nskipped", "absdev: distance from\nthe OFF baseline\n(quiet samples)", "Max of z-scores\nper sample, 3 nodes", "Frozen 19:16"]]
    bad = {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)}
    good = {(2, 0), (2, 1), (2, 2), (1, 3), (2, 3)}
    tb = ax.table(cellText=cells, rowLabels=["Original", "v3", "v4"], colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(7.2); tb.scale(1, 3.4)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID); cell.set_linewidth(0.8)
        if r == 0:
            cell.set_text_props(weight="bold", color=INK); cell.set_facecolor("#f0efec"); cell.set_height(cell.get_height() * 0.55)
        elif c == -1:
            cell.set_text_props(weight="bold", color=DEC[["Original", "v3", "v4"][r - 1]])
        elif (r - 1, c) in bad:
            cell.set_facecolor("#fbe3d6")
        elif (r - 1, c) in good:
            cell.set_facecolor("#d6f1e6")
    ax.set_title("(a) How the decoders differ (orange = source of artificial errors, green = what removed them)", loc="left", fontsize=9)
    # (b) one transmitted frame of the 80 bps validation run 19:27, through each decoder
    run = "stent_ber_val4_80bps_20261022_192702"
    d = os.path.join(DATA, run)
    meta = jload(os.path.join(d, "meta.json")); z = np.load(os.path.join(d, "capture.npz"))
    t, X = z["t"].astype(float), z["X"].astype(float)
    v4 = jload(os.path.join(d, "result_v4.json"))
    G, T, t0, _, _ = quiet(R.regular, t, z["X"])
    Tb = 1 / float(meta["args"]["bps"]); P = FRAME_BITS * Tb
    delay = v4["combined_known_timing"]["delay_ms"] / 1e3; eps = v4["combined_blind"]["clock_ppm"] * 1e-6
    t_start = lambda k: meta["t_first"] + delay + k * P * (1 + eps)
    per = {}
    for name, f, grid in (("Original", "decoded.json", True), ("v3", "decoded_v3.json", True), ("v4", "decoded_v4.json", False)):
        D = jload(os.path.join(d, f))
        for fr in D["blind_frames"]:
            if not fr["valid"]:
                continue
            ts = t0 + fr["s0"] * T if grid else fr["s0"]
            k = int(round((ts - t_start(0)) / (P * (1 + eps))))
            per.setdefault(k, {})[name] = np.array(fr["got"])
        payload = np.array(D["sent_payload"])
    ks = [k for k, v in per.items() if len(v) == 3]
    med = np.median([int(np.sum(per[k]["Original"] != payload)) for k in ks])
    k = min(ks, key=lambda k: (abs(int(np.sum(per[k]["Original"] != payload)) - med), k))   # a typical frame
    b0, nbits = 40, 40
    ax = fig.add_subplot(gs[1])
    tp = t_start(k) + HDR_BITS * Tb
    pre = t < meta["t_prompt"]
    nodes = [int(n[1:]) for n in v4["nodes"]]
    dev = np.max([np.abs(X[:, i] - np.median(X[pre, i])) for i in nodes], axis=0)
    xb = (t - tp) / Tb
    w = (xb >= b0) & (xb < b0 + nbits)
    for b in range(b0, b0 + nbits):
        if payload[b]:
            ax.axvspan(b, b + 1, ymin=0.62, ymax=1.0, color="#f0efec", zorder=0)
    y_raw = 4.6 + 1.6 * np.log10(1 + dev[w]) / 4          # raw |delta| strip, log-scaled into the top band
    ax.plot(xb[w], y_raw, "o", ms=2.8, color=INK2, zorder=3)
    labels = [("Sent", payload, INK)] + [(n, per[k][n], DEC[n]) for n in ("Original", "v3", "v4")]
    for row, (name, bits, c) in enumerate(labels):
        y = 3 - row
        for b in range(b0, b0 + nbits):
            one = bits[b] == 1
            wrong = name != "Sent" and bits[b] != payload[b]
            ax.add_patch(Rectangle((b + 0.06, y + 0.12), 0.88, 0.76, facecolor=c if one else "white",
                                   edgecolor=INK if wrong else (c if one else GRID), lw=2.2 if wrong else 0.8, zorder=2))
            if wrong:                           # ink, not a series colour: visible on every row
                ax.plot(b + 0.5, y + 0.5, marker="x", ms=6, mew=1.8, color="white" if one else INK, zorder=4)
        e_w = 0 if name == "Sent" else int(np.sum(bits[b0:b0 + nbits] != payload[b0:b0 + nbits]))
        e_f = None if name == "Sent" else int(np.sum(bits != payload))
        ax.text(b0 - 0.6, y + 0.5, name, ha="right", va="center", fontsize=8, color=c, weight="bold")
        if e_f is not None:
            ax.text(b0 + nbits + 0.6, y + 0.5, f"{e_w} here\n{e_f}/127 frame", ha="left", va="center", fontsize=7, color=INK)
    ax.text(b0 - 0.6, 5.4, "raw |delta|\nY2-Y4 (log)", ha="right", va="center", fontsize=7, color=INK2)
    ax.set_xlim(b0 - 7.5, b0 + nbits + 6.5); ax.set_ylim(-0.1, 6.4)
    ax.set_yticks([]); ax.grid(False)
    for s in ("left",):
        ax.spines[s].set_visible(False)
    ax.set_xticks(range(b0, b0 + nbits + 1, 5)); ax.set_xlabel("Payload bit index")
    ax.set_title(f"(b) Same transmitted frame through each decoder: 80 bps validation run {stamp(run)}, frame {k + 1} of 24, "
                 f"payload bits {b0}-{b0 + nbits - 1}\n(filled = decided 1; dark outline + x = error; shading = sent 1)", loc="left", fontsize=8.5)
    # (c) BER vs bit rate, validation runs only, pooled per rate
    ax = fig.add_subplot(gs[2])
    vr = [r for r in valid_rows(rows) if r["carrier_hz"] == 155692.0 and r["role_v4"] == "validation"]
    rates = sorted({r["bps"] for r in vr})
    for dec, key, dx in (("Original", "orig", -0.6), ("v3", "v3", 0.0), ("v4", "v4", 0.6)):
        for b in rates:
            e = sum(r[f"{key}_err"] for r in vr if r["bps"] == b); n = sum(r[f"{key}_bits"] for r in vr if r["bps"] == b)
            ber, lo, hi = ci(e, n)
            plot_ber_point(ax, b, e, n, DEC[dec], "o", True, dx)
    ax.set_yscale("log"); ax.set_ylim(1e-2, 40)
    ax.set_xlabel("Physical bit rate (bps)"); ax.set_ylabel("Payload BER (%)")
    ax.set_title("(c) Result: BER vs bit rate, 155.7 kHz, validation runs only (pooled per rate; v = 0 errors, 95 % upper bound)",
                 loc="left", fontsize=8.5)
    ax.legend(handles=[Line2D([], [], color=DEC[n], marker="o", ls="none", label=n) for n in DEC]
              + [Line2D([], [], color=INK2, marker="v", ls="none", label="0 errors: 95 % upper bound")], loc="upper left", ncol=2, fontsize=8)
    save(fig, "fig10_decoder_methods")
    return dict(run=run, frame=k + 1, bits=(b0, b0 + nbits - 1),
                errors_frame={n: int(np.sum(per[k][n] != payload)) for n in ("Original", "v3", "v4")})


# ---------------------------------------------------------------- workbook
def workbook(rows, probes, oo, timing):
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    F, FB, FH = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True), Font(name="Arial", size=10, bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="52514E")

    def sheet(name, header, data, notes=()):
        ws = wb.create_sheet(name)
        for j, h in enumerate(header, 1):
            c = ws.cell(1, j, h); c.font = FH; c.fill = head_fill; c.alignment = Alignment(wrap_text=True, vertical="top")
        for i, row in enumerate(data, 2):
            for j, v in enumerate(row, 1):
                c = ws.cell(i, j, v); c.font = F
        for j, h in enumerate(header, 1):
            ws.column_dimensions[get_column_letter(j)].width = max(10, min(46, len(str(h)) + 2))
        ws.freeze_panes = "B2"
        r0 = len(data) + 3
        for k, n in enumerate(notes):
            ws.cell(r0 + k, 1, n).font = Font(name="Arial", size=9, italic=True)
        return ws

    wb.remove(wb.active)
    # Inventory
    inv = []
    for p in probes:
        inv.append([p["run"], "frequency sweep", stamp(p["run"]), p["config"], "", "",
                    "response above threshold" if p["res"]["significant"] else "no response above threshold"])
    for o in oo:
        a = o["meta"]["args"]
        inv.append([o["run"], "ON/OFF", stamp(o["run"]), f"{a['ny']} nodes from Y{a['y0']}", float(a["carrier"]), "",
                    "carrier detected" if o["detected"] else "no response (transmitter fault, see report)"])
    for r in rows:
        inv.append([r["run"], "BER", r["time"], "stent on top edge, Y0-Y27", r["carrier_hz"], r["bps"],
                    "SIGNAL LOST - excluded" if r["signal_lost"] else f"valid; v4 role {r['role_v4']}"])
    sheet("Inventory", ["Run folder", "Type", "Time", "Configuration", "Carrier (Hz)", "Bit rate (bps)", "Status"], inv,
          ["Status 'SIGNAL LOST': top node >= 10 dB over the lead in < 95 % of 2 s windows (v4 health rule); kept in the inventory, excluded from BER comparisons."])
    # Frequency sweeps
    fs_rows = []
    for p in probes:
        car, E, names = p["aux"]["car"], p["aux"]["E"], p["aux"]["names"]
        thr = p["res"]["noise_floor_db"]["threshold"]
        for c_, f in enumerate(car):
            n = int(np.argmax(E[:, c_]))
            resp = [names[i] for i in np.where(E[:, c_] >= thr)[0]]
            fs_rows.append([p["run"], p["config"], float(f), round(float(E[n, c_]), 2), names[n], round(thr, 2),
                            len(resp), " ".join(resp[:12]) + (" ..." if len(resp) > 12 else ""), round(p["res"]["fs_hz"], 1)])
    sheet("Frequency sweeps", ["Run", "Configuration", "Carrier (Hz)", "Best-node energy increase (dB)", "Best node",
                               "Response threshold (dB)", "Nodes over threshold", "Responding nodes", "Analysis grid rate (samples/s)"],
          fs_rows, ["Recomputed from capture.npz with maxtouch_stent_rx.analyse_probe (start fixed to the known transmitter start).",
                    "Threshold = max(6 dB, median + 6 robust sigma) over all node x carrier pairs of that run.",
                    "Hand run: the saved result.json (older alignment) reported best 176.2 kHz +6.5 dB; recomputed best is listed here. Both are below threshold."])
    # ON/OFF
    on_rows = []
    for o in oo:
        b = o["best"]; a = o["meta"]["args"]
        on_rows.append([o["run"], stamp(o["run"]), float(a["carrier"]), a["ny"], round(o["rates"]["off1"], 1), round(o["rates"]["on"], 1),
                        round(o["rates"]["off2"], 1), b["node"], float(b["std_off1"]), float(b["std_on"]), float(b["std_off2"]),
                        None, "yes" if o["detected"] else "no"])
    ws = sheet("ON-OFF tests", ["Run", "Time", "Carrier (Hz)", "Nodes read", "Rate OFF1 (samples/s)", "Rate ON (samples/s)",
                                "Rate OFF2 (samples/s)", "Node with largest ON/OFF spread ratio", "Std OFF1 (counts)", "Std ON (counts)",
                                "Std OFF2 (counts)", "Spread ratio ON / mean OFF", "Carrier detected (ratio >= 10)"], on_rows,
               ["Spread ratio is a formula: std ON / ((std OFF1 + std OFF2) / 2). 16:10 run: AD9833 was not responding (wiring fault), see report."])
    for i in range(2, len(on_rows) + 2):
        ws.cell(i, 12, f"=J{i}/((I{i}+K{i})/2)").number_format = "0.0"
    # BER results
    ber_hdr = ["Run", "Time", "Carrier (Hz)", "Bit rate (bps)", "v4 role", "Signal present (fraction)", "Signal lost",
               "Fresh samples/s", "Samples per bit", "Frames sent", "v4 frames decoded", "v4 frame success", "v4 errors", "v4 payload bits decoded",
               "v4 BER", "v4 BER 95% lower", "v4 BER 95% upper", "v4 conservative BER (missing frames = 50 %)",
               "Payload throughput (bps)", "Correct payload bits/s", "v4 known-timing errors", "v4 measured delay (ms)", "v4 clock offset (ppm)",
               "v4 error-free frames", "Error-free-frame payload throughput (bps)"]
    ber_data = []
    for r in rows:
        if not r.get("v4_bits"):
            continue
        ber_data.append([r["run"], r["time"], r["carrier_hz"], r["bps"], r["role_v4"], r["signal_fraction"], "yes" if r["signal_lost"] else "no",
                         round(r["fresh_rate_hz"], 1), round(r["samples_per_bit"], 2), r["frames_sent"], r["v4_frames"], None, r["v4_err"],
                         r["v4_bits"], None, None, None, None, None, None, r["v4_known_err"], r["v4_delay_ms"], r["v4_clock_ppm"],
                         r["v4_frames_errorfree"], None])
    ws = sheet("BER results", ber_hdr, ber_data,
               ["BER on decoded payload = errors / payload bits of decoded frames; frame success shown separately.",
                "95 % bounds (formulas): 0 errors -> lower 0, upper 1-0.05^(1/n) (one-sided); else Clopper-Pearson via BETAINV. Assumes independent bit errors.",
                "Conservative BER counts every undecoded frame's 127 payload bits as 50 % errors.",
                "Payload throughput = physical rate x 127/170 x frame success; correct payload bits/s additionally x (1 - BER).",
                "Error-free-frame payload throughput = physical rate x 127/170 x (frames with 0 payload errors / frames sent), from decoded_v4.json.",
                "Signal-lost runs (Signal lost = yes) are listed for completeness and must not be used as link performance."])
    for i in range(2, len(ber_data) + 2):
        ws.cell(i, 12, f"=IF(J{i}>0,K{i}/J{i},\"\")").number_format = "0.0%"
        ws.cell(i, 15, f"=IF(N{i}>0,M{i}/N{i},\"\")").number_format = "0.000%"
        ws.cell(i, 16, f"=IF(N{i}=0,\"\",IF(M{i}=0,0,BETAINV(0.025,M{i},N{i}-M{i}+1)))").number_format = "0.000%"
        ws.cell(i, 17, f"=IF(N{i}=0,\"\",IF(M{i}=0,1-POWER(0.05,1/N{i}),BETAINV(0.975,M{i}+1,N{i}-M{i})))").number_format = "0.000%"
        ws.cell(i, 18, f"=IF(J{i}>0,(M{i}+0.5*(J{i}-K{i})*127)/(N{i}+(J{i}-K{i})*127),\"\")").number_format = "0.000%"
        ws.cell(i, 19, f"=IF(J{i}>0,D{i}*127/170*K{i}/J{i},\"\")").number_format = "0.0"
        ws.cell(i, 20, f"=IF(N{i}>0,S{i}*(1-M{i}/N{i}),\"\")").number_format = "0.0"
        ws.cell(i, 25, f"=IF(J{i}>0,D{i}*127/170*X{i}/J{i},\"\")").number_format = "0.0"
    # Decoder comparison (all runs with all three decoders)
    dc = []
    for r in rows:
        if not r.get("v4_bits"):
            continue
        dc.append([r["run"], r["carrier_hz"], r["bps"], "yes" if r["signal_lost"] else "no", r["role_v3"], r["role_v4"],
                   r["orig_err"], r["orig_bits"], r.get("orig_frames"), r.get("v3_err"), r.get("v3_bits"), r.get("v3_frames"),
                   r.get("v4_single_node"), r.get("v4_single_err"), r.get("v4_single_bits"), r["v4_err"], r["v4_bits"], r["v4_frames"],
                   None, None, None, None])
    ws = sheet("Decoder comparison", ["Run", "Carrier (Hz)", "Bit rate (bps)", "Signal lost", "v3 role", "v4 role",
                                      "Original errors", "Original bits", "Original frames", "v3 errors", "v3 bits", "v3 frames",
                                      "v4 single node", "v4 single errors", "v4 single bits", "v4 combined errors", "v4 combined bits",
                                      "v4 frames", "Original BER", "v3 BER", "v4 single BER", "v4 combined BER"], dc,
               ["All decoders applied to the same capture.npz. Original = maxtouch_stent_rx analyse_ber at recording time; v3/v4 = frozen offline decoders."])
    for i in range(2, len(dc) + 2):
        for col, (e, n) in zip((19, 20, 21, 22), (("G", "H"), ("J", "K"), ("N", "O"), ("P", "Q"))):
            ws.cell(i, col, f"=IF({n}{i}>0,{e}{i}/{n}{i},\"\")").number_format = "0.000%"
    # Carrier comparison
    cc = []
    for r in valid_rows(rows):
        if r["bps"] in (20, 34, 80):
            cc.append([r["bps"], r["carrier_hz"], r["run"], r["role_v4"], r["v4_err"], r["v4_bits"], r["v4_frames"], r["frames_sent"], None])
    cc.sort(key=lambda x: (x[0], x[1], x[2]))
    ws = sheet("Carrier comparison", ["Bit rate (bps)", "Carrier (Hz)", "Run", "v4 role", "v4 errors", "v4 payload bits",
                                      "Frames decoded", "Frames sent", "v4 BER"], cc,
               ["Tested combinations only. 159.6 kHz was not tested at 80 bps. All runs: same stent position and session (not physically independent setups)."])
    for i in range(2, len(cc) + 2):
        ws.cell(i, 9, f"=IF(F{i}>0,E{i}/F{i},\"\")").number_format = "0.000%"
    # Run metadata
    md = []
    for r in rows:
        meta = jload(os.path.join(DATA, r["run"], "meta.json")); a = meta["args"]
        md.append([r["run"], a.get("wave"), a.get("y0"), a.get("ny"), a.get("payload"), a.get("prbs"), meta.get("tx_frames"),
                   round(meta.get("recorded_s", 0), 1), meta.get("samples"), meta.get("hid_errors"),
                   r.get("scan_period_ms"), r.get("v4_nodes"), (meta.get("tx") or {}).get("line", "")])
    sheet("Run metadata", ["Run", "Waveform", "First node Y", "Nodes read", "Payload bits/frame", "PRBS order", "Frames sent",
                           "Recorded (s)", "Samples", "HID errors", "Scan period (ms)", "v4 combined nodes", "Transmitter id"], md,
          [f"Decoders: v3 frozen {V3_FROZEN} (sha 5cb4cecf1de6), v4 frozen {V4_FROZEN} (sha 59cb5e989bc9). Figures' example bits: {json.dumps(timing)}"])
    wb.calculation.fullCalcOnLoad = True
    out = os.path.join(HERE, "Experimental_Summary_20261022.xlsx")
    wb.save(out)
    return out, ber_data, dc, cc, on_rows


# ---------------------------------------------------------------- consistency checks and report tables
def check_and_tables(rows, probes, oo):
    lines, problems = [], []
    for p in probes:                             # recomputed best must match the saved result for the stent runs
        s, b = p["saved"], p["res"]["best"]
        same = abs(s["best"]["carrier_hz"] - b["carrier_hz"]) < 1 and abs(s["best"]["energy_db"] - b["energy_db"]) < 0.05
        lines.append(f"probe {p['run']}: saved best {s['best']['carrier_hz']:.0f} Hz {s['best']['energy_db']:+.1f} dB; "
                     f"recomputed {b['carrier_hz']:.0f} Hz {b['energy_db']:+.1f} dB, threshold {p['res']['noise_floor_db']['threshold']:.1f} dB"
                     f" ({'match' if same else 'differs'})")
        if "Stent" in p["config"] and not same:
            problems.append(p["run"])
    for r in rows:                              # every BER number used comes straight from the saved JSON
        v4 = jload(os.path.join(DATA, r["run"], "result_v4.json"))
        if v4 and (v4["combined_blind"]["errors"] != r["v4_err"] or v4["combined_blind"]["bits"] != r["v4_bits"]):
            problems.append(r["run"])
    return lines, problems


def main():
    rows = ber_rows()
    probes = probe_data()
    oo = onoff_data()
    fig_sweep(probes); fig_heatmap(probes); fig_onoff(oo)
    fig_ber_vs_rate(rows); fig_decoders(rows); fig_carriers(rows); fig_combining(rows)
    timing = fig_timing(rows); fig_throughput(rows); fig_methods(rows)
    out, ber_data, dc, cc, on_rows = workbook(rows, probes, oo, timing)
    lines, problems = check_and_tables(rows, probes, oo)
    summary = dict(ber_runs=[{k: v for k, v in r.items()} for r in rows],
                   probes=[dict(run=p["run"], config=p["config"], fs_hz=p["res"]["fs_hz"], threshold_db=p["res"]["noise_floor_db"]["threshold"],
                                best=p["res"]["best"], top=p["res"]["top"], saved_best=p["saved"]["best"] if p["saved"].get("best") else p["saved"])
                           for p in probes],
                   onoff=[dict(run=o["run"], carrier=o["meta"]["args"]["carrier"], ny=o["meta"]["args"]["ny"], rates=o["rates"],
                               best=o["best"], detected=o["detected"]) for o in oo],
                   timing_examples=timing, checks=lines, problems=problems)
    json.dump(summary, open(os.path.join(HERE, "summary_numbers.json"), "w"), indent=1, default=float)
    print("\n".join(lines))
    print("problems:", problems or "none")
    print("workbook:", out)


if __name__ == "__main__":
    main()
