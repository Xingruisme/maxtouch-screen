#!/usr/bin/env python3
"""
2-FSK vs 4-FSK through-body BER to the maXTouch tablet with the SAME trx frames (2026-11-20).

  2-FSK  1 state per symbol: state 1 = 159.585 kHz, state 0 = 180.556 kHz (panel null) - the existing FSK, sent by the
         unchanged stent_trx_ber path
  4-FSK  2 states per symbol on a ladder of 4 tones the panel answers at 4 different levels (decode_trx_mfsk.py):
         159.585 kHz 0 dB | 151.895 kHz -8.3 dB | 130.979 kHz -19.3 dB | 180.556 kHz null, Gray-mapped
         (1,1) (1,0) (0,0) (0,1)
The panel cannot measure frequency (each scan sees the carrier's energy at a random phase), so the tones are told
apart only by how strongly the panel answers them; see decode_trx_mfsk.py for the survey and the decoding rules.

Compared at the same BIT rate: --bitrates 20,40,60,80 -> 2-FSK at 20/40/60/80 baud, 4-FSK at 10/20/30/40 baud.
Both send the identical frames (trx_psk_link_v2.build_seq with phase number = the bit rate), recorded back to back.
Every recording goes through stent_trx_ber.one_run unchanged (upload, tablet set-up + finger-on calibration, lead,
sgo, tail, transmitter off, tablet restore); this tool only adds the 4-FSK schedule (stent_trx_ber.schedule is
wrapped: 'fsk4' -> 4-tone schedule, everything else -> the original function).
Decoders: 2-FSK runs -> decode_trx_tablet (frozen trx-tablet-1, result_trx.json) AND decode_trx_mfsk (M = 2,
result_mfsk.json); 4-FSK runs -> decode_trx_mfsk (M = 4). Before FEC = code bits of cc4 / uncoded none_full frames,
after FEC = info bits, frame errors, CRC.

   python fsk4_trx_ber.py synth                                        (simulated panel: pipeline + expectation)
   python fsk4_trx_ber.py plan  [--bitrates 20,40,60,80] [--rounds 8]
   python fsk4_trx_ber.py run   --label hand_fsk4 --condition still --current-ma 1     (human contact: the user runs it)
   python fsk4_trx_ber.py analyse data_maxtouch/stent_trx_<label>_*
Output: run folders data_maxtouch/stent_trx_<label>_{fsk,fsk4}_<baud>baud_c<k>_<time>/ (as stent_trx_ber), session
summary data_maxtouch/fsk4ber_<label>_<time>/ (summary.csv/json, frames.csv, table.html, fsk2_vs_fsk4.png/.pdf).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import signal
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stent_trx_ber as B                       # noqa: E402  (one_run, chunks, schedule, save_seq, synth_capture)
import decode_trx_mfsk as MF                    # noqa: E402
T, R, TV2, STX = B.D, B.R, B.TV2, B.STX
TONES4 = (159585.0, 151895.0, 130979.0, 180556.0)
SCHEMES = (("fsk", "trx-tablet-1", "2-FSK, frozen tablet decoder"), ("fsk", MF.VERSION, "2-FSK, M-FSK decoder (M=2)"),
           ("fsk4", MF.VERSION, "4-FSK, M-FSK decoder (M=4)"))
COLORS = {SCHEMES[0][2]: "#2a78d6", SCHEMES[1][2]: "#1baf7a", SCHEMES[2][2]: "#eb6834"}   # fixed order, never cycled
FRAMES = ("cc4", "none_full")


# ================================================================ 4-FSK schedule
def schedule_fsk4(baud, phase, a):
    s = B.sym_us(baud)
    lv = MF.levels_of_states(phase, 4)
    words = [STX.fword(f) for f in a.tones]
    hz = [w * STX.MCLK / 268435456.0 for w in words]
    plan = dict(modulation="fsk4", sym_us=s, baud=1e6 / s, bits_per_symbol=2, bit_rate=2e6 / s, carrier_hz=hz[0],
                zero_hz=hz[-1], tones_hz=hz, ladder_db=list(a.ladder_db), n_symbols=len(lv), n_states=len(phase),
                total_s=len(lv) * s / 1e6, n_steps=len(lv))
    return [(s, 1, [words[i] for i in lv])], plan


_SCHEDULE = B.schedule


def schedule(mod, rate, phase, a):
    return schedule_fsk4(rate, phase, a) if mod == "fsk4" else _SCHEDULE(mod, rate, phase, a)


B.schedule = schedule                           # stent_trx_ber.one_run looks `schedule` up in its own module


# ================================================================ plan
def ber_args(a):
    """the namespace stent_trx_ber.one_run / R.Chip / R.record read"""
    return types.SimpleNamespace(**dict(vars(a), mods="fsk,fsk4", rates=a.bitrates, carrier=TONES4[0],
                                        fsk_zero=TONES4[3]))


def plan_all(a):
    runs = []
    for br in [float(x) for x in a.bitrates.split(",")]:
        ch, per_round = B.chunks(a, br)                             # frames depend on round(bit rate) only
        for mod in ("fsk", "fsk4"):
            for k, (r0, r1, sl) in enumerate(ch):
                runs.append(dict(mod=mod, rate=br if mod == "fsk" else br / 2, bit_rate=br, chunk=k, rounds=(r0, r1),
                                 seq=sl, per_round=per_round))
    return runs


def show_plan(a, runs):
    keys = B.send_list(a)
    print(f"trx frames: {', '.join(keys)} ({sum(TV2.air_len(k) for k in keys)} states per round), {a.rounds} rounds, "
          f"seed {a.seed}; same frames for 2-FSK and 4-FSK at each bit rate")
    print(f"2-FSK: {TONES4[0] / 1e3:g} / {TONES4[3] / 1e3:g} kHz    4-FSK ladder: "
          + "  ".join(f"{f / 1e3:g} kHz ({db:+g} dB)" for f, db in zip(a.tones, a.ladder_db)))
    tot = 0.0
    for r in runs:
        s = len(r["seq"]["phase"]) / (2 if r["mod"] == "fsk4" else 1) * B.sym_us(r["rate"]) / 1e6
        tot += s + a.lead + 0.5 + B.TAIL_S + 4
        print(f"  {r['bit_rate']:4g} bit/s  {'4-FSK' if r['mod'] == 'fsk4' else '2-FSK'} {r['rate']:5g} baud  chunk {r['chunk']}"
              f"  {len(r['seq']['phase'])} states  {s:6.1f} s on air")
    print(f"{len(runs)} recordings, about {tot / 60:.0f} min")


# ================================================================ analysis
def decode(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    out = []
    if meta["modulation"] == "fsk":
        out.append((SCHEMES[0][2], T.decode_run(d)))
        out.append((SCHEMES[1][2], MF.decode_run(d)))
    elif meta["modulation"] == "fsk4":
        out.append((SCHEMES[2][2], MF.decode_run(d)))
    br = float(round(meta["plan"].get("bit_rate") or 1e6 / meta["plan"]["sym_us"]))      # nominal bit rate (whole bit/s)
    return meta, br, out


def collect(dirs):
    rows, runs = [], []
    for d in dirs:
        print(f"decoding {os.path.basename(os.path.normpath(d))} ...", flush=True)
        meta, br, res = decode(d)
        for scheme, r in res:
            runs.append(dict(scheme=scheme, run=r["run"], bit_rate=round(br, 3), baud=r["baud"], timing_r=r.get("timing_r"),
                             symbol_errors=r.get("symbol_errors"), symbols=r.get("symbols_decoded"),
                             state_errors=r.get("state_errors"), state_symbols=r.get("state_symbols"),
                             scans_per_symbol=r.get("scans_per_symbol") or r.get("samples_per_symbol")))
            for f in r.get("frames", []):
                dd = f["decoders"][f["frame"]]
                rows.append(dict(scheme=scheme, run=r["run"], bit_rate=round(br, 3), baud=round(r["baud"], 3),
                                 frame=f["frame"], i=f["i"], pre_err=f["pre_err"], code_bits=f["code_bits"],
                                 info_err=dd["info_err"], info_bits=TV2.N_INFO, frame_err=int(dd["info_err"] > 0),
                                 crc_ok=int(dd["crc_ok"]), crc_missed=int(dd["crc_missed"]), erasures=f["erasures"],
                                 node=f.get("node", "")))
    return rows, runs


def summarise(rows, runs):
    out = []
    for scheme in [s[2] for s in SCHEMES]:
        for br in sorted({r["bit_rate"] for r in rows}):
            for key in FRAMES:
                rr = [r for r in rows if (r["scheme"], r["bit_rate"], r["frame"]) == (scheme, br, key)]
                if not rr:
                    continue
                n, fe = len(rr), sum(r["frame_err"] for r in rr)
                pe, pb = sum(r["pre_err"] for r in rr), sum(r["code_bits"] for r in rr)
                ie, ib = sum(r["info_err"] for r in rr), sum(r["info_bits"] for r in rr)
                lo, hi = TV2.wilson(fe, n)
                rs = [x for x in runs if x["scheme"] == scheme and x["bit_rate"] == br]
                se, sn = sum(x["symbol_errors"] or 0 for x in rs), sum(x["symbols"] or 0 for x in rs)
                out.append(dict(scheme=scheme, bit_rate=br, baud=rr[0]["baud"], frame=key, frames=n, frame_errors=fe,
                                fer=fe / n, fer_lo=lo, fer_hi=hi, crc_pass=sum(r["crc_ok"] for r in rr),
                                crc_missed=sum(r["crc_missed"] for r in rr), pre_fec_errors=pe, pre_fec_bits=pb,
                                pre_fec_ber=pe / pb, post_fec_errors=ie, post_fec_bits=ib, post_fec_ber=ie / ib,
                                symbol_errors=se if sn else None, symbols=sn or None,
                                symbol_error_rate=se / sn if sn else None,
                                info_bps=TV2.N_INFO * br / TV2.air_len(key)))
    return out


def plot(summ, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted = "#1f2328", "#8a8f98"
    brs = sorted({s["bit_rate"] for s in summ})
    xi = {b: i for i, b in enumerate(brs)}
    panels = [("cc4", "pre", "cc4 frames - before FEC (code bits)"), ("cc4", "post", "cc4 frames - after FEC (info bits)"),
              ("none_full", "post", "uncoded none_full frames (info bits)"), ("cc4", "fer", "cc4 frame error rate (95 % CI)")]
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
    for ax, (key, what, ttl) in zip(axs, panels):
        names = [s[2] for s in SCHEMES]
        for j, name in enumerate(names):
            ss = sorted([s for s in summ if s["scheme"] == name and s["frame"] == key], key=lambda s: s["bit_rate"])
            if not ss:
                continue
            x = [xi[s["bit_rate"]] + (j - 1) * 0.08 for s in ss]
            col = COLORS[name]
            if what == "fer":
                ax.plot(x, [s["fer"] for s in ss], color=col, lw=2, marker="o", ms=5, label=name)
                ax.fill_between(x, [s["fer_lo"] for s in ss], [s["fer_hi"] for s in ss], color=col, alpha=0.08, lw=0)
                continue
            e = [s["pre_fec_errors" if what == "pre" else "post_fec_errors"] for s in ss]
            n = [s["pre_fec_bits" if what == "pre" else "post_fec_bits"] for s in ss]
            y = [k / m if k else 3.0 / m for k, m in zip(e, n)]
            ax.plot(x, y, color=col, lw=2, label=name)
            for xx, yy, k in zip(x, y, e):
                ax.plot(xx, yy, ls="none", color=col, marker="v" if k == 0 else "o", ms=7 if k == 0 else 5.5,
                        mfc="white" if k == 0 else col)
        if what != "fer":
            ax.set_yscale("log"); ax.set_ylim(1e-4, 1); ax.axhline(0.5, color=muted, lw=1, ls=":")
        else:
            ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(range(len(brs))); ax.set_xticklabels([f"{b:g}" for b in brs])
        ax.set_xlabel("bit rate [bit/s]")
        ax.set_title(ttl, fontsize=10, color=ink, loc="left")
        ax.grid(axis="y", color="#e3e6ea", lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("BER")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title + "\n2-FSK baud = bit rate, 4-FSK baud = half the bit rate; open triangle = 0 errors (drawn at the 95 % bound 3/n); dotted = chance", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path + ".png", dpi=150); fig.savefig(path + ".pdf")
    plt.close(fig)


def write(out, rows, runs, summ, extra):
    os.makedirs(out, exist_ok=True)
    for name, data in (("summary.csv", summ), ("frames.csv", rows)):
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    json.dump(dict(extra, mfsk_decoder=MF.VERSION, mfsk_rules_sha=MF.RULES_SHA, tablet_decoder=T.VERSION,
                   tablet_rules_sha=T.RULES_SHA, runs=runs, summary=summ),
              open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    pct = lambda e, n: f"0/{n}" if e == 0 else f"{e}/{n} = {100 * e / n:.2g} %"
    h = ["<table border=1 cellspacing=0 cellpadding=4 style='border-collapse:collapse;font-family:sans-serif;font-size:12px'>",
         "<tr><th>Scheme</th><th>Bit rate</th><th>Baud</th><th>Frame</th><th>Symbol errors</th><th>Before FEC</th>"
         "<th>After FEC</th><th>Frame errors</th><th>CRC missed</th></tr>"]
    for s in summ:
        sym = pct(s["symbol_errors"], s["symbols"]) if s["symbols"] else "-"
        h.append(f"<tr><td>{s['scheme']}</td><td>{s['bit_rate']:g}</td><td>{s['baud']:g}</td><td>{s['frame']}</td>"
                 f"<td>{sym}</td><td>{pct(s['pre_fec_errors'], s['pre_fec_bits'])}</td>"
                 f"<td>{pct(s['post_fec_errors'], s['post_fec_bits'])}</td><td>{s['frame_errors']}/{s['frames']}</td>"
                 f"<td>{s['crc_missed']}</td></tr>")
    open(os.path.join(out, "table.html"), "w").write("\n".join(h + ["</table>"]))


def report(summ):
    print(f"\n{'scheme':32s} {'bit/s':>5s} {'baud':>5s} {'frame':9s} {'symbol err':>14s} {'before FEC':>12s} {'after FEC':>11s} {'frames bad':>10s}")
    for s in summ:
        sym = f"{s['symbol_errors']}/{s['symbols']}" if s["symbols"] else "-"
        print(f"{s['scheme'][:32]:32s} {s['bit_rate']:5g} {s['baud']:5g} {s['frame']:9s} {sym:>14s} "
              f"{s['pre_fec_errors']:>5d}/{s['pre_fec_bits']:<6d} {s['post_fec_errors']:>4d}/{s['post_fec_bits']:<6d} "
              f"{s['frame_errors']:>4d}/{s['frames']}")


def analyse(dirs, out, extra=None, title=None):
    rows, runs = collect(dirs)
    if not rows:
        print("no decodable frames"); return
    summ = summarise(rows, runs)
    write(out, rows, runs, summ, extra or {})
    plot(summ, os.path.join(out, "fsk2_vs_fsk4"), title or f"2-FSK vs 4-FSK, maXTouch tablet - {os.path.basename(out)}")
    report(summ)
    print(f"\nSaved {os.path.relpath(out, HERE)}/ (summary.csv/json, frames.csv, table.html, fsk2_vs_fsk4.png/.pdf)")


# ================================================================ run / synth
def run(a):
    if a.current_ma is None:
        raise SystemExit("--current-ma is required (IEC 60601-1 check of every tone)")
    sys.path.append(os.path.join(HERE, "..", "screen-bloodflow-core-main"))
    import phone_body_sweep as PS                                   # the existing IEC 60601-1 check
    PS.check_current(sorted(set(a.tones)), a.current_ma)
    runs = plan_all(a)
    show_plan(a, runs)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    ba = ber_args(a)
    tx, info = STX.open_configured(ba)
    print(f"Transmitter: {info['line']}")
    done = []
    try:
        for i, r in enumerate(runs):
            print(f"\n[{i + 1}/{len(runs)}] {r['bit_rate']:g} bit/s {'4-FSK' if r['mod'] == 'fsk4' else '2-FSK'} "
                  f"{r['rate']:g} baud, chunk {r['chunk']}")
            d, stopped = B.one_run(tx, info, ba, r)
            if d:
                done.append(d)
            if stopped:
                print("  recording ended early (Ctrl+C or link lost) - stopping the session")
                break
    finally:
        tx.close()
    if done:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        analyse(done, os.path.join(B.DATADIR, f"fsk4ber_{a.label}_{stamp}"), dict(current_ma=a.current_ma,
                condition=a.condition, tones=a.tones, ladder_db=a.ladder_db))


def synth_capture4(plan, phase, rng, A=1500.0, sigma=4.0, delay=0.012, fps=125.0, ny=28, jitter_db=1.5):
    """panel-like scans for 4-FSK: random carrier phase per scan, level per symbol from the ladder (+- jitter_db per
    recording, the survey's sweep-to-sweep spread), nodes 2-4"""
    lead, Ts = 4.0, plan["sym_us"] / 1e6
    t_prompt = 1_000_000.0
    t_first = t_prompt + 0.5
    lv = MF.levels_of_states(phase, 4)
    n = len(lv)
    t = np.arange(t_prompt - lead, t_first + n * Ts + B.TAIL_S, 1 / fps) + rng.normal(0, 0.0008, 1)[0]
    t = t[rng.random(len(t)) >= 0.02]
    k = np.floor((t - t_first - delay) / Ts).astype(int)
    on = (k >= 0) & (k < n)
    lad = np.asarray(plan["ladder_db"]) + np.r_[0, rng.uniform(-jitter_db, jitter_db, 3)]
    amp = np.zeros(len(t))
    amp[on] = A * 10 ** (lad[lv[k[on]]] / 20)
    X = rng.normal(0, sigma, (len(t), ny))
    X[:, 2:5] += (amp * np.sqrt(2) * np.cos(rng.uniform(0, 2 * np.pi, len(t))))[:, None] * np.array([0.5, 1.0, 0.8])
    meta = dict(mode="trx_ber", modulation="fsk4", plan=plan, y0=0, ny=ny, patch=None, t_prompt=t_prompt, t_first=t_first)
    return t, np.round(X).astype(np.int16), meta


def synth(a):
    import tempfile
    # 1) mapping check: ideal levels -> states -> every frame decodes
    seq = TV2.build_seq(list(FRAMES), 2, a.seed, 40)
    lv = MF.levels_of_states(seq["phase"], 4)
    assert np.array_equal(MF.states_of_levels(lv, 4), seq["phase"]), "Gray mapping does not invert"
    print("mapping check: 4-FSK levels <-> states invert exactly")
    root = tempfile.mkdtemp(prefix="fsk4_synth_")
    rng = np.random.default_rng(20261120)
    dirs = []
    for r in plan_all(a):
        _, plan = schedule(r["mod"], r["rate"], r["seq"]["phase"], ber_args(a))
        if r["mod"] == "fsk4":
            t, X, meta = synth_capture4(plan, r["seq"]["phase"], rng)
        else:
            t, X, meta = B.synth_capture("fsk", plan, r["seq"]["phase"], rng, fsk_db=-40.0, gap_p=0.02)
        meta.update(send=r["seq"]["keys"], chunk=r["chunk"], round0=r["rounds"][0], round1=r["rounds"][1],
                    frame_index0=int(r["seq"]["frame_index0"]), args=dict(synthetic=True))
        d = os.path.join(root, f"SIM_{r['mod']}_{r['bit_rate']:g}bps_c{r['chunk']}")
        R.save(d, t, X, meta)
        B.save_seq(d, r["seq"])
        dirs.append(d)
    analyse(dirs, os.path.join(root, "fsk4ber_synth"), dict(synthetic=True), "SIMULATED panel scans (not a measurement)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "run", "analyse", "synth"))
    ap.add_argument("what", nargs="*", help="analyse: run folders (globs ok)")
    ap.add_argument("--label", default="fsk4")
    ap.add_argument("--bitrates", default="20,40,60,80", help="bit rates [bit/s]; 2-FSK baud = rate, 4-FSK baud = rate/2")
    ap.add_argument("--tones", default=",".join(f"{f:g}" for f in TONES4), help="4-FSK tones [Hz], strongest level first")
    ap.add_argument("--ladder-db", default=",".join(f"{x:g}" for x in MF.LADDER4_DB), help="nominal level of each tone [dB]")
    ap.add_argument("--send", default="none_full,cc4")
    ap.add_argument("--rounds", type=int, default=8, help="frames of each encoding per (scheme, bit rate) (default 8)")
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--current-ma", type=float, help="nominal current setting (required for run; IEC check)")
    ap.add_argument("--condition", help="run condition for the record, e.g. still")
    ap.add_argument("--calibrate", choices=("finger-on", "finger-off"), default="finger-on")
    ap.add_argument("--pause", action="store_true", help="wait for Enter before each recording")
    ap.add_argument("--lead", type=float, default=4.0)
    ap.add_argument("--y0", type=int, default=0)
    ap.add_argument("--ny", type=int, default=28)
    ap.add_argument("--patch", type=R.parse_patch)
    ap.add_argument("--keep-noise-suppression", action="store_true")
    ap.add_argument("--keep-touch", action="store_true")
    ap.add_argument("--out", help="analyse: output folder")
    STX.add_hw_args(ap)
    a = ap.parse_args()
    a.tones = [float(x) for x in a.tones.split(",")]
    a.ladder_db = [float(x) for x in a.ladder_db.split(",")]
    if len(a.tones) != 4 or len(a.ladder_db) != 4:
        raise SystemExit("--tones and --ladder-db need 4 values each")
    if a.cmd == "plan":
        show_plan(a, plan_all(a))
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "synth":
        synth(a)
    else:
        dirs = sorted({d for p in a.what for d in glob.glob(p) if os.path.exists(os.path.join(d, "meta.json"))})
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        analyse(dirs, a.out or os.path.join(B.DATADIR, f"fsk4ber_analysis_{stamp}"))


if __name__ == "__main__":
    main()
