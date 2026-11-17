#!/usr/bin/env python3
"""
Through-body BER to the maXTouch tablet with the stent link's own data frames and error correction (trx_psk_node,
2026-11-14). OOK, FSK and DBPSK all send the SAME frames as trx_psk_link_v2's FEC phases: trx_psk_link_v2.build_seq
(unchanged; seed --seed, phase number = round(rate) as trx) -> per round one frame of each --send encoding in shuffled
order, 96 info bits (80 random + CRC-16), sync 32 / 64 symbols, codes from stent_fec. The symbol sequence on air is
trx's per-symbol phase state:
   DBPSK  exactly as trx: the firmware's bit-string flips (pb + 'psk <sym_us> buf', 2026-10-27 TeensyTRX), one carrier
   OOK    state 1 = carrier (--carrier, default 168 kHz), state 0 = off
   FSK    state 1 = --carrier (168 kHz), state 0 = --fsk-zero (171 kHz; 3 kHz = a whole multiple of every default
          rate, trx's orthogonality rule)
Results compare BEFORE error correction (code-bit BER as the decoder sees it, and the uncoded frames 'none_full' =
same symbol rate, 'none' = same info rate) with AFTER (info BER, frame error rate, CRC) for every trx decoder of the
frames sent. Decoder: decode_trx_tablet.py (own frozen rules; decode_v4 / stent_modcomp / trx code only called).

   python stent_trx_ber.py plan  [--mods ook,fsk] [--rates 20,40,50,60] [--send none,none_full,cc4] [--rounds 12]
   python stent_trx_ber.py run   --label hand_trx --condition still --current-ma 1 --pause   (human contact: the user runs it)
   python stent_trx_ber.py analyse data_maxtouch/stent_trx_<label>_*      (or a trxber_<label>_<time> folder)
   python stent_trx_ber.py compare 'OOK 168k=data_maxtouch/trxber_A:ook' 'FSK 159/168k=data_maxtouch/trxber_B' --label X
   python stent_trx_ber.py synth                                           (self-test on simulated panel scans)

A schedule holds at most 4096 OOK/FSK symbols, so each (modulation, rate) is sent in chunks of whole rounds, one
recording each (DBPSK uses the same chunks so every modulation sees the same frames in the same order).
Run folders: data_maxtouch/stent_trx_<label>_<mod>_<baud>baud_c<k>_<time>/ (meta.json, capture.npz, tx_seq.npz,
result_trx.json); session summary: data_maxtouch/trxber_<label>_<time>/ (summary.csv/json, frames.csv, table.html,
fec_before_after.png/.pdf).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import math
import os
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode_trx_tablet as D                   # noqa: E402  (imports stent_modcomp, decode_v4, trx_psk_link_v2)
SM, R, TV2, F = D.SM, D.R, D.TV2, D.F
STX = SM.STX
DATADIR = SM.DATADIR
MAXW = 4096                                     # TeensyTRX schedule words
MODS = ("ook", "fsk", "dbpsk")
NAMES = dict(ook="OOK", fsk="FSK", dbpsk="DBPSK")
DEFAULT_SEND = "none,none_full,cc4"             # cc4 = trx's chosen code (TV2.CHOSEN); 'all' = every trx encoding
TAIL_S = 2.0


# ================================================================ plan
def sym_us(rate):
    return int(round(1e6 / rate))


def send_list(a):
    keys = list(TV2.ENCODINGS) if a.send.strip() == "all" else [k.strip() for k in a.send.split(",") if k.strip()]
    bad = [k for k in keys if k not in TV2.ENCODINGS]
    if bad:
        raise SystemExit(f"unknown encodings {bad}; trx has {list(TV2.ENCODINGS)}")
    return keys


def chunks(a, rate):
    """[(round0, round1, seq-slice)] for one rate: whole rounds, <= MAXW symbols each"""
    keys = send_list(a)
    per_round = sum(TV2.air_len(k) for k in keys)
    if per_round > MAXW:
        raise SystemExit(f"one round of {keys} is {per_round} symbols > {MAXW} schedule words - send fewer encodings")
    rpc = max(1, MAXW // per_round)
    seq = TV2.build_seq(keys, a.rounds, a.seed, int(round(rate)))
    K, out = len(keys), []
    for r0 in range(0, a.rounds, rpc):
        r1 = min(a.rounds, r0 + rpc)
        f0, f1 = r0 * K, r1 * K
        s0, s1 = int(seq["start"][f0]), int(seq["start"][f1 - 1] + seq["length"][f1 - 1])
        out.append((r0, r1, dict(keys=keys, enc=seq["enc"][f0:f1], start=seq["start"][f0:f1] - s0,
                                 length=seq["length"][f0:f1], u=seq["u"][f0:f1], phase=seq["phase"][s0:s1],
                                 frame_index0=f0)))
    return out, per_round


def schedule(mod, rate, phase, a):
    """TeensyTRX schedule [(step_us, reps, words)] and plan for one chunk"""
    s = sym_us(rate)
    w1 = STX.fword(a.carrier)
    plan = dict(modulation=mod, sym_us=s, baud=1e6 / s, carrier_hz=w1 * STX.MCLK / 268435456.0, n_symbols=len(phase),
                total_s=len(phase) * s / 1e6, n_steps=len(phase))
    if mod == "dbpsk":
        plan.update(zero_hz=None, n_steps=len(phase))
        return [(s, len(phase), [w1])], plan
    w0 = 0 if mod == "ook" else STX.fword(a.fsk_zero)
    plan.update(zero_hz=0.0 if mod == "ook" else w0 * STX.MCLK / 268435456.0)
    return [(s, 1, [w1 if b else w0 for b in phase])], plan


def plan_all(a):
    mods = [m.strip() for m in a.mods.split(",") if m.strip()]
    bad = [m for m in mods if m not in MODS]
    if bad:
        raise SystemExit(f"--mods: choose from {MODS}")
    rates = [float(r) for r in a.rates.split(",")]
    runs = []
    for rate in rates:                          # same rate -> same frames for every modulation, back to back
        ch, per_round = chunks(a, rate)
        for mod in mods:
            for k, (r0, r1, sl) in enumerate(ch):
                runs.append(dict(mod=mod, rate=rate, chunk=k, rounds=(r0, r1), seq=sl, per_round=per_round))
    return runs


def show_plan(a, runs):
    keys = send_list(a)
    tot = sum(len(r["seq"]["phase"]) * sym_us(r["rate"]) / 1e6 + a.lead + 0.5 + TAIL_S + 4 for r in runs)
    print(f"trx frames: {', '.join(keys)} ({sum(TV2.air_len(k) for k in keys)} symbols per round), {a.rounds} rounds "
          f"= {a.rounds} frames of each per (modulation, rate), seed {a.seed}")
    print(f"carrier {a.carrier / 1e3:g} kHz" + (f", FSK 0 = {a.fsk_zero / 1e3:g} kHz" if "fsk" in a.mods else "")
          + f", wave {a.wave}")
    for r in runs:
        if r["chunk"] == 0:
            n = sum(1 for q in runs if q["mod"] == r["mod"] and q["rate"] == r["rate"])
            s = sum(len(q["seq"]["phase"]) for q in runs if q["mod"] == r["mod"] and q["rate"] == r["rate"])
            print(f"  {NAMES[r['mod']]:6s} {1e6 / sym_us(r['rate']):7.3f} baud: {n} recording(s), {s} symbols, "
                  f"{s * sym_us(r['rate']) / 1e6:6.0f} s on air")
    print(f"{len(runs)} recordings, about {tot / 60:.0f} min")


# ================================================================ record
def load_bits(tx, flips):
    """trx_psk_link_v2.load_bits for StentTX: bit string into the firmware, checked with pbinfo"""
    import re
    lines, bits = TV2.pb_lines(flips)
    tx.ask("pbclr", r"^# pb cleared")
    for ln in lines:
        tx.send(ln)
    r = tx.ask("pbinfo", r"^# pb n=\d+ ones=\d+ checksum=\d+", timeout=5.0)
    m = re.search(r"# pb n=(\d+) ones=(\d+) checksum=(\d+)", r)
    cs, ones = TV2.pb_checksum(bits)
    if not m or int(m.group(1)) != len(bits) or int(m.group(2)) != ones or int(m.group(3)) != cs:
        raise RuntimeError(f"bit string check failed: firmware {r!r}, sent n={len(bits)} ones={ones} checksum={cs}")
    return dict(n=len(bits), ones=ones, checksum=cs)


def run_dir(a, mod, rate, k):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(DATADIR, f"stent_trx_{a.label}_{mod}_{round(1e6 / sym_us(rate)):g}baud_c{k}_{stamp}")


def save_seq(d, sl):
    np.savez_compressed(os.path.join(d, "tx_seq.npz"), keys=np.array(sl["keys"]), enc=sl["enc"], start=sl["start"],
                        length=sl["length"], u=np.packbits(sl["u"], 1), phase=np.packbits(sl["phase"]),
                        n_phase=len(sl["phase"]))


def one_run(tx, tx_info, a, r):
    sl, mod, rate = r["seq"], r["mod"], r["rate"]
    sched, plan = schedule(mod, rate, sl["phase"], a)
    got = tx.upload_schedule(sched)
    pb = None
    if mod == "dbpsk":
        flips = (sl["phase"] ^ np.concatenate([[0], sl["phase"][:-1]])).astype(np.uint8)
        pb = load_bits(tx, flips)
        rr = tx.ask(f"psk {plan['sym_us']} buf", r"^# psk")
        if f"sym_us={plan['sym_us']}" not in rr or "src=buf" not in rr:
            raise RuntimeError(f"psk buf not taken: {rr!r}")
    print(f"  uploaded: {got}" + (f", bit string {pb}" if pb else ""))
    seconds = STX.GO_DELAY_MS / 1e3 + plan["total_s"] + TAIL_S
    finger_off = a.calibrate == "finger-off"
    if finger_off:
        input(f"  [{a.condition or 'no condition'}] electrode on, finger OFF the panel - press Enter to calibrate ")
    elif a.pause:
        input(f"  [{a.condition or 'no condition'}] finger on the panel, electrode on; keep still until the run ends "
              f"(~{seconds + a.lead + 2:.0f} s) - press Enter to start ")
    chip = status = None
    t_cal = None
    try:
        chip = R.Chip(a)
        changed = chip.setup()
        t_cal = time.time()
        if finger_off:
            input(f"  calibrated. Finger ON the panel now (~{seconds + a.lead + 2:.0f} s) - press Enter to start ")
        t, X, t_prompt, t_first, errors = R.record(chip, a, seconds, lambda: tx.go()[0])
    finally:
        try:
            status = tx.status()
        except Exception as e:                  # noqa: BLE001
            status = dict(error=str(e))
        tx.off()
        if chip:
            chip.restore()
    stopped = len(t) < 50 or t_prompt is None or t[-1] < t_prompt + seconds - 1.0
    if len(t) < 50:
        print("  too few samples recorded - nothing saved")
        return None, True
    d = run_dir(a, mod, rate, r["chunk"])
    meta = dict(mode="trx_ber", modulation=mod, args=vars(a), plan=plan, y0=chip.y0, ny=chip.ny, patch=chip.patch,
                t_prompt=t_prompt, t_first=t_first, tx=tx_info, tx_status=status, hid_errors=errors,
                chip_ctrl_orig=changed, recorded_s=float(t[-1] - t[0]), seconds_planned=seconds, samples=int(len(t)),
                send=sl["keys"], seed=a.seed, seq_k=int(round(rate)), chunk=r["chunk"], round0=r["rounds"][0],
                round1=r["rounds"][1], frame_index0=int(sl["frame_index0"]), pb=pb, calibration=a.calibrate,
                t_calibrated=t_cal, decoder_rules_sha=D.RULES_SHA, trx_shas=D.TRX_SHAS,
                tool_sha=__import__("hashlib").sha256(open(__file__, "rb").read()).hexdigest()[:12],
                **SM.lead_touch(t, X, t_prompt, chip.names))
    R.save(d, t, X, meta)
    save_seq(d, sl)
    ev, want = status.get("events", "?"), f"{plan['n_steps']}/{plan['n_steps']}"
    print(f"  saved {os.path.relpath(d, HERE)}   transmitter: state {status.get('state')}, events {ev}"
          + ("" if ev == want and status.get("state") == "done" else f"   ** expected done, {want} **")
          + (f", psk_src {status.get('psk_src')}" if mod == "dbpsk" else ""))
    return d, stopped


def run(a):
    runs = plan_all(a)
    show_plan(a, runs)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    tx, info = STX.open_configured(a)
    print(f"Transmitter: {info['line']}")
    if any(r["mod"] == "dbpsk" for r in runs):
        try:
            tx.ask("pbinfo", r"^# pb n=")
        except TimeoutError:
            tx.close()
            raise SystemExit("this TeensyTRX has no bit-string commands (pb, psk ... buf) - flash "
                             "trx_psk_node/transceiver/firmware/TeensyTRX, or leave dbpsk out of --mods")
    done = []
    try:
        for i, r in enumerate(runs):
            print(f"\n[{i + 1}/{len(runs)}] {NAMES[r['mod']]} {1e6 / sym_us(r['rate']):g} baud, chunk {r['chunk']} "
                  f"(rounds {r['rounds'][0]}-{r['rounds'][1] - 1})")
            d, stopped = one_run(tx, info, a, r)
            if d:
                done.append(d)
            if stopped:
                print("  recording ended early (Ctrl+C or link lost) - stopping the session")
                break
    finally:
        tx.close()
    if done:
        stamp = os.path.basename(done[0]).rsplit("_", 2)[-2:]
        analyse(done, os.path.join(DATADIR, f"trxber_{a.label}_{'_'.join(stamp)}"), a.current_ma)


# ================================================================ analysis
def wilson(k, n):
    return TV2.wilson(k, n)


def collect(dirs):
    rows, runs = [], []
    for d in dirs:
        print(f"decoding {os.path.basename(os.path.normpath(d))} ...", flush=True)
        res = D.decode_run(d)
        runs.append(res)
        for fr in res.get("frames", []):
            for dec, r in fr["decoders"].items():
                rows.append(dict(run=res["run"], modulation=res["modulation"], baud=round(res["baud"], 3), frame=fr["frame"],
                                 decoder=dec, i=fr["i"], round=fr["round"], erasures=fr["erasures"],
                                 reference_ok=fr["reference_ok"], pre_err=fr["pre_err"], code_bits=fr["code_bits"],
                                 info_err=r["info_err"], info_bits=TV2.N_INFO, frame_err=r["frame_err"],
                                 crc_ok=r["crc_ok"], crc_missed=r["crc_missed"]))
    return rows, runs


def label_of(dec, mod):
    """trx's decoder label; its uncoded frames say 'DBPSK', which is only true for the DBPSK runs here"""
    lab = TV2.DECODERS[dec][2]
    return lab if mod == "dbpsk" else lab.replace("(DBPSK, ", "(").replace(", today's DBPSK receiver", ", differential")


def summarise(rows, runs):
    out = []
    groups = sorted({(r["modulation"], r["baud"], r["decoder"]) for r in rows},
                    key=lambda g: (MODS.index(g[0]), g[1], list(TV2.DECODERS).index(g[2])))
    for mod, baud, dec in groups:
        rr = [r for r in rows if (r["modulation"], r["baud"], r["decoder"]) == (mod, baud, dec)]
        key = rr[0]["frame"]
        n = len(rr)
        fe = sum(r["frame_err"] for r in rr)
        ie, ib = sum(r["info_err"] for r in rr), sum(r["info_bits"] for r in rr)
        pe, pb = sum(r["pre_err"] for r in rr), sum(r["code_bits"] for r in rr)
        lo, hi = wilson(fe, n)
        info_rate = TV2.N_INFO * baud / TV2.air_len(key)
        st = [x for x in runs if x["modulation"] == mod and round(x["baud"], 3) == baud and not x.get("signal_lost")]
        out.append(dict(modulation=mod, baud=baud, frame=key, decoder=dec, label=label_of(dec, mod),
                        coded=TV2.make_fec(key).kind != "none", air_symbols=TV2.air_len(key), info_bps=info_rate,
                        frames=n, frame_errors=int(fe), fer=fe / n, fer_lo=lo, fer_hi=hi,
                        crc_pass=int(sum(r["crc_ok"] for r in rr)), crc_missed=int(sum(r["crc_missed"] for r in rr)),
                        pre_fec_errors=int(pe), pre_fec_bits=int(pb), pre_fec_ber=pe / pb if pb else None,
                        post_fec_errors=int(ie), post_fec_bits=int(ib), post_fec_ber=ie / ib if ib else None,
                        post_fec_ber_upper95=3.0 / ib if ib and ie == 0 else None,
                        erasures=int(sum(r["erasures"] for r in rr)), reference_failed=int(sum(not r["reference_ok"] for r in rr)),
                        state_ber=(sum(x["state_errors"] for x in st) / max(sum(x["state_symbols"] for x in st), 1)) if st else None))
    return out


def write_tables(out, rows, runs, summ, current_ma):
    os.makedirs(out, exist_ok=True)
    keys = list(summ[0])
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summ)
    with open(os.path.join(out, "frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(decoder=D.VERSION, rules_sha=D.RULES_SHA, frozen_at=D.FROZEN_AT, trx_shas=D.TRX_SHAS,
                   current_ma=current_ma, runs=[{k: v for k, v in r.items() if k != "frames"} for r in runs],
                   summary=summ), open(os.path.join(out, "summary.json"), "w"), indent=1)
    pct = lambda e, n: "-" if not n else (f"0/{n} (&lt; {300 / n:.2g} %)" if e == 0 else f"{e}/{n} = {100 * e / n:.2g} %")
    h = ["<table border=1 cellspacing=0 cellpadding=4 style='border-collapse:collapse;font-family:sans-serif;font-size:12px'>",
         "<tr><th>Modulation</th><th>Baud</th><th>Frame / decoder</th><th>Info bps</th><th>Before FEC: code-bit BER</th>"
         "<th>After FEC: info BER</th><th>Frame errors</th><th>CRC pass</th><th>CRC missed</th></tr>"]
    for s in summ:
        h.append(f"<tr><td>{NAMES[s['modulation']]}</td><td>{s['baud']:g}</td><td>{s['label']}</td><td>{s['info_bps']:.2f}</td>"
                 f"<td>{pct(s['pre_fec_errors'], s['pre_fec_bits'])}</td><td>{pct(s['post_fec_errors'], s['post_fec_bits'])}</td>"
                 f"<td>{s['frame_errors']}/{s['frames']}</td><td>{s['crc_pass']}/{s['frames']}</td><td>{s['crc_missed']}</td></tr>")
    h.append("</table>")
    open(os.path.join(out, "table.html"), "w").write("\n".join(h))


# fixed categorical order (validated default palette, slots 1-5); a frame keeps its colour whatever else is sent
FRAME_COLORS = dict(cc4="#2a78d6", none_full="#eb6834", none="#1baf7a", cc7="#eda100", h84="#7b3294",
                    bch31="#8c564b", bch63="#e377c2", rs16="#17becf", d_cc4="#1f3f7a", d_bch63="#9e9e9e")


def plot(summ, path, title, columns=None):
    """columns: [(title, value of row["column"])] for a comparison figure; default = one column per modulation"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if columns is None:
        columns = [(NAMES[m], m) for m in MODS if any(s["modulation"] == m for s in summ)]
        summ = [dict(s, column=s["modulation"]) for s in summ]
    mods = [c for _, c in columns]
    bauds = sorted({s["baud"] for s in summ})
    xi = {b: i for i, b in enumerate(bauds)}
    prim = [s for s in summ if s["decoder"] == s["frame"]]           # one (soft) decoder per frame type in the figure
    fig, axs = plt.subplots(2, len(mods), figsize=(max(4.6 * len(mods), 11.0), 8.2), squeeze=False, sharey="row")
    ink, muted = "#1f2328", "#8a8f98"
    floor = 1e-4
    handles = None
    for c, mod in enumerate(mods):
        ax, axf = axs[0, c], axs[1, c]
        present = [k for k in FRAME_COLORS if any(s["frame"] == k and s["column"] == mod for s in prim)]
        for j, key in enumerate(present):
            ss = sorted([s for s in prim if s["frame"] == key and s["column"] == mod], key=lambda s: s["baud"])
            col, coded = FRAME_COLORS[key], ss[0]["coded"]
            dx = (j - (len(present) - 1) / 2) * 0.07                  # dodge: equal values must not hide each other
            x = [xi[s["baud"]] + dx for s in ss]
            name = label_of(key, ss[0]["modulation"])

            def series(err, bits, ls, lw, mfc, label):
                y = [e / n if e else 3.0 / n for e, n in zip(err, bits)]     # zero errors -> 95 % bound 3/n
                ax.plot(x, y, color=col, lw=lw, ls=ls, label=label)
                for xx, yy, e in zip(x, y, err):
                    ax.plot(xx, yy, ls="none", color=col, ms=7 if e == 0 else 5.5, marker="v" if e == 0 else "o",
                            mfc="white" if (e == 0 or mfc == "white") else col)
            if coded:
                series([s["pre_fec_errors"] for s in ss], [s["pre_fec_bits"] for s in ss], "--", 1.4, "white",
                       f"{name}: before FEC (code bits)")
            series([s["post_fec_errors"] for s in ss], [s["post_fec_bits"] for s in ss], "-", 2.0, col,
                   f"{name}: after FEC (info bits)" if coded else f"{name} (info bits)")
            axf.plot(x, [s["fer"] for s in ss], color=col, lw=2, marker="o", ms=5, label=name)
            axf.fill_between(x, [s["fer_lo"] for s in ss], [s["fer_hi"] for s in ss], color=col, alpha=0.08, lw=0)
        for a_ in (ax, axf):
            a_.set_xticks(range(len(bauds))); a_.set_xticklabels([f"{round(b, 1):g}" for b in bauds])
            a_.grid(axis="y", color="#e3e6ea", lw=0.8); a_.set_axisbelow(True)
            for sp in ("top", "right"):
                a_.spines[sp].set_visible(False)
        ax.set_yscale("log"); ax.set_ylim(floor * 0.7, 1)
        ax.axhline(0.5, color=muted, lw=1, ls=":")
        ax.set_title(columns[c][0], fontsize=11, color=ink)
        axf.set_ylim(-0.02, 1.02); axf.set_xlabel("symbol rate [baud]")
        if c == 0:
            ax.set_ylabel("BER"); axf.set_ylabel("frame error rate (95 % CI)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(title + "\n(open triangle = 0 errors, drawn at the 95 % bound 3/n; dotted = chance)", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path + ".png", dpi=150); fig.savefig(path + ".pdf")
    plt.close(fig)


def report(summ):
    print(f"\n{'mod':6s} {'baud':>6s} {'frame/decoder':42s} {'before FEC':>18s} {'after FEC':>18s} {'frames bad':>10s} {'CRC miss':>8s}")
    for s in summ:
        b = f"{s['pre_fec_errors']}/{s['pre_fec_bits']}" if s["coded"] else "-"
        print(f"{NAMES[s['modulation']]:6s} {s['baud']:6g} {s['label'][:42]:42s} {b:>18s} "
              f"{s['post_fec_errors']:>8d}/{s['post_fec_bits']:<9d} {s['frame_errors']:>4d}/{s['frames']:<5d} {s['crc_missed']:>8d}")


def analyse(dirs, out, current_ma=None, title=None):
    rows, runs = collect(dirs)
    if not rows:
        print("no decodable frames"); return
    summ = summarise(rows, runs)
    write_tables(out, rows, runs, summ, current_ma)
    plot(summ, os.path.join(out, "fec_before_after"),
         title or f"{os.path.basename(out)}" + (f", {current_ma:g} mA" if current_ma else "") + f", decoder {D.RULES_SHA}")
    report(summ)
    for r in runs:
        nan = lambda v: float("nan") if v is None else v
        print(f"  {r['run']}: delay {nan(r.get('delay_ms')):.1f} ms, clock {nan(r.get('clock_ppm')):.0f} ppm, "
              f"state BER {r.get('state_errors', 0)}/{r.get('state_symbols', 0)}, erasures {r.get('erasures')}, "
              f"nodes {' '.join(r['nodes'])}" + ("  ** signal lost **" if r.get("signal_lost") else ""))
    print(f"\nSaved {os.path.relpath(out, HERE)}/ (summary.csv/json, frames.csv, table.html, fec_before_after.png/.pdf)")


def compare(specs, out, title=None):
    """NAME=trxber_<folder>[:modulation] ... -> one figure / table, a column per NAME (no re-decode: summary.json)"""
    rows, columns = [], []
    for sp in specs:
        name, rest = sp.split("=", 1)
        path, mod = (rest.rsplit(":", 1) + [None])[:2] if ":" in os.path.basename(rest) else (rest, None)
        js = json.load(open(os.path.join(path, "summary.json")))
        if js["rules_sha"] != D.RULES_SHA:
            print(f"  ! {path}: decoder {js['rules_sha']} != current {D.RULES_SHA}")
        got = [dict(r, column=name, scheme=name, source=os.path.basename(os.path.normpath(path)))
               for r in js["summary"] if mod is None or r["modulation"] == mod]
        if not got:
            raise SystemExit(f"{sp}: no rows")
        rows += got
        columns.append((name, name))
    os.makedirs(out, exist_ok=True)
    keys = ["scheme"] + [k for k in rows[0] if k not in ("scheme", "column")]
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    pct = lambda e, n: "-" if not n else (f"0/{n} (&lt; {300 / n:.2g} %)" if e == 0 else f"{e}/{n} = {100 * e / n:.2g} %")
    h = ["<table border=1 cellspacing=0 cellpadding=4 style='border-collapse:collapse;font-family:sans-serif;font-size:12px'>",
         "<tr><th>Scheme</th><th>Baud</th><th>Frame / decoder</th><th>Before FEC: code-bit BER</th>"
         "<th>After FEC: info BER</th><th>Frame errors</th><th>CRC pass</th></tr>"]
    for r in rows:
        if r["decoder"] != r["frame"]:
            continue
        h.append(f"<tr><td>{r['scheme']}</td><td>{round(r['baud'], 1):g}</td><td>{r['label']}</td>"
                 f"<td>{pct(r['pre_fec_errors'], r['pre_fec_bits']) if r['coded'] else '-'}</td>"
                 f"<td>{pct(r['post_fec_errors'], r['post_fec_bits'])}</td><td>{r['frame_errors']}/{r['frames']}</td>"
                 f"<td>{r['crc_pass']}/{r['frames']}</td></tr>")
    h.append("</table>")
    open(os.path.join(out, "table.html"), "w").write("\n".join(h))
    plot(rows, os.path.join(out, "fec_before_after"), title or os.path.basename(os.path.normpath(out)), columns)
    print(f"Saved {os.path.relpath(out, HERE)}/ (summary.csv, table.html, fec_before_after.png/.pdf)")


def resolve(what):
    dirs = []
    for p in what:
        if os.path.exists(os.path.join(p, "summary.json")) and os.path.basename(os.path.normpath(p)).startswith("trxber_"):
            dirs += [os.path.join(DATADIR, r["run"]) for r in json.load(open(os.path.join(p, "summary.json")))["runs"]]
        else:
            dirs += [q for q in sorted(glob.glob(p)) if os.path.exists(os.path.join(q, "tx_seq.npz"))]
    return dirs


# ================================================================ self-test
def synth_capture(mod, plan, phase, rng, A=1500.0, sigma=4.0, fsk_db=-2.7, delay=0.012, fps=125.0, ny=28, gap_p=0.0):
    """panel-like scans: random carrier phase on every scan (|A sqrt2 cos phi|-type values, signed), nodes 2-4,
    TX edges known, delay = panel latency; gap_p = chance a scan is missing (makes erasures)"""
    lead, Ts = 4.0, plan["sym_us"] / 1e6
    t_prompt = 1_000_000.0
    t_first = t_prompt + 0.5
    n = len(phase)
    t = np.arange(t_prompt - lead, t_first + n * Ts + TAIL_S, 1 / fps) + rng.normal(0, 0.0008, 1)[0]
    t = t[rng.random(len(t)) >= gap_p]
    k = np.floor((t - t_first - delay) / Ts).astype(int)
    on = (k >= 0) & (k < n)
    st = np.zeros(len(t)); st[on] = phase[k[on]]
    amp = np.zeros(len(t))
    if mod == "ook":
        amp[on] = A * st[on]
    elif mod == "fsk":
        amp[on] = np.where(st[on] == 1, A, A * 10 ** (fsk_db / 20))
    else:
        amp[on] = A
    X = rng.normal(0, sigma, (len(t), ny))
    w = np.array([0.5, 1.0, 0.8])
    X[:, 2:5] += (amp * np.sqrt(2) * np.cos(rng.uniform(0, 2 * np.pi, len(t))))[:, None] * w
    meta = dict(mode="trx_ber", modulation=mod, plan=plan, y0=0, ny=ny, patch=None, t_prompt=t_prompt, t_first=t_first)
    return t, np.round(X).astype(np.int16), meta


def synth(a):
    import tempfile
    # 1) mapping check: ideal states -> every trx decoder recovers every frame (no panel involved)
    keys = list(TV2.ENCODINGS)
    seq = TV2.build_seq(keys, 2, a.seed, 50)
    bad = 0
    for i in range(len(seq["enc"])):
        key = keys[seq["enc"][i]]
        ph = seq["phase"][seq["start"][i]:seq["start"][i] + seq["length"][i]]
        _, pre_n, out = D.decode_frame(key, D.code_soft(key, 2.0 * ph - 1), seq["u"][i])
        bad += sum(r["info_err"] > 0 or not r["crc_ok"] for r in out.values())
    print(f"mapping check (ideal states, all {len(keys)} trx encodings x 2 rounds): {bad} decoder failures (want 0)")
    if bad:
        raise SystemExit("mapping check failed")
    # 2) simulated panel: OOK / FSK (-2.7 dB second tone, as the 10-26 body sweeps at 167.7 / 171.9 kHz) / DBPSK
    root = a.datadir or tempfile.mkdtemp(prefix="trxber_synth_")
    rng = np.random.default_rng(20261114)
    a.rounds = a.rounds or 6
    dirs = []
    for rate in [float(r) for r in a.rates.split(",")]:
        ch, _ = chunks(a, rate)
        for mod in MODS:
            for k, (r0, r1, sl) in enumerate(ch):
                _, plan = schedule(mod, rate, sl["phase"], a)
                t, X, meta = synth_capture(mod, plan, sl["phase"], rng, gap_p=a.gap_p)
                meta.update(send=sl["keys"], chunk=k, round0=r0, round1=r1, frame_index0=int(sl["frame_index0"]),
                            args=dict(synthetic=True))
                d = os.path.join(root, f"SIM_stent_trx_{mod}_{round(1e6 / sym_us(rate)):g}baud_c{k}")
                R.save(d, t, X, meta)
                save_seq(d, sl)
                dirs.append(d)
    analyse(dirs, os.path.join(root, "trxber_SIM"), title="SIMULATED panel scans (random phase per scan)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "run", "analyse", "compare", "synth"))
    ap.add_argument("what", nargs="*", help="analyse: run folders (globs ok) or a trxber_<label>_<time> folder")
    ap.add_argument("--label", default="trx")
    ap.add_argument("--mods", default="ook,fsk", help="ook, fsk (dbpsk still possible, not used by default)")
    ap.add_argument("--rates", default="20,40,50,60", help="symbol rates [baud] (symbol = whole us, as trx)")
    ap.add_argument("--carrier", type=float, default=168000.0, help="OOK carrier / FSK state-1 tone / DBPSK carrier [Hz]")
    ap.add_argument("--fsk-zero", type=float, default=171000.0, help="FSK state-0 tone [Hz]")
    ap.add_argument("--send", default=DEFAULT_SEND, help=f"trx encodings (default {DEFAULT_SEND}; 'all' = {','.join(TV2.ENCODINGS)})")
    ap.add_argument("--rounds", type=int, default=12, help="frames of each encoding per (modulation, rate) (default 12)")
    ap.add_argument("--seed", type=int, default=1027, help="trx frame order / info bits seed (trx default 1027)")
    ap.add_argument("--current-ma", type=float, help="nominal current setting, for the record")
    ap.add_argument("--condition", help="run condition for the record, e.g. still")
    ap.add_argument("--calibrate", choices=("finger-on", "finger-off"), default="finger-on")
    ap.add_argument("--pause", action="store_true", help="wait for Enter before each recording")
    ap.add_argument("--lead", type=float, default=4.0)
    ap.add_argument("--y0", type=int, default=0)
    ap.add_argument("--ny", type=int, default=28)
    ap.add_argument("--patch", type=R.parse_patch)
    ap.add_argument("--keep-noise-suppression", action="store_true")
    ap.add_argument("--keep-touch", action="store_true")
    ap.add_argument("--datadir", help="synth: output folder (default: new temp folder)")
    ap.add_argument("--gap-p", type=float, default=0.02, help="synth: share of missing panel scans")
    ap.add_argument("--out", help="analyse / compare: output folder")
    ap.add_argument("--title", help="compare: figure title")
    STX.add_hw_args(ap)
    a = ap.parse_args()
    if a.cmd == "plan":
        show_plan(a, plan_all(a)); return
    if a.cmd == "run":
        run(a); return
    if a.cmd == "synth":
        synth(a); return
    if a.cmd == "compare":
        compare(a.what, a.out or os.path.join(DATADIR, f"trxber_compare_{a.label}"), a.title); return
    dirs = resolve(a.what)
    if not dirs:
        raise SystemExit("no stent_trx run folders found")
    analyse(dirs, a.out or os.path.join(DATADIR, f"trxber_{a.label}_reanalysed_{datetime.datetime.now():%Y%m%d_%H%M%S}"),
            a.current_ma)


if __name__ == "__main__":
    main()
