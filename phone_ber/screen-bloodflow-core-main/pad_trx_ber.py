#!/usr/bin/env python3
"""
Through-body BER to the OnePlus Pad with the stent link's own frames and error correction (trx_psk_node), 2026-11-14.

Same frames as the maXTouch tablet BER (stent_trx_ber.py): trx_psk_link_v2.build_seq (unchanged; seed --seed, phase
number = round(rate) as trx), per round one frame of each --send encoding in shuffled order, 96 info bits (80 random +
CRC-16), codes from stent_fec. State 1 = carrier (--carrier, default 37.4 kHz: the one carrier all three devices
answered in the 1-500 kHz sweeps; the Pad answers 9.9-117 kHz). OOK: state 0 = off. FSK: state 0 = --fsk-zero (default
156.25 kHz, inside the Pad's no-response band 129-172 kHz: 0 % refresh there in the sweep), so the body always carries
a tone and the Pad still tells the states apart. DBPSK makes no sense on the Pad (its map refreshes for any carrier).

Receiver: the Pad's delta map (/proc/touchpanel/debug_info/delta, 38 x 54, read every ~150 ms) is frozen while the
controller is idle and refreshes on every read while the carrier is on (sweep: from +0.15 s after ON to +0.2 s after
OFF). Decoder decode_trx_pad.py (rules fixed before any recording): feature = share of nodes changed since the
previous read; known-timing symbol values; then the tablet decoder's frozen soft-value / FEC / CRC rules.
Before vs after error correction as for the tablet: uncoded none_full (1 symbol per bit) vs cc4 (trx's chosen code)
code-bit BER before, info BER / frame errors / CRC after.

   python pad_trx_ber.py synth                                    (simulated Pad reads: decoder + rate check, no hardware)
   python pad_trx_ber.py plan  [--mods ook,fsk] [--rates 2] [--rounds 3] [--current-ma 0.1]
   python pad_trx_ber.py run   --label hand --site hand2finger --current-ma 0.1        (human contact: the user runs it)
   python pad_trx_ber.py analyse data_oneplus/trx_ber/<run folders or a trxber_<label>_<time> summary folder>

Each recording = --rounds-per-rec rounds (default 1: ~1.6-3.2 min, so the finger only has to stay still that long);
Enter before each. Run folders data_oneplus/trx_ber/pad_trx_<label>_ook_<baud>baud_c<k>_<time>/ (frames.npz,
meta.json, tx_seq.npz, result_trx.json); session summary data_oneplus/trx_ber/trxber_<label>_<time>/ (summary.csv/json,
frames.csv, table.html, fec_before_after.png/.pdf).
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decode_trx_pad as DP                     # noqa: E402  (adds maxtouch-screen-main to sys.path)
import stent_trx_ber as B                       # noqa: E402  (schedule / save_seq / summarise / tables / plot - unchanged)
import pad_body_sweep as PAD                    # noqa: E402
import phone_body_sweep as PS                   # noqa: E402
SW = PS.SW
TV2, STX = B.TV2, B.STX
OUT = HERE / "data_oneplus" / "trx_ber"
DEFAULT_SEND = "none_full,cc4"
LEAD_S, TAIL_S = 6.0, 2.0
TOOL_SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:12]


class Device:
    """what differs per receiving device; the OnePlus Pad is the default (pixel6_trx_ber.py passes its own)"""
    def __init__(self, **kw):
        self.__dict__.update(kw)


PAD_DEV = Device(name="OnePlus Pad", short="Pad", phone=PAD.PHONE, out=OUT, prefix="pad_trx", decoder=DP, read_s=0.150,
                 rates="2", carrier=37.4e3, fsk_zero=156.25e3, synth_rates="2,3,4,5,6", synth=None,
                 rates_help="the Pad reads ~6.7 maps/s; synth: cc4 clean to 6 baud, uncoded errors from 3",
                 carrier_help="37.4 kHz", fsk_help="156.25 kHz: Pad no-response band 129-172 kHz",
                 mods_ok=("ook", "fsk", "dbpsk"))
DEV = PAD_DEV


# ================================================================ plan
def mods_of(a):
    mods = [m.strip() for m in a.mods.split(",") if m.strip()]
    if not mods or any(m not in DEV.mods_ok for m in mods):
        raise SystemExit(f"--mods: {', '.join(DEV.mods_ok)} for the {DEV.name}")
    return mods


def tones(a):
    """AD9833 actual carrier (+ FSK zero tone) [Hz]"""
    hz = lambda f: STX.fword(f) * STX.MCLK / 268435456.0
    return [hz(a.carrier)] + ([hz(a.fsk_zero)] if "fsk" in mods_of(a) else [])


def runs_of(a):
    keys = B.send_list(a)
    per_round = sum(TV2.air_len(k) for k in keys)
    if per_round * a.rounds_per_rec > B.MAXW:
        raise SystemExit(f"{a.rounds_per_rec} rounds = {per_round * a.rounds_per_rec} symbols > {B.MAXW} schedule words")
    runs = []
    for rate in [float(r) for r in a.rates.split(",")]:
        seq = TV2.build_seq(keys, a.rounds, a.seed, int(round(rate)))     # same frames for every modulation
        K = len(keys)
        for mod, k, r0 in [(m, k, r0) for m in mods_of(a) for k, r0 in enumerate(range(0, a.rounds, a.rounds_per_rec))]:
            r1 = min(a.rounds, r0 + a.rounds_per_rec)
            f0, f1 = r0 * K, r1 * K
            s0, s1 = int(seq["start"][f0]), int(seq["start"][f1 - 1] + seq["length"][f1 - 1])
            sl = dict(keys=keys, enc=seq["enc"][f0:f1], start=seq["start"][f0:f1] - s0, length=seq["length"][f0:f1],
                      u=seq["u"][f0:f1], phase=seq["phase"][s0:s1], frame_index0=f0)
            runs.append(dict(mod=mod, rate=rate, chunk=k, rounds=(r0, r1), seq=sl, per_round=per_round))
    return runs


def show_plan(a, runs):
    keys = B.send_list(a)
    carrier = tones(a)[0]
    tot = sum(len(r["seq"]["phase"]) * B.sym_us(r["rate"]) / 1e6 + LEAD_S + TAIL_S + 8 for r in runs)
    print(f"trx frames: {', '.join(keys)} ({sum(TV2.air_len(k) for k in keys)} symbols per round), {a.rounds} rounds = "
          f"{a.rounds} frames of each per (modulation, rate), seed {a.seed}; carrier (state 1) {carrier:.1f} Hz"
          + (f", FSK state 0 {tones(a)[1]:.1f} Hz" if "fsk" in mods_of(a) else "") + f", wave {a.wave}")
    for mod, rate in sorted({(r["mod"], r["rate"]) for r in runs}, key=lambda x: (x[1], x[0] != "ook")):
        rr = [r for r in runs if r["rate"] == rate and r["mod"] == mod]
        s = sum(len(r["seq"]["phase"]) for r in rr)
        print(f"  {mod.upper()} {1e6 / B.sym_us(rate):6.3f} baud ({B.sym_us(rate) / 1e3:.0f} ms/symbol, ~"
              f"{B.sym_us(rate) / 1e6 / DEV.read_s:.1f} {DEV.short} reads): {len(rr)} recording(s) of "
              f"{len(rr[0]['seq']['phase']) * B.sym_us(rate) / 1e6:.0f} s, {s * B.sym_us(rate) / 1e6:.0f} s on air")
    print(f"{len(runs)} recordings, about {tot / 60:.0f} min with the pauses")
    return carrier


# ================================================================ record
def pad_connect(a):
    exe = SW.find_adb(a.adb)
    serial = DEV.phone.resolve_serial(exe, a.serial)
    if not serial:
        raise SystemExit(f"no adb device online - plug in the {DEV.name}, unlock it, accept USB debugging")
    adb = SW.Adb(exe, serial)
    model = adb.shell("getprop ro.product.model").strip()
    device = adb.shell("getprop ro.product.device").strip()
    if not DEV.phone.model_ok(f"{model} {device}"):
        raise SystemExit(f"{serial} is a {model} ({device}), not a {DEV.name} - pass --serial. Nothing was transmitted.")
    g = DEV.phone.read_once(adb)
    print(f"{DEV.name} ({model}, {serial}): map {g.shape[0]} x {g.shape[1]} readable (root ok, screen on)")
    return adb, model, serial


def one_run(tx, tx_info, adb, pad, a, r, baseline=None):
    sl, rate, mod = r["seq"], r["rate"], r["mod"]
    sched, plan = B.schedule(mod, rate, sl["phase"], a)
    got = tx.upload_schedule(sched)            # also 'psk 0': no DBPSK overlay left from an earlier recording
    pb = None
    if mod == "dbpsk":                          # as stent_trx_ber.one_run / trx: firmware bit string of phase flips
        flips = (sl["phase"] ^ np.concatenate([[0], sl["phase"][:-1]])).astype(np.uint8)
        pb = B.load_bits(tx, flips)
        rr = tx.ask(f"psk {plan['sym_us']} buf", r"^# psk")
        if f"sym_us={plan['sym_us']}" not in rr or "src=buf" not in rr:
            raise RuntimeError(f"psk buf not taken: {rr!r}")
    Ts = plan["sym_us"] / 1e6
    print(f"  uploaded: {got}" + (f", bit string {pb}" if pb else ""))
    if not a.yes:
        input(f"  finger ON the {DEV.short}, still, electrode on (~{LEAD_S + plan['total_s'] + TAIL_S + 1:.0f} s) - "
              "press Enter to start ")

    def at(s):
        if s < 0:
            return "off (lead)"
        k = int(s // Ts)
        return "done" if k >= len(sl["phase"]) else f"symbol {k + 1}/{len(sl['phase'])} {'ON ' if sl['phase'][k] else 'off'}"

    status = None
    try:
        off0 = SW.clock_offset(adb)
        Fm, t_dev, t_host, t_first, skipped, F_raw = PS.record(DEV.phone, adb, LEAD_S, plan["total_s"], TAIL_S,
                                                                lambda: tx.go()[0], at, baseline)
        off1 = SW.clock_offset(adb)
    finally:
        try:
            status = tx.status()
        except Exception as e:                  # noqa: BLE001
            status = dict(error=str(e))
        tx.off()
    if len(Fm) < 20 or t_first is None:
        print("  too few reads recorded - nothing saved")
        return None, True
    off = 0.5 * (off0["offset_s"] + off1["offset_s"])
    good = t_dev > 0
    t = np.where(good, t_dev - off, t_host)
    stopped = t[-1] < t_first + plan["total_s"] - 1.0
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = DEV.out / f"{DEV.prefix}_{a.label}_{mod}_{round(1e6 / plan['sym_us']):g}baud_c{r['chunk']}_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    extra = dict(F_raw=F_raw, baseline=baseline) if baseline is not None else {}
    np.savez_compressed(d / "frames.npz", F=Fm, t=t, t_host=t_host, t_dev=t_dev, **extra)
    meta = dict(mode=f"{DEV.prefix}_ber", device=DEV.name, modulation=mod, args=vars(a), plan=plan, t_first=t_first, pad_model=pad,
                tx=tx_info, tx_status=status, clock_offset_start=off0, clock_offset_end=off1, clock_offset_used_s=off,
                reads=int(len(t)), frames_without_device_stamp=int((~good).sum()), skipped_frames=skipped,
                send=sl["keys"], seed=a.seed, seq_k=int(round(rate)), chunk=r["chunk"], round0=r["rounds"][0],
                round1=r["rounds"][1], frame_index0=int(sl["frame_index0"]), pb=pb, current_ma=a.current_ma, site=a.site,
                condition=a.condition, decoder=DEV.decoder.VERSION, decoder_rules_sha=DEV.decoder.RULES_SHA,
                baseline=(f"finger-off median of {DEV.phone.baseline_frames} reads at session start; F = raw - baseline"
                          if baseline is not None else None), tool_sha=TOOL_SHA)
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    B.save_seq(str(d), sl)
    ev, want = status.get("events", "?"), f"{plan['n_steps']}/{plan['n_steps']}"
    print(f"  saved {d}   transmitter: state {status.get('state')}, events {ev}"
          + ("" if ev == want and status.get("state") == "done" else f"   ** expected done, {want} **")
          + (f", psk_src {status.get('psk_src')}" if mod == "dbpsk" else ""))
    res = DEV.decoder.decode_run(str(d))
    print(f"  quick look: state errors {res['state_errors']}/{res['state_symbols']}, delay {res['delay_ms']:.0f} ms, "
          + (f"phase visible? r = {res['dbpsk_r']:+.2f} (0 = no)" if mod == "dbpsk" else f"timing r {res['timing_r']:.2f}"))
    return d, stopped


def run(a):
    runs = runs_of(a)
    show_plan(a, runs)
    if not a.phantom:
        if a.current_ma is None:
            raise SystemExit("person-connected run: give --current-ma (the network's nominal current) so it is checked "
                             "against the IEC 60601-1 limit; --phantom if no person is connected")
        PS.check_current(tones(a), a.current_ma)
        print(f"  {a.current_ma:g} mA is within the IEC 60601-1 type-BF limit at "
              + ", ".join(f"{f / 1e3:.1f} kHz ({PS.iec_limit_ma(f):.2g} mA)" for f in tones(a)))
    adb, pad, _ = pad_connect(a)
    baseline = None
    if DEV.phone.baseline_frames:               # raw panel map (Pixel stm_fts_cmd 13): one finger-off baseline per session
        input(f"BASELINE: lift your finger OFF the {DEV.name} screen (screen on, nothing touching it), then press Enter ")
        baseline = PS.take_baseline(DEV.phone, adb, DEV.phone.baseline_frames)
        print(f"  baseline = median of {DEV.phone.baseline_frames} finger-off reads")
        input(f"Finger ON the {DEV.name} now and keep it still for the whole session, then press Enter ")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    tx, info = STX.open_configured(a)
    print(f"Transmitter: {info['line']}")
    if "dbpsk" in mods_of(a):
        try:
            tx.ask("pbinfo", r"^# pb n=")
        except TimeoutError:
            tx.close()
            raise SystemExit("this TeensyTRX has no bit-string commands (pb, psk ... buf) - flash "
                             "trx_psk_node/transceiver/firmware/TeensyTRX, or leave dbpsk out of --mods")
    done = []
    try:
        for i, r in enumerate(runs):
            print(f"\n[{i + 1}/{len(runs)}] {r['mod'].upper()} {1e6 / B.sym_us(r['rate']):g} baud, chunk {r['chunk']} "
                  f"(rounds {r['rounds'][0]}-{r['rounds'][1] - 1})")
            d, stopped = one_run(tx, info, adb, pad, a, r, baseline)
            if d:
                done.append(str(d))
            if stopped:
                print("  recording ended early (Ctrl+C, screen off or adb lost) - stopping the session")
                break
    finally:
        tx.close()
    if done:
        stamp = "_".join(os.path.basename(done[0]).rsplit("_", 2)[-2:])
        analyse(done, str(DEV.out / f"trxber_{a.label}_{stamp}"), a.current_ma)


# ================================================================ analysis
def collect(dirs):
    rows, runs = [], []
    for d in dirs:
        print(f"decoding {os.path.basename(os.path.normpath(d))} ...", flush=True)
        res = DEV.decoder.decode_run(d)
        runs.append(res)
        for fr in res["frames"]:
            for dec, r in fr["decoders"].items():
                rows.append(dict(run=res["run"], modulation=res["modulation"], baud=round(res["baud"], 3), frame=fr["frame"],
                                 decoder=dec, i=fr["i"], round=fr["round"], erasures=fr["erasures"],
                                 reference_ok=fr["reference_ok"], pre_err=fr["pre_err"], code_bits=fr["code_bits"],
                                 info_err=r["info_err"], info_bits=TV2.N_INFO, frame_err=r["frame_err"],
                                 crc_ok=r["crc_ok"], crc_missed=r["crc_missed"]))
    return rows, runs


def analyse(dirs, out, current_ma=None, title=None):
    rows, runs = collect(dirs)
    if not rows:
        print("no decodable frames")
        return None
    summ = B.summarise(rows, runs)
    B.write_tables(out, rows, runs, summ, current_ma)
    Dm = DEV.decoder
    json.dump(dict(device=DEV.name, decoder=Dm.VERSION, rules_sha=Dm.RULES_SHA, frozen_at=Dm.FROZEN_AT,
                   tablet_rules_sha=Dm.TABLET_SHA, pad_rules_sha=getattr(Dm, "PAD_RULES_SHA", None),
                   trx_shas=DP.DT.TRX_SHAS, current_ma=current_ma,
                   runs=[{k: v for k, v in r.items() if k != "frames"} for r in runs], summary=summ),
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    B.plot(summ, os.path.join(out, "fec_before_after"),
           title or f"{DEV.name} through body, carrier {runs[0]['carrier_hz'] / 1e3:.1f} kHz"
           + (f" (FSK 0 = {runs[-1]['zero_hz'] / 1e3:.2f} kHz)" if any(r["modulation"] == "fsk" for r in runs) else "")
           + (f", {current_ma:g} mA" if current_ma else "") + f" - {os.path.basename(out)}, decoder {Dm.RULES_SHA}")
    B.report(summ)
    for r in runs:
        tr = f"phase r {r['dbpsk_r']:+.2f}" if r["modulation"] == "dbpsk" else f"timing r {r['timing_r']:.2f}"
        print(f"  {r['run']}: delay {r['delay_ms']:.0f} ms, clock {r['clock_ppm']:.0f} ppm, {tr}, "
              f"state BER {r['state_errors']}/{r['state_symbols']}, erasures {r['erasures']}, "
              f"{r['reads_per_symbol']:.1f} reads/symbol")
    print(f"\nSaved {out}/ (summary.csv/json, frames.csv, table.html, fec_before_after.png/.pdf)")
    return summ


def resolve(what):
    dirs = []
    for p in what:
        if os.path.exists(os.path.join(p, "summary.json")) and os.path.basename(os.path.normpath(p)).startswith("trxber_"):
            dirs += [str(DEV.out / r["run"]) for r in json.load(open(os.path.join(p, "summary.json")))["runs"]]
        else:
            dirs += [q for q in sorted(glob.glob(p)) if os.path.exists(os.path.join(q, "tx_seq.npz"))]
    return dirs


# ================================================================ self-test
def synth_reads(phase, Ts, rng, d_on=(0.12, 0.17), d_off=(0.16, 0.23), p_false=0.04, p_miss=0.0, stall_p=0.0):
    """Pad-like reads, from the 1-500 kHz sweep (hand_1k_500k_20261114_211114, clean carriers): a read every 150 ms
    (p5-p95 132-156 ms); the map refreshes from d_on after a carrier ON edge until d_off after the OFF edge
    (measured: P(changed) 0.17 at +0.05-0.15 s, 1.0 from +0.15 s; 1.0 until +0.15 s after OFF, 0.8 at +0.15-0.2,
    0.2 at +0.2-0.25); ~81 % of the nodes change on a refreshing read; false refreshes p_false (sweep OFF windows
    0.04); stall_p = chance per read of a 1-3 s adb stall (none seen in the sweeps)."""
    lead, n = LEAD_S, len(phase)
    edges = np.flatnonzero(np.diff(np.r_[0, phase, 0]))
    on_iv = [(lead + edges[i] * Ts + rng.uniform(*d_on), lead + edges[i + 1] * Ts + rng.uniform(*d_off))
             for i in range(0, len(edges), 2)]
    t, x = [], 0.0
    end = lead + n * Ts + TAIL_S
    while x < end:
        x += rng.normal(0.150, 0.007) + (rng.uniform(1, 3) if rng.random() < stall_p else 0.0)
        t.append(x)
    t = np.array(t)
    act = np.zeros(len(t), bool)
    for a0, a1 in on_iv:
        act |= (t >= a0) & (t < a1)
    ch = np.where(act, rng.random(len(t)) >= p_miss, rng.random(len(t)) < p_false)
    Fm = np.zeros((len(t), 38, 54), np.int16)
    cur = rng.integers(-5, 6, (38, 54)).astype(np.int16)
    for k in range(len(t)):
        if ch[k]:
            m = rng.random((38, 54)) < (0.81 if act[k] else rng.uniform(0.1, 0.8))
            cur = cur + np.where(m, rng.choice([-1, 1], (38, 54)), 0).astype(np.int16)
        Fm[k] = cur
    return t, Fm


def synth(a):
    rng = np.random.default_rng(7)
    keys = B.send_list(a)
    cases = [("sweep-like (false 4 %)", dict()), ("noisy (false 15 %, 5 % missed, stalls)",
                                                 dict(p_false=0.15, p_miss=0.05, stall_p=0.01))]
    print(f"simulated Pad reads, frames {', '.join(keys)}, {a.rounds} rounds per rate; BER before FEC = code bits of "
          "cc4, uncoded = none_full info bits")
    for name, kw in cases:
        print(f"\n{name}")
        for rate in [float(r) for r in a.synth_rates.split(",")]:
            seq = TV2.build_seq(keys, a.rounds, a.seed, int(round(rate)))
            sq = dict(keys=np.array(keys), enc=seq["enc"], start=seq["start"], length=seq["length"],
                      u=np.packbits(seq["u"], 1), phase=np.packbits(seq["phase"]), n_phase=len(seq["phase"]))
            Ts = B.sym_us(rate) / 1e6
            t, Fm = synth_reads(seq["phase"], Ts, rng, **kw)
            meta = dict(plan=dict(sym_us=B.sym_us(rate), carrier_hz=a.carrier), t_first=LEAD_S, frame_index0=0, round0=0)
            res = DP.decode_arrays(t, DP.feature(Fm), meta, sq)
            agg = {}
            for fr in res["frames"]:
                g = agg.setdefault(fr["frame"], [0, 0, 0, 0, 0])
                dd = fr["decoders"][fr["frame"]]
                g[0] += fr["pre_err"]; g[1] += fr["code_bits"]; g[2] += dd["info_err"]; g[3] += TV2.N_INFO; g[4] += dd["frame_err"]
            txt = "   ".join(f"{k}: before {g[0]}/{g[1]}, after {g[2]}/{g[3]}, frames bad {g[4]}/{a.rounds}"
                           for k, g in agg.items())
            print(f"  {1e6 / B.sym_us(rate):5.2f} baud ({res['reads_per_symbol']:.1f} reads/sym, delay "
                  f"{res['delay_ms']:.0f} ms, r {res['timing_r']:.2f}): state {res['state_errors']}/{res['state_symbols']}   {txt}")


def main(dev=None):
    global DEV
    DEV = dev or PAD_DEV
    ap = argparse.ArgumentParser(description=f"TeensyTRX -> body -> {DEV.name}: trx-frame BER, OOK / FSK, before / after FEC")
    ap.add_argument("cmd", choices=("plan", "run", "analyse", "synth"))
    ap.add_argument("what", nargs="*", help="analyse: run folders (globs ok) or a trxber_<label>_<time> folder")
    ap.add_argument("--label", default="hand")
    ap.add_argument("--mods", default="ook,fsk", help=f"{', '.join(DEV.mods_ok)} (default ook,fsk)")
    ap.add_argument("--rates", default=DEV.rates, help=f"symbol rates [baud] (default {DEV.rates}; {DEV.rates_help})")
    ap.add_argument("--carrier", type=float, default=DEV.carrier,
                    help=f"state-1 carrier (OOK carrier / FSK 1 tone) [Hz] (default {DEV.carrier_help})")
    ap.add_argument("--fsk-zero", type=float, default=DEV.fsk_zero, help=f"FSK state-0 tone [Hz] (default {DEV.fsk_help})")
    ap.add_argument("--send", default=DEFAULT_SEND, help=f"trx encodings (default {DEFAULT_SEND}; 'all' = every one)")
    ap.add_argument("--rounds", type=int, default=3, help="frames of each encoding per rate (default 3)")
    ap.add_argument("--rounds-per-rec", type=int, default=1, help="rounds per recording (default 1)")
    ap.add_argument("--seed", type=int, default=1027, help="trx frame order / info bits seed (trx default 1027)")
    ap.add_argument("--current-ma", type=float, help="nominal current of the network [mA]; required for a person-"
                    "connected run, checked against the IEC 60601-1 type-BF limit at the carrier")
    ap.add_argument("--phantom", action="store_true", help="no person connected: no current check")
    ap.add_argument("--site", default="hand2finger")
    ap.add_argument("--condition", default="still")
    ap.add_argument("--wave", default="square", choices=("square", "sine"))
    ap.add_argument("--port", help="TeensyTRX serial port (default: first Teensy)")
    ap.add_argument("--serial", help=f"adb serial of the {DEV.name} (default: auto)")
    ap.add_argument("--adb", help="adb path")
    ap.add_argument("--yes", action="store_true", help="no Enter prompt before each recording")
    ap.add_argument("--out", help="analyse: output folder")
    ap.add_argument("--synth-rates", default=DEV.synth_rates)
    a = ap.parse_args()
    if a.cmd == "synth":
        (DEV.synth or synth)(a)
    elif a.cmd == "plan":
        show_plan(a, runs_of(a))
        print("IEC 60601-1 type-BF limit: " + ", ".join(f"{f / 1e3:.1f} kHz {PS.iec_limit_ma(f):.2g} mA" for f in tones(a)))
        if a.current_ma is not None:
            PS.check_current(tones(a), a.current_ma)
            print(f"  {a.current_ma:g} mA is within it at every tone")
    elif a.cmd == "run":
        run(a)
    else:
        dirs = resolve(a.what)
        if not dirs:
            raise SystemExit("no run folders found")
        cur = json.load(open(os.path.join(dirs[0], "meta.json"))).get("current_ma")
        analyse(dirs, a.out or str(DEV.out / f"trxber_reanalysed_{datetime.datetime.now():%Y%m%d_%H%M%S}"), cur)
    return 0


if __name__ == "__main__":
    sys.exit(main())
