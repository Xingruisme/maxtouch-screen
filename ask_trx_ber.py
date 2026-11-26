#!/usr/bin/env python3
"""
OOK vs 2-ASK vs 4-ASK BER and PER to the maXTouch tablet, ONE carrier (159.585 kHz), the SAME frames
(2026-11-23; three on-air schemes since 2026-11-23 evening, before any ASK recording - until then '2-ASK' meant OOK).

Frame (--send): since 2026-11-24 evening the default is the i2o frame WITHOUT preamble, i2o_frame.py 'i2o_np' -
src 4, type 4, frame no. 4 | payload 20 | CRC-16, 48 info bits, rate-1/2 K=4 code, 96 symbols, nothing uncoded and
nothing known to the receiver. --send i2o = the same frame behind its 32-symbol sync word (128 symbols, the default
earlier that day), --send cc4 = the 256-symbol i2i frame of trx_psk_node. One session = one frame type.
A frame without preamble is decoded by decode_trx_nopre (blind levels from the data symbols; the frozen decoders take
every reference from the sync and cannot decode it). analyse --ignore-sync decodes recordings of frames WITH a sync the
same way, as if the sync had not been sent (its symbols are never looked at) - the with / without comparison on one
recording. The frame start comes from the known schedule in every case, as before.

  OOK    1 state per symbol: state 1 = full level, state 0 = carrier off   (binary ASK at 100 % depth)
  2-ASK  1 state per symbol: state 1 = full level, state 0 = level index --ask2-index (default 2 = 33 %); the carrier
         is never off
  4-ASK  2 states per symbol on 4 transmit levels of the same carrier: full | level index 1 (67 %) | index 2 (33 %) |
         off, Gray-mapped (1,1) (1,0) (0,0) (0,1) strongest first (decode_trx_mfsk's mapping)
The levels come from the ADG729 level switch of "ASK Transceiver Wiring" (2026-11-24): AD9833 -> Ra 1 k -> 3 x 330 Ohm
ladder -> VREF, one ladder node per symbol to E+, switched over I2C by the TeensyTRX ask2 firmware
(firmware/TeensyTRX_ask2/TeensyTRX_ask2.ino; schedule word bits 28-29 = level index): index 0 = 100 %, 1 = 67 %,
2 = 33 %, 'off' = 0 % (DA on VREF; the AD9833 keeps running, so the carrier is continuous from symbol to symbol).
Index 0 = the level of the old Ra/Rb network, so no ASK level is above the --current-ma setting. Until 2026-11-23 the
levels came from a 74HC4066 attenuator (ask1 firmware); its levels files are not accepted any more.

Compared at the same BIT rate: OOK / 2-ASK baud = bit rate, 4-ASK baud = bit rate / 2 (--schemes picks which of the
three are sent). Every recording goes through
stent_trx_ber.one_run unchanged (upload, tablet set-up + finger-on calibration, lead, sgo, tail, transmitter off,
tablet restore). Decoders, both frozen and unchanged (run through a read-only view of the run folder whose meta.json
names the receiver rule set: OOK / 2-ASK -> 'ook', 4-ASK -> 'fsk4'; result files carry modulation = ook/ask2/ask4):
  OOK   -> decode_trx_tablet (trx-tablet-1, the OOK decoder, result_trx.json) AND decode_trx_mfsk (M = 2, result_mfsk.json)
  2-ASK -> decode_trx_mfsk (M = 2, ladder = full / the measured --ask2-index level)
  4-ASK -> decode_trx_mfsk (M = 4, ladder = the measured levels)
PER = packet (frame) error rate: a packet is in error if any of its info bits (i2o: 48 = 32 data + CRC-16; trx
frames: 96 = 80 data + CRC-16) is wrong after decoding, or if it was sent but not decoded (lost). Also reported: CRC pass, CRC missed (CRC ok but bits wrong),
BER before / after FEC, symbol errors.

   python ask_trx_ber.py levels --current-ma 1          measure the 4 switch levels at E+ (A0), same set-up as the run
   python ask_trx_ber.py send   --current-ma 1 [--schemes ask2,ask4] [--bitrates 40] [--bits 512]
                                                        transmit only (PRBS9 by the firmware's ask2 / ask4 commands,
                                                        no tablet recording) - for a scope or a first look
   python ask_trx_ber.py plan   [--bitrates 20,30,40,50,60] [--rounds 50] [--send cc4] [--schemes ook,ask2,ask4]
   python ask_trx_ber.py run    --label body_ask --condition through_body --current-ma 1     (the user runs it)
   python ask_trx_ber.py rx     --label body_ask --condition through_body --current-ma 1     (THIS laptop: maXTouch only)
   python ask_trx_ber.py tx     --label body_ask --current-ma 1     (OTHER laptop: Teensy only - no shared ground)
        same arguments on both, and the same 'plan id' line on both screens (it covers every symbol that is sent);
        --auto on both (with the same --gap, default 30 s): only the first recording is started by hand as below;
        after it tx sends each next recording --gap s after the end of the one before and rx times its calibration
        and recording from the last start it found - nothing more to press until the session ends;
        per run: rx calibrates, prints >>> START TX NOW <<<, then Enter on the tx laptop within
        --start-window s (default 10). rx finds each frame start by correlating the known symbol levels (1 ms steps);
        the frozen decoders refine it as usual (meta.json: split_tx, start_corr).
   python ask_trx_ber.py analyse data_maxtouch/stent_trx_<label>_ask*
   python ask_trx_ber.py synth  [--ladder-db 0,-9,-20,-40]                                    (simulated panel)
Output: run folders data_maxtouch/stent_trx_<label>_{ook,ask2,ask4}_<baud>baud_c<k>_<time>/, session summary
data_maxtouch/askber_<label>_<time>/ (summary.csv/json, frames.csv, table.html, ask2_vs_ask4.png/.pdf).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import shutil
import signal
import sys
import tempfile
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stent_trx_ber as B                       # noqa: E402  (one_run, chunks, schedule, save_seq, synth_capture)
import decode_trx_mfsk as MF                    # noqa: E402
import decode_trx_nopre as NP                   # noqa: E402  (blind level decoder: frames without preamble)
import i2o_frame as I2O                         # noqa: E402  (the short i2o frame; registers itself with the trx tables)
I2O.register()
T, R, TV2, STX = B.D, B.R, B.TV2, B.STX
FW_TAG = "ask2"                                 # TeensyTRX_ask2.ino identifies itself as "TeensyTRX ask2"
DESIGN_DB = [0.0, -3.5, -9.5]                   # open-circuit ladder 100 / 67 / 33 % (plan / synth before a measurement)
CARRIER = 159585.0
AMP_IDLE = 3                                    # level bits of an 'off' word (frequency 0 -> switch on 0 %)
OFF_DB = -40.0                                  # nominal level of 'carrier off' (decode_trx_mfsk treats < NULL_B as a null)
AMP4 = (0, 1, 2, None)                          # 4-ASK level -> level index (None = 'off' word, 0 %)
AMP_OOK = (0, None)
MODS = ("ook", "ask2", "ask4")                  # on-air schemes, in the order they are sent at each bit rate
MOD_NAME = dict(ook="OOK", ask2="2-ASK", ask4="4-ASK")
SCHEMES = (("ook", "OOK, frozen tablet decoder"), ("ook", "OOK, level decoder M=2"),
           ("ask2", "2-ASK (full / reduced level), level decoder M=2"), ("ask4", "4-ASK, level decoder M=4"),
           ("ook", "OOK, blind level decoder M=2"), ("ask2", "2-ASK (full / reduced level), blind level decoder M=2"),
           ("ask4", "4-ASK, blind level decoder M=4"))
BLIND = {m: n for m, n in SCHEMES[4:]}          # no preamble (or sync ignored): decode_trx_nopre
COLORS = {SCHEMES[0][1]: "#2a78d6", SCHEMES[1][1]: "#1baf7a", SCHEMES[2][1]: "#8e5bd0",
          SCHEMES[3][1]: "#eb6834", SCHEMES[4][1]: "#1baf7a", SCHEMES[5][1]: "#8e5bd0",
          SCHEMES[6][1]: "#eb6834"}             # fixed per on-air scheme, never cycled
FRAMES = (I2O.KEY_NP, I2O.KEY, "cc4", "none_full")
IGNORE_SYNC = False                             # analyse --ignore-sync
AS_RULES = dict(ook="ook", ask2="ook", ask4="fsk4")   # receiver rule set each run is decoded with


# ================================================================ schedule
def word(idx, carrier):
    return (AMP_IDLE << 28) if idx is None else ((idx << 28) | STX.fword(carrier))


def schedule_ask(mod, baud, phase, a):
    M = 4 if mod == "ask4" else 2
    s = B.sym_us(baud)
    lv = MF.levels_of_states(phase, M)
    amp = AMP4 if M == 4 else AMP_OOK if mod == "ook" else (0, a.ask2_index)
    words = [word(k, a.carrier) for k in amp]
    hz = STX.fword(a.carrier) * STX.MCLK / 268435456.0
    lad = list(a.ladder_db) if M == 4 else [0.0, OFF_DB] if mod == "ook" else [0.0, a.ladder_db[a.ask2_index]]
    plan = dict(modulation=mod, sym_us=s, baud=1e6 / s, bits_per_symbol=M // 2, bit_rate=(M // 2) * 1e6 / s,
                carrier_hz=hz, zero_hz=0.0, amp_index=[(-1 if k is None else k) for k in amp], ladder_db=lad,
                levels_file=getattr(a, "levels_file", None), n_symbols=len(lv), n_states=len(phase),
                total_s=len(lv) * s / 1e6, n_steps=len(lv))
    return [(s, 1, [words[i] for i in lv])], plan


_SCHEDULE = B.schedule


def schedule(mod, rate, phase, a):
    return schedule_ask(mod, rate, phase, a) if mod in MODS else _SCHEDULE(mod, rate, phase, a)


B.schedule = schedule                           # stent_trx_ber.one_run looks `schedule` up in its own module


# ================================================================ plan
def ber_args(a):
    """the namespace stent_trx_ber.one_run / R.Chip / R.record read"""
    return types.SimpleNamespace(**dict(vars(a), mods=",".join(a.schemes), rates=a.bitrates, fsk_zero=None))


def chunks_even(a, br):
    """stent_trx_ber.chunks with the rounds spread evenly over the recordings (50 frames = 25 + 25, not 42 + 8): the
    blind decoder estimates its levels per recording, so no recording should be much shorter than the others. The
    frames themselves do not depend on the split."""
    per_round = sum(TV2.air_len(k) for k in B.send_list(a))
    rpc = max(1, B.MAXW // per_round)
    n = -(-a.rounds // rpc)
    keep, B.MAXW = B.MAXW, max(per_round, -(-a.rounds // n) * per_round)
    try:
        return B.chunks(a, br)
    finally:
        B.MAXW = keep


def plan_all(a):
    runs = []
    for br in [float(x) for x in a.bitrates.split(",")]:
        ch, per_round = chunks_even(a, br)                          # frames depend on round(bit rate) only
        for mod in a.schemes:
            for k, (r0, r1, sl) in enumerate(ch):
                runs.append(dict(mod=mod, rate=br / 2 if mod == "ask4" else br, bit_rate=br, chunk=k, rounds=(r0, r1),
                                 seq=sl, per_round=per_round))
    return runs


def plan_id(a, runs):
    """fingerprint of everything that goes on air and of the frames' info bits, over all recordings of the plan.
    The two laptops of the split set-up ('tx' / 'rx') must print the same one: it differs if the arguments, the
    frame code (i2o_frame, trx_psk_node) or the random generator differ. The ladder [dB] is not part of it - it
    only tells the receiver where to expect the levels, it does not change what is sent."""
    import hashlib
    h, ba = hashlib.sha256(), ber_args(a)
    for r in runs:
        sched, plan = schedule(r["mod"], r["rate"], r["seq"]["phase"], ba)
        h.update(f"{r['mod']}|{plan['sym_us']}|{r['chunk']}|{r['rounds']}|".encode())
        for step_us, reps, words in sched:
            h.update(np.asarray([step_us, reps] + list(words), np.uint64).tobytes())
        h.update(np.ascontiguousarray(r["seq"]["u"]).tobytes())
    return h.hexdigest()[:8]


def show_plan(a, runs):
    keys = B.send_list(a)
    if set(keys) & set(I2O.KEYS):
        print(frame_note(keys))
    print(f"frames: {', '.join(keys)} ({sum(TV2.air_len(k) for k in keys)} states per round), {a.rounds} rounds, "
          f"seed {a.seed}; same frames for {', '.join(MOD_NAME[m] for m in a.schemes)} at each bit rate")
    print(f"carrier {a.carrier / 1e3:g} kHz   OOK: full / off   2-ASK: full / index {a.ask2_index} "
          f"({a.ladder_db[a.ask2_index]:+.1f} dB)   4-ASK ladder: "
          + "  ".join(f"{'off' if k is None else f'index {k}'} ({db:+.1f} dB)" for k, db in zip(AMP4, a.ladder_db))
          + f"   [{a.levels_file or 'ladder given on the command line'}]")
    tot = 0.0
    for r in runs:
        s = len(r["seq"]["phase"]) / (2 if r["mod"] == "ask4" else 1) * B.sym_us(r["rate"]) / 1e6
        tot += s + a.lead + 0.5 + B.TAIL_S + 4
        print(f"  {r['bit_rate']:4g} bit/s  {MOD_NAME[r['mod']]:5s} {r['rate']:5g} baud  chunk {r['chunk']}"
              f"  {len(r['seq']['phase'])} states  {s:6.1f} s on air")
    print(f"{len(runs)} recordings, about {tot / 60:.0f} min")
    a.plan_id = plan_id(a, runs)
    print(f"plan id {a.plan_id}  (split set-up: 'tx' and 'rx' must show the same id, or the receiver decodes against "
          "other frames than were sent)")


# ================================================================ levels (the 4 switch levels at E+)
def line_amp(v, hz, sps=1e6):
    """carrier amplitude [ADC counts] = Hann-windowed spectrum peak within +-2 kHz of hz (clock offsets of the two
    crystals move the line by a few Hz; the window's scalloping is the same at every level, so ratios are exact)"""
    w = np.hanning(len(v))
    sp = np.abs(np.fft.rfft((v - v.mean()) * w))
    fr = np.fft.rfftfreq(len(v), 1 / sps)
    sel = np.abs(fr - hz) < 2000
    return 2 * sp[sel].max() / w.sum()


def levels(a):
    need_current(a)
    tx, info = open_ask(a)
    out = []
    try:
        print(tx.off())
        floor = line_amp(tx.a0(), a.carrier)
        for k in range(4):
            print(" ", tx.ask(f"amp {k}", r"^# amp"), "|", tx.tone(a.carrier).strip())
            out.append(line_amp(tx.a0(), a.carrier))
            tx.off()
    finally:
        tx.ask("amp 0", r"^# amp")
        tx.off()
        tx.close()
    db = [20 * np.log10(x / out[0]) for x in out]
    print(f"\ncarrier {a.carrier:g} Hz at A0 (E+), off floor {20 * np.log10(max(floor, 1e-9) / out[0]):+.1f} dB")
    for k in range(4):
        print(f"  index {k} (byte 0x{1 << k:02x}, design {('  0.0', ' -3.5', ' -9.5', '  off')[k]} dB): "
              f"{out[k]:8.1f} counts  {db[k]:+6.1f} dB")
    lad = [0.0, round(db[1], 1), round(db[2], 1), OFF_DB]
    warn = []                                   # design: 0 / -3.5 / -9.5 dB open-circuit, about 0 / -3.1 / -7.3 dB into 500 Ohm
    if not db[0] >= max(db[1:]) + 1:
        warn.append("index 0 is not the strongest - check the ladder / ADG729 1A-4A order (1A = ladder top, next to Ra)")
    if not -7 <= db[1] <= -1.5:
        warn.append(f"index 1 (2A, 67 %) at {db[1]:+.1f} dB, wanted -1.5..-7 dB")
    if not (db[2] <= db[1] - 2.5 and db[2] >= -20):
        warn.append(f"index 2 (3A, 33 %) at {db[2]:+.1f} dB, wanted >= 2.5 dB below index 1 and above -20 dB")
    if not db[3] <= -25:
        warn.append(f"index 3 (4A, VREF = 'off') at {db[3]:+.1f} dB, wanted below -25 dB (4A must sit on the VREF node)")
    for w in warn:
        print("  ** " + w)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(B.DATADIR, f"asklevels_{a.label}_{stamp}.json")
    json.dump(dict(kind="ask_levels", created=stamp, carrier_hz=a.carrier, current_ma=a.current_ma,
                   condition=a.condition, firmware=info["line"], point="A0 = E+ through the TX/RX path",
                   counts=[float(x) for x in out], off_counts=float(floor), db=[float(x) for x in db],
                   ladder4_db=lad, amp_index4=[(-1 if k is None else k) for k in AMP4], warnings=warn),
              open(path, "w"), indent=1)
    print(f"\n4-ASK ladder {lad}  saved {os.path.relpath(path, HERE)}" + ("  (NOT usable - see **)" if warn else ""))


def latest_levels():
    """newest levels file measured with THIS circuit (ask2 firmware); the 74HC4066 attenuator's files do not count"""
    fs = [f for f in sorted(glob.glob(os.path.join(B.DATADIR, "asklevels_*.json")), key=os.path.getmtime)
          if FW_TAG in json.load(open(f)).get("firmware", "")]
    return fs[-1] if fs else None


def resolve_ladder(a, need=True):
    if a.ladder_db:
        a.ladder_db = [float(x) for x in a.ladder_db.split(",")]
        a.levels_file = None
    else:
        a.levels_file = a.levels or latest_levels()
        if not a.levels_file:
            if need and a.cmd == "rx":          # split set-up: 'levels' needs the Teensy, which is on the other laptop
                raise SystemExit("2-ASK / 4-ASK need the measured levels, and this laptop has none: on the Teensy "
                                 "laptop run  python ask_trx_ber.py levels --current-ma 1  and copy its "
                                 "data_maxtouch/asklevels_*.json here (or give the same 4 values with --ladder-db). "
                                 "OOK alone needs no levels: --schemes ook")
            if need:
                raise SystemExit("no levels measured with the ADG729 circuit yet: run  python ask_trx_ber.py levels "
                                 "--current-ma 1  first (same set-up as the run), or give --ladder-db")
            a.ladder_db = DESIGN_DB + [OFF_DB]                        # design values (simulation / plan only)
            a.levels_file = "design values, not measured"
            return
        L = json.load(open(a.levels_file))
        if FW_TAG not in L.get("firmware", "") and need:
            raise SystemExit(f"{a.levels_file} was measured with another circuit ({L.get('firmware', '?')[:14]}) - "
                             "rerun 'levels'")
        if L.get("warnings") and need:
            raise SystemExit(f"{a.levels_file}: levels not usable ({'; '.join(L['warnings'])}) - fix and rerun 'levels'"
                             "  (OOK alone needs no levels: add --schemes ook)")
        if L.get("warnings"):                                       # plan / synth / OOK-only run: design values
            a.ladder_db, a.levels_file = DESIGN_DB + [OFF_DB], "design values (newest levels file not usable)"
            return
        a.ladder_db = list(L["ladder4_db"])
        a.levels_file = os.path.relpath(a.levels_file, HERE)
    if len(a.ladder_db) != 4 or not all(x > y for x, y in zip(a.ladder_db, a.ladder_db[1:])):
        raise SystemExit(f"4-ASK ladder {a.ladder_db} must be 4 decreasing values [dB]")


def need_current(a):
    if a.current_ma is None:
        raise SystemExit("--current-ma is required (IEC 60601-1 check; the full ASK level = this setting)")
    sys.path.append(os.path.join(HERE, "..", "screen-bloodflow-core-main"))
    import phone_body_sweep as PS                                   # the existing IEC 60601-1 check
    PS.check_current([a.carrier], a.current_ma)


def open_ask(a):
    tx, info = STX.open_configured(a)
    if FW_TAG not in info["line"]:
        tx.off(); tx.close()
        raise SystemExit(f"the Teensy runs {info['line'][:40]!r}, not TeensyTRX {FW_TAG}: flash "
                         "firmware/TeensyTRX_ask2/TeensyTRX_ask2.ino first (other firmware does not drive the ADG729)")
    if "NOT ANSWERING" in info["line"]:
        tx.off(); tx.close()
        raise SystemExit("the ADG729 level switch does not answer on I2C (SDA 18 / SCL 19, 3.3 V, GND): with every "
                         "switch open nothing would be sent. Check the wiring, then  python stent_tx.py id")
    return tx, info


# ================================================================ send (transmit only, firmware ask2 / ask4 commands)
def send(a):
    need_current(a)
    tx, info = open_ask(a)
    print(f"Transmitter: {info['line']}")
    try:
        for br in [float(x) for x in a.bitrates.split(",")]:
            for mod in a.schemes:
                lo = 3 if mod == "ook" else a.ask2_index
                tx.ask(f"asklo {lo}", r"^# asklo")
                line = (f"ask4 {a.carrier:.3f} {br / 2:g} prbs {a.bits}" if mod == "ask4" else
                        f"ask2 {a.carrier:.3f} {br:g} prbs {a.bits}")
                r = tx.ask(line, r"^# ask\d")
                if "refused" in r:
                    raise SystemExit(f"{line!r}: {r}")
                secs = a.bits / br
                print(f"\n{br:g} bit/s {MOD_NAME[mod]}: {line}\n  {r}", flush=True)
                if not tx.wait_done(1.0 + secs + 5):
                    raise SystemExit("  the schedule did not finish in time")
                st = tx.status()
                print(f"  state {st['state']}, steps {st['events']}, failed switch writes {st.get('adg_err')}")
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("stopped (Ctrl+C)")
    finally:
        try:
            tx.ask(f"asklo {a.ask2_index}", r"^# asklo")
        finally:
            tx.close()


# ================================================================ analysis
def view(d, rules):
    """read-only view of run folder d whose meta.json names the receiver rules (the frozen decoders key on it)"""
    v = tempfile.mkdtemp(prefix="askview_")
    for f in os.listdir(d):
        if f not in ("meta.json", "result_trx.json", "result_mfsk.json"):
            os.symlink(os.path.abspath(os.path.join(d, f)), os.path.join(v, f))
    meta = json.load(open(os.path.join(d, "meta.json")))
    json.dump(dict(meta, modulation=rules, modulation_actual=meta["modulation"]), open(os.path.join(v, "meta.json"), "w"))
    return v, meta


def decode_as(d, rules, which):
    v, meta = view(d, rules)
    try:
        res = (T.decode_run(v) if which == "tablet" else NP.decode_run(v, write=False) if which == "nopre" else
               MF.decode_run(v, write=False))
    finally:
        shutil.rmtree(v, ignore_errors=True)
    res = dict(res, run=os.path.basename(os.path.normpath(d)), modulation=meta["modulation"], decoded_with_rules=rules)
    json.dump(res, open(os.path.join(d, dict(tablet="result_trx.json", nopre="result_nopre.json").get(
        which, "result_mfsk.json")), "w"), indent=1, default=float)
    return res


def decode(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    mod = meta["modulation"]
    run_keys = [str(k) for k in np.load(os.path.join(d, "tx_seq.npz"))["keys"]]
    no_sync = [T.frame_layout(k)[0] == 0 for k in run_keys]
    if mod in BLIND and (IGNORE_SYNC or any(no_sync)):
        if not IGNORE_SYNC and not all(no_sync):
            raise SystemExit(f"{d}: frames with and without preamble in one recording")
        out = [(BLIND[mod], decode_as(d, AS_RULES[mod], "nopre"))]
    elif mod == "ook":
        out = [(SCHEMES[0][1], decode_as(d, "ook", "tablet")), (SCHEMES[1][1], decode_as(d, "ook", "mfsk"))]
    elif mod == "ask2":
        out = [(SCHEMES[2][1], decode_as(d, "ook", "mfsk"))]        # M = 2, ladder = full / reduced level from the plan
    elif mod == "ask4":
        out = [(SCHEMES[3][1], decode_as(d, "fsk4", "mfsk"))]
    else:
        raise SystemExit(f"{d}: modulation {mod} is not an OOK / ASK run")
    br = float(round(meta["plan"].get("bit_rate") or 1e6 / meta["plan"]["sym_us"]))
    sq = np.load(os.path.join(d, "tx_seq.npz"))
    keys = [str(k) for k in sq["keys"]]
    sent = [keys[e] for e in sq["enc"]]
    return meta, br, out, sent


def collect(dirs):
    rows, runs, sent = [], [], {}
    for d in dirs:
        print(f"decoding {os.path.basename(os.path.normpath(d))} ...", flush=True)
        meta, br, res, keys_sent = decode(d)
        for scheme, r in res:
            for k in keys_sent:
                sent[(scheme, br, k)] = sent.get((scheme, br, k), 0) + 1
            runs.append(dict(scheme=scheme, run=r["run"], bit_rate=br, baud=r["baud"], timing_r=r.get("timing_r"),
                             symbol_errors=r.get("symbol_errors"), symbols=r.get("symbols_decoded"),
                             state_errors=r.get("state_errors"), state_symbols=r.get("state_symbols"),
                             frames_sent=len(keys_sent), frames_decoded=len(r.get("frames", [])),
                             ladder_db=meta["plan"].get("ladder_db"), levels_file=meta["plan"].get("levels_file")))
            for f in r.get("frames", []):
                dd = f["decoders"][f["frame"]]
                rows.append(dict(scheme=scheme, run=r["run"], bit_rate=br, baud=round(r["baud"], 3), frame=f["frame"],
                                 i=f["i"], pre_err=f["pre_err"], code_bits=f["code_bits"], info_err=dd["info_err"],
                                 info_bits=TV2.N_INFO, packet_err=int(dd["info_err"] > 0), crc_ok=int(dd["crc_ok"]),
                                 crc_missed=int(dd["crc_missed"]), erasures=f["erasures"], node=f.get("node", "")))
    return rows, runs, sent


def summarise(rows, runs, sent):
    out = []
    for scheme in [s[1] for s in SCHEMES]:
        for br in sorted({k[1] for k in sent}):
            for key in FRAMES:
                n_sent = sent.get((scheme, br, key), 0)
                if not n_sent:
                    continue
                rr = [r for r in rows if (r["scheme"], r["bit_rate"], r["frame"]) == (scheme, br, key)]
                bad, lost = sum(r["packet_err"] for r in rr), n_sent - len(rr)
                lo, hi = TV2.wilson(bad + lost, n_sent)
                pe, pb = sum(r["pre_err"] for r in rr), sum(r["code_bits"] for r in rr)
                ie, ib = sum(r["info_err"] for r in rr), sum(r["info_bits"] for r in rr)
                rs = [x for x in runs if x["scheme"] == scheme and x["bit_rate"] == br]
                se, sn = sum(x["symbol_errors"] or 0 for x in rs), sum(x["symbols"] or 0 for x in rs)
                tr = [x["timing_r"] for x in rs if x["timing_r"] is not None]
                out.append(dict(scheme=scheme, bit_rate=br, baud=rs[0]["baud"] if rs else None, frame=key,
                                packets_sent=n_sent, packets_decoded=len(rr), packets_lost=lost,
                                packet_errors=bad + lost, per=(bad + lost) / n_sent, per_lo=lo, per_hi=hi,
                                crc_pass=sum(r["crc_ok"] for r in rr), crc_missed=sum(r["crc_missed"] for r in rr),
                                pre_fec_errors=pe, pre_fec_bits=pb, pre_fec_ber=pe / pb if pb else None,
                                post_fec_errors=ie, post_fec_bits=ib, post_fec_ber=ie / ib if ib else None,
                                symbol_errors=se if sn else None, symbols=sn or None,
                                symbol_error_rate=se / sn if sn else None,
                                timing_r=float(np.mean(tr)) if tr else None,
                                info_bps=TV2.N_INFO * br / TV2.air_len(key)))
    return out


def plot(summ, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted = "#1f2328", "#8a8f98"
    brs = sorted({s["bit_rate"] for s in summ})
    xi = {b: i for i, b in enumerate(brs)}
    panels = []                                 # only the frame types that were sent
    for key in [k for k in FRAMES if any(s["frame"] == k for s in summ)]:
        if key == "none_full":
            panels += [(key, "post", f"{key} (no FEC) - BER"), (key, "per", f"{key} - PER (95 % CI)")]
        else:
            panels += [(key, "pre", f"{key} - BER before FEC (code bits)"), (key, "post", f"{key} - BER after FEC (info bits)"),
                       (key, "per", f"{key} - PER (95 % CI)")]
    panels.append((panels[0][0], "ser", "symbol error rate"))
    ncol = 2 if len(panels) <= 4 else 3
    nrow = -(-len(panels) // ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow), squeeze=False)
    for ax in axs.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (key, what, ttl) in zip(axs.flat, panels):
        top = 0.0
        shown = [n for _, n in SCHEMES if any(s["scheme"] == n for s in summ)]
        for name in shown:
            j = shown.index(name) + (4 - len(shown)) / 2            # markers side by side, centred on the tick
            ss = sorted([s for s in summ if s["scheme"] == name and s["frame"] == key], key=lambda s: s["bit_rate"])
            if not ss:
                continue
            x = [xi[s["bit_rate"]] + (j - 1.5) * 0.08 for s in ss]
            col = COLORS[name]
            if what == "per":
                ax.plot(x, [s["per"] for s in ss], color=col, lw=2, marker="o", ms=5, label=name)
                ax.fill_between(x, [s["per_lo"] for s in ss], [s["per_hi"] for s in ss], color=col, alpha=0.08, lw=0)
                continue
            if what == "ser":
                ss = [s for s in ss if s["symbols"]]
                if not ss:
                    continue
                x = [xi[s["bit_rate"]] + (j - 1.5) * 0.08 for s in ss]
                e, n = [s["symbol_errors"] for s in ss], [s["symbols"] for s in ss]
            else:
                e = [s["pre_fec_errors" if what == "pre" else "post_fec_errors"] for s in ss]
                n = [max(s["pre_fec_bits" if what == "pre" else "post_fec_bits"], 1) for s in ss]
            y = [k / m for k, m in zip(e, n)]                       # linear axis: 0 errors sit on 0
            top = max(top, max(y))
            ax.plot(x, y, color=col, lw=2, label=name)
            for xx, yy, k in zip(x, y, e):
                ax.plot(xx, yy, ls="none", color=col, marker="v" if k == 0 else "o", ms=7 if k == 0 else 5.5,
                        mfc="white" if k == 0 else col)
        if what == "per":
            ax.set_ylim(-0.02, 1.02); ax.set_ylabel("PER")
        else:
            top = max(1.25 * top, 0.01)                             # linear, scaled to the data (at least 0-1 %)
            ax.set_ylim(-0.04 * top, top)
            if top >= 0.5:
                ax.axhline(0.5, color=muted, lw=1, ls=":")
            ax.set_ylabel("SER" if what == "ser" else "BER")
        ax.set_xticks(range(len(brs))); ax.set_xticklabels([f"{b:g}" for b in brs])
        ax.set_xlim(-0.5, len(brs) - 0.5)
        ax.set_xlabel("bit rate [bit/s]")
        ax.set_title(ttl, fontsize=10, color=ink, loc="left")
        ax.grid(axis="y", color="#e3e6ea", lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9, loc="lower center", ncol=2 if ncol == 2 else 4, frameon=False)
    fig.suptitle(title + "\nOOK / 2-ASK baud = bit rate, 4-ASK baud = half the bit rate; open triangle = 0 errors;\n"
                 "dotted = chance (0.5, when in range); PER counts lost packets as errors", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.07 if ncol == 2 else 0.05, 1, 0.92))
    fig.savefig(path + ".png", dpi=150); fig.savefig(path + ".pdf")
    plt.close(fig)


def write(out, rows, runs, summ, extra):
    os.makedirs(out, exist_ok=True)
    for name, data in (("summary.csv", summ), ("frames.csv", rows)):
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    json.dump(dict(extra, mfsk_decoder=MF.VERSION, mfsk_rules_sha=MF.RULES_SHA, tablet_decoder=T.VERSION,
                   tablet_rules_sha=T.RULES_SHA, nopre_decoder=NP.VERSION, nopre_rules_sha=NP.RULES_SHA,
                   sync_ignored=IGNORE_SYNC, runs=runs, summary=summ),
              open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    pct = lambda e, n: "-" if not n else f"0/{n}" if e == 0 else f"{e}/{n} = {100 * e / n:.2g} %"
    h = ["<table border=1 cellspacing=0 cellpadding=4 style='border-collapse:collapse;font-family:sans-serif;font-size:12px'>",
         "<tr><th>Scheme</th><th>Bit rate</th><th>Baud</th><th>Frame</th><th>PER</th><th>CRC pass</th><th>CRC missed</th>"
         "<th>Symbol errors</th><th>Before FEC</th><th>After FEC</th><th>timing r</th></tr>"]
    for s in summ:
        tr = "" if s["timing_r"] is None else f"{s['timing_r']:.2f}"
        h.append(f"<tr><td>{s['scheme']}</td><td>{s['bit_rate']:g}</td><td>{s['baud'] or 0:g}</td><td>{s['frame']}</td>"
                 f"<td>{pct(s['packet_errors'], s['packets_sent'])}</td><td>{s['crc_pass']}/{s['packets_sent']}</td>"
                 f"<td>{s['crc_missed']}</td><td>{pct(s['symbol_errors'] or 0, s['symbols'])}</td>"
                 f"<td>{pct(s['pre_fec_errors'], s['pre_fec_bits'])}</td><td>{pct(s['post_fec_errors'], s['post_fec_bits'])}</td>"
                 f"<td>{tr}</td></tr>")
    open(os.path.join(out, "table.html"), "w").write("\n".join(h + ["</table>"]))


def report(summ):
    print(f"\n{'scheme':36s} {'bit/s':>5s} {'baud':>5s} {'frame':9s} {'PER':>12s} {'CRC ok':>7s} {'symbol err':>12s} "
          f"{'before FEC':>12s} {'after FEC':>11s} {'timing r':>8s}")
    for s in summ:
        sym = f"{s['symbol_errors']}/{s['symbols']}" if s["symbols"] else "-"
        tr = "" if s["timing_r"] is None else f"{s['timing_r']:.2f}"
        print(f"{s['scheme'][:36]:36s} {s['bit_rate']:5g} {s['baud'] or 0:5.4g} {s['frame']:9s} "
              f"{s['packet_errors']:>3d}/{s['packets_sent']:<3d}{100 * s['per']:4.0f}% {s['crc_pass']:>3d}/{s['packets_sent']:<3d} "
              f"{sym:>12s} {s['pre_fec_errors']:>5d}/{s['pre_fec_bits']:<6d} {s['post_fec_errors']:>4d}/{s['post_fec_bits']:<6d} "
              f"{tr:>8s}")


def analyse(dirs, out, extra=None, title=None):
    rows, runs, sent = collect(dirs)
    if not sent:
        print("no ASK runs"); return
    summ = summarise(rows, runs, sent)
    write(out, rows or [dict(note="no decoded frames")], runs, summ, extra or {})
    plot(summ, os.path.join(out, "ask2_vs_ask4"), title or f"OOK vs 2-ASK vs 4-ASK, maXTouch tablet - {os.path.basename(out)}")
    report(summ)
    print(f"\nSaved {os.path.relpath(out, HERE)}/ (summary.csv/json, frames.csv, table.html, ask2_vs_ask4.png/.pdf)")


# ================================================================ run / synth
def run(a):
    need_current(a)
    runs = plan_all(a)
    show_plan(a, runs)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    ba = ber_args(a)
    tx, info = open_ask(ba)
    print(f"Transmitter: {info['line']}")
    done = []
    try:
        for i, r in enumerate(runs):
            print(f"\n[{i + 1}/{len(runs)}] {r['bit_rate']:g} bit/s {MOD_NAME[r['mod']]} "
                  f"{r['rate']:g} baud, chunk {r['chunk']}")
            d, stopped = B.one_run(tx, info, ba, r)
            if d:
                done.append(d)
            if stopped:
                print("  recording ended early (Ctrl+C or link lost) - stopping the session")
                break
    finally:
        try:
            tx.off()
        finally:
            tx.close()
    if done:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        analyse(done, os.path.join(B.DATADIR, f"askber_{a.label}_{stamp}"),
                dict(frame=frame_note(B.send_list(a)),
                     current_ma=a.current_ma, condition=a.condition, carrier_hz=a.carrier, ladder_db=a.ladder_db,
                     levels_file=a.levels_file, schemes=list(a.schemes), ask2_index=a.ask2_index))


# ================================================================ split set-up: TX on another laptop (no shared ground)
def tx_only(a):
    """the OTHER laptop: drive the Teensy only (same plan as 'run' for the same arguments), one run per Enter"""
    need_current(a)
    runs = plan_all(a)
    show_plan(a, runs)
    ba = ber_args(a)
    tx, info = open_ask(ba)
    print(f"Transmitter: {info['line']}")
    log = dict(info=info["line"], plan_id=a.plan_id, args={k: v for k, v in vars(a).items() if k != "what"}, runs=[])
    path = os.path.join(B.DATADIR, f"asktx_{a.label}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
    if a.auto:
        print(auto_note(a, runs))
    nxt = None                                  # --auto: this laptop's time of the next recording's first symbol
    try:
        for i, r in enumerate(runs):
            sched, plan = schedule(r["mod"], r["rate"], r["seq"]["phase"], ba)
            got = tx.upload_schedule(sched)
            head = (f"\n[{i + 1}/{len(runs)}] {r['bit_rate']:g} bit/s {MOD_NAME[r['mod']]} chunk {r['chunk']} "
                    f"({plan['total_s']:.1f} s) uploaded {got}.")
            if nxt is None:
                input(head + "\n  Press Enter when the maXTouch laptop shows >>> START TX NOW <<< ")
            else:
                print(head + f"\n  auto: starts by itself in {nxt - time.time():.1f} s (carrier off until then)", flush=True)
                if not sleep_until(nxt - STX.GO_DELAY_MS / 1e3):
                    print(f"  ** {time.time() - nxt + STX.GO_DELAY_MS / 1e3:.1f} s late (upload took too long) - the "
                          "receiver may miss this recording; use a longer --gap on both laptops **")
            t_first, _ = tx.go()
            ok = tx.wait_done(STX.GO_DELAY_MS / 1e3 + plan["total_s"] + 5)
            st = tx.status()
            tx.off()
            print(f"  sent: state {st.get('state')}, steps {st.get('events')}" + ("" if ok else "  ** did not finish **"))
            log["runs"].append(dict(mod=r["mod"], rate=r["rate"], chunk=r["chunk"], t_first_local=t_first,
                                    t_first_planned=nxt, status=st))
            json.dump(log, open(path, "w"), indent=1)
            if a.auto:                          # fixed timetable from here on: end of this recording + the gap
                nxt = t_first + plan["total_s"] + a.gap
    except KeyboardInterrupt:
        print("stopped (Ctrl+C)")
    finally:
        try:
            tx.off()
        finally:
            tx.close()
    print(f"log: {os.path.relpath(path, HERE)}")


AUTO_PRE = 3.0            # --auto: the prompt of a recording comes at least this long before its predicted start
AUTO_SETUP = 4.0          # --auto: time allowed for the tablet set-up + calibration before the lead (~2 s measured;
                          # more is allowed by itself if a set-up of this session took longer)
AUTO_SEARCH = 1.5         # --auto: the start is searched within +- this of the prediction
AUTO_GAP_MIN = 22.0       # shortest --gap the receive side can keep up with (see auto_note)


def sleep_until(t_wall):
    """-> False if t_wall has passed already"""
    late = time.time() > t_wall + 0.05
    while True:
        d = t_wall - time.time()
        if d <= 0:
            return not late
        time.sleep(min(d, 0.2))


def auto_note(a, runs):
    """--auto: only the first recording is started by hand. After it the transmitter sends every next recording
    --gap seconds after the end of the one before, by its own clock, and the receiver predicts each start from
    the last start it found (the two clocks only have to agree over one recording + gap: microseconds).
    The gap holds, on the receive side: 2 s tail + 1.5 s search margin, restore / start search / save (~5 s),
    then set-up + calibration (4 s allowed), 4 s lead and 3 s before the predicted start = ~20 s."""
    if a.gap < AUTO_GAP_MIN:
        raise SystemExit(f"--gap {a.gap:g} s is too short for the receive side (needs >= {AUTO_GAP_MIN:g} s)")
    ba, tot = ber_args(a), 0.0
    for r in runs:
        tot += schedule(r["mod"], r["rate"], r["seq"]["phase"], ba)[1]["total_s"]
    tot += a.gap * (len(runs) - 1)
    return (f"AUTO: only recording 1 is started by hand (Enter on both laptops); every later one starts {a.gap:g} s "
            f"after the end of the one before (--gap, the same on both laptops). About {tot / 60:.1f} min from the "
            "first start to the end - finger and electrode stay on the whole time.")


def find_start(t, X, t_prompt, plan, phase, window_s, d0=0.0):
    """frame start from the recording itself: correlate the known symbol levels with the panel feature over
    [t_prompt + d0, t_prompt + window] in 1 ms steps (the frozen decoders then refine it within 0-25 ms)"""
    M = 4 if plan["modulation"] == "ask4" else 2
    lv = MF.levels_of_states(phase, M)
    tmpl = 10 ** (np.asarray(plan["ladder_db"])[lv] / 20)
    _, y, _ = B.D.front(dict(t_prompt=t_prompt, modulation="fsk"), t, X)
    Ts, n = plan["sym_us"] / 1e6, len(lv)
    best = (-np.inf, t_prompt)
    for d in np.arange(max(d0, 0.0), window_s, 0.001):
        c = B.D.corr_states(B.D.sym_values(t, y, t_prompt + d, Ts, 0.0, n), tmpl)
        if c > best[0]:
            best = (c, t_prompt + d)
    return best[1], best[0]


def rx_only(a):
    """THIS laptop: maXTouch only, no Teensy. Records each run of the same plan; the TX laptop starts it by hand."""
    runs = plan_all(a)
    show_plan(a, runs)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    ba = ber_args(a)
    done = []
    if a.auto:
        print(auto_note(a, runs))
    pred = None                                 # --auto: this laptop's time of the next recording's start, as received
    setup_s = 0.0                               # longest tablet set-up + calibration seen in this session
    for i, r in enumerate(runs):
        sl, mod, rate = r["seq"], r["mod"], r["rate"]
        _, plan = schedule(mod, rate, sl["phase"], ba)
        print(f"\n[{i + 1}/{len(runs)}] {r['bit_rate']:g} bit/s {MOD_NAME[mod]} {rate:g} baud, chunk {r['chunk']}")
        if pred is None:
            seconds = a.start_window + plan["total_s"] + B.TAIL_S
            input(f"  TX laptop at the same run, transmitter OFF; finger on the panel, electrode on; keep still "
                  f"(~{seconds + a.lead + 2:.0f} s) - press Enter to calibrate and record ")
        else:
            t_go = pred - AUTO_PRE - a.lead - max(AUTO_SETUP, setup_s + 1.5)
            print(f"  auto: the transmitter starts by itself in {pred - time.time():.1f} s; calibrating in "
                  f"{max(t_go - time.time(), 0):.1f} s - keep still", flush=True)
            if not sleep_until(t_go):
                print(f"  ({time.time() - t_go:.1f} s behind the timetable - a longer --gap on both laptops gives more room)")
        t_open = time.time()
        chip = R.Chip(ba)
        try:
            changed = chip.setup()
            t_cal = time.time()
            setup_s = max(setup_s, t_cal - t_open)
            if pred is not None:                # record until tail + search margin after the predicted end
                if pred - (t_cal + a.lead) < 0.5:
                    print(f"  ** too late for this recording (set-up ended {t_cal + a.lead - pred + AUTO_PRE:.1f} s "
                          "behind the timetable) - skipped; use a longer --gap on both laptops **")
                    pred += plan["total_s"] + a.gap
                    continue
                seconds = pred - (t_cal + a.lead) + AUTO_SEARCH + plan["total_s"] + B.TAIL_S
                print("  (auto: nothing to press at START TX NOW)")
            t, X, t_prompt, _, errors = R.record(chip, ba, seconds)
        finally:
            chip.restore()
        if len(t) < 50 or t_prompt is None:
            print("  too few samples recorded - nothing saved, stopping the session")
            break
        if pred is None:
            tau, c = find_start(t, X, t_prompt, plan, sl["phase"], a.start_window)
            pred_err = None
        else:
            tau, c = find_start(t, X, t_prompt, plan, sl["phase"], pred - t_prompt + AUTO_SEARCH,
                                d0=pred - t_prompt - AUTO_SEARCH)
            pred_err = tau - pred
        print(f"  frame start found {tau - t_prompt:.3f} s after the prompt (correlation {c:.2f})"
              + (f", {pred_err * 1e3:+.0f} ms from the timetable" if pred_err is not None else "")
              + ("   ** weak - was the TX started? **" if c < 0.5 else ""))
        if a.auto:
            if c >= 0.5:
                pred = tau + plan["total_s"] + a.gap            # anchor the timetable on this recording
            elif pred is not None:
                pred += plan["total_s"] + a.gap                 # no clear start: keep the timetable as it was
            else:
                print("  no clear start in the first recording - nothing to time the next ones from, stopping. "
                      "Start again on both laptops.")
                break
        meta = dict(mode="trx_ber", modulation=mod, args=vars(ba), plan=plan, y0=chip.y0, ny=chip.ny, patch=chip.patch,
                    t_prompt=t_prompt, t_first=tau - 0.005, split_tx=True, start_corr=c, start_window_s=a.start_window,
                    plan_id=a.plan_id, auto=bool(a.auto), gap_s=a.gap if a.auto else None,
                    start_pred_err_s=pred_err,
                    tx=None, tx_status=None, hid_errors=errors, chip_ctrl_orig=changed, recorded_s=float(t[-1] - t[0]),
                    seconds_planned=seconds, samples=int(len(t)), send=sl["keys"], seed=a.seed, seq_k=int(round(rate)),
                    chunk=r["chunk"], round0=r["rounds"][0], round1=r["rounds"][1],
                    frame_index0=int(sl["frame_index0"]), pb=None, calibration="finger-on", t_calibrated=t_cal,
                    decoder_rules_sha=B.D.RULES_SHA, trx_shas=B.D.TRX_SHAS,
                    tool_sha=__import__("hashlib").sha256(open(__file__, "rb").read()).hexdigest()[:12],
                    **B.SM.lead_touch(t, X, t_prompt, chip.names))
        d = B.run_dir(ba, mod, rate, r["chunk"])
        R.save(d, t, X, meta)
        B.save_seq(d, sl)
        print(f"  saved {os.path.relpath(d, HERE)}")
        done.append(d)
    if done:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        analyse(done, os.path.join(B.DATADIR, f"askber_{a.label}_{stamp}"),
                dict(frame=frame_note(B.send_list(a)), split_tx=True,
                     current_ma=a.current_ma, condition=a.condition, carrier_hz=a.carrier, ladder_db=a.ladder_db,
                     levels_file=a.levels_file, schemes=list(a.schemes), ask2_index=a.ask2_index))


def frame_mode(keys):
    """one session = one frame type: i2o with or without preamble (48 info bits) or the trx frames (96)"""
    keys = set(keys)
    if keys & set(I2O.KEYS) and len(keys) > 1:
        raise SystemExit(f"one frame type per session / analysis: got {sorted(keys)} (the i2o frames have 48 info bits, "
                         "the trx frames 96; with and without preamble need different decoders) - do them separately")
    I2O.use(bool(keys & set(I2O.KEYS)))


def frame_note(keys):
    k = [x for x in keys if x in I2O.KEYS]
    return I2O.describe(k[0]) if k else f"trx frames {', '.join(keys)} ({TV2.N_INFO} info bits)"


def synth(a):
    import fsk4_trx_ber as F4                   # its 4-level panel simulator (random phase per scan, level jitter)
    seq = TV2.build_seq(B.send_list(a), 2, a.seed, 40)
    lv = MF.levels_of_states(seq["phase"], 4)
    assert np.array_equal(MF.states_of_levels(lv, 4), seq["phase"]), "Gray mapping does not invert"
    print("mapping check: 4-ASK levels <-> states invert exactly")
    root = tempfile.mkdtemp(prefix="ask_synth_")
    rng = np.random.default_rng(20261123)
    dirs = []
    for r in plan_all(a):
        _, plan = schedule(r["mod"], r["rate"], r["seq"]["phase"], ber_args(a))
        if r["mod"] == "ask4":
            t, X, meta = F4.synth_capture4(plan, r["seq"]["phase"], rng)
        elif r["mod"] == "ask2":                # two levels: stent_trx_ber's 2-level ('fsk') panel model, low level
            low = plan["ladder_db"][1] + rng.uniform(-1.5, 1.5)     # jittered like synth_capture4's
            t, X, meta = B.synth_capture("fsk", plan, r["seq"]["phase"], rng, fsk_db=low, gap_p=0.02)
        else:
            t, X, meta = B.synth_capture("ook", plan, r["seq"]["phase"], rng, gap_p=0.02)
        meta.update(modulation=r["mod"], send=r["seq"]["keys"], chunk=r["chunk"], round0=r["rounds"][0],
                    round1=r["rounds"][1], frame_index0=int(r["seq"]["frame_index0"]), args=dict(synthetic=True))
        d = os.path.join(root, f"SIM_{r['mod']}_{r['bit_rate']:g}bps_c{r['chunk']}")
        R.save(d, t, X, meta)
        B.save_seq(d, r["seq"])
        dirs.append(d)
    analyse(dirs, os.path.join(root, "askber_synth"), dict(synthetic=True, frame=frame_note(B.send_list(a)),
                                                           ladder_db=a.ladder_db, ask2_index=a.ask2_index),
            f"SIMULATED panel scans (not a measurement), 4-ASK ladder {a.ladder_db}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("levels", "send", "plan", "run", "tx", "rx", "analyse", "synth"))
    ap.add_argument("what", nargs="*", help="analyse: run folders (globs ok)")
    ap.add_argument("--label", default="ask")
    ap.add_argument("--bitrates", default="20,30,40,50,60", help="bit rates [bit/s]; OOK / 2-ASK baud = rate, 4-ASK baud = rate/2")
    ap.add_argument("--schemes", default="ook,ask2,ask4", help="on-air schemes to send, of ook,ask2,ask4")
    ap.add_argument("--ask2-index", type=int, choices=(1, 2), default=2,
                    help="2-ASK low level = this level index (1 = 67 %%, 2 = 33 %% of the full level; default 2)")
    ap.add_argument("--bits", type=int, default=512, help="send: PRBS9 bits per scheme and bit rate")
    ap.add_argument("--carrier", type=float, default=CARRIER)
    ap.add_argument("--levels", help="asklevels_*.json from 'levels' (default: the newest one in data_maxtouch)")
    ap.add_argument("--ladder-db", help="override: 4 levels [dB], full first, e.g. 0,-7,-17,-40 (off)")
    ap.add_argument("--send", default=I2O.KEY_NP,
                    help="frame type (default i2o_np = the i2o frame without preamble: 48 info bits, rate-1/2 K=4 code, "
                         "96 symbols; i2o = the same behind a 32-symbol sync word, 128 symbols; cc4 = the 256-symbol i2i "
                         "frame of trx_psk_node: SYNC64 + 96 info bits)")
    ap.add_argument("--ignore-sync", action="store_true",
                    help="analyse: decode frames that have a sync word as if it had not been sent (blind decoder)")
    ap.add_argument("--rounds", type=int, default=50, help="frames per (scheme, bit rate) (default 50 since 2026-11-24 "
                                                           "= 4800 states without preamble; 25 before)")
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--current-ma", type=float, help="current network setting = the full ASK level (required for "
                                                      "levels / run; IEC check)")
    ap.add_argument("--condition", help="run condition for the record, e.g. through_body")
    ap.add_argument("--calibrate", choices=("finger-on", "finger-off"), default="finger-on")
    ap.add_argument("--pause", action="store_true", help="wait for Enter before each recording")
    ap.add_argument("--lead", type=float, default=4.0)
    ap.add_argument("--y0", type=int, default=0)
    ap.add_argument("--ny", type=int, default=28)
    ap.add_argument("--patch", type=R.parse_patch)
    ap.add_argument("--keep-noise-suppression", action="store_true")
    ap.add_argument("--keep-touch", action="store_true")
    ap.add_argument("--out", help="analyse: output folder")
    ap.add_argument("--start-window", type=float, default=10.0,
                    help="rx: seconds after START TX NOW in which the TX laptop must start; the frame start is searched there")
    ap.add_argument("--auto", action="store_true",
                    help="tx / rx: only the first recording is started by hand, the rest run on a timetable (both laptops)")
    ap.add_argument("--gap", type=float, default=30.0,
                    help="--auto: seconds between the end of one recording and the start of the next (both laptops)")
    STX.add_hw_args(ap)
    a = ap.parse_args()
    a.schemes = [m.strip() for m in a.schemes.split(",") if m.strip()]
    if not a.schemes or any(m not in MODS for m in a.schemes):
        raise SystemExit(f"--schemes: choose from {','.join(MODS)}")
    a.schemes = [m for m in MODS if m in a.schemes]
    if a.cmd == "levels":
        levels(a); return
    if a.cmd == "send":
        send(a); return
    if a.cmd == "analyse":
        global IGNORE_SYNC
        IGNORE_SYNC = a.ignore_sync
        dirs = sorted({d for p in a.what for d in glob.glob(p) if os.path.exists(os.path.join(d, "meta.json"))})
        keys = sorted({str(k) for d in dirs for k in np.load(os.path.join(d, "tx_seq.npz"))["keys"]})
        frame_mode(keys)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        analyse(dirs, a.out or os.path.join(B.DATADIR, f"askber_analysis_{stamp}"), dict(frame=frame_note(keys)),
                title=("SYNC IGNORED (decoded as if sent without preamble) - " + ", ".join(keys)) if IGNORE_SYNC else None)
        return
    frame_mode(B.send_list(a))
    resolve_ladder(a, need=(a.cmd in ("run", "rx") and a.schemes != ["ook"]))   # OOK = full / off, needs no measured ladder
    if a.cmd == "plan":
        show_plan(a, plan_all(a))
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "tx":
        tx_only(a)
    elif a.cmd == "rx":
        rx_only(a)
    else:
        synth(a)


if __name__ == "__main__":
    main()
