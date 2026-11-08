#!/usr/bin/env python3
"""
Modulation comparison over the stent -> maXTouch link: OOK, FSK and DBPSK at several bit rates, ~5000 known bits per
run, raw recordings decoded offline. The maXTouch version of the stent-to-stent test (deck slides 146 / 155-157).

   python stent_modcomp.py run --carrier 159585 --fsk-zero 180556 --label mc1      # 3 x 5 runs, ~65 min
   python stent_modcomp.py run --carrier 159585 --fsk-zero 180556 --mods ook,fsk --rates 10,50,100 --label mc1b
   python stent_modcomp.py run --carrier 159585 --fsk-zero 180556 --rates 80 --bits 254 --label smoke   # ~1 min check
   python stent_modcomp.py run --carrier 155692 --fsk-zero 180556 --mods ook,fsk --rates 10,20,50 --bits 1016 --pause
                              --current-ma 1 --label body1mA            # through-body: Enter before each run
   python stent_modcomp.py run --carrier 159585 --fsk-zero 180556 --mods ook --rates 60 --bits 1016 --current-ma 1
                              --condition still --calibrate finger-off --label a2f_offcal      # finger-off calibration
   python stent_modcomp.py analyse mc1                     # every run of a label -> table, csv, 3-panel figure
   python stent_modcomp.py analyse data_maxtouch/stent_mod_mc1_ook_10bps_<time> ...     # or explicit run folders
   python stent_modcomp.py compare data_maxtouch/modcomp_mc1_1mA data_maxtouch/modcomp_mc1_5mA --label currents
   python stent_modcomp.py motion hand159_still hand159_swipe hand159_still2 --label hand159   # still/swipe/still
   python stent_modcomp.py synth                           # offline self-test on synthetic recordings (no hardware)

Setup as for maxtouch_stent_rx.py --tx: TeensyTRX (original firmware) and the maXTouch EVK on this laptop, stent over
X line 0, maxtouch_map closed. Runs go rate by rate, all modulations at one rate back to back, so slow drift of the
phantom/panel hits every modulation alike. Each run: calibrate with the carrier off, --lead s of noise, schedule
start (time known to ~1 ms), record until 2 s after the schedule ends. Ctrl+C saves the current run and stops.

Calibration (--calibrate, body runs):
  finger-on   (default, every run up to 2026-11-02) the run calibrates with the finger already on the panel, so the
              deltas show only CHANGES of the finger since then (a still finger reads ~0).
  finger-off  calibrate with the electrode on and the finger OFF the panel, then place the finger and press Enter:
              the deltas then carry the whole touch (~+1500 counts on the finger's node, fingerloc_nat1_20261102_115156),
              so the finger can be located from the recording (touch_vs_carrier.py). Implies the --pause prompts.
              The decoders (frozen v4 / v5) are unchanged: both subtract a baseline anchored on the lead, which now
              includes the touch. Not yet known: whether the touch level plus the carrier clips the int16 deltas, and
              whether the chip recalibrates a long-held touch (none seen within 9 s, RPTEN on).
  Saved in meta.json: calibration, t_calibrated, and the touch seen in the lead (lead_touch_node / _counts, per-node
  lead_median); a finger-off run whose lead shows no touch is flagged when it is saved.

Transmit schedules (one TeensyTRX sseg each, bit_us = round(1e6 / bps)):
  ook   one v4 frame (16-bit preamble, Barker-11 SFD, 127-bit PRBS7 payload, 16 zeros) = 170 words x ceil(bits/127)
        reps; 1 = carrier, 0 = word 0 (output stopped). 5000 -> 40 frames = 5080 payload bits, 6800 bits on air.
  fsk   the same frames, 0 = --fsk-zero instead of off (constant envelope on the stent side).
        The panel is broadband: the 2026-10-23 phantom sweep gave +60.1 dB at 159.6 kHz and still +55.4 dB at
        163.6 kHz, so the slide-146 '+3 kHz' second tone is nearly as loud as the '1' and the panel (one energy
        detector, no tone filters) cannot tell them apart. Put the '0' in a panel null (180.6 kHz: +13.8 dB).
  dbpsk constant carrier, 'psk <bit_us> 9 cont' (phase-continuous, 180 deg flip when the firmware PRBS9 bit is 1),
        payload_bits + 1 symbols; bit k = phase change between symbols k-1 and k = PRBS9 output k (k >= 1).

Decoding (offline, every run):
  ook / fsk  frozen decode_v4.py (sha in result_v4.json), called unchanged -> result_v4.json / decoded_v4.json.
             The figure uses its combined known-timing BER (DBPSK has no energy pattern to sync on blindly, so all
             three are compared with the start given); the csv also has v4's combined blind BER.
  dbpsk      rules fixed here before any DBPSK recording (DBPSK_RULES below, sha in the summary).
  Symbol SNR Es/N0 = k * P1 / (2 sigma^2): k = samples per bit, sigma = OFF noise of the top node (lead MAD),
             P1 = mean (x - OFF baseline)^2 - sigma^2 over the guarded samples of sent-1 bits (DBPSK: all symbols).
             sigma^2 = N0/2 for real samples, so the textbook curves apply as Es/N0.
  Noise added: Gaussian noise on the raw samples of the decoded nodes, the same decoder with the timing of the clean
             recording, one realisation per level (seeded) -> BER vs Es/N0 at --ref-bps (default the lowest rate).
  Theory (dashed, AWGN): DBPSK 0.5 exp(-Es/N0), noncoherent FSK 0.5 exp(-Es/2N0), noncoherent OOK ~0.5 exp(-Es/4N0).

What to expect from DBPSK (measured before writing this, 2026-10-23): during a steady carrier, consecutive scans of a
node are uncorrelated (r = -0.03 / +0.02, signs agree 49 / 51 %, runs c156_20bps_180358 and val4_38bps) while the
nodes within one scan agree perfectly (r = 1.00). Every scan sees the carrier at a fresh random phase, so a 180 deg
flip between symbols leaves nothing a scan-rate receiver can see: expect chance. The synth self-test shows the
same decoder working on a phase-coherent channel, so a chance result is the panel, not the decoder.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import glob
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import time

import numpy as np
from scipy.ndimage import median_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode_v4 as V4                          # frozen; imports maxtouch_stent_rx (R) and stent_frames (TX)
import decode_fska as FA                        # experimental amplitude-discriminated 2-FSK decoder (own rules sha)
import stent_tx as STX

R, RX, SF = V4.R, V4.RX, V4.TX
DATADIR = R.DATADIR
MODS = ("ook", "fsk", "dbpsk")
EXPERIMENTAL = ("fska",)                        # amplitude-discriminated 2-FSK, decode_fska.py (v4 left as it is)
ALL_MODS = MODS + EXPERIMENTAL
PSK_ORDER = 9
TAIL_S = 2.0                                    # recorded after the schedule ends (DBPSK needs the carrier-off edge)
LEAD_TOUCH_MIN = 100.0                          # --calibrate finger-off: below this the lead shows no touch (warning)
SNR_GRID_DB = np.arange(-2.0, 40.1, 2.0)       # energy detection of random-phase samples needs ~25 dB
NOISE_SEED = 20261023
RULES_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]
DBPSK_RULES = ("nodes = top 3 by energy increase after START TX (v4 rule 1); y = sum of (x - OFF baseline) / OFF sigma "
               "over them; timing: carrier-on and carrier-off edges (first / last |y| > 10 lead rms, midpoint to the "
               "neighbouring sample) give the start and the clock (n_symbols bits apart, clipped to +-200 ppm as v4); bit k = 1 if y(last guarded "
               "sample of symbol k-1) * y(first guarded sample of symbol k) < 0, guard 2 ms as v4, no guarded sample -> "
               "any sample of the symbol, none -> decided 0")
COLORS = dict(ook="#ff7f0e", fsk="#2ca02c", dbpsk="#1f77b4", fska="#9467bd")      # slide 155 (+ fska)
MARKERS = dict(ook="s", fsk="^", dbpsk="o", fska="D")
NAMES = dict(ook="OOK", fsk="FSK", dbpsk="DBPSK", fska="FSK-A")


# ================================================================ schedules
def psk_prbs(n, order=PSK_ORDER):
    """TeensyTRX psk_next(): LFSR seeded all ones; symbol k flips the phase when output k is 1"""
    a, b = {7: (6, 5), 11: (10, 8), 15: (14, 13)}.get(order, (8, 4))
    mask = (1 << order) - 1
    l, out = mask, []
    for _ in range(n):
        bit = ((l >> a) ^ (l >> b)) & 1
        l = ((l << 1) | bit) & mask
        out.append(bit)
    return out


def schedule(mod, bps, carrier, fsk_zero, bits):
    """TeensyTRX schedule [(step_us, reps, words)] and its plan"""
    bit_us = int(round(1e6 / bps))
    frame, payload = SF.frame_bits()
    frames = max(1, math.ceil(bits / len(payload)))
    plan = dict(modulation=mod, bit_us=bit_us, bps=1e6 / bit_us, carrier_hz=carrier, payload_bits=frames * len(payload))
    if mod in ("ook", "fsk", "fska"):          # fska: the FSK schedule with a bit-0 tone the panel does see
        zero = 0.0 if mod == "ook" else fsk_zero
        words = [STX.fword(carrier if b else zero) for b in frame]
        plan.update(zero_hz=zero, frames=frames, frame_bits=len(frame), n_steps=frames * len(frame),
                    total_s=frames * len(frame) * bit_us / 1e6)
        return [(bit_us, frames, words)], plan
    n = frames * len(payload) + 1
    plan.update(zero_hz=None, n_symbols=n, n_steps=n, psk_order=PSK_ORDER, total_s=n * bit_us / 1e6)
    return [(bit_us, n, [STX.fword(carrier)])], plan


def run_meta(a, mod, plan, chip_info, t, t_prompt, t_first, errors, changed, tx_info, tx_status, seconds):
    """meta.json; 'args' carries what decode_v4 (via maxtouch_stent_rx.link_from) reads: carrier, bps, payload, prbs"""
    args = dict(vars(a), mode="ber", modulation=mod, carrier=plan["carrier_hz"], bps=plan["bps"], zero_hz=plan["zero_hz"],
                payload=127, prbs=7, block=1.0, reps=3, frames=plan.get("frames", 0))
    return dict(mode="ber" if mod != "dbpsk" else "ber_dbpsk", modulation=mod, args=args, y0=chip_info["y0"],
                ny=chip_info["ny"], patch=chip_info["patch"], t_prompt=t_prompt, t_first=t_first, tx=tx_info,
                tx_frames=plan.get("frames", 0), tx_status=tx_status, hid_errors=errors, chip_ctrl_orig=changed,
                plan=plan, recorded_s=float(t[-1] - t[0]), seconds_planned=seconds, samples=int(len(t)),
                modcomp_sha=RULES_SHA)


def zero_of(a, mod):
    """bit-0 tone of a keyed modulation: FSK's panel null, or FSK-A's visible second tone"""
    return a.fska_zero if mod == "fska" else a.fsk_zero


def lead_touch(t, X, t_prompt, names):
    """touch level before START TX (no carrier): per-node median of the lead. ~0 after a finger-on calibration; the
    whole touch (~+1500 counts on the finger's node) after a finger-off one"""
    lead = t < t_prompt if t_prompt is not None else np.ones(len(t), bool)
    m = np.median(X[lead], axis=0) if lead.any() else np.zeros(X.shape[1])
    j = int(np.argmax(m))
    return dict(lead_touch_node=names[j], lead_touch_counts=float(m[j]), lead_median=[float(v) for v in m])


def run_dir(datadir, label, mod, bps):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(datadir, f"stent_mod_{label}_{mod}_{round(bps):g}bps_{stamp}")


# ================================================================ run
def one_run(tx, tx_info, a, mod, bps):
    sched, plan = schedule(mod, bps, a.carrier, zero_of(a, mod), a.bits)
    got = tx.upload_schedule(sched)
    if mod == "dbpsk":
        r = tx.ask(f"psk {plan['bit_us']} {PSK_ORDER} cont", r"^# psk")
        if "mode=cont" not in r:
            raise RuntimeError(f"psk not taken: {r!r}")
    print(f"  uploaded: {got}")
    seconds = STX.GO_DELAY_MS / 1e3 + plan["total_s"] + TAIL_S
    finger_off = a.calibrate == "finger-off"
    todo = ("keep still until 'Recording nodes', then swipe along the top edge (Y0-Y5, ~1 swipe/s, always "
            "touching) until the run ends" if a.condition == "swipe" else "keep still until the run ends")
    if finger_off:                              # calibrate with nothing on the panel, then place the finger
        input(f"  [{a.condition or 'no condition'}] electrode on, finger OFF the panel - press Enter to calibrate ")
    elif a.pause:                               # body runs: settle the finger before the calibration (it counts as 'off')
        input(f"  [{a.condition or 'no condition'}] finger on the panel, electrode on; {todo} "
              f"(~{seconds + a.lead + 2:.0f} s) - press Enter to start ")
    chip = status = None
    t_cal = None
    try:
        chip = R.Chip(a)
        changed = chip.setup()
        t_cal = time.time()
        if finger_off:
            input(f"  calibrated. Finger ON the panel now; {todo} (~{seconds + a.lead + 2:.0f} s) - press Enter to start ")
        t, X, t_prompt, t_first, errors = R.record(chip, a, seconds, lambda: tx.go()[0])
    finally:
        try:
            status = tx.status()                # before 'off': state done, events, late steps, psk_n
        except Exception as e:
            status = dict(error=str(e))
        tx.off()
        if chip:
            chip.restore()
    stopped = len(t) < 50 or t_prompt is None or t[-1] < t_prompt + seconds - 1.0
    if len(t) < 50:
        print("  too few samples recorded - nothing saved")
        return None, True
    d = run_dir(DATADIR, a.label, mod, plan["bps"])
    meta = run_meta(a, mod, plan, dict(y0=chip.y0, ny=chip.ny, patch=chip.patch), t, t_prompt, t_first,
                    errors, changed, tx_info, status, seconds)
    meta.update(calibration=a.calibrate, t_calibrated=t_cal, **lead_touch(t, X, t_prompt, chip.names))
    R.save(d, t, X, meta)
    print(f"  touch in the lead (before START TX): {meta['lead_touch_node']} {meta['lead_touch_counts']:+.0f} counts"
          + (f"   ** finger-off calibration but no touch above {LEAD_TOUCH_MIN:g}: was the finger on the panel when "
             f"calibrating, or off it while recording? **" if finger_off and meta['lead_touch_counts'] < LEAD_TOUCH_MIN else ""))
    ev = status.get("events", "?")
    want = f"{plan['n_steps']}/{plan['n_steps']}"
    print(f"  saved {os.path.relpath(d, HERE)}   transmitter: state {status.get('state')}, events {ev}"
          + ("" if ev == want and status.get("state") == "done" else f"   ** expected done, {want} **")
          + (f", psk_mode {status.get('psk_mode')}" if mod == "dbpsk" else ""))
    return d, stopped


def run(a):
    mods = [m.strip() for m in a.mods.split(",")]
    rates = [float(r) for r in a.rates.split(",")]
    plan = [(bps, m) for bps in rates for m in mods]
    est = sum(schedule(m, bps, a.carrier, zero_of(a, m), a.bits)[1]["total_s"] + a.lead + 6 for bps, m in plan)
    print(f"{len(plan)} runs, carrier {a.carrier:g} Hz" + (f", FSK 0 = {a.fsk_zero:g} Hz" if "fsk" in mods else "")
          + (f", FSK-A 0 = {a.fska_zero:g} Hz" if "fska" in mods else "")
          + f", {math.ceil(a.bits / 127) * 127} payload bits each, about {est / 60:.0f} min:")
    for bps, m in plan:
        p = schedule(m, bps, a.carrier, zero_of(a, m), a.bits)[1]
        print(f"  {NAMES[m]:6s} {p['bps']:6g} bps  {p['total_s']:6.0f} s")
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))   # kill -> SystemExit -> finally turns the carrier off
    tx, info = STX.open_configured(a)
    print(f"Transmitter: {info['line']}")
    done = []
    try:
        for i, (bps, m) in enumerate(plan):
            print(f"\n[{i + 1}/{len(plan)}] {NAMES[m]} {bps:g} bps")
            d, stopped = one_run(tx, info, a, m, bps)
            if d:
                done.append(d)
            if stopped:
                print("  recording ended early (Ctrl+C or link lost) - stopping the session")
                break
    finally:
        tx.close()
    if done:                                    # one folder per session: a reused label must not overwrite an earlier one
        first = os.path.basename(done[0]).rsplit("_", 2)[-2:]
        analyse(done, a.ref_bps, os.path.join(DATADIR, f"modcomp_{a.label}_{'_'.join(first)}"))


# ================================================================ analysis helpers
def offbase(x, lead):
    """OFF baseline and OFF noise exactly as decode_v4.absdev (running median of the quiet samples, lead MAD)"""
    med = float(np.median(x[lead]))
    sig = max(1.4826 * float(np.median(np.abs(x[lead] - med))), 1.0)
    q = np.where(np.abs(x - med) < 20 * sig)[0]
    b = np.full(len(x), med) if len(q) < 10 else np.interp(np.arange(len(x)), q, RX.baseline(x[q], 50))
    return b, sig


def guarded(t, tau, Tb, eps, nb, n_units):
    """(unit, bit, guarded) per sample; unit = frame (nb bits) or symbol (nb = 1)"""
    m, j, pos = V4.bit_index(t, tau, Tb, eps, nb)
    g = V4.GUARD_MS / 1e3 / (Tb * (1 + eps))
    return m, j, (m >= 0) & (m < n_units) & (pos > g) & (pos < 1 - g)


def clip_fraction(x):
    """share of the strong samples sitting exactly at the run's largest |delta|. A random-phase carrier crowds the
    samples near +-A but spreads them (1 mA, 2026-10-23: Y2/Y3/Y4 maxima 3355/4607/4279, no repeated top value,
    ~0.1 %); a delta range limit stacks them on one value. Above 1 % the SNR and contrast read low."""
    a = np.abs(np.asarray(x, int))
    on = a > 0.1 * a.max()
    return float(np.mean(a[on] == a.max())) if on.any() else 0.0


def es_n0_db(k, p1, sig):
    return float(10 * np.log10(max(k * p1 / (2 * sig ** 2), 1e-12)))


def energy_rank(X, t, t_prompt):
    e = np.diff(X, axis=0) ** 2
    pre, post = t[1:] < t_prompt, t[1:] >= t_prompt
    score = 10 * np.log10((e[post].mean(axis=0) + 1e-9) / (e[pre].mean(axis=0) + 1e-9))
    return [int(i) for i in np.argsort(-score)[:V4.N_NODES]], score


def add_noise(X, cols, sig_add, seed):
    rng = np.random.default_rng(seed)
    Xn = X.astype(float).copy()
    Xn[:, cols] += rng.normal(0.0, sig_add, (len(X), len(cols)))
    return Xn


# ================================================================ OOK / FSK (frozen v4)
def keyed_known(X, t, lead, top, tau, Tb, eps, n_sent):
    """v4 combined known-timing decision with a given timing (v4 absdev, z, max over nodes, decode_frames)"""
    frame, truth = SF.frame_bits()
    comb = np.max([R.zscore(V4.absdev(X[:, i], lead)) for i in top], axis=0)
    fr = V4.decode_frames(comb, t, tau, Tb, eps, frame, truth, np.arange(n_sent), True)
    return sum(f["errors"] for f in fr), sum(f["bits"] for f in fr)


def motion_metrics(x, t, lead, ok, sent, t_first, total_s, signal_rms):
    """finger-motion measures of one node, from the raw samples only (the decoder plays no part; v4's known timing
    only says which samples belong to sent-0 bits, where the carrier is off and only the finger moves the delta):
      drift_counts    95th - 5th percentile of the guarded sent-0 samples (how far the OFF level wanders)
      off_jitter      robust scan-to-scan std over consecutive guarded sent-0 samples (fast motion; for FSK it also
                      holds the '0' tone's own small swing)
      off_excursion   share of guarded sent-0 samples further than 0.25 x the carrier rms from the lead level
                      (what can make an OFF bit look ON)"""
    idx = np.where(ok)[0][sent == 0]
    if len(idx) < 10:
        return dict(drift_counts=None, off_jitter=None, off_excursion=None)
    drift = float(np.subtract(*np.percentile(x[idx], [95, 5])))
    pair = idx[1:][np.diff(idx) == 1]
    dd = x[pair] - x[pair - 1]
    jit = float(1.4826 * np.median(np.abs(dd - np.median(dd))) / math.sqrt(2)) if len(dd) > 10 else None
    exc = float(np.mean(np.abs(x[idx] - np.median(x[lead])) > 0.25 * signal_rms)) if signal_rms else None
    return dict(drift_counts=drift, off_jitter=jit, off_excursion=exc)


def analyse_keyed(d, meta, t, X):
    with contextlib.redirect_stdout(io.StringIO()):
        res = V4.decode_run(d)
    kt, cb = res.get("combined_known_timing") or {}, res["combined_blind"]
    names = R.node_names(meta)
    top = [names.index(n) for n in res["nodes"]]
    frame, _ = SF.frame_bits()
    Tb, n_sent, k = 1.0 / res["bps"], meta["tx_frames"], res["samples_per_bit"]
    lead = t < meta["t_prompt"]
    row = dict(run=os.path.basename(os.path.normpath(d)), modulation=meta["modulation"], bps=res["bps"],
               carrier_hz=meta["plan"]["carrier_hz"], zero_hz=meta["plan"]["zero_hz"], samples_per_bit=k,
               nodes=" ".join(res["nodes"]), decoder=f"v4 {res['rules_sha']}",
               current_ma=meta["args"].get("current_ma"), clip_fraction=clip_fraction(X[:, top[0]]),
               bits=kt.get("bits"), errors=kt.get("errors"), ber=kt.get("ber"), ber_upper95=kt.get("ber_upper95"),
               ber_conservative=kt.get("ber_conservative"), blind_bits=cb["bits"], blind_errors=cb["errors"],
               blind_ber=cb["ber"], delay_ms=kt.get("delay_ms"), clock_ppm=kt.get("clock_ppm"),
               signal_present=(res.get("health") or {}).get("signal_fraction"),
               signal_lost=(res.get("health") or {}).get("signal_lost"),
               blind_frames_valid=cb["frames_valid"], frames_sent=cb["frames_sent"],
               frame_recovery=cb["frames_valid"] / cb["frames_sent"] if cb["frames_sent"] else None,
               blind_ber_conservative=cb["ber_conservative"],
               blind_sent0_read1=cb["errors_by_type"]["sent0_read1"], blind_sent1_read0=cb["errors_by_type"]["sent1_read0"],
               blind_in_bursts=cb["errors_by_type"]["in_bursts"])
    if not kt:
        return row, None
    tau, eps = meta["t_first"] + kt["delay_ms"] / 1e3, kt["clock_ppm"] / 1e6
    x = X[:, top[0]].astype(float)
    b, sig = offbase(x, lead)
    m, j, ok = guarded(t, tau, Tb, eps, len(frame), n_sent)
    sent = np.array(frame)[j[ok]]
    p = (x[ok] - b[ok]) ** 2
    p1, p0 = p[sent == 1].mean() - sig ** 2, p[sent == 0].mean() - sig ** 2
    row.update(es_n0_db=es_n0_db(k, p1, sig), noise_sigma=sig, signal_rms=float(math.sqrt(max(p1, 0))),
               contrast_db=float(10 * np.log10((p1 + sig ** 2) / (p0 + sig ** 2))))
    # motion: worst of the 3 nodes the decoder combines (v4 ranks by energy ratio, so its first node can be a weak
    # edge node that was quiet in the lead; motion on any of the 3 can flip a bit)
    mm = []
    for i in top:
        xi = X[:, i].astype(float)
        bi, si = offbase(xi, lead)
        pi = (xi[ok] - bi[ok]) ** 2
        mm.append(motion_metrics(xi, t, lead, ok, sent, meta["t_first"], meta["plan"]["total_s"],
                                 math.sqrt(max(pi[sent == 1].mean() - si ** 2, 0))))
    row.update({k: max((m[k] for m in mm if m[k] is not None), default=None) for k in mm[0]})
    e, n = keyed_known(X.astype(float), t, lead, top, tau, Tb, eps, n_sent)
    if (e, n) != (kt["errors"], kt["bits"]):     # the noise curve must start from the same decoder as the v4 result
        print(f"  ! {row['run']}: re-decode {e}/{n} differs from v4 known timing {kt['errors']}/{kt['bits']}")

    def decode_with_noise(sig_add, seed):
        Xn = add_noise(X, top, sig_add, seed)
        return keyed_known(Xn, t, lead, top, tau, Tb, eps, n_sent)
    return row, dict(k=k, p1=p1, sig=sig, decode=decode_with_noise)


# ================================================================ FSK-A (experimental, decode_fska.py)
def analyse_fska(d, meta, t, X):
    """amplitude-discriminated 2-FSK: the FSK-A decoder gives the result; v4 (which reads the visible bit-0 tone as
    'on') is kept only as a reference in v4_* columns"""
    fa, (starts, tb) = FA.decode(X, t, meta)
    kt, bl = fa["known"], fa["blind"]
    names = R.node_names(meta)
    plan = meta["plan"]
    Tb, n_sent = plan["bit_us"] / 1e6, meta["tx_frames"]
    k = float(Tb * (len(t) - 1) / (t[-1] - t[0]))
    with contextlib.redirect_stdout(io.StringIO()):
        v4 = V4.decode_run(d)
    v4k, v4b = v4.get("combined_known_timing") or {}, v4["combined_blind"]
    row = dict(run=os.path.basename(os.path.normpath(d)), modulation="fska", bps=plan["bps"], carrier_hz=plan["carrier_hz"],
               zero_hz=plan["zero_hz"], samples_per_bit=k, nodes=" ".join(names[i] for i in fa["nodes"]),
               decoder=f"fska {FA.RULES_SHA}", current_ma=meta["args"].get("current_ma"),
               clip_fraction=clip_fraction(X[:, fa["nodes"][0]]),
               bits=kt["bits"], errors=kt["errors"], ber=kt["ber"],
               ber_upper95=3.0 / kt["bits"] if kt["bits"] and kt["errors"] == 0 else None,
               ber_conservative=(kt["errors"] + 0.5 * (n_sent - kt["frames_scored"]) * len(SF.frame_bits()[1]))
               / (n_sent * len(SF.frame_bits()[1])) if n_sent else None,
               erasures=kt["erasures"], delay_ms=kt["delay_ms"], clock_ppm=kt["clock_ppm"], pilot_r=kt["pilot_r"],
               blind_bits=bl["bits"], blind_errors=bl["errors"], blind_ber=bl["ber"],
               blind_frames_valid=bl["frames_valid"], frames_sent=n_sent,
               frame_recovery=bl["frames_valid"] / n_sent if n_sent else None,
               blind_ber_conservative=(bl["errors"] + 0.5 * (n_sent - bl["frames_valid"]) * len(SF.frame_bits()[1]))
               / (n_sent * len(SF.frame_bits()[1])) if n_sent else None,
               blind_sent0_read1=bl["sent0_read1"], blind_sent1_read0=bl["sent1_read0"], blind_in_bursts=None,
               tone0_over_tone1=(kt["u0_median"] / kt["u1_median"]) if kt.get("u1_median") else None,
               signal_present=None, signal_lost=kt["pilot_r"] < FA.R_MIN,
               v4_errors=v4k.get("errors"), v4_bits=v4k.get("bits"), v4_blind_errors=v4b["errors"],
               v4_blind_bits=v4b["bits"], v4_blind_frames_valid=v4b["frames_valid"])
    if not len(starts):
        return row, None
    lead = t < meta["t_prompt"]
    frame = SF.frame_bits()[0]
    x = X[:, fa["nodes"][0]].astype(float)
    b, sig = offbase(x, lead)
    _, j, ok = guarded(t, starts[0], tb, 0.0, len(frame), len(starts))
    sent = np.array(frame)[j[ok]]
    p = (x[ok] - b[ok]) ** 2
    p1, p0 = p[sent == 1].mean() - sig ** 2, p[sent == 0].mean() - sig ** 2
    row.update(es_n0_db=es_n0_db(k, p1, sig), noise_sigma=sig, signal_rms=float(math.sqrt(max(p1, 0))),
               contrast_db=float(10 * np.log10((p1 + sig ** 2) / (max(p0, 0) + sig ** 2))))

    def decode_with_noise(sig_add, seed):
        r, _ = FA.decode(X, t, meta, extra_noise=lambda Xa, cols: add_noise(Xa, cols, sig_add, seed))
        return r["known"]["errors"], r["known"]["bits"]
    return row, dict(k=k, p1=p1, sig=sig, decode=decode_with_noise)


# ================================================================ DBPSK
def dbpsk_decide(y, t, tau, Tb, eps, n_sym, truth):
    """bit k (1..n-1) = 1 if y changes sign from the last guarded sample of symbol k-1 to the first of symbol k"""
    s, _, ok = guarded(t, tau, Tb, eps, 1, n_sym)
    anyv = (s >= 0) & (s < n_sym)
    first, last = np.full(n_sym, -1), np.full(n_sym, -1)
    for sel in (anyv, ok):                      # guarded samples overwrite the fallback 'any sample'
        idx = np.where(sel)[0]
        if not len(idx):
            continue
        u, i0 = np.unique(s[idx], return_index=True)
        first[u] = idx[i0]
        u, i1 = np.unique(s[idx][::-1], return_index=True)
        last[u] = idx[::-1][i1]
    a, b = last[:-1], first[1:]
    have = (a >= 0) & (b >= 0)
    got = np.zeros(n_sym - 1, int)
    got[have] = (y[a[have]] * y[b[have]] < 0).astype(int)
    errors = int(np.sum(got != np.asarray(truth)))
    # scan-to-scan phase memory: sign agreement of consecutive guarded samples inside one symbol (0.5 = none)
    gi = np.where(ok)[0]
    same = gi[1:][s[gi[1:]] == s[gi[:-1]]]
    agree = float(np.mean(np.sign(y[same]) == np.sign(y[same - 1]))) if len(same) else None
    return errors, int(n_sym - 1), int((~have).sum()), agree


def dbpsk_y(X, lead, top):
    y = np.zeros(len(X))
    for i in top:
        x = X[:, i].astype(float)
        b, sig = offbase(x, lead)
        y += (x - b) / sig
    return y


def analyse_dbpsk(d, meta, t, X):
    plan = meta["plan"]
    n_sym, Tb = plan["n_symbols"], plan["bit_us"] / 1e6
    truth = psk_prbs(n_sym)[1:]
    lead = t < meta["t_prompt"]
    names = R.node_names(meta)
    top, _ = energy_rank(X.astype(float), t, meta["t_prompt"])
    y = dbpsk_y(X, lead, top)
    sy = max(1.4826 * float(np.median(np.abs(y[lead] - np.median(y[lead])))), 1e-9)
    on = np.abs(y) > 10 * sy
    k = float(Tb * (len(t) - 1) / (t[-1] - t[0]))
    row = dict(run=os.path.basename(os.path.normpath(d)), modulation="dbpsk", bps=plan["bps"],
               carrier_hz=plan["carrier_hz"], zero_hz=None, samples_per_bit=k, nodes=" ".join(names[i] for i in top),
               decoder=f"modcomp dbpsk {RULES_SHA}", blind_bits=None, blind_errors=None, blind_ber=None,
               current_ma=meta["args"].get("current_ma"), clip_fraction=clip_fraction(X[:, top[0]]))
    after = np.where(on & (t >= meta["t_first"] - 0.02))[0]
    end = meta["t_first"] + n_sym * Tb + 1.0
    if len(after) < 10 or after[0] == 0:
        row.update(signal_lost=True, signal_present=0.0)
        return row, None
    i1 = after[0]
    i2 = after[t[after] <= end][-1]
    t_on = 0.5 * (t[i1 - 1] + t[i1])
    t_off = 0.5 * (t[i2] + t[min(i2 + 1, len(t) - 1)])
    # edges are +-~4 ms (one scan): on a short run that is >1000 ppm, so keep v4's clock range
    tau, eps = t_on, float(np.clip((t_off - t_on) / (n_sym * Tb) - 1, -200e-6, 200e-6))
    span = (t >= t_on + 0.05) & (t <= t_off - 0.05)
    errors, bits, nodec, agree = dbpsk_decide(y, t, tau, Tb, eps, n_sym, truth)
    x = X[:, top[0]].astype(float)
    b, sig = offbase(x, lead)
    _, _, ok = guarded(t, tau, Tb, eps, 1, n_sym)
    p1 = float(np.mean((x[ok] - b[ok]) ** 2)) - sig ** 2
    row.update(bits=bits, errors=errors, ber=errors / bits, ber_upper95=3.0 / bits if errors == 0 else None,
               ber_conservative=errors / bits, delay_ms=(tau - meta["t_first"]) * 1e3, clock_ppm=eps * 1e6,
               signal_present=float(on[span].mean()) if span.any() else 0.0, signal_lost=False,
               no_decision=nodec, consecutive_sign_agreement=agree,
               es_n0_db=es_n0_db(k, p1, sig), noise_sigma=sig, signal_rms=float(math.sqrt(max(p1, 0))), contrast_db=None)

    def decode_with_noise(sig_add, seed):
        Xn = add_noise(X, top, sig_add, seed)
        return dbpsk_decide(dbpsk_y(Xn, lead, top), t, tau, Tb, eps, n_sym, truth)[:2]
    return row, dict(k=k, p1=p1, sig=sig, decode=decode_with_noise)


# ================================================================ analyse
def analyse_run(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "capture.npz"))
    t, X = z["t"].astype(float), z["X"]
    mod = meta.get("modulation")
    row, aux = (analyse_dbpsk if mod == "dbpsk" else analyse_fska if mod == "fska" else analyse_keyed)(d, meta, t, X)
    if mod == "fsk":                            # the FSK-A decoder on the same FSK recording, reported apart from v4
        fa, _ = FA.decode(X, t, meta)
        row.update(fska_dec_errors=fa["known"]["errors"], fska_dec_bits=fa["known"]["bits"],
                   fska_dec_blind_errors=fa["blind"]["errors"], fska_dec_blind_bits=fa["blind"]["bits"],
                   fska_dec_frames_valid=fa["blind"]["frames_valid"])
    # provenance: raw capture is never rewritten; its hash and first-bit time tie every number to it
    row.update(condition=meta["args"].get("condition"), label=meta["args"].get("label"),
               capture_sha256=hashlib.sha256(open(os.path.join(d, "capture.npz"), "rb").read()).hexdigest()[:16],
               t_first_local=datetime.datetime.fromtimestamp(meta["t_first"]).isoformat(timespec="milliseconds")
               if meta.get("t_first") else None)
    return row, aux


def noise_curve(aux):
    """BER vs Es/N0 with Gaussian noise added to the recording (signal power P1 fixed, noise sigma^2 + added^2)"""
    out = []
    for i, snr in enumerate(SNR_GRID_DB):
        var = aux["k"] * aux["p1"] / (2 * 10 ** (snr / 10)) - aux["sig"] ** 2
        if var <= 0:                            # cleaner than the recording itself
            continue
        e, n = aux["decode"](math.sqrt(var), NOISE_SEED + i)
        out.append(dict(es_n0_db=float(snr), errors=int(e), bits=int(n), ber=e / n if n else None,
                        added_sigma=math.sqrt(var)))
    return out


def theory(mod, snr_db):
    g = 10 ** (np.asarray(snr_db) / 10)
    return 0.5 * np.exp(-g / {"dbpsk": 1, "fsk": 2, "ook": 4}[mod])


def snr_at(curve, target=1e-3):
    """Es/N0 where the noise-added curve crosses target (log-linear), None if it never gets there"""
    pts = [(c["es_n0_db"], max(c["errors"], 0.5) / c["bits"]) for c in curve if c["bits"]]   # 0 errors -> 0.5/N
    for (s0, b0), (s1, b1) in zip(pts, pts[1:]):
        if b0 >= target > b1:
            return s0 + (math.log10(b0) - math.log10(target)) / (math.log10(b0) - math.log10(b1)) * (s1 - s0)
    return None


def analyse(dirs, ref_bps, outdir, current_ma=None):
    """current_ma: label for runs recorded without --current-ma (the raw meta.json is not changed)"""
    rows, aux = [], {}
    for d in sorted(dirs, key=lambda p: (json.load(open(os.path.join(p, "meta.json")))["plan"]["bps"], p)):
        print(f"decoding {os.path.basename(os.path.normpath(d))} ...", flush=True)
        row, ax = analyse_run(d)
        if row.get("current_ma") is None:
            row["current_ma"] = current_ma
        rows.append(row)
        if ax:
            aux[row["run"]] = ax
    if not rows:
        print("no runs"); return
    curves = {}
    for mod in ALL_MODS:
        cand = [r for r in rows if r["modulation"] == mod and r["run"] in aux]
        if not cand:
            continue
        ref = ([r for r in cand if ref_bps and abs(r["bps"] - ref_bps) < 1e-6] or sorted(cand, key=lambda r: r["bps"]))[0]
        print(f"noise-added curve: {NAMES[mod]} from {ref['run']} ...", flush=True)
        curves[mod] = dict(run=ref["run"], bps=ref["bps"], points=noise_curve(aux[ref["run"]]))
    os.makedirs(outdir, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(outdir, "noise_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modulation", "ref_run", "bps", "es_n0_db", "errors", "bits", "ber", "added_sigma"])
        for mod, c in curves.items():
            for p in c["points"]:
                w.writerow([mod, c["run"], c["bps"], p["es_n0_db"], p["errors"], p["bits"], p["ber"], p["added_sigma"]])
    json.dump(dict(rows=rows, curves=curves, modcomp_sha=RULES_SHA, dbpsk_rules=DBPSK_RULES,
                   fska_rules=FA.RULES if any(r["modulation"] == "fska" for r in rows) else None,
                   theory="DBPSK 0.5exp(-g), FSK noncoh 0.5exp(-g/2), OOK noncoh ~0.5exp(-g/4), g = Es/N0"),
              open(os.path.join(outdir, "summary.json"), "w"), indent=1)
    report(rows, curves)
    plot(rows, curves, os.path.join(outdir, "modulation_comparison"))
    print(f"\nSaved {os.path.relpath(outdir, HERE)}/ (summary.csv, noise_curves.csv, summary.json, "
          "modulation_comparison.png/.pdf)")


def fmt_ber(e, n, ub=None):
    if not n:
        return "no frame"
    return f"{e}/{n} (<{ub:.1e})" if e == 0 and ub else f"{e}/{n} = {e / n:.2%}"


def currents(rows):
    return sorted({r.get("current_ma") for r in rows}, key=lambda c: (c is None, c or 0))


def report(rows, curves):
    cur = currents(rows)
    print("\ndrive current: " + ", ".join("not given" if c is None else f"{c:g} mA" for c in cur)
          + ("   ** runs at different currents mixed in one table **" if len(cur) > 1 else ""))
    for r in rows:
        if r.get("clip_fraction", 0) > 0.01:
            print(f"  ! {r['run']}: {r['clip_fraction']:.1%} of the strong samples sit on one top value - the panel "
                  "delta is clipping, Es/N0 and contrast are underestimated")
    print(f"\n{'mod':6s} {'bps':>6} {'smp/bit':>7} {'Es/N0':>7} {'contrast':>8} {'BER known timing':>22} "
          f"{"blind":>18} {"delay":>7} {'clock':>7}  signal")
    for r in rows:
        sig = "LOST" if r.get("signal_lost") else (f"{r['signal_present']:.0%}" if r.get("signal_present") is not None else "-")
        es = f"{r['es_n0_db']:.1f}" if r.get("es_n0_db") is not None else "-"
        ct = f"{r['contrast_db']:.1f}" if r.get("contrast_db") is not None else "-"
        print(f"{NAMES[r['modulation']]:6s} {r['bps']:6g} {r['samples_per_bit']:7.2f} {es:>7} {ct:>8} "
              f"{fmt_ber(r.get('errors'), r.get('bits'), r.get('ber_upper95')):>22} "
              f"{(fmt_ber(r['blind_errors'], r['blind_bits']) if r.get('blind_bits') is not None else '-'):>18} "
              f"{(r.get('delay_ms') or 0):6.1f}ms {(r.get('clock_ppm') or 0):+6.0f}  {sig}"
              + (f"   consecutive-scan sign agreement {r['consecutive_sign_agreement']:.2f}"
                 if r.get("consecutive_sign_agreement") is not None else ""))
    for mod, c in curves.items():
        s = snr_at(c["points"])
        print(f"{NAMES[mod]}: noise-added BER 1e-3 at " + (f"Es/N0 {s:.1f} dB" if s is not None else "not reached")
              + f" ({c['bps']:g} bps run)")


def plot(rows, curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink2, grid = "#52514e", "#e4e3df"
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    notes = []                                  # error-count labels, placed after all series are drawn
    rates = sorted({round(r["bps"]) for r in rows})
    pos = {b: i for i, b in enumerate(rates)}   # rates evenly spaced: 80 / 90 / 100 would crowd on a log axis
    for mod in ALL_MODS:
        rr = sorted([r for r in rows if r["modulation"] == mod and r.get("bits")], key=lambda r: r["bps"])
        if not rr:
            continue
        c, mk = COLORS[mod], MARKERS[mod]
        bps = np.array([pos[round(r["bps"])] for r in rr]) + {"ook": -0.08, "fsk": 0.0, "fska": 0.04, "dbpsk": 0.08}[mod]
        es = np.array([r.get("es_n0_db", np.nan) for r in rr], float)
        ax[0].plot(bps, es, "-", marker=mk, color=c, ms=8, lw=2, label=NAMES[mod])
        ber = np.array([r["ber"] for r in rr], float)
        zero = np.array([r["errors"] == 0 for r in rr])
        y = np.where(zero, [3.0 / r["bits"] for r in rr], ber)
        ax[1].semilogy(bps, y, "-", color=c, lw=1.5, alpha=0.6)
        ax[1].semilogy(bps[~zero], y[~zero], mk, color=c, ms=8, label=NAMES[mod])
        ax[1].semilogy(bps[zero], y[zero], "v", color=c, ms=8)
        for r, x, yy in zip(rr, bps, y):         # every point labelled; /bits shows how many frames v4 could score
            notes.append((round(r["bps"]), x, yy, f"{r['errors']}/{r['bits']}"))
        if mod in curves:
            pts = [p for p in curves[mod]["points"] if p["errors"] > 0]
            ax[2].semilogy([p["es_n0_db"] for p in pts], [p["ber"] for p in pts], "-", marker=mk, color=c, ms=6, lw=2,
                           label=f"{NAMES[mod]} ({curves[mod]['bps']:g} bps)")
        g = np.linspace(-2, 40, 300)
        if mod != "fska":                       # no closed form for two random-phase tones told apart by amplitude
            ax[2].semilogy(g, theory(mod, g), "--", color=c, lw=1.2, alpha=0.8)
    for b, x, yy, txt in notes:                 # at one rate: the highest point labelled above, the others below
        top = max(v for b_, _, v, _ in notes if b_ == b)
        ax[1].annotate(txt, (x, yy), textcoords="offset points", xytext=(0, 9) if yy == top else (0, -15),
                       ha="center", fontsize=8, color=ink2)
    ax[0].set_xlabel("Bit rate (bps)"); ax[0].set_ylabel("Symbol SNR, Es/N0 (dB)")
    ax[1].axhline(0.5, color=ink2, ls="--", lw=0.8)
    ax[1].text(len(rates) - 0.5, 0.5, "chance", ha="right", va="bottom", fontsize=8, color=ink2)
    ax[1].set_xlabel("Bit rate (bps)"); ax[1].set_ylabel("Bit error rate (known timing)")
    ax[1].text(0.02, 0.03, "▼ = 0 errors (95 % upper bound 3/N)", transform=ax[1].transAxes, fontsize=8, color=ink2)
    ax[2].axhline(1e-3, color=ink2, ls=":", lw=0.8)
    ax[2].set_xlabel("Symbol SNR, Es/N0 (dB)"); ax[2].set_ylabel("Bit error rate")
    ax[2].set_xlim(-2, 40); ax[2].set_ylim(1e-4, 1)
    ax[2].text(0.02, 0.03, "solid: noise added to the recording (0-error points left out)\ndashed: theory (AWGN)",
               transform=ax[2].transAxes, ha="left", va="bottom", fontsize=8, color=ink2)
    for a_ in ax[:2]:
        a_.set_xticks(range(len(rates))); a_.set_xticklabels([f"{b:g}" for b in rates])
        a_.set_xlim(-0.5, len(rates) - 0.5)
    ax[1].set_ylim(1e-4, 1)
    cur = currents(rows)
    if len(cur) == 1 and cur[0] is not None:
        fig.suptitle(f"Modulation comparison ({cur[0]:g} mA)", fontsize=13, x=0.01, ha="left")
    for a_, letter in zip(ax, "abc"):
        a_.grid(color=grid, lw=0.6, which="both"); a_.set_axisbelow(True)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
        a_.text(-0.12, 1.04, letter, transform=a_.transAxes, fontsize=14, fontweight="bold")
        if a_.get_legend_handles_labels()[0]:
            a_.legend(fontsize=8, frameon=False, loc={0: "lower left", 1: "lower right", 2: "upper right"}[list(ax).index(a_)])
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=150)
    fig.savefig(path + ".pdf")
    plt.close(fig)


# ================================================================ synthetic self-test
FSKA_ZERO_DB = -10.5                            # 53.5 kHz under 159.6 kHz through the body (fsk_tone_survey, median)


def synth_capture(mod, plan, rng, lead=4.0, coherent=False, A=3000.0, sigma=3.0, ny=28, motion=False):
    """maXTouch-like recording: 7.45 ms scans (+50 ppm vs host), node reads the carrier ~4 ms before the scan time,
    arrives 8-10 ms after it, 0.3 % of samples dropped; Teensy clock +40 ppm. Each scan sees A cos(phase): a fresh
    random phase (what the panel does), or with coherent=True a slow 0.37 Hz beat plus the transmitted 0/180 deg."""
    Tb = plan["bit_us"] / 1e6
    t_start = 1.8e9
    t_prompt = t_start + lead
    t_first = t_prompt + STX.GO_DELAY_MS / 1e3 + 0.0005
    n = int((lead + 0.5 + plan["total_s"] + TAIL_S) / 0.00745)
    ts = t_start + np.arange(n) * 0.00745 * (1 + 5e-5)
    u = (ts - 0.004 - t_first) * (1 + 4e-5) / Tb           # transmitter bit clock
    idx = np.floor(u).astype(int)
    live = (idx >= 0) & (idx < plan["n_steps"])
    amp = np.zeros(n); ph = np.zeros(n)
    if mod in ("ook", "fsk", "fska"):
        frame = np.array(SF.frame_bits()[0])
        bit = frame[idx[live] % len(frame)]
        amp[live] = np.where(bit == 1, A, dict(ook=0.0, fsk=A * 10 ** (-46 / 20), fska=A * 10 ** (FSKA_ZERO_DB / 20))[mod])
    else:
        flips = np.cumsum(psk_prbs(plan["n_symbols"])) % 2
        amp[live] = A
        ph[live] = np.pi * flips[idx[live]]
    phase = (2 * np.pi * 0.37 * (ts - t_start) + ph) if coherent else rng.uniform(0, 2 * np.pi, n)
    gains = np.zeros(ny); gains[1:6] = [0.3, 0.8, 1.0, 0.7, 0.2]
    sig_part = np.outer(amp * np.cos(phase), gains)
    if motion:                                  # finger swipe like 10-26 11:42: +-1500 count drift, 60 count jitter
        tt = ts - t_start
        drift = sum(rng.uniform(300, 600) * np.sin(2 * np.pi * f * tt + rng.uniform(0, 6.3)) for f in (0.2, 0.5, 1.1, 1.7))
        sig_part = np.outer(amp * np.cos(phase), gains) * (1 + 0.4 * np.sin(2 * np.pi * 0.3 * tt))[:, None]
        sig_part += np.outer(drift, gains) + rng.normal(0, 60, (n, ny)) * (gains > 0.25)
    X = np.clip(np.round(sig_part + rng.normal(0, sigma, (n, ny))), -4700, 4700)
    t = ts + 0.008 + rng.uniform(0, 0.002, n)
    keep = rng.random(n) > 0.003
    return t[keep], X[keep].astype(np.int16), t_prompt, t_first


def synth(a):
    base = a.datadir or tempfile.mkdtemp(prefix="modcomp_synth_")
    rng = np.random.default_rng(1)
    rates = [float(r) for r in a.rates.split(",")]
    sets = {"synth": [(bps, m) for bps in rates for m in MODS], "synthcoh": [(bps, "dbpsk") for bps in rates],
            "synthmotion": [(bps, m) for bps in rates for m in ("ook", "fsk")],
            "synthfska": [(bps, m) for bps in rates for m in ("fsk", "fska")]}
    for label, plan_list in sets.items():
        dirs = []
        for bps, mod in plan_list:
            sched, plan = schedule(mod, bps, a.carrier, zero_of(a, mod), a.bits)
            t, X, t_prompt, t_first = synth_capture(mod, plan, rng, coherent=label == "synthcoh",
                                                    motion=label == "synthmotion")
            d = run_dir(base, label, mod, plan["bps"])
            status = dict(state="done", events=f"{plan['n_steps']}/{plan['n_steps']}", synthetic=True)
            R.save(d, t, X, run_meta(argparse.Namespace(**dict(vars(a), label=label)), mod, plan,
                                     dict(y0=0, ny=X.shape[1], patch=None), t, t_prompt, t_first, 0, {},
                                     dict(line="synthetic"), status, plan["total_s"]))
            dirs.append(d)
            time.sleep(1.0)                     # distinct folder stamps
        print(f"\n===== {label}: " + {"synth": "random phase per scan (like the panel)",
                                         "synthcoh": "phase-coherent channel (DBPSK decoder check)",
                                         "synthmotion": "finger swipe: +-1500 count drift, 60 count jitter",
                                         "synthfska": f"FSK vs FSK-A, bit-0 tone {FSKA_ZERO_DB:g} dB under bit 1"}[label])
        analyse(dirs, None, os.path.join(base, f"modcomp_{label}"))


# ================================================================ compare currents
CURRENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]     # categorical slots 1-4 (validated palette), fixed order by current


def compare(outdirs, out):
    """one figure + csv over several analysed sessions (modcomp_<label>/summary.json), one series per current:
    a = Es/N0 vs bit rate (OOK), b-d = BER vs bit rate for OOK, FSK, DBPSK"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets = []
    for d in outdirs:
        j = json.load(open(os.path.join(d, "summary.json")))
        cur = currents(j["rows"])
        sets.append((cur[0] if len(cur) == 1 else None, os.path.basename(os.path.normpath(d)), j["rows"]))
    sets.sort(key=lambda x: (x[0] is None, x[0] or 0))
    ink2, grid = "#52514e", "#e4e3df"
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))
    rates = sorted({r["bps"] for _, _, rows in sets for r in rows})
    table = []
    for n, (cur, name, rows) in enumerate(sets):
        c = CURRENT_COLORS[n % len(CURRENT_COLORS)]
        lab = f"{cur:g} mA" if cur is not None else name
        for mod, a_ in zip(("ook",) + MODS, ax):
            rr = sorted([r for r in rows if r["modulation"] == mod and r.get("bits")], key=lambda r: r["bps"])
            if not rr:
                continue
            bps = np.array([r["bps"] for r in rr])
            if a_ is ax[0]:
                a_.semilogx(bps, [r.get("es_n0_db", np.nan) for r in rr], "-o", color=c, ms=7, lw=2, label=lab)
                continue
            zero = np.array([r["errors"] == 0 for r in rr])
            y = np.array([3.0 / r["bits"] if r["errors"] == 0 else r["ber"] for r in rr])
            a_.loglog(bps, y, "-", color=c, lw=1.5)
            a_.loglog(bps[~zero], y[~zero], MARKERS[mod], color=c, ms=7, label=lab)
            a_.loglog(bps[zero], y[zero], "v", color=c, ms=7)
        for r in rows:
            table.append(dict(current_ma=cur, session=name, **{k: r.get(k) for k in (
                "modulation", "bps", "samples_per_bit", "es_n0_db", "contrast_db", "errors", "bits", "ber",
                "blind_errors", "blind_bits", "blind_ber", "signal_present", "signal_lost", "run")}))
    ax[0].set_ylabel("Symbol SNR, Es/N0 (dB)"); ax[0].set_title("OOK symbol SNR", fontsize=10, loc="left")
    for a_, mod in zip(ax[1:], MODS):
        a_.set_title(f"{NAMES[mod]} BER (known timing)", fontsize=10, loc="left")
        a_.axhline(0.5, color=ink2, ls="--", lw=0.8)
        a_.set_ylim(1e-4, 1); a_.set_ylabel("Bit error rate")
    ax[1].text(0.02, 0.03, "\u25bc = 0 errors (95 % upper bound 3/N)", transform=ax[1].transAxes, fontsize=8, color=ink2)
    for a_, letter in zip(ax, "abcd"):
        a_.set_xlabel("Bit rate (bps)")
        a_.set_xticks(rates); a_.set_xticks([], minor=True); a_.set_xticklabels([f"{r:g}" for r in rates])
        a_.set_xlim(min(rates) / 1.3, max(rates) * 1.3)
        a_.grid(color=grid, lw=0.6, which="both"); a_.set_axisbelow(True)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.text(-0.14, 1.04, letter, transform=a_.transAxes, fontsize=14, fontweight="bold")
        if a_.get_legend_handles_labels()[0]:
            a_.legend(fontsize=8, frameon=False, loc="upper right" if a_ is ax[0] else "lower right")
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "currents.png"), dpi=150); fig.savefig(os.path.join(out, "currents.pdf"))
    plt.close(fig)
    with open(os.path.join(out, "currents.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    print(f"Saved {os.path.relpath(out, HERE)}/ (currents.png/.pdf, currents.csv)")


# ================================================================ still / swipe comparison
def motion(labels, out):
    """sessions in the given order (e.g. still, swipe, still again), each one label = one condition: decodes every run
    with the same decoder (frozen v4 for OOK/FSK) and reports, separately, blind BER on the recovered frames, frame
    recovery, the conservative BER (lost frames count as 50 %), known-timing BER, error types and the motion measures.
    The raw captures are only read; each row carries the capture hash and the first-bit time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sessions = []
    for lab in labels:
        dirs = sorted(glob.glob(os.path.join(DATADIR, f"stent_mod_{lab}_*")))
        if not dirs:
            print(f"no runs for label {lab!r}"); continue
        rows = []
        for d in dirs:
            print(f"decoding {os.path.basename(d)} ...", flush=True)
            r, _ = analyse_run(d)
            if r["modulation"] != "dbpsk":
                rows.append(r)
        cond = sorted({r.get("condition") or "?" for r in rows})
        sessions.append((lab, "/".join(cond), rows))
    if not sessions:
        return
    os.makedirs(out, exist_ok=True)
    keys = ["label", "condition", "carrier_hz", "modulation", "bps", "samples_per_bit", "blind_errors", "blind_bits",
            "blind_ber", "blind_frames_valid", "frames_sent", "frame_recovery", "blind_ber_conservative", "errors", "bits",
            "ber", "blind_sent0_read1", "blind_sent1_read0", "blind_in_bursts", "drift_counts", "off_jitter",
            "off_excursion", "signal_rms", "es_n0_db", "signal_present", "signal_lost", "t_first_local",
            "capture_sha256", "run"]
    table = [dict((k, r.get(k)) for k in keys) for _, _, rows in sessions for r in rows]
    with open(os.path.join(out, "motion_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(table)
    print(f"\n{'session':16s} {'cond':6s} {'carrier':>8} {'mod':4s} {'bps':>4} {'blind BER (recovered)':>22} "
          f"{'frames':>9} {'conserv.':>8} {'known':>14} {'0->1':>5} {'1->0':>5} {'drift':>6} {'jitter':>6} {'excur.':>6}")
    for r in table:
        fv = f"{r['blind_frames_valid']}/{r['frames_sent']}"
        cons = f"{r['blind_ber_conservative']:.2%}" if r["blind_ber_conservative"] is not None else "-"
        print(f"{r['label']:16s} {(r['condition'] or '?'):6s} {r['carrier_hz'] / 1e3:7.1f}k {r['modulation']:4s} "
              f"{r['bps']:4g} {fmt_ber(r['blind_errors'], r['blind_bits'], 3.0 / r['blind_bits'] if r['blind_bits'] else None):>22} "
              f"{fv:>9} {cons:>8} {fmt_ber(r['errors'], r['bits']):>14} {r['blind_sent0_read1']:>5} {r['blind_sent1_read0']:>5} "
              f"{(r['drift_counts'] or 0):6.0f} {(r['off_jitter'] or 0):6.1f} {(r['off_excursion'] or 0):6.1%}"
              + ("  LOST" if r.get("signal_lost") else ""))
    # figure: one x position per session (in the given order), one series per carrier x modulation x rate
    series = sorted({(r["carrier_hz"], r["modulation"], r["bps"]) for r in table})
    carriers = sorted({c for c, _, _ in series})
    rates = sorted({b for _, _, b in series})
    styles = ["-", "--", ":", "-."]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    xs = np.arange(len(sessions))
    xl = [f"{lab}\n({cond})" for lab, cond, _ in sessions]
    for si, (c_hz, mod, bps) in enumerate(series):
        xd = xs + 0.04 * (si - (len(series) - 1) / 2)   # small sideways dodge: equal values stay visible
        pts = [next((r for r in rows if r["carrier_hz"] == c_hz and r["modulation"] == mod and r["bps"] == bps), None)
               for _, _, rows in sessions]
        lab = f"{NAMES[mod]} {bps:g} bps" + (f" @ {c_hz / 1e3:.1f} kHz" if len(carriers) > 1 else "")
        # rate: solid + filled marker (lowest rate) / dashed + hollow marker (next); carrier, if several, by line style
        ri = rates.index(bps)
        kw = dict(color=COLORS[mod], ls=styles[(ri if len(carriers) == 1 else carriers.index(c_hz)) % 4],
                  marker=MARKERS[mod], ms=8, lw=2, mew=1.8, label=lab,
                  mfc=COLORS[mod] if ri % 2 == 0 else "white")
        ber = [((p["blind_errors"] or 3.0) / p["blind_bits"]) if p and p["blind_bits"] else np.nan
               for p in pts]                    # 0 errors plotted at the 95 % bound 3/N
        ax[0].semilogy(xd, ber, **kw)
        last = next((i for i in range(len(ber) - 1, -1, -1) if np.isfinite(ber[i])), None)
        if last is not None:                    # direct label at the line end
            ax[0].annotate(lab, (xd[last], ber[last]), xytext=(8, 0), textcoords="offset points", va="center",
                           fontsize=8, color="#0b0b0b")
        ax[1].plot(xd, [100 * p["frame_recovery"] if p and p["frame_recovery"] is not None else np.nan for p in pts], **kw)
        ax[2].plot(xd, [p["drift_counts"] if p and p["drift_counts"] is not None else np.nan for p in pts], **kw)
        ax[3].plot(xd, [p["off_jitter"] if p and p["off_jitter"] is not None else np.nan for p in pts], **kw)
    titles = ("Blind BER, recovered frames (0 errors at 3/N)", "Frames recovered blind (%)",
              "OFF-level drift, worst decoded node (counts, 5-95 %)", "Scan-to-scan jitter in OFF bits (counts)")
    ink2, grid = "#52514e", "#e4e3df"
    for a_, title, letter in zip(ax, titles, "abcd"):
        a_.set_xticks(xs); a_.set_xticklabels(xl, fontsize=8)
        a_.set_xlim(-0.4, len(sessions) - 0.6 + (0.5 if a_ is ax[0] else 0))
        a_.set_title(title, fontsize=10, loc="left")
        a_.grid(color=grid, lw=0.6, which="both"); a_.set_axisbelow(True)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.text(-0.14, 1.04, letter, transform=a_.transAxes, fontsize=14, fontweight="bold")
    ax[0].axhline(0.5, color=ink2, ls="--", lw=0.8)
    ax[1].set_ylim(-5, 105)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=9, handlelength=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(out, "motion_comparison.png"), dpi=150); fig.savefig(os.path.join(out, "motion_comparison.pdf"))
    plt.close(fig)
    print(f"\nSaved {os.path.relpath(out, HERE)}/ (motion_comparison.csv, .png, .pdf)")


# ================================================================ condition comparison
def pool_rows(rows):
    """one row per modulation x rate: errors / bits / frames summed over sessions, Es/N0 and motion measures averaged"""
    out = []
    for key in sorted({(r["modulation"], round(r["bps"])) for r in rows}):
        rr = [r for r in rows if (r["modulation"], round(r["bps"])) == key]
        q = dict(rr[0], run="+".join(r["run"] for r in rr), t_first_local=rr[0].get("t_first_local"),
                 capture_sha256="+".join(r.get("capture_sha256") or "" for r in rr))
        for k in ("errors", "bits", "blind_errors", "blind_bits", "blind_frames_valid", "frames_sent",
                  "blind_sent0_read1", "blind_sent1_read0"):
            q[k] = sum(r.get(k) or 0 for r in rr)
        q["ber"] = q["errors"] / q["bits"] if q["bits"] else None
        q["blind_ber"] = q["blind_errors"] / q["blind_bits"] if q["blind_bits"] else None
        for k in ("es_n0_db", "drift_counts", "off_jitter", "off_excursion", "signal_rms"):
            v = [r[k] for r in rr if r.get(k) is not None]
            q[k] = float(np.mean(v)) if v else None
        v = [r["es_n0_db"] for r in rr if r.get("es_n0_db") is not None]
        q["es_n0_min"], q["es_n0_max"] = (min(v), max(v)) if len(v) > 1 else (None, None)    # spread over sessions
        out.append(q)
    return out


def pool_noise(noise):
    """noise-added points of several sessions: errors and bits summed per modulation, recording rate and Es/N0"""
    acc = {}
    for p in noise:
        k = (p["modulation"], round(float(p["bps"])), float(p["es_n0_db"]))
        a = acc.setdefault(k, dict(p, errors=0, bits=0))
        a["errors"] = int(a["errors"]) + int(p["errors"]); a["bits"] = int(a["bits"]) + int(p["bits"])
    return list(acc.values())


CONDITION_COLORS = ("#1f5fbf", "#d62728", "#222222", "#9467bd")     # --color-by condition: blue, red, black, purple


def conditions(specs, out, by_condition=False, colors=None, keep_rates=None, ber_min=None, snr_lim=None,
               letters="abc", labels=True, percent=False, counts=False, panel_model=False,
               noise_bps=None):
    """two or three analysed sessions of the same schedule under different conditions (NAME=modcomp_folder, e.g.
    still=... walking=...), from their summary.json / noise_curves.csv (nothing re-decoded), in the layout of the
    modulation-comparison slide: a = Es/N0 vs rate, b = known-timing BER vs rate, c = noise-added BER vs Es/N0.
    Modulation = colour, condition = line style (first: solid + filled, then dashed / dash-dot + hollow); with
    by_condition the other way round: condition = colour, OOK solid + filled, FSK dashed + hollow.
    The table (known-timing and blind BER, frame recovery, OFF drift) is written as csv, tsv and an html page
    whose table pastes into Google Slides as a table."""
    from scipy.stats import beta
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets, kref, sets_noise_bps = [], [], []
    for s in specs:                             # NAME=folder, or NAME=folder1+folder2: sessions pooled per mod x rate
        name, ds = s.split("=", 1)
        ds = ds.split("+")
        rows, noise = [], []
        for d in ds:
            rows += [r for r in json.load(open(os.path.join(d, "summary.json")))["rows"] if r["modulation"] != "dbpsk"]
            with open(os.path.join(d, "noise_curves.csv")) as f:
                noise += list(csv.DictReader(f))
        if keep_rates:                          # only these rates (rows, and noise curves recorded at them)
            rows = [r for r in rows if round(r["bps"]) in keep_rates]
            noise = [p for p in noise if round(float(p["bps"])) in keep_rates]
        sets_noise_bps.append({round(float(p["bps"])) for p in noise})
        nref = {round(float(p["bps"])) for p in noise}           # scans per bit of the noise-curve recordings
        kref += [(r["samples_per_bit"], round(r["bps"])) for r in rows if round(r["bps"]) in nref and r.get("samples_per_bit")]
        if len(ds) > 1:
            rows, noise = pool_rows(rows), pool_noise(noise)
        sets.append((name, "+".join(os.path.basename(os.path.normpath(d)) for d in ds), rows, noise))
    nb = noise_bps or min(set.union(*sets_noise_bps)) if any(sets_noise_bps) else None
    if nb is not None and any(len(b) > 1 or b != {nb} for b in sets_noise_bps):
        print(f"  panel c: noise-added curves from the {nb:g} bps recordings only "
              f"(found {sorted(set.union(*sets_noise_bps))} bps; curves of different rates are not pooled)")
    sets = [(name, folder, rows, [p for p in noise if round(float(p["bps"])) == nb]) for name, folder, rows, noise in sets]
    kref = [k_ for k_, b_ in kref if b_ == nb]
    ink2, grid = "#52514e", "#e4e3df"
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.0))
    rates = sorted({round(r["bps"]) for _, _, rows, _ in sets for r in rows})
    mods = [m for m in ALL_MODS if any(r["modulation"] == m for _, _, rows, _ in sets for r in rows)]
    nudge = {m: 0.07 * (i - (len(mods) - 1) / 2) for i, m in enumerate(mods)}     # rates evenly spaced
    styles = [dict(ls="-", filled=True), dict(ls="--", filled=False), dict(ls="-.", filled=False)]
    table, notes = [], []
    for k, (name, folder, rows, noise) in enumerate(sets):
        st = styles[k % len(styles)]
        for mod in mods:
            rr = sorted([r for r in rows if r["modulation"] == mod], key=lambda r: r["bps"])
            if not rr:
                continue
            if by_condition:
                c = (colors or CONDITION_COLORS)[k % len(colors or CONDITION_COLORS)]
                st = dict(ls="-" if mod == "ook" else "--", filled=mod == "ook")
            else:
                c = COLORS[mod]
            kw = dict(color=c, ls=st["ls"], lw=2, ms=8, mew=1.8, mfc=c if st["filled"] else "white", marker=MARKERS[mod])
            lab = f"{NAMES[mod]}, {name}"
            bps = np.array([rates.index(round(r["bps"])) for r in rr]) + nudge[mod] + 0.025 * (k - (len(sets) - 1) / 2)
            ax[0].plot(bps, [r.get("es_n0_db", np.nan) for r in rr], label=lab, **kw)
            sp = [(x, r) for x, r in zip(bps, rr) if r.get("es_n0_min") is not None]
            if sp:                              # pooled sessions: bar = lowest to highest session Es/N0
                ax[0].errorbar([x for x, _ in sp], [r["es_n0_db"] for _, r in sp], fmt="none", ecolor=c,
                               elinewidth=1.2, capsize=4, capthick=1.2,
                               yerr=[[r["es_n0_db"] - r["es_n0_min"] for _, r in sp],
                                     [r["es_n0_max"] - r["es_n0_db"] for _, r in sp]])
            e = np.array([r["errors"] for r in rr], float); n = np.array([r["bits"] for r in rr], float)
            if counts:                          # error counts on a linear axis; bars = 95 % Clopper-Pearson x bits
                lo_all = np.where(e > 0, beta.ppf(0.025, np.maximum(e, 1), n - e + 1), 0.0) * n
                hi_all = beta.ppf(0.975, e + 1, n - e) * n
                ax[1].plot(bps, e, **kw)
                ax[1].errorbar(bps, e, yerr=[e - lo_all, hi_all - e], fmt="none", ecolor=c, elinewidth=1.2,
                               capsize=4, capthick=1.2)
                for r, x, yy, l_, h_ in zip(rr, bps, e, lo_all, hi_all):
                    notes.append((round(r["bps"]), x, yy, f"{r['errors']}/{r['bits']}", c, l_, h_))
            else:
                ax[1].semilogy(bps, np.where(e == 0, 3.0 / n, e / n), **kw)     # 0 errors at the 95 % bound 3/N
                ok = e > 0                      # 95 % Clopper-Pearson bars; a 0-error point is already its upper bound
                lo = beta.ppf(0.025, e[ok], n[ok] - e[ok] + 1); hi = beta.ppf(0.975, e[ok] + 1, n[ok] - e[ok])
                ax[1].errorbar(bps[ok], e[ok] / n[ok], yerr=[e[ok] / n[ok] - lo, hi - e[ok] / n[ok]], fmt="none",
                               ecolor=c, elinewidth=1.2, capsize=4, capthick=1.2)
                if (e == 0).any():
                    ax[1].plot(bps[e == 0], 3.0 / n[e == 0], "v", color=c, ms=8)
                lo_all = np.where(ok, beta.ppf(0.025, np.maximum(e, 1), n - e + 1), 3.0 / n)   # bar ends, for labels
                hi_all = np.where(ok, beta.ppf(0.975, e + 1, n - e), 3.0 / n)
                for r, x, yy, l_, h_ in zip(rr, bps, np.where(e == 0, 3.0 / n, e / n), lo_all, hi_all):
                    notes.append((round(r["bps"]), x, yy, f"{r['errors']}/{r['bits']}", c, l_, h_))   # all labelled
            nn = sorted([p for p in noise if p["modulation"] == mod], key=lambda p: float(p["es_n0_db"]))
            if nn:
                g = np.array([float(p["es_n0_db"]) for p in nn])
                e = np.array([int(p["errors"]) for p in nn]); n = np.array([int(p["bits"]) for p in nn])
                ax[2].semilogy(g, np.where(e == 0, np.nan, e / n), **dict(kw, ms=6))   # 0 errors: gap (3/N > 1/N)
            for r in rr:
                table.append(dict(condition=name, session=folder, **{q: r.get(q) for q in (
                    "modulation", "bps", "errors", "bits", "ber", "blind_errors", "blind_bits", "blind_ber",
                    "blind_frames_valid", "frames_sent", "blind_sent0_read1", "blind_sent1_read0", "drift_counts",
                    "off_jitter", "off_excursion", "es_n0_db", "signal_rms", "t_first_local", "capture_sha256", "run")}))
    g = np.linspace(-3, 30, 200)
    for mod in mods:
        if mod != "fska":
            ax[2].semilogy(g, theory(mod, g), ":", color=ink2 if by_condition else COLORS[mod], lw=1.2, alpha=0.7)
    if panel_model and kref:                    # simulated LLR detector on sparse random-phase scans (panel_model.py)
        import panel_model as PM
        kk = float(np.median(kref))
        pm = PM.curve(kk)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "panel_model.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pm[0])); w.writeheader(); w.writerows(pm)
        gp = np.array([p["es_n0_db"] for p in pm]); bp = np.array([p["ber"] for p in pm])
        ax[2].semilogy(gp, np.where(bp > 0, bp, np.nan), "-.", color="black", lw=1.8, zorder=4)
        from matplotlib.lines import Line2D
        ax[2].legend([Line2D([], [], color=ink2, lw=2, marker="s", ms=5),
                      Line2D([], [], color="black", lw=1.8, ls="-."), Line2D([], [], color=ink2, lw=1.2, ls=":")],
                     ["1  experimental: v4 decoder, noise added to the recording",
                      f"2  simulated panel-aware detector (model, k = {kk:.2f} scans/bit)",
                      "3  textbook AWGN reference (FSK left, OOK right)"],
                     loc="upper right", fontsize=7.5, frameon=True, framealpha=0.9, edgecolor="none")
    ax[0].set_ylabel("Symbol SNR, Es/N0 (dB)")
    if any(r.get("es_n0_min") is not None for _, _, rows, _ in sets for r in rows):
        ax[0].text(0.02, 0.03, "points: mean of sessions; bars: lowest to highest session", transform=ax[0].transAxes,
                   fontsize=8, color=ink2)
    ax[1].set_ylabel("Bit error rate (%, known timing)" if percent else "Bit error rate (known timing)")
    if percent:                                 # same log axis, ticks written as percentages
        from matplotlib.ticker import FuncFormatter
        ax[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:g}")); ax[1].set_ylim(1e-4, 1)
    if counts:
        top = max(100.0, 1.15 * max(q[6] for q in notes))
        ax[1].set_ylabel("Bit errors (known timing)"); ax[1].set_ylim(-0.1 * top, top)
        ax[1].axhline(0, color=ink2, lw=0.6)
        ax[1].text(0.02, 0.97, "bars: 95 % Clopper-Pearson interval; labels: errors / bits sent",
                   transform=ax[1].transAxes, fontsize=8, color=ink2, va="top")
    else:
        ax[1].axhline(0.5, color=ink2, ls="--", lw=0.8)
        ax[1].text(0.97, 0.52, "chance", transform=ax[1].get_yaxis_transform(), ha="right", va="bottom", fontsize=8,
                   color=ink2)
        ax[1].text(0.02, 0.88, "bars: 95 % Clopper-Pearson interval" + ("\n\u25bc = 0 errors (95 % upper bound 3/N)"
                   if any(r["errors"] == 0 for r in table) else ""), transform=ax[1].transAxes, fontsize=8, color=ink2,
                   va="top")
        ax[1].set_ylim(ber_min or min(1e-4, min(q[5] for q in notes) / 4), 1)     # room for labels under the lowest bars
    if snr_lim:
        ax[0].set_ylim(*snr_lim)
    ax[2].set_ylabel("Bit error rate"); ax[2].set_ylim(1e-4, 1); ax[2].set_xlim(-3, 30)
    ax[2].set_xlabel("Symbol SNR, Es/N0 (dB)")
    ax[2].axhline(1e-3, color=ink2, ls=":", lw=0.8)
    ref = {float(p["bps"]) for _, _, _, noise in sets for p in noise}
    if panel_model and kref:
        ax[2].text(0.02, 0.03, f"1: {'/'.join(f'{b:g}' for b in sorted(ref))} bps recordings, colour as panels a-b "
                   "(gap = 0 errors)\n2: random-phase scans, Gaussian noise, one node, likelihood-ratio decision\n"
                   "all three: Es/N0 = k P1 / 2 sigma^2", transform=ax[2].transAxes, fontsize=7.5, color=ink2)
    else:
        ax[2].text(0.02, 0.03, f"lines: noise added to the {'/'.join(f'{b:g}' for b in sorted(ref))} bps recordings "
                   "(gap = 0 errors)\ndotted: theory (AWGN)" + (", FSK left, OOK right" if by_condition else ""),
                   transform=ax[2].transAxes, fontsize=8, color=ink2)
    for a_, letter in zip(ax, letters):
        a_.grid(color=grid, lw=0.6, which="both"); a_.set_axisbelow(True)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.text(-0.14, 1.04, letter, transform=a_.transAxes, fontsize=14, fontweight="bold")
        if a_ is not ax[2]:
            a_.set_xlabel("Bit rate (bps)")
            a_.set_xticks(range(len(rates))); a_.set_xticklabels([f"{r:g}" for r in rates])
            a_.set_xlim(-0.5, len(rates) - 0.5)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=10, handlelength=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    to_px, gap = ax[1].transData, 11 * fig.dpi / 72
    for b in (rates if labels else []):         # labels errors/bits: one set above/below, several in a column beside the rate
        grp = sorted([q for q in notes if q[0] == b], key=lambda q: -q[2])
        if len(sets) == 1:                      # highest point labelled above its error bar, the others below theirs
            for i, (_, x, yy, txt, _, l_, h_) in enumerate(grp):
                ax[1].annotate(txt, (x, h_ if i == 0 else l_), textcoords="offset points", xytext=(0, 4 if i == 0 else -4),
                               ha="center", va="bottom" if i == 0 else "top", fontsize=8, color=ink2, zorder=5,
                               bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
            continue
        py = [to_px.transform((x, yy))[1] for _, x, yy, *_ in grp]
        ly = [py[0]]
        for v in py[1:]:
            ly.append(min(v, ly[-1] - gap))
        ly = np.array(ly) + (np.mean(py) - np.mean(ly))
        lx = to_px.transform((rates.index(b) + 0.16, 1))[0]
        for (_, x, yy, txt, c, _, _), v in zip(grp, ly):
            ax[1].annotate(txt, (x, yy), xytext=to_px.inverted().transform((lx, v)), textcoords="data",
                           ha="left", va="center", fontsize=7.5, color=ink2, zorder=5,
                           bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85),
                           arrowprops=dict(arrowstyle="-", color=c, lw=0.6, shrinkA=1, shrinkB=4))
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "conditions.png"), dpi=150); fig.savefig(os.path.join(out, "conditions.pdf"))
    plt.close(fig)
    with open(os.path.join(out, "conditions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)
    # slide table: one row per modulation x rate, conditions side by side
    names = [nm for nm, _, _, _ in sets]
    cell = lambda e, n: f"{e}/{n} ({e / n:.2%})" if n else "-"
    head = (["Modulation", "Bit rate (bps)"] + [f"BER known timing, {nm}" for nm in names]
            + [f"BER blind, {nm}" for nm in names] + [f"Frames recovered, {nm}" for nm in names]
            + [f"OFF drift (counts), {nm}" for nm in names])
    body = []
    for mod in mods:
        for bps in rates:
            rr = [next((r for r in table if r["condition"] == nm and r["modulation"] == mod and round(r["bps"]) == bps), None)
                  for nm in names]
            if not any(rr):
                continue
            body.append([NAMES[mod], f"{bps:g}"] + [cell(r["errors"], r["bits"]) if r else "-" for r in rr]
                        + [cell(r["blind_errors"], r["blind_bits"]) if r else "-" for r in rr]
                        + [f"{r['blind_frames_valid']}/{r['frames_sent']}" if r else "-" for r in rr]
                        + [f"{r['drift_counts']:.0f}" if r else "-" for r in rr])
    with open(os.path.join(out, "conditions_table.tsv"), "w") as f:
        f.write("\n".join("\t".join(x) for x in [head] + body) + "\n")
    td = 'style="border:1px solid #999;padding:4px 8px;font-family:Arial;font-size:11pt;text-align:center"'
    th = td.replace('text-align:center"', 'text-align:center;font-weight:bold;background:#eeeeee"')
    html = ("<!doctype html><meta charset='utf-8'><title>Condition comparison table</title>"
            "<p style='font-family:Arial'>Select the table (Ctrl+A), copy (Ctrl+C), paste into Google Slides (Ctrl+V).</p>"
            "<table style='border-collapse:collapse'>"
            + "<tr>" + "".join(f"<th {th}>{x}</th>" for x in head) + "</tr>"
            + "".join("<tr>" + "".join(f"<td {td}>{x}</td>" for x in row) + "</tr>" for row in body) + "</table>")
    with open(os.path.join(out, "conditions_table.html"), "w") as f:
        f.write(html)
    for row in [head] + body:
        print("  ".join(f"{x:>14s}" for x in row))
    print(f"\nSaved {os.path.relpath(out, HERE)}/ (conditions.png/.pdf, conditions.csv, conditions_table.tsv/.html)")


# ================================================================ repeated sessions
def sessions(specs, out, rates=None):
    """repeats of one test (NAME=modcomp_folder ...), from their summary.json (nothing re-decoded, v4 results):
    each session on its own, merged (errors and bits pooled over the sessions, 95 % Clopper-Pearson interval) and
    averaged (mean of the per-session BERs, bars = lowest to highest session). Figure a / b / c = sessions / merged /
    averaged, known-timing BER; the table (csv, tsv, html for Google Slides) also holds blind BER and frames."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import beta
    sets = []
    for s in specs:
        name, d = s.split("=", 1)
        rows = [r for r in json.load(open(os.path.join(d, "summary.json")))["rows"] if r["modulation"] in ("ook", "fsk")]
        sets.append((name, os.path.basename(os.path.normpath(d)), rows))
    common = sorted(set.intersection(*[{round(r["bps"]) for r in rows} for _, _, rows in sets]))
    rates = rates or common
    mods = [m for m in ("ook", "fsk") if any(r["modulation"] == m for _, _, rows, in sets for r in rows)]
    pick = lambda rows, mod, b: next((r for r in rows if r["modulation"] == mod and round(r["bps"]) == b), None)
    cp = lambda e, n: (beta.ppf(0.025, e, n - e + 1) if e else 0.0, beta.ppf(0.975, e + 1, n - e) if e < n else 1.0)
    table = []
    for mod in mods:
        for b in rates:
            per = [pick(rows, mod, b) for _, _, rows in sets]
            got = [p for p in per if p and p.get("bits")]
            if not got:
                continue
            row = dict(modulation=NAMES[mod], bps=b)
            for (name, _, _), p in zip(sets, per):
                row[f"{name} known"] = f"{p['errors']}/{p['bits']} ({p['errors'] / p['bits']:.2%})" if p and p.get("bits") else "-"
                row[f"{name} blind"] = (f"{p['blind_errors']}/{p['blind_bits']} ({p['blind_errors'] / p['blind_bits']:.2%}), "
                                        f"{p['blind_frames_valid']}/{p['frames_sent']} fr") if p and p.get("blind_bits") else "-"
            e, n = sum(p["errors"] for p in got), sum(p["bits"] for p in got)
            eb, nb = sum(p["blind_errors"] or 0 for p in got), sum(p["blind_bits"] or 0 for p in got)
            lo, hi = cp(e, n)
            bers = [p["errors"] / p["bits"] for p in got]
            row.update({"merged known": f"{e}/{n} ({e / n:.2%})", "merged known 95% CI": f"{lo:.2%} - {hi:.2%}",
                        "merged blind": f"{eb}/{nb} ({eb / nb:.2%}), {sum(p['blind_frames_valid'] for p in got)}/"
                                        f"{sum(p['frames_sent'] for p in got)} fr" if nb else "-",
                        "average known": f"{np.mean(bers):.2%} (range {min(bers):.2%} - {max(bers):.2%}, n={len(bers)})",
                        "_m": mod, "_e": e, "_n": n, "_lo": lo, "_hi": hi, "_bers": bers,
                        "_per": [p["errors"] / p["bits"] if p and p.get("bits") else np.nan for p in per],
                        "_nbits": [p["bits"] if p and p.get("bits") else np.nan for p in per]})
            table.append(row)
    ink2, grid = "#52514e", "#e4e3df"
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.0), sharey=True)
    styles = [dict(ls="-", filled=True), dict(ls="--", filled=False), dict(ls=":", filled=False), dict(ls="-.", filled=False)]
    x = np.arange(len(rates))
    off = {m: 0.07 * (i - (len(mods) - 1) / 2) for i, m in enumerate(mods)}
    floor = lambda v, n: max(v, 0.5 / n)        # 0 errors drawn at half a bit's share, marked v
    for mod in mods:
        rr = [r for r in table if r["_m"] == mod]
        xs = np.array([rates.index(r["bps"]) for r in rr]) + off[mod]
        c = COLORS[mod]
        for k, (name, _, _) in enumerate(sets):
            st = styles[k % len(styles)]
            y = np.array([r["_per"][k] for r in rr])
            nb = np.array([r["_nbits"][k] for r in rr], float)
            yd = np.where(y == 0, 0.5 / np.maximum(nb, 1), y)          # 0 errors: drawn at half a bit's share, as v
            xk = xs + 0.03 * (k - 1)
            ax[0].semilogy(xk, yd, ls=st["ls"], marker=MARKERS[mod], color=c, mfc=c if st["filled"] else "white", ms=7,
                           mew=1.6, lw=1.6, label=f"{NAMES[mod]}, {name}")
            if (y == 0).any():
                ax[0].plot(xk[y == 0], yd[y == 0], "v", color=c, ms=9, mfc=c if st["filled"] else "white", mew=1.6)
        y = np.array([r["_e"] / r["_n"] for r in rr])
        ax[1].errorbar(xs, np.maximum(y, 0.5 / np.array([r["_n"] for r in rr])),
                       yerr=[np.maximum(y - [r["_lo"] for r in rr], 0), np.array([r["_hi"] for r in rr]) - y],
                       fmt="-" + MARKERS[mod], color=c, ms=8, lw=2, capsize=4, label=NAMES[mod])
        m = np.array([np.mean(r["_bers"]) for r in rr])
        ax[2].errorbar(xs, m, yerr=[m - [min(r["_bers"]) for r in rr], np.array([max(r["_bers"]) for r in rr]) - m],
                       fmt="-" + MARKERS[mod], color=c, ms=8, lw=2, capsize=4, label=NAMES[mod])
    for r in table:                             # merged counts: at one rate the higher point above, the lower below
        yy = max(r["_e"] / r["_n"], 0.5 / r["_n"])
        top = max(max(q["_e"] / q["_n"], 0.5 / q["_n"]) for q in table if q["bps"] == r["bps"])
        ax[1].annotate(f"{r['_e']}/{r['_n']}", (rates.index(r["bps"]) + off[r["_m"]], yy), textcoords="offset points",
                       xytext=(0, 9) if yy == top else (0, -15), ha="center", fontsize=8, color=ink2)
    titles = ("Each session (known timing)", "Merged: errors / bits pooled (95 % CI)", "Averaged: mean of sessions (bars = min - max)")
    for a_, title, letter in zip(ax, titles, "abc"):
        a_.set_title(title, fontsize=10, loc="left")
        a_.set_xticks(x); a_.set_xticklabels([f"{b:g}" for b in rates]); a_.set_xlim(-0.5, len(rates) - 0.5)
        a_.set_xlabel("Bit rate (bps)")
        a_.axhline(0.5, color=ink2, ls="--", lw=0.8)
        a_.grid(color=grid, lw=0.6, which="both"); a_.set_axisbelow(True)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        a_.text(-0.12, 1.04, letter, transform=a_.transAxes, fontsize=14, fontweight="bold")
    ax[0].set_ylabel("Bit error rate"); ax[0].set_ylim(1e-4, 1)
    ax[0].legend(fontsize=8, frameon=False, loc="upper left", ncol=2, handlelength=3.5)
    ax[0].text(0.02, 0.03, "\u25bc = 0 errors (drawn at 0.5 / bits)", transform=ax[0].transAxes, fontsize=8, color=ink2)
    ax[2].text(0.02, 0.03, "a bar reaching the bottom: one session had 0 errors", transform=ax[2].transAxes, fontsize=8,
               color=ink2)
    ax[1].legend(fontsize=8, frameon=False, loc="upper left"); ax[2].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "sessions.png"), dpi=150); fig.savefig(os.path.join(out, "sessions.pdf"))
    plt.close(fig)
    head = [k for k in table[0] if not k.startswith("_")]
    body = [[str(r[k]) for k in head] for r in table]
    with open(os.path.join(out, "sessions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(body)
    with open(os.path.join(out, "sessions_table.tsv"), "w") as f:
        f.write("\n".join("\t".join(x) for x in [head] + body) + "\n")
    td = 'style="border:1px solid #999;padding:4px 8px;font-family:Arial;font-size:10pt;text-align:center"'
    th = td.replace('text-align:center"', 'text-align:center;font-weight:bold;background:#eeeeee"')
    open(os.path.join(out, "sessions_table.html"), "w").write(
        "<!doctype html><meta charset='utf-8'><title>Session table</title>"
        "<p style='font-family:Arial'>Select the table (Ctrl+A), copy (Ctrl+C), paste into Google Slides (Ctrl+V).</p>"
        "<table style='border-collapse:collapse'><tr>" + "".join(f"<th {th}>{x}</th>" for x in head) + "</tr>"
        + "".join("<tr>" + "".join(f"<td {td}>{x}</td>" for x in row) + "</tr>" for row in body) + "</table>")
    json.dump(dict(sessions={n: f for n, f, _ in sets}, rates=rates, decoder="v4 (frozen)",
                   rows=[{k: v for k, v in r.items() if not k.startswith("_")} for r in table]),
              open(os.path.join(out, "sessions.json"), "w"), indent=1)
    for r in table:
        print(f"{r['modulation']:4s} {r['bps']:4g}  " + "  ".join(f"{n}: {r[n + ' known']:>18}" for n, _, _ in sets)
              + f"  | merged {r['merged known']:>17} [{r['merged known 95% CI']}]  | average {r['average known']}")
    print(f"\nSaved {os.path.relpath(out, HERE)}/ (sessions.png/.pdf, sessions.csv, sessions_table.tsv/.html, sessions.json)")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description="OOK / FSK / DBPSK comparison over the stent -> maXTouch link")
    ap.add_argument("cmd", choices=("run", "analyse", "compare", "motion", "conditions", "sessions", "synth"))
    ap.add_argument("what", nargs="*", help="analyse: a label, or run folders; compare: modcomp_<label> folders; conditions / sessions: NAME=modcomp_<label> ...")
    ap.add_argument("--label", default="mc")
    ap.add_argument("--carrier", type=float, help="carrier = '1' tone [Hz] (2026-10-23 phantom sweep best: 159585)")
    ap.add_argument("--fsk-zero", type=float, help="FSK '0' tone [Hz]; pick a panel null (2026-10-23: 180556)")
    ap.add_argument("--fska-zero", type=float, default=53513.0,
                    help="FSK-A (experimental, --mods fska) '0' tone [Hz]: a tone the panel does see, ~10 dB under the "
                         "carrier through the body (fsk_tone_survey.py); decoded by decode_fska.py")
    ap.add_argument("--mods", default="ook,fsk,dbpsk")
    ap.add_argument("--rates", default="10,20,50,80,100", help="bit rates [bps]")
    ap.add_argument("--bits", type=int, default=5000, help="payload bits per run (rounded up to whole 127-bit frames)")
    ap.add_argument("--current-ma", type=float, help="drive current [mA], for the record and the figure title "
                                                       "(set on the hardware; this program does not change it)")
    ap.add_argument("--ref-bps", type=float, help="rate of the noise-added curves (default: lowest rate)")
    ap.add_argument("--condition", help="run condition for the record, e.g. still / swipe (also sets the --pause prompt)")
    ap.add_argument("--calibrate", choices=("finger-on", "finger-off"), default="finger-on",
                    help="body runs: calibrate with the finger on the panel (default, as before) or off it, then "
                         "place the finger (deltas show the whole touch; implies the --pause prompts)")
    ap.add_argument("--pause", action="store_true",
                    help="wait for Enter before each run (through-body: place the finger first, the run calibrates on it)")
    ap.add_argument("--lead", type=float, default=4.0, help="noise recorded before the schedule [s]")
    ap.add_argument("--y0", type=int, default=0, help="first Y node of X line 0 (stent responds on Y2-Y4)")
    ap.add_argument("--ny", type=int, default=28, help="nodes read (<= 28: one HID transfer, ~130 samples/s)")
    ap.add_argument("--patch", type=R.parse_patch, help="X0,Y0,SIZE: read a square elsewhere (slow)")
    ap.add_argument("--keep-noise-suppression", action="store_true")
    ap.add_argument("--keep-touch", action="store_true")
    ap.add_argument("--color-by", choices=("modulation", "condition"), default="modulation",
                    help="conditions: colour the lines by modulation (default) or by condition")
    ap.add_argument("--ber-min", type=float, help="conditions: bottom of the BER axis (same on figures shown side by side)")
    ap.add_argument("--counts", action="store_true", help="conditions: panel b shows error counts (linear axis)")
    ap.add_argument("--panel-model", action="store_true",
                    help="conditions: add the simulated panel-aware detector curve (panel_model.py) to panel c")
    ap.add_argument("--noise-bps", type=float, help="conditions: recording rate of the panel c curves (default: lowest)")
    ap.add_argument("--percent", action="store_true", help="conditions: BER axis of panel b in percent")
    ap.add_argument("--no-labels", action="store_true", help="conditions: no errors/bits numbers on the BER panel")
    ap.add_argument("--letters", default="abc", help="conditions: panel letters (e.g. def for a second row)")
    ap.add_argument("--snr-lim", help="conditions: Es/N0 axis range LO,HI [dB]")
    ap.add_argument("--colors", help="conditions --color-by condition: colours in condition order, e.g. '#d62728'")
    ap.add_argument("--datadir", help="synth: where to write (default: a new temp folder)")
    STX.add_hw_args(ap)
    a = ap.parse_args()
    if a.cmd == "motion":
        motion(a.what, os.path.join(DATADIR, f"modcomp_motion_{a.label}"))
        return
    if a.cmd == "sessions":
        sessions(a.what, os.path.join(DATADIR, f"modcomp_sessions_{a.label}"),
                 [int(float(r)) for r in a.rates.split(",")] if a.rates != "10,20,50,80,100" else None)
        return
    if a.cmd == "conditions":
        conditions(a.what, os.path.join(DATADIR, f"modcomp_conditions_{a.label}"), a.color_by == "condition",
                   a.colors.split(",") if a.colors else None,
                   [int(float(r)) for r in a.rates.split(",")] if a.rates != "10,20,50,80,100" else None,
                   a.ber_min, [float(v) for v in a.snr_lim.split(",")] if a.snr_lim else None, a.letters,
                   not a.no_labels, a.percent, a.counts, a.panel_model,
                   a.noise_bps)
        return
    if a.cmd == "compare":
        compare(a.what, os.path.join(DATADIR, f"modcomp_compare_{a.label}"))
        return
    if a.cmd == "analyse":
        if not a.what:
            ap.error("analyse needs a label or run folders")
        if len(a.what) == 1 and not os.path.isdir(a.what[0]):
            dirs = sorted(glob.glob(os.path.join(DATADIR, f"stent_mod_{a.what[0]}_*")))
            out = os.path.join(DATADIR, f"modcomp_{a.what[0]}")
        else:
            dirs = a.what
            out = os.path.join(DATADIR, f"modcomp_{a.label}")
        if not dirs:
            ap.error(f"no runs for {a.what[0]!r} in {DATADIR}")
        analyse(dirs, a.ref_bps, out, a.current_ma)
        return
    mods = [m.strip() for m in a.mods.split(",")]
    if any(m not in ALL_MODS for m in mods):
        ap.error(f"--mods: choose from {','.join(ALL_MODS)}")
    if a.cmd == "synth":
        a.carrier, a.fsk_zero = a.carrier or 159585.0, a.fsk_zero or 180556.0
        if a.bits == 5000:
            a.bits = 508                        # 4 frames: quick
        if a.rates == "10,20,50,80,100":
            a.rates = "20,80"
        synth(a)
        return
    if a.carrier is None:
        ap.error("run needs --carrier (from the probe)")
    if "fsk" in mods and a.fsk_zero is None:
        ap.error("FSK needs --fsk-zero (a frequency the panel barely hears, see the probe)")
    if not 0 <= a.y0 < R.NODES_PER_PAGE:
        ap.error("--y0 must be 0-63 (page 0)")
    run(a)


if __name__ == "__main__":
    main()
