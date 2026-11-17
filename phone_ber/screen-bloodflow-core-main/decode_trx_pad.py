#!/usr/bin/env python3
"""
Decoder for the trx-frame BER runs through the body to the OnePlus Pad (pad_trx_ber.py), 2026-11-14.
Same frames and same FEC accounting as the maXTouch tablet runs (decode_trx_tablet.py, frozen e76afd59a7b5 - only its
frame / soft-value / decoding functions are CALLED here, nothing in it is changed). What differs is the Pad front end:
the Pad's /proc/touchpanel/debug_info/delta map is frozen (identical reads) while the controller is idle and refreshes
on every read while a through-body carrier is on (sweep 2026-11-14: changed in 100 % of ON frames at 9.9-117 kHz,
0.7 % in the no-TX control). So the per-read feature is "how much of the map changed since the previous read".

Rules (fixed 2026-11-14 before any Pad recording with trx frames):
  1. Feature per read k: y_k = share of the 38 x 54 nodes whose value differs from read k-1 (0..1); y_0 = 0. Its time
     is the read's device stamp on the host clock (t in frames.npz).
  2. Timing (known-timing BER, as the tablet decoder / trx): symbol edge j at tau + j * Ts * (1 + eps). Coarse: tau =
     t_first + 0..600 ms in 10 ms steps, eps = 0 (the Pad's refresh starts ~0.2 s after the carrier, sweep median
     0.20 s); fine: tau +-10 ms in 1 ms steps, eps +-200 ppm in 10 ppm steps; maximise the Pearson correlation of the
     per-symbol values with the sent states over the whole recording.
  3. Symbol value = mean y of the reads inside the symbol (decode_trx_tablet.sym_values: its 2 ms edge guard only
     matters when a read sits within 2 ms of an edge); no read = ERASURE (soft value 0).
  FSK (state 1 = --carrier, which the Pad answers; state 0 = --fsk-zero, a tone in the Pad's no-response band, sweep
     129-172 kHz: 0 % refresh at 156 and 172 kHz) uses the same feature and rules: the Pad refreshes for state 1 only.
  DBPSK (rule D, added 2026-11-14 23:00 before any Pad DBPSK recording; OOK/FSK rules above unchanged): the carrier is
     on for the whole frame, so "changed share" carries nothing; the phase can only show up as the SIGN of the carrier
     interference the panel samples. D1 feature: reads inside the transmission (tau .. tau + n Ts) -> subtract their
     mean map, first principal component (SVD) of those reads = the dominant interference pattern; y_k = projection of
     read k on it (sign arbitrary - rule 5 resolves it on the sync). D2 timing: no amplitude pattern to sync on, so
     tau = t_first + 134 ms (median Pad delay of the four OOK/FSK recordings 2026-11-14 22:06-22:18: 156, 132, 120,
     136 ms), eps = 0 (their clocks: -80..+130 ppm = < 20 ms over a recording, << one symbol). D3 = rule 3. Then rules
     4-8 with decode_trx_tablet's DBPSK rule 5 (sign from the sync, scale = mean |v|). Diagnostic only (not used):
     dbpsk_r = Pearson r of the symbol values with the sent phase states (0 = the phase is invisible).
  4-8. Unchanged from decode_trx_tablet rules 5-8 (OOK/FSK): state soft value from the frame's own sync symbols
     (hi / lo = mean of sync symbols sent as carrier / off, threshold their midpoint, clipped +-3), code-bit soft values
     (bpsk / dpsk mapping, pairs), every trx_psk_link_v2.DECODERS entry of the frame's encoding, CRC-16; BEFORE FEC =
     hard code-bit errors, AFTER FEC = info-bit errors of the 96, frame error, CRC pass / missed; state BER of the run.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.normpath(os.path.join(HERE, "..", "maxtouch-screen-main"))
sys.path.insert(0, TAB)
import decode_trx_tablet as DT                 # noqa: E402  (frozen; frame_layout, sym_values, state_soft, code_soft, decode_frame)
TV2, F = DT.TV2, DT.F

VERSION = "trx-pad-1"
FROZEN_AT = "2026-11-14 22:00"
RULES_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]
TABLET_SHA = DT.RULES_SHA
COARSE_MS = np.arange(0.0, 600.0 + 1e-9, 10.0)
DBPSK_DELAY_S = 0.134
EPS_PPM = np.arange(-200, 201, 10)


def feature(Fm):
    """rule 1: share of nodes changed since the previous read"""
    Fm = np.asarray(Fm)
    y = np.zeros(len(Fm))
    if len(Fm) > 1:
        y[1:] = (np.diff(Fm.astype(np.int32), axis=0) != 0).reshape(len(Fm) - 1, -1).mean(1)
    return y


def dbpsk_feature(Fm, t, tau, Ts, n):
    """rule D1"""
    X = np.asarray(Fm, float).reshape(len(Fm), -1)
    on = (t >= tau) & (t < tau + n * Ts)
    if on.sum() < 3:
        return np.zeros(len(X))
    mu = X[on].mean(0)
    _, _, vt = np.linalg.svd(X[on] - mu, full_matrices=False)
    return (X - mu) @ vt[0]


def timing(t, y, t_first, Ts, states):
    """rule 2"""
    n = len(states)
    best = (-np.inf, t_first, 0.0)
    for d in COARSE_MS:
        c = DT.corr_states(DT.sym_values(t, y, t_first + d / 1e3, Ts, 0.0, n), states)
        if c > best[0]:
            best = (c, t_first + d / 1e3, 0.0)
    tc = best[1]
    for eps in EPS_PPM * 1e-6:
        for tau in np.arange(tc - 0.010, tc + 0.010 + 1e-9, 0.001):
            c = DT.corr_states(DT.sym_values(t, y, tau, Ts, eps, n), states)
            if c > best[0]:
                best = (c, tau, eps)
    return best[1], best[2], best[0]


def decode_arrays(t, y, meta, sq, Fm=None):
    """rules 2-8 on a feature series -> result dict (frames with before / after FEC accounting)"""
    keys = [str(k) for k in sq["keys"]]
    enc, start, length = sq["enc"], sq["start"], sq["length"]
    u = np.unpackbits(sq["u"], axis=1)[:, :TV2.N_INFO]
    states = np.unpackbits(sq["phase"])[:int(sq["n_phase"])].astype(np.uint8)
    Ts, n = meta["plan"]["sym_us"] / 1e6, len(states)
    mod = meta.get("modulation", "ook")
    extra = {}
    if mod == "dbpsk":
        tau, eps, r_time = meta["t_first"] + DBPSK_DELAY_S, 0.0, None
        y = dbpsk_feature(Fm, t, tau, Ts, n)
        v = DT.sym_values(t, y, tau, Ts, eps, n)
        ok = ~np.isnan(v)
        extra = dict(dbpsk_r=float(np.corrcoef(v[ok], states[ok])[0, 1]) if ok.sum() > 10 and np.std(v[ok]) > 0 else None)
    else:
        tau, eps, r_time = timing(t, y, meta["t_first"], Ts, states)
        v = DT.sym_values(t, y, tau, Ts, eps, n)
    frames, s_all, have = [], np.zeros(n), np.zeros(n, bool)
    for i in range(len(enc)):
        key = keys[enc[i]]
        a, L = int(start[i]), int(length[i])
        if a + L > n:
            continue
        nsync, pair, mapping, sync = DT.frame_layout(key)
        sync_air = np.repeat(sync, 2) if pair else sync
        s, ok_ref = DT.state_soft(mod, v[a:a + L], sync_air)
        s_all[a:a + L], have[a:a + L] = s, True
        pre, n_code, out = DT.decode_frame(key, DT.code_soft(key, s), u[i])
        frames.append(dict(i=int(meta["frame_index0"] + i), round=int(meta["round0"] + i // len(keys)), frame=key,
                           t0=float(tau + a * Ts * (1 + eps)), erasures=int(np.isnan(v[a:a + L]).sum()),
                           reference_ok=ok_ref, pre_err=pre, code_bits=n_code,
                           decoders={k: dict(r, frame_err=r["info_err"] > 0, crc_missed=r["crc_ok"] and r["info_err"] > 0)
                                     for k, r in out.items()}))
    st_ok = have & (s_all != 0)
    return dict(decoder=VERSION, rules_sha=RULES_SHA, frozen_at=FROZEN_AT, tablet_rules_sha=TABLET_SHA,
                trx_shas=DT.TRX_SHAS, modulation=mod, baud=1 / Ts, carrier_hz=meta["plan"]["carrier_hz"],
                zero_hz=meta["plan"].get("zero_hz"),
                symbols=n, reads_per_symbol=float(Ts * (len(t) - 1) / (t[-1] - t[0])), signal_lost=False,
                delay_ms=(tau - meta["t_first"]) * 1e3, clock_ppm=eps * 1e6, timing_r=r_time,
                erasures=int(np.isnan(v).sum()),
                state_errors=int(np.sum((s_all[st_ok] > 0) != states[st_ok].astype(bool))),
                state_symbols=int(st_ok.sum()), nodes=["first principal component of the on-air reads" if mod == "dbpsk"
                                                       else "changed share (whole map)"], frames=frames, **extra)


def decode_run(d):
    """one run folder (meta.json, frames.npz, tx_seq.npz) -> result_trx.json"""
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "frames.npz"))
    sq = np.load(os.path.join(d, "tx_seq.npz"))
    t = z["t"].astype(float)
    res = decode_arrays(t, feature(z["F"]), meta, sq, z["F"])
    res["run"] = os.path.basename(os.path.normpath(d))
    json.dump(res, open(os.path.join(d, "result_trx.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = decode_run(p)
        print(p, f"delay {r['delay_ms']:.0f} ms", r["state_errors"], "/", r["state_symbols"], len(r["frames"]), "frames")
