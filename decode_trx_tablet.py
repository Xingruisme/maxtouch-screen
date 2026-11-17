#!/usr/bin/env python3
"""
Decoder for the trx-frame BER runs through the body to the maXTouch tablet (stent_trx_ber.py), 2026-11-14.
The frames are the stent-to-stent link's FEC frames from trx_psk_node (trx_psk_link_v2.build_seq, unchanged): per round
one frame of each --send encoding in shuffled order, 96 info bits each (80 random + CRC-16), sync 32 (uncoded) or 64
(coded) symbols, encoders/decoders from stent_fec (unchanged). What differs from trx is only the receiver front end:
the panel gives ~125 scans/s of one number per node, not I/Q, so the symbol soft values come from the panel feature.

Rules (fixed 2026-11-14 before any recording with these frames; decode_v4 / stent_modcomp / trx code are only called):
  1. Nodes: the 3 with the largest energy increase after START TX vs the lead (v4 rule 1, stent_modcomp.energy_rank).
  2. Feature: OOK/FSK = v4 absdev z-scored, max over the 3 nodes (v4 rule 2). DBPSK = stent_modcomp.dbpsk_y (signed sum
     of (x - OFF baseline) / OFF sigma over the 3 nodes).
  3. Timing (known-timing BER, as trx's physical FEC decode, which also cuts frames at their known positions):
     OOK/FSK: symbol edge k at tau + k * Ts * (1 + eps); tau in the known first-symbol time + 0..25 ms (1 ms, then
     +-1 ms in 0.1 ms), eps in +-200 ppm (10 ppm) - as v4's known window and clock range - maximising the Pearson
     correlation of the per-symbol values with the sent symbol states over the whole run.
     DBPSK (constant envelope, no state in the amplitude): the stent_modcomp DBPSK rule - first / last |y| > 10 lead
     rms give the carrier-on / -off edges -> tau and eps (clipped to +-200 ppm).
  4. Symbol value = mean feature of the samples more than 2 ms from both symbol edges (v4 rule 4), else of all its
     samples; a symbol with no sample is an ERASURE (soft value 0 = 'unknown' for every decoder; v4 dropped the frame).
  5. State soft value per frame, from that frame's own sync symbols only (never its body):
     OOK/FSK: s = (v - thr) / ((hi - lo) / 2), hi / lo = mean value of the sync symbols sent as state 1 / 0, thr their
       midpoint (v4 rule 5 on the sync instead of header + gap). State 1 = carrier (OOK) / carrier tone (FSK).
     DBPSK: s = v * sign / scale, sign = sign of sum over sync symbols of v * (2 state - 1), scale = mean |v| over them.
     s clipped to +-3 (one outlier scan must not outweigh the code).
  6. Code-bit soft values d (positive = 0, the stent_fec convention): bpsk-mapped frames (body state = code bit):
     d = -s; dpsk-mapped frames (none, none_full, d_cc4, d_bch63: code bit = state change): d_k = s_k * s_(k-1), the
     first body symbol against the last sync symbol. Paired frames ('none', two symbols per bit): s = mean of the pair.
  7. Decoding: every trx_psk_link_v2.DECODERS entry of the frame's encoding (stent_fec.Fec.decode, its quantisation);
     uncoded = sign of d. CRC-16 checked with stent_fec.crc_ok.
  8. Accounting per frame: BEFORE FEC = hard decisions on the code bits (d < 0) vs the sent code bits (coded-bit BER,
     the channel as the decoder sees it; erasures decided 0 and counted); AFTER FEC = decoded info bits vs the sent 96
     (info BER), frame error = any info bit wrong, crc_ok, crc_missed = CRC passed but bits wrong. Also the state BER
     of all symbols of the run (sign of s vs sent state). A frame with any symbol outside the recording is skipped.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stent_modcomp as SM                      # noqa: E402  (energy_rank, dbpsk_y, offbase; imports decode_v4)
V4, R = SM.V4, SM.R
TRX_DIR = os.path.normpath(os.path.join(HERE, "..", "trx_psk_node", "transceiver"))
sys.path.append(TRX_DIR)
with contextlib.redirect_stdout(io.StringIO()):
    import trx_psk_link_v2 as TV2               # noqa: E402  (build_seq, ENCODINGS, DECODERS, make_fec) - not edited
F = TV2.F                                       # stent_fec

VERSION = "trx-tablet-1"
FROZEN_AT = "2026-11-14 14:00"                  # rules fixed before any recording with trx frames
RULES_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]
TRX_SHAS = {n: hashlib.sha256(open(os.path.join(TRX_DIR, n), "rb").read()).hexdigest()[:12]
            for n in ("stent_fec.py", "trx_psk_link_v2.py")}
GUARD_MS = V4.GUARD_MS
DELAY_MS = (0.0, 25.0)
EPS_PPM = np.arange(-200, 201, 10)
CLIP = 3.0


# ================================================================ frames
def frame_layout(key):
    """(sync symbols on air, pair, mapping, sync states) of an encoding"""
    f = TV2.make_fec(key)
    nsync = getattr(f, "nsync", F.NSYNC)
    sync = TV2.PS32 if nsync == 32 else F.PS
    pair = TV2.ENCODINGS[key][1]
    return nsync, pair, f.mapping, np.asarray(sync, np.uint8)


def code_bits(key, u):
    """sent code bits (air order, before differential encoding) of info rows u (B, 96)"""
    f = TV2.make_fec(key)
    return np.atleast_2d(u).astype(np.uint8) if f.kind == "none" else f.encode(u)


# ================================================================ front end + timing
def front(meta, t, X):
    lead = t < meta["t_prompt"]
    top, _ = SM.energy_rank(X.astype(float), t, meta["t_prompt"])
    if meta["modulation"] == "dbpsk":
        y = SM.dbpsk_y(X, lead, top)
    else:
        y = np.max([R.zscore(V4.absdev(X[:, i].astype(float), lead)) for i in top], axis=0)
    return top, y, lead


def sym_index(t, tau, Ts, eps):
    u = (t - tau) / (Ts * (1 + eps))
    k = np.floor(u).astype(int)
    return k, u - k


def sym_values(t, y, tau, Ts, eps, n):
    """v4 rule 4 per symbol; NaN = no sample (erasure)"""
    k, pos = sym_index(t, tau, Ts, eps)
    g = GUARD_MS / 1e3 / (Ts * (1 + eps))
    inside = (k >= 0) & (k < n)
    v = np.full(n, np.nan)
    for sel in (inside, inside & (pos > g) & (pos < 1 - g)):      # guarded samples overwrite the all-sample fallback
        s = np.bincount(k[sel], y[sel], n); c = np.bincount(k[sel], minlength=n)
        v = np.where(c > 0, s / np.maximum(c, 1), v)
    return v


def corr_states(v, states):
    ok = ~np.isnan(v)
    if ok.sum() < 10 or np.std(v[ok]) == 0:
        return -np.inf
    return float(np.corrcoef(v[ok], states[ok])[0, 1])


def timing_keyed(t, y, t_first, Ts, states):
    n = len(states)
    best = (-np.inf, t_first, 0.0)
    for d in np.arange(DELAY_MS[0], DELAY_MS[1] + 1e-9, 1.0):
        c = corr_states(sym_values(t, y, t_first + d / 1e3, Ts, 0.0, n), states)
        if c > best[0]:
            best = (c, t_first + d / 1e3, 0.0)
    tau_c = best[1]
    for eps in EPS_PPM * 1e-6:
        for tau in np.arange(tau_c - 0.001, tau_c + 0.001 + 1e-9, 0.0001):
            c = corr_states(sym_values(t, y, tau, Ts, eps, n), states)
            if c > best[0]:
                best = (c, tau, eps)
    return best[1], best[2], best[0]


def timing_dbpsk(t, y, lead, t_first, Ts, n):
    sy = max(1.4826 * float(np.median(np.abs(y[lead] - np.median(y[lead])))), 1e-9)
    on = np.abs(y) > 10 * sy
    after = np.where(on & (t >= t_first - 0.02))[0]
    end = t_first + n * Ts + 1.0
    if len(after) < 10 or after[0] == 0:
        return None, None
    i1, i2 = after[0], after[t[after] <= end][-1]
    t_on = 0.5 * (t[i1 - 1] + t[i1])
    t_off = 0.5 * (t[i2] + t[min(i2 + 1, len(t) - 1)])
    return t_on, float(np.clip((t_off - t_on) / (n * Ts) - 1, -200e-6, 200e-6))


# ================================================================ soft values + decoding
def state_soft(mod, v, sync_states):
    """rule 5 for one frame's symbol values v (sync first)"""
    ns = len(sync_states)
    vs, ss = v[:ns], sync_states
    ok = ~np.isnan(vs)
    s = np.zeros(len(v))
    if mod == "dbpsk":
        if ok.sum() < 4:
            return s, False
        sign = np.sign(np.sum(vs[ok] * (2.0 * ss[ok] - 1))) or 1.0
        scale = np.mean(np.abs(vs[ok])) + 1e-12
        s = np.where(np.isnan(v), 0.0, v * sign / scale)
    else:
        hi, lo = vs[ok & (ss == 1)], vs[ok & (ss == 0)]
        if not len(hi) or not len(lo) or hi.mean() <= lo.mean():
            return s, False                     # no usable reference: every symbol unknown
        thr, half = 0.5 * (hi.mean() + lo.mean()), 0.5 * (hi.mean() - lo.mean())
        s = np.where(np.isnan(v), 0.0, (v - thr) / half)
    return np.clip(s, -CLIP, CLIP), True


def code_soft(key, s_frame):
    """rule 6: state soft values of one frame (air symbols, pairs included) -> code-bit soft values d (positive = 0)"""
    nsync, pair, mapping, _ = frame_layout(key)
    s = s_frame.reshape(-1, 2).mean(1) if pair else s_frame
    if mapping == "bpsk":
        return -s[nsync:]
    return s[nsync:] * s[nsync - 1:-1]


def decode_frame(key, d, u):
    """rule 7-8 for one frame -> {decoder: dict(info_err, crc_ok)} and the coded-bit errors before FEC"""
    c = code_bits(key, u)[0]
    pre = int(np.sum((d < 0).astype(np.uint8) != c))
    out = {}
    for name, (k2, q, _) in TV2.DECODERS.items():
        if k2 != key:
            continue
        f = TV2.make_fec(key, q)
        bits = (np.atleast_2d(d) < 0).astype(np.uint8) if f.kind == "none" else f.decode(d)
        bits = bits[0]
        out[name] = dict(info_err=int(np.sum(bits != u)), crc_ok=bool(F.crc_ok(bits)[0]))
    return pre, len(c), out


def decode_run(d):
    """one run folder (meta.json, capture.npz, tx_seq.npz) -> result_trx.json"""
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "capture.npz"))
    t, X = z["t"].astype(float), z["X"].astype(float)
    sq = np.load(os.path.join(d, "tx_seq.npz"))
    keys = [str(k) for k in sq["keys"]]
    enc, start, length = sq["enc"], sq["start"], sq["length"]
    u = np.unpackbits(sq["u"], axis=1)[:, :TV2.N_INFO]
    states = np.unpackbits(sq["phase"])[:int(sq["n_phase"])].astype(np.uint8)
    mod, Ts, n = meta["modulation"], meta["plan"]["sym_us"] / 1e6, len(states)
    top, y, lead = front(meta, t, X)
    names = R.node_names(meta)
    res = dict(decoder=VERSION, rules_sha=RULES_SHA, frozen_at=FROZEN_AT, trx_shas=TRX_SHAS, run=os.path.basename(
        os.path.normpath(d)), modulation=mod, baud=1 / Ts, carrier_hz=meta["plan"]["carrier_hz"],
        zero_hz=meta["plan"].get("zero_hz"), nodes=[names[i] for i in top], symbols=n,
        samples_per_symbol=float(Ts * (len(t) - 1) / (t[-1] - t[0])))
    if mod == "dbpsk":
        tau, eps = timing_dbpsk(t, y, lead, meta["t_first"], Ts, n)
        if tau is None:
            res.update(signal_lost=True, frames=[])
            json.dump(res, open(os.path.join(d, "result_trx.json"), "w"), indent=1)
            return res
        r_time = None
    else:
        tau, eps, r_time = timing_keyed(t, y, meta["t_first"], Ts, states)
    v = sym_values(t, y, tau, Ts, eps, n)
    frames, s_all = [], np.zeros(n)
    have = np.zeros(n, bool)
    for i in range(len(enc)):
        key = keys[enc[i]]
        a, L = int(start[i]), int(length[i])
        if a + L > n:
            continue
        nsync, pair, mapping, sync = frame_layout(key)
        sync_air = np.repeat(sync, 2) if pair else sync
        s, ok_ref = state_soft(mod, v[a:a + L], sync_air)
        s_all[a:a + L], have[a:a + L] = s, True
        dd = code_soft(key, s)
        pre, n_code, out = decode_frame(key, dd, u[i])
        frames.append(dict(i=int(meta["frame_index0"] + i), round=int(meta["round0"] + i // len(keys)), frame=key,
                           t0=float(tau + a * Ts * (1 + eps)), erasures=int(np.isnan(v[a:a + L]).sum()),
                           reference_ok=ok_ref, pre_err=pre, code_bits=n_code,
                           decoders={k: dict(r, frame_err=r["info_err"] > 0, crc_missed=r["crc_ok"] and r["info_err"] > 0)
                                     for k, r in out.items()}))
    st_ok = have & (s_all != 0)
    res.update(signal_lost=False, delay_ms=(tau - meta["t_first"]) * 1e3, clock_ppm=eps * 1e6, timing_r=r_time,
               erasures=int(np.isnan(v).sum()), state_errors=int(np.sum((s_all[st_ok] > 0) != states[st_ok].astype(bool))),
               state_symbols=int(st_ok.sum()), frames=frames)
    json.dump(res, open(os.path.join(d, "result_trx.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = decode_run(p)
        print(p, r.get("delay_ms"), r.get("state_errors"), "/", r.get("state_symbols"), len(r["frames"]), "frames")
