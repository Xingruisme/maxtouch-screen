#!/usr/bin/env python3
"""
Blind level decoder for trx frames WITHOUT a preamble on the maXTouch tablet (2026-11-24): OOK / 2-ASK (M = 2) and
4-ASK / 4-FSK (M = 4) runs whose frames carry no known symbol (i2o_frame 'i2o_np'), and - for comparison - frames WITH
a sync word decoded as if it were not there (its symbols are never looked at, not for levels and not for decoding).

Why a separate decoder: both frozen decoders take every reference from the frame's sync symbols (decode_trx_tablet
rule 5: the threshold; decode_trx_mfsk M1 / M5: the node and the level amplitudes). Without a sync word they have
nothing and lose every frame. Here the references come from the data symbols themselves. The transmit levels differ
only in amplitude, so they need no labels: the strongest cluster is level 0, the weakest the last one.

What 'no preamble' does NOT change in the PER tests: the frame start. These tests always cut the frames at their known
positions in the schedule (decode_trx_tablet rule 3, 'known-timing BER'); the sync word was never used to find a frame.
A stand-alone receiver without a preamble needs that start from somewhere else (the touch-trigger timing) - not tested
here.

Rules (RULES_SHA below; first set 2026-11-24, variants compared on the 30 recordings of
askber_ankle_sitting_still_i2o_1mA_20261124_180111 with their sync ignored - so those recordings are development data):
 N1  feature, timing, scans per symbol, scan likelihood: decode_trx_mfsk M1 (feature), M2, M3, M4, called unchanged.
     The timing template is the sent symbols' nominal amplitudes (known-timing, as before).
 N2  body symbols = every frame's symbols after its sync (all of them for a frame without one). Only body symbols are
     used below.
 N3  per symbol an amplitude estimate r = sqrt(2 (mean a^2 - 1)) (a^2 of a level B has mean B^2 / 2 + 1).
 N4  levels per RECORDING and node, blind: B_top = the (1 - 1 / 2M) quantile of r over the recording's body symbols
     (the median of the strongest 1 / M of the symbols; the code bits are balanced, so each level holds ~1 / M of them).
     Start = nominal ladder x B_top, then EM_REC rounds of EM over all body symbols (update B^2 = 2 (weighted mean
     a^2 - 1)), each level clamped to [start / 2, 2 x start] (null: [0, max(2 x start, 5)]) and the levels kept in
     order. B_top < 5 lead sigma = no signal: every frame of that node undecodable.
 N5  per FRAME: node = the one of the 3 whose body symbols spread most = mean a^2 of its strongest quarter of symbols
     over that of its weakest quarter (the blind form of M1's sync score). Levels = that node's recording levels,
     NOT re-estimated per frame (EM_FRAME = 0). On the development recordings 4 rounds of per-frame EM made it worse
     (packet errors of 50, at 20..60 bit/s: 2-ASK 0 0 0 0 7 instead of 0 0 0 0 1, 4-ASK 0 0 2 1 2 instead of
     0 0 0 0 0): one frame's 48-96 symbols estimate the levels more noisily than the whole recording does, and these
     recordings (sitting still) drift little. A recording with strong drift is where this choice can lose.
     Node per frame or one node per recording made no difference (412 vs 408 symbol errors in all).
 N6  posterior over levels -> state soft values (decode_trx_mfsk.state_soft), code bits = body states (bpsk-mapped
     frames only), LLR = 2 atanh, then decode_trx_tablet.decode_frame (trx's decoders + CRC) - as M6 / M7.
Output: result_nopre.json in the run folder (the frozen decoders' result files are never touched).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode_trx_mfsk as MF                    # noqa: E402  (frozen; only called)

T = MF.T
R, TV2 = T.R, T.TV2
VERSION = "trx-nopre-1"
RULES_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]
EM_REC = 8
EM_FRAME = 0
NULL_B = MF.NULL_B


def sym_amp(scans):
    """N3 -> (r, has scans)"""
    has = np.array([len(x) > 0 for x in scans])
    r = np.array([np.sqrt(max(2.0 * (float(np.mean(x ** 2)) - 1.0), 0.0)) if len(x) else 0.0 for x in scans])
    return r, has


def em(scans, B0, iters):
    """clamped, ordered EM from start levels B0 (strongest first)"""
    if not iters or not len(scans):
        return B0.copy()
    M = len(B0)
    ns = np.array([len(x) for x in scans], float)
    a2 = np.array([float(np.sum(x ** 2)) for x in scans])
    lo = np.where(B0 >= NULL_B, B0 / 2, 0.0)
    hi = np.where(B0 >= NULL_B, 2 * B0, np.maximum(2 * B0, NULL_B))
    B = B0.copy()
    for _ in range(iters):
        w = MF.posteriors(scans, B) * (ns > 0)[:, None]
        num, den = (w * a2[:, None]).sum(0), (w * ns[:, None]).sum(0)
        Bn = np.sqrt(np.maximum(2.0 * (num / np.maximum(den, 1e-9) - 1.0), 0.0))
        B = np.sort(np.where(den > 1, np.clip(Bn, lo, hi), B))[::-1]
    return B


def recording_levels(scans, ladder_db, iters=EM_REC):
    """N4 -> (B, ok)"""
    M = len(ladder_db)
    r, has = sym_amp(scans)
    if has.sum() < 4 * M:
        return np.full(M, np.nan), False
    top = float(np.quantile(r[has], 1.0 - 0.5 / M))
    if top < NULL_B:
        return np.full(M, np.nan), False
    return em(scans, 10 ** (np.asarray(ladder_db, float) / 20) * top, iters), True


def spread_db(scans):
    """N5 node choice: strongest quarter of the symbols over the weakest quarter, mean a^2 [dB]"""
    e = np.sort([float(np.mean(x ** 2)) for x in scans if len(x)])
    q = len(e) // 4
    if q < 1:
        return -np.inf
    return 10 * np.log10((e[-q:].mean() + 1e-9) / (e[:q].mean() + 1e-9))


def decode_run(d, write=True, em_frame=EM_FRAME, em_rec=EM_REC, node_per_frame=True):
    meta = json.load(open(os.path.join(d, "meta.json")))
    if meta["modulation"] not in ("fsk", "fsk4", "ook"):
        raise ValueError(f"{d}: modulation {meta['modulation']} is not keyed")
    z = np.load(os.path.join(d, "capture.npz"))
    t, X = z["t"].astype(float), z["X"].astype(float)
    sq = np.load(os.path.join(d, "tx_seq.npz"))
    keys = [str(k) for k in sq["keys"]]
    enc, start, length = sq["enc"], sq["start"], sq["length"]
    u = np.unpackbits(sq["u"], axis=1)[:, :TV2.N_INFO]
    states = np.unpackbits(sq["phase"])[:int(sq["n_phase"])].astype(np.uint8)
    M, lad = MF.run_M(meta), MF.ladder(meta)
    per = 2 if M == 4 else 1
    lv_all = MF.levels_of_states(states, M)
    n = len(lv_all)
    Ts = meta["plan"]["sym_us"] / 1e6
    top, y, lead = T.front(dict(meta, modulation="fsk"), t, X)
    tau, eps, r_time = T.timing_keyed(t, y, meta["t_first"], Ts, 10 ** (np.asarray(lad)[lv_all] / 20))
    node_scans = {j: MF.symbol_scans(t, MF.scan_feature(X, j, lead), tau, Ts, eps, n) for j in top}
    names = R.node_names(meta)
    body = []                                   # (frame, first body symbol, body symbols, sync states)
    for i in range(len(enc)):
        key = keys[enc[i]]
        s0, L = int(start[i]), int(length[i])
        nsync, pair, mapping, _ = T.frame_layout(key)
        if pair or mapping != "bpsk":
            raise ValueError(f"frame {key}: only bpsk-mapped, unpaired frames (body state = code bit)")
        if s0 % per or L % per or nsync % per:
            raise ValueError(f"frame {i} not aligned to {per}-state symbols")
        if (s0 + L) // per <= n:
            body.append((i, (s0 + nsync) // per, (L - nsync) // per, nsync))
    in_body = np.zeros(n, bool)
    for _, b0, nb, _ in body:
        in_body[b0:b0 + nb] = True
    rec = {j: recording_levels([node_scans[j][k] for k in np.where(in_body)[0]], lad, em_rec) for j in top}
    sc0 = node_scans[top[0]]
    res = dict(decoder=VERSION, rules_sha=RULES_SHA, run=os.path.basename(os.path.normpath(d)),
               modulation=meta["modulation"], M=M, baud=1 / Ts, bits_per_symbol=per, bit_rate=per / Ts,
               ladder_db=list(lad), nodes=[names[j] for j in top], symbols=n,
               scans_per_symbol=float(np.mean([len(x) for x in sc0])), delay_ms=(tau - meta["t_first"]) * 1e3,
               clock_ppm=eps * 1e6, timing_r=r_time, erasures=int(sum(len(x) == 0 for x in sc0)), signal_lost=False,
               sync_symbols_ignored=int(sum(b[3] for b in body)),
               recording_levels={names[j]: [float(x) for x in rec[j][0]] for j in top})
    rec_node = max(top, key=lambda j: spread_db([node_scans[j][k] for k in np.where(in_body)[0]]))
    frames, sym_err, sym_n, st_err, st_n = [], 0, 0, 0, 0
    for i, b0, nb, nsync in body:
        key = keys[enc[i]]
        node = max(top, key=lambda j: spread_db(node_scans[j][b0:b0 + nb])) if node_per_frame else rec_node
        scans = node_scans[node][b0:b0 + nb]
        Brec, ok = rec[node]
        B = em(scans, Brec, em_frame) if ok else Brec
        post = MF.posteriors(scans, B) if ok else np.full((nb, M), 1.0 / M)
        s = MF.state_soft(post, M)              # body states, air order
        sent_lv = lv_all[b0:b0 + nb]
        sym_err += int(np.sum(post.argmax(1) != sent_lv)); sym_n += nb
        sent_st = states[b0 * per:(b0 + nb) * per]
        st_err += int(np.sum((s > 0).astype(np.uint8) != sent_st)); st_n += len(sent_st)
        dd = 2 * np.arctanh(np.clip(-s, -1 + 1e-12, 1 - 1e-12))     # code bit = body state; positive = 0
        pre, n_code, out = T.decode_frame(key, dd, u[i])
        frames.append(dict(i=int(meta["frame_index0"] + i), round=int(meta["round0"] + i // len(keys)), frame=key,
                           t0=float(tau + b0 * Ts * (1 + eps)), erasures=int(sum(len(x) == 0 for x in scans)),
                           node=names[node], reference_ok=bool(ok), levels=[float(b) for b in B], pre_err=pre,
                           code_bits=n_code,
                           decoders={k: dict(r, frame_err=r["info_err"] > 0, crc_missed=r["crc_ok"] and r["info_err"] > 0)
                                     for k, r in out.items()}))
    res.update(symbol_errors=sym_err, symbols_decoded=sym_n, state_errors=st_err, state_symbols=st_n, frames=frames)
    if write:
        json.dump(res, open(os.path.join(d, "result_nopre.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = decode_run(p)
        print(p, f"M={r['M']} r={r['timing_r']:.2f} symbol errors {r['symbol_errors']}/{r['symbols_decoded']}",
              sum(f["decoders"][f["frame"]]["info_err"] > 0 for f in r["frames"]), "of", len(r["frames"]), "frames wrong")
