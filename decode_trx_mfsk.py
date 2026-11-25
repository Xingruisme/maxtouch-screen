#!/usr/bin/env python3
"""
M-ary amplitude-discriminated FSK decoder for the maXTouch tablet (trx frames), 2026-11-20. M = 2 (the existing 2-FSK
159.585 / 180.556 kHz runs, decoded a second time with the same likelihood rules as 4-FSK) or M = 4 (fsk4 runs).

Why 'amplitude-discriminated': the panel cannot measure frequency. Each scan reports the energy it picked up from the
carrier at a random phase (consecutive scans uncorrelated). A tone is recognised only by HOW STRONGLY the panel answers
it. 2-FSK works because 180.556 kHz falls in a panel null; 4-FSK needs 4 tones the panel answers at 4 different,
stable levels. From the 11 valid 10-26 body sweeps (scratchpad fsk4_survey.py), relative to 159.585 kHz:
   level 0  159.585 kHz   0 dB
   level 1  151.895 kHz  -8.3 dB  [-10.3 .. -5.7]
   level 2  130.979 kHz -19.3 dB  [-21.4 .. -18.2]
   level 3  180.556 kHz  ~-40 dB  (null, the 2-FSK '0' tone)

Frames are unchanged: the air sequence is trx's binary state stream (decode_trx_tablet / stent_trx_ber). 4-FSK sends
states (2k, 2k+1) as ONE symbol, Gray-mapped onto the level ladder (neighbouring levels differ in one state):
   (1,1) -> level 0   (1,0) -> level 1   (0,0) -> level 2   (0,1) -> level 3
so 2 states per symbol; at the same symbol rate 4-FSK carries twice the bits of 2-FSK.

Rules (RULES_SHA below; fixed 2026-11-20 before any fsk4 recording - change only as a new VERSION):
 M1  feature per scan and node, for the 3 nodes stent_modcomp.energy_rank picks (as v4): a = |x - b| / sigma with
     decode_v4.absdev's own baseline b (running median of quiet samples) and lead noise sigma (a = sqrt(10^absdev - 1)).
     Per FRAME one node: the one whose known sync symbols separate best = mean a^2 of the strongest sync level present
     over mean a^2 of the weakest (null) sync level. (Energy rank alone picked the finger node, whose touch drift lifts
     its null scans to 100-580 in 2 of the 2-FSK development runs while a neighbour stayed at 3-30.)
 M2  timing: decode_trx_tablet.timing_keyed on the v4 feature with the symbols' nominal amplitudes as the template.
 M3  scans per symbol: inside the symbol and >= GUARD_MS from both edges; a symbol without such scans uses all its
     scans (v4 rule 4); none at all = erasure (both states soft 0).
 M4  likelihood of one scan for level amplitude B: a = |B cos(theta) + n|, theta uniform, n ~ N(0, 1), mixed with a
     1e-3 uniform outlier floor; a symbol's log-likelihood = sum over its scans.
 M5  level amplitudes from KNOWN symbols only (no decision-directed updates): per RECORDING and node, B_i =
     sqrt2 x median of the scans of all sync symbols of level i in the recording (the median of |B cos theta| is
     B / sqrt2; robust to scans leaking in from a neighbouring symbol); a level with < 4 such symbols = nominal ladder x
     the median of the measured levels' ratios. Per FRAME start: a level with >= 4 sync symbols in THIS frame = sqrt2 x
     median of those scans; the others = recording level x the frame gain c = sqrt2 x median(a / B_level) over the
     frame's sync scans of non-null levels (B >= 5). Then EM_ITERS rounds of EM over the frame's symbols (sync
     fixed to its known level; update B^2 = 2(weighted mean a^2 - 1)), each level CLAMPED to [start / 2, 2 x start]
     (null: [0, max(2 x start, 5)]) - follows drift within a recording (fixed per-recording levels: 52 vs 19 frame
     errors on the 2-FSK development runs) without collapsing (unclamped EM: simulated level 0 188-238 instead of ~470,
     a null pulled to ~200 by leaking scans).
 M6  per symbol: posterior over levels -> LLR of each state (log-sum over levels), clipped to +-LLR_CLIP;
     state soft value s = tanh(LLR / 2).
 M7  code bits: decode_trx_tablet.code_soft (unchanged; for the differential mapping it multiplies neighbouring s, the
     exact soft XOR), converted back to an LLR 2 atanh(d), then decode_trx_tablet.decode_frame (trx's decoders + CRC).
Output: result_mfsk.json in the run folder (result_trx.json of the frozen tablet decoder is never touched).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode_trx_tablet as T                   # noqa: E402  (frozen; only called)

V4, SM, R, TV2, F = T.V4, T.SM, T.R, T.TV2, T.F
VERSION = "trx-mfsk-1"
FROZEN_AT = "2026-11-20 12:40"                  # rules fixed before any fsk4 recording
RULES_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]
GUARD_MS = T.GUARD_MS
MIN_SYNC = 4
NULL_B = 5.0
EM_ITERS = 4
LLR_CLIP = 12.0
OUTLIER = 1e-3
NTHETA = 64
LADDER4_DB = (0.0, -8.3, -19.3, -40.0)          # nominal level of each tone (survey medians), strongest first
LADDER2_DB = (0.0, -40.0)
LEVEL_OF_PAIR = {(1, 1): 0, (1, 0): 1, (0, 0): 2, (0, 1): 3}
PAIR_OF_LEVEL = {v: k for k, v in LEVEL_OF_PAIR.items()}


# ================================================================ mapping
def levels_of_states(states, M):
    """binary air states -> level index per symbol (M=2: state 1 = level 0, state 0 = level 1)"""
    s = np.asarray(states, np.uint8)
    if M == 2:
        return (1 - s).astype(int)
    if len(s) % 2:
        raise ValueError("4-FSK needs an even number of states")
    p = s.reshape(-1, 2)
    return np.array([LEVEL_OF_PAIR[(int(a), int(b))] for a, b in p], int)


def states_of_levels(lv, M):
    lv = np.asarray(lv, int)
    if M == 2:
        return (1 - lv).astype(np.uint8)
    return np.array([PAIR_OF_LEVEL[int(i)] for i in lv], np.uint8).reshape(-1)


def state_bits(M):
    """bits[level] = tuple of the states that level carries"""
    return [(1 - i,) for i in range(2)] if M == 2 else [PAIR_OF_LEVEL[i] for i in range(4)]


# ================================================================ likelihood (M4)
_EDGES = np.linspace(-6.0, 6.0, 2 * NTHETA + 1)               # noise kernel window +-6 sigma, 0.09 sigma bins
_KER = np.exp(-0.5 * (0.5 * (_EDGES[:-1] + _EDGES[1:])) ** 2) / np.sqrt(2 * np.pi)


def _arcsine_cdf(u, B):
    """CDF of B cos(theta), theta uniform (arcsine law on [-B, B]); B ~ 0 = point mass at 0"""
    return np.where(B > 1e-6, 0.5 + np.arcsin(np.clip(u / np.maximum(B, 1e-6), -1, 1)) / np.pi, (u >= 0).astype(float))


def scan_loglik(a, B):
    """log p(a | B) for scans a (n,) and level amplitudes B (M,) -> (n, M): density of |B cos(theta) + n| at a,
    = g(a) + g(-a), g = arcsine law convolved with N(0, 1), integrated exactly bin by bin (no phase grid)"""
    a = np.asarray(a, float)[:, None, None]
    Bm = np.asarray(B, float)[None, :, None]
    g = lambda x: (np.diff(_arcsine_cdf(x + _EDGES[None, None, :], Bm), axis=2) * _KER).sum(2)
    f = g(a) + g(-a)
    amax = max(float(np.max(a)) if a.size else 1.0, float(np.max(B)) + 6.0, 10.0)
    return np.log((1 - OUTLIER) * f + OUTLIER / amax + 1e-300)


def wmedian(x, w):
    if len(x) == 0 or w.sum() <= 0:
        return np.nan
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def robust_B(scans, weights):
    """B = sqrt2 x weighted median of the scans (symbol weight shared by its scans)"""
    x = np.concatenate([s for s in scans] or [np.zeros(0)])
    w = np.concatenate([np.full(len(s), wt) for s, wt in zip(scans, weights)] or [np.zeros(0)])
    m = wmedian(x, w)
    return np.nan if np.isnan(m) else np.sqrt(2.0) * m


# ================================================================ levels + posteriors (M5-M6)
def recording_levels(scans, known, ladder_db):
    """B per level from all known (sync) symbols of a recording; ok = at least one non-null level measured"""
    M = len(ladder_db)
    nom = 10 ** (np.asarray(ladder_db) / 20)
    B = np.full(M, np.nan)
    for i in range(M):
        idx = [j for j in np.where(known == i)[0] if len(scans[j])]
        if len(idx) >= MIN_SYNC:
            B[i] = robust_B([scans[j] for j in idx], np.ones(len(idx)))
    meas = [B[i] / nom[i] for i in range(M) if not np.isnan(B[i]) and nom[i] > 0.05]
    if not meas or np.nanmax(B) < NULL_B:
        return B, False
    return np.where(np.isnan(B), float(np.median(meas)) * nom, B), True


def frame_gain(scans, known, B):
    r = [scans[j] / B[known[j]] for j in np.where(known >= 0)[0] if len(scans[j]) and B[known[j]] >= NULL_B]
    if not r:
        return 1.0
    return float(np.sqrt(2.0) * np.median(np.concatenate(r)))


def refine(scans, known, B0):
    """clamped EM (M5) from start levels B0"""
    M = len(B0)
    ns = np.array([len(x) for x in scans], float)
    a2 = np.array([float(np.sum(x ** 2)) for x in scans])
    lo = np.where(B0 >= NULL_B, B0 / 2, 0.0)
    hi = np.where(B0 >= NULL_B, 2 * B0, np.maximum(2 * B0, NULL_B))
    B = B0.copy()
    kn = known >= 0
    for _ in range(EM_ITERS):
        post = posteriors(scans, B)
        post[kn] = np.eye(M)[known[kn]]
        w = post * (ns > 0)[:, None]
        num, den = (w * a2[:, None]).sum(0), (w * ns[:, None]).sum(0)
        Bn = np.sqrt(np.maximum(2.0 * (num / np.maximum(den, 1e-9) - 1.0), 0.0))
        B = np.where(den > 1, np.clip(Bn, lo, hi), B)
    return B


def posteriors(scans, B):
    M = len(B)
    ll = np.zeros((len(scans), M))
    for j, x in enumerate(scans):
        if len(x):
            ll[j] = scan_loglik(x, B).sum(0)
    post = np.exp(ll - ll.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    post[np.array([len(x) == 0 for x in scans])] = 1.0 / M
    return post


def state_soft(post, M):
    """posterior (n_sym, M) -> state soft values s = tanh(LLR/2) in air order (n_sym * bits)"""
    bits = state_bits(M)
    nb = len(bits[0])
    out = np.zeros((len(post), nb))
    for b in range(nb):
        one = np.array([bits[i][b] == 1 for i in range(M)])
        p1, p0 = post[:, one].sum(1), post[:, ~one].sum(1)
        llr = np.clip(np.log(p1 + 1e-300) - np.log(p0 + 1e-300), -LLR_CLIP, LLR_CLIP)
        out[:, b] = np.tanh(llr / 2)
    return out.reshape(-1)


# ================================================================ run (M1-M3, M7)
def run_M(meta):
    return 4 if meta["modulation"] == "fsk4" else 2


def ladder(meta):
    p = meta["plan"]
    if p.get("ladder_db"):
        return tuple(p["ladder_db"])
    return LADDER4_DB if run_M(meta) == 4 else LADDER2_DB


def scan_feature(X, j, lead):
    """M1: a = |x - b| / sigma on node j (decode_v4.absdev baseline and noise)"""
    ad = V4.absdev(X[:, j].astype(float), lead)
    return np.sqrt(np.maximum(10 ** ad - 1, 0.0))


def sync_score(scans, known):
    """M1 node choice: mean a^2 of the strongest known sync level present / that of the weakest (dB)"""
    lv = [i for i in np.unique(known[known >= 0])
          if sum(len(scans[j]) for j in np.where(known == i)[0]) >= 2]
    if len(lv) < 2:
        return -np.inf
    e = lambda i: np.mean(np.concatenate([scans[j] for j in np.where(known == i)[0]]) ** 2)
    return 10 * np.log10((e(min(lv)) + 1e-9) / (e(max(lv)) + 1e-9))


def symbol_scans(t, a, tau, Ts, eps, n):
    k, pos = T.sym_index(t, tau, Ts, eps)
    g = GUARD_MS / 1e3 / (Ts * (1 + eps))
    out = []
    inside = (k >= 0) & (k < n)
    guarded = inside & (pos > g) & (pos < 1 - g)
    order = np.argsort(k, kind="stable")
    ks = k[order]
    for j in range(n):
        lo, hi = np.searchsorted(ks, j), np.searchsorted(ks, j, side="right")
        idx = order[lo:hi]
        gi = idx[guarded[idx]]
        out.append(a[gi] if len(gi) else a[idx])
    return out


def decode_run(d, write=True):
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
    M, lad = run_M(meta), ladder(meta)
    per = 2 if M == 4 else 1
    lv_all = levels_of_states(states, M)
    n = len(lv_all)
    Ts = meta["plan"]["sym_us"] / 1e6
    top, y, lead = T.front(dict(meta, modulation="fsk"), t, X)
    template = 10 ** (np.asarray(lad)[lv_all] / 20)
    tau, eps, r_time = T.timing_keyed(t, y, meta["t_first"], Ts, template)
    node_scans = {j: symbol_scans(t, scan_feature(X, j, lead), tau, Ts, eps, n) for j in top}
    names = R.node_names(meta)
    sc0 = node_scans[top[0]]
    res = dict(decoder=VERSION, rules_sha=RULES_SHA, frozen_at=FROZEN_AT, run=os.path.basename(os.path.normpath(d)),
               modulation=meta["modulation"], M=M, baud=1 / Ts, bits_per_symbol=per, bit_rate=per / Ts,
               ladder_db=list(lad), nodes=[names[j] for j in top], symbols=n,
               scans_per_symbol=float(np.mean([len(x) for x in sc0])), delay_ms=(tau - meta["t_first"]) * 1e3,
               clock_ppm=eps * 1e6, timing_r=r_time, erasures=int(sum(len(x) == 0 for x in sc0)), signal_lost=False)
    known_all = np.full(n, -1)
    for i in range(len(enc)):
        s0, L = int(start[i]), int(length[i])
        ks = T.frame_layout(keys[enc[i]])[0] // per
        if (s0 + L) // per <= n:
            known_all[s0 // per:s0 // per + ks] = lv_all[s0 // per:s0 // per + ks]
    rec = {j: recording_levels(node_scans[j], known_all, lad) for j in top}
    res["recording_levels"] = {names[j]: [float(x) for x in rec[j][0]] for j in top}
    frames, sym_err, sym_n, st_err, st_n = [], 0, 0, 0, 0
    for i in range(len(enc)):
        key = keys[enc[i]]
        s0, L = int(start[i]), int(length[i])
        if s0 % per or L % per:
            raise ValueError(f"frame {i} not aligned to {per}-state symbols")
        a0, nL = s0 // per, L // per
        if a0 + nL > n:
            continue
        nsync = T.frame_layout(key)[0]
        known = np.full(nL, -1)
        ksync = nsync // per
        known[:ksync] = lv_all[a0:a0 + ksync]
        node = max(top, key=lambda j: sync_score(node_scans[j][a0:a0 + nL], known))
        scans = node_scans[node][a0:a0 + nL]
        Brec, ok = rec[node]
        c = frame_gain(scans, known, Brec) if ok else 1.0
        B = np.where(Brec >= NULL_B, c * Brec, Brec)
        for lvl in range(M):
            idx = [j for j in np.where(known == lvl)[0] if len(scans[j])]
            if ok and len(idx) >= MIN_SYNC:
                B[lvl] = robust_B([scans[j] for j in idx], np.ones(len(idx)))
        if ok:
            B = refine(scans, known, B)
        post = posteriors(scans, B) if ok else np.full((nL, M), 1.0 / M)
        s = state_soft(post, M)
        dec_lv = post.argmax(1)
        sym_err += int(np.sum(dec_lv != lv_all[a0:a0 + nL])); sym_n += nL
        st_hat = (s > 0).astype(np.uint8)
        st_err += int(np.sum(st_hat != states[s0:s0 + L])); st_n += L
        dd = T.code_soft(key, s)
        dd = 2 * np.arctanh(np.clip(dd, -1 + 1e-12, 1 - 1e-12))
        pre, n_code, out = T.decode_frame(key, dd, u[i])
        frames.append(dict(i=int(meta["frame_index0"] + i), round=int(meta["round0"] + i // len(keys)), frame=key,
                           t0=float(tau + a0 * Ts * (1 + eps)), erasures=int(sum(len(x) == 0 for x in scans)),
                           node=names[node], reference_ok=bool(ok), gain=c, levels=[float(b) for b in B], pre_err=pre,
                           code_bits=n_code,
                           decoders={k: dict(r, frame_err=r["info_err"] > 0, crc_missed=r["crc_ok"] and r["info_err"] > 0)
                                     for k, r in out.items()}))
    res.update(symbol_errors=sym_err, symbols_decoded=sym_n, state_errors=st_err, state_symbols=st_n, frames=frames)
    if write:
        json.dump(res, open(os.path.join(d, "result_mfsk.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = decode_run(p)
        print(p, f"M={r['M']} r={r['timing_r']:.2f} symbol errors {r['symbol_errors']}/{r['symbols_decoded']}",
              [(f["frame"], f["pre_err"], f["decoders"][f["frame"]]["info_err"]) for f in r["frames"]])
