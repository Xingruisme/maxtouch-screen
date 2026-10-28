"""
Payload-blind decoder for the audit of decode_v4.py (frozen, sha 59cb5e989bc9).

Written from the rules text of decode_v4.py, without importing it or any other project module, so that no code path
can reach the transmitted payload. It knows only the frame STRUCTURE:
    16-bit preamble 1010..10 | 11-bit Barker SFD 11100010010 | 127 payload bits (values unknown) | 16 gap zeros
and reads only what a receiver has: sample timestamps t, raw delta counts X, the bit rate, and which samples are the
transmitter-off lead. It never opens result*.json, decoded*.json, stent_frames.py or meta['t_first'].

Frame acceptance is the header-only half of v4's rule: correlation r of the 27 header decision values with the
known header >= R_MIN (0.3). v4 additionally requires payload BER < 0.35, which needs the payload; that half is gone.

Options (defaults = the frozen v4 rules) exist only for the separate sensitivity and negative-control analyses.
"""
from __future__ import annotations

import sys

import numpy as np

FORBIDDEN = ("stent_frames", "cap_ook_tx", "cap_ook_rx", "decode_v2", "decode_v3", "decode_v4", "maxtouch_stent_rx")

PREAMBLE = [1, 0] * 8
SFD = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
HDR = np.array(PREAMBLE + SFD, float)
N_HDR, N_PAY, N_GAP = len(HDR), 127, 16
NB = N_HDR + N_PAY + N_GAP                       # 170 bits per frame
KNOWN = np.zeros(NB, bool); KNOWN[:N_HDR] = True; KNOWN[N_HDR + N_PAY:] = True
KNOWN_VAL = np.zeros(NB); KNOWN_VAL[:N_HDR] = HDR      # payload positions: unknown (never used, KNOWN is False)
PAY = slice(N_HDR, N_HDR + N_PAY)
R_MIN = 0.3
EPS_PPM = np.arange(-200, 201, 10)


def assert_blind():
    bad = [m for m in FORBIDDEN if m in sys.modules]
    if bad:
        raise RuntimeError(f"blind decoder loaded next to payload-carrying modules {bad}: run it in its own process")


# ------------------------------------------------------------------ features
def block_median_baseline(x, block=50):
    """medians of consecutive `block`-sample chunks (the last chunk takes the remainder), linearly interpolated
    between chunk centres"""
    block = max(4, int(block))
    nb = max(1, len(x) // block)
    centers = (np.arange(nb) + 0.5) * block
    meds = np.array([np.median(x[i * block:(i + 1) * block if i < nb - 1 else len(x)]) for i in range(nb)])
    return np.interp(np.arange(len(x)), centers, meds) if nb > 1 else np.full(len(x), meds[0])


def off_baseline(x, lead):
    """(baseline per sample, lead median, lead sigma): the baseline follows only the quiet samples
    (within 20 sigma of the lead median)"""
    med = float(np.median(x[lead]))
    sig = max(1.4826 * float(np.median(np.abs(x[lead] - med))), 1.0)
    q = np.where(np.abs(x - med) < 20 * sig)[0]
    b = np.full(len(x), med) if len(q) < 10 else np.interp(np.arange(len(x)), q, block_median_baseline(x[q], 50))
    return b, med, sig


def absdev(x, lead):
    b, _, sig = off_baseline(x, lead)
    return np.log10(1 + (x - b) ** 2 / sig ** 2)


def zscore(v):
    med = np.median(v)
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    return (v - med) / (iqr if iqr > 0 else (np.std(v) or 1.0))


def node_ranking(X, lead):
    """energy of successive differences after vs during the lead [dB], per node"""
    e = np.diff(X, axis=0) ** 2
    pre, post = lead[1:], ~lead[1:]
    return 10 * np.log10((e[post].mean(axis=0) + 1e-9) / (e[pre].mean(axis=0) + 1e-9))


# ------------------------------------------------------------------ timing
def bit_index(t, tau, Tb, eps):
    u = (t - tau) / (Tb * (1 + eps))
    m = np.floor(u / NB).astype(int)
    jb = u - m * NB
    j = np.floor(jb).astype(int)
    return m, j, jb - j


def slots(t, tau, eps, P):
    """frame slots whose start lies inside the recording span (as v4's all_frames)"""
    last = int((t[-1] - tau) / (P * (1 + eps)))
    return np.arange(max(0, int(np.ceil((t[0] - tau) / (P * (1 + eps))))), last + 1)


def header_score(f, t, tau, Tb, eps, frames):
    m, j, _ = bit_index(t, tau, Tb, eps)
    sel = (j < N_HDR) & np.isin(m, frames)
    if not sel.any():
        return -np.inf
    idx = np.searchsorted(frames, m[sel]) * N_HDR + j[sel]
    n = len(frames) * N_HDR
    s = np.bincount(idx, f[sel], n); c = np.bincount(idx, minlength=n)
    H = np.where(c > 0, s / np.maximum(c, 1), np.nan).reshape(len(frames), N_HDR)
    ok = ~np.isnan(H).any(axis=1)
    if not ok.any():
        return -np.inf
    Hc = H[ok] - H[ok].mean(axis=1, keepdims=True); tc = HDR - HDR.mean()
    den = np.linalg.norm(Hc, axis=1) * np.linalg.norm(tc)
    return float(np.sum(np.where(den > 0, Hc @ tc / np.where(den > 0, den, 1), 0)))


def search(f, t, Tb, eps_ppm=EPS_PPM):
    """coarse tau over one frame period at eps = 0 (4 ms), then tau +-6 ms (0.25 ms) x eps"""
    P = NB * Tb
    best = (-np.inf, t[0], 0.0)
    for tau in np.arange(t[0], t[0] + P, 0.004):
        s = header_score(f, t, tau, Tb, 0.0, slots(t, tau, 0.0, P))
        if s > best[0]:
            best = (s, tau, 0.0)
    tau_c = best[1]
    for eps in np.asarray(eps_ppm) * 1e-6:
        for tau in np.arange(tau_c - 0.006, tau_c + 0.006 + 1e-9, 0.00025):
            s = header_score(f, t, tau, Tb, eps, slots(t, tau, eps, P))
            if s > best[0]:
                best = (s, tau, eps)
    return best[1], best[2], best[0]


# ------------------------------------------------------------------ bits
def decode_frames(f, t, tau, Tb, eps, frames, guard_ms=2.0, decision="mean_inner", r_min=R_MIN):
    m, j, pos = bit_index(t, tau, Tb, eps)
    g = guard_ms / 1e3 / (Tb * (1 + eps))
    inner = (pos > g) & (pos < 1 - g)
    out = []
    for mm in frames:
        sel = m == mm
        if not sel.any():
            continue
        jj, ff, inn, pp = j[sel], f[sel], inner[sel], pos[sel]
        v = np.full(NB, np.nan); n_all = np.zeros(NB, int); n_in = np.zeros(NB, int)
        for b in range(NB):
            w = jj == b
            if w.any():
                wi = w & inn
                n_all[b], n_in[b] = w.sum(), wi.sum()
                if decision == "nearest_center":
                    v[b] = ff[w][np.argmin(np.abs(pp[w] - 0.5))]
                else:
                    v[b] = ff[wi].mean() if wi.any() else ff[w].mean()
        if np.isnan(v[:N_HDR]).any() or np.isnan(v[PAY]).any():
            continue
        kn = KNOWN & ~np.isnan(v)
        hi, lo = v[kn & (KNOWN_VAL == 1)].mean(), v[kn & (KNOWN_VAL == 0)].mean()
        thr = (hi + lo) / 2
        got = (v[PAY] > thr).astype(int)
        hc = v[:N_HDR] - v[:N_HDR].mean(); tc = HDR - HDR.mean()
        r = float(hc @ tc / (np.linalg.norm(hc) * np.linalg.norm(tc) + 1e-12))
        hdr_bits = (v[:N_HDR] > thr).astype(int)
        out.append(dict(m=int(mm), s0=float(tau + mm * NB * Tb * (1 + eps)), r=r, valid=bool(r >= r_min),
                        thr=float(thr), hi=float(hi), lo=float(lo), got=got.tolist(),
                        norm=((v[PAY] - thr) / max(hi - lo, 1e-9)).tolist(),
                        header_errors=int(np.sum(hdr_bits != HDR)), v=v.tolist(),
                        n_samples=n_all.tolist(), n_inner=n_in.tolist()))
    return out


def decode(t, X, bps, lead, n_nodes=3, nodes=None, combine="max", guard_ms=2.0, decision="mean_inner",
           timing=None, eps_ppm=EPS_PPM, r_min=R_MIN):
    """the whole receiver. lead: boolean mask of transmitter-off samples. nodes: force these node indices.
    timing: force (tau, eps) instead of searching. Returns frames + everything the diagnostic plots need."""
    assert_blind()
    t = np.asarray(t, float); X = np.asarray(X, float)
    Tb = 1.0 / bps
    rank = node_ranking(X, lead)
    top = list(nodes) if nodes is not None else [int(i) for i in np.argsort(-rank)[:n_nodes]]
    z = {i: zscore(absdev(X[:, i], lead)) for i in top}
    f = np.max([z[i] for i in top], axis=0) if combine == "max" else np.mean([z[i] for i in top], axis=0)
    if timing is None:
        tau, eps, score = search(f, t, Tb, eps_ppm)
    else:
        tau, eps = timing
        score = header_score(f, t, tau, Tb, eps, slots(t, tau, eps, NB * Tb))
    frames = decode_frames(f, t, tau, Tb, eps, slots(t, tau, eps, NB * Tb), guard_ms, decision, r_min)
    return dict(nodes=top, node_rank_db=rank.tolist(), tau=float(tau), eps=float(eps), header_score=float(score),
                Tb=Tb, frames=frames, f=f, z=z)
