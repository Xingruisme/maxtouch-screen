#!/usr/bin/env python3
"""
Amplitude-discriminated 2-FSK decoder (experimental, 2026-10-27), for frames whose bit-0 tone the panel does see
(bit 1 = 159.585 kHz, bit 0 = 53.513 kHz: about 10 dB weaker through the body). decode_v4 is not used or changed.

The panel gives one delta per node per scan and cannot tell frequencies apart; each scan sees a tone of rms A as
A sqrt2 cos(phi) with a fresh random phase (125 scans/s: consecutive scans r ~ 0, kurtosis 1.42 ~ a sinusoid's 1.5).
So both symbols are 'carrier on', and they differ only in the spread of |delta|. This decoder learns that spread for
each symbol from the known pilot bits of every frame (preamble + SFD) and decides each payload bit by the summed
log-likelihood ratio of its scans. It also decodes frames whose bit 0 is (nearly) off, e.g. the 180.556 kHz pair.

Rules (RULES below) were fixed before any recording with a visible bit-0 tone existed. They were developed on synthetic
captures and on existing 180.556 kHz FSK runs (exploratory data); recordings made with the 53.513 kHz pair are
validation data and must not be used to change them. RULES_SHA (this file) goes into every result.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

import stent_frames as SF

RULES = dict(
    nodes="top 3 by scan-to-scan energy after the START prompt over before it",
    baseline="per node, running median of all scans over 0.5 s (tone deltas are symmetric about the baseline)",
    feature="v = sqrt(mean over the 3 nodes of ((x - baseline) / node rms after the prompt)^2), one value per scan",
    guard_ms=2.0,
    scans_per_bit="guarded scans (2 ms from each bit edge); none -> every scan inside the bit; none -> erasure (error)",
    pilot="16 preamble + 11 SFD bits of each frame (13 ones, 14 zeros)",
    calibration="u = v / frame scale (median v over the frame's pilot-1 scans); p1, p0 = Gaussian-kernel densities "
                "of u over all pilot-1 / pilot-0 scans of the run, bandwidth per symbol by Silverman's rule "
                "(0.9 min(sd, IQR/1.34) n^-1/5, at least 0.005), floor 1e-3",
    decision="bit 1 if the sum over its scans of log p1(u) / p0(u) > 0",
    known_timing="t_first + delay 0-25 ms (0.25 ms) x clock -200..+200 ppm (50 ppm), maximising the mean per-frame "
                 "pilot correlation (Pearson r of per-bit mean v^2 vs the pilot bits)",
    blind="pilot correlation at every Tb/4 from the recording start (no t_first), greedy non-overlapping peaks with "
          "r >= 0.6, each refined over +-Tb/4 in Tb/16; a frame counts as found when its refined r >= 0.6 and its "
          "scale (median v of its pilot-1 scans) >= 5 x the median v of the transmitter-off lead; with >= 3 frames "
          "found, the frame most others agree with (within one bit of a whole number of frame periods) anchors a "
          "least-squares line start = a + b x frame number through the agreeing frames; disagreeing ones are dropped",
)
RULES_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]

N_NODES, BASE_S, GUARD_S, BW_MIN, FLOOR, R_MIN, LEAD_X = 3, 0.5, 2e-3, 0.005, 1e-3, 0.6, 5.0
FRAME, TRUTH = (np.array(b) for b in SF.frame_bits())
NP = len(SF.PREAMBLE) + len(SF.SFD)
PILOT = FRAME[:NP]


# ---------------------------------------------------------------- per-scan feature
def nodes(X, t, t_prompt):
    e = np.diff(X, axis=0) ** 2
    pre, post = t[1:] < t_prompt, t[1:] >= t_prompt
    score = (e[post].mean(axis=0) + 1e-9) / (e[pre].mean(axis=0) + 1e-9)
    return [int(i) for i in np.argsort(-score)[:N_NODES]]


def running_median(x, t, span):
    """median of x over [t - span/2, t + span/2] at every sample (a centred window of fixed duration)"""
    n = max(3, int(round(span / np.median(np.diff(t)))) | 1)
    if len(x) <= n:
        return np.full(len(x), np.median(x))
    from scipy.ndimage import median_filter
    return median_filter(x, size=n, mode="nearest")


def feature(X, t, t_prompt, cols):
    after = t >= t_prompt
    d = []
    for i in cols:
        x = X[:, i].astype(float)
        dev = x - running_median(x, t, BASE_S)
        d.append(dev / max(float(np.sqrt(np.mean(dev[after] ** 2))), 1e-9))
    return np.sqrt(np.mean(np.square(d), axis=0))


# ---------------------------------------------------------------- bit windows
def windows(t, starts, bits, Tb, guard):
    """scan index ranges [lo, hi) of bit j of the frame starting at each start (shape starts x bits)"""
    a = starts[:, None] + bits[None, :] * Tb
    return np.searchsorted(t, a + guard), np.searchsorted(t, a + Tb - guard)


def bit_ranges(t, starts, bits, Tb):
    """guarded range where it holds a scan, else the whole bit"""
    lo, hi = windows(t, starts, bits, Tb, GUARD_S)
    lo2, hi2 = windows(t, starts, bits, Tb, 0.0)
    empty = hi <= lo
    return np.where(empty, lo2, lo), np.where(empty, hi2, hi)


def range_sum(c, lo, hi):
    """sum of the per-scan values (cumulative c, c[0] = 0) over [lo, hi)"""
    return c[hi] - c[lo]


def pilot_r(e_cum, n_cum, t, starts, Tb):
    """Pearson r of the per-bit mean energy over the pilot bits vs the pilot pattern, per start"""
    lo, hi = bit_ranges(t, starts, np.arange(NP), Tb)
    n = range_sum(n_cum, lo, hi)
    m = np.where(n > 0, range_sum(e_cum, lo, hi) / np.maximum(n, 1), np.nan)
    ok = ~np.isnan(m).any(axis=1)
    p = PILOT - PILOT.mean()
    mc = m - np.nanmean(m, axis=1, keepdims=True)
    r = (mc @ p) / (np.sqrt(np.nansum(mc ** 2, axis=1)) * np.sqrt(np.sum(p ** 2)) + 1e-12)
    return np.where(ok, r, -1.0)


def cums(v):
    return np.concatenate([[0.0], np.cumsum(v ** 2)]), np.arange(len(v) + 1, dtype=float)


# ---------------------------------------------------------------- calibration and decision
def kde(u, grid):
    if len(u) < 5:
        return np.full(len(grid), FLOOR)
    iqr = np.subtract(*np.percentile(u, [75, 25]))
    h = max(0.9 * min(np.std(u), iqr / 1.34 if iqr > 0 else np.std(u)) * len(u) ** -0.2, BW_MIN)
    z = (grid[:, None] - u[None, :]) / h
    p = np.exp(-0.5 * z ** 2).sum(axis=1) / (len(u) * h * np.sqrt(2 * np.pi))
    return np.maximum(p, FLOOR)


def decode_frames(v, t, starts, Tb):
    """per frame: scale from its pilot-1 scans, then run-wide pilot densities, then LLR decisions of every bit"""
    nb = len(FRAME)
    lo, hi = bit_ranges(t, starts, np.arange(nb), Tb)
    scale, u = np.full(len(starts), np.nan), np.full(len(v), np.nan)
    for f in range(len(starts)):
        p1 = np.concatenate([np.arange(lo[f, j], hi[f, j]) for j in range(NP) if PILOT[j] == 1] or [np.array([], int)])
        if len(p1):
            scale[f] = np.median(v[p1])
        if np.isfinite(scale[f]) and scale[f] > 0:
            a, b = lo[f, 0], hi[f, nb - 1]
            u[a:b] = v[a:b] / scale[f]
    pil = {s: np.concatenate([np.arange(lo[f, j], hi[f, j]) for f in range(len(starts)) if np.isfinite(scale[f])
                              for j in range(NP) if PILOT[j] == s] or [np.array([], int)]) for s in (0, 1)}
    grid = np.linspace(0, 4, 801)
    dens = {s: kde(u[pil[s]], grid) for s in (0, 1)}
    llr_grid = np.log(dens[1]) - np.log(dens[0])
    llr = np.interp(np.nan_to_num(u, nan=0.0), grid, llr_grid)
    c = np.concatenate([[0.0], np.cumsum(llr)])
    L = range_sum(c, lo, hi)
    got = (L > 0).astype(int)
    erased = hi <= lo
    cal = dict(pilot_scans1=len(pil[1]), pilot_scans0=len(pil[0]),
               u0_median=float(np.median(u[pil[0]])) if len(pil[0]) else None,
               u1_median=float(np.median(u[pil[1]])) if len(pil[1]) else None)
    return got, erased, scale, cal


def score(got, erased, scale, frames):
    """payload errors of the given frames (rows of got); an erased bit counts as an error"""
    pl = slice(NP, NP + len(TRUTH))
    e0 = e1 = er = 0
    for f in frames:
        if not np.isfinite(scale[f]):
            continue
        g, z = got[f, pl], erased[f, pl]
        er += int(z.sum())
        e0 += int(((g == 1) & (TRUTH == 0) & ~z).sum())
        e1 += int(((g == 0) & (TRUTH == 1) & ~z).sum())
    n = sum(len(TRUTH) for f in frames if np.isfinite(scale[f]))
    return dict(errors=e0 + e1 + er, bits=n, sent0_read1=e0, sent1_read0=e1, erasures=er,
                ber=(e0 + e1 + er) / n if n else None)


# ---------------------------------------------------------------- known timing and blind
def known(v, t, t_first, Tb, n_frames):
    e_cum, n_cum = cums(v)
    nb = len(FRAME)
    best = (-2.0, 0.0, 0.0)
    for eps in np.arange(-200e-6, 200e-6 + 1e-12, 50e-6):
        tb = Tb * (1 + eps)
        for dly in np.arange(0.0, 0.025 + 1e-12, 0.00025):
            starts = t_first + dly + np.arange(n_frames) * nb * tb
            starts = starts[starts + nb * tb <= t[-1]]
            if not len(starts):
                continue
            r = float(np.mean(pilot_r(e_cum, n_cum, t, starts, tb)))
            if r > best[0]:
                best = (r, dly, eps)
    r, dly, eps = best
    tb = Tb * (1 + eps)
    starts = t_first + dly + np.arange(n_frames) * nb * tb
    starts = starts[starts + nb * tb <= t[-1]]
    got, erased, scale, cal = decode_frames(v, t, starts, tb)
    res = score(got, erased, scale, range(len(starts)))
    res.update(frames_scored=int(np.isfinite(scale).sum()), frames_sent=n_frames, delay_ms=dly * 1e3,
               clock_ppm=eps * 1e6, pilot_r=r, **cal)
    return res, (starts, tb)


def blind(v, t, Tb, n_frames, noise):
    e_cum, n_cum = cums(v)
    nb = len(FRAME)
    cand = np.arange(t[0], t[-1] - nb * Tb, Tb / 4)
    r = pilot_r(e_cum, n_cum, t, cand, Tb)
    taken = []
    for i in np.argsort(-r):
        if r[i] < R_MIN:
            break
        if all(abs(cand[i] - s) > 0.9 * nb * Tb for s in taken):
            taken.append(cand[i])
    found = []
    for s in sorted(taken):
        fine = s + np.arange(-4, 5) * Tb / 16
        rf = pilot_r(e_cum, n_cum, t, fine, Tb)
        if rf.max() >= R_MIN:
            found.append(float(fine[int(np.argmax(rf))]))
    starts = np.array(found)
    if len(starts):                             # a noise stretch can reach r >= 0.6 over thousands of candidates
        lo, hi = bit_ranges(t, starts, np.arange(NP), Tb)
        lvl = np.array([np.median(np.concatenate([v[lo[f, j]:hi[f, j]] for j in range(NP) if PILOT[j] == 1]))
                        for f in range(len(starts))])
        starts = starts[lvl >= LEAD_X * noise]
    if len(starts) >= 3:                        # one period and phase for the run, from the found frames only
        P = len(FRAME) * Tb
        best = None
        for s0 in starts:                       # anchor = the found frame most others agree with (within one bit)
            m = np.round((starts - s0) / P)
            inl = np.abs(starts - s0 - m * P) < Tb
            if best is None or inl.sum() > best[2].sum():
                best = (s0, m, inl)
        _, m, inl = best
        b, a = np.polyfit(m[inl], starts[inl], 1)
        starts = a + b * m[inl]
        Tb = b / len(FRAME)
    if not len(starts):
        return dict(errors=0, bits=0, ber=None, frames_valid=0, frames_sent=n_frames, sent0_read1=0, sent1_read0=0,
                    erasures=0)
    got, erased, scale, cal = decode_frames(v, t, starts, Tb)
    res = score(got, erased, scale, range(len(starts)))
    res.update(frames_valid=int(np.isfinite(scale).sum()), frames_sent=n_frames, **cal)
    return res


def decode(X, t, meta, extra_noise=None):
    """known-timing and blind results for one recording (X: scans x nodes). extra_noise(X, cols) -> X adds noise to
    the selected nodes before the feature (noise-added curves)."""
    plan = meta["plan"]
    Tb = plan["bit_us"] / 1e6
    cols = nodes(X.astype(float), t, meta["t_prompt"])
    Xf = X.astype(float) if extra_noise is None else extra_noise(X, cols)
    v = feature(Xf, t, meta["t_prompt"], cols)
    kt, timing = known(v, t, meta["t_first"], Tb, meta["tx_frames"])
    bl = blind(v, t, Tb, meta["tx_frames"], float(np.median(v[t < meta["t_prompt"]])))
    return dict(nodes=cols, known=kt, blind=bl, rules_sha=RULES_SHA), timing
