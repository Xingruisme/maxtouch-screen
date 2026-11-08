#!/usr/bin/env python3
"""
Simulated panel-aware detector: BER vs Es/N0 for a receiver that sees the carrier only through sparse, random-phase
panel scans. Used as the third reference in panel c of stent_modcomp.py conditions --panel-model.

   python panel_model.py [--k 6.25] [--bits 2000000]      # prints the curve and the assumptions

Observation model (assumptions, all of them):
  * Samples per bit: the bit is observed by N scans, N = floor(k) or ceil(k) with P(ceil) = k - floor(k), so the mean
    is k (k = scans per bit, the same k as in Es/N0; 20 bps reference runs: k = 6.25, 125 scans/s). Every scan of the
    bit is used (no guard band: the model has no bit-edge transition and no timing error).
  * Phase: during a sent 1 each scan sees the tone at an independent phase phi ~ Uniform(0, 2 pi), giving a sample
    A sqrt(2) cos(phi) (rms A). Measured: consecutive scans of a node are uncorrelated (r ~ 0) and the sample
    kurtosis is ~1.42, near a sinusoid's 1.5 (stent_modcomp.py notes, 2026-10-23).
  * Sent 0: no carrier. OOK by definition; FSK as this panel sees it, because the '0' tone (180.556 kHz) sits at a
    panel null: it arrives a median 32 dB below the '1' tone in the 20 bps runs (OOK's OFF level: 37 dB below, set
    by drift). So one simulated curve serves both; the textbook FSK curve assumes both tones are received.
  * Noise: additive, white, Gaussian, N(0, sigma^2), independent from scan to scan and of the phase; one node.
  * Es/N0 = k A^2 / (2 sigma^2): the definition of stent_modcomp.py (P1 = A^2 = mean carrier power of sent-1 samples
    above the noise, sigma = OFF noise), and of the noise-added curves (P1 fixed, sigma^2 raised).
  * Decision: per bit, the log-likelihood ratio sum_n log p1(x_n) / p0(x_n) > 0, with p0 = N(0, sigma^2) and p1 the
    exact density of A sqrt(2) cos(phi) + noise (numerical average over phi), A and sigma known, equal priors.
Optimality: given this model (independent scans, known A, sigma, N and bit timing, independent equiprobable bits),
the LLR test is the minimum-error (MAP) decision, so no detector working on the SAME observations does better. It is
NOT a bound for the hardware: the model leaves out the extra nodes (the decoders combine 3 nodes, which see the
same carrier with partly independent noise), any correlation between scans, coding across bits, and possible other
observables (raw signal instead of deltas, more scans per bit at other panel settings). It also leaves out what makes
the real decoders worse (bit-edge scans, timing search, unknown and drifting A and baseline, non-Gaussian noise).
"""
from __future__ import annotations

import argparse

import numpy as np

SEED = 20261102
GRID_DB = np.arange(-2.0, 30.01, 1.0)


def llr_table(a, lim=None, n=4001, nphi=720):
    """log p1(x) / p0(x) on a grid, sigma = 1, rms carrier amplitude a"""
    lim = lim or a * np.sqrt(2) + 8
    x = np.linspace(-lim, lim, n)
    s = a * np.sqrt(2) * np.cos(np.linspace(0, 2 * np.pi, nphi, endpoint=False))
    # log of the phase average of exp(-(x - s)^2 / 2), relative to exp(-x^2 / 2) (common factors cancel)
    e = -0.5 * (x[:, None] - s[None, :]) ** 2 + 0.5 * x[:, None] ** 2
    m = e.max(axis=1, keepdims=True)
    return x, (m[:, 0] + np.log(np.mean(np.exp(e - m), axis=1)))


def ber(es_n0_db, k, bits, rng):
    g = 10 ** (es_n0_db / 10)
    a = np.sqrt(2 * g / k)                      # Es/N0 = k a^2 / 2 with sigma = 1
    x, L = llr_table(a)
    nlo = int(np.floor(k)); p_hi = k - nlo
    nmax = nlo + (p_hi > 0)
    sent = rng.integers(0, 2, bits)
    N = nlo + (rng.random(bits) < p_hi)
    phi = rng.uniform(0, 2 * np.pi, (bits, nmax))
    xs = rng.normal(0, 1, (bits, nmax)) + sent[:, None] * a * np.sqrt(2) * np.cos(phi)
    use = np.arange(nmax)[None, :] < N[:, None]
    llr = np.where(use, np.interp(np.clip(xs, x[0], x[-1]), x, L), 0.0).sum(axis=1)
    err = int(np.sum((llr > 0).astype(int) != sent))
    return err, bits


def curve(k, bits=2_000_000, grid=GRID_DB):
    rng = np.random.default_rng(SEED)
    out = []
    for s in grid:
        e, n = 0, 0
        for _ in range(max(1, bits // 500_000)):   # in chunks (memory)
            de, dn = ber(float(s), k, min(bits, 500_000), rng)
            e += de; n += dn
        out.append(dict(es_n0_db=float(s), errors=e, bits=n, ber=e / n, k=k))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=float, default=6.25)
    p.add_argument("--bits", type=int, default=2_000_000)
    a = p.parse_args()
    for r in curve(a.k, a.bits):
        print(f"Es/N0 {r['es_n0_db']:5.1f} dB  BER {r['ber']:.2e}  ({r['errors']}/{r['bits']})   "
              f"textbook OOK {0.5 * np.exp(-10 ** (r['es_n0_db'] / 10) / 4):.1e}")


if __name__ == "__main__":
    main()
