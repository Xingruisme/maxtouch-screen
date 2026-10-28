"""Frame and probe schedule of the TeensyTRX stent transmitter (transceiver/cap_ook_tx.py), for receive-only use.

The transmitter code is not on this laptop, so this reproduces just what a receiver needs, recovered from the
zero-error CapTIvate recordings in ../captivate/data (stentber_okfsk_ook_* and stentber_adv_ook_*, 2026-10-15):

  frame = PREAMBLE (16 bits 1010...10) + SFD (Barker-11 11100010010) + payload + GAP_BITS zeros
  payload = 127-bit PRBS7 (x^7 + x^6 + 1, seed all ones), starting 7 bits into the sequence
  probe   = marker (marker_s on at 1012.5 Hz, marker_s off), then per carrier `reps` x (block_s on, block_s off)

It exposes the same names cap_ook_rx.py imports from cap_ook_tx (PREAMBLE, SFD, GAP_BITS, build, frame_bits,
probe_plan, PROBE_CARRIERS), so ../captivate/cap_ook_rx.py's analysis runs unchanged on maXTouch samples.
"""

from __future__ import annotations

import numpy as np

PREAMBLE = [1, 0] * 8
SFD = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]
GAP_BITS = 16
PRBS_SHIFT = 120                        # payload[i] = prbs7[(i - 120) % 127], i.e. it starts at sequence index 7
PROBE_CARRIERS = [100, 300, 1012.5, 2012.5, 4012.5, 8012.5, 20e3, 50e3, 163e3, 500e3, 1e6, 2e6, 3e6, 4e6]
MARKER_HZ, MARKER_S = 1012.5, 3.0


def prbs7():
    s, out = [1] * 7, []
    for _ in range(127):
        out.append(s[-1])
        s = [s[5] ^ s[6]] + s[:-1]
    return out


def frame_bits(payload=127, prbs=7):
    """(all frame bits, payload bits) like cap_ook_tx.frame_bits"""
    if prbs != 7 or payload != 127:
        raise ValueError("only the recorded default (127-bit PRBS7 payload) is known; check the transmitter's --payload/--prbs")
    pl = np.roll(prbs7(), PRBS_SHIFT).tolist()
    return PREAMBLE + SFD + pl + [0] * GAP_BITS, pl


def build(link):
    """(None, plan) with the plan fields the receiver uses (bps, frame_bits, bit_us, carrier_hz, zero_hz, frame_s)"""
    bits, _ = frame_bits(link.payload, link.prbs)
    bit_us = int(round(1e6 / link.bps))
    return None, dict(kind="cap_ook", carrier_hz=getattr(link, "carrier", None), zero_hz=getattr(link, "zero_hz", None),
                      wave=getattr(link, "wave", "square"), bps=link.bps, bit_us=bit_us, payload=link.payload,
                      prbs=link.prbs, frame_bits=len(bits), header_bits=len(PREAMBLE) + len(SFD),
                      frame_s=len(bits) * bit_us / 1e6)


def probe_plan(link):
    """(None, plan) of the transmitter's --probe schedule"""
    car = list(link.probe_carriers)
    total = 2 * MARKER_S + len(car) * 2 * link.probe_reps * link.probe_block
    return None, dict(kind="cap_ook_probe", carriers_hz=car, marker_hz=MARKER_HZ, marker_s=MARKER_S,
                      block_s=link.probe_block, reps=link.probe_reps, per_carrier_s=2 * link.probe_reps * link.probe_block,
                      total_s=total)
