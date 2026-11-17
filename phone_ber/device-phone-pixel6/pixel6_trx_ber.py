#!/usr/bin/env python3
"""
Through-body BER to the Pixel 6 with the stent link's own frames and error correction (trx_psk_node), 2026-11-15.
Same tool as the OnePlus Pad BER (screen-bloodflow-core-main/pad_trx_ber.py: trx_psk_link_v2 frames none_full + cc4,
before / after FEC, IEC current check, summary tables / figures) with the Pixel 6 as the receiver:

  reader   pixel6_capacitance.MatrixStream (stm_fts_cmd 13 raw map, 16 x 34, ~2.4 reads/s measured) minus one
           finger-off baseline taken at the start of the session (as pixel6_body_sweep.py)
  carrier  54.76 kHz (AD9833 54763.72 Hz, the sweep's exact tone): the whole panel steps +4 -> -4 counts within one
           read of the carrier edge and back within one read (sweep hand_1k_500k_20261114_202504; control flat).
           FSK state 0 = 445.79 kHz (no Pixel 6 response in the sweep).
  decoder  decode_trx_level.py (panel-level feature, rules fixed before any Pixel 6 BER recording) + decode_trx_pad
           rules 2-8 (which call the frozen decode_trx_tablet)
  rates    default 1 baud (~2.4 reads per symbol, like the Pad's 3 baud that decoded cleanly after FEC)

   python pixel6_trx_ber.py synth
   python pixel6_trx_ber.py plan --rounds 1 --current-ma 1
   python pixel6_trx_ber.py run --label hand --site hand2finger --current-ma 1 --rounds 1        (the user runs it)
   python pixel6_trx_ber.py analyse data/trx_ber/<run folders or trxber_ summary folder>
Output data/trx_ber/px6_trx_<label>_<mod>_<baud>baud_c<k>_<time>/ and data/trx_ber/trxber_<label>_<time>/.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pixel6_body_sweep as SWP                         # noqa: E402  (PHONE spec; adds screen-bloodflow-core-main)
sys.path.append(str(HERE.parent / "screen-bloodflow-core-main"))
import pad_trx_ber as P                                 # noqa: E402
import decode_trx_level as DL                           # noqa: E402

B, TV2 = P.B, P.TV2
READ_S = 0.416                                          # sweep: read interval p5 / p50 / p95 = 327 / 416 / 443 ms


def synth_reads(phase, Ts, rng, lat=(0.10, 0.30), p_glitch=0.03, node_sd=5.0):
    """Pixel-6-like reads from the sweep at 54.76 kHz: a read every ~416 ms (p5-p95 327-443); panel level +4 counts
    with the carrier off, -4 while it is on (from lat after each edge); node noise ~5 counts per read; a finger streak
    along column 13; p_glitch = share of reads whose panel level jumps to the other state by itself (sweep: a few
    single '-4' reads in OFF blocks)."""
    lead, n = P.LEAD_S, len(phase)
    t, x = [], 0.0
    while x < lead + n * Ts + P.TAIL_S:
        x += float(np.clip(rng.normal(0.416, 0.03), 0.32, 0.46))
        t.append(x)
    t = np.array(t)
    j = np.floor((t - lead - rng.uniform(*lat, len(t))) / Ts).astype(int)
    on = np.where((j >= 0) & (j < n), phase[np.clip(j, 0, n - 1)], 0).astype(bool)
    on ^= rng.random(len(t)) < p_glitch
    lvl = np.where(on, -4.0, 4.0)
    F = rng.normal(0, node_sd, (len(t), 16, 34)) + lvl[:, None, None]
    F[:, 2:6, 12:15] -= 1300.0
    F[:, :, 12:15] -= 40.0
    return t, np.round(F).astype(np.int16)


def synth(a):
    rng = np.random.default_rng(11)
    keys = B.send_list(a)
    cases = [("sweep-like (3 % glitch reads)", dict()), ("noisy (10 % glitch reads, slower 0.1-0.5 s response)",
                                                         dict(p_glitch=0.10, lat=(0.1, 0.5)))]
    print(f"simulated Pixel 6 reads at 54.76 kHz, frames {', '.join(keys)}, {a.rounds} rounds per rate")
    for name, kw in cases:
        print(f"\n{name}")
        for rate in [float(r) for r in a.synth_rates.split(",")]:
            seq = TV2.build_seq(keys, a.rounds, a.seed, int(round(rate)))
            sq = dict(keys=np.array(keys), enc=seq["enc"], start=seq["start"], length=seq["length"],
                      u=np.packbits(seq["u"], 1), phase=np.packbits(seq["phase"]), n_phase=len(seq["phase"]))
            Ts = B.sym_us(rate) / 1e6
            t, Fm = synth_reads(seq["phase"], Ts, rng, **kw)
            meta = dict(modulation="ook", plan=dict(sym_us=B.sym_us(rate), carrier_hz=a.carrier), t_first=P.LEAD_S,
                        frame_index0=0, round0=0)
            y, _, _ = DL.feature(Fm, t, P.LEAD_S)
            res = DL.DP.decode_arrays(t, y, meta, sq)
            agg = {}
            for fr in res["frames"]:
                g = agg.setdefault(fr["frame"], [0, 0, 0, 0, 0])
                dd = fr["decoders"][fr["frame"]]
                g[0] += fr["pre_err"]; g[1] += fr["code_bits"]; g[2] += dd["info_err"]; g[3] += TV2.N_INFO; g[4] += dd["frame_err"]
            txt = "   ".join(f"{k}: before {g[0]}/{g[1]}, after {g[2]}/{g[3]}, frames bad {g[4]}/{a.rounds}"
                           for k, g in agg.items())
            print(f"  {1e6 / B.sym_us(rate):5.2f} baud ({res['reads_per_symbol']:.1f} reads/sym, delay "
                  f"{res['delay_ms']:.0f} ms, r {res['timing_r']:.2f}): state {res['state_errors']}/{res['state_symbols']}   {txt}")


DEV = P.Device(name="Pixel 6", short="Pixel 6", phone=SWP.PHONE, out=HERE / "data" / "trx_ber", prefix="px6_trx",
               decoder=DL, read_s=READ_S, rates="1", carrier=54763.72, fsk_zero=445.79e3, synth_rates="0.5,0.75,1,1.5,2",
               synth=synth, rates_help="the Pixel 6 reads ~2.4 maps/s",
               carrier_help="54.76 kHz: panel steps +4 -> -4 counts within one read",
               fsk_help="445.79 kHz: no Pixel 6 response in the sweep", mods_ok=("ook", "fsk"))

if __name__ == "__main__":
    sys.exit(P.main(DEV))
