#!/usr/bin/env python3
"""
i2o frame (implant -> outside screen), 2026-11-24: a shorter frame than the i2i one of trx_psk_node, for the PER tests
on the tablet. 128 symbols instead of 256.

  i2i (trx 'cc4'):  preamble 32 | SFD 32 || header 20 (src 4, dst 4, type 4, power 4, frame no. 4) | payload 60 | CRC 16
                    64 uncoded + 96 info bits x 2 = 256 symbols
  i2o ('i2o'):      sync 32 || header 12 (src 4, type 4, frame no. 4) | payload 20 | CRC 16
                    32 uncoded + 48 info bits x 2 = 128 symbols

What changed and why
  - SFD dropped as a field. Its two jobs move into the 32-symbol sync word: marking the frame start, and (for ASK)
    showing the receiver every transmit level. The i2i preamble is 32 alternating symbols on air, which can do neither
    (it is periodic, and in 4-ASK it is one level only), so it is replaced, not kept.
  - Sync word SYNC_WORD = 0xEED09709, given as AIR STATES (1 = high), first state first. Read as 16 pairs it holds
    each 4-ASK level exactly 4 times (the level decoder's minimum for a per-frame level estimate, decode_trx_mfsk
    MIN_SYNC); read as 32 two-level symbols it is balanced 16 / 16. Aperiodic autocorrelation peak sidelobe 5 of 32
    (two-level) and 2.75 of 20 (four-level) - best of 30 million random arrangements (scratchpad sync_search.py).
    It starts on the full level.
  - Dst dropped (one receiver: the screen). Power dropped (no power-control loop towards the screen).
  - Payload 60 -> 20 bits. CRC-16 unchanged (same polynomial, stent_fec.with_crc / crc_ok, over header + payload).
  - Coding unchanged: rate-1/2 convolutional code K = 4 (17,13), zero-terminated and punctured to exactly 2 x 48 = 96
    symbols, body state = code bit, soft Viterbi (stent_fec.Fec(kind='cc4', n_info=48, interleave=False)).

No-preamble variant 'i2o_np' (2026-11-24, asked for the PER tests): the same frame WITHOUT the 32 sync symbols -
  i2o_np:           header 12 (src 4, type 4, frame no. 4) | payload 20 | CRC 16
                    48 info bits x 2 = 96 symbols, all coded, nothing known to the receiver
The receiver then has no known symbol: no level / threshold reference and no start mark in the frame. The frozen
decoders cannot decode it (both take their reference from the sync); decode_trx_nopre.py does, with the levels
estimated blind from the data symbols and the frame start taken from the known schedule (as the PER tests always did).

Nothing in trx_psk_node or in the frozen decoders is edited. register() adds the frame to trx_psk_link_v2's tables in
this process and wraps two lookups (make_fec, decode_trx_tablet.frame_layout) so that the frozen decoders see the
i2o layout; use() sets the info length (trx_psk_link_v2.N_INFO) to 48 for a session that sends i2o frames. A session
is one frame type - i2o and i2i frames are not mixed in one recording or one analysis.
"""
from __future__ import annotations

import numpy as np

import decode_trx_tablet as T                   # frozen; only called / wrapped

TV2, F = T.TV2, T.F
KEY = "i2o"
KEY_NP = "i2o_np"                               # the same frame without the sync word
KEYS = (KEY, KEY_NP)
SYNC_WORD = 0xEED09709
SYNC = np.array([(SYNC_WORD >> (31 - i)) & 1 for i in range(32)], np.uint8)     # air states, first state first
FIELDS = (("src", 4), ("type", 4), ("frame_no", 4), ("payload", 20), ("crc", 16))
N_INFO = sum(n for _, n in FIELDS)              # 48
N_CODED = 2 * N_INFO                            # 96
N_AIR = len(SYNC) + N_CODED                     # 128
N_INFO_I2I = TV2.N_INFO                         # 96
LABEL = "i2o short frame (sync 32 + 48 info bits, conv K=4)"
LABEL_NP = "i2o frame without preamble (48 info bits, conv K=4, 96 symbols)"
NO_SYNC = np.zeros(0, np.uint8)


class I2OFec(F.Fec):
    nsync = len(SYNC)

    def __init__(self, quant=None):
        super().__init__(kind="cc4", n_info=N_INFO, interleave=False, quant=quant)

    def frame(self, u):
        """info (B, 48) -> the bits whose running XOR is the air state row [SYNC | code bits] (what build_seq expects)"""
        c = self.encode(u)
        s = np.concatenate([np.repeat(SYNC[None], c.shape[0], 0), c], 1)
        return s ^ np.concatenate([np.zeros((len(s), 1), np.uint8), s[:, :-1]], 1)


class I2ONoPreFec(I2OFec):
    nsync = 0

    def frame(self, u):
        """info (B, 48) -> the bits whose running XOR is the air state row [code bits] (no sync)"""
        s = self.encode(u)
        return s ^ np.concatenate([np.zeros((len(s), 1), np.uint8), s[:, :-1]], 1)


_make_fec, _frame_layout = TV2.make_fec, T.frame_layout


def _mk(key, quant=None):
    return I2OFec(quant) if key == KEY else I2ONoPreFec(quant) if key == KEY_NP else _make_fec(key, quant)


def _layout(key):
    if key == KEY_NP:
        return 0, False, "bpsk", NO_SYNC.copy()
    return (len(SYNC), False, "bpsk", SYNC.copy()) if key == KEY else _frame_layout(key)


def register():
    TV2.ENCODINGS[KEY] = (dict(kind="cc4", n_info=N_INFO, interleave=False), False, LABEL)
    TV2.DECODERS[KEY] = (KEY, None, LABEL)
    TV2.ENCODINGS[KEY_NP] = (dict(kind="cc4", n_info=N_INFO, interleave=False), False, LABEL_NP)
    TV2.DECODERS[KEY_NP] = (KEY_NP, None, LABEL_NP)
    TV2.make_fec = _mk
    T.frame_layout = _layout


def use(on=True):
    """info length of the session: 48 (i2o) or back to 96 (i2i)"""
    TV2.N_INFO = N_INFO if on else N_INFO_I2I


def pack(src, ftype, frame_no, payload):
    """field values -> 48 info bits (MSB first per field, CRC-16 appended)"""
    bits = []
    for v, (name, n) in zip((src, ftype, frame_no, payload), FIELDS[:-1]):
        if not 0 <= v < 1 << n:
            raise ValueError(f"{name} = {v} does not fit {n} bits")
        bits += [(v >> (n - 1 - i)) & 1 for i in range(n)]
    return F.with_crc(np.array(bits, np.uint8))[0]


def unpack(bits):
    bits, out, i = np.asarray(bits, np.uint8), {}, 0
    for name, n in FIELDS[:-1]:
        out[name] = int("".join(map(str, bits[i:i + n])), 2)
        i += n
    out["crc_ok"] = bool(F.crc_ok(bits)[0])
    return out


def describe(key=KEY):
    body = " | ".join(f"{n} {b}" for n, b in FIELDS) + f" = {N_INFO} info bits -> {N_CODED} coded"
    if key == KEY_NP:
        return f"i2o frame without preamble: {body}; {N_CODED} symbols (no sync, nothing uncoded)"
    return f"i2o frame: sync {len(SYNC)} (0x{SYNC_WORD:08X}, uncoded) | {body}; {N_AIR} symbols"


if __name__ == "__main__":
    register(); use()
    print(describe())
    rng = np.random.default_rng(1)
    u = pack(3, 1, 7, 0xABCDE)
    print("pack / unpack:", unpack(u))
    f = I2OFec()
    st = np.cumsum(f.frame(u), 1)[0] % 2
    assert np.array_equal(st[:32], SYNC) and len(st) == N_AIR
    d = T.code_soft(KEY, 2.0 * st - 1)          # state soft values (positive = state 1) -> code-bit soft (positive = 0)
    assert np.array_equal(d, 1.0 - 2.0 * st[32:])
    print("noiseless decode:", T.decode_frame(KEY, d, u))
    print("noisy decode    :", T.decode_frame(KEY, d + rng.normal(0, 0.8, N_CODED), u))
    print(describe(KEY_NP))
    st = np.cumsum(I2ONoPreFec().frame(u), 1)[0] % 2
    assert len(st) == N_CODED == TV2.air_len(KEY_NP) and np.array_equal(st, f.encode(u)[0])
    d = T.code_soft(KEY_NP, 2.0 * st - 1)
    print("no preamble, noiseless decode:", T.decode_frame(KEY_NP, d, u))
