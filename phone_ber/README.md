# Through-body BER on phones and the OnePlus Pad (trx frames + FEC)

Copies of the phone / tablet BER tools and their data, taken 2026-11-17 from the working folders
`screen-bloodflow-core-main/` and `device-phone-pixel6/` (siblings of this repo's checkout `maxtouch-screen-main/`).
The scripts expect that sibling layout (`decode_trx_pad.py` loads `../maxtouch-screen-main/decode_trx_tablet.py`,
`pad_trx_ber.py` loads `stent_trx_ber.py` from there), so run them from the original folders, not from here.

Same frames and FEC accounting as the maXTouch tablet runs (`stent_trx_ber.py`, frames from
`trx_psk_node/transceiver/trx_psk_link_v2.build_seq`):

- `none_full`: sync 32 (preamble 16 + SFD 0x0A6E) + 96 info bits (80 random + CRC-16), 1 symbol per bit
- `cc4`: sync 64 (preamble 32 + SFD32 0x06669758) + 96 info bits coded K=4 rate 1/2 to 192 symbols
- before FEC = cc4 code-bit errors, after FEC = info-bit errors of the 96 (soft Viterbi), CRC-16 checked

| Device | Tool | Decoder | Notes |
|---|---|---|---|
| OnePlus Pad | `screen-bloodflow-core-main/pad_trx_ber.py` | `decode_trx_pad.py` (trx-pad-1) | feature = share of map nodes changed since previous read; OOK 37.4 kHz, FSK 37.4 / 445 kHz |
| Pixel 6 | `device-phone-pixel6/pixel6_trx_ber.py` (Device spec on `pad_trx_ber.py`) | `decode_trx_level.py` (trx-level-1) | panel-median level; OOK 54.76 kHz, FSK 54.76 / 445.79 kHz; no BER session recorded yet |

Both decoders call the frozen `decode_trx_tablet.py` functions unchanged; their rules were fixed before any recording.

Data:

- `screen-bloodflow-core-main/data_oneplus/trx_ber/`: Pad BER sessions 2026-11-14, among them
  `trxber_ankle_full_1mA_20261114_232032` (ankle to finger, 1 mA, 2/3/4 baud x 3 rounds, figures + scripts)
- `screen-bloodflow-core-main/data_oneplus/body_sweep/`, `device-phone-pixel6/data/body_sweep/`: 1-500 kHz body sweeps
- `screen-bloodflow-core-main/data_body_sweep_all/`: all-device response-vs-frequency figure (`make_figure.py`)
