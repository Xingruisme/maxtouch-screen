#!/usr/bin/env python3
"""
Touch-triggered I2O proof of concept (2026-11-19): finger touches the maXTouch screen -> the stent receiver (TeensyTRX
A0) detects the 70.6 kHz drive (touch_detector.Detector, frozen 87fc749001d6, unchanged) -> ONE trx BER frame is
sent with the existing tablet BER path (stent_trx_ber.one_run, unchanged: same schedule, framing, FEC, receiver set-up,
recording, decoder) -> the tablet decodes it -> the terminal reports the result.

   python touch_ber.py sim                        # no hardware: detector fed from a recording, MOCK transmitter,
                                                  # simulated tablet capture of the frame (stent_trx_ber.synth_capture)
   python touch_ber.py sim --fail-every 2         # every 2nd frame simulated as never arriving (tests NOT_RECEIVED)
   python touch_ber.py run                        # hardware, transmitter MOCKED: real detector + real tablet recording,
                                                  # the AD9833 is never commanded -> every frame must be NOT_RECEIVED
   python touch_ber.py run --bench-tx --phantom-ok --current-ma 1
                                                  # BENCH ONLY (electrically reviewed phantom, NO person connected):
                                                  # real AD9833 output, one frame per touch, typed confirmation

Start rule (--trigger, default long since 2026-11-23): LONG PRESS = detector TOUCH and no RELEASE for --hold 1.0 s;
the frame starts when the hold time is reached, keep the finger on; one start per press. --trigger double = DOUBLE TAP
(default 2026-11-19 .. 11-22): TOUCH -> RELEASE within --max-tap 0.5 s -> TOUCH within --max-gap 0.5 s, frame at the 2nd
touch. --trigger single = every TOUCH (first runs, e.g. touchber_phantom_tx1_20261119_183236). The default sim recording
has 8 s holds (long / single), no double taps.

Device (--device): maxtouch (default, everything below) or pad = OnePlus Pad (2026-11-23, no hardware run yet):
  detector  same frozen Detector code, centred on --f0 = the Pad's own drive frequency seen at A0. NOT MEASURED YET, so
            `run --device pad` refuses without --f0 (find it receive-only with touch_drive_rx.py). Its thresholds were
            validated on the maXTouch only. The Pad's own touch reports are not used.
  frame     pad_trx_ber.one_run / decode_trx_pad unchanged: 37.4 kHz OOK, 4 baud, cc4 = 64 s (was 2 baud = 128 s; user:
            too long), 6 s lead, 2 s tail. --frame none_full = 32 s, NO error correction. Real 4 baud Pad frames before
            (1 mA, ankle 11-14, hand 11-20), CRC ok: OOK cc4 5/6, none_full 4/6; FSK cc4 6/6, none_full 3/6. On-air cap
            70 s per frame (maXTouch: 20 s, unchanged). Same RECEIVED_R: simulated noise-only Pad frames timing_r
            <= 0.25 cc4 (2-8 baud, n = 600), 0.29 none_full (n = 160); simulated 4 baud frames >= 0.83 / 0.75 (2026-11-23).

Sequence per trigger (all modes):
  start event (LONG PRESS, DOUBLE TAP or TOUCH) -> [cooldown / max-frames / must have seen RELEASE since the last frame] ->
  stop the A0 stream (detector paused: the BER receiver turns the tablet's noise suppression T72 off, which changes the
  drive the detector was validated on, and the AD9833 shares the electrode) -> stop the tablet touch-message reader
  (it and the BER recording both use the EVK control interface; the BER set-up also clears touch reports) ->
  stent_trx_ber.one_run: upload schedule, chip set-up + finger-on calibration, 4 s lead, sgo, frame, 2 s tail,
  transmitter off, chip restored -> decode_trx_tablet.decode_run -> classify -> restart the stream (0.3 s guard) and
  the touch-message reader -> wait for RELEASE.

BER frame (defaults = the 2026-11-16/18 tablet sessions): OOK, carrier 159.585 kHz (FSK 0 = 180.556 kHz), 40 baud, one
trx 'cc4' frame (sync 64 + 192 coded symbols = 256 symbols, 6.4 s), seed 1027, phase number round(rate), frame i of
the session = round i of trx_psk_link_v2.build_seq; finger-on calibration, lead 4 s, nodes y0 0 / ny 28.

Outcome (rule fixed 2026-11-19 before any hardware run; noise-only simulations gave timing_r <= 0.26, OOK
'reference_ok' passed on 78 % of noise-only frames, so it cannot tell a missing frame):
  SUCCESS          CRC-16 passes and 0 info-bit errors
  CRC_MISSED       CRC passes but info bits wrong (counted as a failure)
  RECEIVED_ERRORS  CRC fails and the decoder's timing correlation timing_r >= 0.40 -> BER reported
  NOT_RECEIVED     CRC fails and timing_r < 0.40, or signal lost, or the transmitter / tablet recording failed ->
                   NO BER (never counted as 0)
Log: data_touchdrive/touchber_<label>_<time>/triggers.jsonl (one line per trigger: detection, trigger, rx ready,
tx start / end, transmitter status, run folder, decoder result, outcome, BER) + summary.json; the BER run folders are
the normal stent_trx_ber ones (data_maxtouch/stent_trx_<label>_...).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import json
import os
import re
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stent_trx_ber as B                                           # noqa: E402  (one_run, schedule, synth_capture, D, R, STX, TV2)
import touch_detector as TDT                                        # noqa: E402  (Detector, FileSource, fingerprint)
import touch_drive_rx as T                                          # noqa: E402  (parse_pairs, T5Reader)

RECEIVED_R = 0.40
OUT = TDT.D
DETECT_CMDS = re.compile(r"^(x|off|\?|mono \d+)$")
TX_CMDS = re.compile(r"^(x|off|\?|mono \d+|psk 0|sclr|sseg \d+ \d+ \d+|sw( \d+)+|sinfo|sgo( \d+)*|sstat|wave (square|sine))$")


# ---------------------------------------------------------------------------------------------- transmitter objects
class MockTX:
    """stands in for stent_tx.StentTX inside stent_trx_ber.one_run; never touches any hardware"""

    def __init__(self):
        self.cmds, self.planned_steps = [], 0

    def upload_schedule(self, sched):
        self.cmds.append(("upload", [(st, reps, len(w)) for st, reps, w in sched]))
        self.planned_steps = sum(reps * len(w) for _, reps, w in sched)
        return dict(segs=len(sched), words=sum(len(w) for _, _, w in sched), mock=True)

    def go(self, delay_ms=B.STX.GO_DELAY_MS):
        t = time.time()
        self.cmds.append(("sgo (MOCK - not sent)", t))
        return t + delay_ms / 1e3, "# MOCK sgo"

    def status(self):
        return dict(state="mock", events=f"0/{self.planned_steps}", line="MOCK transmitter: nothing was sent")

    def off(self):
        self.cmds.append(("off", time.time()))
        return "# off (mock)"

    def ask(self, line, want, timeout=None):
        raise RuntimeError(f"MockTX: unexpected command {line!r}")


class Teensy:
    """one owner of the TeensyTRX serial port: A0 stream for the detector, and (bench mode only) one_run's commands.
    Every write is checked: detection phase = x / off / ? / mono only; TX window (bench mode) = the commands
    stent_trx_ber.one_run uses for an OOK/FSK frame."""

    def __init__(self, port, bench_tx):
        import stent_tx as STX
        self.stx = STX.StentTX(port)
        self.allowed = DETECT_CMDS
        self.bench_tx = bench_tx
        raw = self.stx.dev.write

        def guarded(data):
            for ln in data.decode(errors="replace").splitlines():
                if ln.strip() and not self.allowed.match(ln.strip()):
                    raise RuntimeError(f"blocked command {ln!r} ({'TX window' if self.allowed is TX_CMDS else 'detection phase'})")
            return raw(data)
        self.stx.dev.write = guarded
        self.off_before = self.stx.off()
        self.info = self.stx.info()
        if bench_tx:
            self.allowed = TX_CMDS
            try:
                self.stx.configure(wave="square")          # stent_trx_ber default (--wave square)
            finally:
                self.allowed = DETECT_CMDS

    def start(self, sps):
        self.stx.dev.reset_input_buffer()
        self.stx.dev.write(f"mono {int(sps)}\n".encode())

    def read(self, n):
        return self.stx.dev.read(n)

    def pause(self):
        self.stx.dev.write(b"x\n")
        time.sleep(0.3)
        tail = self.stx.dev.read(1 << 16).decode(errors="replace")
        self.stx.dev.reset_input_buffer()
        sl = re.findall(r"slips\s+(\d+)", tail)
        return int(sl[-1]) if sl else None

    @contextlib.contextmanager
    def tx_window(self):
        self.allowed = TX_CMDS if self.bench_tx else DETECT_CMDS
        try:
            yield self.stx if self.bench_tx else MockTX()
        finally:
            self.allowed = DETECT_CMDS
            self.off_after_frame = self.stx.off()

    def close(self):
        try:
            self.pause()
        finally:
            off = self.stx.off()
            self.stx.dev.close()
        return off


class DoubleTap:
    """start rule on top of the frozen detector's TOUCH / RELEASE events (times = A0 sample time, s):
    TOUCH -> RELEASE within max_tap -> TOUCH within max_gap of that RELEASE = double tap; the frame starts at the
    2nd TOUCH (the finger stays on for the finger-on calibration and the frame). Spliced real A0 (td_touch2, 11-19):
    the detector resolves taps and gaps down to 80 ms (5/5), event times 30-40 ms after the true edges."""

    def __init__(self, max_tap=0.5, max_gap=0.5):
        self.max_tap, self.max_gap = max_tap, max_gap
        self.reset()

    def reset(self):
        self.t_down = self.t_up = None

    def feed(self, ev, t):
        """returns (is_double_tap, info)"""
        if ev == "TOUCH":
            if self.t_up is not None and t - self.t_up <= self.max_gap:
                info = dict(tap1_s=round(self.t_up - self.t_down, 3), gap_s=round(t - self.t_up, 3))
                self.reset()
                return True, info
            self.t_down, self.t_up = t, None
            return False, "tap 1"
        if ev == "RELEASE" and self.t_down is not None:
            if t - self.t_down <= self.max_tap:
                self.t_up = t
                return False, f"tap 1 {t - self.t_down:.2f} s - tap again within {self.max_gap:g} s"
            held = t - self.t_down
            self.reset()
            return False, f"press {held:.2f} s - too long for a double tap"
        return False, None


class LongPress:
    """start rule on the frozen detector's blocks (times = A0 sample time, s): TOUCH, then no RELEASE for `hold` s.
    Fed EVERY detector block (ev None between events). Fires once per press; a RELEASE before `hold` cancels. The
    detector reports a release ~30 ms late (3 blocks off), so presses a few 10 ms short of `hold` can still count."""

    def __init__(self, hold=1.0):
        self.hold = hold
        self.reset()

    def reset(self):
        self.t_down = None

    def feed(self, ev, t):
        """returns (start, info)"""
        if ev == "TOUCH":
            self.t_down = t
            return False, f"hold {self.hold:g} s to start"
        if ev == "RELEASE":
            note = None if self.t_down is None else f"press {t - self.t_down:.2f} s - too short, hold {self.hold:g} s"
            self.reset()
            return False, note
        if self.t_down is not None and t - self.t_down >= self.hold - 1e-9:
            info = dict(hold_s=round(t - self.t_down, 3), touch_s=self.t_down)
            self.reset()
            return True, info
        return False, None


def pad_tool():
    """the OnePlus Pad BER tool (imported only for --device pad)"""
    sys.path.append(str(HERE.parent / "screen-bloodflow-core-main"))
    import pad_trx_ber as PT
    return PT


def pad_args(a):
    """the pad_trx_ber settings pad_connect / one_run read (its defaults)"""
    return types.SimpleNamespace(**dict(vars(ber_args(a)), site=a.site, rounds_per_rec=1, yes=True, adb=None,
                                        serial=a.serial, phantom=a.phantom_ok))


class SimSource(TDT.FileSource):
    """recording replay for `sim`; skip() jumps over the time a frame takes, as the real detector is paused then"""

    def pause(self):
        return None

    def start(self, sps):
        if self.t0 is None:                                         # restarts after a frame keep the replay position
            super().start(sps)

    def skip(self, seconds):
        """the samples during a frame are never seen by the paused detector: jump over them"""
        self.i = min(len(self.x), self.i + int(seconds * self.sps))
        self.t0 -= seconds / self.speed


# ---------------------------------------------------------------------------------------------- one frame
def ber_args(a):
    """the stent_trx_ber settings one_run / R.Chip / R.record read (values of the 2026-11-16/18 tablet sessions)"""
    return types.SimpleNamespace(label=a.label, condition=a.condition, carrier=a.carrier, fsk_zero=a.fsk_zero,
                                 seed=a.seed, send=a.frame, rounds=1, current_ma=a.current_ma, calibrate="finger-on",
                                 pause=False, lead=4.0, y0=0, ny=28, patch=None, keep_noise_suppression=False,
                                 keep_touch=False, wave="square", port=a.port, mods=a.mod, rates=str(a.rate))


def frame_plan(a, i):
    """frame i of the session = round i of trx_psk_link_v2.build_seq(['cc4'], ...) as stent_trx_ber.chunks slices it"""
    seq = B.TV2.build_seq([a.frame], i + 1, a.seed, int(round(a.rate)))
    s0, L = int(seq["start"][i]), int(seq["length"][i])
    sl = dict(keys=[a.frame], enc=seq["enc"][i:i + 1], start=np.array([0]), length=seq["length"][i:i + 1],
              u=seq["u"][i:i + 1], phase=seq["phase"][s0:s0 + L], frame_index0=i)
    return dict(mod=a.mod, rate=a.rate, chunk=i, rounds=(i, i + 1), seq=sl, per_round=L)


def sim_frame(a, r, fail, rng, root):
    """simulated tablet capture of the frame (stent_trx_ber.synth_capture), saved and decoded like a real run"""
    sl = r["seq"]
    _, plan = B.schedule(r["mod"], r["rate"], sl["phase"], ber_args(a))
    t, X, meta = B.synth_capture(r["mod"], plan, sl["phase"], rng, A=0.0 if fail else 1500.0)
    meta.update(send=sl["keys"], chunk=r["chunk"], round0=r["rounds"][0], round1=r["rounds"][1],
                frame_index0=int(sl["frame_index0"]), args=dict(synthetic=True, simulated_missing=bool(fail)),
                tx_status=dict(state="mock", events="0/0"))
    d = os.path.join(root, f"SIM_frame{r['chunk']}")
    B.R.save(d, t, X, meta)
    B.save_seq(d, sl)
    return d, plan


def sim_frame_pad(a, r, fail, rng, root):
    """simulated Pad reads of the frame (pad_trx_ber.synth_reads), saved and decoded like a real run"""
    PT, sl = pad_tool(), r["seq"]
    _, plan = B.schedule(r["mod"], r["rate"], sl["phase"], pad_args(a))
    t, F = PT.synth_reads(np.zeros_like(sl["phase"]) if fail else sl["phase"], plan["sym_us"] / 1e6, rng)
    d = os.path.join(root, f"SIM_frame{r['chunk']}")
    os.makedirs(d)
    np.savez_compressed(os.path.join(d, "frames.npz"), F=F, t=t)
    json.dump(dict(modulation=r["mod"], plan=plan, t_first=PT.LEAD_S, send=sl["keys"], round0=r["rounds"][0],
                   round1=r["rounds"][1], chunk=r["chunk"], frame_index0=int(sl["frame_index0"]),
                   args=dict(synthetic=True, simulated_missing=bool(fail)), tx_status=dict(state="mock", events="0/0")),
              open(os.path.join(d, "meta.json"), "w"), indent=1)
    B.save_seq(d, sl)
    return d, plan


def classify(d, tx_ok, decode=B.D.decode_run):
    res = decode(d)
    f = (res.get("frames") or [None])[0]
    out = dict(run=os.path.basename(d), timing_r=res.get("timing_r"), signal_lost=res.get("signal_lost"),
               delay_ms=res.get("delay_ms"), erasures=res.get("erasures"))
    if f is None or res.get("signal_lost"):
        return dict(out, outcome="NOT_RECEIVED", ber_pre=None, ber_post=None, crc_ok=False)
    dd = f["decoders"][f["frame"]]
    out.update(crc_ok=dd["crc_ok"], info_err=dd["info_err"], pre_err=f["pre_err"], code_bits=f["code_bits"],
               reference_ok=f["reference_ok"])
    r = res.get("timing_r") or 0.0
    if dd["crc_ok"]:
        oc = "SUCCESS" if dd["info_err"] == 0 else "CRC_MISSED"
    elif r >= RECEIVED_R and tx_ok:
        oc = "RECEIVED_ERRORS"
    else:
        oc = "NOT_RECEIVED"
    got = oc != "NOT_RECEIVED"
    out.update(outcome=oc, ber_pre=f["pre_err"] / f["code_bits"] if got else None,
               ber_post=dd["info_err"] / B.TV2.N_INFO if got else None)
    return out


# ---------------------------------------------------------------------------------------------- main loop
def confirm_bench(a):
    if not (a.bench_tx and a.phantom_ok):
        return
    if a.current_ma is None:
        raise SystemExit("--bench-tx needs --current-ma (the current network setting, checked and logged)")
    sys.path.append(str(HERE.parent / "screen-bloodflow-core-main"))
    import phone_body_sweep as PS                                   # the existing IEC 60601-1 check
    PS.check_current([a.carrier] + ([a.fsk_zero] if a.mod == "fsk" else []), a.current_ma)
    print("\n*** BENCH TX MODE: the AD9833 WILL transmit one frame per touch. ***\n"
          "Only for the electrically reviewed phantom set-up. NO person may be connected to the RX/TX electrodes or the "
          "phantom; the artificial finger is moved by its insulating handle only.")
    if input('Type exactly  PHANTOM ONLY - NO PERSON CONNECTED  to continue: ').strip() != "PHANTOM ONLY - NO PERSON CONNECTED":
        raise SystemExit("not confirmed - nothing transmitted")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("sim", "run"))
    ap.add_argument("--device", default="maxtouch", choices=("maxtouch", "pad"), help="pad = OnePlus Pad")
    ap.add_argument("--frame", default="cc4", choices=("cc4", "none_full"),
                    help="cc4 = with FEC (default); none_full = half as long, no FEC (pad only)")
    ap.add_argument("--f0", type=float, help="pad: the Pad's drive frequency at A0 [Hz] (required for run)")
    ap.add_argument("--serial", help="pad: adb serial (default: auto)")
    ap.add_argument("--site", default="", help="pad: electrode site, for the record")
    ap.add_argument("--label", default="touchber")
    ap.add_argument("--condition", default="phantom")
    ap.add_argument("--mod", default="ook", choices=("ook", "fsk"))
    ap.add_argument("--rate", type=float, help="baud (default maxtouch 40, pad 4)")
    ap.add_argument("--carrier", type=float, help="Hz (default maxtouch 159585, pad 37400)")
    ap.add_argument("--fsk-zero", type=float, help="Hz (default maxtouch 180556, pad 156250)")
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--current-ma", type=float, help="current network setting (required with --bench-tx)")
    ap.add_argument("--trigger", default="long", choices=("long", "double", "single"),
                    help="long = long press starts a frame (default since 2026-11-23); double = double tap; single = every touch")
    ap.add_argument("--hold", type=float, default=1.0, help="long press: finger on this long starts the frame (s)")
    ap.add_argument("--max-tap", type=float, default=0.5, help="double tap: longest 1st tap (s)")
    ap.add_argument("--max-gap", type=float, default=0.5, help="double tap: longest release -> 2nd touch (s)")
    ap.add_argument("--max-frames", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=5.0, help="min s from the end of one frame to the next trigger")
    ap.add_argument("--seconds", type=float, help="stop after this long")
    ap.add_argument("--port")
    ap.add_argument("--sps", type=int, default=1_000_000)
    ap.add_argument("--no-tablet-truth", action="store_true", help="do not log the tablet's own touch messages")
    ap.add_argument("--bench-tx", action="store_true", help="REAL AD9833 output (bench phantom only; needs --phantom-ok)")
    ap.add_argument("--phantom-ok", action="store_true", help="I confirm: reviewed phantom set-up, no person connected")
    ap.add_argument("--source", default=str(OUT / "td_touch2_20261118_171250"), help="sim: recording for the detector")
    ap.add_argument("--speed", type=float, default=4.0, help="sim: replay speed")
    ap.add_argument("--fail-every", type=int, default=0, help="sim: every n-th frame is simulated as never arriving")
    a = ap.parse_args()
    if a.bench_tx and (a.cmd != "run" or not a.phantom_ok):
        raise SystemExit("--bench-tx needs `run` and --phantom-ok")
    if not 0.2 <= a.hold <= 10:
        raise SystemExit("--hold: 0.2 - 10 s")
    pad = a.device == "pad"
    if not pad and a.frame != "cc4":
        raise SystemExit("--frame none_full is for --device pad only")
    if not pad and a.f0 is not None:
        raise SystemExit("--f0 is for --device pad only (the maXTouch detector is frozen at 70.6 kHz)")
    if pad and a.cmd == "run" and a.f0 is None:
        raise SystemExit("--device pad: give --f0, the Pad's drive frequency at A0 (not measured yet; record it receive-only "
                         "with: touch_drive_rx.py run --no-evk --truth prompts --label pad_touch). Nothing was transmitted.")
    if a.f0 is not None and not 5e3 < a.f0 < a.sps / 2 - 5e3:
        raise SystemExit("--f0 must lie inside the A0 band")
    for k, m, o in (("rate", 40.0, 4.0), ("carrier", 159585.0, 37.4e3), ("fsk_zero", 180556.0, 156.25e3)):
        if getattr(a, k) is None:
            setattr(a, k, o if pad else m)
    fp = TDT.detector_fingerprint()
    if fp != TDT.FROZEN_DETECTOR_SHA:
        raise SystemExit(f"detector changed ({fp} != frozen {TDT.FROZEN_DETECTOR_SHA}) - refusing")
    ba = ber_args(a)
    r0 = frame_plan(a, 0)
    _, plan0 = B.schedule(a.mod, a.rate, r0["seq"]["phase"], ba)
    cap = 70.0 if pad else 20.0
    if plan0["total_s"] > cap:
        raise SystemExit(f"one frame is {plan0['total_s']:.1f} s on air (> {cap:g} s) - choose a higher --rate")
    confirm_bench(a)
    PT = pad_tool() if pad else None
    adb = pad_model = None
    if pad and a.cmd == "run":
        adb, pad_model, _ = PT.pad_connect(pad_args(a))
    mode = "SIM (recording + mock TX + simulated tablet)" if a.cmd == "sim" else \
        ("BENCH TX (real AD9833)" if a.bench_tx else "HARDWARE, TX MOCKED (AD9833 never commanded)")
    d = OUT / f"touchber_{a.label}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    d.mkdir(parents=True)
    log = open(d / "triggers.jsonl", "w")
    src = SimSource(a.source, a.speed) if a.cmd == "sim" else Teensy(a.port, a.bench_tx)
    det = TDT.Detector(fs=a.sps, f0=a.f0) if a.f0 is not None else TDT.Detector(fs=a.sps)
    taps = LongPress(a.hold) if a.trigger == "long" else DoubleTap(a.max_tap, a.max_gap)
    scr = "OnePlus Pad" if pad else "maXTouch"
    lead_s = PT.LEAD_S if pad else 4.0
    rng = np.random.default_rng(20261119)
    print(f"{scr}   {mode}\ndetector {TDT.VERSION} {fp} (frozen), centre {det.f0 / 1e3:g} kHz"
          + (" ** thresholds NOT validated on the Pad **" if pad else "") + f"\nframe: {a.mod.upper()} {a.rate:g} baud, trx {a.frame}, "
          f"{plan0['total_s']:.1f} s on air, carrier {a.carrier / 1e3:g} kHz   max {a.max_frames} frames, cooldown {a.cooldown:g} s\n"
          + (f"Start a frame: LONG PRESS the screen (>= {a.hold:g} s), keep the finger on.\n" if a.trigger == "long" else
             f"Start a frame: DOUBLE TAP the screen (tap <= {a.max_tap:g} s, gap <= {a.max_gap:g} s), keep the finger "
             f"on after the 2nd tap.\n" if a.trigger == "double" else "Start a frame: touch the screen.\n") +
          f"Per trigger: ~1.5 s set-up + {lead_s:g} s lead + {plan0['total_s'] + 2.5:.1f} s frame -> keep the finger on the "
          f"screen until the result is printed. First {det.params['init_s']:.0f} s: do not touch (baseline).\n")

    def start_truth():
        if a.cmd == "sim" or a.no_tablet_truth or pad:
            return None
        try:
            rd = T.T5Reader()
            rd.start()
            return rd
        except Exception as e:
            print(f"  (tablet touch messages unavailable: {e})")
            return None
    truth_msgs, t5 = [], start_truth()
    src.start(a.sps)
    t_start, carry, seg_n, seg_off, seg_s0 = time.time(), b"", 0, np.inf, 0
    n_frames, armed, last_end, guard_until, results, stop = 0, True, -1e9, 0.0, [], False
    try:
        while not stop and (a.seconds is None or time.time() - t_start < a.seconds):
            b = src.read(16384)
            if b is None:
                break
            if not b:
                continue
            raw = carry + b
            c, used = T.parse_pairs(np.frombuffer(raw, np.uint8))
            carry = raw[used:]
            if not len(c):
                continue
            now = time.time()
            seg_n += len(c)
            seg_off = min(seg_off, now - seg_n / a.sps)
            if now < guard_until:
                continue
            for r in det.feed(c.astype(float) * TDT.MV):
                if a.trigger == "long":
                    dbl, note = taps.feed(r["event"], r["s"] / a.sps)
                    if not r["event"] and not dbl:
                        continue
                elif not r["event"]:
                    continue
                else:
                    dbl, note = taps.feed(r["event"], r["s"] / a.sps) if a.trigger == "double" else (True, None)
                if r["event"] == "RELEASE":
                    armed = True
                    print(f"  {now - t_start:8.2f} s  RELEASE DETECTED" + (f"  ({note})" if note else ""))
                    continue
                s_touch = r["s"] if not (a.trigger == "long" and dbl) else note.pop("touch_s") * a.sps
                t_det = seg_off + (s_touch - seg_s0) / a.sps            # long press: the TOUCH, not the end of the hold
                if not dbl:
                    print(f"  {now - t_start:8.2f} s  TOUCH DETECTED  ({note})")
                    continue
                why = ("max frames reached" if n_frames >= a.max_frames else
                       "no RELEASE since the last frame" if not armed else
                       "cooldown" if now - last_end < a.cooldown else None)
                what = dict(long="LONG PRESS", double="DOUBLE TAP", single="TOUCH")[a.trigger]
                print(f"  {now - t_start:8.2f} s  {what} DETECTED" + (f" {note}" if isinstance(note, dict) else "")
                      + (f"  (no trigger: {why})" if why else f"  -> frame {n_frames + 1}, keep the finger on"))
                if why:
                    continue
                # ---- trigger one frame
                rec = dict(frame=n_frames, mode=mode, device=a.device, trigger=a.trigger, t_touch_detected=t_det, t_decision=now,
                           level_db=r["level_db"], baseline_db=r["baseline_db"])
                if isinstance(note, dict):
                    rec.update(note)
                rec["slips_before"] = src.pause()
                if t5 is not None:
                    t5.stop.set()
                    t5.join(2.0)
                    truth_msgs += t5.msgs
                    t5 = None
                rec["t_trigger"] = time.time()
                fr = frame_plan(a, n_frames)
                tx_ok = False
                try:
                    if a.cmd == "sim":
                        fail = bool(a.fail_every) and (n_frames + 1) % a.fail_every == 0
                        with contextlib.redirect_stdout(io.StringIO()):
                            rd, plan = (sim_frame_pad if pad else sim_frame)(a, fr, fail, rng, str(d))
                        meta = json.load(open(os.path.join(rd, "meta.json")))
                        tx_ok = True
                        rec.update(t_rx_ready=None, t_tx_start=None, tx_status=meta["tx_status"], simulated_missing=fail)
                        src.skip(1.5 + lead_s + plan["total_s"] + 2.5)
                    else:
                        with src.tx_window() as tx:
                            if pad:
                                rd, stop = PT.one_run(tx, dict(line=src.info["line"]), adb, pad_model, pad_args(a), fr)
                            else:
                                rd, _ = B.one_run(tx, dict(line=src.info["line"]), ba, fr)
                        if rd is None:
                            raise RuntimeError(f"{scr} recording too short")
                        meta = json.load(open(os.path.join(rd, "meta.json")))
                        st = meta.get("tx_status") or {}
                        want = f"{meta['plan']['n_steps']}/{meta['plan']['n_steps']}"
                        tx_ok = a.bench_tx and st.get("state") == "done" and st.get("events") == want
                        rec.update(t_rx_ready=meta.get("t_calibrated"), t_tx_start=meta.get("t_first"),
                                   t_tx_end=(meta.get("t_first") or 0) + meta["plan"]["total_s"], tx_status=st,
                                   tx_off_after=getattr(src, "off_after_frame", None))
                    rec.update(run_dir=str(rd), **classify(str(rd), tx_ok, *([PT.DP.decode_run] if pad else [])))
                    if stop:
                        print("  recording ended early (Ctrl+C, screen off or adb lost) - stopping the session")
                except Exception as e:                                  # noqa: BLE001
                    rec.update(outcome="NOT_RECEIVED", error=str(e), ber_pre=None, ber_post=None)
                rec["t_done"] = time.time()
                n_frames += 1
                armed, last_end = False, time.time()
                results.append(rec)
                log.write(json.dumps(rec, default=str) + "\n")
                log.flush()
                ber = (f"BER before FEC {rec['ber_pre']:.4f}, after FEC {rec['ber_post']:.4f}"
                       if rec.get("ber_pre") is not None else "no BER (frame not received)")
                lat = (f", touch -> first symbol {rec['t_tx_start'] - rec['t_touch_detected']:.2f} s"
                       if rec.get("t_tx_start") else "")
                print(f"  ===> frame {n_frames}: {rec['outcome']}  (CRC {'ok' if rec.get('crc_ok') else 'fail'}, "
                      f"timing_r {rec.get('timing_r') if rec.get('timing_r') is None else round(rec['timing_r'], 2)}) "
                      f"{ber}{lat}" + (f"  error: {rec['error']}" if rec.get("error") else ""))
                if stop:
                    break
                # ---- resume detection
                t5 = start_truth()
                carry, seg_n, seg_off = b"", 0, np.inf
                det.buf = np.zeros(0)                                   # drop the partial block from before the pause
                seg_s0 = det.n
                taps.reset()
                src.start(a.sps)
                guard_until = time.time() + 0.3
                break
    except KeyboardInterrupt:
        print("\n  stopped (Ctrl+C)")
    finally:
        off = src.close() if hasattr(src, "close") else "n/a (sim)"
        if t5 is not None:
            t5.stop.set()
            t5.join(2.0)
            truth_msgs += t5.msgs
        log.close()
    counts = {k: sum(1 for x in results if x["outcome"] == k)
              for k in ("SUCCESS", "CRC_MISSED", "RECEIVED_ERRORS", "NOT_RECEIVED")}
    summ = dict(mode=mode, device=a.device, detector=fp, detector_f0=det.f0, trigger=a.trigger, received_r=RECEIVED_R, frames=len(results), **counts,
                transmitter_after=off, args=vars(a), tablet_touch_messages=len(truth_msgs))
    json.dump(summ, open(d / "summary.json", "w"), indent=1, default=str)
    json.dump(dict(t5=truth_msgs), open(d / "truth.json", "w"))
    got = [x for x in results if x.get("ber_pre") is not None]
    pooled = (f"   pooled BER over {len(got)} received frame(s): before FEC "
              f"{sum(x['pre_err'] for x in got)}/{sum(x['code_bits'] for x in got)}, after FEC "
              f"{sum(x['info_err'] for x in got)}/{len(got) * B.TV2.N_INFO}" if got else "")
    print(f"\n{len(results)} frame(s): {counts}   transmitter after: {off}\n"
          f"frame success rate {counts['SUCCESS']}/{len(results)}{pooled}\nlog: {d}")


if __name__ == "__main__":
    main()
