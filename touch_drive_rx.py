#!/usr/bin/env python3
"""
Can the stent receiver hear the TABLET? Touch-triggered link experiment (advisor's idea, 2026-11-18).

The maXTouch drives its X lines with bursts (expected around 100 kHz). A finger on the panel couples that drive into
the body, so the TeensyTRX receiver (A0 = E+, the same electrode pair the stent link uses) may see it grow when the
finger touches and drop when it leaves. If the three states (no touch / touch and hold / release) are separable, the
tablet's own drive could tell the stent "a finger is on the screen, start sending".

RECEIVE ONLY. The AD9833 is switched off ('off', confirmed) before the stream and again at the end; this tool never
sends cw / sgo / psk, so no current is put into anybody. No BER, no data.

   python touch_drive_rx.py plan    [--cycles 6 --hold 8 --rest 8 --lead 10]
   python touch_drive_rx.py run     --label hand --site wrist --condition still    (the user runs it, prompts on screen)
   python touch_drive_rx.py run     --label ctl_notouch --free 60    (free run: no prompts, e.g. a no-touch control)
   python touch_drive_rx.py run     --label ctl_evk_unplugged --free 30 --no-evk   (EVK unplugged, Teensy only)
   python touch_drive_rx.py analyse data_touchdrive/<run folder> [...]
   python touch_drive_rx.py synth                                    (analysis self-test on a simulated recording)

Recording (data_touchdrive/td_<label>_<time>/):
  rx.npy        A0 stream, 12-bit ADC counts (uint16), --sps samples/s (default 1 MS/s, the trx_recip rate; Nyquist
                500 kHz)
  clock.npy     (host time, samples received) after every USB read -> host time of each sample
  truth.json    ground truth from the tablet itself: every report of the maXTouch DIGITIZER interface (passive hidraw
                read, the chip is not polled or reconfigured), plus the on-screen prompts and their times
  meta.json     arguments, transmitter 'off' replies, stream stats (slips), maXTouch T7 power config (idle / active
                acquisition intervals) and T100 CTRL (touch reports must be on for the ground truth)

Analysis (fixed before the first recording, 2026-11-18):
  A1 truth: touch intervals = runs of digitizer reports with gaps < 0.15 s. If the digitizer gave nothing, the prompt
     schedule is the truth (prompted HOLD = touched). Blocks within 0.3 s of an edge are left out of the statistics.
  A1b (added 2026-11-18 before the second round): --truth prompts uses the prompt schedule even when the tablet
     reports touches (bezel / hover controls); the tablet's reports inside / outside the prompted intervals are counted.
  A1c (added 2026-11-18 after 0 digitizer reports in all runs, before any run with it): touch intervals from the
     controller's own T100 messages (T44/T5 polled read-only every ~2 ms) - touched while any touch ID has DETECT set;
     used before the digitizer and the prompts when present.
  A2 spectra: 20 ms blocks, mean of Hann-windowed 2048-point periodograms (0.49 kHz bins), 10-490 kHz.
     Touched vs untouched median spectra -> ratio in dB; the 8 strongest lines (peaks of the ratio) are listed.
  A3 features per block: (a) BAND = power in the bin with the largest touched/untouched ratio +-2 bins, the bin chosen
     on the FIRST half of the recording only; (b) BROAD = total power 10-490 kHz (no choice made). log10 power.
  A4 separation on the SECOND half: AUC (touched vs untouched), threshold = midpoint of the two medians of the first
     half, block error rates (false 'touch' while untouched, missed touch), and detection delay of each touch-down /
     release edge = first time the feature (3-block median) crosses the threshold the right way and stays for 3 blocks.
  A5 burst rate: the maXTouch scans in bursts (7.31 ms scans measured 2026-10-20) and may scan slower while idle
     (T7 IDLEACQINT vs ACTVACQINT). Envelope at the chosen line (complex demodulation, 0.1 ms) -> spectrum of 2 s
     stretches -> strongest repetition rate 5-500 Hz, per state. A rate change is a touch cue independent of amplitude.
Figures: overview.png (spectrogram with the touch intervals, feature vs time with threshold, spectra touched /
untouched, feature histograms) + result.json.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "data_touchdrive"
VERSION = "touchdrive-1"
BLOCK_S, NFFT = 0.020, 2048
F_LO, F_HI = 10e3, 490e3
TRUTH_GAP_S, EDGE_GUARD_S = 0.15, 0.3


# ----------------------------------------------------------------------------------------------- schedule / plan
def schedule(a):
    """[(t_rel, state, text)] state 0 = hands off, 1 = touch and hold"""
    if a.free:
        return [(0.0, -1, f"FREE RUN {a.free:.0f} s - do what the label says ({a.label})")], float(a.free)
    ev, t = [(0.0, 0, "hands OFF the screen (lead)")], a.lead
    for c in range(a.cycles):
        ev.append((t, 1, f"[{c + 1}/{a.cycles}] {a.action.upper()} and HOLD still"))
        t += a.hold
        ev.append((t, 0, f"[{c + 1}/{a.cycles}] RELEASE - hand away (off the screen and bezel)"))
        t += a.rest
    return ev, t


def plan(a):
    ev, total = schedule(a)
    print(f"receive only (AD9833 off), A0 stream {a.sps / 1e3:.0f} kS/s, {total:.0f} s, "
          f"{2 * a.sps * total / 1e6:.0f} MB")
    for t, s, txt in ev:
        print(f"  {t:6.1f} s  {txt}")


# ----------------------------------------------------------------------------------------------- recording
class Digitizer(threading.Thread):
    """passive read of every maXTouch EVK HID interface except the control one (interface 0): the touchscreen
    collection (hid-multitouch) AND interface 2 (configuration + a mouse collection - the EVK may report touches
    there when the host does not switch it to multitouch mode; 2026-11-18 runs got 0 reports on interface 1 alone).
    Every report is kept as (host time, 'if<N>:<hex>')."""

    def __init__(self):
        super().__init__(daemon=True)
        import maxtouch_hid as MH
        self.hs, seen = [], set()
        for d in MH.hid.enumerate(MH.VENDOR_ID, MH.PRODUCT_ID):
            if d["interface_number"] == 0 or d["path"] in seen:
                continue
            seen.add(d["path"])
            try:
                h = MH.hid.device()
                h.open_path(d["path"])
                h.set_nonblocking(1)
                self.hs.append((d["interface_number"], h))
            except Exception as e:
                print(f"  ! could not open EVK interface {d['interface_number']} ({d['path']}): {e}")
        self.h = self.hs or None
        self.reports, self.stop = [], threading.Event()

    def run(self):
        if not self.hs:
            return
        while not self.stop.is_set():
            got = False
            for i, h in self.hs:
                d = h.read(64)
                if d:
                    self.reports.append((time.time(), f"if{i}:" + bytes(d).hex()))
                    got = True
            if not got:
                time.sleep(0.0005)
        for _, h in self.hs:
            h.close()


class T5Reader(threading.Thread):
    """touch messages straight from the controller (added 2026-11-18 after 0 HID digitizer reports in every run):
    poll T44 (message count) on the control interface and read T5 (message processor) - the path maxtouch_hid's
    stream_t100_peaks uses. Read only: no chip setting is changed. Every message is kept as (host time of the poll
    that found it, hex) - stamping at the start of the poll since 2026-11-18 evening (was: after the T5 read);
    T100 touch report IDs (all T100 IDs except the first two: screen status, reserved) are stored for A1c."""

    def __init__(self, poll_s=0.002):
        super().__init__(daemon=True)
        import maxtouch_hid as MH
        self.msgs, self.stop, self.err, self.poll_s = [], threading.Event(), None, poll_s
        self.m = MH.MaxTouchHid()
        self.m.load_info()
        self.t44 = self.m.object_address(44, 0)
        self.t5 = self.m.object_address(5, 0)
        self.t5n = self.m.object_size(5, 0)
        ids = sorted(r for r, ty in self.m._report_id_to_type.items() if ty == 100)
        self.touch_ids = ids[2:]
        if self.t44 is None or self.t5 is None:
            raise RuntimeError("T44 / T5 not found")

    def run(self):
        """each message is stamped with the host time at the START of the T44 poll that found it (it was queued
        between the previous poll and this one, so its timing error is at most one poll gap); poll start times are
        kept in self.polls (for the gap statistics of a session)"""
        self.polls = []
        try:
            while not self.stop.is_set():
                tp = time.time()
                self.polls.append(tp)
                n = self.m.read_register(self.t44, 1)[0]
                for _ in range(min(n, 32)):
                    msg = self.m.read_register(self.t5, self.t5n)
                    if msg and msg[0] not in (0x00, 0xFF):
                        self.msgs.append((tp, msg.hex()))
                if not n:
                    time.sleep(self.poll_s)
        except Exception as e:
            self.err = str(e)
        finally:
            try:
                self.m.close()
            except Exception:
                pass

    def poll_stats(self):
        g = np.diff(np.array(getattr(self, "polls", []), float)) * 1e3
        if not len(g):
            return dict(polls=0)
        return dict(polls=int(len(g) + 1), gap_ms_p50=round(float(np.median(g)), 2),
                    gap_ms_p99=round(float(np.percentile(g, 99)), 2), gap_ms_max=round(float(g.max()), 2))


class NoDigitizer:
    """--no-evk: stands in for Digitizer, records nothing and never opens HID"""
    h, reports = None, []

    def __init__(self):
        self.stop = threading.Event()

    def start(self):
        pass

    def join(self, timeout=None):
        pass


def chip_info():
    """read-only: T7 power config and T100 CTRL (RPTEN must be set or the digitizer reports nothing)"""
    try:
        import maxtouch_hid as MH
        m = MH.MaxTouchHid()
        try:
            m.load_info()
            out = {}
            t7 = m.object_address(7, 0)
            if t7 is not None:
                r = m.read_register(t7, m.object_size(7, 0))
                out["T7"] = dict(raw=r.hex(), idleacqint_ms=r[0], actvacqint_ms=r[1],
                                 actv2idleto_200ms=r[2] if len(r) > 2 else None)
            t100 = m.object_address(100, 0)
            if t100 is not None:
                c = m.read_register(t100, 1)[0]
                out["T100_ctrl"] = c
                out["touch_reports_on"] = bool(c & 0x02)
            t72 = m.object_address(72, 0)
            if t72 is not None:
                out["T72_ctrl"] = m.read_register(t72, 1)[0]
            return out
        finally:
            m.close()
    except Exception as e:
        return {"error": str(e)}


def parse_pairs(arr):
    """TeensyLink stream bytes -> (codes, bytes used). 0x80 | high 7 bits, then low 7 bits, of adc12 << 2"""
    out, i, n = [], 0, len(arr)
    while n - i >= 2:
        pr = arr[i:i + (n - i) // 2 * 2].reshape(-1, 2)
        bad = np.flatnonzero(((pr[:, 0] & 0x80) == 0) | ((pr[:, 1] & 0x80) != 0))
        k = int(bad[0]) if len(bad) else len(pr)
        out.append((((pr[:k, 0] & 0x7F).astype(np.uint16) << 7) | pr[:k, 1]) >> 2)
        i += 2 * k
        if not len(bad):
            break
        j = i + 1
        while j < n - 1 and not ((arr[j] & 0x80) and not (arr[j + 1] & 0x80)):
            j += 1
        i = j
    return (np.concatenate(out) if out else np.zeros(0, np.uint16)), i


def run(a):
    import stent_tx as STX
    ev, total = schedule(a)
    plan(a)
    if a.no_evk:
        import maxtouch_hid as MH                   # same HID backend (hidraw or hid) as the EVK tools
        if MH.hid.enumerate(MH.VENDOR_ID, MH.PRODUCT_ID):
            raise SystemExit("--no-evk: the maXTouch EVK is still on USB - unplug it for this control (Teensy stays)")
        chip = {"evk_connected": False, "note": "receiver-only control, EVK unplugged (no drive, no touch reports)"}
    else:
        chip = chip_info()
    print(f"maXTouch: {chip}")
    if chip.get("touch_reports_on") is False:
        print("  ! T100 touch reports are OFF (an earlier BER run left RPTEN cleared?) - the digitizer will give no "
              "ground truth; the prompts are used instead. Unplug/replug the EVK to restore its defaults.")
    tx = STX.StentTX(a.port)
    off1 = tx.off()
    info = tx.info()
    print(f"Teensy: {info['line']}   output: {off1}")
    d = OUT / f"td_{a.label}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    d.mkdir(parents=True)
    dig = NoDigitizer() if a.no_evk else Digitizer()
    dig.start()
    t5 = None
    if not a.no_evk:
        try:
            t5 = T5Reader()
            t5.start()
            print(f"touch messages: polling T44/T5, T100 touch report IDs {t5.touch_ids}")
        except Exception as e:
            print(f"  ! controller touch messages not available ({e}) - tablet timing from the digitizer / prompts")
            t5 = None
    if not a.yes:
        input(f"\nRX electrodes on ({a.site or 'site?'}); {ev[0][2]}. Press Enter to start ")
    buf = np.empty(int(a.sps * (total + 10)), np.uint16)
    n, carry, clock, prompts, k = 0, b"", [], [], 0
    dev = tx.dev
    dev.reset_input_buffer()
    dev.write(f"mono {int(a.sps)}\n".encode())
    t0 = time.time()
    aborted = False
    try:
        while True:
            now = time.time() - t0
            while k < len(ev) and now >= ev[k][0]:
                print(f"\a  {now:6.1f} s  >>> {ev[k][2]}", flush=True)
                prompts.append(dict(t=t0 + ev[k][0], t_shown=time.time(), state=ev[k][1], text=ev[k][2]))
                k += 1
            if now >= total:
                break
            raw = carry + dev.read(1 << 16)
            arr = np.frombuffer(raw, np.uint8)
            c, used = parse_pairs(arr)
            carry = raw[used:]
            m = min(len(c), len(buf) - n)
            buf[n:n + m] = c[:m]
            n += m
            clock.append((time.time(), n))
    except KeyboardInterrupt:
        aborted = True
        print("\n  stopped (Ctrl+C)")
    finally:
        dev.write(b"x\n")
        time.sleep(0.4)
        tail = dev.read(1 << 16).decode(errors="replace")
        dev.reset_input_buffer()
        sl = re.findall(r"slips\s+(\d+)", tail)
        try:
            off2 = tx.off()
        except Exception as e:
            off2 = f"! {e}"
        tx.dev.close()
        dig.stop.set()
        dig.join(1.0)
        if t5 is not None:
            t5.stop.set()
            t5.join(2.0)
    el = time.time() - t0
    np.save(d / "rx.npy", buf[:n])
    np.save(d / "clock.npy", np.array(clock, float))
    json.dump(dict(digitizer=dig.reports, prompts=prompts, t5=t5.msgs if t5 else [],
                   t5_touch_ids=t5.touch_ids if t5 else [], t5_error=t5.err if t5 else None),
              open(d / "truth.json", "w"))
    meta = dict(version=VERSION, args=vars(a), t0=t0, sps=a.sps, samples=int(n), host_s=el,
                rate=n / max(el, 1e-9), slips=int(sl[-1]) if sl else None, aborted=aborted, teensy=info["line"],
                tx_off_before=off1, tx_off_after=off2, chip=chip, digitizer_reports=len(dig.reports),
                digitizer_found=dig.h is not None, no_evk=bool(a.no_evk))
    json.dump(meta, open(d / "meta.json", "w"), indent=1)
    print(f"\nsaved {d}  ({n} samples, {n / max(el, 1e-9) / 1e3:.1f} kS/s, slips {meta['slips']}, "
          f"digitizer reports {len(dig.reports)}, output after: {off2})")
    per_if = {}
    for _, h in dig.reports:
        k = h.split(":", 1)[0]
        per_if[k] = per_if.get(k, 0) + 1
    print(f"tablet touch reports per interface: {per_if or 'none'}")
    if t5 is not None:
        nt = sum(1 for _, h in t5.msgs if int(h[:2], 16) in t5.touch_ids)
        iv, _ = truth_intervals(dict(t5=t5.msgs, t5_touch_ids=t5.touch_ids), time.time())
        print(f"controller touch messages: {len(t5.msgs)} ({nt} touch), touches found: {len(iv)}"
              + (f"   ! reader stopped: {t5.err}" if t5.err else ""))
    if n > a.sps:
        analyse_dir(d)


# ----------------------------------------------------------------------------------------------- analysis
def sample_times(clock, sps, n):
    """host time of each sample: arrival times lag, so the earliest-arrival bound of t - n/sps is the offset"""
    off = float(np.min(clock[:, 0] - clock[:, 1] / sps))
    return off, off + np.arange(n) / sps


def t5_intervals(msgs, ids, t_end):
    """A1c: touch intervals from controller T100 messages - touched while any touch report ID has DETECT (bit 7)"""
    ids, act, iv, t_on = set(ids), {}, [], None
    for t, h in msgs:
        b = bytes.fromhex(h)
        if len(b) < 2 or b[0] not in ids:
            continue
        act[b[0]] = bool(b[1] & 0x80)
        now = any(act.values())
        if now and t_on is None:
            t_on = t
        elif not now and t_on is not None:
            iv.append((float(t_on), float(t)))
            t_on = None
    if t_on is not None:
        iv.append((float(t_on), float(t_end)))
    return [(s, e) for s, e in iv if e - s > 0.05]


def truth_intervals(truth, t_end, mode="auto"):
    if mode != "prompts" and truth.get("t5"):
        iv = t5_intervals(truth["t5"], truth.get("t5_touch_ids", []), t_end)
        if iv:
            return iv, "controller"
    rep = np.array([t for t, _ in truth.get("digitizer", [])], float)
    if mode != "prompts" and len(rep) >= 2:
        br = np.flatnonzero(np.diff(rep) > TRUTH_GAP_S)
        st = np.r_[rep[0], rep[br + 1]]
        en = np.r_[rep[br], rep[-1]]
        return [(float(s), float(e)) for s, e in zip(st, en) if e - s > 0.05], "digitizer"
    iv, cur = [], None
    for p in truth.get("prompts", []):
        if p["state"] == 1:
            cur = p["t"]
        elif cur is not None:
            iv.append((cur, p["t"]))
            cur = None
    if cur is not None:
        iv.append((cur, t_end))
    return iv, "prompts"


def block_spectra(x, sps):
    L = int(round(BLOCK_S * sps))
    nseg = L // NFFT
    nb = len(x) // L
    w = np.hanning(NFFT).astype(np.float32)
    X = x[:nb * L].reshape(nb, L)[:, :nseg * NFFT].reshape(nb, nseg, NFFT)
    X = X - X.mean(axis=2, keepdims=True)
    P = (np.abs(np.fft.rfft(X * w, axis=2)) ** 2).mean(axis=1) / (np.sum(w ** 2) * sps)
    f = np.fft.rfftfreq(NFFT, 1 / sps)
    return f, P.astype(np.float64), L


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    allv = np.r_[pos, neg]
    r = np.argsort(np.argsort(allv)) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def edge_delays(tb, y, thr, sign_up, edges, horizon=3.0):
    """first time after each edge where the 3-block median feature is on the far side of thr for 3 blocks"""
    ys = np.convolve(y, np.ones(3) / 3, mode="same")
    out = []
    for te, up in edges:
        k = np.flatnonzero((tb >= te) & (tb <= te + horizon))
        hit = None
        on = (ys[k] > thr) if (up == sign_up) else (ys[k] < thr)
        for j in range(len(k) - 2):
            if on[j] and on[j + 1] and on[j + 2]:
                hit = float(tb[k[j]] - te)
                break
        out.append(dict(t=float(te), down=bool(up), delay_s=hit))
    return out


def burst_rate(x, sps, f0, tsec, segs):
    """A5: strongest envelope repetition rate 5-500 Hz in each (t0, t1) stretch (>= 1 s)"""
    dec = int(sps // 10000)
    res = []
    for s0, s1 in segs:
        i0, i1 = int((s0 - tsec[0]) * sps), int((s1 - tsec[0]) * sps)
        if i1 - i0 < sps:
            continue
        seg = x[i0:i1].astype(np.float64)
        seg -= seg.mean()
        ph = np.exp(-2j * np.pi * f0 * np.arange(len(seg)) / sps)
        z = seg * ph
        m = len(z) // dec
        env = np.abs(z[:m * dec].reshape(m, dec).mean(1))
        env -= env.mean()
        F = np.abs(np.fft.rfft(env * np.hanning(len(env))))
        fr = np.fft.rfftfreq(len(env), dec / sps)
        k = (fr >= 5) & (fr <= 500)
        if k.any():
            j = np.argmax(F[k])
            res.append(dict(t0=float(s0), t1=float(s1), rate_hz=float(fr[k][j]),
                            prominence=float(F[k][j] / (np.median(F[k]) + 1e-12))))
    return res


def analyse_arrays(x, tsec, truth, sps, meta=None, truth_mode="auto"):
    f, P, L = block_spectra(x.astype(np.float32), sps)
    nb = len(P)
    tb = tsec[0] + (np.arange(nb) + 0.5) * L / sps
    iv, src = truth_intervals(truth, tsec[-1], truth_mode)
    touched = np.zeros(nb, bool)
    guard = np.zeros(nb, bool)
    edges = []
    for s, e in iv:
        touched |= (tb >= s) & (tb < e)
        guard |= (np.abs(tb - s) < EDGE_GUARD_S) | (np.abs(tb - e) < EDGE_GUARD_S)
        edges += [(s, True), (e, False)]
    edges.sort()
    use = ~guard
    fk = (f >= F_LO) & (f <= F_HI)
    half = tb < tb[0] + (tb[-1] - tb[0]) / 2
    rep_t = np.array([t for t, _ in truth.get("digitizer", [])], float)
    if truth.get("t5"):
        ids = set(truth.get("t5_touch_ids", []))
        rep_t = np.array([t for t, h in truth["t5"] if int(h[:2], 16) in ids], float)
    inside = sum(int(((rep_t >= s) & (rep_t < e)).sum()) for s, e in iv)
    res = dict(version=VERSION, truth_mode=truth_mode, truth_source=src, timing_valid=src in ("digitizer", "controller"),
               tablet_reports=int(len(rep_t)), tablet_reports_in_intervals=inside,
               tablet_reports_outside=int(len(rep_t) - inside),
               touch_intervals=iv, blocks=int(nb), block_s=BLOCK_S,
               touched_blocks=int((touched & use).sum()), untouched_blocks=int((~touched & use).sum()))
    if (touched & use).sum() < 5 or (~touched & use).sum() < 5:
        res["note"] = "not enough touched / untouched blocks for statistics - spectrum lines over the floor only"
        lp = np.log10(P + 1e-30)
        med = np.median(lp, 0)
        floor = np.median(med[fk])
        rs = np.where(fk, med, -np.inf)
        pk = [i for i in range(1, len(rs) - 1) if rs[i] >= rs[i - 1] and rs[i] >= rs[i + 1] and np.isfinite(rs[i])]
        pk = sorted(pk, key=lambda i: -rs[i])[:8]
        res["lines_all"] = [dict(f_khz=float(f[i] / 1e3), over_floor_db=float(10 * (med[i] - floor))) for i in pk]
        return res, dict(f=f, P=P, tb=tb, touched=touched, iv=iv, med_all=med)
    lp = np.log10(P + 1e-30)
    med_t = np.median(lp[touched & use], 0)
    med_u = np.median(lp[~touched & use], 0)
    ratio_db = 10 * (med_t - med_u)
    floor = np.median(med_u[fk])
    # A2 lines
    rs = np.where(fk, ratio_db, -np.inf)
    pk = [i for i in range(1, len(rs) - 1) if rs[i] >= rs[i - 1] and rs[i] >= rs[i + 1] and np.isfinite(rs[i])]
    pk = sorted(pk, key=lambda i: -rs[i])[:8]
    res["lines"] = [dict(f_khz=float(f[i] / 1e3), ratio_db=float(ratio_db[i]),
                         touched_over_floor_db=float(10 * (med_t[i] - floor)),
                         untouched_over_floor_db=float(10 * (med_u[i] - floor))) for i in pk]
    # A3 band chosen on the first half
    h1 = use & half
    if (touched & h1).sum() >= 3 and (~touched & h1).sum() >= 3:
        r1 = 10 * (np.median(lp[touched & h1], 0) - np.median(lp[~touched & h1], 0))
    else:
        r1 = ratio_db
    i0 = int(np.argmax(np.where(fk, r1, -np.inf)))
    band = slice(max(0, i0 - 2), i0 + 3)
    feats = dict(band=np.log10(P[:, band].sum(1) + 1e-30), broad=np.log10(P[:, fk].sum(1) + 1e-30))
    res["band_khz"] = [float(f[band.start] / 1e3), float(f[band.stop - 1] / 1e3)]
    res["band_center_khz"] = float(f[i0] / 1e3)
    h2 = use & ~half
    for name, y in feats.items():
        a_ = (touched & h1)
        b_ = (~touched & h1)
        thr = (np.median(y[a_]) + np.median(y[b_])) / 2 if a_.sum() and b_.sum() else np.median(y)
        up = (np.median(y[a_]) > np.median(y[b_])) if a_.sum() and b_.sum() else True
        pos, neg = y[touched & h2], y[~touched & h2]
        call = (y > thr) if up else (y < thr)
        e2 = [(t, u) for t, u in edges if t >= tb[0] + (tb[-1] - tb[0]) / 2]
        dl = edge_delays(tb, y, thr, up, e2)
        res[name] = dict(threshold=float(thr), touch_raises=bool(up),
                         auc_second_half=auc(pos, neg) if up else auc(-pos, -neg),
                         false_touch=float(call[~touched & h2].mean()) if (~touched & h2).any() else None,
                         missed_touch=float((~call)[touched & h2].mean()) if (touched & h2).any() else None,
                         median_touched=float(np.median(y[touched & use])), median_untouched=float(np.median(y[~touched & use])),
                         contrast_db=float(10 * (np.median(y[touched & use]) - np.median(y[~touched & use]))),
                         edges=dl)
    # A5
    segs_t = [(s + EDGE_GUARD_S, e - EDGE_GUARD_S) for s, e in iv]
    gaps = [(tsec[0] + 1, iv[0][0] - EDGE_GUARD_S)] if iv else [(tsec[0] + 1, tsec[-1])]
    gaps += [(iv[i][1] + EDGE_GUARD_S, iv[i + 1][0] - EDGE_GUARD_S) for i in range(len(iv) - 1)]
    if iv:
        gaps.append((iv[-1][1] + EDGE_GUARD_S, tsec[-1]))
    res["burst_touched"] = burst_rate(x, sps, f[i0], tsec, segs_t)
    res["burst_untouched"] = burst_rate(x, sps, f[i0], tsec, gaps)
    return res, dict(f=f, P=P, tb=tb, touched=touched, use=use, med_t=med_t, med_u=med_u, feats=feats,
                     band=band, iv=iv)


def figure(d, res, A, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, P, tb = A["f"], A["P"], A["tb"]
    t = tb - tb[0]
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 1], width_ratios=[2.2, 1])
    a = fig.add_subplot(gs[0, :])
    fk = (f >= F_LO) & (f <= F_HI)
    a.pcolormesh(t, f[fk] / 1e3, 10 * np.log10(P[:, fk].T + 1e-30), shading="auto", cmap="magma", rasterized=True)
    for s, e in A["iv"]:
        a.axvspan(s - tb[0], e - tb[0], ymin=0.97, ymax=1.0, color="#2ecc71")
    a.set_ylabel("frequency [kHz]")
    a.set_title(f"{title}: A0 spectrogram (green bar = finger on screen, {res['truth_source']})", loc="left", fontsize=10)
    if "med_all" in A:
        s = fig.add_subplot(gs[1:, :])
        s.plot(f[fk] / 1e3, 10 * A["med_all"][fk], color="#555", lw=0.8, label="median spectrum (whole run)")
        s.set_xlabel("frequency [kHz]")
        s.set_ylabel("PSD [dB]")
        s.legend(fontsize=8, frameon=False)
    if "band" in res:
        b = fig.add_subplot(gs[1, 0], sharex=a)
        y = A["feats"]["band"]
        b.plot(t, 10 * y, lw=0.7, color="#1f5fbf", label=f"band {res['band_center_khz']:.1f} kHz")
        b.axhline(10 * res["band"]["threshold"], color="#555", ls="--", lw=0.8, label="threshold (1st half)")
        for s, e in A["iv"]:
            b.axvspan(s - tb[0], e - tb[0], color="#2ecc71", alpha=0.15, lw=0)
        b.axvline((tb[-1] - tb[0]) / 2, color="#999", lw=0.6, ls=":")
        b.set_ylabel("band power [dB]")
        b.set_xlabel("time [s]")
        b.legend(fontsize=8, loc="upper right", frameon=False)
        c = fig.add_subplot(gs[1, 1])
        for m, lab, col in ((A["touched"] & A["use"], "touched", "#2e8b57"), (~A["touched"] & A["use"], "untouched", "#999")):
            c.hist(10 * y[m], bins=40, alpha=0.6, color=col, label=lab)
        c.axvline(10 * res["band"]["threshold"], color="#555", ls="--", lw=0.8)
        c.set_xlabel("band power [dB]")
        c.legend(fontsize=8, frameon=False)
        c.set_title(f"AUC (2nd half) {res['band']['auc_second_half']:.3f}", fontsize=9)
        s = fig.add_subplot(gs[2, :])
        s.plot(f[fk] / 1e3, 10 * A["med_u"][fk], color="#999", lw=0.8, label="untouched (median)")
        s.plot(f[fk] / 1e3, 10 * A["med_t"][fk], color="#2e8b57", lw=0.8, label="touched (median)")
        s.axvspan(res["band_khz"][0], res["band_khz"][1], color="#1f5fbf", alpha=0.2, lw=0)
        s.set_xlabel("frequency [kHz]")
        s.set_ylabel("PSD [dB]")
        s.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(d / "overview.png", dpi=120)
    plt.close(fig)


def report(res):
    print(f"  truth: {res['truth_source']}, {len(res['touch_intervals'])} touches, blocks touched/untouched "
          f"{res['touched_blocks']}/{res['untouched_blocks']}")
    if res.get("note_run"):
        print(f"  NOTE: {res['note_run']}")
    print(f"  tablet touch reports: {res['tablet_reports']} ({res['tablet_reports_in_intervals']} inside the "
          f"{'prompted' if res['truth_source'] == 'prompts' else 'touch'} intervals, {res['tablet_reports_outside']} outside)")
    if res["truth_source"] == "prompts" and res["truth_mode"] != "prompts" and res["touched_blocks"]:
        print("  ! timing from the PROMPTS (no tablet touch reports): accuracy and latency below include reaction "
              "time and missed prompts - not final")
    if "note" in res:
        print("  " + res["note"])
        for ln in res.get("lines_all", [])[:8]:
            print(f"    {ln['f_khz']:7.2f} kHz  {ln['over_floor_db']:+6.1f} dB over the spectrum median")
        return
    print("  lines touched vs untouched (A2):")
    for ln in res["lines"][:5]:
        print(f"    {ln['f_khz']:7.2f} kHz  {ln['ratio_db']:+6.1f} dB   (touched {ln['touched_over_floor_db']:+.1f} dB "
              f"over floor, untouched {ln['untouched_over_floor_db']:+.1f})")
    for name in ("band", "broad"):
        r = res[name]
        dn = [e["delay_s"] for e in r["edges"] if e["down"]]
        upd = [e["delay_s"] for e in r["edges"] if not e["down"]]
        fmt = lambda v: "-" if not v else ", ".join("miss" if x is None else f"{x * 1e3:.0f}" for x in v)
        lab = f"{name} {res['band_center_khz']:.1f} kHz" if name == "band" else "broad 10-490 kHz"
        print(f"  {lab}: contrast {r['contrast_db']:+.1f} dB, AUC {r['auc_second_half']:.3f}, false touch "
              f"{r['false_touch']:.3f}, missed {r['missed_touch']:.3f}; delay touch [ms] {fmt(dn)}; release {fmt(upd)}")
    for k in ("burst_untouched", "burst_touched"):
        v = [b["rate_hz"] for b in res[k]]
        if v:
            print(f"  scan repetition {k.split('_')[1]}: median {np.median(v):.1f} Hz ({len(v)} stretches)")


def analyse_dir(d, truth_mode=None):
    d = Path(d)
    meta = json.load(open(d / "meta.json"))
    truth = json.load(open(d / "truth.json"))
    x = np.load(d / "rx.npy")
    clock = np.load(d / "clock.npy")
    sps = float(meta["sps"])
    _, tsec = sample_times(clock, sps, len(x))
    mode = truth_mode or meta.get("args", {}).get("truth") or "auto"
    res, A = analyse_arrays(x, tsec, truth, sps, meta, mode)
    if meta.get("note"):
        res["note_run"] = meta["note"]
    res.update(run=d.name, slips=meta.get("slips"), chip=meta.get("chip"))
    json.dump(res, open(d / "result.json", "w"), indent=1)
    print(f"\n== {d.name}")
    report(res)
    figure(d, res, A, d.name)
    print(f"  -> {d / 'overview.png'}")
    return res


# ----------------------------------------------------------------------------------------------- synth
def synth(a):
    """touch -> drive line at 112 kHz 20 dB stronger, scans every 7.3 ms (touched) / 20 ms (idle)"""
    rng = np.random.default_rng(3)
    sps, T = 1_000_000, 40.0
    t = np.arange(int(sps * T)) / sps
    iv = [(5 + 7 * c, 9 + 7 * c) for c in range(5)]
    touch = np.zeros(len(t), bool)
    for s, e in iv:
        touch |= (t >= s) & (t < e)
    period = np.where(touch, 7.3e-3, 20e-3)
    phase = np.cumsum(1 / period) / sps
    burst = (phase % 1) < 0.35
    amp = np.where(touch, 40.0, 4.0) * burst
    x = 2048 + amp * np.sin(2 * np.pi * 112e3 * t) + rng.normal(0, 6, len(t))
    x = np.clip(np.round(x), 0, 4095).astype(np.uint16)
    t0 = 1.8e9
    truth = dict(digitizer=[(t0 + s + k * 0.008, "") for s, e in iv for k in range(int((e - s) / 0.008))], prompts=[])
    res, A = analyse_arrays(x, t0 + t, truth, sps)
    report(res)
    out = Path(a.out or "/tmp") / "touchdrive_synth"
    out.mkdir(parents=True, exist_ok=True)
    figure(out, res, A, "SIMULATED")
    print(f"  -> {out / 'overview.png'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "run", "analyse", "synth"))
    ap.add_argument("what", nargs="*")
    ap.add_argument("--label", default="touch")
    ap.add_argument("--site", help="where the RX electrodes are, for the record (e.g. wrist, forearm, agar)")
    ap.add_argument("--condition", help="for the record, e.g. still")
    ap.add_argument("--action", default="touch the screen",
                    help="what the prompt asks for during HOLD, e.g. 'touch the bezel', 'hover 1 cm above the screen'")
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--hold", type=float, default=8.0, help="touch-and-hold time per cycle [s]")
    ap.add_argument("--rest", type=float, default=8.0, help="finger-off time per cycle [s]")
    ap.add_argument("--lead", type=float, default=10.0, help="finger-off time before the first touch [s]")
    ap.add_argument("--free", type=float, help="free run of this many seconds, no prompts (controls)")
    ap.add_argument("--sps", type=int, default=1_000_000)
    ap.add_argument("--port")
    ap.add_argument("--yes", action="store_true", help="start without waiting for Enter")
    ap.add_argument("--truth", choices=("auto", "prompts"), default=None,
                    help="touch timing for the analysis: auto = tablet touch reports, prompts if none (default); "
                         "prompts = the prompt schedule (bezel / hover controls, where the tablet should report no "
                         "touch - its reports are then only counted as a check)")
    ap.add_argument("--no-evk", action="store_true",
                    help="receiver-only control with the maXTouch EVK unplugged: no HID access at all, the script "
                         "stops if the EVK is still connected; use with --free")
    ap.add_argument("--out", help="synth: output folder")
    a = ap.parse_args()
    if a.cmd == "plan":
        plan(a)
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "analyse":
        for p in a.what:
            analyse_dir(p, a.truth)
    else:
        synth(a)


if __name__ == "__main__":
    main()
