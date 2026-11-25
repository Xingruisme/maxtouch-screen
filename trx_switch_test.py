#!/usr/bin/env python3
"""
Bench test of the TeensyTRX RX <-> TX switching and of its fail-safe (firmware TeensyTRX ask2 rev 2, 2026-11-24).

Why: on 2026-11-24 the board stopped answering twice right after an A0 stream (stent_tx.py selftest), once with the
carrier still on. touch_drive_rx.py / touch_ber.py use the same stream, and a two-way link switches between receiving
(A0 stream) and sending all the time. This test repeats those transitions and counts every step that fails.

    python trx_switch_test.py run                       # the bench test: REAL carrier on E+ (electrode off the body!)
    python trx_switch_test.py run --dry                 # rehearsal: firmware 'dry 1', nothing reaches E+
    python trx_switch_test.py run --quick 300 --cycles 0 --no-failsafe     # only the fast RX / TX / RX loop
    options: --carrier 159585  --label bench  --port /dev/ttyACM0  --wave square

What one run does
  1) --quick N (default 100) fast cycles:  RX (A0 stream, output off) -> TX (cw on) -> RX while sending -> TX off -> RX.
     Three stream stops per cycle = three stops of the 1 MS/s sample clock, the step the board died in.
  2) --cycles N (default 20) full cycles, on top of the above with scheduled sending as the PER / touch tools do it:
       a) 2-ASK schedule, no stream: the firmware stops the sample clock itself when the schedule ends
       b) schedule + A0 stream that outlasts it (stream stop after the schedule)
       c) schedule + A0 stream stopped while the schedule still runs
  3) fail-safe (skip with --no-failsafe):
       A) carrier on, then 'fs hang' (loop() stopped on purpose): the watchdog must reset the board within ~3 s, the
          board must answer again WITHOUT a replug and the carrier must be gone
       B) the same with interrupts disabled ('fs hang irq'), as in an interrupt storm
       C) cw hold: hold time 3 s, carrier on, host silent for 5 s -> the firmware switches the carrier off itself
       D) keep-alive: hold time 3 s, carrier on, a byte every second for 6 s -> the carrier stays on
Every command's reply is checked. A step without a reply is a FAILURE; the script then waits for the board to come
back (watchdog), reads where it was stuck ('fs': last_stage, stuck_pc, stuck_exc) and goes on.
In a real run the carrier line on A0 is checked too: >= 30 dB over the spectrum median and 20 dB over the 'off'
reading when on (2026-11-24 self-test: ~80 dB on, 8-22 dB off), back down when off. In --dry there is no carrier to see.

Output: data_maxtouch/trxswitch_<label>_<time>.json and a summary. Exit code 0 only if nothing failed.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

import stent_tx as STX

HERE = os.path.dirname(os.path.abspath(__file__))
ON_DB, ON_OVER_OFF_DB = 30.0, 20.0
BAUD = 200.0                                    # schedule symbol rate of the full cycles


class Bench:
    def __init__(self, a):
        self.a = a
        self.tx = None
        self.log = dict(args={k: v for k, v in vars(a).items()}, steps=0, failures=[], resets=[], levels=dict(off=[], on=[]),
                        failsafe={})
        self.t_step = {}
        self.wd = None                          # watchdog resets the firmware has counted (fs wdog_resets)

    # ---------------------------------------------------------------- link
    def open(self):
        self.tx, info = STX.open_configured(self.a)
        self.info = info["line"]
        if self.a.dry:                          # after every (re)boot the firmware is in real mode again
            r = self.tx.ask("dry 1", r"^# dry")
            if not r.endswith("1"):
                raise SystemExit(f"'dry 1' not taken ({r!r}) - not TeensyTRX ask2, stopping before anything is sent")
        return info["line"]

    def close(self):
        if self.tx is None:
            return
        try:
            self.tx.off()
            if self.a.dry:
                self.tx.ask("dry 0", r"^# dry")
        except Exception as e:
            print(f"  ! could not confirm the carrier is off ({e})")
        try:
            self.tx.dev.close()
        except Exception:
            pass
        self.tx = None

    def reopen(self, wait_s=20.0):
        """after a failure / a reset: wait for the Teensy's port and open it again -> seconds waited, or None"""
        t0 = time.time()
        try:
            self.tx.dev.close()
        except Exception:
            pass
        self.tx = None
        last = None
        while time.time() - t0 < wait_s:
            try:
                self.open()
                return time.time() - t0
            except SystemExit:
                raise
            except Exception as e:
                last = e
                try:
                    if self.tx is not None:
                        self.tx.dev.close()
                except Exception:
                    pass
                self.tx = None
                time.sleep(0.25)
        print(f"  ! the board did not come back within {wait_s:g} s ({last})")
        return None

    # ---------------------------------------------------------------- steps
    def probe(self):
        """after a failure, on the port as it is: does the board still answer? Tells a firmware that stopped
        streaming (answers here) from a USB receive path that stalled (answers only after the port is re-opened)."""
        out = {}
        try:
            dev = self.tx.dev
            out["in_waiting"] = dev.in_waiting
            for cmd, wait in (("x", 0.4), ("fs", 0.6)):
                dev.write((cmd + "\n").encode())
                end, got = time.time() + wait, b""
                while time.time() < end:
                    got += dev.read(1 << 16)
                txt = [ln for ln in got.decode(errors="replace").splitlines() if ln.startswith("#")]
                out[cmd] = dict(bytes=len(got), reply=txt[-1][:200] if txt else None)
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"[:160]
        return out

    def step(self, name, fn):
        """one checked step -> its result, or None after a failure (logged, board re-opened)"""
        self.log["steps"] += 1
        t = time.time()
        try:
            r = fn()
            self.t_step.setdefault(name, []).append(time.time() - t)
            return r if r is not None else True
        except (TimeoutError, RuntimeError, OSError, StopIteration, ValueError) as e:
            f = dict(step=name, n=self.log["steps"], error=f"{type(e).__name__}: {e}"[:200], t=time.time())
            print(f"\n  ** FAILED at step {self.log['steps']} ({name}): {f['error']}")
            f["probe"] = self.probe()
            print(f"     same port, before re-opening: {f['probe']}")
            back = self.reopen()
            f["back_after_s"] = back
            if back is not None:
                fs = self.tx.fs()
                f["fs"] = fs and fs["line"]
                print(f"     board back after {back:.1f} s: {fs['line'] if fs else 'no fs command (firmware older than rev 2)'}")
                if fs and self.wd is not None and int(fs["wdog_resets"]) > self.wd:
                    self.log["resets"].append(fs["line"])       # the watchdog brought it back
                if fs:
                    self.wd = int(fs["wdog_resets"])
            self.log["failures"].append(f)
            if back is None:
                raise SystemExit("the board does not answer - replug it; results so far are saved")
            return None

    def level(self, kind, seconds=0.1):
        """A0 stream -> carrier line [dB over the median]; kind 'on' / 'off' is what it should be"""
        def go():
            db = STX.carrier_line_db(self.tx.a0(seconds=seconds), self.a.carrier)
            self.log["levels"][kind].append(db)
            return db
        return self.step(f"rx_{kind}", go)

    def check_levels(self, off, on, off2, where):
        if self.a.dry or None in (off, on, off2):
            return
        bad = []
        if on < ON_DB or on < max(off, off2) + ON_OVER_OFF_DB:
            bad.append(f"carrier ON reads {on:.1f} dB (off {off:.1f} / {off2:.1f})")
        if off2 > on - ON_OVER_OFF_DB:
            bad.append(f"carrier still there after off: {off2:.1f} dB (on {on:.1f})")
        for b in bad:
            self.log["failures"].append(dict(step=where, n=self.log["steps"], error=b, t=time.time()))
            print(f"\n  ** FAILED ({where}): {b}")

    def quick(self, k):
        off = self.level("off")
        self.step("tx_on", lambda: self.tx.tone(self.a.carrier))
        on = self.level("on")
        self.step("tx_off", lambda: self.tx.off())
        off2 = self.level("off")
        self.check_levels(off, on, off2, f"quick cycle {k}")

    def sched(self, n_sym):
        r = self.tx.ask(f"ask2 {self.a.carrier:.3f} {BAUD:g} prbs {n_sym}", r"^# ask\d")
        if "refused" in r:
            raise RuntimeError(r)
        return time.time() + 1.0 + n_sym / BAUD     # the firmware starts 1 s after the command

    def done(self, n_sym):
        st = self.tx.status()
        if st["state"] != "done" or st["events"] != f"{n_sym}/{n_sym}" or st.get("adg_err", "0") != "0":
            raise RuntimeError(f"schedule not done cleanly: {st['line']}")
        return st

    def full(self, k):
        self.quick(k)

        def a():                                # schedule alone
            t_end = self.sched(100)
            if not self.tx.wait_done(t_end - time.time() + 3):
                raise RuntimeError("schedule did not finish")
            return self.done(100)
        self.step("sched", a)

        def b():                                # stream outlasts the schedule
            t_end = self.sched(100)
            self.tx.a0(seconds=t_end - time.time() + 0.3)
            return self.done(100)
        self.step("sched+stream", b)

        def c():                                # stream stopped while the schedule still runs
            t_end = self.sched(300)
            self.tx.a0(seconds=0.4)
            if not self.tx.wait_done(t_end - time.time() + 3):
                raise RuntimeError("schedule did not finish")
            return self.done(300)
        self.step("stream-stop-in-sched", c)
        self.step("off", lambda: self.tx.off())

    # ---------------------------------------------------------------- fail-safe
    def fs_or_stop(self):
        fs = self.tx.fs()
        if fs is None:
            print("  the firmware has no 'fs' command (older than rev 2): fail-safe tests skipped")
        return fs

    def hang(self, irq):
        name = "hang_irq_off" if irq else "hang"
        res = dict(ok=False)
        self.log["failsafe"][name] = res
        fs0 = self.fs_or_stop()
        if fs0 is None:
            return
        print(f"  {'B' if irq else 'A'}) carrier on, then 'fs hang{' irq' if irq else ''}' ...", end="", flush=True)
        self.tx.tone(self.a.carrier)
        on = STX.carrier_line_db(self.tx.a0(seconds=0.1), self.a.carrier)
        t0 = time.time()
        self.tx.send("fs hang irq" if irq else "fs hang")
        time.sleep(0.3)
        back = self.reopen(wait_s=20.0)
        if back is None:
            res["error"] = "the board did not come back - the watchdog did not reset it"
            print(f"\n  ** FAILED: {res['error']}")
            raise SystemExit("replug the board; results so far are saved")
        t_back = time.time() - t0
        fs1 = self.tx.fs()
        off = STX.carrier_line_db(self.tx.a0(seconds=0.1), self.a.carrier)
        st = self.tx.status()
        res.update(back_after_s=t_back, fs=fs1["line"], on_db=on, after_db=off, state=st["line"])
        self.wd = int(fs1["wdog_resets"])
        ok = (fs1.get("boot") == "wdog" and int(fs1["wdog_resets"]) == int(fs0["wdog_resets"]) + 1
              and (int(fs1["last_stage"], 16) & 0xFFFF) == 0x7F)
        if not self.a.dry:
            ok = ok and on >= ON_DB and off <= on - ON_OVER_OFF_DB
        res["ok"] = bool(ok)
        print(f" back after {t_back:.1f} s, boot={fs1.get('boot')}, last_stage={fs1.get('last_stage')}, "
              f"stuck_exc={fs1.get('stuck_exc')} stuck_pc={fs1.get('stuck_pc')}"
              + ("" if self.a.dry else f", carrier {on:.1f} dB -> {off:.1f} dB") + ("   OK" if ok else "   ** FAILED **"))

    def hold(self, keep_alive):
        name = "keep_alive" if keep_alive else "cw_hold"
        res = dict(ok=False)
        self.log["failsafe"][name] = res
        fs0 = self.fs_or_stop()
        if fs0 is None:
            return
        print(f"  {'D' if keep_alive else 'C'}) hold time 3 s, carrier on, "
              + ("a byte every second for 6 s ..." if keep_alive else "host silent for 5 s ..."), end="", flush=True)
        try:
            self.tx.ask("fs hold 3", r"^# fs ")
            self.tx.tone(self.a.carrier)
            if keep_alive:
                for _ in range(6):
                    time.sleep(1.0)
                    self.tx.send("ka")
            else:
                time.sleep(5.0)
            time.sleep(0.2)
            self.tx.dev.reset_input_buffer()    # the firmware's '# failsafe: cw off' line
            fs1 = self.tx.fs()
            db = STX.carrier_line_db(self.tx.a0(seconds=0.1), self.a.carrier)
            trips = int(fs1["trips"]) - int(fs0["trips"])
            ok = trips == (0 if keep_alive else 1)
            if not self.a.dry:
                ok = ok and ((db >= ON_DB) if keep_alive else (db < ON_DB))
            res.update(ok=bool(ok), trips=trips, line_db=db, fs=fs1["line"])
            print(f" switched off by the firmware: {trips}" + ("" if self.a.dry else f", carrier line {db:.1f} dB")
                  + ("   OK" if ok else "   ** FAILED **"))
        finally:
            self.tx.off()
            self.tx.ask("fs hold 60", r"^# fs ")

    def failsafe(self):
        print("\nFail-safe tests")
        self.hang(False)
        self.hang(True)
        self.hold(False)
        self.hold(True)
        bad = [k for k, v in self.log["failsafe"].items() if not v.get("ok")]
        for k in bad:
            self.log["failures"].append(dict(step=f"failsafe {k}", n=self.log["steps"], error=str(self.log["failsafe"][k])[:200],
                                             t=time.time()))


def run(a):
    b = Bench(a)
    path = os.path.join(HERE, "data_maxtouch", f"trxswitch_{a.label}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
    if not a.dry and not a.yes:
        input("REAL carrier on E+ (1 mA limit as wired). Electrode OFF the body, EVK may stay unplugged.\n"
              "Press Enter to start (Ctrl+C to leave) ")
    t0 = time.time()
    try:
        print(f"Transmitter: {b.open()}")
        fs = b.tx.fs()
        b.log["firmware"], b.log["fs_start"] = b.info, fs and fs["line"]
        b.wd = int(fs["wdog_resets"]) if fs else None
        print(f"  {fs['line'] if fs else 'no fs command: firmware older than rev 2 (no watchdog, no cw hold)'}")
        print(f"  mode: {'DRY rehearsal (nothing reaches E+)' if a.dry else 'REAL carrier'}, {a.carrier:g} Hz")
        for name, n, fn in (("quick", a.quick, b.quick), ("full", a.cycles, b.full)):
            if n:
                print(f"\n{n} {name} cycles", flush=True)
            for k in range(n):
                fn(k)
                if (k + 1) % 10 == 0 or k + 1 == n:
                    print(f"  {name} {k + 1}/{n}: {b.log['steps']} steps, {len(b.log['failures'])} failures, "
                          f"{len(b.log['resets'])} watchdog resets, {time.time() - t0:.0f} s", flush=True)
                json.dump(b.log, open(path, "w"), indent=1)
        if a.failsafe:
            b.failsafe()
        fs = b.tx.fs()
        b.log["fs_end"] = fs and fs["line"]
    except KeyboardInterrupt:
        print("\nstopped (Ctrl+C)")
        b.log["stopped"] = True
    finally:
        b.close()
        b.log["seconds"] = time.time() - t0
        b.log["step_seconds"] = {k: [float(np.median(v)), float(np.max(v)), len(v)] for k, v in b.t_step.items()}
        json.dump(b.log, open(path, "w"), indent=1)
    L = b.log
    print(f"\nSummary ({L['seconds']:.0f} s): {L['steps']} checked steps, {len(L['failures'])} failures, "
          f"{len(L['resets'])} watchdog resets during the cycles")
    for k in ("off", "on"):
        v = L["levels"][k]
        if v and not a.dry:
            print(f"  carrier line with the output {k:3s}: median {np.median(v):5.1f} dB, min {np.min(v):5.1f}, "
                  f"max {np.max(v):5.1f}  ({len(v)} streams)")
    for k, (med, mx, n) in L["step_seconds"].items():
        print(f"  {k:22s} {n:5d} x   median {med * 1e3:7.1f} ms   slowest {mx * 1e3:7.1f} ms")
    for f in L["failures"][:20]:
        print(f"  FAILED {f['step']}: {f['error']}" + (f"\n         {f['fs']}" if f.get("fs") else ""))
    print(f"log: {os.path.relpath(path, HERE)}")
    print("RESULT: " + ("PASS" if not L["failures"] and not L.get("stopped") else "FAIL"))
    return 0 if not L["failures"] and not L.get("stopped") else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--quick", type=int, default=100, help="fast RX / TX / RX cycles (default 100)")
    ap.add_argument("--cycles", type=int, default=20, help="full cycles with schedules (default 20)")
    ap.add_argument("--no-failsafe", dest="failsafe", action="store_false", help="skip the watchdog / cw-hold tests")
    ap.add_argument("--dry", action="store_true", help="rehearsal: firmware 'dry 1', nothing reaches E+")
    ap.add_argument("--carrier", type=float, default=159585.0)
    ap.add_argument("--label", default="bench")
    ap.add_argument("--yes", action="store_true", help="do not wait for Enter before a real run")
    STX.add_hw_args(ap)
    a = ap.parse_args()
    sys.exit(run(a))


if __name__ == "__main__":
    main()
