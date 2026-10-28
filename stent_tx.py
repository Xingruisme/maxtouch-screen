#!/usr/bin/env python3
"""
Host driver for the stent transmitter: Teensy 4.0 + AD9833 running the ORIGINAL TeensyTRX firmware
(../transceiver/firmware/TeensyTRX/TeensyTRX.ino, restored 2026-10-22). Frames and probe schedules come from
stent_frames.py, i.e. the same bits the TeensyTRX sent in the CapTIvate tests.

   python stent_tx.py id                                    # is TeensyTRX there? (first line of its '?' help)
   python stent_tx.py tone 163000                           # continuous carrier ('cw'), until 'python stent_tx.py off'
   python stent_tx.py selftest 163000                       # A0 (E+) stream with the carrier off / on / off
   python stent_tx.py off
   python stent_tx.py status                                # 'sstat' of the last schedule
   python stent_tx.py ber --carrier 168000 --bps 20 --frames 4      # OOK (add --zero-hz 171000 for FSK)
   python stent_tx.py probe --carriers log:50e3:1.5e6:2.5% --block 0.5 --reps 2
   options for every command: --wave square|sine (default square) --port /dev/ttyACM0

TeensyTRX commands used (see the firmware header): wave, cw, off, psk 0, sclr, sseg <step_us> <reps> <n>,
sw <words>, sinfo (checksum), sgo <delay_ms>, sstat, mono/x (A0 stream for the self-test), ?.
Pins and clock are fixed in the firmware: AD9833 FSYNC 10 / SDATA 11 / SCLK 13, MCLK 25 MHz, pin 2 = TX/RX switch.
A schedule is one sseg on the common step of all segment lengths; every step is 'off -> FREQ0 -> on' in the firmware
(~20 us gap), word 0 = 0 Hz = the OOK 'off' level, the same as cap_ook_tx.py. The firmware turns the output off when
the schedule ends; every command here that starts the output also turns it off on exit, error or Ctrl+C.

Normally maxtouch_stent_rx.py --tx drives this itself (it knows the exact start time then).
Needs: pip install pyserial numpy
"""

from __future__ import annotations

import argparse
import math
import re
import signal
import sys
import time

import numpy as np

import stent_frames as SF

TEENSY_VID = 0x16C0
MCLK = 25_000_000.0                             # TeensyTRX.ino MCLK (fixed in the firmware)
MAXW, MAXSEG = 4096, 16                         # TeensyTRX.ino schedule limits
GO_DELAY_MS = 500                               # 'sgo' arms the schedule this long ahead (USB latency is ~1 ms)


def fword(hz):
    return int(hz * 268435456.0 / MCLK + 0.5) & 0x0FFFFFFF


# ================================================================ schedules
def ber_segments(carrier, bps, frames, zero_hz=None, payload=127, prbs=7):
    """frames x (preamble + SFD + payload + gap) as run-length segments; 1 = carrier, 0 = off (OOK) or zero_hz (FSK)"""
    bits = SF.frame_bits(payload, prbs)[0] * frames
    bit_us = int(round(1e6 / bps))
    segs = []
    for b in bits:
        hz = carrier if b else (zero_hz or 0.0)
        if segs and segs[-1][0] == hz:
            segs[-1][1] += bit_us
        else:
            segs.append([hz, bit_us])
    return [tuple(s) for s in segs], dict(bit_us=bit_us, n_bits=len(bits), frames=frames,
                                          total_s=len(bits) * bit_us / 1e6)


def probe_segments(carriers, block_s, reps, marker_hz=SF.MARKER_HZ, marker_s=SF.MARKER_S):
    """the transmitter --probe schedule: marker on/off, then per carrier `reps` x (block on, block off)"""
    b = int(round(block_s * 1e6)); m = int(round(marker_s * 1e6))
    segs = [(marker_hz, m), (0.0, m)]
    for f in carriers:
        segs += [(float(f), b), (0.0, b)] * reps
    return segs, dict(total_s=sum(s[1] for s in segs) / 1e6)


def to_schedule(segs):
    """run-length [(hz, us)] -> TeensyTRX schedule [(step_us, reps, words)]: one sseg on the common step"""
    step = 0
    for _, us in segs:
        step = math.gcd(step, int(us))
    words = [fword(hz) for hz, us in segs for _ in range(int(us) // step)]
    if len(words) > MAXW:
        raise ValueError(f"schedule needs {len(words)} steps of {step} us, TeensyTRX holds {MAXW} - "
                         "use segment lengths with a larger common step")
    return [(step, 1, words)]


def checksum(sched):
    """the same sum as TeensyTRX sinfo (schedSum)"""
    cs = 0
    for st, reps, w in sched:
        cs = (cs + (st ^ reps ^ len(w))) & 0xFFFFFFFF
    for _, _, w in sched:
        cs = (cs + sum(w)) & 0xFFFFFFFF
    return cs


# ================================================================ link
class StentTX:
    def __init__(self, port=None, timeout=1.0):
        import serial
        from serial.tools import list_ports
        if port is None:
            cands = [p.device for p in list_ports.comports() if p.vid == TEENSY_VID]
            if not cands:
                raise RuntimeError("no Teensy serial port (TeensyTRX with USB Type 'Serial')")
            port = cands[0]
        self.port = port
        self.dev = serial.Serial(port, 921600, timeout=0.05, write_timeout=2.0)
        self.timeout = timeout
        self.send("x")                          # stop an A0 stream left running: the firmware is silent while streaming
        time.sleep(0.2)
        self.dev.reset_input_buffer()

    def close(self):
        try:
            self.off()
        except Exception as e:
            print(f"  ! could not confirm the carrier is off ({e}) - run: python stent_tx.py off")
        try:
            self.dev.close()
        except Exception:                       # the Teensy may have dropped off USB together with the EVK
            pass

    def send(self, line):
        self.dev.write((line + "\n").encode())

    def ask(self, line, want, timeout=None):
        """send one command, return the first reply line matching `want` (firmware refusals raise)"""
        self.send(line)
        end = time.time() + (timeout or self.timeout)
        buf = b""
        while time.time() < end:
            buf += self.dev.read(256)
            while b"\n" in buf:
                ln, buf = buf.split(b"\n", 1)
                s = ln.decode(errors="replace").strip()
                if re.match(r"# (sgo:|sseg: refused|cw: bad|busy)", s):
                    raise RuntimeError(f"Teensy: {s} (command {line[:40]!r})")
                if re.search(want, s):
                    return s
        raise TimeoutError(f"no reply to {line[:40]!r} (wanted {want!r}) - not TeensyTRX, or port {self.port}?")

    def info(self):
        s = self.ask("?", r"^TeensyTRX")
        time.sleep(0.2)
        self.dev.reset_input_buffer()           # rest of the help text
        return {"line": s}

    def configure(self, wave=None):
        if wave:
            r = self.ask(f"wave {wave}", r"^# wave")
            if wave not in r:
                raise RuntimeError(f"wave {wave} not taken: {r!r}")
        return self.info()

    def upload(self, segs):
        """send the schedule, then check the firmware's checksum and totals"""
        return self.upload_schedule(to_schedule(segs))

    def upload_schedule(self, sched):
        """[(step_us, reps, words)] as given: a segment can repeat (one 170-bit frame x 40 reps = 6800 bits in 170
        words, where to_schedule's single unrolled segment stops at MAXW = 4096)"""
        self.off()
        self.ask("psk 0", r"^# psk")            # the firmware keeps a DBPSK overlay from an earlier session
        self.ask("sclr", r"^# sched cleared")
        for st, reps, words in sched:
            self.send(f"sseg {st} {reps} {len(words)}")
            for i in range(0, len(words), 24):
                self.send("sw " + " ".join(map(str, words[i:i + 24])))
        s = self.ask("sinfo", r"^# sched segs=", timeout=5.0)
        got = dict((k, int(v)) for k, v in re.findall(r"(\w+)=(\d+)", s))
        want = dict(segs=len(sched), words=sum(len(w) for _, _, w in sched),
                    total_us=sum(st * reps * len(w) for st, reps, w in sched), checksum=checksum(sched))
        bad = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
        if bad:
            raise RuntimeError(f"upload check failed (Teensy, sent): {bad}")
        return got

    def go(self, delay_ms=GO_DELAY_MS):
        """start the uploaded schedule; returns the host time of its first step (+-~1 ms)"""
        t_cmd = time.time()
        r = self.ask(f"sgo {int(delay_ms)}", r"^# sweep armed in")
        return t_cmd + delay_ms / 1e3 + 0.0005, r

    def status(self):
        s = self.ask("sstat", r"^# sweep state=")
        return dict(re.findall(r"(\w+)=(\S+)", s)) | {"line": s}

    def wait_done(self, timeout):
        end = time.time() + timeout
        while time.time() < end:
            if self.status()["state"] in ("done", "stop", "idle"):
                return True
            time.sleep(0.5)
        return False

    def tone(self, hz):
        return self.ask(f"cw {hz}", r"^# cw")

    def off(self):
        return self.ask("off", r"^# off")

    def a0(self, sps=1000000, seconds=0.2):
        """A0 (E+ through the TX/RX path) via the 'mono' stream -> 12-bit ADC counts.
        Stream format: 2 bytes per sample, 0x80 | high 7 bits, then low 7 bits, of adc12 << 2."""
        need = int(2 * sps * seconds) + 40000
        self.dev.reset_input_buffer()
        self.send(f"mono {sps}")
        buf, end = b"", time.time() + seconds + 5
        while len(buf) < need and time.time() < end:
            buf += self.dev.read(1 << 16)
        self.send("x")
        time.sleep(0.3)
        self.dev.reset_input_buffer()           # '# mono stopped ...'
        a = np.frombuffer(buf[20000:], np.uint8)            # skip the first ~10 ms
        j = next(i for i in range(len(a) - 1) if a[i] & 0x80 and not a[i + 1] & 0x80)
        pr = a[j:j + (len(a) - j) // 2 * 2].reshape(-1, 2)
        ok = (pr[:, 0] & 0x80 != 0) & (pr[:, 1] & 0x80 == 0)
        return ((((pr[:, 0] & 0x7F).astype(np.uint16) << 7) | pr[:, 1]) >> 2)[ok].astype(float)


def add_hw_args(ap):
    ap.add_argument("--port", help="Teensy serial port (default: first VID 16C0 port)")
    ap.add_argument("--wave", default="square", choices=("square", "sine"))


def open_configured(a):
    tx = StentTX(a.port)
    info = tx.configure(wave=a.wave)
    return tx, info


def selftest(tx, hz):
    """Does a carrier really come out? Stream A0 (the TeensyTRX electrode input) with the carrier off, on, off and
    look for the carrier line. 2026-10-22, restored TeensyTRX, square: ~2 Vpp and the carrier 80-85 dB over the
    spectrum median at 20-400 kHz, 3-7 dB with the output off. Above 500 kHz the 1 MS/s stream shows it aliased."""
    sps = 1e6
    alias = abs(hz - round(hz / sps) * sps)

    def show(name, v):
        sp = np.abs(np.fft.rfft((v - v.mean()) * np.hanning(len(v))))
        fr = np.fft.rfftfreq(len(v), 1 / sps)
        j = int(np.argmin(np.abs(fr - alias)))
        db = 20 * np.log10(sp[max(j - 3, 0):j + 4].max() / np.median(sp))
        print(f"  {name:9s} {v.min():5.0f}..{v.max():5.0f} ({(v.max() - v.min()) * 3.3 / 4095:4.2f} Vpp)  "
              f"std {v.std():6.1f}   line at {alias / 1e3:g} kHz: {db:5.1f} dB over the median")
        return db

    print(tx.off())
    d0 = show("off", tx.a0())
    print(tx.tone(hz))
    d1 = show("ON", tx.a0())
    print(tx.off())
    d2 = show("off again", tx.a0())
    if d1 >= 30 and d1 > max(d0, d2) + 20:
        print(f"carrier present at {hz:g} Hz")
    else:
        print(f"no clear carrier at {hz:g} Hz on A0 - check the AD9833 wiring / power")


def main():
    ap = argparse.ArgumentParser(description="stent transmitter (Teensy 4.0 + AD9833, original TeensyTRX firmware)")
    ap.add_argument("cmd", choices=("id", "tone", "off", "status", "selftest", "ber", "probe"))
    ap.add_argument("hz", nargs="?", type=float, help="tone / selftest: carrier [Hz]")
    ap.add_argument("--carrier", type=float)
    ap.add_argument("--zero-hz", type=float, help="ber: FSK '0' frequency (default: OOK, carrier off)")
    ap.add_argument("--bps", type=float, default=10.0)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--carriers", default="default")
    ap.add_argument("--block", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=3)
    add_hw_args(ap)
    a = ap.parse_args()
    if a.cmd == "id":
        tx = StentTX(a.port); print(tx.info()["line"]); tx.dev.close(); return
    if a.cmd == "status":
        tx = StentTX(a.port); print(tx.status()["line"]); tx.dev.close(); return
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))     # kill -> SystemExit -> the finally below turns it off
    tx, info = open_configured(a)
    print(info["line"])
    if a.cmd == "tone":                         # deliberately left on: the user turns it off
        print(tx.tone(a.hz)); print("carrier on until 'python stent_tx.py off'"); tx.dev.close(); return
    try:
        if a.cmd == "off":
            print(tx.off()); return
        if a.cmd == "selftest":
            selftest(tx, a.hz or 163000); return
        if a.cmd == "ber":
            segs, plan = ber_segments(a.carrier, a.bps, a.frames, a.zero_hz)
        else:
            from maxtouch_stent_rx import parse_carriers
            segs, plan = probe_segments(parse_carriers(a.carriers), a.block, a.reps)
        print(f"uploading {len(segs)} segments ({plan['total_s']:.1f} s): {tx.upload(segs)}")
        t_first, r = tx.go()
        print(r)
        ok = tx.wait_done(plan["total_s"] + GO_DELAY_MS / 1e3 + 5)
        print(tx.status()["line"] if ok else "schedule did not finish in time")
    except KeyboardInterrupt:
        print("stopped (Ctrl+C)")
    finally:
        tx.close()


if __name__ == "__main__":
    main()
