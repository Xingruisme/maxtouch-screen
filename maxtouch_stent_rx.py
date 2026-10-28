#!/usr/bin/env python3
"""
Stent -> maXTouch touchscreen receiver: carrier frequency probe and bit error rate (BER).
The maXTouch version of ../captivate/cap_ook_rx.py. The TeensyTRX transmitter runs on another computer as before.

   python maxtouch_stent_rx.py probe --label p1                     # pair with: cap_ook_tx.py --probe
   python maxtouch_stent_rx.py probe --carriers log:50e3:1.5e6:2.5% --block 0.5 --reps 1 --label fine
   python maxtouch_stent_rx.py probe --tx --patch 12,30,4 --label p12          # stent anywhere: 4x4 square at X12-15, Y30-33
   python maxtouch_stent_rx.py ber --carrier 312e3 --bps 20 --label b20      # pair with: cap_ook_tx.py --carrier 312e3 --bps 20
   python maxtouch_stent_rx.py onoff                                # carrier 163 kHz OFF 10 s / ON 10 s / OFF 10 s (--tx implied)
   python maxtouch_stent_rx.py onoff --patch 11,29,4 --carrier 500e3 --phase-s 15
   python maxtouch_stent_rx.py reanalyse data_maxtouch/stent_ber_b20_<time>
   python maxtouch_stent_rx.py summary                              # BER vs bps table + figure over all ber runs

Order (same as cap_ook_rx): close maxtouch_map, put the stent in place with the transmitter OFF and start this first.
It sets up the chip, records --lead seconds of noise, then prints  >>> START TX NOW <<< . Then start cap_ook_tx.py with the
same settings (ber: --carrier/--bps/--payload/--prbs; probe: --probe-carriers/--probe-block/--probe-reps).
The probe prints the exact carrier list to give the transmitter. Recording stops after --seconds (Ctrl+C stops early).

What is sampled: page 0 of the T37 mutual-delta diagnostics = X line 0, Y lines 0-63. Re-issuing the T6 'deltas'
command waits for the next acquisition and refreshes page 0, so every sample is a fresh scan (no repeated frames).
Measured 2026-10-20 with T72 off: reading <= 28 nodes (one HID transfer) = 137 samples/s, locked to the 7.31 ms scans;
reading all 64 = ~64 samples/s at the read-loop pace (the chip seems not to scan while serving the long read).
Any other X line costs one more acquisition per page (see maxtouch_map.py), so the stent signal must reach X line 0's
nodes: put the stent over/next to X line 0 (top row of the maxtouch_map landscape view). The probe reads all 64 to
show which Y nodes respond; then use --y0/--ny (<= 28) around them for the ber runs to get the full 137 samples/s.
--patch X0,Y0,SIZE (or X0,Y0,W,H) reads a fixed square anywhere instead, like maxtouch_map.py's patch mode: pages are
stepped from 0 up to the square's last row, one acquisition per step, so a sample takes ~1.1 x (X0 + W) scans
(4x4 measured 2026-10-21: 26.7 / 11.6 / 3.9 samples/s at X 0 / 5 / 20). Probe blocks default to 3 s then (give
--block 5 for squares far down the panel); too slow for ber above ~1-2 bps.
The nodes of one sample come from successive scans (a few tens of ms apart), which the 0.5-1 s probe blocks absorb.

Chip settings changed only while recording (RAM only, never saved to NVM, restored at the end, also on Ctrl+C):
  T72 noise suppression off  - it would hop the acquisition frequency away from the carrier, which looks like noise
  T100 touch reports off (RPTEN bit only; ENABLE must stay or acquisition stops) - no ghost-touch clicks here
  T6 calibrate once with the transmitter off, so all deltas start at 0
--keep-noise-suppression / --keep-touch leave those alone.

Analysis reuses ../captivate/cap_ook_rx.py unchanged (sample numbering, features level/energy/logenergy/absdev,
frame-grid search, decision-directed thresholds, BER, probe alignment). stent_frames.py stands in for the
transmitter module (frame/probe schedule recovered from the CapTIvate recordings; re-decodes all 8 of them exactly).
ber: blind BER per responding node, plus 'combined' = mean of the z-scored absdev of the --combine best nodes
(ranked by energy increase after START TX, chosen before decoding) - like the CapTIvate two-electrode receiver.
The combined result is the headline number; the per-node table is for reference (picking the best of many nodes
after decoding would flatter the BER).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stent_frames as TX

sys.modules["cap_ook_tx"] = TX                  # cap_ook_rx imports the transmitter module by this name
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "captivate"))
import cap_ook_rx as RX

from maxtouch_hid import DELTAS_MODE, PAGE_UP, MaxTouchHid

DATADIR = os.path.join(HERE, "data_maxtouch")
T_NOM = 0.010                                   # nominal acquisition period (free-run, measured ~10 ms)
T6_CALIBRATE, T6_DIAGNOSTIC = 2, 5
T100_RPTEN = 0x02
NODES_PER_PAGE = 64


def parse_patch(s):
    """'X0,Y0,SIZE' or 'X0,Y0,W,H' -> [x0, y0, w, h]"""
    v = [int(p) for p in s.split(",")]
    if len(v) == 3:
        v.append(v[2])
    if len(v) != 4 or min(v) < 0 or v[2] < 1 or v[3] < 1:
        raise argparse.ArgumentTypeError("--patch needs X0,Y0,SIZE or X0,Y0,W,H")
    return v


def node_names(meta):
    """label of each recorded node: 'Y12' on X line 0 (page-0 mode), 'X12Y30' in patch mode"""
    if meta.get("patch"):
        x0, y0, w, h = meta["patch"]
        return [f"X{x0 + i}Y{y0 + j}" for i in range(w) for j in range(h)]
    return [f"Y{meta['y0'] + i}" for i in range(meta["ny"])]


# ================================================================ chip
class Chip:
    """maXTouch control link for the stent tests: setup/restore and page-0 (or --patch) delta sampling."""

    def __init__(self, a):
        self.m = MaxTouchHid()
        self._load_checked()
        idn = self.m.id_info
        self.ny_matrix = idn.matrix_y
        self.t6 = self.m.object_address(6)
        self.t37 = self.m.object_address(37)
        self.t72 = self.m.object_address(72)
        self.t100 = self.m.object_address(100)
        self.patch = a.patch
        if self.patch:
            x0, y0, w, h = self.patch
            if x0 + w > idn.matrix_x or y0 + h > idn.matrix_y:
                raise ValueError(f"--patch {x0},{y0},{w},{h} is outside the {idn.matrix_x}x{idn.matrix_y} matrix")
            page_size = self.m.object_size(37) - 2
            self.wanted = {}                    # page -> [(offset in page, node index)], T37 is X-major
            for i in range(w):
                for j in range(h):
                    off = ((x0 + i) * idn.matrix_y + y0 + j) * 2
                    self.wanted.setdefault(off // page_size, []).append((off % page_size, i * h + j))
            self.y0, self.ny = y0, w * h
        else:
            self.y0, self.ny = a.y0, min(a.ny, NODES_PER_PAGE - a.y0)
        self.single = self.ny <= 28             # one 58-byte HID transfer, no header check
        self.names = node_names(dict(patch=self.patch, y0=self.y0, ny=self.ny))
        self.orig = {}
        self.a = a

    def _load_checked(self, attempts=4):
        """Read the info block + object table, refusing a garbled one before anything is written. Once the first
        read after opening returned a stale T100 touch message left in the EVK's HID buffer (38 90 8c 02 ...) as
        the 'info block', and setup then wrote to addresses taken from that table (2026-10-20). So: drain pending
        reports first, and require two identical reads that contain the objects this program uses."""
        need = {5, 6, 37, 44, 72, 100}
        prev = None
        for _ in range(attempts):
            while self.m._h.read(64):          # stale reports from an earlier session
                pass
            try:
                self.m.load_info()
            except Exception:
                continue
            table = [(o.type_id, o.address, o.size) for o in self.m.objects]
            types = {o.type_id for o in self.m.objects}
            if need <= types and all(addr < 0x4000 for _, addr, _ in table) and table == prev:
                return
            prev = table
        raise RuntimeError("could not read a consistent maXTouch object table - unplug/replug the EVK and retry")

    def setup(self):
        if not self.a.keep_noise_suppression and self.t72 is not None:
            self.orig["T72"] = self.m.read_register(self.t72, 1)[0]
            self.m.write_register(self.t72, bytes([0]))
        if not self.a.keep_touch and self.t100 is not None:
            # clear only RPTEN (bit 1, report touches). CTRL = 0 also cleared ENABLE and stopped the touchscreen
            # acquisition itself: every delta read 0 (measured 2026-10-20)
            self.orig["T100"] = self.m.read_register(self.t100, 1)[0]
            self.m.write_register(self.t100, bytes([self.orig["T100"] & ~T100_RPTEN & 0xFF]))
        self.m.write_register(self.t6 + T6_CALIBRATE, bytes([1]))
        time.sleep(1.5)
        return {k: int(v) for k, v in self.orig.items()}

    def restore(self):
        for name, addr in (("T100", self.t100), ("T72", self.t72)):
            if name in self.orig:
                try:
                    self.m.write_register(addr, bytes([self.orig[name]]))
                except Exception as e:
                    print(f"  ! could not restore {name} CTRL={self.orig[name]}: {e} - unplug/replug the EVK to reset it")
        self.m.close()

    def _step(self, cmd):
        """T6 diagnostic command (deltas = back to page 0, or page up); waits for the next scan"""
        diag = self.t6 + T6_DIAGNOSTIC
        self.m.write_register(diag, bytes([cmd]))
        deadline = time.perf_counter() + 0.5
        while self.m.read_register(diag, 1)[0] != 0:
            if time.perf_counter() > deadline:
                raise TimeoutError("T6 diagnostic command timed out")

    def _sample_patch(self):
        """pages can only be stepped up from page 0, so walk to the square's last page, reading the wanted bytes"""
        vals = np.zeros(self.ny, np.int16)
        for page in range(max(self.wanted) + 1):
            self._step(DELTAS_MODE if page == 0 else PAGE_UP)
            if page in self.wanted:
                last = max(o for o, _ in self.wanted[page]) + 2
                raw = self.m.read_register(self.t37, 2 + last)
                if len(raw) != 2 + last or raw[0] != DELTAS_MODE or raw[1] != page:
                    raise RuntimeError(f"unexpected T37 reply ({len(raw)} bytes, header {bytes(raw[:2]).hex()}, page {page})")
                for o, k in self.wanted[page]:
                    vals[k] = int.from_bytes(bytes(raw[2 + o:4 + o]), "little", signed=True)
        return vals

    def sample(self):
        """one fresh acquisition: T6 deltas command (waits for the next scan) + read the chosen page-0 nodes"""
        if self.patch:
            return self._sample_patch()
        self._step(DELTAS_MODE)
        if self.single:
            raw = self.m.read_register(self.t37 + 2 + 2 * self.y0, 2 * self.ny)
        else:
            page = self.m.read_register(self.t37, 2 + 2 * (self.y0 + self.ny))
            if page[0] != DELTAS_MODE or page[1] != 0:
                raise RuntimeError(f"unexpected T37 header mode=0x{page[0]:02X} page={page[1]}")
            raw = page[2 + 2 * self.y0:]
        if len(raw) != 2 * self.ny:
            # a reply meant for another command (e.g. maxtouch_map.py opened mid-run, 2026-10-21) - one bad sample
            # used to crash the save and lose the whole recording
            raise RuntimeError(f"T37 read returned {len(raw)} bytes, expected {2 * self.ny}")
        return np.frombuffer(bytes(raw), "<i2")


def record(chip, a, seconds, start_tx=None):
    """Record page-0 samples: --lead s of noise, then START TX, until `seconds` after START TX or Ctrl+C.
    start_tx: callable that starts the transmitter itself at that moment and returns its first-segment time."""
    t_list, x_list = [], []
    t_start = time.time()
    t_prompt = t_first = None
    perf0, unix0 = time.perf_counter(), time.time()
    last_print, n_print = time.time(), 0
    errors = 0
    where = (f"X{chip.patch[0]}-{chip.patch[0] + chip.patch[2] - 1} Y{chip.patch[1]}-{chip.patch[1] + chip.patch[3] - 1}"
             if chip.patch else f"X0 Y{chip.y0}-{chip.y0 + chip.ny - 1}")
    print(f"Recording nodes {where} ({chip.ny} nodes). Transmitter must be OFF now.")
    try:
        while True:
            now = time.time()
            if t_prompt is None and now - t_start >= a.lead:
                t_prompt = now
                if start_tx:
                    t_first = start_tx()
                    print(f"\n  transmitter started (first segment at +{t_first - now:.3f} s), recording {seconds:.0f} s",
                          flush=True)
                else:
                    print("\n" + "=" * 60 + "\n  >>> START TX NOW <<<   (recording for "
                          f"{seconds:.0f} s more, Ctrl+C to stop early)\n" + "=" * 60, flush=True)
            if t_prompt is not None and now - t_prompt >= seconds:
                break
            try:
                v = chip.sample()
            except Exception as e:              # a dropped HID transfer: count it and carry on
                errors += 1
                if errors > 50:                 # the EVK fell off USB (happened 2026-10-20): keep what was recorded
                    print(f"\n  ! maXTouch link lost ({e}) - stopping, the samples so far are saved and analysed")
                    break
                continue
            t_list.append(unix0 + (time.perf_counter() - perf0))
            x_list.append(v)
            if now - last_print >= 1.0:
                X = np.array(x_list[n_print:], float)
                rate = len(X) / (now - last_print)
                if len(X) > 3:
                    d = np.diff(X, axis=0).std(axis=0) / math.sqrt(2)
                    j = int(np.argmax(d))
                    print(f"  {now - t_start:6.1f} s  {rate:5.1f} samples/s   most active node {chip.names[j]} "
                          f"(rms {d[j]:.1f}, median node {np.median(d):.1f} counts)", flush=True)
                last_print, n_print = now, len(x_list)
    except KeyboardInterrupt:
        print("  stopped (Ctrl+C)")
    # keep only full-length samples (belt and braces: sample() already rejects short replies)
    from collections import Counter
    n = Counter(len(v) for v in x_list).most_common(1)[0][0] if x_list else 0
    keep = [i for i, v in enumerate(x_list) if len(v) == n]
    return (np.array([t_list[i] for i in keep]), np.array([x_list[i] for i in keep], np.int16).reshape(len(keep), -1),
            t_prompt, t_first, errors)


# ================================================================ helpers
def parse_carriers(s):
    """'100e3,200e3' | 'log:start:stop:step%' | 'lin:start:stop:step' | 'default' (the transmitter's --probe list)"""
    if s in (None, "", "default"):
        return list(TX.PROBE_CARRIERS)
    if s.startswith("log:"):
        _, f0, f1, st = s.split(":")
        r = 1 + float(st.rstrip("%")) / 100
        n = int(math.floor(math.log(float(f1) / float(f0)) / math.log(r))) + 1
        return [round(float(f0) * r ** i, 1) for i in range(n)]
    if s.startswith("lin:"):
        _, f0, f1, st = s.split(":")
        return [round(float(v), 1) for v in np.arange(float(f0), float(f1) + float(st) / 2, float(st))]
    return [float(v) for v in s.split(",")]


def link_from(a, probe):
    return argparse.Namespace(carrier=a.carrier if not probe else None, zero_hz=None, bps=a.bps, payload=a.payload,
                              prbs=a.prbs, wave="square", probe_carriers=parse_carriers(a.carriers) if probe else None,
                              probe_block=a.block, probe_reps=a.reps)


def scan_period(t):
    """Acquisition period from the arrival times, to ~1e-5. Arrivals are scan times plus a short positive delay, so at
    the true period T every t mod T lands at nearly the same phase: |mean exp(2 pi i t / T)| is high, and over
    thousands of samples it decoheres for any other T - except whole fractions T/2, T/3 (multiples of T are multiples
    of T/k too). Twice the period does not fit unless every sample skipped exactly one scan.
    1) Coarse: search 2 ms to 1.8 x the 5th-percentile interval on the first ~15 s and take the LARGEST period that
       fits within 80 % of the best fit.
    2) Fine: refine on the whole record around it.
    Simple interval statistics fail here (measured 2026-10-20): reading 64 nodes takes longer than one 7.3 ms scan, so
    samples land every 2nd or 3rd scan, and even reading 28 nodes the ~2 ms USB polling splits intervals into ~6 and
    ~8 ms groups. The numbering below needs the period to ~0.3 % or better (the CapTIvate period was crystal-exact;
    the maXTouch free-run period is not a set value): a 0.5 % error misnumbered the synthetic test and gave 5 % BER."""
    t = np.asarray(t, float) - t[0]
    dt = np.diff(t)
    dt = dt[(dt > 0) & (dt < 0.1)]
    if len(dt) < 10:
        return T_NOM

    def conc(tt, Ts):
        Ts = np.asarray(Ts)
        return np.concatenate([np.abs(np.exp(2j * np.pi * tt[None, :] / Ts[i:i + 200, None]).mean(axis=1))
                               for i in range(0, len(Ts), 200)])

    tc = t[t <= min(t[-1], 15.0)]
    span_c = max(tc[-1], 1.0)
    # arrivals come on the 1 ms USB frame grid (28 nodes: intervals of 6 or 8 ms), so every period dividing 2 ms fits
    # them perfectly - with the old 2 ms lower bound that picked T = 2.000 ms instead of 7.43 (20 bps run 2026-10-22).
    # A sample never arrives faster than the scan, so the period cannot be far below the typical interval.
    lo, hi = max(0.002, 0.25 * np.percentile(dt, 5), 0.6 * np.median(dt)), 1.8 * np.percentile(dt, 5)
    step_c = lo * lo / span_c / 4               # phase over the coarse span changes by <= 1/4 cycle per step
    grid = np.arange(lo, hi, step_c)
    c = conc(tc, grid)
    T_edge = float(grid[c >= 0.8 * c.max()].max())
    near = np.abs(grid - T_edge) <= 0.003 * T_edge     # that peak's own maximum, not its upper edge (149 ppm off)
    T0 = float(grid[near][int(np.argmax(c[near]))])
    span = max(t[-1], 1.0)
    best = T0
    for w, st in ((2 * step_c, T0 * T0 / span / 4), (None, T0 * T0 / span / 80)):
        if w is None:
            w = 2 * T0 * T0 / span / 4
        g = np.arange(best - w, best + w, st)
        best = float(g[int(np.argmax(conc(t, g)))])
    return best


def number(t, T, W=301):
    """Scan number of each sample: round (t / T - typical delay phase), the delay phase tracked as a running circular
    mean over W samples (follows slow oscillator drift). Not cap_ook_rx.regularise: that numbers each sample from the
    least-late recent one and needs every delay < 0.85 T, else one late sample shifts all later numbers by one. Here a
    sample's delay after its scan is ~2-8 ms (poll + read) on a ~10 ms period, and a wrong number stays local."""
    tr = np.asarray(t, float) - t[0]
    ph = tr / T
    z = np.exp(2j * np.pi * ph)
    k = np.ones(min(W, len(z))) / min(W, len(z))
    zm = np.convolve(z, k, mode="same")
    # typical fractional phase of the arrivals around each sample, unwrapped: near +-0.5 it would otherwise flip by
    # a whole period and shift a stretch of numbers by one
    phi = np.unwrap(np.angle(zm)) / (2 * np.pi)
    dev = (ph - phi + 0.5) % 1.0 - 0.5          # each arrival's phase relative to the typical one
    low = float(np.percentile(dev, 2))          # least-late arrivals: arrivals are only ever late, never early
    x = ph - phi - low
    n = np.floor(x + 0.15).astype(int)          # tolerates delays up to 0.85 T past the least-late ones
    clashes = 0
    for i in range(1, len(n)):
        # two samples can't be the same scan. Arrivals are only late, so the earlier one of the pair is the one that
        # came in > 0.85 T late and was counted a scan too far: move it back if that slot is free.
        if n[i] <= n[i - 1]:
            clashes += 1
            if i == 1 or n[i] - 1 > n[i - 2]:
                n[i - 1] = n[i] - 1
            else:
                n[i] = n[i - 1] + 1
    n -= n[0]
    r = tr - n * T
    t0 = float(t[0] + np.median(r) - (np.median(r) - r.min()))
    return n, float(t0), float(np.std(r - np.convolve(r, k, mode="same"))), clashes


def regular(t, X):
    """host times -> uniform sample grid.
    Scan-locked (reading <= 28 nodes, measured 7.31 ms scans, 0 clashes): number each sample's scan, interpolate the
    few missing ones. Not scan-locked: reading 64 nodes, samples came a steady 16.0 ms apart = the read loop, not a
    multiple of the 7.31 ms scan (the chip apparently cannot scan while serving the long reads). Numbering then makes
    no sense (201 clashes in 451 samples), so the samples are put on a uniform grid at the mean interval by their own
    timestamps (+-~2 ms, fine for 10-100 ms bits). Returns grid, period, grid start time, jitter, (missing, clashes)."""
    T = scan_period(t)
    n, t0, jit, clashes = number(t, T)
    if clashes <= 0.01 * len(t) and jit <= 0.3 * T:
        G = np.column_stack([RX.on_grid(n, X[:, i].astype(float))[0] for i in range(X.shape[1])])
        return G, T, t0, jit, (int(n[-1] + 1 - len(t)), clashes)
    dt = np.diff(t)
    Tm = float(np.mean(dt[(dt > 0.3 * np.median(dt)) & (dt < 1.6 * np.median(dt))]))
    tg = t[0] + np.arange(int((t[-1] - t[0]) / Tm) + 1) * Tm
    G = np.column_stack([np.interp(tg, t, X[:, i].astype(float)) for i in range(X.shape[1])])
    print(f"  samples not locked to a fixed scan period ({clashes} numbering clashes): placed on a {Tm * 1e3:.2f} ms "
          f"grid by timestamp")
    return G, Tm, float(t[0]), float(np.std(dt - Tm)), (0, -1)


def zscore(v):
    med = np.median(v)
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    return (v - med) / (iqr if iqr > 0 else (np.std(v) or 1.0))


def node_scores(G, T, t0, t_prompt):
    """energy increase [dB] per node after START TX vs the lead (transmitter off)"""
    s = int((t_prompt - t0) / T) if t_prompt else len(G) // 5
    s = int(np.clip(s, 5, len(G) - 5))
    e = np.diff(G, axis=0) ** 2
    pre = e[:s].mean(axis=0) + 1e-9
    post = e[s:].mean(axis=0) + 1e-9
    return 10 * np.log10(post / pre), s


def save(d, t, X, meta):
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "capture.npz"), t=t, X=X)
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)


def tx_command(a, probe):
    if probe:
        car = ",".join(f"{f:g}" for f in parse_carriers(a.carriers))
        return f"cap_ook_tx.py --probe --probe-carriers {car} --probe-block {a.block:g} --probe-reps {a.reps}"
    return f"cap_ook_tx.py --carrier {a.carrier:g} --bps {a.bps:g} --payload {a.payload} --prbs {a.prbs}"


# ================================================================ probe
def analyse_probe(t, X, meta):
    a = argparse.Namespace(**meta["args"])
    link = link_from(a, True)
    G, T, t0, jit, (missing, clashes) = regular(t, X)
    y0 = meta["y0"]
    names = node_names(meta)
    # Schedule start: only near the START TX prompt. The transmitter's 1 kHz start marker is invisible to the touch
    # sensor, and with a single responding carrier, shifting the schedule by one carrier slot fits just as well
    # (synthetic test: 500 kHz reported as 1 MHz). So the operator starts the transmitter right at the prompt and
    # the start is searched in [prompt + start_delay - start_tol, prompt + start_delay + start_tol].
    hint = tol = window = None
    if meta.get("t_first"):                     # --tx: the start is known to ~1 ms
        # a few samples of slack only: +-0.3 s let a run without any response align on noise (picked 0.1 s for a
        # known 0.5 s start and reported a nonsense level z +1582 with 0.5 s blocks, hand/finger run 2026-10-22)
        hint, tol = (meta["t_first"] - t0) / T, 3 * T + 0.02
        window = (meta["t_first"] - meta["t_prompt"] - tol, meta["t_first"] - meta["t_prompt"] + tol)
    elif meta.get("t_prompt"):
        hint = (meta["t_prompt"] - t0) / T + a.start_delay / T
        tol = a.start_tol
        window = (a.start_delay - tol, a.start_delay + tol)
        if tol >= 0.5 * link.probe_block * 2 * link.probe_reps and not meta.get("t_first"):
            print(f"  note: start window +-{tol:g} s is not shorter than half a carrier slot "
                  f"({link.probe_block * 2 * link.probe_reps:g} s), the carrier could be misassigned by one slot")
    if hint is not None:
        L = int(round(TX.probe_plan(link)[1]["total_s"] / T))
        if hint - tol / T > len(G) - L - 2:
            need = (hint * T + t0 - meta["t_prompt"]) + tol + TX.probe_plan(link)[1]["total_s"]
            return dict(error=f"recording ends before the schedule could have finished: it needs {need:.0f} s after "
                              f"the START TX prompt, got {(len(G) * T + t0 - meta['t_prompt']):.0f} s "
                              f"(cap_ook_rx would silently search the whole record instead)"), None
    first = [RX.analyse_probe(G[:, i], T, link, tau_hint=hint, hint_tol_s=tol or 1.5) for i in range(G.shape[1])]
    ok = [p for p in first if "error" not in p]
    if not ok:
        return dict(error=first[0].get("error", "probe failed")), None
    # one schedule start for all nodes: the node with the strongest carrier response, then every node again near it.
    # (Picking by align_peak chose a noise-only node inside the short start window, where that statistic is noisy.)
    ref = max(ok, key=lambda p: max(r["energy_db"] for r in p["rows"]))
    tau = ref["tau_samples"] if not meta.get("t_first") else int(round(hint))
    # every node again around that start; tight, so the refinement can't wander out of the searched window
    probes = [RX.analyse_probe(G[:, i], T, link, tau_hint=tau, hint_tol_s=min(0.3, 3 * T + 0.02))
              for i in range(G.shape[1])]
    car = probes[0]["plan"]["carriers_hz"]
    E = np.array([[r["energy_db"] for r in p["rows"]] for p in probes])        # node x carrier
    Lz = np.array([[r["level_z"] for r in p["rows"]] for p in probes])
    best_c = int(np.argmax(E.max(axis=0)))
    best_n = int(np.argmax(E[:, best_c]))
    top = sorted(range(len(car)), key=lambda c: -E[:, c].max())[:8]
    # A response must stand clear of the noise floor of the whole node x carrier matrix (most pairs see nothing):
    # >= 6 dB and >= median + 6 robust sigma. With 64 nodes x a few carriers and short blocks, a plain 6 dB rule let
    # noise and post-calibration drift through (+10 dB 'response' with the transmitter off, 2026-10-20).
    med = float(np.median(E)); sig = float(1.4826 * np.median(np.abs(E - med))) or 0.5
    thr = max(6.0, med + 6 * sig)
    resp = E[:, best_c]
    good = np.where((resp >= thr) & (resp >= resp.max() - 6))[0]
    start_delay = float(tau * T + t0 - meta["t_prompt"]) if meta.get("t_prompt") else None
    res = dict(mode="probe", fs_hz=1 / T, missing=missing, clashes=clashes, arrival_jitter_ms=jit * 1e3, tau_s=float(tau * T),
               start_delay_s=start_delay, start_window_s=window,
               align_peak=float(ref["align_peak"]), carriers_hz=car, y0=y0, patch=meta.get("patch"),
               noise_floor_db=dict(median=med, robust_sigma=sig, threshold=thr), significant=bool(E.max() >= thr),
               best=dict(carrier_hz=car[best_c], node=names[best_n], energy_db=float(E[best_n, best_c]),
                         level_z=float(Lz[best_n, best_c]),
                         responding_nodes=[names[i] for i in good]),
               top=[dict(carrier_hz=car[c], node=names[int(np.argmax(E[:, c]))], energy_db=float(E[:, c].max()),
                         nodes_over_6db=int((E[:, c] >= 6).sum())) for c in top])
    return res, dict(E=E, Lz=Lz, G=G, T=T, car=car, best_n=best_n, tau=tau, names=names)


def plot_probe(res, aux, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    E, car, names = aux["E"], np.array(aux["car"]), aux["names"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 11), gridspec_kw=dict(height_ratios=[1.2, 1, 1]))
    im = ax[0].imshow(E, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=max(10, np.nanmax(E)))
    step = max(1, len(car) // 20)
    ax[0].set_xticks(range(0, len(car), step))
    ax[0].set_xticklabels([f"{c / 1e3:g}k" for c in car[::step]], rotation=60, fontsize=7)
    nstep = max(1, len(names) // 16)
    ax[0].set_yticks(range(0, len(names), nstep))
    ax[0].set_yticklabels(names[::nstep], fontsize=7)
    ax[0].set_ylabel("node")
    ax[0].set_title("Energy increase while the carrier is on [dB] (on/off blocks)", fontsize=10, loc="left")
    fig.colorbar(im, ax=ax[0], fraction=0.03)
    ax[1].semilogx(car, E.max(axis=0), "o-", ms=3, label="best node at each carrier")
    ax[1].semilogx(car, E[aux["best_n"]], ".-", ms=3, alpha=0.7, label=f"node {names[aux['best_n']]}")
    ax[1].axhline(6, color="0.6", ls="--", lw=0.8)
    ax[1].set_xlabel("carrier [Hz]"); ax[1].set_ylabel("energy increase [dB]"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    b = res["best"]
    ax[1].set_title(f"Best: {b['carrier_hz'] / 1e3:g} kHz on node {b['node']} (+{b['energy_db']:.1f} dB)", fontsize=10, loc="left")
    G, T = aux["G"], aux["T"]
    tt = np.arange(len(G)) * T
    ax[2].plot(tt, G[:, aux["best_n"]], lw=0.6, color="k")
    ax[2].axvline(aux["tau"] * T, color="tab:orange", lw=1, label="schedule start (aligned)")
    ax[2].set_xlabel("time [s]"); ax[2].set_ylabel(f"delta, node {names[aux['best_n']]}"); ax[2].legend(fontsize=8)
    ax[2].set_title(f"Raw node trace ({res['fs_hz']:.1f} samples/s)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ================================================================ ber
def decode(f, T, link):
    _, plan = TX.build(link)
    _, truth = TX.frame_bits(link.payload, link.prbs)
    k = (1.0 / plan["bps"]) / T
    RX.set_guard(k)
    fr = RX.find_frames(f, k, plan["frame_bits"], truth)
    valid = [d for d in fr["frames"] if d["valid"]]
    bits = sum(d["bits"] for d in valid); errs = sum(d["errors"] for d in valid)
    return dict(frames=fr["frames"], n_valid=len(valid), bits=bits, errors=errs, ber=errs / bits if bits else None,
                q=float(np.mean([d["q"] for d in valid])) if valid else None, k=k, plan=plan)


def analyse_ber(t, X, meta):
    a = argparse.Namespace(**meta["args"])
    link = link_from(a, False)
    G, T, t0, jit, (missing, clashes) = regular(t, X)
    y0 = meta["y0"]
    names = node_names(meta)
    score, s_prompt = node_scores(G, T, t0, meta.get("t_prompt"))
    order = np.argsort(-score)
    cand = [int(i) for i in order[:a.candidates]]
    per_node = []
    for i in cand:
        r = RX.analyse(G[:, i], T, link)
        per_node.append(dict(node=names[i], score_db=float(score[i]), feature=r["feature"], bits=r.get("bits", 0),
                             errors=r.get("errors"), ber=r.get("ber"), q=r.get("q")))
    comb_nodes = cand[:a.combine]
    # weight = energy increase ratio - 1 (fixed before decoding). Equal weights let the weaker nodes dilute the best
    # one (synthetic: combined 2.6 % vs best node alone 2.4 %)
    w = np.array([max(10 ** (score[i] / 10) - 1, 0.0) for i in comb_nodes])
    w = w / w.sum() if w.sum() > 0 else np.full(len(comb_nodes), 1 / len(comb_nodes))
    f = np.sum([wi * zscore(RX.features(G[:, i])["absdev"]) for wi, i in zip(w, comb_nodes)], axis=0)
    comb = decode(f, T, link)
    plan = comb["plan"]
    genie, fr = None, []
    if meta.get("t_first"):                     # --tx: decisions at the known bit times (no sync errors) = upper bound
        _, truth = TX.frame_bits(link.payload, link.prbs)
        k = comb["k"]
        P = plan["frame_bits"] * k
        s0g = (meta["t_first"] - t0) / T
        fr = [RX.decode_frame(f, k, s0g + m * P, truth) for m in range(meta.get("tx_frames", 0))
              if s0g + m * P + (len(RX.HDR) + len(truth)) * k < len(f)]
        for m, d in enumerate(fr):
            d.update(m=m, s0=float(s0g + m * P))
        gb = sum(d["bits"] for d in fr)
        ge = sum(d["errors"] for d in fr)
        genie = dict(bits=gb, errors=ge, ber=ge / gb if gb else None, frames=len(fr))
    # how fresh the samples are: arrivals per second, and samples identical to the previous one on every node
    # (a re-read of an old scan instead of a new acquisition)
    dt = np.diff(t)
    fresh = dict(arrival_rate_hz=float((len(t) - 1) / (t[-1] - t[0])), median_interval_ms=float(np.median(dt) * 1e3),
                 repeat_fraction=float(np.mean(np.all(X[1:] == X[:-1], axis=1))), scan_period_ms=float(T * 1e3))
    res = dict(mode="ber", bps=a.bps, carrier_hz=a.carrier, fs_hz=1 / T, samples_per_bit=comb["k"],
               missing=missing, clashes=clashes, arrival_jitter_ms=jit * 1e3, y0=y0, patch=meta.get("patch"),
               combined=dict(nodes=[names[i] for i in comb_nodes], bits=comb["bits"], errors=comb["errors"],
                             ber=comb["ber"], ber_upper95=(3.0 / comb["bits"] if comb["bits"] and comb["errors"] == 0 else None),
                             frames_valid=comb["n_valid"], q=comb["q"]),
               per_node=per_node, frame_bits=plan["frame_bits"], known_timing=genie, sampling=fresh)
    return res, dict(G=G, T=T, f=f, comb=comb, score=score, s_prompt=s_prompt,
                     truth=TX.frame_bits(link.payload, link.prbs)[1], genie_frames=fr if genie else [])


def save_decoded(d, res, aux):
    """decoded.json: every frame of the blind (combined) and the known-timing decoder, with the bits it decided,
    its decision values and the payload that was sent - for comparing runs later without re-decoding"""
    keep = ("m", "s0", "valid", "bits", "errors", "ber", "q", "thr", "inverted", "got", "norm")
    pack = lambda frs: [{k: v for k, v in fr.items() if k in keep} for fr in frs]
    json.dump(dict(bps=res["bps"], carrier_hz=res["carrier_hz"], samples_per_bit=res["samples_per_bit"],
                   combined_nodes=res["combined"]["nodes"], sent_payload=[int(b) for b in aux["truth"]],
                   blind_frames=pack(aux["comb"]["frames"]), known_timing_frames=pack(aux["genie_frames"])),
              open(os.path.join(d, "decoded.json"), "w"))


def plot_ber(res, aux, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(11, 10))
    T, f, comb = aux["T"], aux["f"], aux["comb"]
    tt = np.arange(len(f)) * T
    ax[0].plot(tt, f, lw=0.5, color="k")
    for d in comb["frames"]:
        ax[0].axvline(d["s0"] * T, color="tab:green" if d["valid"] else "tab:red", lw=1)
    ax[0].axvline(aux["s_prompt"] * T, color="tab:orange", ls="--", lw=1, label="START TX prompt")
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("combined z(absdev)"); ax[0].legend(fontsize=8)
    c = res["combined"]
    ber = f"{c['errors']}/{c['bits']}" if c["bits"] else "no valid frame"
    ax[0].set_title(f"{res['bps']:g} bps @ {res['fs_hz']:.1f} samples/s ({res['samples_per_bit']:.1f} per bit), "
                    f"combined nodes {', '.join(c['nodes'])}: BER {ber}   (frame starts: green valid / red invalid)",
                    fontsize=9, loc="left")
    pn = res["per_node"]
    lab = [p["node"] for p in pn] + ["combined"]
    val = [(p["ber"] if p["ber"] is not None else np.nan) for p in pn] + [c["ber"] if c["ber"] is not None else np.nan]
    ax[1].bar(range(len(val)), val, color=["tab:blue"] * len(pn) + ["tab:green"])
    ax[1].set_xticks(range(len(val))); ax[1].set_xticklabels(lab, fontsize=8)
    ax[1].set_ylabel("blind BER"); ax[1].set_title("Per node (ranked by energy increase) and combined", fontsize=9, loc="left")
    truth = aux["truth"]
    for d in comb["frames"]:
        if not d["valid"]:
            continue
        nv = np.array(d["norm"]); tr = np.array(truth, bool)
        x0 = d["m"] * len(truth)
        ax[2].plot(x0 + np.where(tr)[0], nv[tr], ".", ms=3, color="tab:orange")
        ax[2].plot(x0 + np.where(~tr)[0], nv[~tr], ".", ms=3, color="tab:blue")
    ax[2].axhline(0, color="0.4", lw=0.8)
    ax[2].set_xlabel("payload bit"); ax[2].set_ylabel("decision value (0 = threshold)")
    ax[2].set_title("Decision values: orange = sent 1, blue = sent 0", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ================================================================ onoff
ONOFF_PHASES = (("off1", 0), ("on", 1), ("off2", 0))
ONOFF_SETTLE_S = 0.5                            # left out of the statistics after every switch


def run_onoff(a):
    """Carrier OFF / continuous ON / OFF, --phase-s each, recording the per-node T37 deltas (the chip's raw
    mutual-capacitance measurement minus its reference; calibrated with the carrier off, so OFF sits near 0).
    The carrier is switched off in a finally: Ctrl+C, SIGTERM or any error still turns it off."""
    import signal
    import stent_tx as STX
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))   # kill -> SystemExit -> the finally below runs
    a.carrier = a.carrier or 163000.0
    tx = chip = None
    rows, changed = [], {}
    try:
        tx, info = STX.open_configured(a)
        print(f"Transmitter: {info['line']}")
        print(f"  {tx.off()}")
        chip = Chip(a)
        changed = chip.setup()
        print(f"Chip set up: original CTRL bytes {changed} (restored at the end), calibrated with the carrier off")
        print(f"Recording {chip.ny} nodes, {len(ONOFF_PHASES)} x {a.phase_s:g} s: "
              + " / ".join(f"{p.upper()}" for p, _ in ONOFF_PHASES) + f"  (carrier {a.carrier:g} Hz)")
        t0 = time.time()
        errors = 0
        for k, (phase, on) in enumerate(ONOFF_PHASES):
            print(f"  {phase}: " + (tx.tone(a.carrier) if on else tx.off()), flush=True)
            t_end = t0 + (k + 1) * a.phase_s
            while time.time() < t_end:
                try:
                    v = chip.sample()
                except Exception:
                    errors += 1
                    if errors > 50:
                        raise
                    continue
                rows.append((time.time(), phase, on, v))
            n = sum(r[1] == phase for r in rows)
            print(f"    {n} samples ({n / a.phase_s:.1f} /s)")
    except KeyboardInterrupt:
        print("  stopped (Ctrl+C) - saving what was recorded")
    finally:
        if tx:
            try:
                print(f"  carrier: {tx.off()}")
            except Exception as e:
                print(f"  ! could not confirm the carrier is off ({e}) - run: python stent_tx.py off")
            tx.close()
        if chip:
            chip.restore()
    if len(rows) < 10:
        print("too few samples recorded"); return
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(DATADIR, f"stent_onoff_{a.label + '_' if a.label else ''}{a.carrier / 1e3:g}kHz_{stamp}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "onoff.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "unix_time", "phase", "tx_on"] + chip.names)
        for t, phase, on, v in rows:
            w.writerow([f"{t - rows[0][0]:.4f}", f"{t:.4f}", phase, on] + [int(x) for x in v])
    json.dump(dict(mode="onoff", args=vars(a), names=chip.names, patch=chip.patch, y0=chip.y0, ny=chip.ny,
                   phase_s=a.phase_s, carrier_hz=a.carrier, chip_ctrl_orig=changed, samples=len(rows)),
              open(os.path.join(d, "meta.json"), "w"), indent=1)
    analyse_onoff(d)


def analyse_onoff(d):
    """per-node mean/std per phase and the ON step, from onoff.csv; writes onoff_stats.csv and result.png"""
    with open(os.path.join(d, "onoff.csv")) as f:
        r = csv.reader(f)
        head = next(r)
        data = list(r)
    names = head[4:]
    t = np.array([float(x[0]) for x in data])
    phase = np.array([x[2] for x in data])
    on = np.array([int(x[3]) for x in data], bool)
    X = np.array([[float(v) for v in x[4:]] for x in data])
    # switch times = first sample of each phase; drop ONOFF_SETTLE_S after each switch from the statistics
    starts = {p: float(t[phase == p][0]) for p, _ in ONOFF_PHASES if (phase == p).any()}
    settled = np.ones(len(t), bool)
    for s in starts.values():
        settled &= ~((t >= s) & (t < s + ONOFF_SETTLE_S))

    def stats(p):
        m = (phase == p) & settled
        return X[m].mean(axis=0), X[m].std(axis=0), int(m.sum())

    if not all(p in starts for p, _ in ONOFF_PHASES):
        print(f"incomplete run (phases recorded: {list(starts)}) - no statistics"); return None
    (m1, s1, n1), (mo, so, no), (m2, s2, n2) = stats("off1"), stats("on"), stats("off2")
    off_mean, off_std = (m1 + m2) / 2, np.sqrt((s1 ** 2 + s2 ** 2) / 2)
    step = mo - off_mean
    drift = m2 - m1                             # how much OFF moved by itself = the yardstick for a real step
    ratio = so / np.maximum(off_std, 1e-9)
    # a response: the ON mean moves by >= 0.5 x the OFF noise AND >= 3 x the OFF1 -> OFF2 drift, or the scatter grows
    # >= 1.5 x (a carrier near the acquisition frequency shows up as extra noise rather than a shift)
    level = (np.abs(step) >= 0.5 * off_std) & (np.abs(step) >= 3 * np.abs(drift))
    noisy = ratio >= 1.5
    score = np.maximum(np.abs(step) / np.maximum(off_std, 0.5), ratio - 1)
    with open(os.path.join(d, "onoff_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "mean_off1", "std_off1", "mean_on", "std_on", "mean_off2", "std_off2",
                    "step_on_minus_off", "drift_off2_minus_off1", "std_ratio_on_off", "level_response", "noise_response"])
        for i, n in enumerate(names):
            w.writerow([n] + [f"{v:.3f}" for v in (m1[i], s1[i], mo[i], so[i], m2[i], s2[i], step[i], drift[i], ratio[i])]
                       + [int(level[i]), int(noisy[i])])
    order = np.argsort(-score)
    print(f"\nPer node (statistics skip {ONOFF_SETTLE_S:g} s after each switch; {n1} / {no} / {n2} samples):")
    print(f"{'node':>8} {'OFF1 mean±std':>15} {'ON mean±std':>15} {'OFF2 mean±std':>15} {'ON-OFF':>8} {'drift':>7} {'std ON/OFF':>10}")
    for i in order:
        flag = ("  <-- level" if level[i] else "") + ("  <-- noise" if noisy[i] else "")
        print(f"{names[i]:>8} {m1[i]:7.2f}±{s1[i]:<6.2f} {mo[i]:7.2f}±{so[i]:<6.2f} {m2[i]:7.2f}±{s2[i]:<6.2f} "
              f"{step[i]:+8.2f} {drift[i]:+7.2f} {ratio[i]:10.2f}{flag}")
    resp = [names[i] for i in order if level[i] or noisy[i]]
    if resp:
        print(f"\nResponse on {len(resp)} node(s): {', '.join(resp)}")
    else:
        print("\nNo node responded: the ON mean stayed within the OFF noise / drift and the scatter did not grow.")
    plot_onoff(d, t, X, names, starts, step, drift, order, level | noisy, ratio)
    print(f"Saved {d}")
    return dict(names=names, step=step, drift=drift, ratio=ratio, level=level, noisy=noisy)


def plot_onoff(d, t, X, names, starts, step, drift, order, flagged, ratio):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]       # categorical slots 1-4, fixed order
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(4, 1, figsize=(11, 13), gridspec_kw=dict(height_ratios=[1.3, 1, 1, 1]))
    t_on, t_off2 = starts["on"], starts["off2"]
    for a_ in ax[:2]:
        a_.axvspan(t_on, t_off2, color="#f0efec", zorder=0)
        a_.axvline(t_on, color=ink2, lw=0.8, ls="--"); a_.axvline(t_off2, color=ink2, lw=0.8, ls="--")
        a_.grid(color=grid, lw=0.6); a_.set_axisbelow(True)
        a_.set_xlim(t[0], t[-1])
    # raw samples overplotted are a solid band (+-3 counts of noise hides a 1-2 count shift): show 0.5 s means
    W = max(1, int(round(0.5 * (len(t) - 1) / (t[-1] - t[0]))))
    smooth = lambda v: np.convolve(v, np.ones(W) / W, mode="same")
    top = order[:4]
    for c, i in zip(colors, top):
        ax[0].plot(t, smooth(X[:, i]), lw=1.5, color=c, label=f"{names[i]} (ON-OFF {step[i]:+.1f})")
    ax[0].set_ylabel("delta, 0.5 s mean [counts]", color=ink2)
    ax[0].legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False)
    ax[0].set_title("Node deltas, the 4 nodes that changed most (shaded = carrier ON; raw values in onoff.csv)",
                    fontsize=10, loc="left", color=ink, pad=30)
    ymax = ax[0].get_ylim()[1]
    for x, lab in ((t[0], "OFF"), (t_on, "ON"), (t_off2, "OFF")):
        ax[0].text(x + 0.2, ymax, lab, va="top", fontsize=9, color=ink2)
    ax[1].plot(t, smooth(X.mean(axis=1)), lw=1.5, color=colors[0])
    ax[1].set_ylabel("0.5 s mean [counts]", color=ink2)
    ax[1].set_xlabel("time [s]", color=ink2)
    ax[1].set_title(f"Mean over all {len(names)} nodes", fontsize=10, loc="left", color=ink)
    xs = np.arange(len(names))
    ax[2].bar(xs, step, width=0.8, color=[colors[1] if f else colors[0] for f in flagged])
    ax[2].errorbar(xs, step, yerr=np.abs(drift), fmt="none", ecolor=ink2, lw=0.8, capsize=0)
    ax[2].axhline(0, color=ink2, lw=0.8)
    nstep = max(1, len(names) // 32)
    ax[2].set_xticks(xs[::nstep]); ax[2].set_xticklabels(names[::nstep], rotation=90, fontsize=7)
    ax[2].set_xlim(-0.6, len(names) - 0.4)
    ax[2].set_ylabel("ON - OFF mean [counts]", color=ink2)
    ax[2].grid(axis="y", color=grid, lw=0.6); ax[2].set_axisbelow(True)
    ax[2].set_title("Step per node (whisker = OFF1->OFF2 drift; orange = counted as a response)",
                    fontsize=10, loc="left", color=ink)
    # 2026-10-22, TeensyTRX at 163 kHz: the carrier showed up as huge zero-mean swings (std x250-530 on Y1-Y4),
    # not as a mean step - the scatter ratio is the measure that shows it
    ax[3].bar(xs, ratio, width=0.8, color=[colors[1] if f else colors[0] for f in flagged])
    ax[3].set_yscale("log")
    ax[3].axhline(1.0, color=ink2, lw=0.8)
    ax[3].axhline(1.5, color=ink2, lw=0.8, ls="--")
    ax[3].set_xticks(xs[::nstep]); ax[3].set_xticklabels(names[::nstep], rotation=90, fontsize=7)
    ax[3].set_xlim(-0.6, len(names) - 0.4)
    ax[3].set_ylabel("std ON / std OFF", color=ink2)
    ax[3].grid(axis="y", color=grid, lw=0.6, which="both"); ax[3].set_axisbelow(True)
    ax[3].set_title("Scatter ratio per node (log; dashed = 1.5, counted as a response above it)",
                    fontsize=10, loc="left", color=ink)
    for a_ in ax:
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "result.png"), dpi=110)
    plt.close(fig)


# ================================================================ report
def report(res, d):
    if res.get("error"):
        print(f"\nanalysis failed: {res['error']}")
        return
    if res["clashes"] < 0:
        timing = f"not scan-locked, placed on a time grid (interval jitter {res['arrival_jitter_ms']:.1f} ms)"
    else:
        timing = (f"scan-locked, {res['missing']} missing scans interpolated, arrival jitter "
                  f"{res['arrival_jitter_ms']:.1f} ms, {res['clashes']} numbering clashes resolved")
    print(f"\nSamples: {res['fs_hz']:.1f} /s, {timing}")
    if res["mode"] == "probe" and res.get("start_delay_s") is not None:
        print(f"Transmitter schedule started {res['start_delay_s']:.1f} s after the START TX prompt "
              f"(searched {res['start_window_s'][0]:.1f} to {res['start_window_s'][1]:.1f} s)")
    if res["mode"] == "probe":
        b = res["best"]
        print(f"Best carrier {b['carrier_hz'] / 1e3:g} kHz on node {b['node']}: +{b['energy_db']:.1f} dB "
              f"(level z {b['level_z']:+.1f}); responding nodes [{', '.join(b['responding_nodes'])}]")
        print("Strongest carriers:  " + "   ".join(f"{r['carrier_hz'] / 1e3:g}k +{r['energy_db']:.1f} dB ({r['node']})"
                                            for r in res["top"]))
        nf = res["noise_floor_db"]
        print(f"Noise floor over all node x carrier pairs: median {nf['median']:+.1f} dB, robust sigma {nf['robust_sigma']:.1f} dB "
              f"-> response threshold {nf['threshold']:.1f} dB")
        resp = b["responding_nodes"]
        if res["significant"] and resp and res.get("patch"):
            print(f"Next: ber runs with  --carrier {b['carrier_hz']:g} --patch {','.join(map(str, res['patch']))}  "
                  f"(only ~{res['fs_hz']:.0f} samples/s here: keep --bps <= {res['fs_hz'] / 5:.1f}, "
                  f"or move the stent to X line 0 for ~137 samples/s)")
        elif res["significant"] and resp:
            c = int(np.clip(round(np.mean([int(n[1:]) for n in resp])) - 14, 0, NODES_PER_PAGE - 28))
            print(f"Next: ber runs with  --carrier {b['carrier_hz']:g} --y0 {c} --ny 28  (~137 samples/s)")
        else:
            where = "inside the --patch square" if res.get("patch") else "over X line 0"
            print(f"No clear response (nothing above the threshold). Check the stent is {where}, use more --reps, "
                  "or try a finer --carriers sweep near the expected frequency.")
    else:
        c = res["combined"]
        ber = f"{c['errors']}/{c['bits']} = {c['ber']:.2%}" if c["bits"] else "no valid frame"
        s = res.get("sampling")
        if s:
            print(f"Fresh samples: {s['arrival_rate_hz']:.1f} /s arriving (median interval {s['median_interval_ms']:.2f} ms), "
                  f"{s['repeat_fraction']:.2%} identical to the previous sample, scan period {s['scan_period_ms']:.3f} ms")
        print(f"{res['bps']:g} bps, {res['samples_per_bit']:.1f} samples per bit")
        print(f"Blind BER, combined nodes {', '.join(c['nodes'])}: {ber}" + (f"  (95 % upper bound {c['ber_upper95']:.1e})"
                                                                    if c["ber_upper95"] else ""))
        g = res.get("known_timing")
        if g and g["bits"]:
            print(f"Known-timing BER (transmitter driven from here, no sync errors): {g['errors']}/{g['bits']} = {g['ber']:.2%}")
        print("Per node:  " + "   ".join(f"{p['node']} {p['errors']}/{p['bits']} ({p['feature']}, +{p['score_db']:.0f} dB)"
                                       if p["bits"] else f"{p['node']} no frame (+{p['score_db']:.0f} dB)"
                                       for p in res["per_node"]))
    print(f"Saved {d}")


def analyse_dir(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "capture.npz"))
    if meta["mode"] == "probe":
        res, aux = analyse_probe(z["t"], z["X"], meta)
        if aux is not None:
            plot_probe(res, aux, os.path.join(d, "result.png"))
    else:
        res, aux = analyse_ber(z["t"], z["X"], meta)
        plot_ber(res, aux, os.path.join(d, "result.png"))
        save_decoded(d, res, aux)
    json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1)
    report(res, d)
    return res


def summary(a):
    rows = []
    for p in sorted(glob.glob(os.path.join(DATADIR, "stent_ber_*", "result.json"))):
        r = json.load(open(p))
        if a.label and a.label not in p:
            continue
        c = r["combined"]
        rows.append((r["bps"], r["fs_hz"], r["samples_per_bit"], c["errors"], c["bits"], os.path.basename(os.path.dirname(p))))
    if not rows:
        print("no ber runs found"); return
    rows.sort()
    print(f"{'bps':>6} {'samples/s':>9} {'per bit':>7} {'BER':>14}  run")
    for bps, fs, k, e, b, name in rows:
        print(f"{bps:6g} {fs:9.1f} {k:7.1f} {(f'{e}/{b} = {e / b:.2%}' if b else 'no frame'):>14}  {name}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ok = [r for r in rows if r[4]]
    bps = np.array([r[0] for r in ok]); ber = np.array([max(r[3], 0.5) / r[4] for r in ok])
    zero = np.array([r[3] == 0 for r in ok])
    ax.semilogy(bps[~zero], ber[~zero], "o", color="tab:blue", label="measured")
    ax.semilogy(bps[zero], ber[zero], "v", color="tab:green", label="0 errors (plotted at 0.5/N)")
    ax.axhline(0.01, color="0.5", ls="--", lw=0.8, label="1 %")
    ax.set_xlabel("bit rate [bps]"); ax.set_ylabel("blind BER (combined nodes)"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("Stent -> maXTouch: BER vs bit rate", fontsize=10)
    out = os.path.join(DATADIR, "stent_ber_summary.png")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"figure: {os.path.relpath(out, HERE)}")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description="Stent -> maXTouch receiver: carrier probe and BER")
    ap.add_argument("mode", choices=("probe", "ber", "onoff", "reanalyse", "summary"))
    ap.add_argument("dir", nargs="?", help="reanalyse: the run folder")
    ap.add_argument("--label", default="")
    ap.add_argument("--carrier", type=float, help="ber: carrier [Hz] the transmitter sends (for the record); "
                                                   "onoff: continuous carrier (default 163000)")
    ap.add_argument("--phase-s", type=float, default=10.0, help="onoff: length of each OFF / ON / OFF phase [s]")
    ap.add_argument("--bps", type=float, default=10.0)
    ap.add_argument("--payload", type=int, default=127)
    ap.add_argument("--prbs", type=int, default=7)
    ap.add_argument("--carriers", default="default",
                    help="probe: 'default' (transmitter's list), 'a,b,c', 'log:f0:f1:step%%' or 'lin:f0:f1:step'")
    ap.add_argument("--block", type=float,
                    help="probe: on/off block [s] (transmitter --probe-block; default 1, 3 with --patch)")
    ap.add_argument("--reps", type=int, default=3, help="probe: repeats per carrier (transmitter --probe-reps)")
    ap.add_argument("--start-delay", type=float, default=1.5,
                    help="probe: expected seconds from the START TX prompt to the transmitter schedule start")
    ap.add_argument("--start-tol", type=float, default=2.0,
                    help="probe: +- seconds searched around --start-delay; keep it under half a carrier slot "
                         "(2 x block x reps) so the carrier can't be misassigned")
    ap.add_argument("--seconds", type=float, help="record this long after START TX (default: probe schedule + 20 s, ber 4 frames)")
    ap.add_argument("--lead", type=float, default=4.0, help="noise recording before START TX [s]")
    ap.add_argument("--y0", type=int, default=0, help="first Y node of page 0 to read")
    ap.add_argument("--ny", type=int, default=64, help="number of nodes (<= 28 = one HID transfer, ~100 samples/s)")
    ap.add_argument("--patch", type=parse_patch,
                    help="X0,Y0,SIZE or X0,Y0,W,H: read this fixed square anywhere instead of X line 0 "
                         "(slower: pages are stepped from 0; 4x4 measured 27 / 12 / 4 samples/s at X 0 / 5 / 20)")
    ap.add_argument("--candidates", type=int, default=8, help="ber: nodes decoded individually")
    ap.add_argument("--combine", type=int, default=3, help="ber: best nodes averaged into the combined decision")
    ap.add_argument("--tx", action="store_true",
                    help="drive the Teensy (original TeensyTRX firmware, via stent_tx.py) from here: exact timing, no prompt")
    ap.add_argument("--frames", type=int, default=4, help="ber --tx: frames to send")
    ap.add_argument("--zero-hz", type=float, help="ber --tx: FSK '0' frequency (default OOK)")
    ap.add_argument("--keep-noise-suppression", action="store_true")
    ap.add_argument("--keep-touch", action="store_true")
    import stent_tx as STX
    STX.add_hw_args(ap)
    a = ap.parse_args()
    if a.block is None:
        # ~8 samples/s in a patch: 1 s blocks left ~5 samples each and pure noise reached +10 dB somewhere among
        # 16 nodes x 14 carriers, which then set the schedule alignment (synthetic, 2026-10-21). 2-3 s blocks: clean
        a.block = 3.0 if a.patch else 1.0

    if a.mode == "reanalyse":
        if os.path.exists(os.path.join(a.dir, "onoff.csv")):
            analyse_onoff(a.dir); return
        analyse_dir(a.dir); return
    if a.mode == "summary":
        summary(a); return
    if a.mode == "onoff":
        if not 0 <= a.y0 < NODES_PER_PAGE:
            ap.error("--y0 must be 0-63 (page 0)")
        run_onoff(a); return
    if a.mode == "ber" and a.carrier is None:
        ap.error("ber needs --carrier (what the transmitter sends; from the probe)")
    if not 0 <= a.y0 < NODES_PER_PAGE:
        ap.error("--y0 must be 0-63 (page 0)")
    probe = a.mode == "probe"
    link = link_from(a, probe)
    if probe:
        _, plan = TX.probe_plan(link)
        seconds = a.seconds or plan["total_s"] + 20
        print(f"Probe: {len(plan['carriers_hz'])} carriers, {plan['total_s']:.0f} s schedule")
    else:
        _, plan = TX.build(link)
        seconds = a.seconds or (a.frames if a.tx else 4) * plan["frame_s"] + 5
        print(f"BER: {a.bps:g} bps, frame {plan['frame_bits']} bits = {plan['frame_s']:.1f} s, recording {seconds:.0f} s after START TX")
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))   # kill -> SystemExit -> the finally below runs
    tx = start_tx = tx_info = chip = None
    try:
        if a.tx:
            tx, tx_info = STX.open_configured(a)
            print(f"Transmitter: {tx_info['line']}")
            if probe:
                segs, tplan = STX.probe_segments(link.probe_carriers, a.block, a.reps)
            else:
                segs, tplan = STX.ber_segments(a.carrier, a.bps, a.frames, a.zero_hz, a.payload, a.prbs)
            print(f"  uploaded {len(segs)} segments ({tplan['total_s']:.1f} s): {tx.upload(segs)}")
            seconds = a.seconds or STX.GO_DELAY_MS / 1e3 + tplan["total_s"] + 2
            start_tx = lambda: tx.go()[0]
        else:
            print(f"Transmitter command (other computer):\n  {tx_command(a, probe)}\n")
        chip = Chip(a)
        changed = chip.setup()
        print(f"Chip set up: original CTRL bytes {changed} (restored at the end), calibrated with TX off")
        t, X, t_prompt, t_first, errors = record(chip, a, seconds, start_tx)
    finally:
        if tx:                                  # TeensyTRX 'off' (also stops a schedule still running)
            tx.close()
        if chip:
            chip.restore()
    if len(t) < 50:
        print("too few samples recorded"); return
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"stent_{a.mode}_{a.label + '_' if a.label else ''}" + (f"{a.bps:g}bps_" if not probe else "") + stamp
    d = os.path.join(DATADIR, name)
    meta = dict(mode=a.mode, args=vars(a), y0=chip.y0, ny=chip.ny, patch=chip.patch, t_prompt=t_prompt, t_first=t_first,
                tx=tx_info, tx_frames=(a.frames if a.tx and not probe else 0), hid_errors=errors,
                chip_ctrl_orig=changed, plan=plan, tx_command=tx_command(a, probe),
                recorded_s=float(t[-1] - t[0]), samples=int(len(t)))
    save(d, t, X, meta)
    analyse_dir(d)


if __name__ == "__main__":
    main()
