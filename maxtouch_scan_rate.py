#!/usr/bin/env python3
"""Read the maXTouch configured scan (acquisition) rate from the GEN_POWERCONFIG_T7 object. Read-only, no touch needed.

T7 byte 0 IDLEACQINT / byte 1 ACTVACQINT: interval between acquisitions in ms while idle / while touched.
255 = free-run (scan back to back, rate set by the acquisition timing), 0 = no acquisition (deep sleep).
For the measured rate with a finger on the panel, use: python maxtouch_hid.py --mode peaks
"""

from __future__ import annotations

from maxtouch_hid import MaxTouchHid

GEN_POWERCONFIG_T7 = 7


def describe(interval_ms: int) -> str:
    if interval_ms == 0:
        return "0 = acquisition off (deep sleep)"
    if interval_ms == 255:
        return "255 = free-run (scans back to back as fast as the acquisition settings allow)"
    return f"{interval_ms} ms -> {1000.0 / interval_ms:.1f} Hz"


def main() -> None:
    mxt = MaxTouchHid()
    try:
        mxt.load_info()
        assert mxt.id_info is not None
        idn = mxt.id_info
        print(
            f"Chip: family=0x{idn.family:02X} variant=0x{idn.variant:02X} "
            f"version={idn.version}.{idn.build:02X} matrix={idn.matrix_x}x{idn.matrix_y}"
        )
        t7 = mxt.object_address(GEN_POWERCONFIG_T7, 0)
        t7_size = mxt.object_size(GEN_POWERCONFIG_T7, 0)
        if t7 is None or t7_size is None:
            raise RuntimeError("T7 object not found in object table")
        raw = mxt.read_register(t7, t7_size)
        print(f"T7 @ 0x{t7:04X} size={t7_size}: {raw.hex(' ')}")
        print(f"  Idle   scan interval (IDLEACQINT): {describe(raw[0])}")
        print(f"  Active scan interval (ACTVACQINT): {describe(raw[1])}")
        if t7_size > 2:
            print(f"  Active-to-idle timeout (ACTV2IDLETO): {raw[2] * 200} ms")
    finally:
        mxt.close()


if __name__ == "__main__":
    main()
