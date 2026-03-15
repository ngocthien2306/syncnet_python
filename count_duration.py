#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Usage: python count_duration.py <folder>
"""

import sys, wave, contextlib
from pathlib import Path

TARGET_H = 720


def get_wav_duration(path: Path) -> float:
    try:
        with contextlib.closing(wave.open(str(path), 'r')) as f:
            return f.getnframes() / float(f.getframerate())
    except:
        return 0.0


def main():
    if len(sys.argv) != 2:
        print("Usage: python count_duration.py <folder>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"ERROR: {folder} is not a directory")
        sys.exit(1)

    wav_files = sorted(folder.rglob("*.wav"))
    if not wav_files:
        print("No .wav files found.")
        sys.exit(0)

    total_sec = 0.0
    for i, f in enumerate(wav_files, 1):
        dur = get_wav_duration(f)
        total_sec += dur
        print(f"\r  Scanning {i}/{len(wav_files)} files...", end="", flush=True)

    print()

    total_h   = total_sec / 3600
    target_s  = TARGET_H * 3600
    pct       = total_sec / target_s * 100
    remaining = max(0.0, target_s - total_sec)

    h  = int(total_sec // 3600)
    m  = int((total_sec % 3600) // 60)
    s  = int(total_sec % 60)

    rh = int(remaining // 3600)
    rm = int((remaining % 3600) // 60)

    bar_len  = 40
    filled   = int(bar_len * min(pct, 100) / 100)
    bar      = "█" * filled + "░" * (bar_len - filled)

    print(f"\n  Files     : {len(wav_files):,}")
    print(f"  Total     : {h}h {m}m {s}s  ({total_h:.2f}h)")
    print(f"  Target    : {TARGET_H}h")
    print(f"  Remaining : {rh}h {rm}m")
    print(f"\n  [{bar}] {pct:.1f}%\n")


if __name__ == "__main__":
    main()
