#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Usage: python status.py <input_dir>
Hiển thị trạng thái xử lý tất cả scene.
"""

import sys, json, time
from pathlib import Path

SUPPORTED = {".mp4", ".MP4", ".avi", ".mov", ".mkv", ".webm"}


def main():
    if len(sys.argv) != 2:
        print("Usage: python status.py <input_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    if not input_dir.is_dir():
        print(f"ERROR: {input_dir} is not a directory")
        sys.exit(1)

    total = done = processing = error = pending = 0
    now = time.time()

    for video_dir in sorted(input_dir.iterdir()):
        if not video_dir.is_dir() or video_dir.name.startswith("."):
            continue

        scenes = sorted(f for f in video_dir.iterdir()
                        if f.is_file() and f.suffix in SUPPORTED)
        if not scenes:
            continue

        v_done = v_proc = v_err = v_pend = 0
        rows = []
        for scene in scenes:
            total += 1
            done_f    = video_dir / f".done_{scene.stem}"
            error_f   = video_dir / f".error_{scene.stem}"
            claimed_f = video_dir / f".claimed_{scene.stem}"

            if done_f.exists():
                status = "✅ done"
                done += 1; v_done += 1
            elif error_f.exists():
                try:
                    data = json.loads(error_f.read_text())
                    err_msg = data.get("error", "")[:60]
                except:
                    err_msg = ""
                status = f"❌ error: {err_msg}"
                error += 1; v_err += 1
            elif claimed_f.exists():
                try:
                    data  = json.loads(claimed_f.read_text())
                    age   = int(now - data.get("t", now))
                    host  = data.get("host", "?")
                    pid   = data.get("pid", "?")
                    status = f"⏳ processing ({host}:{pid}, {age}s ago)"
                except:
                    status = "⏳ processing"
                processing += 1; v_proc += 1
            else:
                status = "🔲 pending"
                pending += 1; v_pend += 1

            rows.append((scene.name, status))

        # Video header
        total_scenes = len(scenes)
        print(f"\n📁 {video_dir.name}  [{v_done}/{total_scenes} done"
              f"{f'  {v_proc} processing' if v_proc else ''}"
              f"{f'  {v_err} error' if v_err else ''}"
              f"{f'  {v_pend} pending' if v_pend else ''}]")
        for name, status in rows:
            print(f"   {name:45s}  {status}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Total  : {total}")
    print(f"✅ Done       : {done}")
    print(f"⏳ Processing : {processing}")
    print(f"❌ Error      : {error}")
    print(f"🔲 Pending    : {pending}")


if __name__ == "__main__":
    main()
