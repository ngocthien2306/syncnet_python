# AVSR Pipeline

## Cấu trúc thư mục

```
input_dir/
├── 0pGOSWYMJio/
│   ├── 0pGOSWYMJio-Scene-001.mp4
│   └── 0pGOSWYMJio-Scene-002.mp4
└── 02ZGoRLlsRg/
    └── ...

output_base/
├── 0pGOSWYMJio/
│   ├── 0pGOSWYMJio-Scene-001/
│   │   ├── videos/       ← clip mp4
│   │   ├── audios/       ← clip wav
│   │   └── transcripts/  ← transcript txt
│   └── ...
└── worker_<hostname>_<pid>.log
```

---

## Chạy worker

```bash
# 1 scene tại 1 thời điểm
python worker.py <input_dir> <output_base>

# N scene song song (tùy VRAM)
python worker.py <input_dir> <output_base> --parallel 2
```

**Ví dụ:**
```bash
python worker.py /workspace/youtube_download_scenes /workspace/output --parallel 2
```

Nhiều máy/terminal chạy cùng lúc được — mỗi máy tự động nhận scene khác nhau, không bị trùng.

---

## Kiểm tra trạng thái

```bash
python status.py <input_dir>
```

**Output:**
```
📁 0pGOSWYMJio  [3/10 done  1 processing  1 error  5 pending]
   0pGOSWYMJio-Scene-001.mp4    ✅ done
   0pGOSWYMJio-Scene-002.mp4    ⏳ processing (machine1:1234, 45s ago)
   0pGOSWYMJio-Scene-003.mp4    ❌ error: cuda out of memory
   0pGOSWYMJio-Scene-004.mp4    🔲 pending

============================================================
Total  : 18  |  ✅ 3  |  ⏳ 1  |  ❌ 1  |  🔲 13
```

---

## Reset để chạy lại

```bash
# Xóa toàn bộ — chạy lại hết
find <input_dir> -name ".done_*" -o -name ".error_*" -o -name ".claimed_*" | xargs rm -f

# Chỉ reset scene chưa có output (giữ lại scene đã xong thật sự)
find <input_dir> -name ".done_*" | while read f; do
    video=$(basename $(dirname $f))
    scene=$(basename $f | sed 's/^\.done_//')
    out="<output_base>/$video/$scene/videos"
    if [ ! -d "$out" ] || [ -z "$(ls -A $out 2>/dev/null)" ]; then
        rm "$f"
    fi
done
```

---

## Xem log realtime

```bash
tail -f /workspace/output/worker_*.log
```

---

## Đo tổng thời lượng audio (target 720h)

```bash
python count_duration.py <output_base>
```

**Output:**
```
  Scanning 1842/1842 files...

  Files     : 1,842
  Total     : 48h 32m 17s  (48.54h)
  Target    : 720h
  Remaining : 671h 27m

  [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 6.7%
```

Scan toàn bộ file `.wav` trong output folder (bao gồm tất cả subfolder).