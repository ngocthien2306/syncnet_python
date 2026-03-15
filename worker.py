#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Usage:
  python worker.py <input_dir> <output_base> [--parallel N]

Example:
  python worker.py /workspace/youtube_download_scenes /workspace/output --parallel 2
"""

import os, sys, time, json, socket, traceback, argparse, threading, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from process_scene import process_scene

SUPPORTED       = {".mp4", ".MP4", ".avi", ".mov", ".mkv", ".webm"}
CLAIM_TIMEOUT   = 3600
HEARTBEAT_EVERY = 60
SCAN_INTERVAL   = 5

_hostname  = socket.gethostname()
_scan_lock = threading.Lock()
logger     = logging.getLogger("worker")


def setup_logging(output_base: str) -> Path:
    log_path = Path(output_base) / f"worker_{_hostname}_{os.getpid()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ]
    )
    return log_path


# ===== CLAIM HELPERS =====

def _cf(scene: Path) -> Path:
    return scene.parent / f".claimed_{scene.stem}"

def _done_f(scene: Path) -> Path:
    return scene.parent / f".done_{scene.stem}"

def _error_f(scene: Path) -> Path:
    return scene.parent / f".error_{scene.stem}"


def _read_claim(scene: Path) -> dict:
    try:
        return json.loads(_cf(scene).read_text())
    except:
        return {}


def _write_claim(scene: Path, pid: int):
    _cf(scene).write_text(json.dumps({
        "host": _hostname, "pid": pid, "t": time.time()
    }))


def try_claim(scene: Path) -> bool:
    if _done_f(scene).exists() or _error_f(scene).exists():
        return False

    cf = _cf(scene)
    if cf.exists():
        data = _read_claim(scene)
        if time.time() - data.get("t", 0) < CLAIM_TIMEOUT:
            return False
        try:
            cf.unlink(missing_ok=True)
        except:
            return False

    try:
        with open(cf, 'x') as f:
            json.dump({"host": _hostname, "pid": os.getpid(), "t": time.time()}, f)
        return True
    except FileExistsError:
        return False


def mark_done(scene: Path):
    _done_f(scene).write_text(json.dumps({
        "host": _hostname, "pid": os.getpid(), "t": time.time()
    }))
    _cf(scene).unlink(missing_ok=True)


def mark_error(scene: Path, error: str):
    _error_f(scene).write_text(json.dumps({
        "host": _hostname, "pid": os.getpid(), "t": time.time(), "error": error
    }))
    _cf(scene).unlink(missing_ok=True)


# ===== HEARTBEAT =====

class Heartbeat:
    def __init__(self, scene: Path):
        self.scene   = scene
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(HEARTBEAT_EVERY):
            try:
                _write_claim(self.scene, os.getpid())
            except:
                pass

    def stop(self):
        self._stop.set()


# ===== SCAN =====

def scan_scenes(input_dir: Path) -> list:
    scenes = []
    for video_dir in sorted(input_dir.iterdir()):
        if not video_dir.is_dir() or video_dir.name.startswith("."):
            continue
        for f in sorted(video_dir.iterdir()):
            if f.is_file() and f.suffix in SUPPORTED:
                scenes.append(f)
    return scenes


def claim_next(input_dir: Path) -> Path | None:
    with _scan_lock:
        for scene in scan_scenes(input_dir):
            if try_claim(scene):
                return scene
    return None


# ===== WORKER THREAD =====

def worker_thread(tid: int, input_dir: Path, output_base: str):
    pid = os.getpid()
    tag = f"[{_hostname}:{pid}:T{tid}]"
    logger.info(f"{tag} started")

    while True:
        scene = claim_next(input_dir)

        if scene is None:
            pending = [
                s for s in scan_scenes(input_dir)
                if not _done_f(s).exists() and not _error_f(s).exists()
            ]
            if not pending:
                logger.info(f"{tag} No more scenes. Exiting.")
                break
            logger.info(f"{tag} Waiting ({len(pending)} pending)...")
            time.sleep(SCAN_INTERVAL)
            continue

        video_name = scene.parent.name
        label      = f"{video_name}/{scene.name}"
        logger.info(f"{tag} Claimed: {label}")

        hb = Heartbeat(scene)
        t0 = time.time()
        try:
            out = str(Path(output_base) / video_name)
            process_scene(str(scene), out)
            mark_done(scene)
            logger.info(f"{tag} Done: {label} ({time.time()-t0:.1f}s)")
        except Exception as e:
            mark_error(scene, str(e))
            logger.error(f"{tag} Error: {label}: {e}\n{traceback.format_exc()}")
        finally:
            hb.stop()


# ===== MAIN =====

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",   help="Folder chứa các video subfolder")
    parser.add_argument("output_base", help="Folder output gốc")
    parser.add_argument("--parallel",  type=int, default=1,
                        help="Số scene xử lý song song (default: 1)")
    args = parser.parse_args()

    input_dir   = Path(args.input_dir)
    output_base = args.output_base
    n           = args.parallel

    if not input_dir.is_dir():
        print(f"ERROR: {input_dir} is not a directory")
        sys.exit(1)

    Path(output_base).mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(output_base)

    logger.info(f"[{_hostname}:{os.getpid()}] Worker started | parallel={n}")
    logger.info(f"  Input : {input_dir.resolve()}")
    logger.info(f"  Output: {Path(output_base).resolve()}")
    logger.info(f"  Log   : {log_path}")

    if n == 1:
        worker_thread(1, input_dir, output_base)
    else:
        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = [ex.submit(worker_thread, i + 1, input_dir, output_base)
                       for i in range(n)]
            for f in as_completed(futures):
                f.result()

    logger.info(f"[{_hostname}:{os.getpid()}] All workers finished.")


if __name__ == "__main__":
    main()
