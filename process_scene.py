#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Usage: python process_scene.py <scene_video_path> <output_dir>
"""

import os, shutil, subprocess, sys, time, pickle, glob, gc, warnings, threading, logging
import cv2
import numpy as np
np.int = int
import torch
import requests
from pathlib import Path
from shutil import rmtree
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import interp1d
from scipy import signal

from detectors import S3FD
from SyncNetInstance import SyncNetInstance
from torchvision import transforms
from models import get_model, SCRFD
from utils.general import compute_euler_angles_from_rotation_matrices

warnings.filterwarnings("ignore")

# ===== CONFIG =====
BASE_CODE_DIR = Path(".")
_CPU_COUNT = os.cpu_count() or 8
_CPU_WORKERS = max(4, min(8, _CPU_COUNT))

CONFIG = {
    "initial_model":        str(BASE_CODE_DIR / "data/syncnet_v2.model"),
    "head_pose_weights":    str(BASE_CODE_DIR / "weights/resnet18.pt"),
    "head_pose_network":    "resnet18",
    "head_pose_det_model":  str(BASE_CODE_DIR / "weights/det_10g.onnx"),
    "head_pose_angle_thresh": 45.0,
    "head_pose_ratio_thresh": 0.10,
    "facedet_scale":        0.25,
    "crop_scale":           0.40,
    "min_track":            50,
    "frame_rate":           30,
    "num_failed_det":       25,
    "min_face_size":        100,
    "batch_size":           20,
    "vshift":               15,
    "min_confidence":       3.0,
    "offset_min":           -5,
    "offset_max":           5,
    # "asr_api_url":          "https://implicatively-unpale-susann.ngrok-free.dev/transcribe",
    "asr_api_url":          "http://localhost:8000/transcribe",
    "asr_language":         "zh",
    "asr_use_punctuation":  True,
    "asr_full_text_mode":   True,
    "asr_timeout":          120,
    "max_audio_duration":   25.0,
}

# ===== MODEL CACHE =====
_s3fd_models:     dict = {}
_syncnet_models:  dict = {}
_headpose_models: dict = {}
_models_lock = threading.Lock()


def _load_s3fd(device):
    if device not in _s3fd_models:
        with _models_lock:
            if device not in _s3fd_models:
                _s3fd_models[device] = S3FD(device=device)
    return _s3fd_models[device]


def _load_syncnet(model_path, device):
    if device not in _syncnet_models:
        with _models_lock:
            if device not in _syncnet_models:
                s = SyncNetInstance(device=device)
                s.loadParameters(model_path)
                _syncnet_models[device] = s
    return _syncnet_models[device]


def _load_head_pose_models(device):
    key = str(device)
    if key not in _headpose_models:
        with _models_lock:
            if key not in _headpose_models:
                face_det   = SCRFD(model_path=CONFIG["head_pose_det_model"])
                pose_model = get_model(CONFIG["head_pose_network"], num_classes=6, pretrained=False)
                state      = torch.load(CONFIG["head_pose_weights"], map_location=device)
                pose_model.load_state_dict(state)
                pose_model.to(device).eval()
                _headpose_models[key] = (pose_model, face_det)
    return _headpose_models[key]


# ===== HEAD POSE =====
_FACE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def _preprocess_face(image):
    return _FACE_TRANSFORM(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0)

def _expand_bbox(x_min, y_min, x_max, y_max, factor=0.2):
    w, h = x_max - x_min, y_max - y_min
    return (max(0, x_min - int(factor * h)), max(0, y_min - int(factor * w)),
            x_max + int(factor * h), y_max + int(factor * w))


def check_head_pose_video(video_path, device, batch_size=32, frame_step=5):
    try:
        pose_model, face_det = _load_head_pose_models(device)
    except:
        return {"approved": True}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"approved": False}

    face_crops, total_frames = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if total_frames % frame_step == 0:
            fh, fw = frame.shape[:2]
            bboxes, kps = face_det.detect(frame)
            for bbox, _ in zip(bboxes, kps):
                x1, y1, x2, y2 = _expand_bbox(*map(int, bbox[:4]))
                x2, y2 = min(x2, fw), min(y2, fh)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    face_crops.append((total_frames, crop))
        total_frames += 1
    cap.release()

    if not total_frames or not face_crops:
        return {"approved": True}

    angle_thresh = CONFIG["head_pose_angle_thresh"]
    excessive = set()
    with torch.no_grad():
        for i in range(0, len(face_crops), batch_size):
            batch = face_crops[i:i + batch_size]
            idxs = [x[0] for x in batch]
            t = torch.cat([_preprocess_face(c) for _, c in batch], dim=0).to(device)
            angles = np.degrees(
                compute_euler_angles_from_rotation_matrices(pose_model(t).cpu()).numpy())
            for j, fidx in enumerate(idxs):
                if any(abs(angles[j, k]) > angle_thresh for k in range(3)):
                    excessive.add(fidx)

    ratio = len(excessive) / total_frames
    return {"approved": ratio <= CONFIG["head_pose_ratio_thresh"]}


def filter_by_head_pose(opt):
    crop_dir     = os.path.join(opt.data_dir, "pycrop", opt.reference)
    rejected_dir = os.path.join(opt.data_dir, "pycrop_headpose_rejected", opt.reference)
    os.makedirs(rejected_dir, exist_ok=True)
    crop_files = sorted(glob.glob(os.path.join(crop_dir, "0*.avi")))
    if not crop_files:
        return

    with ThreadPoolExecutor(max_workers=min(len(crop_files), _CPU_WORKERS)) as ex:
        futures = {ex.submit(check_head_pose_video, cp, opt.device): cp for cp in crop_files}
        for future in as_completed(futures):
            cp = futures[future]
            if not future.result().get("approved", True):
                shutil.move(cp, os.path.join(rejected_dir, os.path.basename(cp)))


# ===== CORE PIPELINE =====
def bb_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0, xB - xA) * max(0, yB - yA)
    return inter / float((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def track_shot(opt, scenefaces):
    tracks = []
    while True:
        track = []
        for framefaces in scenefaces:
            for face in framefaces:
                if not track:
                    track.append(face); framefaces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= opt.num_failed_det:
                    if bb_iou(face["bbox"], track[-1]["bbox"]) > 0.5:
                        track.append(face); framefaces.remove(face)
                    continue
                else:
                    break
        if not track:
            break
        if len(track) > opt.min_track:
            framenum = np.array([f["frame"] for f in track])
            bboxes   = np.array([f["bbox"]  for f in track])
            frame_i  = np.arange(framenum[0], framenum[-1] + 1)
            bboxes_i = np.stack([interp1d(framenum, bboxes[:, j])(frame_i) for j in range(4)], axis=1)
            if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]),
                   np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > opt.min_face_size:
                tracks.append({"frame": frame_i, "bbox": bboxes_i})
    return tracks


def crop_video(opt, track, cropfile, avi_path):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vOut   = cv2.VideoWriter(cropfile + "t.avi", fourcc, opt.frame_rate, (224, 224))
    dets   = {k: [] for k in ("x", "y", "s")}
    for det in track["bbox"]:
        dets["s"].append(max(det[3]-det[1], det[2]-det[0]) / 2)
        dets["y"].append((det[1]+det[3]) / 2)
        dets["x"].append((det[0]+det[2]) / 2)
    for k in dets:
        dets[k] = signal.medfilt(dets[k], kernel_size=13)

    cap = cv2.VideoCapture(avi_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(track["frame"][0]))
    for fidx in range(len(track["frame"])):
        ret, img = cap.read()
        if not ret: break
        cs  = opt.crop_scale
        bs  = dets["s"][fidx]
        bsi = int(bs * (1 + 2 * cs))
        pad = np.pad(img, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110,))
        my, mx = dets["y"][fidx] + bsi, dets["x"][fidx] + bsi
        face = pad[int(my-bs):int(my+bs*(1+2*cs)), int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    cap.release()

    stem      = Path(cropfile).stem
    audiotmp  = os.path.join(opt.tmp_dir, opt.reference, f"audio_{stem}.wav")
    audiostart = track["frame"][0] / opt.frame_rate
    audioend   = (track["frame"][-1] + 1) / opt.frame_rate
    vOut.release()

    ref_avi = os.path.join(opt.avi_dir, opt.reference)
    subprocess.call(
        f"ffmpeg -y -i {ref_avi}/audio.wav -ss {audiostart:.3f} -to {audioend:.3f} {audiotmp}",
        shell=True, stderr=subprocess.DEVNULL)
    subprocess.call(
        f"ffmpeg -y -i {cropfile}t.avi -i {audiotmp} -c:v copy -c:a copy {cropfile}.avi",
        shell=True, stderr=subprocess.DEVNULL)
    os.remove(cropfile + "t.avi")
    return {"track": track, "proc_track": dets}


def inference_video(opt, batch_size=32):
    DET      = _load_s3fd(opt.device)
    avi_path = os.path.join(opt.avi_dir, opt.reference, "video.avi")
    cap      = cv2.VideoCapture(avi_path)
    dets, batch, start = [], [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if batch:
                bboxes = DET.detect_faces_batch(batch, conf_th=0.9, scales=[opt.facedet_scale])
                for fidx, bb in zip(range(start, start + len(batch)), bboxes):
                    dets.append([{"frame": fidx, "bbox": b[:-1].tolist(), "conf": float(b[-1])} for b in bb])
            break
        batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(batch) == batch_size:
            bboxes = DET.detect_faces_batch(batch, conf_th=0.9, scales=[opt.facedet_scale])
            for fidx, bb in zip(range(start, start + batch_size), bboxes):
                dets.append([{"frame": fidx, "bbox": b[:-1].tolist(), "conf": float(b[-1])} for b in bb])
            start += batch_size
            batch = []
    cap.release()
    with open(os.path.join(opt.work_dir, opt.reference, "faces.pckl"), "wb") as f:
        pickle.dump(dets, f)
    return dets


logger = logging.getLogger("process_scene")


def _t(label, t0):
    import time
    elapsed = time.time() - t0
    logger.info(f"    [{label}] {elapsed:.2f}s")
    return time.time()


def run_pipeline(opt):
    import time
    for d in [opt.work_dir, opt.crop_dir, opt.avi_dir, opt.tmp_dir]:
        p = os.path.join(d, opt.reference)
        if os.path.exists(p): rmtree(p)
        os.makedirs(p)

    ref_avi  = os.path.join(opt.avi_dir, opt.reference)
    avi_path = f"{ref_avi}/video.avi"
    wav_path = f"{ref_avi}/audio.wav"

    # convert video + extract audio in parallel (both read from original source)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(subprocess.call,
            f"ffmpeg -y -i {opt.videofile} -qscale:v 2 -async 1 -r 25 {avi_path}",
            shell=True, stderr=subprocess.DEVNULL)
        f2 = ex.submit(subprocess.call,
            f"ffmpeg -y -i {opt.videofile} -ac 1 -vn -acodec pcm_s16le -ar 16000 {wav_path}",
            shell=True, stderr=subprocess.DEVNULL)
        f1.result(); f2.result()
    t0 = _t("ffmpeg convert+audio (parallel)", t0)

    faces = inference_video(opt)
    t0 = _t("face detection (S3FD)", t0)

    alltracks = track_shot(opt, faces)
    t0 = _t("track_shot", t0)

    vidtracks = [None] * len(alltracks)
    with ThreadPoolExecutor(max_workers=min(len(alltracks) or 1, _CPU_WORKERS)) as ex:
        futures = {
            ex.submit(crop_video, opt, track,
                      os.path.join(opt.crop_dir, opt.reference, "%05d" % ii),
                      avi_path): ii
            for ii, track in enumerate(alltracks)
        }
        for future in as_completed(futures):
            vidtracks[futures[future]] = future.result()
    _t(f"crop_video ({len(alltracks)} tracks)", t0)

    with open(os.path.join(opt.work_dir, opt.reference, "tracks.pckl"), "wb") as f:
        pickle.dump(vidtracks, f)
    rmtree(os.path.join(opt.tmp_dir, opt.reference))


# ===== SYNCNET =====
def run_syncnet(opt):
    s     = _load_syncnet(opt.initial_model, opt.device)
    flist = sorted(glob.glob(os.path.join(opt.crop_dir, opt.reference, "0*.avi")))
    if not flist:
        return

    prepared = [None] * len(flist)
    with ThreadPoolExecutor(max_workers=min(len(flist), max(_CPU_WORKERS, 16))) as ex:
        futures = {ex.submit(s.prepare_data, fname): i for i, fname in enumerate(flist)}
        for future in as_completed(futures):
            prepared[futures[future]] = future.result()

    results = []
    for idx, (fname, data) in enumerate(zip(flist, prepared)):
        if data is None: continue
        offset, conf, dist = s.evaluate_tensors(opt, *data)
        dist_str = " ".join(map(str, dist.flatten())) if hasattr(dist, "flatten") else str(dist)
        results.append({"index": idx, "file": os.path.basename(fname),
                        "offset": offset, "conf": conf, "dist": dist_str})

    with open(os.path.join(opt.work_dir, opt.reference, "activesd.txt"), "w") as f:
        f.write("# index\tfile\toffset\tconfidence\tdist\n")
        for r in results:
            f.write(f'{r["index"]}\t{r["file"]}\t{r["offset"]}\t{r["conf"]}\t{r["dist"]}\n')


# ===== FILTER =====
def filter_videos(opt):
    result_file = os.path.join(opt.work_dir, opt.reference, "activesd.txt")
    if not os.path.exists(result_file):
        return
    results = []
    with open(result_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split("\t")
            if len(parts) < 4: continue
            try:
                results.append({"index": int(parts[0]), "filename": parts[1],
                                 "offset": int(parts[2]), "confidence": float(parts[3])})
            except: pass

    good = [r for r in results if r["confidence"] >= opt.min_confidence
            and opt.offset_min <= r["offset"] <= opt.offset_max]
    bad  = [r for r in results if r not in good]
    crop_dir = os.path.join(opt.data_dir, "pycrop")
    for label, lst in [("save_good", good), ("save_bad", bad)]:
        dst = os.path.join(getattr(opt, label), opt.reference)
        os.makedirs(dst, exist_ok=True)
        for r in lst:
            src = os.path.join(crop_dir, opt.reference, r["filename"])
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst, r["filename"]))


# ===== ASR =====
def get_audio_duration(audio_path):
    import wave
    try:
        with wave.open(str(audio_path)) as wf:
            return wf.getnframes() / float(wf.getframerate())
    except: return 0.0


def call_asr_api(audio_path):
    with open(audio_path, "rb") as f:
        resp = requests.post(
            CONFIG["asr_api_url"],
            params={"language": CONFIG["asr_language"],
                    "use_punctuation": str(CONFIG["asr_use_punctuation"]).lower(),
                    "full_text_mode":  str(CONFIG["asr_full_text_mode"]).lower()},
            files={"file": (audio_path.name, f, "audio/wav")},
            timeout=CONFIG["asr_timeout"])
    resp.raise_for_status()
    return resp.json().get("text", "").strip()


def save_chunk(src_mp4, src_wav, start, end, video_out, audio_out, text_out):
    dur = end - start
    r1 = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src_mp4),
         "-ss", f"{start:.3f}", "-i", str(src_wav),
         "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "fast",
         "-c:a", "aac", "-ar", "44100", "-ac", "1",
         "-map", "0:v:0", "-map", "1:a:0", str(video_out)], capture_output=True)
    r2 = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src_wav),
         "-t", f"{dur:.3f}", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
         str(audio_out)], capture_output=True)
    if r1.returncode != 0 or r2.returncode != 0:
        return False
    try:
        text_out.write_text(call_asr_api(audio_out), encoding="utf-8")
        return True
    except: return False


def run_asr(opt, output_dir: Path):
    input_path = Path(opt.save_good) / opt.reference
    tmp_dir    = output_dir / "_tmp"
    video_dir  = output_dir / "videos"
    audio_dir  = output_dir / "audios"
    trans_dir  = output_dir / "transcripts"
    for d in [video_dir, audio_dir, trans_dir, tmp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    avi_files = sorted(input_path.glob("*.avi"))
    if not avi_files:
        return

    max_dur = CONFIG["max_audio_duration"]

    def _process_one(avi_file):
        uid     = f"{avi_file.stem}_{threading.get_ident()}"
        tmp_mp4 = tmp_dir / f"{uid}.mp4"
        tmp_wav = tmp_dir / f"{uid}.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(avi_file), "-c:v", "libx264",
                        "-c:a", "aac", "-preset", "fast", str(tmp_mp4)], capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(avi_file), "-vn", "-acodec", "pcm_s16le",
                        "-ar", "16000", "-ac", "1", str(tmp_wav)], capture_output=True)
        if not tmp_wav.exists() or os.path.getsize(str(tmp_wav)) < 2000:
            tmp_mp4.unlink(missing_ok=True); tmp_wav.unlink(missing_ok=True)
            return []

        audio_dur   = get_audio_duration(tmp_wav)
        chunk_starts = []
        s = 0.0
        while s < audio_dur:
            chunk_starts.append(s)
            if audio_dur <= max_dur: break
            s += max_dur

        chunks = []
        for ci, cs in enumerate(chunk_starts):
            ce = min(cs + max_dur, audio_dur)
            if ce - cs < 1.0: break
            t_vid = tmp_dir / f"{uid}_c{ci}_vid.mp4"
            t_aud = tmp_dir / f"{uid}_c{ci}_aud.wav"
            t_txt = tmp_dir / f"{uid}_c{ci}.txt"
            if save_chunk(tmp_mp4, tmp_wav, cs, ce, t_vid, t_aud, t_txt):
                chunks.append((t_vid, t_aud, t_txt))

        tmp_mp4.unlink(missing_ok=True); tmp_wav.unlink(missing_ok=True)
        return chunks

    all_chunks = []
    with ThreadPoolExecutor(max_workers=max(4, _CPU_WORKERS * 2)) as ex:
        futures = {ex.submit(_process_one, f): f for f in avi_files}
        for future in as_completed(futures):
            try: all_chunks.extend(future.result())
            except: pass

    for idx, (t_vid, t_aud, t_txt) in enumerate(all_chunks, 1):
        fid = f"{idx:06d}"
        t_vid.rename(video_dir / f"{fid}.mp4")
        t_aud.rename(audio_dir / f"{fid}.wav")
        t_txt.rename(trans_dir / f"{fid}.txt")

    try: shutil.rmtree(tmp_dir)
    except: pass


# ===== MAIN =====
class _Opt:
    def __init__(self, d):
        for k, v in d.items(): setattr(self, k, v)


def process_scene(video_path: str, output_base: str):
    video_path = Path(video_path).resolve()
    video_name = video_path.stem
    output_dir = Path(output_base) / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device    = "cuda:0" if torch.cuda.is_available() else "cpu"
    workspace  = output_dir / "_workspace"

    opt = _Opt({
        **CONFIG,
        "videofile":  str(video_path),
        "reference":  video_name,
        "device":     device,
        "data_dir":   str(workspace),
        "save_good":  str(workspace / "save_good"),
        "save_bad":   str(workspace / "save_bad"),
        "avi_dir":    str(workspace / "pyavi"),
        "tmp_dir":    str(workspace / "pytmp"),
        "work_dir":   str(workspace / "pywork"),
        "crop_dir":   str(workspace / "pycrop"),
    })

    import time
    steps = [
        ("run_pipeline",        lambda: run_pipeline(opt)),
        ("filter_by_head_pose", lambda: filter_by_head_pose(opt)),
        ("run_syncnet",         lambda: run_syncnet(opt)),
        ("filter_videos",       lambda: filter_videos(opt)),
        ("run_asr",             lambda: run_asr(opt, output_dir)),
    ]
    timings = {}
    t_total = time.time()
    for name, fn in steps:
        t0 = time.time()
        fn()
        timings[name] = time.time() - t0
        logger.info(f"  [{name}] {timings[name]:.2f}s")

    try: rmtree(workspace)
    except: pass

    total = len(list((output_dir / "videos").glob("*.mp4"))) if (output_dir / "videos").exists() else 0
    logger.info(f"Done: {total} clips → {output_dir} | total={time.time()-t_total:.2f}s")
    return str(output_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_scene.py <input_folder|video_path> <output_base>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_base = sys.argv[2]

    if input_path.is_dir():
        scene_files = sorted(input_path.glob("*.mp4")) + sorted(input_path.glob("*.MP4"))
        if not scene_files:
            print(f"No MP4 files found in {input_path}")
            sys.exit(1)
        print(f"Found {len(scene_files)} scenes in {input_path}")
        for scene in scene_files:
            print(f"\n=== {scene.name} ===")
            process_scene(str(scene), output_base)
    else:
        process_scene(str(input_path), output_base)
