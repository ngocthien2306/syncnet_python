#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
AVSR Pipeline - CLI Version
Chạy: python run.py
Thả video vào ./input_videos/ rồi chạy lại
"""

import os, shutil, subprocess, sys, time, pickle, glob, gc, fcntl, json, hashlib, warnings, logging
import cv2
import numpy as np
np.int = int
import torch
from pathlib import Path
from shutil import rmtree
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal
import requests

import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.detectors import ContentDetector

from detectors import S3FD
from SyncNetInstance import SyncNetInstance
from torchvision import transforms
from models import get_model, SCRFD
from utils.general import compute_euler_angles_from_rotation_matrices

warnings.filterwarnings("ignore")

# ==================== CONFIG ====================
BASE_CODE_DIR  = Path(".")

CONFIG = {
    # --- THƯ MỤC ---
    "input_dir":            "./input_videos",
    "output_dir":           str(BASE_CODE_DIR / "results"),
    "workspace_dir":        str(BASE_CODE_DIR / "cli_workspace"),
    "log_file":             str(BASE_CODE_DIR / "pipeline.log"),

    # --- MODEL ---
    "initial_model":        str(BASE_CODE_DIR / "data/syncnet_v2.model"),
    "head_pose_weights":    str(BASE_CODE_DIR / "weights/resnet18.pt"),
    "head_pose_network":    "resnet18",
    "head_pose_det_model":  str(BASE_CODE_DIR / "weights/det_10g.onnx"),

    # --- HEAD POSE ---
    "head_pose_angle_thresh": 45.0,
    "head_pose_ratio_thresh": 0.10,

    # --- FACE DETECTION ---
    "facedet_scale":        0.25,
    "crop_scale":           0.40,
    "min_track":            50,
    "frame_rate":           25,
    "num_failed_det":       25,
    "min_face_size":        100,

    # --- SYNCNET ---
    "batch_size":           20,
    "vshift":               15,

    # --- FILTER ---
    "min_confidence":       3.0,
    "offset_min":           -5,
    "offset_max":           5,

    # --- ASR API ---
    "asr_api_url":          "https://implicatively-unpale-susann.ngrok-free.dev/transcribe",
    "asr_language":         "zh",
    "asr_use_punctuation":  True,
    "asr_full_text_mode":   True,
    "asr_timeout":          120,

    # --- ASR CHUNK ---
    "max_audio_duration":   25.0,

    # --- PARALLEL ---
    # Auto-scale to number of GPUs × slots per GPU
    "max_workers":          max(torch.cuda.device_count(), 1) * _SLOTS_PER_GPU if torch.cuda.is_available() else 1,
}

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
DONE_LOG_FILE = Path(CONFIG["output_dir"]) / ".done_videos.json"
LOCK_FILE     = Path(CONFIG["output_dir"]) / ".queue.lock"

# ==================== LOGGING ====================
Path(CONFIG["log_file"]).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(CONFIG["log_file"], encoding="utf-8"),
    ]
)
logger = logging.getLogger("AVSR")

# ==================== GLOBAL MODEL CACHE ====================
import threading
import queue as _queue_mod

# Per-device model caches (key = device string e.g. "cuda:0", "cpu")
_s3fd_models:     dict = {}   # device -> S3FD
_syncnet_models:  dict = {}   # device -> SyncNetInstance
_headpose_models: dict = {}   # device -> (pose_model, face_det)
_models_lock = threading.Lock()

# GPU pool — each entry is a device string; size = number of available GPUs
def _build_gpu_pool(slots_per_gpu: int = 1) -> _queue_mod.Queue:
    pool = _queue_mod.Queue()
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n > 0:
        for i in range(n):
            for _ in range(slots_per_gpu):
                pool.put(f"cuda:{i}")
    else:
        pool.put("cpu")
    return pool

_SLOTS_PER_GPU = 4   # scenes sharing each GPU concurrently; tune based on GPU memory
_gpu_pool = _build_gpu_pool(slots_per_gpu=_SLOTS_PER_GPU)

# Legacy alias kept for head-pose serialization (CPU-bound SCRFD ONNX session)
_headpose_lock = threading.Lock()

# ==================== FILE LOCK (multi-terminal safe) ====================
class FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def __enter__(self):
        self._fh = open(self.path, "w")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *args):
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()


def load_done_log() -> dict:
    if DONE_LOG_FILE.exists():
        with open(DONE_LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_done_log(done: dict):
    DONE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DONE_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(done, f, indent=2, ensure_ascii=False)


def claim_next_video(input_dir: Path) -> Path | None:
    """Lấy 1 video chưa xử lý, đánh dấu processing ngay để terminal khác không lấy trùng"""
    with FileLock(LOCK_FILE):
        done = load_done_log()
        for video in sorted(input_dir.iterdir()):
            if not video.is_file() or video.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            status = done.get(video.name, {}).get("status")
            if status in ("done", "processing"):
                continue
            done[video.name] = {"status": "processing", "pid": os.getpid(),
                                  "time": datetime.now().isoformat()}
            save_done_log(done)
            return video
    return None


def mark_done(video_name: str, output_folder: str):
    with FileLock(LOCK_FILE):
        done = load_done_log()
        done[video_name] = {"status": "done", "output": output_folder,
                             "pid": os.getpid(), "time": datetime.now().isoformat()}
        save_done_log(done)


def mark_error(video_name: str, error: str):
    with FileLock(LOCK_FILE):
        done = load_done_log()
        done[video_name] = {"status": "error", "error": error,
                             "pid": os.getpid(), "time": datetime.now().isoformat()}
        save_done_log(done)


_folder_lock = threading.Lock()
def get_next_result_folder(output_base: Path) -> Path:
    with _folder_lock:
        existing = sorted(output_base.glob("result*"))
        max_num = 0
        for folder in existing:
            try:
                max_num = max(max_num, int(folder.name.replace("result", "")))
            except:
                pass
        folder = output_base / f"result{max_num + 1}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder


# ==================== HELPER ====================
class DictToObject:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.debug("GPU memory cleared.")


def _load_s3fd(device: str) -> "S3FD":
    if device in _s3fd_models:
        return _s3fd_models[device]
    with _models_lock:
        if device not in _s3fd_models:
            logger.info(f"Loading S3FD on {device}...")
            _s3fd_models[device] = S3FD(device=device)
            logger.info(f"S3FD loaded on {device} OK.")
    return _s3fd_models[device]


def _load_syncnet(model_path: str, device: str) -> "SyncNetInstance":
    if device in _syncnet_models:
        return _syncnet_models[device]
    with _models_lock:
        if device not in _syncnet_models:
            logger.info(f"Loading SyncNet on {device}...")
            s = SyncNetInstance(device=device)
            s.loadParameters(model_path)
            _syncnet_models[device] = s
            logger.info(f"SyncNet loaded on {device} OK.")
    return _syncnet_models[device]


# ==================== HEAD POSE ====================
def _load_head_pose_models(device):
    """Load (and cache) head pose models per device string."""
    device_str = str(device)
    if device_str in _headpose_models:
        return _headpose_models[device_str]
    with _models_lock:
        if device_str not in _headpose_models:
            logger.info(f"Loading head pose models on {device}...")
            face_det   = SCRFD(model_path=CONFIG["head_pose_det_model"])
            pose_model = get_model(CONFIG["head_pose_network"], num_classes=6, pretrained=False)
            state_dict = torch.load(CONFIG["head_pose_weights"], map_location=device)
            pose_model.load_state_dict(state_dict)
            pose_model.to(device).eval()
            _headpose_models[device_str] = (pose_model, face_det)
            logger.info(f"Head pose models loaded on {device} OK.")
    return _headpose_models[device_str]


_FACE_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def _preprocess_face(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return _FACE_TRANSFORM(image).unsqueeze(0)


def _expand_bbox(x_min, y_min, x_max, y_max, factor=0.2):
    w, h = x_max - x_min, y_max - y_min
    return (max(0, x_min - int(factor * h)), max(0, y_min - int(factor * w)),
            x_max + int(factor * h), y_max + int(factor * w))


def check_head_pose_video(video_path: str, device=None, batch_size: int = 32) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # No _headpose_lock needed: model loading is protected by _models_lock inside
    # _load_head_pose_models; ONNX Runtime and PyTorch inference are thread-safe.
    try:
        pose_model, face_det = _load_head_pose_models(device)
    except Exception as e:
        logger.warning(f"check_head_pose_video: model unavailable => {e}")
        return {"approved": True, "total_frames": 0, "excessive_frames": 0,
                "excessive_ratio": 0.0, "reason": "model_unavailable"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"approved": False, "total_frames": 0, "excessive_frames": 0,
                "excessive_ratio": 0.0, "reason": "cannot_open_video"}

    angle_thresh = CONFIG["head_pose_angle_thresh"]
    # Collect all face crops across all frames first
    face_crops = []   # list of (frame_idx, face_crop_array)
    total_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fh, fw = frame.shape[:2]
        bboxes, keypoints = face_det.detect(frame)
        for bbox, _ in zip(bboxes, keypoints):
            x_min, y_min, x_max, y_max = _expand_bbox(*map(int, bbox[:4]))
            x_max, y_max = min(x_max, fw), min(y_max, fh)  # clip to frame bounds
            face_crop = frame[y_min:y_max, x_min:x_max]
            if face_crop.size == 0:
                continue
            face_crops.append((total_frames, face_crop))
        total_frames += 1
    cap.release()

    if total_frames == 0:
        return {"approved": False, "total_frames": 0, "excessive_frames": 0,
                "excessive_ratio": 0.0, "reason": "no_frames"}
    if not face_crops:
        return {"approved": True, "total_frames": total_frames, "excessive_frames": 0,
                "excessive_ratio": 0.0, "reason": "no_faces_detected"}

    # Batch pose inference over all collected face crops
    excessive_frame_set = set()
    with torch.no_grad():
        for i in range(0, len(face_crops), batch_size):
            batch = face_crops[i:i + batch_size]
            frame_indices = [x[0] for x in batch]
            tensors = torch.cat([_preprocess_face(c) for _, c in batch], dim=0).to(device)
            rotation_matrices = pose_model(tensors).cpu()
            euler_angles = np.degrees(
                compute_euler_angles_from_rotation_matrices(rotation_matrices).numpy())
            for j, fidx in enumerate(frame_indices):
                pitch, yaw, roll = float(euler_angles[j, 0]), float(euler_angles[j, 1]), float(euler_angles[j, 2])
                if abs(pitch) > angle_thresh or abs(yaw) > angle_thresh or abs(roll) > angle_thresh:
                    excessive_frame_set.add(fidx)

    excessive_frames = len(excessive_frame_set)
    excessive_ratio = excessive_frames / total_frames
    approved = excessive_ratio <= CONFIG["head_pose_ratio_thresh"]
    reason = f"{'OK' if approved else 'Excessive'} head pose: {excessive_frames}/{total_frames} ({excessive_ratio*100:.1f}%)"
    return {"approved": approved, "total_frames": total_frames,
            "excessive_frames": excessive_frames, "excessive_ratio": excessive_ratio, "reason": reason}


def filter_by_head_pose(opt) -> dict:
    crop_dir     = os.path.join(opt.data_dir, "pycrop", opt.reference)
    rejected_dir = os.path.join(opt.data_dir, "pycrop_headpose_rejected", opt.reference)
    os.makedirs(rejected_dir, exist_ok=True)
    crop_files = sorted(glob.glob(os.path.join(crop_dir, "0*.avi")))
    if not crop_files:
        return {"total": 0, "approved": 0, "rejected": 0}
    total = len(crop_files)
    logger.info(f"  filter_by_head_pose [{opt.reference}]: checking {total} crops...")
    device = getattr(opt, "device", None)

    # Check all crop files in parallel: SCRFD (ONNX) + VideoCapture are CPU/IO-bound
    # and thread-safe; GPU pose inference serializes automatically on the same device.
    hp_workers = min(total, 4)
    results_map = {}
    with ThreadPoolExecutor(max_workers=hp_workers) as executor:
        futures = {executor.submit(check_head_pose_video, cp, device): cp
                   for cp in crop_files}
        for future in as_completed(futures):
            cp = futures[future]
            results_map[cp] = future.result()

    approved = rejected = 0
    for crop_path in crop_files:
        result = results_map[crop_path]
        if not result["approved"]:
            shutil.move(crop_path, os.path.join(rejected_dir, os.path.basename(crop_path)))
            rejected += 1
        else:
            approved += 1
    logger.info(f"  filter_by_head_pose [{opt.reference}]: total={total}, approved={approved}, rejected={rejected}")
    return {"total": total, "approved": approved, "rejected": rejected}


# ==================== CORE PIPELINE ====================
def bb_intersection_over_union(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea  = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea  = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def track_shot(opt, scenefaces):
    iouThres = 0.5
    tracks = []
    while True:
        track = []
        for framefaces in scenefaces:
            for face in framefaces:
                if track == []:
                    track.append(face); framefaces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= opt.num_failed_det:
                    if bb_intersection_over_union(face["bbox"], track[-1]["bbox"]) > iouThres:
                        track.append(face); framefaces.remove(face)
                    continue
                else:
                    break
        if track == []:
            break
        elif len(track) > opt.min_track:
            framenum  = np.array([f["frame"] for f in track])
            bboxes    = np.array([np.array(f["bbox"]) for f in track])
            frame_i   = np.arange(framenum[0], framenum[-1] + 1)
            bboxes_i  = []
            for ij in range(4):
                interpfn = interp1d(framenum, bboxes[:, ij])
                bboxes_i.append(interpfn(frame_i))
            bboxes_i = np.stack(bboxes_i, axis=1)
            if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]),
                   np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > opt.min_face_size:
                tracks.append({"frame": frame_i, "bbox": bboxes_i})
    return tracks


def crop_video(opt, track, cropfile, flist=None):
    if flist is None:
        flist = sorted(glob.glob(os.path.join(opt.frames_dir, opt.reference, "*.jpg")))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vOut   = cv2.VideoWriter(cropfile + "t.avi", fourcc, opt.frame_rate, (224, 224))
    dets   = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)
        dets["x"].append((det[0] + det[2]) / 2)
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)
    for fidx, frame in enumerate(track["frame"]):
        cs  = opt.crop_scale
        bs  = dets["s"][fidx]
        bsi = int(bs * (1 + 2 * cs))
        image     = cv2.imread(flist[frame])
        frame_pad = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
        my   = dets["y"][fidx] + bsi
        mx   = dets["x"][fidx] + bsi
        face = frame_pad[int(my - bs):int(my + bs * (1 + 2 * cs)),
                         int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
        vOut.write(cv2.resize(face, (224, 224)))
    # Use cropfile stem as unique name to avoid race condition in parallel calls
    audiotmp   = os.path.join(opt.tmp_dir, opt.reference, f"audio_{Path(cropfile).stem}.wav")
    audiostart = track["frame"][0] / opt.frame_rate
    audioend   = (track["frame"][-1] + 1) / opt.frame_rate
    vOut.release()
    subprocess.call(
        f"ffmpeg -y -i {os.path.join(opt.avi_dir, opt.reference, 'audio.wav')} "
        f"-ss {audiostart:.3f} -to {audioend:.3f} {audiotmp}",
        shell=True, stderr=subprocess.DEVNULL)
    subprocess.call(
        f"ffmpeg -y -i {cropfile}t.avi -i {audiotmp} -c:v copy -c:a copy {cropfile}.avi",
        shell=True, stderr=subprocess.DEVNULL)
    os.remove(cropfile + "t.avi")
    return {"track": track, "proc_track": dets}


def inference_video(opt):
    logger.info(f"  inference_video [{opt.reference}]: device={opt.device}")
    DET   = _load_s3fd(opt.device)
    flist = sorted(glob.glob(os.path.join(opt.frames_dir, opt.reference, "*.jpg")))
    logger.info(f"  inference_video [{opt.reference}]: {len(flist)} frames")
    dets  = []
    for fidx, fname in enumerate(flist):
        image    = cv2.imread(fname)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes   = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])
        dets.append([{"frame": fidx, "bbox": (bbox[:-1]).tolist(), "conf": bbox[-1]}
                      for bbox in bboxes])
    savepath = os.path.join(opt.work_dir, opt.reference, "faces.pckl")
    with open(savepath, "wb") as f:
        pickle.dump(dets, f)
    return dets


def scene_detect(opt):
    logger.info(f"  scene_detect [{opt.reference}]: running scenedetect...")
    video_path = os.path.join(opt.avi_dir, opt.reference, "video.avi")
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager, frame_skip=3)
    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]
    savepath = os.path.join(opt.work_dir, opt.reference, "scene.pckl")
    with open(savepath, "wb") as f:
        pickle.dump(scene_list, f)
    logger.info(f"  scene_detect [{opt.reference}]: {len(scene_list)} scenes")
    return scene_list


def run_pipeline(opt):
    logger.info(f"  run_pipeline [{opt.reference}]: START")
    for dir_path in [opt.work_dir, opt.crop_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
        full_path = os.path.join(dir_path, opt.reference)
        if os.path.exists(full_path):
            rmtree(full_path)
        os.makedirs(full_path)
    ref_avi     = os.path.join(opt.avi_dir, opt.reference)
    frames_dir  = os.path.join(opt.frames_dir, opt.reference)

    # Step 1: convert source video → .avi (must finish before steps below)
    subprocess.call(
        f"ffmpeg -y -i {opt.videofile} -qscale:v 2 -async 1 -r 25 {ref_avi}/video.avi",
        shell=True, stderr=subprocess.DEVNULL)

    # Step 2: extract frames + audio in parallel (both read from video.avi independently)
    def _extract_frames():
        subprocess.call(
            f"ffmpeg -y -i {ref_avi}/video.avi -qscale:v 2 -f image2 {frames_dir}/%06d.jpg",
            shell=True, stderr=subprocess.DEVNULL)

    def _extract_audio():
        subprocess.call(
            f"ffmpeg -y -i {ref_avi}/video.avi -ac 1 -vn -acodec pcm_s16le -ar 16000 {ref_avi}/audio.wav",
            shell=True, stderr=subprocess.DEVNULL)

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_frames = executor.submit(_extract_frames)
        f_audio  = executor.submit(_extract_audio)
        f_frames.result()
        f_audio.result()

    # Run face detection (GPU) and scene detection (CPU) in parallel — independent inputs
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_faces = executor.submit(inference_video, opt)
        f_scene = executor.submit(scene_detect, opt)
        faces = f_faces.result()
        scene = f_scene.result()

    alltracks = []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= opt.min_track:
            alltracks.extend(track_shot(opt, faces[shot[0].frame_num:shot[1].frame_num]))

    # Compute flist once — all tracks in this scene share the same frames dir
    flist = sorted(glob.glob(os.path.join(opt.frames_dir, opt.reference, "*.jpg")))

    # Step 3: crop all face tracks in parallel (CPU/IO-bound, no GPU)
    vidtracks = [None] * len(alltracks)
    crop_workers = min(len(alltracks), 4) if alltracks else 1
    with ThreadPoolExecutor(max_workers=crop_workers) as executor:
        futures = {
            executor.submit(
                crop_video, opt, track,
                os.path.join(opt.crop_dir, opt.reference, "%05d" % ii),
                flist
            ): ii
            for ii, track in enumerate(alltracks)
        }
        for future in as_completed(futures):
            vidtracks[futures[future]] = future.result()

    with open(os.path.join(opt.work_dir, opt.reference, "tracks.pckl"), "wb") as f:
        pickle.dump(vidtracks, f)
    rmtree(os.path.join(opt.tmp_dir, opt.reference))
    logger.info(f"  run_pipeline [{opt.reference}]: {len(vidtracks)} face tracks. END")


def run_syncnet(opt):
    logger.info(f"  run_syncnet [{opt.reference}]: START device={opt.device}")
    s     = _load_syncnet(opt.initial_model, opt.device)
    flist = sorted(glob.glob(os.path.join(opt.crop_dir, opt.reference, "0*.avi")))
    logger.info(f"  run_syncnet [{opt.reference}]: {len(flist)} crops to evaluate")
    if not flist:
        logger.info(f"  run_syncnet [{opt.reference}]: no crops, skip")
        return

    # Phase 1 — CPU: read video/audio + MFCC in parallel (no GPU needed)
    cpu_workers = max(4, len(flist))
    prepared = [None] * len(flist)
    with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
        futures = {executor.submit(s.prepare_data, fname): i
                   for i, fname in enumerate(flist)}
        for future in as_completed(futures):
            prepared[futures[future]] = future.result()

    # Phase 2 — GPU: inference sequentially on pre-computed CPU tensors
    results = []
    for idx, (fname, data) in enumerate(zip(flist, prepared)):
        if data is None:
            continue
        offset, conf, dist = s.evaluate_tensors(opt, *data)
        dist_str = " ".join(map(str, dist.flatten())) if hasattr(dist, "flatten") else str(dist)
        results.append({"index": idx, "file": os.path.basename(fname),
                         "offset": offset, "conf": conf, "dist": dist_str})
    out_path = os.path.join(opt.work_dir, opt.reference, "activesd.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# index\tfile\toffset\tconfidence\tdist\n")
        for r in results:
            f.write(f'{r["index"]}\t{r["file"]}\t{r["offset"]}\t{r["conf"]}\t{r["dist"]}\n')
    logger.info(f"  run_syncnet [{opt.reference}]: END")


def filter_videos(opt):
    logger.info(f"  filter_videos [{opt.reference}]: START")
    result_file = os.path.join(opt.work_dir, opt.reference, "activesd.txt")
    if not os.path.exists(result_file):
        logger.warning(f"  filter_videos [{opt.reference}]: no activesd.txt found")
        return
    results = []
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            try:
                results.append({"index": int(parts[0]), "filename": parts[1],
                                  "offset": int(parts[2]), "confidence": float(parts[3])})
            except:
                pass
    good = [r for r in results if r["confidence"] >= opt.min_confidence
            and opt.offset_min <= r["offset"] <= opt.offset_max]
    bad  = [r for r in results if r not in good]
    logger.info(f"  filter_videos [{opt.reference}]: good={len(good)}, bad={len(bad)}")
    crop_dir = os.path.join(opt.data_dir, "pycrop")
    for label, lst in [("save_good", good), ("save_bad", bad)]:
        dst_dir = os.path.join(getattr(opt, label), opt.reference)
        os.makedirs(dst_dir, exist_ok=True)
        for r in lst:
            src = os.path.join(crop_dir, opt.reference, r["filename"])
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst_dir, r["filename"]))
    logger.info(f"  filter_videos [{opt.reference}]: END")


# ==================== ASR via API ====================
def get_audio_duration(audio_path: Path) -> float:
    import wave
    try:
        with wave.open(str(audio_path), "r") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except:
        return 0.0


def call_asr_api(audio_path: Path) -> str:
    """Gọi ASR API, trả về text"""
    with open(audio_path, "rb") as audio_file:
        response = requests.post(
            CONFIG["asr_api_url"],
            params={
                "language":        CONFIG["asr_language"],
                "use_punctuation": str(CONFIG["asr_use_punctuation"]).lower(),
                "full_text_mode":  str(CONFIG["asr_full_text_mode"]).lower(),
            },
            files={"file": (audio_path.name, audio_file, "audio/wav")},
            timeout=CONFIG["asr_timeout"],
        )
    response.raise_for_status()
    return response.json().get("text", "").strip()


def save_chunk(src_mp4: Path, src_wav: Path, start: float, end: float,
               video_out: Path, audio_out: Path, text_out: Path) -> bool:
    """Cắt đoạn video/audio theo [start, end] và gọi API lấy transcript"""
    dur = end - start
    # Video mp4 có cả video lẫn audio (dùng src_wav làm nguồn audio)
    r1 = subprocess.run(
        ["ffmpeg", "-y",
         "-ss", f"{start:.3f}", "-i", str(src_mp4),
         "-ss", f"{start:.3f}", "-i", str(src_wav),
         "-t", f"{dur:.3f}",
         "-c:v", "libx264", "-preset", "fast",
         "-c:a", "aac", "-ar", "44100", "-ac", "1",
         "-map", "0:v:0", "-map", "1:a:0",
         str(video_out)],
        capture_output=True)
    # WAV riêng để gửi API
    r2 = subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src_wav),
         "-t", f"{dur:.3f}", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_out)],
        capture_output=True)
    if r1.returncode != 0 or r2.returncode != 0:
        return False
    try:
        text = call_asr_api(audio_out)
        text_out.write_text(text, encoding="utf-8")
        return True
    except Exception as e:
        logger.error(f"    ASR API error: {e}")
        return False


def run_asr(opt, session_dir: Path):
    logger.info(f"  run_asr [{opt.reference}]: START (API mode)")
    input_path = Path(opt.save_good) / opt.reference
    tets_dir   = session_dir / "tetss" / opt.reference
    tmp_dir    = tets_dir / "_tmp"
    video_dir  = tets_dir / "videos"
    audio_dir  = tets_dir / "audios"
    trans_dir  = tets_dir / "transcripts"
    for d in [video_dir, audio_dir, trans_dir, tmp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    avi_files = sorted(input_path.glob("*.avi"))
    if not avi_files:
        logger.warning(f"  run_asr [{opt.reference}]: no AVI files found")
        return

    logger.info(f"  run_asr [{opt.reference}]: {len(avi_files)} clips to process in parallel")
    max_dur = CONFIG["max_audio_duration"]

    def _process_one_avi(avi_file: Path) -> list:
        """Process single AVI; returns list of (tmp_vid, tmp_aud, tmp_txt) Paths."""
        uid     = f"{avi_file.stem}_{threading.get_ident()}"
        tmp_mp4 = tmp_dir / f"{uid}.mp4"
        tmp_wav = tmp_dir / f"{uid}.wav"

        subprocess.run(["ffmpeg", "-y", "-i", str(avi_file), "-c:v", "libx264",
                        "-c:a", "aac", "-preset", "fast", str(tmp_mp4)], capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", str(avi_file), "-vn", "-acodec", "pcm_s16le",
                        "-ar", "16000", "-ac", "1", str(tmp_wav)], capture_output=True)

        wav_size = os.path.getsize(str(tmp_wav)) if tmp_wav.exists() else 0
        if wav_size < 2000:
            logger.warning(f"  run_asr: {avi_file.name} wav too small, skipping")
            tmp_mp4.unlink(missing_ok=True); tmp_wav.unlink(missing_ok=True)
            return []

        audio_duration = get_audio_duration(tmp_wav)
        chunks_out = []   # list of (tmp_vid_path, tmp_aud_path, tmp_txt_path)

        chunk_starts = []
        s = 0.0
        while s < audio_duration:
            chunk_starts.append(s)
            if audio_duration <= max_dur:
                break
            s += max_dur

        for ci, chunk_start in enumerate(chunk_starts):
            chunk_end = min(chunk_start + max_dur, audio_duration)
            if chunk_end - chunk_start < 1.0:
                break
            t_vid = tmp_dir / f"{uid}_c{ci}_vid.mp4"
            t_aud = tmp_dir / f"{uid}_c{ci}_aud.wav"
            t_txt = tmp_dir / f"{uid}_c{ci}.txt"
            if save_chunk(tmp_mp4, tmp_wav, chunk_start, chunk_end, t_vid, t_aud, t_txt):
                chunks_out.append((t_vid, t_aud, t_txt))

        tmp_mp4.unlink(missing_ok=True)
        tmp_wav.unlink(missing_ok=True)
        return chunks_out

    # Run all AVI files in parallel (CPU + IO + network, no GPU)
    asr_workers = max(4, CONFIG["max_workers"] * 2)
    all_chunks: list = []
    with ThreadPoolExecutor(max_workers=asr_workers) as executor:
        futures = {executor.submit(_process_one_avi, f): f for f in avi_files}
        for future in as_completed(futures):
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"  run_asr: {futures[future].name} failed: {e}")

    # Rename temp files to sequential IDs
    for idx, (t_vid, t_aud, t_txt) in enumerate(all_chunks, 1):
        file_id = f"{idx:06d}"
        t_vid.rename(video_dir / f"{file_id}.mp4")
        t_aud.rename(audio_dir / f"{file_id}.wav")
        t_txt.rename(trans_dir / f"{file_id}.txt")

    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    logger.info(f"  run_asr [{opt.reference}]: END. Total clips = {len(all_chunks)}")


# ==================== CLEANUP ====================
def cleanup_temp_directories(opt):
    logger.info(f"  cleanup [{opt.reference}]: removing temp dirs...")
    temp_dirs = [
        os.path.join(opt.data_dir, d, opt.reference)
        for d in ["pyavi", "pytmp", "pywork", "pycrop", "pyframes", "pycrop_headpose_rejected"]
    ] + [os.path.join(opt.save_good, opt.reference),
         os.path.join(opt.save_bad,  opt.reference)]
    for d in temp_dirs:
        if os.path.exists(d):
            try:
                rmtree(d)
            except Exception as e:
                logger.warning(f"  cleanup [{opt.reference}]: cannot remove {d} => {e}")


# ==================== MERGE ====================
def merge_and_reorganize(session_dir: Path, final_output_dir: Path):
    logger.info(f"merge_and_reorganize: session={session_dir}, output={final_output_dir}")
    tetss_path = session_dir / "tetss"
    if not tetss_path.exists():
        logger.warning(f"merge_and_reorganize: tetss not found at {tetss_path}")
        return
    videos_dir = final_output_dir / "videos"
    audios_dir = final_output_dir / "audios"
    trans_dir  = final_output_dir / "transcripts"
    for d in [videos_dir, audios_dir, trans_dir]:
        d.mkdir(parents=True, exist_ok=True)
    scene_folders = sorted([d for d in tetss_path.iterdir() if d.is_dir()])
    logger.info(f"merge_and_reorganize: {len(scene_folders)} scene folders")
    counter = 1
    for scene_folder in scene_folders:
        video_src = scene_folder / "videos"
        if not video_src.exists():
            continue
        for video_file in sorted(video_src.glob("*.mp4")):
            new_id = f"{counter:06d}"
            shutil.copy2(video_file, videos_dir / f"{new_id}.mp4")
            audio_src = scene_folder / "audios" / video_file.with_suffix(".wav").name
            if audio_src.exists():
                shutil.copy2(audio_src, audios_dir / f"{new_id}.wav")
            trans_src = scene_folder / "transcripts" / video_file.with_suffix(".txt").name
            if trans_src.exists():
                shutil.copy2(trans_src, trans_dir / f"{new_id}.txt")
            counter += 1
    logger.info(f"merge_and_reorganize: total {counter - 1} clips merged")
    try:
        rmtree(tetss_path)
    except Exception as e:
        logger.warning(f"merge_and_reorganize: cannot clean tetss => {e}")


# ==================== SCENE SPLIT ====================
def run_scenedetect(video_path: str, scene_output: str) -> list:
    os.makedirs(scene_output, exist_ok=True)

    # Step 1: Detect scene boundaries via Python API (downscale + frame_skip)
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)  # no frame_skip: accuracy matters here
    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]
    video_manager.release()
    logger.info(f"run_scenedetect: detected {len(scene_list)} scenes")

    # Step 2: Split in parallel with ffmpeg stream-copy (no re-encode → ~50x faster)
    stem = Path(video_path).stem
    ext  = Path(video_path).suffix

    def _split_one(idx, start_tc, end_tc):
        start_sec = start_tc.get_seconds()
        duration  = end_tc.get_seconds() - start_sec
        out_path  = os.path.join(scene_output, f"{stem}-Scene-{idx+1:03d}{ext}")
        subprocess.run(
            ["ffmpeg", "-y",
             "-ss", f"{start_sec:.3f}", "-i", video_path,
             "-t",  f"{duration:.3f}",
             "-c:v", "libx264", "-preset", "ultrafast",
             "-c:a", "copy", out_path],
            capture_output=True)
        return out_path if os.path.exists(out_path) else None

    split_workers = min(len(scene_list), os.cpu_count() or 8)
    scenes = [None] * len(scene_list)
    with ThreadPoolExecutor(max_workers=split_workers) as executor:
        futures = {executor.submit(_split_one, i, s[0], s[1]): i
                   for i, s in enumerate(scene_list)}
        for future in as_completed(futures):
            scenes[futures[future]] = future.result()

    scenes = sorted(s for s in scenes if s)
    logger.info(f"run_scenedetect: {len(scenes)} scenes produced")
    return scenes


# ==================== PROCESS ONE SCENE ====================
def process_single_scene(video_path: str, session_dir: Path, thread_id: int) -> dict:
    video_name = Path(video_path).stem
    # Acquire a GPU device for this scene's lifetime
    device = _gpu_pool.get()
    logger.info(f"[Thread-{thread_id}] START: {video_name} on {device}")
    pid = os.getpid()

    # Workspace riêng theo video_name + pid để không đụng nhau
    workspace  = session_dir / "processing" / f"{video_name}_{pid}_{thread_id}"
    workspace.mkdir(parents=True, exist_ok=True)

    opt = DictToObject({
        **CONFIG,
        "videofile":   video_path,
        "reference":   video_name,
        "device":      device,
        "data_dir":    str(workspace),
        "save_good":   str(workspace / "save_good"),
        "save_bad":    str(workspace / "save_bad"),
        "avi_dir":     str(workspace / "pyavi"),
        "tmp_dir":     str(workspace / "pytmp"),
        "work_dir":    str(workspace / "pywork"),
        "crop_dir":    str(workspace / "pycrop"),
        "frames_dir":  str(workspace / "pyframes"),
    })

    try:
        t0 = time.time()
        logger.info(f"[Thread-{thread_id}] {video_name}: STEP A - run_pipeline")
        run_pipeline(opt)

        logger.info(f"[Thread-{thread_id}] {video_name}: STEP B - filter_by_head_pose")
        hp_stats = filter_by_head_pose(opt)

        logger.info(f"[Thread-{thread_id}] {video_name}: STEP C - run_syncnet")
        run_syncnet(opt)

        logger.info(f"[Thread-{thread_id}] {video_name}: STEP D - filter_videos")
        filter_videos(opt)

        logger.info(f"[Thread-{thread_id}] {video_name}: STEP E - run_asr")
        run_asr(opt, session_dir)

        logger.info(f"[Thread-{thread_id}] {video_name}: STEP F - cleanup")
        cleanup_temp_directories(opt)
        if workspace.exists():
            rmtree(workspace)

        elapsed = time.time() - t0
        logger.info(f"[Thread-{thread_id}] DONE: {video_name} in {elapsed:.1f}s | head_pose={hp_stats}")
        return {"video": video_name, "status": "success", "time": elapsed, "head_pose": hp_stats}

    except Exception as e:
        import traceback
        logger.error(f"[Thread-{thread_id}] FAILED: {video_name}\n{traceback.format_exc()}")
        return {"video": video_name, "status": "failed", "error": str(e)}

    finally:
        # Always return the GPU to the pool, even on error
        _gpu_pool.put(device)


# ==================== PROCESS ONE VIDEO ====================
def process_one_video(video_path: Path, output_base: Path) -> str:
    pid        = os.getpid()
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(CONFIG["workspace_dir"]) / f"session_{video_path.stem}_{timestamp}_{pid}"
    session_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[PID {pid}] === Processing: {video_path.name} ===")
    logger.info(f"[PID {pid}] session_dir: {session_dir}")

    # Copy video vào session workspace
    workspace_video = session_dir / video_path.name
    shutil.copy2(video_path, workspace_video)

    result_folder = get_next_result_folder(output_base)
    shutil.copy2(workspace_video, result_folder / f"original_{video_path.name}")
    logger.info(f"[PID {pid}] result_folder: {result_folder}")

    # STEP 1: Scene split
    logger.info(f"[PID {pid}] STEP 1/4: Scene detection...")
    scene_output = str(session_dir / "scenes")
    scene_videos = run_scenedetect(str(workspace_video), scene_output)
    if not scene_videos:
        raise RuntimeError("No scenes detected from video")
    logger.info(f"[PID {pid}] STEP 1 OK: {len(scene_videos)} scenes")

    # STEP 2: Process scenes in parallel
    max_workers = CONFIG["max_workers"]
    logger.info(f"[PID {pid}] STEP 2/4: Processing {len(scene_videos)} scenes (max_workers={max_workers})...")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_scene, v, session_dir, i + 1): v
            for i, v in enumerate(scene_videos)
        }
        for future in as_completed(futures):
            try:
                r = future.result()
                results.append(r)
                logger.info(f"[PID {pid}] Scene done: {r['video']} status={r['status']}")
            except Exception as exc:
                logger.error(f"[PID {pid}] Scene exception: {exc}")
                results.append({"video": futures[future], "status": "exception", "error": str(exc)})

    successful = sum(1 for r in results if r["status"] == "success")
    logger.info(f"[PID {pid}] STEP 2 OK: {successful}/{len(scene_videos)} scenes successful")

    # STEP 3: Merge
    if os.path.exists(scene_output):
        rmtree(scene_output)
    logger.info(f"[PID {pid}] STEP 3/4: Merging results into {result_folder}...")
    merge_and_reorganize(session_dir, result_folder)

    # STEP 4: Cleanup session
    try:
        rmtree(session_dir)
        logger.info(f"[PID {pid}] session_dir cleaned up")
    except Exception as e:
        logger.warning(f"[PID {pid}] Cannot clean session_dir: {e}")

    total_clips = len(list((result_folder / "videos").glob("*.mp4"))) if (result_folder / "videos").exists() else 0
    logger.info(f"[PID {pid}] STEP 4 OK: {total_clips} clips in {result_folder}")
    return str(result_folder)


# ==================== MAIN ====================
def main():
    input_dir  = Path(CONFIG["input_dir"])
    output_dir = Path(CONFIG["output_dir"])
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(CONFIG["workspace_dir"]).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"[PID {os.getpid()}] AVSR CLI Worker started")
    logger.info(f"  Input  : {input_dir.absolute()}")
    logger.info(f"  Output : {output_dir.absolute()}")
    logger.info(f"  Log    : {CONFIG['log_file']}")
    logger.info("=" * 60)

    claimed = 0
    while True:
        video_path = claim_next_video(input_dir)
        if video_path is None:
            if claimed == 0:
                logger.info(f"[PID {os.getpid()}] Không có video mới. Thả video vào: {input_dir.absolute()}")
            else:
                logger.info(f"[PID {os.getpid()}] Hoàn tất {claimed} video.")
            break

        claimed += 1
        logger.info(f"[PID {os.getpid()}] Claimed: {video_path.name}")
        try:
            output_folder = process_one_video(video_path, output_dir)
            mark_done(video_path.name, output_folder)
            logger.info(f"[PID {os.getpid()}] ✅ DONE: {video_path.name} → {output_folder}")
        except Exception as e:
            import traceback
            logger.error(f"[PID {os.getpid()}] ❌ ERROR: {video_path.name}\n{traceback.format_exc()}")
            mark_error(video_path.name, str(e))

    logger.info(f"[PID {os.getpid()}] Worker exiting.")


if __name__ == "__main__":
    main()
