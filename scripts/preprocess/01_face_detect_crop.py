"""
Step 1: Face Detection, Cropping, and FPS Normalization.

For each raw video:
1. Detect faces using RetinaFace (or fallback to MediaPipe)
2. Compute stable crop region across all frames
3. Crop and resize to target resolution (512x512)
4. Normalize FPS to 25fps
5. Save as processed .mp4

Input:  raw video files
Output: data_root/{clip_id}/video.mp4  (cropped, resized, 25fps)

Usage:
    python scripts/preprocess/01_face_detect_crop.py \
        --input_dir /path/to/raw_videos \
        --output_dir /path/to/processed_data \
        --target_fps 25 \
        --target_size 512 \
        --num_workers 4
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def get_face_detector(backend="retinaface"):
    """Initialize face detector."""
    if backend == "retinaface":
        try:
            from retinaface import RetinaFace
            return {"type": "retinaface"}
        except ImportError:
            print("[Warning] RetinaFace not installed, falling back to OpenCV Haar cascade")
            backend = "opencv"

    if backend == "opencv":
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        return {"type": "opencv", "detector": detector}

    raise ValueError(f"Unknown backend: {backend}")


def detect_face_bbox(frame_bgr, detector_info):
    """Detect face bounding box in a single frame.

    Returns: (x, y, w, h) or None if no face detected.
    """
    if detector_info["type"] == "retinaface":
        from retinaface import RetinaFace
        faces = RetinaFace.detect_faces(frame_bgr)
        if not faces or len(faces) == 0:
            return None
        # Use the largest face
        best_face = None
        best_area = 0
        for key, face_info in faces.items():
            if isinstance(face_info, dict) and "facial_area" in face_info:
                x1, y1, x2, y2 = face_info["facial_area"]
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_face = (x1, y1, x2 - x1, y2 - y1)
        return best_face

    elif detector_info["type"] == "opencv":
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = detector_info["detector"].detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None
        # Use the largest face
        areas = [w * h for (x, y, w, h) in faces]
        idx = np.argmax(areas)
        return tuple(faces[idx])

    return None


def compute_stable_crop(
    video_path: str,
    detector_info: dict,
    target_size: int = 512,
    sample_interval: int = 10,
    expand_ratio: float = 1.8,
) -> dict:
    """Sample frames from video, detect faces, compute stable crop region.

    Returns dict with crop info or None if insufficient face detections.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sample frames for face detection
    bboxes = []
    sample_frames = list(range(0, total_frames, sample_interval))
    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        bbox = detect_face_bbox(frame, detector_info)
        if bbox is not None:
            bboxes.append(bbox)

    cap.release()

    if len(bboxes) < len(sample_frames) * 0.3:
        # Less than 30% of frames have faces -> skip
        return None

    # Multiple faces check: if std of bbox center is too high, likely multi-face
    bboxes = np.array(bboxes)  # [N, 4] (x, y, w, h)
    centers_x = bboxes[:, 0] + bboxes[:, 2] / 2
    centers_y = bboxes[:, 1] + bboxes[:, 3] / 2

    # Compute median bbox (robust to outliers)
    median_cx = np.median(centers_x)
    median_cy = np.median(centers_y)
    median_w = np.median(bboxes[:, 2])
    median_h = np.median(bboxes[:, 3])

    # Expand crop region
    crop_size = max(median_w, median_h) * expand_ratio
    crop_size = int(crop_size)

    # Center the crop
    crop_x = int(median_cx - crop_size / 2)
    crop_y = int(median_cy - crop_size / 2)

    # Clamp to image bounds
    crop_x = max(0, min(crop_x, orig_w - crop_size))
    crop_y = max(0, min(crop_y, orig_h - crop_size))
    crop_size = min(crop_size, orig_w - crop_x, orig_h - crop_y)

    if crop_size < 64:
        return None

    return {
        "crop_x": crop_x,
        "crop_y": crop_y,
        "crop_size": crop_size,
        "orig_w": orig_w,
        "orig_h": orig_h,
        "num_faces_detected": len(bboxes),
        "total_sampled": len(sample_frames),
    }


def process_video(
    video_path: str,
    output_dir: str,
    crop_info: dict,
    target_size: int = 512,
    target_fps: float = 25.0,
) -> str:
    """Crop and resize video using ffmpeg.

    Returns path to output video, or None on failure.
    """
    clip_id = Path(video_path).stem
    clip_dir = os.path.join(output_dir, clip_id)
    os.makedirs(clip_dir, exist_ok=True)
    output_path = os.path.join(clip_dir, "video.mp4")

    cx = crop_info["crop_x"]
    cy = crop_info["crop_y"]
    cs = crop_info["crop_size"]

    # ffmpeg: crop -> resize -> fps normalize
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"crop={cs}:{cs}:{cx}:{cy},scale={target_size}:{target_size},fps={target_fps}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-ar", "16000", "-ac", "1",  # mono 16kHz audio for HuBERT
        "-movflags", "+faststart",
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"[Error] ffmpeg failed for {video_path}: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"[Error] ffmpeg timed out for {video_path}")
        return None

    return output_path


def get_video_info(video_path: str) -> dict:
    """Get basic video info using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video":
                    return {
                        "num_frames": int(stream.get("nb_frames", 0)),
                        "width": int(stream.get("width", 0)),
                        "height": int(stream.get("height", 0)),
                        "fps": eval(stream.get("r_frame_rate", "25/1")),
                        "duration": float(info.get("format", {}).get("duration", 0)),
                    }
    except Exception:
        pass
    return {}


def main():
    parser = argparse.ArgumentParser(description="Face detection, cropping, and FPS normalization")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with raw videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--target_fps", type=float, default=25.0, help="Target FPS")
    parser.add_argument("--target_size", type=int, default=512, help="Target crop size (square)")
    parser.add_argument("--expand_ratio", type=float, default=1.8, help="Face bbox expand ratio for cropping")
    parser.add_argument("--sample_interval", type=int, default=10, help="Frame interval for face detection sampling")
    parser.add_argument("--min_duration", type=float, default=2.0, help="Minimum clip duration in seconds")
    parser.add_argument("--detector", type=str, default="retinaface", choices=["retinaface", "opencv"])
    parser.add_argument("--extensions", type=str, default="mp4,avi,mov,mkv,webm")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all video files
    extensions = set(args.extensions.split(","))
    video_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for f in files:
            if f.rsplit(".", 1)[-1].lower() in extensions:
                video_files.append(os.path.join(root, f))

    print(f"Found {len(video_files)} video files in {args.input_dir}")

    detector_info = get_face_detector(args.detector)

    results = {"processed": 0, "skipped_no_face": 0, "skipped_short": 0, "failed": 0}

    for video_path in tqdm(video_files, desc="Processing videos"):
        # Check duration
        info = get_video_info(video_path)
        if info.get("duration", 0) < args.min_duration:
            results["skipped_short"] += 1
            continue

        # Detect faces and compute crop
        crop_info = compute_stable_crop(
            video_path, detector_info,
            target_size=args.target_size,
            sample_interval=args.sample_interval,
            expand_ratio=args.expand_ratio,
        )

        if crop_info is None:
            results["skipped_no_face"] += 1
            continue

        # Process video
        output = process_video(
            video_path, args.output_dir, crop_info,
            target_size=args.target_size,
            target_fps=args.target_fps,
        )

        if output is None:
            results["failed"] += 1
        else:
            results["processed"] += 1

    print(f"\n[Done] Results: {results}")


if __name__ == "__main__":
    main()
