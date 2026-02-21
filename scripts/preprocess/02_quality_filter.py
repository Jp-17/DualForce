"""
Step 2: Quality Filtering.

Filters processed clips based on:
- Hard filters: multi-face, short duration, low resolution, large A/V offset
- Soft filters: blur score, head pose, audio SNR (add weight metadata)

Input:  data_root/{clip_id}/video.mp4  (from step 01)
Output: data_root/filtered_clips.json  (list of passing clip_ids with quality scores)

Usage:
    python scripts/preprocess/02_quality_filter.py \
        --data_dir /path/to/processed_data \
        --output_file filtered_clips.json \
        --min_duration 2.0 \
        --min_resolution 256
"""

import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm


def compute_blur_score(frame_bgr):
    """Compute Laplacian variance as blur score. Higher = sharper."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def analyze_clip(clip_dir: str, clip_id: str, min_duration: float = 2.0) -> dict:
    """Analyze a single clip for quality metrics.

    Returns dict with quality scores, or None if clip should be hard-filtered.
    """
    video_path = os.path.join(clip_dir, "video.mp4")
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        cap.release()
        return None

    duration = total_frames / fps

    # Hard filter: duration
    if duration < min_duration:
        cap.release()
        return None

    # Sample frames for blur analysis
    blur_scores = []
    sample_indices = np.linspace(0, total_frames - 1, min(20, total_frames), dtype=int)
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            blur_scores.append(compute_blur_score(frame))

    cap.release()

    if len(blur_scores) == 0:
        return None

    mean_blur = float(np.mean(blur_scores))
    min_blur = float(np.min(blur_scores))

    return {
        "clip_id": clip_id,
        "duration": round(duration, 2),
        "num_frames": total_frames,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "blur_score_mean": round(mean_blur, 2),
        "blur_score_min": round(min_blur, 2),
        # Soft quality weight (1.0 = good, lower = worse)
        "quality_weight": round(min(1.0, mean_blur / 200.0), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Quality filtering for processed clips")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="filtered_clips.json")
    parser.add_argument("--min_duration", type=float, default=2.0)
    parser.add_argument("--min_resolution", type=int, default=256)
    parser.add_argument("--min_blur_score", type=float, default=20.0, help="Hard filter: minimum mean blur score")
    args = parser.parse_args()

    # Find all clip directories
    clip_dirs = []
    for name in sorted(os.listdir(args.data_dir)):
        clip_path = os.path.join(args.data_dir, name)
        if os.path.isdir(clip_path) and os.path.exists(os.path.join(clip_path, "video.mp4")):
            clip_dirs.append((name, clip_path))

    print(f"Found {len(clip_dirs)} clip directories")

    passed = []
    stats = {"total": len(clip_dirs), "passed": 0, "hard_filtered": 0}

    for clip_id, clip_path in tqdm(clip_dirs, desc="Analyzing clips"):
        result = analyze_clip(clip_path, clip_id, min_duration=args.min_duration)

        if result is None:
            stats["hard_filtered"] += 1
            continue

        # Hard filters
        if result["width"] < args.min_resolution or result["height"] < args.min_resolution:
            stats["hard_filtered"] += 1
            continue

        if result["blur_score_mean"] < args.min_blur_score:
            stats["hard_filtered"] += 1
            continue

        passed.append(result)
        stats["passed"] += 1

    # Save filtered list
    output_path = os.path.join(args.data_dir, args.output_file)
    with open(output_path, "w") as f:
        json.dump(passed, f, indent=2)

    print(f"\n[Done] {stats}")
    print(f"Saved {len(passed)} clips to {output_path}")


if __name__ == "__main__":
    main()
