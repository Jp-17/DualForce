"""
Step 7: Build metadata.json for DualForce training.

Scans processed data directory and builds the metadata file expected by
DualForceDataset. Verifies that all required features exist and records
per-clip statistics.

Input:  data_root/{clip_id}/ directories with extracted features
Output: data_root/metadata.json

Usage:
    python scripts/preprocess/07_build_metadata.py \
        --data_dir /path/to/processed_data \
        --output_file metadata.json \
        --require_all_features
"""

import argparse
import json
import os

import cv2
import torch
from safetensors.torch import load_file
from tqdm import tqdm


REQUIRED_FILES = [
    "video_latents.safetensors",
    "struct_latents.safetensors",
    "audio_features.safetensors",
    "first_frame.safetensors",
]

OPTIONAL_FILES = [
    "flame_params.safetensors",
]


def get_video_num_frames(video_path: str) -> int:
    """Get frame count from video file."""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def verify_clip(clip_dir: str, clip_id: str, require_all: bool = False) -> dict:
    """Verify a clip directory has all required features.

    Returns metadata dict or None if clip is invalid.
    """
    # Check required files
    for fname in REQUIRED_FILES:
        fpath = os.path.join(clip_dir, fname)
        if not os.path.exists(fpath):
            return None

    # Probe feature shapes
    info = {"clip_id": clip_id}

    try:
        # Video latents
        vl = load_file(os.path.join(clip_dir, "video_latents.safetensors"))
        vl_tensor = vl["video_latents"]
        info["video_latent_shape"] = list(vl_tensor.shape)
        # T' in latent space; raw frames = T' * vae_temporal_stride
        latent_T = vl_tensor.shape[1]
        info["latent_frames"] = latent_T

        # Struct latents
        sl = load_file(os.path.join(clip_dir, "struct_latents.safetensors"))
        sl_tensor = sl["struct_latents"]
        info["struct_latent_shape"] = list(sl_tensor.shape)
        info["struct_frames"] = sl_tensor.shape[1]

        # Audio features
        af = load_file(os.path.join(clip_dir, "audio_features.safetensors"))
        af_tensor = af["audio_features"]
        info["audio_feature_shape"] = list(af_tensor.shape)
        info["audio_frames"] = af_tensor.shape[0]

        # First frame
        ff = load_file(os.path.join(clip_dir, "first_frame.safetensors"))
        ff_tensor = ff["first_frame"]
        info["first_frame_shape"] = list(ff_tensor.shape)

    except Exception as e:
        print(f"[Error] Failed to load features for {clip_id}: {e}")
        return None

    # Optional: FLAME params
    flame_path = os.path.join(clip_dir, "flame_params.safetensors")
    info["has_flame"] = os.path.exists(flame_path)
    if info["has_flame"]:
        try:
            fp = load_file(flame_path)
            fp_tensor = fp["flame_params"]
            info["flame_shape"] = list(fp_tensor.shape)
        except Exception:
            info["has_flame"] = False

    if require_all and not info["has_flame"]:
        return None

    # Get raw frame count from video if available
    video_path = os.path.join(clip_dir, "video.mp4")
    if os.path.exists(video_path):
        info["num_frames"] = get_video_num_frames(video_path)
    else:
        # Estimate from latent frames (T' * 4)
        info["num_frames"] = latent_T * 4

    # Caption: check if caption.txt exists in clip dir
    caption_path = os.path.join(clip_dir, "caption.txt")
    if os.path.exists(caption_path):
        with open(caption_path, "r") as f:
            info["caption"] = f.read().strip()
    else:
        info["caption"] = "A person speaking."

    return info


def main():
    parser = argparse.ArgumentParser(description="Build metadata.json for DualForce training")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="metadata.json")
    parser.add_argument("--require_all_features", action="store_true",
                        help="Require all features including FLAME params")
    parser.add_argument("--min_latent_frames", type=int, default=4,
                        help="Minimum video latent frames (T') to include clip")
    args = parser.parse_args()

    # Find clip directories
    clip_dirs = sorted([
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ])

    print(f"Scanning {len(clip_dirs)} directories...")

    metadata = []
    stats = {"total": len(clip_dirs), "valid": 0, "invalid": 0, "too_short": 0}

    for clip_id in tqdm(clip_dirs, desc="Building metadata"):
        clip_dir = os.path.join(args.data_dir, clip_id)

        info = verify_clip(clip_dir, clip_id, require_all=args.require_all_features)

        if info is None:
            stats["invalid"] += 1
            continue

        if info.get("latent_frames", 0) < args.min_latent_frames:
            stats["too_short"] += 1
            continue

        metadata.append(info)
        stats["valid"] += 1

    # Save metadata
    output_path = os.path.join(args.data_dir, args.output_file)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print(f"\n[Done] Metadata built: {stats}")
    print(f"Total valid clips: {len(metadata)}")
    if metadata:
        total_latent_frames = sum(m.get("latent_frames", 0) for m in metadata)
        total_raw_frames = sum(m.get("num_frames", 0) for m in metadata)
        flame_count = sum(1 for m in metadata if m.get("has_flame", False))
        print(f"Total latent frames: {total_latent_frames}")
        print(f"Total raw frames: {total_raw_frames} (~{total_raw_frames / 25 / 3600:.1f}h at 25fps)")
        print(f"Clips with FLAME params: {flame_count}/{len(metadata)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
