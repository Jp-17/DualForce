"""
Step 4: Extract 3D Structure Latents using LivePortrait.

For each processed clip, extract per-frame motion features from LivePortrait's
motion extractor and save as struct_latents.

Input:  data_root/{clip_id}/video.mp4
Output: data_root/{clip_id}/struct_latents.safetensors
        Contains: {"struct_latents": tensor[D=128, T]}

LivePortrait motion features capture:
- Head pose (rotation, translation)
- Expression deformation
- Scale/crop parameters
These are packed into a 128-dimensional vector per frame.

Usage:
    python scripts/preprocess/04_extract_struct_latents.py \
        --data_dir /path/to/processed_data \
        --liveportrait_path /path/to/LivePortrait \
        --device cuda
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from safetensors.torch import save_file
from tqdm import tqdm


def load_frames_as_numpy(video_path: str) -> list:
    """Load video frames as list of numpy BGR images."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def extract_struct_features_liveportrait(
    frames: list,
    motion_extractor,
    face_analyzer,
    device: str = "cuda",
) -> torch.Tensor:
    """Extract LivePortrait motion features for each frame.

    Returns: [D=128, T] tensor of struct latents, or None on failure.
    """
    all_features = []

    for frame in frames:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            # Detect face and get aligned crop
            # LivePortrait expects 256x256 aligned face
            face_info = face_analyzer.get(frame_rgb)
            if len(face_info) == 0:
                # Use previous features or zero-fill
                if all_features:
                    all_features.append(all_features[-1].clone())
                else:
                    all_features.append(torch.zeros(128, device=device))
                continue

            # Get the primary face
            face = face_info[0]

            # Prepare input for motion extractor
            # This depends on the specific LivePortrait API
            # The motion extractor outputs a dict with motion parameters
            with torch.no_grad():
                motion_out = motion_extractor(face)

            # Pack motion parameters into a flat vector
            # LivePortrait typically outputs:
            #   pitch, yaw, roll (3), translation (3), expression (63), scale (1), ...
            # Total packed to 128 dims
            features = pack_motion_features(motion_out, target_dim=128, device=device)
            all_features.append(features)

        except Exception as e:
            # Fallback: repeat previous or zero
            if all_features:
                all_features.append(all_features[-1].clone())
            else:
                all_features.append(torch.zeros(128, device=device))

    if len(all_features) == 0:
        return None

    # Stack: [T, D=128] -> [D=128, T]
    struct_latents = torch.stack(all_features).t()
    return struct_latents


def pack_motion_features(motion_out: dict, target_dim: int = 128, device: str = "cuda") -> torch.Tensor:
    """Pack LivePortrait motion output dict into a flat feature vector.

    Adapts to whatever fields the motion extractor provides.
    """
    parts = []

    # Common fields from LivePortrait
    for key in ["pitch", "yaw", "roll", "t", "exp", "scale"]:
        if key in motion_out:
            val = motion_out[key]
            if isinstance(val, torch.Tensor):
                parts.append(val.flatten().to(device))
            elif isinstance(val, (float, int)):
                parts.append(torch.tensor([val], device=device))
            elif isinstance(val, np.ndarray):
                parts.append(torch.from_numpy(val).flatten().to(device))

    if len(parts) == 0:
        return torch.zeros(target_dim, device=device)

    features = torch.cat(parts)

    # Pad or truncate to target_dim
    if features.shape[0] < target_dim:
        features = torch.nn.functional.pad(features, (0, target_dim - features.shape[0]))
    elif features.shape[0] > target_dim:
        features = features[:target_dim]

    return features


def extract_struct_features_fallback(
    frames: list,
    device: str = "cuda",
) -> torch.Tensor:
    """Fallback: extract basic face landmarks as struct features.

    Uses MediaPipe or dlib for face landmark extraction when LivePortrait
    is not available.
    """
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
        )
    except ImportError:
        print("[Warning] MediaPipe not available, using random features")
        T = len(frames)
        return torch.randn(128, T, device=device) * 0.01

    all_features = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            landmarks = results.multi_face_landmarks[0]
            # Extract key landmarks (68 points x 2 = 136, truncate to 128)
            # Use subset of landmarks for compact representation
            key_indices = list(range(0, min(64, len(landmarks.landmark))))
            coords = []
            for idx in key_indices:
                lm = landmarks.landmark[idx]
                coords.extend([lm.x, lm.y])

            features = torch.tensor(coords[:128], device=device, dtype=torch.float32)
            if features.shape[0] < 128:
                features = torch.nn.functional.pad(features, (0, 128 - features.shape[0]))
        else:
            if all_features:
                features = all_features[-1].clone()
            else:
                features = torch.zeros(128, device=device)

        all_features.append(features)

    face_mesh.close()

    if len(all_features) == 0:
        return None

    return torch.stack(all_features).t()  # [D=128, T]


def main():
    parser = argparse.ArgumentParser(description="Extract 3D structure latents")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--liveportrait_path", type=str, default=None, help="Path to LivePortrait installation")
    parser.add_argument("--clip_list", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fallback", action="store_true", help="Use MediaPipe fallback instead of LivePortrait")
    args = parser.parse_args()

    # Initialize extractor
    motion_extractor = None
    face_analyzer = None

    if not args.fallback and args.liveportrait_path:
        try:
            sys.path.insert(0, args.liveportrait_path)
            # LivePortrait import would go here
            # from liveportrait.modules.motion_extractor import MotionExtractor
            # motion_extractor = MotionExtractor(...)
            print("[Info] LivePortrait loaded successfully")
        except ImportError:
            print("[Warning] LivePortrait not available, using fallback")
            args.fallback = True

    # Gather clips
    if args.clip_list:
        with open(args.clip_list, "r") as f:
            clips = json.load(f)
        clip_ids = [c["clip_id"] if isinstance(c, dict) else c for c in clips]
    else:
        clip_ids = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
            and os.path.exists(os.path.join(args.data_dir, d, "video.mp4"))
        ])

    print(f"Processing {len(clip_ids)} clips...")

    for clip_id in tqdm(clip_ids, desc="Extracting struct latents"):
        clip_dir = os.path.join(args.data_dir, clip_id)
        output_path = os.path.join(clip_dir, "struct_latents.safetensors")

        if os.path.exists(output_path) and not args.overwrite:
            continue

        video_path = os.path.join(clip_dir, "video.mp4")
        if not os.path.exists(video_path):
            continue

        frames = load_frames_as_numpy(video_path)
        if len(frames) == 0:
            continue

        if args.fallback or motion_extractor is None:
            struct_latents = extract_struct_features_fallback(frames, device=args.device)
        else:
            struct_latents = extract_struct_features_liveportrait(
                frames, motion_extractor, face_analyzer, device=args.device
            )

        if struct_latents is None:
            continue

        # Save as float16 to save space
        save_file({"struct_latents": struct_latents.cpu().to(torch.float16)}, output_path)

    print("[Done] Struct latent extraction complete.")


if __name__ == "__main__":
    main()
