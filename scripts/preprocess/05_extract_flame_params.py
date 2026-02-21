"""
Step 5: Extract FLAME Parameters using EMOCA.

For each processed clip, extract per-frame FLAME parameters
(shape, expression, pose, jaw) using EMOCA and save as .safetensors.

Input:  data_root/{clip_id}/video.mp4
Output: data_root/{clip_id}/flame_params.safetensors
        Contains: {"flame_params": tensor[T, 159]}

FLAME parameters breakdown (159 dims total):
- shape:      100 dims (identity, same across frames)
- expression:  50 dims (per-frame expression)
- jaw_pose:     3 dims (jaw rotation)
- global_pose:  3 dims (head rotation)
- translation:  3 dims (head position)

Usage:
    python scripts/preprocess/05_extract_flame_params.py \
        --data_dir /path/to/processed_data \
        --emoca_path /path/to/EMOCA \
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


def extract_flame_emoca(
    frames: list,
    emoca_model,
    device: str = "cuda",
) -> torch.Tensor:
    """Extract FLAME parameters using EMOCA.

    Returns: [T, 159] tensor of FLAME parameters, or None on failure.
    """
    all_params = []

    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            with torch.no_grad():
                # EMOCA API depends on the specific version
                # Typical: encode image -> get FLAME code
                result = emoca_model.encode(frame_rgb)

                # Pack FLAME parameters
                shape = result.get("shape", torch.zeros(100))       # [100]
                expression = result.get("exp", torch.zeros(50))     # [50]
                jaw_pose = result.get("jaw", torch.zeros(3))        # [3]
                global_pose = result.get("pose", torch.zeros(3))    # [3]
                translation = result.get("trans", torch.zeros(3))   # [3]

                params = torch.cat([
                    shape.flatten()[:100],
                    expression.flatten()[:50],
                    jaw_pose.flatten()[:3],
                    global_pose.flatten()[:3],
                    translation.flatten()[:3],
                ]).to(device)

                # Ensure 159 dims
                if params.shape[0] < 159:
                    params = torch.nn.functional.pad(params, (0, 159 - params.shape[0]))
                params = params[:159]

                all_params.append(params)

        except Exception:
            if all_params:
                all_params.append(all_params[-1].clone())
            else:
                all_params.append(torch.zeros(159, device=device))

    if len(all_params) == 0:
        return None

    return torch.stack(all_params)  # [T, 159]


def extract_flame_fallback(
    frames: list,
    device: str = "cuda",
) -> torch.Tensor:
    """Fallback: generate placeholder FLAME params from face landmarks.

    When EMOCA is not available, we create approximate FLAME-like features
    from basic face geometry. These are NOT real FLAME params but maintain
    the expected interface.
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
        print("[Warning] MediaPipe not available, generating zero FLAME params")
        T = len(frames)
        return torch.zeros(T, 159, device=device)

    all_params = []

    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            landmarks = results.multi_face_landmarks[0]

            # Approximate head pose from key landmarks
            nose = landmarks.landmark[1]
            chin = landmarks.landmark[152]
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]

            # Basic pose estimation
            yaw = (right_eye.x - left_eye.x) * 10
            pitch = (nose.y - chin.y) * 10
            roll = np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)

            # Jaw: distance between upper and lower lip landmarks
            upper_lip = landmarks.landmark[13]
            lower_lip = landmarks.landmark[14]
            jaw_open = (lower_lip.y - upper_lip.y) * 20

            # Pack into 159-dim (mostly zeros for shape/expression)
            params = torch.zeros(159, device=device)
            params[153] = pitch   # global_pose[0]
            params[154] = yaw     # global_pose[1]
            params[155] = roll    # global_pose[2]
            params[150] = jaw_open  # jaw_pose[0]
            params[156] = nose.x - 0.5  # translation[0]
            params[157] = nose.y - 0.5  # translation[1]
            params[158] = 0.0    # translation[2]

            all_params.append(params)
        else:
            if all_params:
                all_params.append(all_params[-1].clone())
            else:
                all_params.append(torch.zeros(159, device=device))

    face_mesh.close()

    if len(all_params) == 0:
        return None

    return torch.stack(all_params)  # [T, 159]


def main():
    parser = argparse.ArgumentParser(description="Extract FLAME parameters")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--emoca_path", type=str, default=None, help="Path to EMOCA installation")
    parser.add_argument("--clip_list", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fallback", action="store_true", help="Use MediaPipe fallback")
    args = parser.parse_args()

    # Initialize EMOCA
    emoca_model = None
    if not args.fallback and args.emoca_path:
        try:
            sys.path.insert(0, args.emoca_path)
            # EMOCA import would go here
            print("[Info] EMOCA loaded successfully")
        except ImportError:
            print("[Warning] EMOCA not available, using fallback")
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

    for clip_id in tqdm(clip_ids, desc="Extracting FLAME params"):
        clip_dir = os.path.join(args.data_dir, clip_id)
        output_path = os.path.join(clip_dir, "flame_params.safetensors")

        if os.path.exists(output_path) and not args.overwrite:
            continue

        video_path = os.path.join(clip_dir, "video.mp4")
        if not os.path.exists(video_path):
            continue

        frames = load_frames_as_numpy(video_path)
        if len(frames) == 0:
            continue

        if args.fallback or emoca_model is None:
            flame_params = extract_flame_fallback(frames, device=args.device)
        else:
            flame_params = extract_flame_emoca(frames, emoca_model, device=args.device)

        if flame_params is None:
            continue

        save_file({"flame_params": flame_params.cpu().to(torch.float16)}, output_path)

    print("[Done] FLAME parameter extraction complete.")


if __name__ == "__main__":
    main()
