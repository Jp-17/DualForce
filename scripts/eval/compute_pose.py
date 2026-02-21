#!/usr/bin/env python3
"""
Compute head pose metrics: APD (Average Pose Distance).

Estimates 3D head pose (yaw, pitch, roll) from generated and reference videos,
then measures how well the generated pose trajectory matches the reference.

Usage:
    python scripts/eval/compute_pose.py \
        --ref_dir /path/to/real_videos/ \
        --gen_dir /path/to/generated_videos/
"""

import argparse
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm


def load_pose_estimator():
    """Load head pose estimator.

    Tries MediaPipe face mesh first (good 3D landmarks),
    falls back to OpenCV DNN-based approach.
    """
    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        return {"type": "mediapipe", "estimator": face_mesh}
    except ImportError:
        print("Warning: mediapipe not installed. Using OpenCV solvePnP fallback.")
        print("Install mediapipe: pip install mediapipe")
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return {"type": "opencv", "estimator": face_cascade}


def estimate_pose_mediapipe(face_mesh, frame_rgb):
    """Estimate head pose using MediaPipe face mesh.

    Returns:
        (yaw, pitch, roll) in degrees, or None if no face detected
    """
    results = face_mesh.process(frame_rgb)
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0]
    h, w = frame_rgb.shape[:2]

    # Key landmarks for pose estimation
    # Nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
    key_indices = [1, 152, 33, 263, 61, 291]
    image_points = np.array([
        [landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
        for i in key_indices
    ], dtype=np.float64)

    # 3D model points (generic face model)
    model_points = np.array([
        [0.0, 0.0, 0.0],          # Nose tip
        [0.0, -330.0, -65.0],     # Chin
        [-225.0, 170.0, -135.0],  # Left eye corner
        [225.0, 170.0, -135.0],   # Right eye corner
        [-150.0, -150.0, -125.0], # Left mouth corner
        [150.0, -150.0, -125.0],  # Right mouth corner
    ], dtype=np.float64)

    # Camera intrinsics (approximate)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat([rotation_mat, translation_vec])
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
        cv2.hconcat([pose_mat, np.array([[0, 0, 0, 1]], dtype=np.float64)])[:3]
    )

    yaw = euler_angles[1, 0]
    pitch = euler_angles[0, 0]
    roll = euler_angles[2, 0]

    return (yaw, pitch, roll)


def estimate_pose_opencv(face_cascade, frame_bgr):
    """Rough pose estimation using face detection bounding box position.

    This is a very rough approximation — only useful as a fallback.
    Returns (yaw, pitch, roll) based on face position heuristics.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    img_h, img_w = frame_bgr.shape[:2]

    # Rough yaw from horizontal position
    center_x = (x + w / 2) / img_w
    yaw = (center_x - 0.5) * 60  # Approximate degrees

    # Rough pitch from vertical position
    center_y = (y + h / 2) / img_h
    pitch = (center_y - 0.45) * 40

    roll = 0.0  # Can't estimate from bbox

    return (yaw, pitch, roll)


def extract_pose_trajectory(video_path, pose_estimator, max_frames=None):
    """Extract per-frame head pose from a video.

    Returns:
        poses: [T, 3] numpy array of (yaw, pitch, roll) in degrees
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    poses = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        if pose_estimator["type"] == "mediapipe":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose = estimate_pose_mediapipe(pose_estimator["estimator"], frame_rgb)
        else:
            pose = estimate_pose_opencv(pose_estimator["estimator"], frame)

        if pose is not None:
            poses.append(pose)
        else:
            # Use previous pose or zeros
            if poses:
                poses.append(poses[-1])
            else:
                poses.append((0.0, 0.0, 0.0))

        frame_idx += 1

    cap.release()

    if len(poses) == 0:
        return None
    return np.array(poses)


def compute_apd(ref_poses, gen_poses):
    """Compute Average Pose Distance between two pose trajectories.

    Args:
        ref_poses: [T, 3] reference (yaw, pitch, roll)
        gen_poses: [T, 3] generated (yaw, pitch, roll)

    Returns:
        APD: mean L2 distance in degrees
    """
    min_len = min(len(ref_poses), len(gen_poses))
    ref = ref_poses[:min_len]
    gen = gen_poses[:min_len]
    distances = np.linalg.norm(ref - gen, axis=1)
    return float(np.mean(distances))


def compute_pose_metrics(ref_dir, gen_dir, pose_estimator, max_frames=None):
    """Compute APD between paired reference and generated videos.

    Directory structure:
        ref_dir/
            clip_001.mp4
            ...
        gen_dir/
            clip_001.mp4
            ...
    """
    gen_videos = sorted(glob.glob(os.path.join(gen_dir, "*.mp4")))
    if not gen_videos:
        gen_videos = sorted(glob.glob(os.path.join(gen_dir, "**/*.mp4"), recursive=True))

    all_apd = []
    per_axis = {"yaw": [], "pitch": [], "roll": []}
    skipped = 0

    for gen_path in tqdm(gen_videos, desc="Computing pose metrics"):
        stem = os.path.splitext(os.path.basename(gen_path))[0]
        ref_path = os.path.join(ref_dir, stem + ".mp4")

        if not os.path.exists(ref_path):
            skipped += 1
            continue

        ref_poses = extract_pose_trajectory(ref_path, pose_estimator, max_frames)
        gen_poses = extract_pose_trajectory(gen_path, pose_estimator, max_frames)

        if ref_poses is None or gen_poses is None:
            skipped += 1
            continue

        apd = compute_apd(ref_poses, gen_poses)
        all_apd.append(apd)

        # Per-axis distances
        min_len = min(len(ref_poses), len(gen_poses))
        for i, axis in enumerate(["yaw", "pitch", "roll"]):
            axis_dist = np.mean(np.abs(ref_poses[:min_len, i] - gen_poses[:min_len, i]))
            per_axis[axis].append(float(axis_dist))

    if skipped > 0:
        print(f"  Skipped {skipped} videos")

    if not all_apd:
        return {"apd": float("nan"), "num_pairs": 0}

    return {
        "apd": float(np.mean(all_apd)),
        "apd_std": float(np.std(all_apd)),
        "yaw_error": float(np.mean(per_axis["yaw"])) if per_axis["yaw"] else float("nan"),
        "pitch_error": float(np.mean(per_axis["pitch"])) if per_axis["pitch"] else float("nan"),
        "roll_error": float(np.mean(per_axis["roll"])) if per_axis["roll"] else float("nan"),
        "num_pairs": len(all_apd),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute head pose metrics (APD)")
    parser.add_argument("--ref_dir", type=str, required=True, help="Directory of reference videos")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory of generated videos")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    print("Loading pose estimator...")
    pose_estimator = load_pose_estimator()

    results = compute_pose_metrics(
        args.ref_dir, args.gen_dir, pose_estimator,
        max_frames=args.max_frames,
    )

    print(f"\nHead Pose Metrics:")
    print(f"  APD (Average Pose Distance): {results['apd']:.2f}° (lower = better)")
    print(f"  Yaw error:   {results.get('yaw_error', 'N/A'):.2f}°")
    print(f"  Pitch error: {results.get('pitch_error', 'N/A'):.2f}°")
    print(f"  Roll error:  {results.get('roll_error', 'N/A'):.2f}°")
    print(f"  Num pairs:   {results['num_pairs']}")

    if args.output_json:
        import json
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
