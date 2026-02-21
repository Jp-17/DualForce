#!/usr/bin/env python3
"""
Compute lip-sync quality metrics: Sync-C (confidence) and Sync-D (distance).

Uses SyncNet to evaluate audio-visual synchronization in generated videos.
Higher Sync-C and lower Sync-D indicate better lip-sync quality.

SyncNet processes pairs of (audio_chunk, video_frames) and outputs embeddings
for each modality. In-sync pairs should have high cosine similarity.

Usage:
    python scripts/eval/compute_sync.py \
        --video_dir /path/to/generated_videos/ \
        --batch_size 8

Note: Requires SyncNet weights. Download from:
    https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model
"""

import argparse
import glob
import math
import os
import subprocess
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class SyncNetModel(nn.Module):
    """SyncNet architecture for audio-visual synchronization.

    Based on SyncNet_python by Chung & Zisserman.
    Processes 5-frame video windows and corresponding audio spectrograms.
    """

    def __init__(self):
        super().__init__()

        # Visual encoder (5 grayscale frames, 48x96 mouth crops)
        self.face_encoder = nn.Sequential(
            nn.Conv2d(5, 96, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.face_fc = nn.Sequential(
            nn.Linear(256 * 5 * 11, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
        )

        # Audio encoder (13 MFCC coefficients, 20 time steps)
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.audio_fc = nn.Sequential(
            nn.Linear(256 * 4 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
        )

    def forward_visual(self, x):
        """x: [B, 5, 48, 96] grayscale mouth crops."""
        out = self.face_encoder(x)
        out = out.view(out.size(0), -1)
        out = self.face_fc(out)
        return F.normalize(out, p=2, dim=1)

    def forward_audio(self, x):
        """x: [B, 1, 13, 20] MFCC spectrogram."""
        out = self.audio_encoder(x)
        out = out.view(out.size(0), -1)
        out = self.audio_fc(out)
        return F.normalize(out, p=2, dim=1)


def load_syncnet(weights_path=None, device="cuda"):
    """Load SyncNet model with optional pretrained weights."""
    model = SyncNetModel().to(device)
    if weights_path is not None and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded SyncNet weights from {weights_path}")
    else:
        print("  WARNING: No SyncNet weights provided. Results will be random.")
        print("  Download from: https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model")
    model.eval()
    return model


def extract_audio_mfcc(video_path, sr=16000, n_mfcc=13):
    """Extract MFCC features from video's audio track.

    Returns:
        mfcc: [T_audio, n_mfcc] numpy array
    """
    try:
        import librosa
    except ImportError:
        print("Warning: librosa not installed. pip install librosa")
        return None

    # Extract audio to temp wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ar", str(sr), "-ac", "1", "-f", "wav", tmp_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        audio, _ = librosa.load(tmp_path, sr=sr)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc).T  # [T, n_mfcc]
    return mfcc


def extract_mouth_crops(video_path, max_frames=None):
    """Extract grayscale mouth region crops from video frames.

    Uses face detection to locate the mouth region.

    Returns:
        crops: [T, 48, 96] numpy array of grayscale mouth crops
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Load face detector
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    except Exception:
        cap.release()
        return None

    crops = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            # Use largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            # Mouth region: lower half of face
            mouth_y = y + int(h * 0.55)
            mouth_h = int(h * 0.45)
            mouth_x = x + int(w * 0.15)
            mouth_w = int(w * 0.7)

            mouth = gray[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w]
            if mouth.size > 0:
                mouth = cv2.resize(mouth, (96, 48))
                crops.append(mouth.astype(np.float32) / 255.0)
            else:
                crops.append(np.zeros((48, 96), dtype=np.float32))
        else:
            crops.append(np.zeros((48, 96), dtype=np.float32))

        frame_idx += 1

    cap.release()

    if len(crops) == 0:
        return None
    return np.array(crops)


def compute_sync_metrics(video_dir, syncnet, device="cuda", syncnet_window=5, fps=25):
    """Compute Sync-C and Sync-D for all videos in a directory.

    Args:
        video_dir: Directory containing generated .mp4 files
        syncnet: Loaded SyncNet model
        syncnet_window: Number of frames per SyncNet window (5 for standard SyncNet)
        fps: Video frame rate

    Returns:
        dict with sync_c (confidence), sync_d (distance), and counts
    """
    video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    if not video_paths:
        video_paths = sorted(glob.glob(os.path.join(video_dir, "**/*.mp4"), recursive=True))

    all_sync_c = []
    all_sync_d = []
    skipped = 0

    for video_path in tqdm(video_paths, desc="Computing sync metrics"):
        # Extract mouth crops
        mouth_crops = extract_mouth_crops(video_path)
        if mouth_crops is None or len(mouth_crops) < syncnet_window:
            skipped += 1
            continue

        # Extract audio MFCC
        mfcc = extract_audio_mfcc(video_path)
        if mfcc is None:
            skipped += 1
            continue

        T_video = len(mouth_crops)
        # Audio frames per video frame at 16kHz/25fps
        audio_frames_per_video_frame = 20  # ~0.04s per frame at 25fps, 20 MFCC frames per window

        num_windows = T_video - syncnet_window + 1
        if num_windows <= 0:
            skipped += 1
            continue

        window_sync_scores = []

        for wi in range(0, num_windows, syncnet_window):
            # Visual: 5-frame window
            visual_window = mouth_crops[wi:wi + syncnet_window]  # [5, 48, 96]
            visual_tensor = torch.from_numpy(visual_window).unsqueeze(0).to(device)  # [1, 5, 48, 96]

            # Audio: corresponding MFCC window
            audio_start = int(wi * audio_frames_per_video_frame / syncnet_window)
            audio_end = audio_start + 20
            if audio_end > len(mfcc):
                break
            audio_window = mfcc[audio_start:audio_end]  # [20, 13]
            audio_tensor = torch.from_numpy(audio_window.T).unsqueeze(0).unsqueeze(0).float().to(device)
            # [1, 1, 13, 20]

            with torch.no_grad():
                v_emb = syncnet.forward_visual(visual_tensor)
                a_emb = syncnet.forward_audio(audio_tensor)

            # Cosine similarity
            cos_sim = F.cosine_similarity(v_emb, a_emb).item()
            window_sync_scores.append(cos_sim)

        if window_sync_scores:
            # Sync-C: mean confidence (higher = better sync)
            sync_c = np.mean(window_sync_scores)
            # Sync-D: mean euclidean distance (lower = better sync)
            sync_d = np.mean([1.0 - s for s in window_sync_scores])

            all_sync_c.append(sync_c)
            all_sync_d.append(sync_d)
        else:
            skipped += 1

    if skipped > 0:
        print(f"  Skipped {skipped} videos")

    if not all_sync_c:
        return {"sync_c": float("nan"), "sync_d": float("nan"), "num_videos": 0}

    return {
        "sync_c": float(np.mean(all_sync_c)),
        "sync_d": float(np.mean(all_sync_d)),
        "sync_c_std": float(np.std(all_sync_c)),
        "sync_d_std": float(np.std(all_sync_d)),
        "num_videos": len(all_sync_c),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute lip-sync metrics (Sync-C, Sync-D)")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory of generated videos with audio")
    parser.add_argument("--syncnet_weights", type=str, default=None, help="Path to SyncNet weights")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading SyncNet...")
    syncnet = load_syncnet(args.syncnet_weights, device)

    results = compute_sync_metrics(args.video_dir, syncnet, device)

    print(f"\nLip-Sync Metrics:")
    print(f"  Sync-C (confidence): {results['sync_c']:.4f} (higher = better)")
    print(f"  Sync-D (distance):   {results['sync_d']:.4f} (lower = better)")
    print(f"  Num videos:          {results['num_videos']}")

    if args.output_json:
        import json
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
