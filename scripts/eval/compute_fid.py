#!/usr/bin/env python3
"""
Compute Fréchet Inception Distance (FID) between generated and real video frames.

Extracts per-frame features using Inception-v3 and computes FID.
This measures single-frame visual quality (not temporal coherence).

Usage:
    python scripts/eval/compute_fid.py \
        --real_dir /path/to/real_videos/ \
        --gen_dir /path/to/generated_videos/ \
        --batch_size 32
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from tqdm import tqdm


def load_inception_model(device="cuda"):
    """Load pretrained Inception-v3 as a feature extractor."""
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights)
    except ImportError:
        from torchvision.models import inception_v3
        model = inception_v3(pretrained=True)

    # Use features before the final FC (2048-dim)
    model.fc = torch.nn.Identity()
    model.aux_logits = False
    model = model.to(device).eval()
    return model


class VideoFrameDataset(Dataset):
    """Dataset that loads video files and extracts all frames."""

    def __init__(self, video_dir, resolution=299, max_frames_per_video=16,
                 extensions=("mp4", "avi", "webm")):
        self.resolution = resolution
        self.max_frames = max_frames_per_video
        self.frames = []  # List of (video_path, frame_idx) or preloaded frames

        video_paths = []
        for ext in extensions:
            video_paths.extend(glob.glob(os.path.join(video_dir, f"**/*.{ext}"), recursive=True))
        video_paths.sort()

        if len(video_paths) == 0:
            raise ValueError(f"No video files found in {video_dir}")

        # Pre-scan to build frame index
        for path in tqdm(video_paths, desc="Scanning videos"):
            try:
                video, _, info = read_video(path, pts_unit="sec")
                T = video.shape[0]
                if T == 0:
                    continue
                # Sample up to max_frames evenly
                if T > self.max_frames:
                    indices = torch.linspace(0, T - 1, self.max_frames).long().tolist()
                else:
                    indices = list(range(T))
                for fi in indices:
                    self.frames.append((path, fi))
            except Exception:
                continue

        print(f"  Total frames indexed: {len(self.frames)}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        path, frame_idx = self.frames[idx]
        try:
            video, _, _ = read_video(path, pts_unit="sec")
            frame = video[frame_idx]  # [H, W, C] uint8
        except Exception:
            return torch.zeros(3, self.resolution, self.resolution)

        # [H, W, C] -> [C, H, W] float [0, 1]
        frame = frame.permute(2, 0, 1).float() / 255.0

        # Resize to Inception input size
        frame = F.interpolate(
            frame.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Normalize for Inception (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame = (frame - mean) / std

        return frame


def extract_features(model, dataloader, device="cuda"):
    """Extract Inception features from all frames."""
    all_features = []
    for batch in tqdm(dataloader, desc="Extracting Inception features"):
        batch = batch.to(device)
        with torch.no_grad():
            features = model(batch)
            if isinstance(features, tuple):
                features = features[0]
        all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)


def compute_fid(real_features, gen_features):
    """Compute Fréchet Inception Distance.

    FID = ||mu_r - mu_g||^2 + Tr(C_r + C_g - 2*sqrt(C_r * C_g))
    """
    mu_r = np.mean(real_features, axis=0)
    mu_g = np.mean(gen_features, axis=0)

    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(gen_features, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum(diff ** 2) + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)


def main():
    parser = argparse.ArgumentParser(description="Compute FID on video frames")
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--gen_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_frames_per_video", type=int, default=16)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading Inception-v3 model...")
    model = load_inception_model(device)

    print(f"Loading real video frames from: {args.real_dir}")
    real_dataset = VideoFrameDataset(args.real_dir, max_frames_per_video=args.max_frames_per_video)
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    print(f"Loading generated video frames from: {args.gen_dir}")
    gen_dataset = VideoFrameDataset(args.gen_dir, max_frames_per_video=args.max_frames_per_video)
    gen_loader = DataLoader(gen_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    print("Extracting real frame features...")
    real_features = extract_features(model, real_loader, device)

    print("Extracting generated frame features...")
    gen_features = extract_features(model, gen_loader, device)

    fid = compute_fid(real_features, gen_features)
    print(f"\nFID: {fid:.2f}")

    if args.output_json:
        import json
        results = {
            "fid": fid,
            "num_real_frames": len(real_dataset),
            "num_gen_frames": len(gen_dataset),
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")

    return fid


if __name__ == "__main__":
    main()
