#!/usr/bin/env python3
"""
Compute Fréchet Video Distance (FVD) between generated and real videos.

Uses I3D features (Kinetics-pretrained) to compute the Fréchet distance
in feature space. Lower FVD = better temporal coherence and visual quality.

Usage:
    python scripts/eval/compute_fvd.py \
        --real_dir /path/to/real_videos/ \
        --gen_dir /path/to/generated_videos/ \
        --num_frames 16 \
        --batch_size 8
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


def load_i3d_model(device="cuda"):
    """Load pretrained I3D model for feature extraction.

    Uses torchvision's video classification model as a feature extractor.
    Falls back to a ResNet3D-18 if I3D is not available.
    """
    try:
        from torchvision.models.video import r3d_18, R3D_18_Weights
        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights)
    except ImportError:
        from torchvision.models.video import r3d_18
        model = r3d_18(pretrained=True)

    # Remove final classification layer — use features before it
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    return model


class VideoDataset(Dataset):
    """Dataset that loads video files and samples fixed-length clips."""

    def __init__(self, video_dir, num_frames=16, resolution=224, extensions=("mp4", "avi", "webm")):
        self.num_frames = num_frames
        self.resolution = resolution
        self.video_paths = []
        for ext in extensions:
            self.video_paths.extend(glob.glob(os.path.join(video_dir, f"**/*.{ext}"), recursive=True))
        self.video_paths.sort()
        if len(self.video_paths) == 0:
            raise ValueError(f"No video files found in {video_dir}")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        try:
            video, _, info = read_video(path, pts_unit="sec")
            # video: [T, H, W, C] uint8
        except Exception as e:
            print(f"Warning: Failed to read {path}: {e}")
            return torch.zeros(3, self.num_frames, self.resolution, self.resolution)

        T = video.shape[0]
        if T == 0:
            return torch.zeros(3, self.num_frames, self.resolution, self.resolution)

        # Sample num_frames evenly
        if T >= self.num_frames:
            indices = torch.linspace(0, T - 1, self.num_frames).long()
        else:
            indices = torch.arange(T)
            # Pad by repeating last frame
            pad = torch.full((self.num_frames - T,), T - 1, dtype=torch.long)
            indices = torch.cat([indices, pad])

        video = video[indices]  # [num_frames, H, W, C]

        # Convert to [C, T, H, W] float32 [0, 1]
        video = video.permute(3, 0, 1, 2).float() / 255.0

        # Resize to resolution
        video = F.interpolate(
            video.unsqueeze(0),
            size=(self.num_frames, self.resolution, self.resolution),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        # Normalize for I3D (ImageNet stats)
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        video = (video - mean) / std

        return video


def extract_features(model, dataloader, device="cuda"):
    """Extract I3D features from all videos in dataloader."""
    all_features = []
    for batch in tqdm(dataloader, desc="Extracting features"):
        batch = batch.to(device)
        with torch.no_grad():
            features = model(batch)  # [B, feat_dim]
        all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)


def compute_fvd(real_features, gen_features):
    """Compute Fréchet Video Distance between two sets of features.

    FVD = ||mu_r - mu_g||^2 + Tr(C_r + C_g - 2*sqrt(C_r * C_g))
    """
    mu_r = np.mean(real_features, axis=0)
    mu_g = np.mean(gen_features, axis=0)

    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(gen_features, rowvar=False)

    diff = mu_r - mu_g

    # Compute sqrt of product of covariances
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)

    # Handle numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = np.sum(diff ** 2) + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fvd)


def main():
    parser = argparse.ArgumentParser(description="Compute FVD")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory of real videos")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory of generated videos")
    parser.add_argument("--num_frames", type=int, default=16, help="Frames per clip for I3D")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resolution", type=int, default=224, help="I3D input resolution")
    parser.add_argument("--output_json", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading I3D model...")
    model = load_i3d_model(device)

    print(f"Loading real videos from: {args.real_dir}")
    real_dataset = VideoDataset(args.real_dir, num_frames=args.num_frames, resolution=args.resolution)
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    print(f"  Found {len(real_dataset)} real videos")

    print(f"Loading generated videos from: {args.gen_dir}")
    gen_dataset = VideoDataset(args.gen_dir, num_frames=args.num_frames, resolution=args.resolution)
    gen_loader = DataLoader(gen_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    print(f"  Found {len(gen_dataset)} generated videos")

    print("Extracting real video features...")
    real_features = extract_features(model, real_loader, device)
    print(f"  Real features shape: {real_features.shape}")

    print("Extracting generated video features...")
    gen_features = extract_features(model, gen_loader, device)
    print(f"  Generated features shape: {gen_features.shape}")

    fvd = compute_fvd(real_features, gen_features)
    print(f"\nFVD: {fvd:.2f}")

    if args.output_json:
        import json
        results = {
            "fvd": fvd,
            "num_real": len(real_dataset),
            "num_gen": len(gen_dataset),
            "num_frames": args.num_frames,
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")

    return fvd


if __name__ == "__main__":
    main()
