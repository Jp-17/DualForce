"""
Step 3: Extract Video Latents using MOVA Video VAE.

For each processed clip, encode the video through the frozen Video VAE
and save latents as .safetensors.

Input:  data_root/{clip_id}/video.mp4
Output: data_root/{clip_id}/video_latents.safetensors
        Contains: {"video_latents": tensor[C=16, T', H', W']}

The VAE has:
- Spatial compression: 8x (512 -> 64)
- Temporal compression: 4x (e.g. 100 frames -> 25 latent frames)
- Latent channels: 16

Usage:
    python scripts/preprocess/03_extract_video_latents.py \
        --data_dir /path/to/processed_data \
        --vae_path /path/to/MOVA-360p/video_vae \
        --batch_frames 49 \
        --device cuda
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def load_video_frames(video_path: str, max_frames: int = None) -> torch.Tensor:
    """Load video frames as tensor [T, C, H, W] in [-1, 1] range."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB, [H, W, 3] -> [3, H, W]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frame = frame * 2 - 1  # Normalize to [-1, 1]
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()

    if len(frames) == 0:
        return None

    return torch.stack(frames)  # [T, C, H, W]


def main():
    parser = argparse.ArgumentParser(description="Extract video latents using MOVA Video VAE")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--vae_path", type=str, required=True, help="Path to MOVA Video VAE checkpoint directory")
    parser.add_argument("--clip_list", type=str, default=None, help="Optional JSON file with clip list (from step 02)")
    parser.add_argument("--batch_frames", type=int, default=49, help="Number of frames to process at once")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Load Video VAE
    print(f"Loading Video VAE from {args.vae_path}...")
    from diffusers.models.autoencoders import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(args.vae_path, torch_dtype=dtype)
    vae = vae.to(args.device)
    vae.eval()

    # Get normalization stats
    latents_mean = torch.tensor(vae.config.latents_mean, device=args.device, dtype=dtype)
    latents_std = torch.tensor(vae.config.latents_std, device=args.device, dtype=dtype)

    # Gather clips to process
    if args.clip_list:
        with open(args.clip_list, "r") as f:
            clips = json.load(f)
        clip_ids = [c["clip_id"] for c in clips]
    else:
        clip_ids = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
            and os.path.exists(os.path.join(args.data_dir, d, "video.mp4"))
        ])

    print(f"Processing {len(clip_ids)} clips...")

    for clip_id in tqdm(clip_ids, desc="Extracting video latents"):
        clip_dir = os.path.join(args.data_dir, clip_id)
        output_path = os.path.join(clip_dir, "video_latents.safetensors")

        if os.path.exists(output_path) and not args.overwrite:
            continue

        video_path = os.path.join(clip_dir, "video.mp4")
        if not os.path.exists(video_path):
            continue

        # Load frames
        frames = load_video_frames(video_path)
        if frames is None or frames.shape[0] < 1:
            print(f"[Skip] {clip_id}: no frames loaded")
            continue

        # VAE expects [B, C, T, H, W]
        # Process in chunks if video is very long
        T = frames.shape[0]
        all_latents = []

        for start in range(0, T, args.batch_frames):
            end = min(start + args.batch_frames, T)
            chunk = frames[start:end]  # [T_chunk, C, H, W]

            # Rearrange to [1, C, T_chunk, H, W]
            chunk = chunk.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
            chunk = chunk.to(device=args.device, dtype=dtype)

            with torch.no_grad():
                latent_dist = vae.encode(chunk).latent_dist
                latents = latent_dist.mode()  # [1, C_latent, T', H', W']

                # Normalize
                mean = latents_mean.view(1, -1, 1, 1, 1)
                std = latents_std.view(1, -1, 1, 1, 1)
                latents = (latents - mean) / std

            all_latents.append(latents.cpu().to(torch.float16))

        if len(all_latents) == 0:
            continue

        # Concatenate along temporal dimension
        video_latents = torch.cat(all_latents, dim=2).squeeze(0)  # [C=16, T', H', W']

        # Save
        save_file({"video_latents": video_latents}, output_path)

    print("[Done] Video latent extraction complete.")


if __name__ == "__main__":
    main()
