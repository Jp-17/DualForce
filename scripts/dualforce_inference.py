#!/usr/bin/env python3
"""
DualForce Inference Script.

Generate talking head videos autoregressively from a reference image and prompt.

Usage:
    python scripts/dualforce_inference.py \
        --checkpoint_path ./checkpoints/dualforce/best \
        --image_path ./test_image.jpg \
        --prompt "A person speaking naturally" \
        --num_frames 49 \
        --output_path ./output.mp4

    # With struct/audio conditioning:
    python scripts/dualforce_inference.py \
        --checkpoint_path ./checkpoints/dualforce/best \
        --image_path ./test_image.jpg \
        --struct_path ./struct_latents.safetensors \
        --audio_path ./audio_features.safetensors \
        --prompt "A person speaking" \
        --num_frames 97 \
        --output_path ./output.mp4
"""

import argparse
import os

import torch
import torchvision.io as tvio
from PIL import Image
from torchvision import transforms

from mova.diffusion.pipelines.pipeline_dualforce import DualForceInference


def load_image(image_path: str, height: int, width: int) -> torch.Tensor:
    """Load and preprocess reference image."""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(img).unsqueeze(0)  # [1, 3, H, W]


def save_video(frames, output_path: str, fps: float = 25.0):
    """Save generated video frames."""
    if isinstance(frames, list):
        # PIL frames
        import imageio
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
        for frame in frames[0]:  # batch dim
            import numpy as np
            writer.append_data(np.array(frame))
        writer.close()
    elif isinstance(frames, torch.Tensor):
        # [B, C, T, H, W] tensor
        video = frames[0].permute(1, 0, 2, 3)  # [T, C, H, W]
        video = (video * 255).clamp(0, 255).byte()
        tvio.write_video(output_path, video.permute(0, 2, 3, 1).cpu(), fps=fps)

    print(f"Saved video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DualForce Inference")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to trained DualForce checkpoint")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Reference image for first frame")
    parser.add_argument("--prompt", type=str, default="A person speaking naturally.",
                        help="Text prompt")
    parser.add_argument("--struct_path", type=str, default=None,
                        help="Path to struct_latents.safetensors (optional)")
    parser.add_argument("--audio_path", type=str, default=None,
                        help="Path to audio_features.safetensors (optional)")
    # Generation params
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--window_stride", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    # Output
    parser.add_argument("--output_path", type=str, default="./output_dualforce.mp4")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    print(f"Loading DualForce from {args.checkpoint_path}...")
    pipeline = DualForceInference.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    pipeline.eval()

    # Load reference image
    image = load_image(args.image_path, args.height, args.width).to(device)

    # Load optional struct/audio conditioning
    struct_latents = None
    if args.struct_path:
        from safetensors.torch import load_file
        data = load_file(args.struct_path)
        struct_latents = data["struct_latents"].unsqueeze(0).to(device)

    audio_features = None
    if args.audio_path:
        from safetensors.torch import load_file
        data = load_file(args.audio_path)
        audio_features = data["audio_features"].unsqueeze(0).to(device)

    # Generate
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"Generating {args.num_frames} frames at {args.height}x{args.width}...")
    output = pipeline(
        prompt=args.prompt,
        image=image,
        struct_latents=struct_latents,
        audio_features=audio_features,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        window_size=args.window_size,
        window_stride=args.window_stride,
        cfg_scale=args.cfg_scale,
        device=device,
        generator=generator,
    )

    # Save
    save_video(output, args.output_path)
    print("Done!")


if __name__ == "__main__":
    main()
