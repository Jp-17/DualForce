#!/usr/bin/env python3
"""
Batch inference script for DualForce evaluation.

Generates videos for all test samples in a directory, producing output
compatible with the evaluation scripts in scripts/eval/.

Usage:
    python scripts/eval/batch_generate.py \
        --checkpoint /path/to/dualforce_checkpoint \
        --test_dir /path/to/test_data/ \
        --output_dir ./eval_results/generated/ \
        --ref_output_dir ./eval_results/references/

The test directory should contain per-clip subdirectories with:
    clip_001/
        first_frame.safetensors   (required)
        audio_features.safetensors (optional)
        struct_latents.safetensors (optional)
        caption.txt               (optional)
    clip_002/
        ...

Or a metadata.json listing clips.
"""

import argparse
import glob
import json
import os
import sys
import time

import torch
from safetensors.torch import load_file
from tqdm import tqdm


def find_test_clips(test_dir):
    """Find all test clips in the directory.

    Returns list of (clip_id, clip_dir) tuples.
    """
    # Check for metadata.json
    metadata_path = os.path.join(test_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

        clips = []
        if isinstance(metadata, list):
            for item in metadata:
                if isinstance(item, str):
                    clips.append((item, os.path.join(test_dir, item)))
                elif isinstance(item, dict):
                    clip_id = item.get("clip_id", item.get("id"))
                    clips.append((clip_id, os.path.join(test_dir, clip_id)))
        elif isinstance(metadata, dict):
            for key in ("clips", "samples", "test"):
                if key in metadata:
                    for item in metadata[key]:
                        if isinstance(item, str):
                            clips.append((item, os.path.join(test_dir, item)))
                        elif isinstance(item, dict):
                            clip_id = item.get("clip_id", item.get("id"))
                            clips.append((clip_id, os.path.join(test_dir, clip_id)))
                    break
        return clips

    # Fallback: scan for subdirectories with first_frame.safetensors
    clips = []
    for d in sorted(os.listdir(test_dir)):
        clip_dir = os.path.join(test_dir, d)
        if os.path.isdir(clip_dir):
            first_frame_path = os.path.join(clip_dir, "first_frame.safetensors")
            if os.path.exists(first_frame_path):
                clips.append((d, clip_dir))

    return clips


def main():
    parser = argparse.ArgumentParser(description="Batch generate DualForce videos for evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="DualForce checkpoint path")
    parser.add_argument("--test_dir", type=str, required=True, help="Directory with test clips")
    parser.add_argument("--output_dir", type=str, default="./eval_results/generated",
                        help="Output directory for generated videos")
    parser.add_argument("--ref_output_dir", type=str, default="./eval_results/references",
                        help="Output directory for reference images")

    # Generation parameters
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--window_stride", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=5.0)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--offload", choices=["none", "cpu"], default="cpu")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--max_clips", type=int, default=None, help="Max clips to generate")
    parser.add_argument("--default_caption", type=str, default="A person speaking naturally.",
                        help="Default caption if none provided")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ref_output_dir, exist_ok=True)

    # Find test clips
    clips = find_test_clips(args.test_dir)
    if not clips:
        print(f"No test clips found in {args.test_dir}")
        sys.exit(1)

    if args.max_clips:
        clips = clips[:args.max_clips]

    print(f"Found {len(clips)} test clips")

    # Load pipeline
    print("Loading DualForce pipeline...")
    from mova.diffusion.pipelines.pipeline_dualforce import DualForceInference

    pipe = DualForceInference.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16
    )

    if args.offload == "cpu":
        pipe.enable_model_cpu_offload(0)
    else:
        pipe.to(args.device)

    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    # Generate videos
    total_time = 0
    generated = 0
    skipped = 0

    for clip_id, clip_dir in tqdm(clips, desc="Generating videos"):
        output_path = os.path.join(args.output_dir, f"{clip_id}.mp4")
        ref_path = os.path.join(args.ref_output_dir, f"{clip_id}.png")

        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            continue

        # Load first frame
        first_frame_path = os.path.join(clip_dir, "first_frame.safetensors")
        if not os.path.exists(first_frame_path):
            print(f"  Skipping {clip_id}: no first_frame.safetensors")
            skipped += 1
            continue

        try:
            first_frame_data = load_file(first_frame_path)
            first_frame = first_frame_data.get(
                "first_frame", first_frame_data.get("ref_image")
            )
            if first_frame is None:
                first_frame = list(first_frame_data.values())[0]
        except Exception as e:
            print(f"  Skipping {clip_id}: {e}")
            skipped += 1
            continue

        # Ensure [1, C, H, W]
        if first_frame.dim() == 3:
            first_frame = first_frame.unsqueeze(0)

        # Load optional audio features
        audio_features = None
        audio_path = os.path.join(clip_dir, "audio_features.safetensors")
        if os.path.exists(audio_path):
            try:
                audio_data = load_file(audio_path)
                audio_features = audio_data.get("audio_features")
                if audio_features is not None and audio_features.dim() == 2:
                    audio_features = audio_features.unsqueeze(0)
            except Exception:
                pass

        # Load optional struct latents
        struct_latents = None
        struct_path = os.path.join(clip_dir, "struct_latents.safetensors")
        if os.path.exists(struct_path):
            try:
                struct_data = load_file(struct_path)
                struct_latents = struct_data.get("struct_latents")
                if struct_latents is not None and struct_latents.dim() == 2:
                    struct_latents = struct_latents.unsqueeze(0)
            except Exception:
                pass

        # Load caption
        caption = args.default_caption
        caption_path = os.path.join(clip_dir, "caption.txt")
        if os.path.exists(caption_path):
            with open(caption_path) as f:
                caption = f.read().strip() or args.default_caption

        # Generate
        start = time.time()
        try:
            video = pipe(
                prompt=caption,
                image=first_frame,
                struct_latents=struct_latents,
                audio_features=audio_features,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                video_fps=args.fps,
                num_inference_steps=args.num_inference_steps,
                window_size=args.window_size,
                window_stride=args.window_stride,
                cfg_scale=args.cfg_scale,
                output_type="pil",
                generator=generator,
            )
            elapsed = time.time() - start
            total_time += elapsed

            # Save video as mp4
            if isinstance(video, list) and len(video) > 0:
                from PIL import Image
                frames = video if isinstance(video[0], Image.Image) else video[0]

                # Save using imageio or ffmpeg
                try:
                    import imageio
                    writer = imageio.get_writer(output_path, fps=args.fps)
                    for frame in frames:
                        import numpy as np
                        writer.append_data(np.array(frame))
                    writer.close()
                except ImportError:
                    # Fallback: save frames as individual images
                    frame_dir = os.path.join(args.output_dir, f"{clip_id}_frames")
                    os.makedirs(frame_dir, exist_ok=True)
                    for i, frame in enumerate(frames):
                        frame.save(os.path.join(frame_dir, f"{i:04d}.png"))
                    # Use ffmpeg to combine
                    import subprocess
                    subprocess.run([
                        "ffmpeg", "-y", "-framerate", str(args.fps),
                        "-i", os.path.join(frame_dir, "%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        output_path,
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Save reference image
            from torchvision.utils import save_image
            ref_img = first_frame.squeeze(0)
            if ref_img.min() < 0:
                ref_img = (ref_img + 1) / 2  # [-1,1] -> [0,1]
            save_image(ref_img, ref_path)

            generated += 1

        except Exception as e:
            print(f"  Error generating {clip_id}: {e}")
            skipped += 1

    print(f"\nGeneration complete:")
    print(f"  Generated: {generated}")
    print(f"  Skipped:   {skipped}")
    if generated > 0:
        print(f"  Avg time:  {total_time / generated:.1f}s per video")
    print(f"  Output:    {args.output_dir}")
    print(f"  Refs:      {args.ref_output_dir}")

    # Save generation metadata
    meta = {
        "checkpoint": args.checkpoint,
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "generated": generated,
        "skipped": skipped,
    }
    with open(os.path.join(args.output_dir, "generation_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
