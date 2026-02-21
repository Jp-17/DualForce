#!/usr/bin/env python3
"""
MOVA Inference Verification Script.

Verifies that the base MOVA-360p checkpoint loads and runs correctly.
This is Phase 0.1 of the DualForce execution plan.

Usage (single GPU, with CPU offload):
    python scripts/verify_mova_inference.py \
        --ckpt_path /path/to/MOVA-360p \
        --ref_path /path/to/reference_image.jpg \
        --prompt "A person speaking" \
        --output_path ./output/mova_test.mp4

Usage (minimal test, just check model loads):
    python scripts/verify_mova_inference.py \
        --ckpt_path /path/to/MOVA-360p \
        --load-only
"""

import argparse
import os
import sys
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Verify MOVA inference")
    parser.add_argument("--ckpt_path", type=str,
                        default="/root/autodl-tmp/checkpoints/MOVA-360p",
                        help="Path to MOVA-360p checkpoint directory")
    parser.add_argument("--ref_path", type=str, default=None,
                        help="Path to reference image (required for full inference)")
    parser.add_argument("--prompt", type=str, default="A person speaking naturally.",
                        help="Text prompt for generation")
    parser.add_argument("--output_path", type=str, default="./output/mova_test.mp4",
                        help="Output video path")
    parser.add_argument("--num_frames", type=int, default=49,
                        help="Number of frames (lower for faster test)")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_inference_steps", type=int, default=20,
                        help="Denoising steps (lower for faster test)")
    parser.add_argument("--load-only", action="store_true",
                        help="Only load model and verify components, don't run inference")
    parser.add_argument("--offload", choices=["none", "cpu", "group"], default="cpu",
                        help="Offload strategy (cpu recommended for single GPU)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def verify_checkpoint_files(ckpt_path):
    """Check that all required checkpoint files exist."""
    required_dirs = [
        "video_dit",
        "video_dit_2",
        "audio_dit",
        "dual_tower_bridge",
        "video_vae",
        "audio_vae",
        "text_encoder",
    ]

    print(f"Checking checkpoint at: {ckpt_path}")
    all_ok = True
    for d in required_dirs:
        full_path = os.path.join(ckpt_path, d)
        exists = os.path.isdir(full_path)
        status = "[+]" if exists else "[-]"
        config_exists = os.path.exists(os.path.join(full_path, "config.json"))
        print(f"  {status} {d}/ {'(config.json found)' if config_exists else '(MISSING)' if not exists else ''}")
        if not exists:
            all_ok = False

    return all_ok


def load_and_verify(ckpt_path, offload="cpu"):
    """Load MOVA pipeline and verify components."""
    from mova.diffusion.pipelines.pipeline_mova import MOVA

    print("\nLoading MOVA pipeline...")
    start = time.time()

    pipe = MOVA.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16)

    elapsed = time.time() - start
    print(f"  Loaded in {elapsed:.1f}s")

    # Verify components
    components = {
        "video_dit": pipe.video_dit,
        "video_dit_2": pipe.video_dit_2,
        "audio_dit": pipe.audio_dit,
        "dual_tower_bridge": pipe.dual_tower_bridge,
        "video_vae": pipe.video_vae,
        "audio_vae": pipe.audio_vae,
        "text_encoder": pipe.text_encoder,
        "tokenizer": pipe.tokenizer,
    }

    print("\nComponent verification:")
    for name, component in components.items():
        if component is not None:
            if hasattr(component, 'parameters'):
                params = sum(p.numel() for p in component.parameters()) / 1e6
                print(f"  [+] {name}: {params:.1f}M params")
            else:
                print(f"  [+] {name}: loaded")
        else:
            print(f"  [-] {name}: None")

    # Apply offload
    if offload == "cpu":
        print("\nEnabling CPU offload...")
        pipe.enable_model_cpu_offload(0)
    elif offload == "none":
        print("\nMoving to CUDA...")
        pipe.to("cuda:0")
    elif offload == "group":
        print("\nEnabling group offload...")
        pipe.enable_group_offload(
            onload_device=torch.device("cuda:0"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=True,
        )

    return pipe


def run_inference(pipe, args):
    """Run a test inference."""
    from PIL import Image
    from mova.datasets.transforms.custom import crop_and_resize
    from mova.utils.data import save_video_with_audio

    print(f"\nRunning inference:")
    print(f"  Prompt: {args.prompt}")
    print(f"  Frames: {args.num_frames}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Steps: {args.num_inference_steps}")

    img = Image.open(args.ref_path).convert("RGB")
    ref_img = crop_and_resize(img, height=args.height, width=args.width)

    torch.manual_seed(args.seed)

    start = time.time()
    video, audio = pipe(
        prompt=args.prompt,
        negative_prompt="",
        num_frames=args.num_frames,
        image=ref_img,
        height=args.height,
        width=args.width,
        video_fps=24.0,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=5.0,
        cfg_scale=5.0,
        seed=args.seed,
    )
    elapsed = time.time() - start

    print(f"  Inference completed in {elapsed:.1f}s")
    print(f"  Video shape: {[v.size for v in video[0]] if isinstance(video[0], list) else 'tensor'}")

    # Save output
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    audio_save = audio[0].cpu().squeeze()

    save_video_with_audio(
        video[0],
        audio_save,
        args.output_path,
        fps=24.0,
        sample_rate=pipe.audio_sample_rate,
        quality=9,
    )
    print(f"  Saved to: {args.output_path}")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  Peak GPU memory: {peak_mem:.2f} GB")


def main():
    args = parse_args()

    print("=" * 60)
    print("  MOVA Inference Verification")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available. This script requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Step 1: Verify checkpoint files
    if not verify_checkpoint_files(args.ckpt_path):
        print("\nCheckpoint files incomplete. Cannot proceed.")
        sys.exit(1)

    # Step 2: Load model
    pipe = load_and_verify(args.ckpt_path, offload=args.offload)

    if args.load_only:
        print("\n[+] Load-only mode. Model loaded successfully.")
        print("    To run full inference, provide --ref_path and remove --load-only")
        sys.exit(0)

    # Step 3: Run inference
    if args.ref_path is None:
        print("\nNo --ref_path provided. Skipping inference.")
        print("Provide a reference image to run full inference.")
        sys.exit(0)

    if not os.path.exists(args.ref_path):
        print(f"\nReference image not found: {args.ref_path}")
        sys.exit(1)

    run_inference(pipe, args)

    print("\n[+] MOVA inference verification PASSED.")


if __name__ == "__main__":
    main()
