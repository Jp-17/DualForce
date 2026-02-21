#!/usr/bin/env python3
"""
DualForce Evaluation Runner — runs all metrics and produces a summary report.

This is the main entry point for evaluation. It:
1. Generates videos from test samples using the DualForce pipeline
2. Computes all metrics: FVD, FID, ACD/CSIM, Sync-C/Sync-D, APD
3. Saves a JSON report with all results

Usage (full evaluation):
    python scripts/eval/run_eval.py \
        --checkpoint /path/to/dualforce_checkpoint \
        --test_data /path/to/test_data/ \
        --output_dir ./eval_results/

Usage (metrics only, videos already generated):
    python scripts/eval/run_eval.py \
        --metrics_only \
        --real_dir /path/to/real_videos/ \
        --gen_dir /path/to/generated_videos/ \
        --ref_dir /path/to/reference_images/ \
        --output_dir ./eval_results/

Usage (specific metrics):
    python scripts/eval/run_eval.py \
        --metrics_only \
        --real_dir ... --gen_dir ... \
        --metrics fvd fid identity \
        --output_dir ./eval_results/
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime


def generate_videos(checkpoint, test_data, output_dir, args):
    """Generate videos from test samples using the DualForce pipeline.

    This reads test metadata and generates a video for each sample.
    """
    import glob
    import torch
    from safetensors.torch import load_file

    gen_dir = os.path.join(output_dir, "generated")
    ref_dir = os.path.join(output_dir, "references")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    # Load pipeline
    print("Loading DualForce pipeline...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from mova.diffusion.pipelines.pipeline_dualforce import DualForceInference

    pipe = DualForceInference.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload(0)

    # Load test metadata
    metadata_path = os.path.join(test_data, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Test metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    clips = metadata.get("clips", metadata.get("samples", []))
    if isinstance(metadata, list):
        clips = metadata

    print(f"Generating {len(clips)} test videos...")

    for clip_info in clips:
        if isinstance(clip_info, str):
            clip_id = clip_info
            clip_dir = os.path.join(test_data, clip_id)
        else:
            clip_id = clip_info.get("clip_id", clip_info.get("id", "unknown"))
            clip_dir = os.path.join(test_data, clip_id)

        output_path = os.path.join(gen_dir, f"{clip_id}.mp4")
        if os.path.exists(output_path) and not args.overwrite:
            continue

        # Load first frame
        first_frame_path = os.path.join(clip_dir, "first_frame.safetensors")
        if not os.path.exists(first_frame_path):
            print(f"  Skipping {clip_id}: no first_frame found")
            continue

        first_frame_data = load_file(first_frame_path)
        first_frame = first_frame_data.get("first_frame", first_frame_data.get("ref_image"))
        if first_frame is None:
            continue
        first_frame = first_frame.unsqueeze(0)  # [1, C, H, W]

        # Load audio features (optional)
        audio_path = os.path.join(clip_dir, "audio_features.safetensors")
        audio_features = None
        if os.path.exists(audio_path):
            audio_data = load_file(audio_path)
            audio_features = audio_data.get("audio_features")
            if audio_features is not None:
                audio_features = audio_features.unsqueeze(0)

        # Load struct latents (optional)
        struct_path = os.path.join(clip_dir, "struct_latents.safetensors")
        struct_latents = None
        if os.path.exists(struct_path):
            struct_data = load_file(struct_path)
            struct_latents = struct_data.get("struct_latents")
            if struct_latents is not None:
                struct_latents = struct_latents.unsqueeze(0)

        # Caption
        caption = "A person speaking naturally."
        caption_path = os.path.join(clip_dir, "caption.txt")
        if os.path.exists(caption_path):
            with open(caption_path) as f:
                caption = f.read().strip()

        # Generate
        try:
            video = pipe(
                prompt=caption,
                image=first_frame,
                struct_latents=struct_latents,
                audio_features=audio_features,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                output_type="pil",
            )

            # Save generated video
            from mova.utils.data import save_video
            save_video(video, output_path, fps=25.0)

            # Save reference image
            from torchvision.utils import save_image
            save_image(
                (first_frame.squeeze(0) + 1) / 2,  # [-1,1] -> [0,1]
                os.path.join(ref_dir, f"{clip_id}.png"),
            )
        except Exception as e:
            print(f"  Error generating {clip_id}: {e}")

    return gen_dir, ref_dir


def run_fvd(real_dir, gen_dir, output_dir, args):
    """Run FVD computation."""
    from scripts.eval.compute_fvd import (
        VideoDataset, load_i3d_model, extract_features, compute_fvd,
    )
    from torch.utils.data import DataLoader
    import torch

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_i3d_model(device)

    real_ds = VideoDataset(real_dir, num_frames=16)
    gen_ds = VideoDataset(gen_dir, num_frames=16)

    real_loader = DataLoader(real_ds, batch_size=args.batch_size, num_workers=4)
    gen_loader = DataLoader(gen_ds, batch_size=args.batch_size, num_workers=4)

    real_feats = extract_features(model, real_loader, device)
    gen_feats = extract_features(model, gen_loader, device)

    fvd = compute_fvd(real_feats, gen_feats)
    return {"fvd": fvd, "num_real": len(real_ds), "num_gen": len(gen_ds)}


def run_fid(real_dir, gen_dir, output_dir, args):
    """Run FID computation."""
    from scripts.eval.compute_fid import (
        VideoFrameDataset, load_inception_model, extract_features, compute_fid,
    )
    from torch.utils.data import DataLoader
    import torch

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_inception_model(device)

    real_ds = VideoFrameDataset(real_dir, max_frames_per_video=16)
    gen_ds = VideoFrameDataset(gen_dir, max_frames_per_video=16)

    real_loader = DataLoader(real_ds, batch_size=args.batch_size, num_workers=4)
    gen_loader = DataLoader(gen_ds, batch_size=args.batch_size, num_workers=4)

    real_feats = extract_features(model, real_loader, device)
    gen_feats = extract_features(model, gen_loader, device)

    fid = compute_fid(real_feats, gen_feats)
    return {"fid": fid}


def run_identity(ref_dir, gen_dir, output_dir, args):
    """Run identity preservation metrics."""
    from scripts.eval.compute_identity import load_arcface_model, compute_identity_metrics

    face_model = load_arcface_model(args.device)
    return compute_identity_metrics(ref_dir, gen_dir, face_model)


def run_sync(gen_dir, output_dir, args):
    """Run lip-sync metrics."""
    from scripts.eval.compute_sync import load_syncnet, compute_sync_metrics
    import torch

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    syncnet = load_syncnet(args.syncnet_weights, device)
    return compute_sync_metrics(gen_dir, syncnet, device)


def run_pose(ref_dir, gen_dir, output_dir, args):
    """Run head pose metrics."""
    from scripts.eval.compute_pose import load_pose_estimator, compute_pose_metrics

    pose_estimator = load_pose_estimator()
    return compute_pose_metrics(ref_dir, gen_dir, pose_estimator)


ALL_METRICS = ["fvd", "fid", "identity", "sync", "pose"]


def main():
    parser = argparse.ArgumentParser(description="DualForce Evaluation Runner")

    # Mode
    parser.add_argument("--metrics_only", action="store_true",
                        help="Skip generation, only compute metrics on existing videos")

    # Generation args
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="DualForce checkpoint path (required if not --metrics_only)")
    parser.add_argument("--test_data", type=str, default=None,
                        help="Test data directory with metadata.json")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")

    # Pre-existing directories (for --metrics_only)
    parser.add_argument("--real_dir", type=str, default=None, help="Real videos directory")
    parser.add_argument("--gen_dir", type=str, default=None, help="Generated videos directory")
    parser.add_argument("--ref_dir", type=str, default=None, help="Reference images directory")

    # Metrics selection
    parser.add_argument("--metrics", nargs="+", default=ALL_METRICS,
                        choices=ALL_METRICS, help="Which metrics to compute")

    # Common
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--syncnet_weights", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  DualForce Evaluation")
    print("=" * 60)
    print(f"  Metrics: {', '.join(args.metrics)}")
    print(f"  Output:  {args.output_dir}")
    print()

    # Step 1: Generate videos (or use existing)
    if args.metrics_only:
        real_dir = args.real_dir
        gen_dir = args.gen_dir
        ref_dir = args.ref_dir
        if gen_dir is None:
            parser.error("--gen_dir required with --metrics_only")
    else:
        if args.checkpoint is None or args.test_data is None:
            parser.error("--checkpoint and --test_data required for generation")
        gen_dir, ref_dir = generate_videos(args.checkpoint, args.test_data, args.output_dir, args)
        real_dir = args.real_dir  # May still need real videos for FVD/FID

    # Step 2: Compute metrics
    results = {"timestamp": datetime.now().isoformat(), "metrics": {}}
    start_time = time.time()

    if "fvd" in args.metrics:
        if real_dir is None:
            print("\nSkipping FVD: --real_dir not provided")
        else:
            print("\n--- Computing FVD ---")
            try:
                results["metrics"]["fvd"] = run_fvd(real_dir, gen_dir, args.output_dir, args)
            except Exception as e:
                print(f"FVD failed: {e}")
                results["metrics"]["fvd"] = {"error": str(e)}

    if "fid" in args.metrics:
        if real_dir is None:
            print("\nSkipping FID: --real_dir not provided")
        else:
            print("\n--- Computing FID ---")
            try:
                results["metrics"]["fid"] = run_fid(real_dir, gen_dir, args.output_dir, args)
            except Exception as e:
                print(f"FID failed: {e}")
                results["metrics"]["fid"] = {"error": str(e)}

    if "identity" in args.metrics:
        if ref_dir is None:
            print("\nSkipping identity: --ref_dir not provided")
        else:
            print("\n--- Computing Identity (ACD, CSIM) ---")
            try:
                results["metrics"]["identity"] = run_identity(ref_dir, gen_dir, args.output_dir, args)
            except Exception as e:
                print(f"Identity failed: {e}")
                results["metrics"]["identity"] = {"error": str(e)}

    if "sync" in args.metrics:
        print("\n--- Computing Lip-Sync (Sync-C, Sync-D) ---")
        try:
            results["metrics"]["sync"] = run_sync(gen_dir, args.output_dir, args)
        except Exception as e:
            print(f"Sync failed: {e}")
            results["metrics"]["sync"] = {"error": str(e)}

    if "pose" in args.metrics:
        if real_dir is None:
            print("\nSkipping pose: --real_dir not provided (using gen as ref)")
        print("\n--- Computing Head Pose (APD) ---")
        pose_ref_dir = real_dir or gen_dir
        try:
            results["metrics"]["pose"] = run_pose(pose_ref_dir, gen_dir, args.output_dir, args)
        except Exception as e:
            print(f"Pose failed: {e}")
            results["metrics"]["pose"] = {"error": str(e)}

    elapsed = time.time() - start_time
    results["total_time_seconds"] = elapsed

    # Step 3: Save report
    report_path = os.path.join(args.output_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    # Step 4: Print summary
    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)

    for metric_name, metric_data in results["metrics"].items():
        if "error" in metric_data:
            print(f"  {metric_name}: ERROR - {metric_data['error']}")
        else:
            if metric_name == "fvd":
                print(f"  FVD:    {metric_data['fvd']:.2f}")
            elif metric_name == "fid":
                print(f"  FID:    {metric_data['fid']:.2f}")
            elif metric_name == "identity":
                print(f"  ACD:    {metric_data['acd']:.4f} (lower=better)")
                print(f"  CSIM:   {metric_data['csim']:.4f} (higher=better)")
            elif metric_name == "sync":
                print(f"  Sync-C: {metric_data['sync_c']:.4f} (higher=better)")
                print(f"  Sync-D: {metric_data['sync_d']:.4f} (lower=better)")
            elif metric_name == "pose":
                print(f"  APD:    {metric_data['apd']:.2f}° (lower=better)")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Report saved to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
