#!/usr/bin/env python3
"""
Compute identity preservation metrics: ACD (Average Content Distance) and CSIM.

Uses ArcFace (InsightFace) embeddings to measure how well the generated video
preserves the identity of the reference face.

Metrics:
- ACD: Mean L2 distance between reference and generated frame embeddings (lower = better)
- CSIM: Mean cosine similarity between reference and generated frame embeddings (higher = better)

Usage:
    python scripts/eval/compute_identity.py \
        --ref_dir /path/to/reference_images/ \
        --gen_dir /path/to/generated_videos/ \
        --batch_size 16
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from tqdm import tqdm


def load_arcface_model(device="cuda"):
    """Load ArcFace model for face identity embedding.

    Tries insightface first, falls back to a pretrained ResNet with
    ArcFace-like behavior.
    """
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(160, 160))
        return {"type": "insightface", "app": app}
    except ImportError:
        print("Warning: insightface not installed. Using torchvision ResNet50 as fallback.")
        print("Install insightface for accurate identity metrics: pip install insightface onnxruntime-gpu")
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except ImportError:
            from torchvision.models import resnet50
            model = resnet50(pretrained=True)
        model.fc = torch.nn.Identity()
        model = model.to(device).eval()
        return {"type": "resnet", "model": model, "device": device}


def extract_embedding_insightface(app, image_bgr):
    """Extract ArcFace embedding using insightface.

    Args:
        app: FaceAnalysis instance
        image_bgr: [H, W, 3] uint8 BGR image

    Returns:
        embedding: [512] float32 numpy array, or None if no face detected
    """
    faces = app.get(image_bgr)
    if len(faces) == 0:
        return None
    # Use the largest face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.normed_embedding


def extract_embedding_resnet(model, image_rgb, device="cuda"):
    """Extract embedding using ResNet50 fallback.

    Args:
        model: ResNet50 with fc replaced by Identity
        image_rgb: [H, W, 3] uint8 RGB numpy array

    Returns:
        embedding: [2048] float32 numpy array
    """
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor).squeeze(0).cpu().numpy()
    return emb / (np.linalg.norm(emb) + 1e-8)


def get_embedding(face_model, image_rgb):
    """Extract face embedding from an RGB image.

    Args:
        face_model: dict with 'type' and model
        image_rgb: [H, W, 3] uint8 RGB numpy array

    Returns:
        embedding: normalized float32 numpy array, or None
    """
    if face_model["type"] == "insightface":
        # insightface expects BGR
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return extract_embedding_insightface(face_model["app"], image_bgr)
    else:
        return extract_embedding_resnet(face_model["model"], image_rgb, face_model["device"])


def compute_identity_metrics(ref_dir, gen_dir, face_model, max_frames_per_video=16):
    """Compute ACD and CSIM between reference images and generated video frames.

    Expects paired data: for each video in gen_dir, there should be a
    corresponding reference image in ref_dir with the same stem name.

    Directory structure:
        ref_dir/
            clip_001.jpg (or .png)
            clip_002.jpg
            ...
        gen_dir/
            clip_001.mp4
            clip_002.mp4
            ...
    """
    gen_videos = sorted(glob.glob(os.path.join(gen_dir, "*.mp4")))
    if not gen_videos:
        gen_videos = sorted(glob.glob(os.path.join(gen_dir, "**/*.mp4"), recursive=True))

    all_acd = []
    all_csim = []
    skipped = 0

    for video_path in tqdm(gen_videos, desc="Computing identity metrics"):
        stem = os.path.splitext(os.path.basename(video_path))[0]

        # Find reference image
        ref_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            candidate = os.path.join(ref_dir, stem + ext)
            if os.path.exists(candidate):
                ref_path = candidate
                break

        if ref_path is None:
            skipped += 1
            continue

        # Load reference image
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            skipped += 1
            continue
        ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_emb = get_embedding(face_model, ref_img_rgb)
        if ref_emb is None:
            skipped += 1
            continue

        # Load generated video frames
        try:
            video, _, _ = read_video(video_path, pts_unit="sec")
        except Exception:
            skipped += 1
            continue

        T = video.shape[0]
        if T == 0:
            skipped += 1
            continue

        # Sample frames
        if T > max_frames_per_video:
            indices = torch.linspace(0, T - 1, max_frames_per_video).long().tolist()
        else:
            indices = list(range(T))

        frame_acds = []
        frame_csims = []

        for fi in indices:
            frame = video[fi].numpy()  # [H, W, C] uint8 RGB
            gen_emb = get_embedding(face_model, frame)
            if gen_emb is None:
                continue

            # ACD: L2 distance
            acd = np.linalg.norm(ref_emb - gen_emb)
            frame_acds.append(acd)

            # CSIM: cosine similarity
            csim = np.dot(ref_emb, gen_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(gen_emb) + 1e-8)
            frame_csims.append(csim)

        if frame_acds:
            all_acd.append(np.mean(frame_acds))
            all_csim.append(np.mean(frame_csims))

    if skipped > 0:
        print(f"  Skipped {skipped} videos (missing ref, no face detected, or read error)")

    if len(all_acd) == 0:
        print("  No valid pairs found!")
        return {"acd": float("nan"), "csim": float("nan"), "num_pairs": 0}

    acd_mean = float(np.mean(all_acd))
    csim_mean = float(np.mean(all_csim))

    return {
        "acd": acd_mean,
        "csim": csim_mean,
        "acd_std": float(np.std(all_acd)),
        "csim_std": float(np.std(all_csim)),
        "num_pairs": len(all_acd),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute identity preservation metrics (ACD, CSIM)")
    parser.add_argument("--ref_dir", type=str, required=True, help="Directory of reference face images")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory of generated videos")
    parser.add_argument("--max_frames_per_video", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    print("Loading face recognition model...")
    face_model = load_arcface_model(args.device)

    results = compute_identity_metrics(
        args.ref_dir, args.gen_dir, face_model,
        max_frames_per_video=args.max_frames_per_video,
    )

    print(f"\nIdentity Preservation Metrics:")
    print(f"  ACD (Average Content Distance): {results['acd']:.4f} (lower = better)")
    print(f"  CSIM (Cosine Similarity):       {results['csim']:.4f} (higher = better)")
    print(f"  Num valid pairs:                {results['num_pairs']}")

    if args.output_json:
        import json
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
