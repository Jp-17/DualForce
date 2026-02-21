"""
Step 6: Extract Audio Features using HuBERT-Large.

For each processed clip, extract audio features from HuBERT-Large
and save as .safetensors. Also saves the first frame image.

Input:  data_root/{clip_id}/video.mp4
Output: data_root/{clip_id}/audio_features.safetensors
        Contains: {"audio_features": tensor[T_audio, 1024]}
        data_root/{clip_id}/first_frame.safetensors
        Contains: {"first_frame": tensor[C=3, H, W]}

HuBERT-Large outputs 50Hz features (one 1024-dim vector per 20ms).
For 25fps video of T frames: T_audio = T * 2 (= T * 50/25).

Usage:
    python scripts/preprocess/06_extract_audio_features.py \
        --data_dir /path/to/processed_data \
        --hubert_model facebook/hubert-large-ls960-ft \
        --device cuda
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torchaudio
from safetensors.torch import save_file
from tqdm import tqdm


def extract_audio_from_video(video_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Extract audio waveform from video file.

    Returns: [1, T_samples] tensor at target_sr sample rate, or None.
    """
    try:
        waveform, sr = torchaudio.load(video_path)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform
    except Exception as e:
        print(f"[Warning] Failed to load audio from {video_path}: {e}")
        return None


def extract_first_frame(video_path: str) -> torch.Tensor:
    """Extract first frame as tensor [C=3, H, W] in [-1, 1] range."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    frame_tensor = frame_tensor * 2 - 1  # Normalize to [-1, 1]
    return frame_tensor


def main():
    parser = argparse.ArgumentParser(description="Extract audio features and first frame")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--hubert_model", type=str, default="facebook/hubert-large-ls960-ft",
                        help="HuBERT model name or path")
    parser.add_argument("--clip_list", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Load HuBERT model
    print(f"Loading HuBERT model: {args.hubert_model}...")
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.hubert_model)
    hubert = HubertModel.from_pretrained(args.hubert_model)
    hubert = hubert.to(args.device)
    hubert.eval()

    # Gather clips
    if args.clip_list:
        with open(args.clip_list, "r") as f:
            clips = json.load(f)
        clip_ids = [c["clip_id"] if isinstance(c, dict) else c for c in clips]
    else:
        clip_ids = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
            and os.path.exists(os.path.join(args.data_dir, d, "video.mp4"))
        ])

    print(f"Processing {len(clip_ids)} clips...")

    for clip_id in tqdm(clip_ids, desc="Extracting audio features"):
        clip_dir = os.path.join(args.data_dir, clip_id)
        audio_output = os.path.join(clip_dir, "audio_features.safetensors")
        frame_output = os.path.join(clip_dir, "first_frame.safetensors")

        video_path = os.path.join(clip_dir, "video.mp4")
        if not os.path.exists(video_path):
            continue

        # Extract audio features
        if not os.path.exists(audio_output) or args.overwrite:
            waveform = extract_audio_from_video(video_path, target_sr=16000)

            if waveform is not None and waveform.shape[1] > 0:
                # Process through HuBERT
                # Feature extractor expects raw waveform at 16kHz
                inputs = feature_extractor(
                    waveform.squeeze(0).numpy(),
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                )
                input_values = inputs.input_values.to(args.device)

                with torch.no_grad():
                    outputs = hubert(input_values)
                    # last_hidden_state: [1, T_audio, 1024]
                    audio_features = outputs.last_hidden_state.squeeze(0)  # [T_audio, 1024]

                save_file(
                    {"audio_features": audio_features.cpu().to(torch.float16)},
                    audio_output,
                )

        # Extract first frame
        if not os.path.exists(frame_output) or args.overwrite:
            first_frame = extract_first_frame(video_path)
            if first_frame is not None:
                save_file(
                    {"first_frame": first_frame.to(torch.float16)},
                    frame_output,
                )

    print("[Done] Audio feature and first frame extraction complete.")


if __name__ == "__main__":
    main()
