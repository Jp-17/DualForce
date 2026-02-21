"""
DualForce Multi-Modal Dataset.

Loads pre-extracted features from .safetensors files for training:
- video_latents: Video VAE latents [C=16, T', H', W']
- struct_latents: LivePortrait motion features [D=128, T]
- audio_features: HuBERT features [T_audio, 1024]
- flame_params: EMOCA FLAME parameters [T, 159] (optional)
- first_frame_latent: Pre-encoded first frame [C=16, 1, H', W'] (optional)

Data layout:
    data_root/
    ├── metadata.json   # [{"clip_id": "clip_001", "caption": "...", "num_frames": 100, ...}]
    ├── clip_001/
    │   ├── video_latents.safetensors
    │   ├── struct_latents.safetensors
    │   ├── audio_features.safetensors
    │   ├── flame_params.safetensors       (optional)
    │   └── first_frame.safetensors
    ├── clip_002/
    │   └── ...
"""

import json
import os
import random

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from mova.registry import DATASETS


@DATASETS.register_module()
class DualForceDataset(Dataset):
    """
    Multi-modal dataset for DualForce training.

    Loads pre-extracted features and performs random temporal cropping.
    All features are expected to be temporally aligned at the same frame rate.
    """

    def __init__(
        self,
        data_root: str,
        metadata_file: str = "metadata.json",
        clip_length: int = 32,
        video_fps: float = 25.0,
        # Feature file names within each clip directory
        video_latent_file: str = "video_latents.safetensors",
        struct_latent_file: str = "struct_latents.safetensors",
        audio_feature_file: str = "audio_features.safetensors",
        flame_param_file: str = "flame_params.safetensors",
        first_frame_file: str = "first_frame.safetensors",
        # Tensor key names within safetensors files
        video_latent_key: str = "video_latents",
        struct_latent_key: str = "struct_latents",
        audio_feature_key: str = "audio_features",
        flame_param_key: str = "flame_params",
        first_frame_key: str = "first_frame",
        # Optional features
        load_flame: bool = True,
        # VAE temporal stride (for aligning video latent frames to raw frames)
        vae_temporal_stride: int = 4,
        # Audio feature rate (HuBERT features per second)
        audio_feature_rate: float = 50.0,
    ):
        super().__init__()
        self.data_root = data_root
        self.clip_length = clip_length
        self.video_fps = video_fps
        self.load_flame = load_flame
        self.vae_temporal_stride = vae_temporal_stride
        self.audio_feature_rate = audio_feature_rate

        # File names
        self.video_latent_file = video_latent_file
        self.struct_latent_file = struct_latent_file
        self.audio_feature_file = audio_feature_file
        self.flame_param_file = flame_param_file
        self.first_frame_file = first_frame_file

        # Tensor keys
        self.video_latent_key = video_latent_key
        self.struct_latent_key = struct_latent_key
        self.audio_feature_key = audio_feature_key
        self.flame_param_key = flame_param_key
        self.first_frame_key = first_frame_key

        # Load metadata
        metadata_path = os.path.join(data_root, metadata_file)
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Filter clips that are long enough
        # clip_length is in raw video frames; video latent temporal dim = raw_frames / vae_temporal_stride
        min_latent_frames = max(1, clip_length // vae_temporal_stride)
        valid = []
        for item in self.metadata:
            clip_dir = os.path.join(data_root, item["clip_id"])
            if not os.path.isdir(clip_dir):
                continue
            # Check minimum frame count if provided in metadata
            num_frames = item.get("num_frames", clip_length)
            if num_frames >= clip_length:
                valid.append(item)

        self.metadata = valid
        print(f"[DualForceDataset] Loaded {len(self.metadata)} clips from {metadata_path}")
        print(f"[DualForceDataset] clip_length={clip_length}, vae_stride={vae_temporal_stride}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        item = self.metadata[idx]
        clip_id = item["clip_id"]
        clip_dir = os.path.join(self.data_root, clip_id)
        caption = item.get("caption", "A person speaking.")

        # --------------------------------------------------
        # Load video latents: [C=16, T_latent, H', W']
        # --------------------------------------------------
        video_data = load_file(os.path.join(clip_dir, self.video_latent_file))
        video_latents = video_data[self.video_latent_key]  # [C, T_latent, H', W']

        T_latent = video_latents.shape[1]
        target_latent_frames = self.clip_length // self.vae_temporal_stride

        # Random temporal crop in latent space
        if T_latent > target_latent_frames:
            start = random.randint(0, T_latent - target_latent_frames)
        else:
            start = 0
        end = start + target_latent_frames

        video_latents = video_latents[:, start:end]

        # Pad if shorter than target
        if video_latents.shape[1] < target_latent_frames:
            pad_t = target_latent_frames - video_latents.shape[1]
            video_latents = torch.nn.functional.pad(video_latents, (0, 0, 0, 0, 0, pad_t))

        # --------------------------------------------------
        # Load struct latents: [D=128, T_struct]
        # --------------------------------------------------
        struct_data = load_file(os.path.join(clip_dir, self.struct_latent_file))
        struct_latents = struct_data[self.struct_latent_key]  # [D, T_struct]

        # Struct latents are at raw frame rate; align temporal crop
        raw_start = start * self.vae_temporal_stride
        raw_end = end * self.vae_temporal_stride

        struct_latents = struct_latents[:, raw_start:raw_end]

        # Downsample struct to match video latent temporal dim
        # struct is at raw fps, video latents are at raw_fps / vae_temporal_stride
        # We take every vae_temporal_stride-th struct frame
        struct_latents = struct_latents[:, ::self.vae_temporal_stride]

        if struct_latents.shape[1] < target_latent_frames:
            pad_t = target_latent_frames - struct_latents.shape[1]
            struct_latents = torch.nn.functional.pad(struct_latents, (0, pad_t))

        struct_latents = struct_latents[:, :target_latent_frames]

        # --------------------------------------------------
        # Load audio features: [T_audio, 1024]
        # --------------------------------------------------
        audio_data = load_file(os.path.join(clip_dir, self.audio_feature_file))
        audio_features = audio_data[self.audio_feature_key]  # [T_audio, 1024]

        # Align audio features to the temporal crop
        # Audio features are at audio_feature_rate Hz, video at video_fps Hz
        audio_per_video_frame = self.audio_feature_rate / self.video_fps
        audio_start = int(raw_start * audio_per_video_frame)
        audio_end = int(raw_end * audio_per_video_frame)

        audio_features = audio_features[audio_start:audio_end]

        # Target audio length for this clip
        target_audio_len = int(self.clip_length * audio_per_video_frame)
        if audio_features.shape[0] < target_audio_len:
            pad_t = target_audio_len - audio_features.shape[0]
            audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, pad_t))

        audio_features = audio_features[:target_audio_len]

        # --------------------------------------------------
        # Load FLAME parameters (optional): [T, 159]
        # --------------------------------------------------
        flame_params = None
        if self.load_flame:
            flame_path = os.path.join(clip_dir, self.flame_param_file)
            if os.path.exists(flame_path):
                flame_data = load_file(flame_path)
                flame_params = flame_data[self.flame_param_key]  # [T, 159]
                flame_params = flame_params[raw_start:raw_end]
                # Downsample to match video latent temporal dim
                flame_params = flame_params[::self.vae_temporal_stride]
                if flame_params.shape[0] < target_latent_frames:
                    pad_t = target_latent_frames - flame_params.shape[0]
                    flame_params = torch.nn.functional.pad(flame_params, (0, 0, 0, pad_t))
                flame_params = flame_params[:target_latent_frames]

        # --------------------------------------------------
        # Load first frame: [C, H, W] or [C, 1, H', W']
        # --------------------------------------------------
        first_frame_path = os.path.join(clip_dir, self.first_frame_file)
        if os.path.exists(first_frame_path):
            first_frame_data = load_file(first_frame_path)
            first_frame = first_frame_data[self.first_frame_key]  # [C, H, W]
        else:
            # Fallback: use zeros (will be re-encoded from video latents in training)
            C_img = 3
            H_img = video_latents.shape[-2] * 8  # Approximate from latent spatial dims
            W_img = video_latents.shape[-1] * 8
            first_frame = torch.zeros(C_img, H_img, W_img)

        result = {
            "video_latents": video_latents,       # [C=16, T', H', W']
            "struct_latents": struct_latents,      # [D=128, T']
            "audio_features": audio_features,      # [T_audio, 1024]
            "first_frame": first_frame,            # [C, H, W]
            "caption": caption,
            "clip_id": clip_id,
        }

        if flame_params is not None:
            result["flame_params"] = flame_params  # [T', 159]

        return result


def dualforce_collate_fn(batch):
    """Custom collate function for DualForce dataset."""
    video_latents = torch.stack([item["video_latents"] for item in batch])
    struct_latents = torch.stack([item["struct_latents"] for item in batch])
    audio_features = torch.stack([item["audio_features"] for item in batch])
    first_frames = torch.stack([item["first_frame"] for item in batch])
    captions = [item["caption"] for item in batch]
    clip_ids = [item["clip_id"] for item in batch]

    result = {
        "video_latents": video_latents,       # [B, C=16, T', H', W']
        "struct_latents": struct_latents,      # [B, D=128, T']
        "audio_features": audio_features,      # [B, T_audio, 1024]
        "first_frame": first_frames,           # [B, C, H, W]
        "caption": captions,
        "clip_id": clip_ids,
    }

    # FLAME params may not exist for all clips
    if "flame_params" in batch[0]:
        flame_params = torch.stack([item["flame_params"] for item in batch])
        result["flame_params"] = flame_params  # [B, T', 159]

    return result
