# ============================================================
# Ablation 1: No 3D Structure Stream
# ============================================================
# Remove the struct_dit entirely. Video-only with audio conditioning.
# Tests whether the 3D structure stream contributes to quality.
# Expected: worse ACD, worse APD, possibly worse FVD on long sequences.
# ============================================================

_base_ = "../dualforce_train_8gpu.py"

# Override: disable struct stream
struct_dit = None

# Override: no bridge (nothing to bridge)
bridge = None

# Override: loss weights (no struct loss, no flame loss)
diffusion_pipeline = dict(
    loss_config=dict(
        video_weight=1.0,
        struct_weight=0.0,    # Disabled
        flame_weight=0.0,     # No struct to align
        lip_sync_weight=0.3,  # Audio still used
    ),
    struct_dit_config=None,
    bridge_config=None,
)

# Trainer: only train video_dit + audio conditioning
trainer = dict(
    train_modules=[
        "video_dit",
        "audio_conditioning", "video_adaln",
        "lip_sync_video_proj", "lip_sync_audio_proj",
    ],
    save_path="./checkpoints/dualforce_ablation_no_struct",
)

logger = dict(
    log_dir="./tensorboard/dualforce_ablation_no_struct",
)
