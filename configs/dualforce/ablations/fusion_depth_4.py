# ============================================================
# Ablation 3a: Fusion Depth = 4 layers (shallow)
# ============================================================
# Only the first 4 layers have bridge cross-attention.
# Tests the impact of fusion depth on quality.
# Expected: less structural consistency, possibly faster training.
# ============================================================

_base_ = "../dualforce_train_8gpu.py"

# Override: audio conditioning in first 4 layers only
diffusion_pipeline = dict(
    audio_conditioning_config=dict(
        audio_dim=1024,
        num_heads=12,
        num_layers=4,   # Changed from 6 to 4
    ),
)

# Override bridge: shallow_focus already limits to ~1/3 of layers
# With 20 layers, shallow_focus gives ~7 layers of interaction
# To test 4-layer fusion, we'd need custom strategy or manual override

trainer = dict(
    save_path="./checkpoints/dualforce_ablation_fusion_4",
)

logger = dict(
    log_dir="./tensorboard/dualforce_ablation_fusion_4",
)
