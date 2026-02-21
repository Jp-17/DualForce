# ============================================================
# Ablation 3b: Fusion Depth = 12 layers (deep)
# ============================================================
# First 12 layers have bridge cross-attention + audio conditioning.
# Tests whether deeper fusion improves quality.
# Expected: better structural consistency, higher compute cost.
# ============================================================

_base_ = "../dualforce_train_8gpu.py"

# Override: audio conditioning in first 12 layers
diffusion_pipeline = dict(
    audio_conditioning_config=dict(
        audio_dim=1024,
        num_heads=12,
        num_layers=12,   # Changed from 6 to 12
    ),
)

# Override bridge: use "full" strategy for all-layer interaction
bridge = dict(
    visual_layers=20,
    audio_layers=20,
    visual_hidden_dim=1536,
    audio_hidden_dim=1536,
    head_dim=128,
    interaction_strategy="full",  # All layers (not just shallow)
    apply_cross_rope=True,
    apply_first_frame_bias_in_rope=False,
    trainable_condition_scale=False,
    pooled_adaln=False,
    audio_fps=25.0,
)

trainer = dict(
    save_path="./checkpoints/dualforce_ablation_fusion_12",
)

logger = dict(
    log_dir="./tensorboard/dualforce_ablation_fusion_12",
)
