# ============================================================
# Ablation 2: Unidirectional Bridge (struct→video only)
# ============================================================
# Only allow 3D structure to influence video, not the reverse.
# Tests whether bidirectional cross-attention is needed or if
# one-way influence (3D guides video) suffices.
# Expected: moderate degradation in 3D consistency, video quality similar.
# ============================================================

_base_ = "../dualforce_train_8gpu.py"

# Override bridge to unidirectional
bridge = dict(
    visual_layers=20,
    audio_layers=20,
    visual_hidden_dim=1536,
    audio_hidden_dim=1536,
    head_dim=128,
    interaction_strategy="shallow_focus",
    apply_cross_rope=True,
    apply_first_frame_bias_in_rope=False,
    trainable_condition_scale=False,
    pooled_adaln=False,
    audio_fps=25.0,
    # DualForce addition: only a2v (struct→video), no v2a (video→struct)
    unidirectional="a2v",
)

trainer = dict(
    save_path="./checkpoints/dualforce_ablation_unidirectional",
)

logger = dict(
    log_dir="./tensorboard/dualforce_ablation_unidirectional",
)
