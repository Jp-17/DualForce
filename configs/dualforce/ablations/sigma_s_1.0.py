# ============================================================
# Ablation 4b: sigma_s_max = 1.0 (symmetric noise)
# ============================================================
# Same noise range for struct as video (no asymmetry).
# Tests whether asymmetric noise (lower struct noise) matters.
# Expected: worse struct quality, potentially more diverse generation.
# ============================================================

_base_ = "../dualforce_train_8gpu.py"

diffusion_forcing = dict(
    sigma_v_max=1.0,
    sigma_v_min=0.0,
    sigma_s_max=1.0,    # Changed from 0.7 to 1.0 (symmetric)
    sigma_s_min=0.0,
    noise_sampling="uniform",
)

trainer = dict(
    save_path="./checkpoints/dualforce_ablation_sigma_s_1.0",
)

logger = dict(
    log_dir="./tensorboard/dualforce_ablation_sigma_s_1.0",
)
