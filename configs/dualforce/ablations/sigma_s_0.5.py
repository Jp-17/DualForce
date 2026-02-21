# ============================================================
# Ablation 4a: sigma_s_max = 0.5 (less struct noise)
# ============================================================
# Reduce maximum noise on struct tokens.
# With lower noise, struct tokens carry more structure info at each step.
# Expected: better early convergence but potentially worse diversity.
# ============================================================

_base_ = "../dualforce_train_8gpu.py"

diffusion_forcing = dict(
    sigma_v_max=1.0,
    sigma_v_min=0.0,
    sigma_s_max=0.5,    # Changed from 0.7 to 0.5
    sigma_s_min=0.0,
    noise_sampling="uniform",
)

trainer = dict(
    save_path="./checkpoints/dualforce_ablation_sigma_s_0.5",
)

logger = dict(
    log_dir="./tensorboard/dualforce_ablation_sigma_s_0.5",
)
