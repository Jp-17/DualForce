# ============================================================
# Ablation 5: Long Sequence Evaluation Configs
# ============================================================
# These configs are for EVALUATION ONLY (not training).
# Test how quality degrades as sequence length increases.
# Evaluated at: 32, 128, 256, 512 frames.
# Key metrics: ACD over time (identity drift), FVD, APD.
# ============================================================

# This file defines evaluation-time parameters only.
# Training uses clip_length=32 as in the base config.

eval_configs = {
    "short_32": dict(
        num_frames=32,
        window_size=8,
        window_stride=4,
        num_inference_steps=20,
    ),
    "medium_128": dict(
        num_frames=128,
        window_size=16,
        window_stride=8,
        num_inference_steps=20,
    ),
    "long_256": dict(
        num_frames=256,
        window_size=16,
        window_stride=8,
        num_inference_steps=20,
    ),
    "very_long_512": dict(
        num_frames=512,
        window_size=32,
        window_stride=16,
        num_inference_steps=20,
    ),
}

# For each eval config, expected measurements:
# - ACD@t: identity distance at frame t vs reference (should stay low)
# - FVD: computed on 16-frame clips sampled throughout the video
# - APD: head pose drift over time
# - Visual quality: qualitative assessment of later frames
