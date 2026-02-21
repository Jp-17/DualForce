# ============================================================
# DualForce Training Configuration
# ============================================================
# Based on MOVA but with:
# - Shrunk video DiT (5120->1536, 40->20 layers)
# - 3D structure stream replacing audio tower
# - Single DiT (no video_dit_2)
# - Causal temporal attention
# - Diffusion Forcing (per-frame independent noise)
# ============================================================

# --------------------------------------------------
# Model Configuration
# --------------------------------------------------
# MOVA-Lite Video DiT: significantly shrunk from MOVA-360p
# Original: dim=5120, ffn=13824, heads=40, layers=40
# DualForce: dim=1536, ffn=6144, heads=12, layers=20
video_dit = dict(
    dim=1536,
    in_dim=36,          # 16 (VAE z) + 4 (mask) + 16 (first_frame latent)
    ffn_dim=6144,       # 4x hidden dim
    out_dim=16,         # VAE z_dim
    text_dim=4096,      # UMT5 text embedding dim
    freq_dim=256,       # Frequency embedding dim
    eps=1e-6,
    patch_size=(1, 2, 2),   # No temporal downsampling (matches MOVA)
    num_heads=12,       # head_dim = 1536/12 = 128
    num_layers=20,
    has_image_input=False,
    has_image_pos_emb=False,
    has_ref_conv=False,
    require_vae_embedding=True,
    require_clip_embedding=False,
    # DualForce additions
    causal_temporal=True,   # Enable causal temporal attention
)

# 3D Structure DiT: replaces MOVA's audio DiT
# Same dim as audio DiT was (1536) to preserve bridge compatibility
struct_dit = dict(
    dim=1536,
    in_dim=128,         # LivePortrait struct latent dim
    ffn_dim=6144,       # 4x hidden dim
    out_dim=128,        # Output same dim as input
    text_dim=4096,      # UMT5 text embedding dim
    freq_dim=256,
    eps=1e-6,
    patch_size=(1,),    # 1D temporal tokens, no downsampling
    num_heads=12,       # head_dim = 128
    num_layers=20,      # Same depth as video DiT
    has_image_input=False,
    require_vae_embedding=False,    # No VAE embedding for struct
    require_clip_embedding=True,     # Keep CLIP conditioning
    causal_temporal=True,
)

# Bridge: video <-> 3D structure bidirectional attention
bridge = dict(
    visual_layers=20,       # Match video DiT
    audio_layers=20,        # Match struct DiT (using audio_layers key for compat)
    visual_hidden_dim=1536, # Match video DiT dim
    audio_hidden_dim=1536,  # Match struct DiT dim (was 1536 in MOVA)
    head_dim=128,
    interaction_strategy="shallow_focus",  # DualForce: shallow fusion (MOVA was "full")
    apply_cross_rope=True,
    apply_first_frame_bias_in_rope=False,
    trainable_condition_scale=False,
    pooled_adaln=False,
    audio_fps=25.0,     # 3D struct shares video FPS (not audio 50Hz)
)

# Diffusion pipeline
diffusion_pipeline = dict(
    type="DualForceTrain_from_pretrained",
    from_pretrained=None,   # Train from scratch (set to checkpoint path to resume)
    use_gradient_checkpointing=True,
    use_gradient_checkpointing_offload=True,
    # Component configs (used when from_pretrained=None)
    video_dit_config=video_dit,
    struct_dit_config=struct_dit,
    bridge_config=bridge,
    scheduler_config=dict(),  # DiffusionForcingScheduler defaults
    loss_config=dict(
        video_weight=1.0,
        struct_weight=0.5,
        flame_weight=0.1,
        lip_sync_weight=0.3,
    ),
    # Frozen model paths (required for from-scratch training)
    vae_path="/root/autodl-tmp/checkpoints/MOVA-360p/video_vae",
    text_encoder_path="/root/autodl-tmp/checkpoints/MOVA-360p/text_encoder",
)

# --------------------------------------------------
# Data Configuration
# --------------------------------------------------
data = dict(
    dataset=dict(
        type="DualForceDataset",
        data_root="/path/to/preprocessed_data",
        metadata_file="metadata.json",
        num_frames=32,          # Shorter clips for training
        height=352,
        width=640,
        video_fps=25.0,         # 25fps for talking head
        clip_length=32,         # Frames per training clip
        # Feature cache paths (pre-extracted)
        video_latent_key="video_latents",
        struct_latent_key="struct_latents",
        audio_feature_key="audio_features",
        flame_param_key="flame_params",
        ref_feature_key="ref_features",
    ),
    transform=None,
    batch_size=2,       # Per GPU
    num_workers=4,
)

# --------------------------------------------------
# Optimizer Configuration
# --------------------------------------------------
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-8,
)

# --------------------------------------------------
# Loss Configuration (DualForce specific)
# --------------------------------------------------
loss = dict(
    video_weight=1.0,       # L_video: flow matching on video latents
    struct_weight=0.5,      # L_struct: flow matching on 3D struct latents
    flame_weight=0.1,       # L_flame: FLAME parameter alignment
    lip_sync_weight=0.3,    # L_lip_sync: contrastive lip-sync loss
)

# --------------------------------------------------
# Diffusion Forcing Configuration
# --------------------------------------------------
diffusion_forcing = dict(
    # Per-frame noise schedule
    sigma_v_max=1.0,        # Video noise range [0, 1.0]
    sigma_v_min=0.0,
    sigma_s_max=0.7,        # Struct noise range [0, 0.7] (smaller = more stable)
    sigma_s_min=0.0,
    # Noise sampling: each frame gets independent sigma
    noise_sampling="uniform",   # "uniform" or "logit_normal"
)

# --------------------------------------------------
# FSDP Configuration
# --------------------------------------------------
fsdp = dict(
    sharding_strategy="FULL_SHARD",
    cpu_offload=True,
    backward_prefetch="BACKWARD_PRE",
    reshard_after_forward=True,
)

# --------------------------------------------------
# Logger Configuration
# --------------------------------------------------
logger = dict(
    log_dir="./tensorboard/dualforce",
)

# --------------------------------------------------
# Trainer Configuration
# --------------------------------------------------
trainer = dict(
    # Training steps
    max_steps=100000,
    num_train_timesteps=1000,

    # Gradient
    gradient_accumulation_steps=4,
    gradient_clip_norm=1.0,

    # Mixed precision
    mixed_precision="bf16",

    # FSDP
    use_fsdp=True,

    # Warmup
    warmup_steps=1000,
    lr_scheduler_type="cosine",
    min_lr=1e-6,

    # Logging
    log_interval=1,
    logger_type="tensorboard",

    # Checkpointing
    save_interval=500,
    save_path="./checkpoints/dualforce",
    resume_from=None,

    # Modules to train (no video_dit_2!)
    train_modules=["video_dit", "struct_dit", "dual_tower_bridge"],

    # Full fine-tuning (no LoRA)
    use_lora=False,

    # Context Parallel
    enable_cp=True,
)

# --------------------------------------------------
# Model Parameter Summary (estimated)
# --------------------------------------------------
# Video DiT (dim=1536, layers=20):
#   Attention: 4 * 1536^2 * 20 = ~189M
#   FFN: 2 * 1536 * 6144 * 20 = ~378M
#   Other (embedding, head, norms): ~50M
#   Total: ~617M
#
# Struct DiT (same dims):
#   Total: ~617M
#
# Bridge (20 interaction layers):
#   ~200M (cross-attention for both directions)
#
# Frozen modules: Video VAE (~200M), Text Encoder (~3B), HuBERT (~300M)
# Total trainable: ~1.5B
# Total with frozen: ~5B
