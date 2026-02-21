# MOVA Codebase Analysis Report

> Date: 2026-02-21
> Project: DualForce (based on MOVA fork)
> Purpose: Thorough analysis of the MOVA codebase to inform DualForce development

---

## 1. Directory Structure

```
DualForce/ (forked from OpenMOSS/MOVA)
├── mova/
│   ├── diffusion/
│   │   ├── models/
│   │   │   ├── wan_video_dit.py      # Video DiT backbone (WanModel)
│   │   │   ├── wan_audio_dit.py      # Audio DiT backbone (WanAudioModel)
│   │   │   ├── interactionv2.py      # Bridge CrossAttention (DualTowerConditionalBridge)
│   │   │   └── dac_vae.py            # Audio VAE (DAC codec)
│   │   ├── pipelines/
│   │   │   ├── pipeline_mova.py      # Inference pipeline (MOVA class)
│   │   │   └── mova_train.py         # Training pipeline (MOVATrain)
│   │   └── schedulers/
│   │       ├── flow_match.py         # FlowMatchScheduler
│   │       └── flow_match_pair.py    # FlowMatchPairScheduler (independent video/audio sigma)
│   ├── datasets/
│   │   └── video_audio_dataset.py    # Data loading (VideoAudioDataset)
│   ├── engine/trainer/
│   │   ├── accelerate/
│   │   │   ├── accelerate_trainer.py # Main distributed trainer (AccelerateTrainer)
│   │   │   └── lora_utils.py         # LoRA implementation
│   │   └── low_resource/
│   │       └── low_resource_trainer.py # Single-GPU memory-efficient trainer
│   └── distributed/
│       └── functional.py             # Context parallelism utilities
├── configs/training/
│   ├── mova_train_accelerate.py      # 1-GPU Accelerate config
│   ├── mova_train_accelerate_8gpu.py # 8-GPU FSDP config
│   ├── mova_train_low_resource.py    # Low-resource single-GPU config
│   └── accelerate/
│       └── fsdp_8gpu.yaml            # FSDP Accelerate config
├── scripts/
│   ├── inference_single.py           # Inference entry point
│   ├── inference_single_lora.py      # LoRA inference
│   └── training_scripts/
│       ├── accelerate_train.py       # Training entry point
│       └── low_resource_train.py     # Low-resource training entry
└── pyproject.toml                     # Dependencies
```

---

## 2. Model Architecture Detail

### 2.1 Video DiT (WanModel)

**Location:** `mova/diffusion/models/wan_video_dit.py`

**Config:**
```
dim = 3072           # Hidden dimension
ffn_dim = 12288      # FFN hidden (4x)
num_heads = 24       # Attention heads (head_dim=128)
num_layers = 30      # DiT blocks
patch_size = (2,2,2) # 3D patchification (temporal, height, width)
in_dim = 16          # VAE latent channels
out_dim = 16         # Output channels
text_dim = 4096      # UMT5 text embedding dim
freq_dim = 256       # Frequency embedding dim
```

**DiT Block Structure (per block):**
```
SelfAttention(dim=3072, heads=24)
  ├── 3D RoPE (precompute_freqs_cis_3d)
  ├── RMSNorm on Q, K
  ├── FlashAttention (v2/v3, SageAttention, LongContextAttention)
  └── Output projection

CrossAttention(dim=3072, heads=24)
  ├── Q from video tokens, KV from text embeddings
  ├── Optional image input handling (first 257 tokens for DINO/CLIP)
  └── RMSNorm on Q, K

FFN: Linear(3072→12288) → GELU → Linear(12288→3072)

AdaLN Modulation: timestep_emb → MLP → 6 params (shift, scale, gate × 2)
```

**Forward Flow:**
1. Patchify: Conv3d(16, 3072, kernel=(2,2,2), stride=(2,2,2))
2. Add positional info via 3D RoPE
3. Process through 30 DiT blocks
4. Unpatchify: reshape to [B, 16, T/4+1, H/8, W/8]

**Important:** MOVA uses TWO video DiTs:
- `video_dit`: processes high-noise timesteps
- `video_dit_2`: processes low-noise timesteps (after boundary_ratio ~0.9)
- Same architecture, different weights

### 2.2 Audio DiT (WanAudioModel)

**Location:** `mova/diffusion/models/wan_audio_dit.py`

**Config:**
```
dim = 1536           # Half of video (asymmetric)
ffn_dim = 6144       # 4x
num_heads = 12       # Half of video (head_dim=128, same)
num_layers = 30      # Same as video
patch_size = (2,1,1) # Only temporal patching (1D audio)
in_dim = 128         # DAC latent dim
out_dim = 128
text_dim = 4096
```

**Key Differences from Video:**
- 1D Conv1d patchification (vs 3D Conv3d)
- 1D RoPE (vs 3D RoPE)
- Half hidden dim (asymmetric dual tower)
- Different VAE (DAC vs AutoencoderKLWan)

### 2.3 Bridge CrossAttention (DualTowerConditionalBridge)

**Location:** `mova/diffusion/models/interactionv2.py`

**Config:**
```
visual_layers = 30
audio_layers = 30
visual_hidden_dim = 3072
audio_hidden_dim = 1536
audio_fps = 44100.0 / 2048.0  # ~21.5 Hz
head_dim = 128
interaction_strategy = "shallow_focus"  # DEFAULT
```

**Interaction Strategies:**
- `shallow_focus`: First ~1/3 of layers (emphasis on structural layout)
- `distributed`: Every 3rd layer (sparse)
- `progressive`: Dense shallow + sparse deep
- `custom`: Explicit layer indices [0,2,4,6,8,12,16,20]
- `full`: All 30 layers

**ConditionalCrossAttentionBlock:**
```
CrossAttention(Q=primary_tower, KV=conditioning_tower)
├── RMSNorm on Q, K
├── Optional cross-modal RoPE alignment
├── FlashAttention
├── Output projection
└── Optional PerFrameAttentionPooling
```

**Aligned-RoPE for Cross-Modal Sync:**
```python
def build_aligned_freqs(video_fps, grid_size, audio_steps, ...):
    # Aligns video frames (24fps) to audio steps (21.5Hz)
    # Scale factor: audio_fps / (video_fps / 4.0) accounting for VAE stride
    # Returns: (visual_cos_sin, audio_cos_sin) for RoPE application
```

### 2.4 Audio VAE (DAC)

**Location:** `mova/diffusion/models/dac_vae.py`

```
Encoder: Conv1d chain, stride [2,4,8,8] = total stride 256
Latent dim: 128
Decoder: Transposed Conv1d, decoder_dim=1536
VQ-VAE: 9 codebooks × 1024 entries × 8 dims
Sample rate: 44100 Hz
Hop length: 2048 samples
```

### 2.5 Video VAE

```
Type: AutoencoderKLWan (from diffusers)
z_dim: 16 channels
Temporal stride: 4
Spatial stride: 8
Latent normalization: (latent - mean) / std using config stats
```

---

## 3. Training Infrastructure

### 3.1 Training Loop (AccelerateTrainer)

**Location:** `mova/engine/trainer/accelerate/accelerate_trainer.py`

**Main Loop:**
```python
while global_step < max_steps:
    batch = next(dataloader)  # video, audio, caption, first_frame
    with accelerator.accumulate():
        loss_dict = model.training_step(batch)  # forward pass
        accelerator.backward(loss_dict['loss'])  # backward
        clip_grad_norm(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()
    if global_step % save_interval == 0:
        save_checkpoint(global_step)
```

**Key Features:**
- Gradient accumulation via `accelerator.accumulate()` context
- BF16 mixed precision via autocast
- FSDP distributed training with sharded optimizer
- Gradient checkpointing (recompute activations)
- FP8 CPU offloading for frozen modules (50% memory saving)

### 3.2 Noise Scheduling

**Location:** `mova/diffusion/schedulers/flow_match.py`

```python
class FlowMatchScheduler:
    sigma_max = 1.0
    sigma_min = 0.003 / 1.002 ≈ 0.00299
    shift = 3.0

    def add_noise(original, noise, timestep):
        sigma = sigmas[timestep_id]
        return (1 - sigma) * original + sigma * noise

    def training_target(sample, noise, timestep):
        return noise - sample  # v-prediction
```

**FlowMatchPairScheduler** (flow_match_pair.py):
- Supports independent visual and audio timesteps
- `dual_sigma_shift`: different shift factors per modality
- Returns paired timesteps for dual-tower training

### 3.3 Training Forward Pass

**Location:** `mova/diffusion/pipelines/mova_train.py`

```python
def training_step(batch):
    # 1. Encode video to latents via VAE (frozen)
    video_latents = video_vae.encode(batch['video'])
    # 2. Encode audio to latents via DAC (frozen)
    audio_latents = audio_vae.encode(batch['audio'])
    # 3. Sample timesteps (optionally independent per modality)
    timestep, audio_timestep = sample_timestep_pair(config)
    # 4. Add noise via flow matching
    noisy_video = scheduler.add_noise(video_latents, video_noise, timestep)
    noisy_audio = scheduler.add_noise(audio_latents, audio_noise, audio_timestep)
    # 5. Forward through dual tower + bridge
    video_pred, audio_pred = inference_single_step(
        noisy_video, noisy_audio, text_emb, timestep, audio_timestep)
    # 6. Compute loss
    video_target = video_noise - video_latents
    audio_target = audio_noise - audio_latents
    loss = MSE(video_pred, video_target) + MSE(audio_pred, audio_target)
    return {'loss': loss, 'video_loss': video_loss, 'audio_loss': audio_loss}
```

### 3.4 Data Pipeline

**Location:** `mova/datasets/video_audio_dataset.py`

```
Input format:
  data_root/
  ├── metadata.json  # [{"video_path": "xxx.mp4", "caption": "..."}]
  └── videos/

Processing:
  Video: load frames -> center crop -> resize (352×640) -> normalize [-1,1]
  Audio: decode from MP4 -> resample 48kHz -> mono -> pad/truncate
  First frame: separate encoding for conditioning
```

### 3.5 LoRA Implementation

**Location:** `mova/engine/trainer/accelerate/lora_utils.py`

```python
LoRA config:
  rank = 16
  alpha = 16.0
  target_modules = ["q", "k", "v", "o", "to_q", "to_k", "to_v", "proj"]

LoRALinear: h = Wx + (BA)x * (alpha/rank)
  A: Linear(in_features, rank, bias=False)
  B: Linear(rank, out_features, bias=False)
```

### 3.6 Checkpoint Structure

```
checkpoints/step-N/
├── video_dit/pytorch_model.bin
├── video_dit_2/pytorch_model.bin
├── audio_dit/pytorch_model.bin
├── dual_tower_bridge/pytorch_model.bin
├── trainer_state.pt  # {global_step, epoch}
└── accelerator/      # optimizer, scheduler, scaler
```

### 3.7 FSDP Configuration

```python
# Frozen modules (excluded from FSDP):
ignored = ['text_encoder', 'video_vae', 'audio_vae', 'prompter']

# FSDP wrapping:
FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True),
    reshard_after_forward=True
)
```

---

## 4. Inference Pipeline

### 4.1 Denoising Loop (pipeline_mova.py)

```python
# Two-stage denoising:
for i, (visual_t, audio_t) in enumerate(paired_timesteps):
    if visual_t > boundary_threshold:
        active_dit = video_dit      # Stage 1: high noise
    else:
        active_dit = video_dit_2    # Stage 2: low noise (refinement)

    noise_pred = inference_single_step(
        latents, audio_latents, context, visual_t, audio_t, video_fps)

    latents = scheduler.step(noise_pred, ...)
```

### 4.2 Reference Image Processing

```python
# First frame conditioning:
video_condition = torch.cat([ref_image, zero_padding], dim=2)
latent_condition = normalize(vae.encode(video_condition))
mask = torch.ones(B, 1, T, H, W)
mask[:, :, 1:] = 0  # Only first frame is real
latent_input = torch.cat([latents, mask, latent_condition], dim=1)
```

### 4.3 Text Encoding

```python
# UMT5 encoder:
text_inputs = tokenizer(prompt, max_length=512, truncation=True)
prompt_embeds = text_encoder(input_ids).last_hidden_state  # [B, 512, 4096]
```

---

## 5. Key Dependencies

```
torch, torchvision
diffusers (VAE, schedulers)
transformers<5 (T5 encoder)
mmengine (registry)
yunchang (long-context attention)
einops
descript-audiotools
accelerate
ftfy

Training extras: bitsandbytes, torchcodec
```

---

## 6. Implications for DualForce

### What's Directly Reusable
- Training loop (AccelerateTrainer, FSDP, gradient checkpointing, checkpoint mgmt)
- Data loading framework (extend VideoAudioDataset)
- FlowMatchPairScheduler (independent sigma per modality)
- DualTowerConditionalBridge with shallow_focus (matches our shallow fusion design)
- Video VAE (frozen, for latent encoding/decoding)
- RoPE implementation (Aligned-RoPE for cross-modal sync)
- All distributed training utilities

### What Needs Modification
- WanModel: shrink config (30->20 layers, 3072->1536 dim), add causal mask, add KV-cache
- WanAudioModel: replace with Wan3DStructModel (keep dim=1536 for bridge compatibility)
- Bridge: rename audio->struct, simplify aligned frequencies (same FPS)
- Scheduler: extend to per-frame sigma (currently per-sample)
- Training pipeline: drop video_dit_2, add per-frame noise, add multi-modal losses
- Inference: autoregressive loop with KV-cache instead of full-sequence denoising

### What Must Be Built From Scratch
- Implicit3DEncoder (LivePortrait wrapper + projector)
- StructTokenProjector
- Audio conditioning path (HuBERT + cross-attention)
- DualAdaLNZero (separate sigma_v/sigma_s modulation)
- MultiModalKVCache
- Lip-sync contrastive loss
- FLAME alignment loss
- Full evaluation pipeline (FVD, FID, ACD, Sync-C/D)
- Data preprocessing scripts
