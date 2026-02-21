# DualForce Progress Log

> Project: DualForce - 3D-Aware Autoregressive Diffusion for Talking Head Generation
> Started: 2026-02-21
> Last Updated: 2026-02-21 (Session 8)

---

## Version History

| Date | Version | Summary |
|------|---------|---------|
| 2026-02-21 | v0.1 | Initial analysis complete. MOVA codebase analyzed, execution plan created. |
| 2026-02-21 | v0.2 | CRITICAL correction: actual MOVA-360p is dim=5120/40L (not 3072/30L). Created core DualForce code. |
| 2026-02-21 | v0.3 | Dataset class, preprocessing pipeline, causal attention, KV-cache all implemented. Phase 1 architecture ~90% complete. |
| 2026-02-21 | v0.4 | Training/inference scripts, factory function, trainer generalization, FSDP config. Phase 1 CODE COMPLETE. |
| 2026-02-21 | v0.5 | Phase 3 audio conditioning: AudioConditioningModule, DualAdaLNZero, gated cross-attention. |
| 2026-02-21 | v0.6 | Evaluation pipeline (5 metrics), inference rewrite with dual-tower+CFG+audio. |

---

## 2026-02-21 - Day 1: Project Kickoff & Analysis

### Completed
- [x] Read DualForce research proposal (v3.4)
- [x] Read dataset download guide
- [x] Read MOVA README
- [x] Thorough exploration of MOVA codebase:
  - Model architecture (WanModel, WanAudioModel, DualTowerConditionalBridge)
  - Training infrastructure (AccelerateTrainer, FSDP, data pipeline)
  - Inference pipeline, RoPE, VAE, schedulers
- [x] Identified 5 discrepancies between proposal and actual code
- [x] Created MOVA analysis document: `cc_todo/20260221-mova-codebase-analysis.md`
- [x] Created execution plan: `cc_todo/20260221-dualforce-execution-plan.md`
- [x] Confirmed key decisions with user:
  - 8x A100 80GB compute
  - Single DiT (drop video_dit_2)
  - Shrink & retrain from scratch
  - Quality-first, no deadline
  - MOVA weights already downloaded

### Key Findings
1. **CRITICAL CORRECTION:** Actual MOVA-360p checkpoint has dim=5120, 40 layers, NOT 3072/30 as initially reported from code defaults
2. Bridge uses `interaction_strategy="full"` (all layers), NOT `shallow_focus`
3. Video DiT patch_size=(1,2,2) not (2,2,2) -- no temporal downsampling
4. Audio VAE: 48kHz/960 hop = 50Hz tokens (not 44.1kHz/2048 = 21.5Hz)
5. in_dim=36: 16(VAE) + 4(mask) + 16(first_frame_condition)
6. Two video DiTs (video_dit + video_dit_2) - decided to drop dit_2

### Issues Encountered
- Initial code exploration read default constructor values instead of actual checkpoint configs. **Always verify from checkpoint configs, not code defaults!**

---

## 2026-02-21 - Session 2: Core Architecture Implementation

### Phase 1 Architecture Code Created
- [x] `configs/dualforce/dualforce_train_8gpu.py` - Training config with MOVA-Lite parameters
- [x] `mova/diffusion/models/wan_struct_dit.py` - 3D Structure DiT (WanStructModel)
- [x] `mova/diffusion/schedulers/diffusion_forcing.py` - Per-frame noise scheduler
- [x] `mova/diffusion/pipelines/dualforce_train.py` - DualForceTrain pipeline
- [x] Updated `__init__.py` files for model and scheduler registration

**Git commit:** `f505ce6` - "Add DualForce core architecture: struct DiT, DF scheduler, training pipeline"

---

## 2026-02-21 - Session 3: Dataset, Preprocessing, Causal Attention, KV-Cache

### Completed

#### DualForce Dataset
- [x] `mova/datasets/dualforce_dataset.py` - Multi-modal dataset class
  - Loads pre-extracted .safetensors: video_latents, struct_latents, audio_features, flame_params, first_frame
  - Random temporal cropping with proper cross-modal alignment
  - Custom collate function (dualforce_collate_fn)
- [x] Registered in `mova/datasets/__init__.py`

#### Data Preprocessing Pipeline
- [x] `scripts/preprocess/01_face_detect_crop.py` - Face detection (RetinaFace/OpenCV), stable crop, ffmpeg resize to 512x512 @25fps
- [x] `scripts/preprocess/02_quality_filter.py` - Blur/resolution/duration/multi-face filtering
- [x] `scripts/preprocess/03_extract_video_latents.py` - MOVA Video VAE encoding
- [x] `scripts/preprocess/04_extract_struct_latents.py` - LivePortrait motion extraction (+ MediaPipe fallback)
- [x] `scripts/preprocess/05_extract_flame_params.py` - EMOCA FLAME parameter extraction (+ MediaPipe fallback)
- [x] `scripts/preprocess/06_extract_audio_features.py` - HuBERT-Large feature extraction + first_frame latent
- [x] `scripts/preprocess/07_build_metadata.py` - Feature validation, metadata.json builder
- [x] `scripts/preprocess/run_pipeline.sh` - End-to-end pipeline runner

#### Causal Temporal Attention
- [x] Added `_build_block_causal_mask()` - Block-causal attention mask (frame t attends only to frames <= t)
- [x] Added `flash_attention_causal()` - Causal attention via `scaled_dot_product_attention` with block mask
- [x] Modified `AttentionModule`, `SelfAttention`, `DiTBlock` - Optional `causal_grid_size` parameter
- [x] Added `causal_temporal` config param to `WanModel.__init__()`
- [x] Updated `WanModel.forward()`, `WanStructModel.forward()`, `DualForceTrain.forward_dual_tower()`

#### KV-Cache for Autoregressive Inference
- [x] `mova/diffusion/models/kv_cache.py` - LayerKVCache, MultiModalKVCache, CachedSelfAttention
- [x] Registered in `mova/diffusion/models/__init__.py`

**Git commit:** `7567c46` - "Add dataset, preprocessing pipeline, causal attention, and KV-cache"

---

## Current Phase: Phase 1 - Architecture Implementation (COMPLETE, pending GPU verification)

### Phase 0 Progress
| Task | Status | Notes |
|------|--------|-------|
| MOVA inference verification | Pending | Needs GPU server |
| Environment setup | Pending | Need to verify deps |
| HDTF download | Pending | Priority 1 |
| CelebV-HQ download | Pending | Priority 2 |
| VFHQ-512 download | Pending | Priority 3 |
| Preprocessing pipeline | **Done** | 7-step pipeline + runner script |
| Feature extraction scripts | **Done** | All 5 modalities covered |

### Phase 1 Progress (Architecture)
| Task | Status | Notes |
|------|--------|-------|
| MOVA-Lite config | **Done** | configs/dualforce/dualforce_train_8gpu.py |
| WanStructModel | **Done** | mova/diffusion/models/wan_struct_dit.py |
| DiffusionForcingScheduler | **Done** | mova/diffusion/schedulers/diffusion_forcing.py |
| DualForceTrain pipeline | **Done** | mova/diffusion/pipelines/dualforce_train.py |
| DualForceDataset | **Done** | mova/datasets/dualforce_dataset.py |
| Causal temporal attention | **Done** | Block-causal mask in SelfAttention |
| KV-Cache | **Done** | MultiModalKVCache + CachedSelfAttention |
| DualForceTrain_from_pretrained | **Done** | Factory function for model building |
| DualForceTrain.forward | **Done** | Bridges trainer call convention |
| DualForce training script | **Done** | scripts/training_scripts/dualforce_train.py |
| AccelerateTrainer generalization | **Done** | Batch-key agnostic, DualForce metrics |
| DualForceInference pipeline | **Done** | Sliding window AR generation |
| Inference script | **Done** | scripts/dualforce_inference.py |
| Forward pass verification | **Pending** | Needs GPU |

---

## 2026-02-21 - Session 4: Training/Inference Scripts, Factory, Trainer Update

### Completed

#### DualForceTrain_from_pretrained (Factory Function)
- [x] Registered factory in `DIFFUSION_PIPELINES` registry
- [x] Two modes: build from scratch (config dicts) or load from checkpoint
- [x] Accepts `video_dit_config`, `struct_dit_config`, `bridge_config` for from-scratch builds
- [x] Requires `vae_path` and `text_encoder_path` for frozen components

#### DualForceTrain.forward()
- [x] Added `forward()` method that dispatches batch kwargs to `training_step()`
- [x] Handles both DualForce dataset keys (`video_latents`, `struct_latents`) and MOVA keys (`video`, `audio`)

#### AccelerateTrainer Generalization
- [x] Training loop now passes full batch dict as `**kwargs` instead of hardcoded `video=`, `audio=`
- [x] Logs DualForce-specific metrics: `struct_loss`, `flame_loss`, `lip_sync_loss`, `sigma_v_mean`, `sigma_s_mean`
- [x] Backward-compatible with MOVA (uses `.get()` with fallbacks for `audio_loss`)

#### Training Config Update
- [x] `dualforce_train_8gpu.py` now passes component configs through `diffusion_pipeline` dict
- [x] Includes frozen model paths (vae_path, text_encoder_path) pointing to MOVA-360p checkpoints

#### DualForceInference Pipeline
- [x] `pipeline_dualforce.py`: Sliding window autoregressive denoising
  - Configurable `window_size` and `window_stride`
  - Per-frame sigma schedule from `sigma_max` → 0
  - First frame conditioning via VAE encoding
  - Optional struct/audio conditioning inputs
  - CFG guidance support
  - VAE decoding to video output
- [x] Registered `DualForceInference_from_pretrained` in `DIFFUSION_PIPELINES`

#### Inference Script
- [x] `scripts/dualforce_inference.py`: CLI tool for video generation
  - Takes reference image, prompt, optional struct/audio features
  - Saves output as MP4

**Git commit:** `222b2ed` - "Add training/inference scripts, factory function, and trainer generalization"

---

## 2026-02-21 - Session 5: FSDP Config, Launch Script, Plan Update

### Completed

#### Accelerate FSDP Configuration
- [x] `configs/dualforce/accelerate/fsdp_8gpu.yaml` - Full FSDP config for 8-GPU training
  - FSDP wrapping: DiTBlock + ConditionalCrossAttentionBlock
  - Context Parallel size=2, DP shard size=4
  - bf16 mixed precision, CPU parameter offload
  - Gradient checkpointing enabled

#### Training Launch Script
- [x] `scripts/training_scripts/dualforce_train_8gpu.sh` - Bash launcher
  - Sets PYTHONPATH, CUDA_VISIBLE_DEVICES, TOKENIZERS_PARALLELISM
  - Uses `accelerate launch` with FSDP config
  - Supports `--cfg-options` for config overrides

#### Execution Plan Update
- [x] Updated `cc_todo/20260221-dualforce-execution-plan.md`
  - Marked Phase 0.3 preprocessing scripts as complete
  - Marked all Phase 1 code items as complete
  - Added new items for DualForce training script, inference pipeline, dataset, factory function

### Phase 1 Status: CODE COMPLETE
All Phase 1 code has been written. Remaining items require GPU access:
- Forward pass verification (random input → correct output shape)
- KV-cache consistency verification (cached vs non-cached output match)

---

## 2026-02-21 - Session 6: Code Review, Bugfixes, Verification & Download Scripts

### Code Review
- [x] Thorough review of all DualForce code for correctness issues
- [x] Identified and fixed 3 critical bugs:
  1. Hardcoded `"cuda"` in `torch.autocast` — now detects device type dynamically
  2. Loss placeholder tensors created in wrong dtype (float32 vs model's bfloat16) — now matches video_loss dtype
  3. Missing error handling for first_frame file in dataset — now falls back to zeros
- [x] Cleaned up config: removed misleading `patch_size`, `require_clip_embedding`, `require_vae_embedding` from struct_dit config
- [x] Removed unused `ref_feature_key`, `num_frames`, `height`, `width` from data config

### GPU Verification Script
- [x] `scripts/verify_dualforce.py` — 6-test suite:
  1. Video DiT forward pass (random input → correct shape)
  2. Struct DiT forward pass
  3. Dual tower with bridge cross-attention
  4. KV-cache consistency
  5. Full training pipeline (with VAE + text encoder)
  6. Memory estimation for full 20-layer model
- [x] Supports `--cpu-only`, `--skip-training`, `--device` options

### MOVA Inference Verification Script
- [x] `scripts/verify_mova_inference.py` — Phase 0.1 verification
  - Checks all checkpoint files exist
  - Loads MOVA pipeline and reports component param counts
  - Optional full inference with reference image
  - Supports `--load-only` mode for quick check
  - Supports CPU offload for single-GPU testing

### HDTF Download Script
- [x] `scripts/download/download_hdtf.py` — Phase 0.2 data download
  - Parses HDTF metadata (URL files, annotation files)
  - Downloads videos via yt-dlp with parallel workers
  - Clips videos according to annotation timestamps
  - Normalizes to 512x512 @ 25fps
  - Generates metadata.json for the preprocessing pipeline

---

## 2026-02-21 - Session 7: Phase 3 Audio Conditioning & Per-Frame AdaLN

### Phase 3.4 — Audio Conditioning Path (COMPLETE)
- [x] `mova/diffusion/models/audio_conditioning.py` — New module with 5 classes:
  1. **AudioProjector**: LayerNorm → Linear(1024, 1536) → GELU → Linear → projects HuBERT features
  2. **AudioCondCrossAttention**: One-directional cross-attention with gated residual (`sigmoid(gate) * attn_out`, gate init=0)
  3. **AudioConditioningModule**: Container managing per-layer audio→video and audio→struct cross-attention. Includes `align_audio_to_tokens()` for temporal resampling via F.interpolate. Configurable `num_layers` (default 6) for shallow-only conditioning.
  4. **DualAdaLNZero**: Per-frame sigma → sinusoidal embedding → MLP → 6*dim modulation vectors. Handles both 3D video grids (f,h,w) and 1D struct grids (f,). Output: [B, L, 6, dim]
  5. **PerFrameDiTBlockWrapper**: Utility to reshape per-frame t_mod to DiTBlock-compatible format

### Integration into DualForceTrain
- [x] Updated `__init__()`: accepts `audio_conditioning`, `video_adaln`, `struct_adaln` modules
- [x] Updated `forward_dual_tower()`: applies audio conditioning per layer in shallow blocks
- [x] Updated `training_step()`: DualAdaLNZero overrides global t_mod with per-frame modulation
- [x] Updated `freeze_for_training()`: includes new modules in default trainable set
- [x] Updated factory function: accepts `audio_conditioning_config`, builds all new modules

### Config Update
- [x] `audio_conditioning_config` added to `diffusion_pipeline` in training config
- [x] `train_modules` expanded to include `audio_conditioning`, `video_adaln`, `struct_adaln`
- [x] Updated parameter summary with new module estimates (~43M total new params)

### Design Decisions
- **Gated residual with near-zero init**: Audio conditioning starts with zero influence, progressively learned. Prevents destabilizing early training.
- **Shallow-only conditioning (6 layers)**: Audio influences visual/structural features in early layers only; deep layers specialize without audio interference.
- **Temporal alignment via interpolation**: HuBERT operates at different temporal resolution than video/struct tokens. F.interpolate aligns them.

**Git commit:** `9b5bcd7` — "Add audio conditioning module and per-frame AdaLN (Phase 3)"

---

## 2026-02-21 - Session 8: Evaluation Pipeline & Inference Rewrite

### Phase 5.1 — Evaluation Pipeline (COMPLETE)
All 5 metrics implemented with individual scripts and an orchestrator:

- [x] `scripts/eval/compute_fvd.py` — FVD using R3D-18 (I3D-like) video features
  - VideoDataset with temporal sampling, trilinear resize, ImageNet normalization
  - Fréchet distance computation with scipy.linalg.sqrtm
- [x] `scripts/eval/compute_fid.py` — FID using Inception-v3 frame features
  - VideoFrameDataset for per-frame extraction across all videos
  - Fréchet distance in 2048-dim Inception feature space
- [x] `scripts/eval/compute_identity.py` — ACD/CSIM using ArcFace (insightface)
  - Paired reference-image + generated-video evaluation
  - ACD = mean L2 distance, CSIM = mean cosine similarity
  - Fallback to ResNet50 if insightface not installed
- [x] `scripts/eval/compute_sync.py` — Sync-C/Sync-D using SyncNet
  - SyncNetModel with visual (mouth crop) + audio (MFCC) encoders
  - Sliding window evaluation with 5-frame windows
  - Mouth crop extraction via OpenCV Haar cascade
- [x] `scripts/eval/compute_pose.py` — APD using MediaPipe face mesh
  - Per-frame head pose (yaw, pitch, roll) via solvePnP
  - Per-axis error breakdown
  - Fallback to OpenCV cascade if MediaPipe unavailable
- [x] `scripts/eval/run_eval.py` — Orchestrator
  - Two modes: end-to-end (generate + evaluate) or metrics-only
  - Selective metric computation via `--metrics fvd fid identity sync pose`
  - JSON report output with all results

**Git commit:** `7056b56` — "Add evaluation pipeline: FVD, FID, ACD/CSIM, Sync-C/D, APD (Phase 5)"

### Inference Pipeline Rewrite (COMPLETE)
Major overhaul of `mova/diffusion/pipelines/pipeline_dualforce.py`:

- [x] Full dual-tower forward with bridge cross-attention (was a TODO placeholder)
- [x] Classifier-free guidance (CFG) with negative prompt support
- [x] Audio conditioning integration in shallow layers during inference
- [x] Per-frame DualAdaLNZero modulation during inference
- [x] Fixed hardcoded `"cuda"` in `torch.autocast` (same bug as training pipeline)
- [x] Proper patchify/unpatchify and tokenize/detokenize flow matching DiT conventions
- [x] Struct latent joint generation (when not provided as input, generated alongside video)
- [x] Cross-RoPE alignment for bridge during inference

**Git commit:** `5a9d66b` — "Rewrite inference pipeline with dual-tower, CFG, audio conditioning"

---

## Current Phase Status

### Phase 0 Progress
| Task | Status | Notes |
|------|--------|-------|
| MOVA inference verification | Pending | Needs GPU server |
| Environment setup | Pending | Script ready: `scripts/setup_env.sh` |
| HDTF download | Pending | Script ready: `scripts/download/download_hdtf.py` |
| CelebV-HQ download | Pending | Script ready: `scripts/download/download_celebvhq.py` |
| VFHQ-512 download | Pending | Priority 3 |
| Preprocessing pipeline | **Done** | 7-step pipeline + runner script |

### Phase 1 Progress (Architecture) — COMPLETE
All code written. Forward pass verification pending GPU.

### Phase 3 Progress (Multi-Modal Architecture) — COMPLETE
- AudioConditioningModule, DualAdaLNZero, gated cross-attention all implemented
- Integrated into both training and inference pipelines

### Phase 5 Progress (Evaluation) — COMPLETE
All 5 metrics implemented: FVD, FID, ACD/CSIM, Sync-C/D, APD

## Next Immediate Tasks (requires GPU)

1. **Forward pass verification** - Run DualForceTrain with random data, verify output shapes and loss computation
2. **MOVA inference test** - Verify base MOVA-360p weights produce valid output
3. **HDTF data download** - Start downloading from YouTube (time-sensitive, links may expire)
4. **Run preprocessing on test clips** - Validate full pipeline end-to-end on 5-10 clips
5. **Begin Phase 2** - Causal video pretraining once data is ready

## Git Commit History

| Commit | Description |
|--------|-------------|
| `5a9d66b` | Rewrite inference pipeline with dual-tower, CFG, audio conditioning |
| `7056b56` | Add evaluation pipeline: FVD, FID, ACD/CSIM, Sync-C/D, APD (Phase 5) |
| `6f502db` | Update progress log: Phase 3 audio conditioning complete |
| `9b5bcd7` | Add audio conditioning module and per-frame AdaLN (Phase 3) |
| `5f404cf` | Add CelebV-HQ download script and environment setup |
| `98aefc6` | Fix bugs, add verification scripts and HDTF download tool |
| `b4d5899` | Add FSDP config, launch script; finalize Phase 1 code completion |
| `222b2ed` | Add training/inference scripts, factory function, trainer generalization |
| `7567c46` | Add dataset, preprocessing pipeline, causal attention, KV-cache |
| `f505ce6` | Add DualForce core architecture: struct DiT, DF scheduler, training pipeline |
| `a14eb9a` | Add project analysis and execution plan docs |

### New Files Created (Session 5)
| File | Purpose |
|------|---------|
| `configs/dualforce/accelerate/fsdp_8gpu.yaml` | FSDP config for 8-GPU distributed training |
| `scripts/training_scripts/dualforce_train_8gpu.sh` | Bash launch script for accelerate |
| `scripts/verify_dualforce.py` | GPU verification script (6 tests) |
| `scripts/verify_mova_inference.py` | MOVA inference verification |
| `scripts/download/download_hdtf.py` | HDTF dataset download + clip script |

## Code File Summary

### New Files Created
| File | Purpose |
|------|---------|
| `configs/dualforce/dualforce_train_8gpu.py` | Training config |
| `mova/diffusion/models/wan_struct_dit.py` | 3D Structure DiT |
| `mova/diffusion/models/kv_cache.py` | KV-cache for AR inference |
| `mova/diffusion/models/audio_conditioning.py` | Audio cond + DualAdaLNZero |
| `mova/diffusion/schedulers/diffusion_forcing.py` | Per-frame noise scheduler |
| `mova/diffusion/pipelines/dualforce_train.py` | Training pipeline |
| `mova/diffusion/pipelines/pipeline_dualforce.py` | Inference pipeline |
| `mova/datasets/dualforce_dataset.py` | Multi-modal dataset |
| `scripts/training_scripts/dualforce_train.py` | Training launch script |
| `scripts/dualforce_inference.py` | Inference CLI script |
| `scripts/eval/compute_fvd.py` | FVD metric (R3D-18 features) |
| `scripts/eval/compute_fid.py` | FID metric (Inception-v3 features) |
| `scripts/eval/compute_identity.py` | ACD/CSIM identity metrics (ArcFace) |
| `scripts/eval/compute_sync.py` | Sync-C/Sync-D lip-sync metrics (SyncNet) |
| `scripts/eval/compute_pose.py` | APD head pose metric (MediaPipe) |
| `scripts/eval/run_eval.py` | Evaluation orchestrator |
| `scripts/preprocess/01-07_*.py` | Preprocessing pipeline (7 scripts) |
| `scripts/preprocess/run_pipeline.sh` | Pipeline runner |

### Modified Files
| File | Change |
|------|--------|
| `mova/diffusion/models/wan_video_dit.py` | Block-causal mask, `causal_temporal` config |
| `mova/diffusion/models/__init__.py` | Register WanStructModel, KV-cache |
| `mova/diffusion/schedulers/__init__.py` | Register DiffusionForcingScheduler |
| `mova/diffusion/pipelines/__init__.py` | Register DualForceTrain, DualForceInference |
| `mova/datasets/__init__.py` | Register DualForceDataset |
| `mova/engine/trainer/accelerate/accelerate_trainer.py` | Batch-key agnostic training loop |

---

*This file is continuously updated as the project progresses.*
