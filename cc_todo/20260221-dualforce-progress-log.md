# DualForce Progress Log

> Project: DualForce - 3D-Aware Autoregressive Diffusion for Talking Head Generation
> Started: 2026-02-21
> Last Updated: 2026-02-21 (Session 3)

---

## Version History

| Date | Version | Summary |
|------|---------|---------|
| 2026-02-21 | v0.1 | Initial analysis complete. MOVA codebase analyzed, execution plan created. |
| 2026-02-21 | v0.2 | CRITICAL correction: actual MOVA-360p is dim=5120/40L (not 3072/30L). Created core DualForce code. |
| 2026-02-21 | v0.3 | Dataset class, preprocessing pipeline, causal attention, KV-cache all implemented. Phase 1 architecture ~90% complete. |

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

## Current Phase: Phase 1 - Architecture Implementation (~90% complete)

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
| DualForce training script | Pending | Adapt accelerate_train.py |
| DualForce AR inference pipeline | Pending | Sliding window generation |
| Forward pass verification | Pending | Needs GPU |

---

## Next Immediate Tasks

1. Create DualForce training launch script (adapt accelerate_train.py)
2. Create DualForce AR inference pipeline (sliding window + KV-cache)
3. Forward pass test on GPU (when available)
4. Start HDTF data download

---

*This file is continuously updated as the project progresses.*
