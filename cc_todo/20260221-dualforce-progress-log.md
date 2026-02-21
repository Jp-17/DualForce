# DualForce Progress Log

> Project: DualForce - 3D-Aware Autoregressive Diffusion for Talking Head Generation
> Started: 2026-02-21
> Last Updated: 2026-02-21 (Session 2)

---

## Version History

| Date | Version | Summary |
|------|---------|---------|
| 2026-02-21 | v0.1 | Initial analysis complete. MOVA codebase analyzed, execution plan created. |
| 2026-02-21 | v0.2 | CRITICAL correction: actual MOVA-360p is dim=5120/40L (not 3072/30L). Created core DualForce code. |

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

### Created Code (Phase 1 Architecture)
- [x] `configs/dualforce/dualforce_train_8gpu.py` - Training config with MOVA-Lite parameters
- [x] `mova/diffusion/models/wan_struct_dit.py` - 3D Structure DiT (WanStructModel)
- [x] `mova/diffusion/schedulers/diffusion_forcing.py` - Per-frame noise scheduler
- [x] `mova/diffusion/pipelines/dualforce_train.py` - DualForceTrain pipeline
- [x] Updated `__init__.py` files for model and scheduler registration

### Next Steps
- [ ] Verify MOVA inference with downloaded weights (needs GPU)
- [ ] Start HDTF data download
- [ ] Build data preprocessing pipeline
- [ ] Test forward pass of new architecture on GPU
- [ ] Implement causal temporal attention mask

---

## Current Phase: Phase 0/1 - Environment Setup + Architecture Implementation

### Phase 0 Progress
| Task | Status | Notes |
|------|--------|-------|
| MOVA inference verification | Pending | Needs GPU server |
| Environment setup | Pending | Need to verify deps |
| HDTF download | Pending | Priority 1 |
| CelebV-HQ download | Pending | Priority 2 |
| VFHQ-512 download | Pending | Priority 3 |
| Preprocessing pipeline | Pending | |
| Feature extraction | Pending | |

### Phase 1 Progress (Architecture)
| Task | Status | Notes |
|------|--------|-------|
| MOVA-Lite config | Done | configs/dualforce/dualforce_train_8gpu.py |
| WanStructModel | Done | mova/diffusion/models/wan_struct_dit.py |
| DiffusionForcingScheduler | Done | mova/diffusion/schedulers/diffusion_forcing.py |
| DualForceTrain pipeline | Done | mova/diffusion/pipelines/dualforce_train.py |
| Causal temporal attention | Pending | Need to modify SelfAttention |
| KV-Cache | Pending | New module needed |
| Forward pass verification | Pending | Needs GPU |

---

*This file is continuously updated as the project progresses.*
