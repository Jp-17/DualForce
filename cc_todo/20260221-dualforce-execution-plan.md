# DualForce Project Execution Plan

> Date: 2026-02-21
> Version: v1.0
> Project: DualForce - 3D-Aware Autoregressive Diffusion for Talking Head Generation
> Base: MOVA (OpenMOSS) fork

---

## 1. Project Overview

DualForce combines three innovations atop a scaled-down MOVA backbone ("MOVA-Lite"):
1. **Diffusion Forcing**: Per-token independent noise levels for native autoregressive generation (no distillation)
2. **Implicit 3D Structure Latents**: LivePortrait motion features as parallel modality for structural consistency
3. **Shallow Fusion Architecture**: Bidirectional video<->3D attention in early layers, independent processing in deep layers

**Target**: Research publication at NeurIPS 2026 / CVPR 2027 / ECCV 2026

---

## 2. Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Compute | 8x A100 80GB | Full setup for distributed training |
| Video DiT | Single DiT (drop video_dit_2) | Simplifies architecture, halves model complexity |
| Model size | Shrink & retrain (~2B) | Clean architecture, full control |
| Layer count | 30->20 layers | MOVA actual is 30 (not 48 as proposal stated) |
| Hidden dim | 3072->1536 (video), keep 1536 (3D) | Bridge compatibility |
| Timeline | Quality-first, no fixed deadline | Thorough ablations and polish |
| MOVA weights | Already downloaded | Can verify inference immediately |

---

## 3. Proposal Corrections

| Proposal Statement | Actual Finding (from checkpoint configs) | Correction |
|---|---|---|
| "Wan 2.2 14B, 48 layers, MoE" | WanModel has **40 layers, dim=5120, no MoE** | Shrink 40->20 layers, 5120->1536 dim |
| Not mentioned | MOVA has video_dit + video_dit_2 (two-stage) | Drop video_dit_2, use single DiT |
| "Need to build Diffusion Forcing from scratch" | FlowMatchPairScheduler already supports independent sigma | Extend to per-frame (not just per-sample) |
| "Need new shallow fusion mechanism" | Bridge uses `interaction_strategy="full"` (all layers!) | Change to shallow_focus for DualForce |
| "Audio tower → 3D stream" | Bridge projects 5120<->1536 | Keep 3D dim=1536 for bridge compatibility |
| Assumed patch_size=(2,2,2) | Actual patch_size=(1,2,2) - no temporal downsampling | Keep (1,2,2) to match VAE output |
| Assumed audio at 44.1kHz/2048 hop | Actual: 48kHz/960 hop = 50Hz audio tokens | Audio token count is ~2.3x more than expected |
| "in_dim=16" | Actual in_dim=36 = 16(VAE) + 4(mask) + 16(first_frame) | Keep in_dim=36 for video_dit |

---

## 4. Architecture Mapping

### MOVA → DualForce

```
MOVA (actual from checkpoint):        DualForce:
──────────────────────────            ─────────
video_dit (5120, 40L, patch(1,2,2))  → video_dit (1536, 20L, patch(1,2,2), causal+KV-cache)
video_dit_2 (5120, 40L)             → [REMOVED]
audio_dit (1536, 30L)               → struct_dit (1536, 20L, 3D stream)
Bridge (full, all layers)           → Bridge (shallow_focus, video<->3D)
UMT5 text encoder                   → [KEEP, frozen]
Video VAE (z=16, stride_t=4, s=8)   → [KEEP, frozen]
Audio VAE (DAC, 128d, 48kHz, 960hp) → [REMOVED, replaced by HuBERT conditioning]
                                      + HuBERT (frozen) + audio cross-attn [NEW]
                                      + DualAdaLNZero (sigma_v, sigma_s) [NEW]
                                      + MultiModalKVCache [NEW]
                                      + Lip-sync loss, FLAME loss [NEW]
```

**Note on in_dim:** Video DiT input = concat(noisy_latent[16ch], mask[4ch], cond_latent[16ch]) = 36ch total.
For DualForce we keep in_dim=36 for the video backbone.

### Key Files to Modify

| File | Change |
|------|--------|
| `mova/diffusion/models/wan_video_dit.py` | Shrink config, causal mask, KV-cache |
| `mova/diffusion/models/wan_audio_dit.py` | Fork → `wan_struct_dit.py` |
| `mova/diffusion/models/interactionv2.py` | Adapt bridge for video<->3D |
| `mova/diffusion/pipelines/mova_train.py` | DF noise, multi-modal loss, drop dit_2 |
| `mova/diffusion/pipelines/pipeline_mova.py` | AR inference, KV-cache, drop dit_2 |
| `mova/diffusion/schedulers/flow_match.py` | Per-frame sigma |
| `mova/datasets/video_audio_dataset.py` | 3D/FLAME/audio features |
| `mova/engine/trainer/accelerate/accelerate_trainer.py` | New loss terms |

---

## 5. Execution Plan (6 Phases, ~12 weeks)

### Phase 0: Environment & Data (Week 1-2)

**0.1 Environment**
- [ ] Verify MOVA inference with downloaded weights
- [ ] Install additional deps: LivePortrait, EMOCA, HuBERT, RetinaFace
- [ ] Create DualForce conda environment

**0.2 Data Download**
- [ ] HDTF (~16h, 362 clips) - Priority 1
- [ ] CelebV-HQ (~65h, 35K clips) - Priority 2
- [ ] VFHQ-512 (~50h, 15K clips) - Priority 3

**0.3 Data Preprocessing**
- [x] Build preprocessing scripts (face detect, crop, filter, FPS normalize) — `scripts/preprocess/01-07`
- [ ] Extract video_latents (MOVA Video VAE) — script ready: `03_extract_video_latents.py`
- [ ] Extract struct_latents (LivePortrait) — script ready: `04_extract_struct_latents.py`
- [ ] Extract flame_params (EMOCA) — script ready: `05_extract_flame_params.py`
- [ ] Extract audio_features (HuBERT-Large) — script ready: `06_extract_audio_features.py`
- [ ] Extract ref_features (DINOv2/CLIP)
- [x] Store as per-clip .safetensors — format defined in `DualForceDataset`

**Milestone:** 10 clips spot-checked, all 5 feature types correct

### Phase 1: MOVA-Lite Backbone (Week 2-3) — ✅ CODE COMPLETE

- [x] Create DualForce model config (dim=1536, layers=20, heads=12) — `dualforce_train_8gpu.py`
- [x] Remove video_dit_2 from pipeline — `DualForceTrain` has no video_dit_2
- [ ] Verify forward pass (random input → correct output shape) — needs GPU
- [x] Convert training from LoRA to full FT — `use_lora=False` in config
- [x] Implement causal temporal attention mask — `_build_block_causal_mask` in `wan_video_dit.py`
- [x] Implement MultiModalKVCache — `mova/diffusion/models/kv_cache.py`
- [ ] Verify KV-cache matches non-cached output — needs GPU
- [x] DualForce training script + FSDP launch — `dualforce_train.py`, `dualforce_train_8gpu.sh`
- [x] DualForce inference pipeline — `pipeline_dualforce.py` with sliding window
- [x] DualForce dataset class — `dualforce_dataset.py` with multi-modal features
- [x] Factory function — `DualForceTrain_from_pretrained` registered in DIFFUSION_PIPELINES
- [x] Trainer generalization — AccelerateTrainer batch-key agnostic

**Milestone:** MOVA-Lite forward pass works, <50GB VRAM

### Phase 2: Causal Pretraining (Week 3-4)

- [ ] Prepare general video data (VoxCeleb2 + Panda-70M subsets)
- [ ] Train MOVA-Lite with causal attention (video-only)
- [ ] Verify loss convergence
- [ ] Save `mova_lite_causal_base.pt`

**Milestone:** Causal video generation produces coherent frames

### Phase 3: Multi-Modal Architecture (Week 4-6)

- [ ] Create Wan3DStructModel (fork from WanAudioModel)
- [ ] Implement Implicit3DEncoder (LivePortrait wrapper)
- [ ] Implement StructTokenProjector
- [ ] Adapt Bridge for video<->3D
- [ ] Add HuBERT audio conditioning path
- [ ] Implement DualAdaLNZero
- [ ] Implement gated residual connections
- [ ] Verify full dual-tower forward pass

**Milestone:** Video + 3D + audio all flow through, correct output shapes

### Phase 4: Diffusion Forcing Training (Week 6-8)

- [ ] Implement per-frame noise schedule
- [ ] Build MultiModalTalkingHeadDataset
- [ ] Implement multi-modal losses (video, struct, FLAME, lip-sync)
- [ ] Train on HDTF + CelebV-HQ (~81h)
- [ ] Monitor training milestones (1K/5K/20K/50K steps)

**Milestone:** Both video and struct losses decrease, recognizable faces generated

### Phase 5: Evaluation & Ablation (Week 8-10)

- [ ] Build evaluation pipeline (FVD, FID, ACD, Sync-C/D, APD)
- [ ] Run 5 ablation studies
- [ ] Compare with baselines
- [ ] Long-sequence consistency tests (32/128/256/512 frames)

**Milestone:** Complete ablation tables with clear trends

### Phase 6: Paper (Week 10-12+)

- [ ] Compile results (tables, figures, visualizations)
- [ ] Draft paper (NeurIPS/CVPR format)
- [ ] Internal review and iteration

**Milestone:** Complete manuscript

---

## 6. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Training from scratch doesn't converge | Medium | High | Progressive training, more data, longer schedule |
| 3D latent quality unstable | Medium | High | FLAME cross-check filtering, soft alignment loss |
| YouTube videos unavailable | High | Medium | Download ASAP, prioritize direct-download datasets |
| OOM on 8x A100 | Low | Medium | Gradient checkpointing, reduce batch/frames |
| Diffusion Forcing training instability | Low | High | Start with small model validation, ablate sigma ranges |

---

## 7. Immediate Next Steps

1. Verify MOVA inference works
2. Start HDTF download (YouTube links may expire)
3. Explore LivePortrait motion extractor interface
4. Create MOVA-Lite config and verify forward pass
5. Set up preprocessing pipeline
