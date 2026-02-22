# 3D-Aware Native Autoregressive Diffusion for Real-Time Avatar Video Generation

## 研究思路分析与细化方案（v3.4）

---

## 1. 背景与问题

### 1.1 研究背景

Audio-driven talking head video generation 是 AIGC 领域的核心应用之一。当前主流方案面临三个根本性挑战：

- **长序列时间一致性差**：自回归视频扩散模型（如 Hallo2/3）在生成长视频时面临身份漂移、表情抖动等问题，缺乏有效的跨帧结构约束。
- **训练依赖大规模蒸馏**：实现自回归扩散（如 Self-Forcing、CausVid）需要复杂的 teacher-student 蒸馏训练，计算开销大、工程复杂度高。
- **3D 结构信息利用不足**：人脸具有天然的 3D 几何先验，但当前方法要么完全忽略（纯 2D diffusion），要么过度依赖显式 3D（FLAME 参数，表征力有限），要么采用串行 pipeline（3D 预测 → video 生成，误差单向传播）。

### 1.2 赛道选择

避开大规模通用视频生成（需要千卡级 GPU），聚焦 **实时自回归扩散 + Avatar 场景**。8 卡 A100 可做有意义的增量研究。人脸/人体 avatar 场景具有明确的 3D 几何约束，是最适合引入结构先验的视频生成子领域。

### 1.3 核心问题

> **如何在不引入显式 3D 输入的前提下，让自回归视频扩散模型获得 3D 结构感知能力，从而同时提升长序列一致性、降低训练复杂度？**

---

## 2. 设计动机与核心创新

### 2.1 三个关键洞察

**洞察 1：Diffusion Forcing 天然支持多模态联合去噪**

Diffusion Forcing（Chen et al., 2024, ICML Oral）的核心机制——序列中每个 token 独立赋予不同噪声水平——使得不同模态（video latent、3D structure latent）可以在同一框架下以不同噪声 schedule 联合去噪。无需蒸馏即可实现自回归生成。

**洞察 2：隐式 3D latent 比显式 FLAME 参数更优**

LivePortrait-style implicit 3D latent 既保留了 3D 结构先验（低维、结构化），又具备比 FLAME 更强的表征力（能建模细微表情、非刚性动态）。将其作为与 video latent 平行的 "第二模态" 参与联合去噪，而非串行 pipeline 中的中间产物。

**洞察 3：浅层融合 + 深层特化的多模态架构最有效**

来自 OVI（Character.AI, 2025）、OmniTalker（NeurIPS 2025）、LTX-2（Lightricks, 2026）的经验表明：多模态 attention 融合应集中在网络浅层（structural layout），深层各模态独立特化（appearance details），避免过度耦合导致质量下降。

### 2.2 核心创新

> **在 Diffusion Forcing 框架下，将隐式 3D 结构 latent 作为与 video latent 平行的模态，通过浅层双向 attention 融合 + 深层独立特化实现联合自回归生成。3D 模态在低维空间中提供结构先验和跨帧一致性锚点，video 模态在高维空间中生成高质量外观细节，两者互相增益而非互相限制。Audio 作为 conditioning-only 信号驱动时序动态。推理时无需任何显式 3D 输入。**

**本方案首次统一了三大前沿方向**：
1. **3D-aware video generation**（Geometry Forcing, NeurIPS 2025 验证）
2. **多模态双向 attention 融合**（OVI/LTX-2/OmniTalker 验证）
3. **Diffusion Forcing 自回归训练**（clean paradigm, 无需蒸馏）

### 2.3 创新点分解

| # | 创新点 | 技术描述 | 解决的问题 |
|---|--------|---------|-----------|
| 1 | 多模态 Diffusion Forcing | 首次将 Diffusion Forcing 扩展到 video + 3D latent 联合生成 | 消除蒸馏依赖 |
| 2 | Shallow Fusion + Deep Specialization | 浅层（0-7）双向跨模态 attention + 深层（8-27）独立特化 | 3D 增益而不限制 video |
| 3 | Asymmetric Noise Schedule | σ_s ~ U(0,0.7) vs σ_v ~ U(0,1.0)，隐式"先结构后外观"生成 | 无需两阶段 pipeline |
| 4 | Audio Conditioning via Dual-path | audio→video + audio→3D→video 双路径驱动 | 鲁棒的唇音同步 |
| 5 | Inference-time 3D-free | 训练时 3D 监督，推理时从噪声联合生成 | 实用性 |

---

## 3. 系统概览：训练与推理流程图

### 3.1 训练流程（Training Pipeline）

```
                              Training Pipeline
 ╔═══════════════════════════════════════════════════════════════════════════╗
 ║                                                                         ║
 ║   Input Video Clip (T frames)                                           ║
 ║   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               ║
 ║   │ Frame_0  │  │ Frame_1  │  │ Frame_2  │  │  ...     │               ║
 ║   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘               ║
 ║        │              │              │              │                    ║
 ║   ┌────▼─────────────▼──────────────▼──────────────▼───────┐            ║
 ║   │               Pre-computed Feature Extraction           │            ║
 ║   │  ┌────────────┐ ┌────────────┐ ┌─────────────────┐     │            ║
 ║   │  │ Video VAE  │ │ LivePort.  │ │  HuBERT-Large   │     │            ║
 ║   │  │ Encoder ❄️  │ │ Motion ❄️  │ │  Audio Enc. ❄️   │     │            ║
 ║   │  └─────┬──────┘ └─────┬──────┘ └───────┬─────────┘     │            ║
 ║   │        │              │                 │               │            ║
 ║   │    z_t ∈ R^{C×H×W}  s_t ∈ R^{D_s}   a_t ∈ R^{D_a}    │            ║
 ║   │   (video latent)  (3D struct lat.)  (audio feat.)      │            ║
 ║   └────────┬──────────────┬─────────────────┬───────────────┘            ║
 ║            │              │                 │                            ║
 ║   ┌────────▼──────────────▼─────────────────▼───────────────┐            ║
 ║   │         Diffusion Forcing: Per-Token Noise Sampling      │            ║
 ║   │                                                          │            ║
 ║   │  For each frame t independently:                         │            ║
 ║   │    σ_v(t) ~ U(0, 1.0)    ← video noise level            │            ║
 ║   │    σ_s(t) ~ U(0, 0.7)    ← 3D noise level (偏小)        │            ║
 ║   │                                                          │            ║
 ║   │  z̃_t = (1-σ_v)·z_t + σ_v·ε_v    ← noisy video          │            ║
 ║   │  s̃_t = (1-σ_s)·s_t + σ_s·ε_s    ← noisy structure      │            ║
 ║   │  a_t = a_t (clean)               ← audio conditioning   │            ║
 ║   └────────┬──────────────┬─────────────────┬───────────────┘            ║
 ║            │              │                 │                            ║
 ║   Ref ─────┤              │                 │                            ║
 ║   Image    │              │                 │                            ║
 ║   ┌────┐   │              │                 │                            ║
 ║   │DINO│   │              │                 │                            ║
 ║   │/CLIP│  │              │                 │                            ║
 ║   │ ❄️  │   │              │                 │                            ║
 ║   └──┬─┘   │              │                 │                            ║
 ║      │     │              │                 │                            ║
 ║   ┌──▼─────▼──────────────▼─────────────────▼───────────────┐            ║
 ║   │            MOVA-Lite Multi-Modal Causal DiT              │            ║
 ║   │                                                          │            ║
 ║   │  ┌─────────────────────────────────────────────────────┐ │            ║
 ║   │  │ Shallow Blocks (0 ~ N_fusion-1):                    │ │            ║
 ║   │  │   Spatial Self-Attn (video)                         │ │            ║
 ║   │  │   → Bidirectional Cross-Modal Attn (video↔3D)       │ │            ║
 ║   │  │   → Audio Cross-Attn (audio→video, audio→3D)        │ │            ║
 ║   │  │   → Temporal Causal Attn (per-modality, masked)     │ │            ║
 ║   │  │   → Dual FFN + Dual AdaLN(σ_v, σ_s)                │ │            ║
 ║   │  └─────────────────────────────────────────────────────┘ │            ║
 ║   │  ┌─────────────────────────────────────────────────────┐ │            ║
 ║   │  │ Deep Blocks (N_fusion ~ L-1):                       │ │            ║
 ║   │  │   Spatial Self-Attn (video)                         │ │            ║
 ║   │  │   → Temporal Causal Attn (per-modality, masked)     │ │            ║
 ║   │  │   → Dual FFN + Dual AdaLN(σ_v, σ_s)                │ │            ║
 ║   │  └─────────────────────────────────────────────────────┘ │            ║
 ║   │                                                          │            ║
 ║   │  Output: v̂_video (velocity), v̂_struct (velocity)        │            ║
 ║   └──────────┬──────────────────────┬───────────────────────┘            ║
 ║              │                      │                                    ║
 ║   ┌──────────▼──────────────────────▼───────────────────────┐            ║
 ║   │                    Loss Computation                      │            ║
 ║   │                                                          │            ║
 ║   │  L_total = L_video + λ_s·L_struct                        │            ║
 ║   │           + λ_f·L_flame_align + λ_lip·L_lip_sync        │            ║
 ║   │                                                          │            ║
 ║   │  L_video:  Flow matching (v-pred) on video latents       │            ║
 ║   │  L_struct: Flow matching (v-pred) on 3D struct latents   │            ║
 ║   │  L_flame:  Soft FLAME alignment (decaying weight)        │            ║
 ║   │  L_lip:    Contrastive lip-sync (audio vs mouth region)  │            ║
 ║   └─────────────────────────────────────────────────────────┘            ║
 ╚═══════════════════════════════════════════════════════════════════════════╝
```

### 3.2 推理流程（Inference Pipeline）

```
                           Inference Pipeline
 ╔═══════════════════════════════════════════════════════════════════════════╗
 ║                                                                         ║
 ║   Inputs:                                                               ║
 ║     ┌─────────────┐         ┌───────────────────────────────┐           ║
 ║     │ Ref Image   │         │ Audio Stream (streaming input) │           ║
 ║     └──────┬──────┘         └──────────────┬────────────────┘           ║
 ║            │                               │                            ║
 ║     ┌──────▼──────┐              ┌─────────▼──────────┐                 ║
 ║     │ DINO/CLIP ❄️ │              │  HuBERT-Large ❄️    │                 ║
 ║     └──────┬──────┘              └─────────┬──────────┘                 ║
 ║            │                               │                            ║
 ║     ref_tokens ∈ R^{N_ref×D}       a_t ∈ R^{N_a×D}                     ║
 ║     (16 tokens, 一次计算)           (4 tokens/frame, 流式)               ║
 ║            │                               │                            ║
 ║     ┌──────▼───────────────────────────────▼──────────────────┐         ║
 ║     │          Autoregressive Frame-by-Frame Loop              │         ║
 ║     │                                                          │         ║
 ║     │  For t = 0, 1, 2, ...:                                   │         ║
 ║     │                                                          │         ║
 ║     │   Step 1: Initialize from noise                          │         ║
 ║     │     z_t ← N(0, I) ∈ R^{C_v×H×W}    (video noise)       │         ║
 ║     │     s_t ← N(0, I) ∈ R^{D_s}         (3D struct noise)   │         ║
 ║     │                                                          │         ║
 ║     │   Step 2: Multi-step denoising (K steps, e.g. K=20)     │         ║
 ║     │   ┌────────────────────────────────────────────────┐     │         ║
 ║     │   │  For k = K, K-1, ..., 1:                       │     │         ║
 ║     │   │                                                │     │         ║
 ║     │   │   ┌─────────────────────────────────────────┐  │     │         ║
 ║     │   │   │  MOVA-Lite Multi-Modal Causal DiT       │  │     │         ║
 ║     │   │   │                                         │  │     │         ║
 ║     │   │   │  Inputs:                                │  │     │         ║
 ║     │   │   │    z̃_t (noisy video)                    │  │     │         ║
 ║     │   │   │    s̃_t (noisy struct)                   │  │     │         ║
 ║     │   │   │    a_t (clean audio)                    │  │     │         ║
 ║     │   │   │    ref_tokens (clean ref)               │  │     │         ║
 ║     │   │   │    KV_cache (past frames t'<t)          │  │     │         ║
 ║     │   │   │    σ_v^(k), σ_s^(k) (noise levels)     │  │     │         ║
 ║     │   │   │                                         │  │     │         ║
 ║     │   │   │  Outputs:                               │  │     │         ║
 ║     │   │   │    v̂_video, v̂_struct (velocities)       │  │     │         ║
 ║     │   │   └───────────────┬─────────────────────────┘  │     │         ║
 ║     │   │                   │                             │     │         ║
 ║     │   │   z̃_t ← update(z̃_t, v̂_video, σ_v^(k))        │     │         ║
 ║     │   │   s̃_t ← update(s̃_t, v̂_struct, σ_s^(k))       │     │         ║
 ║     │   │                                                │     │         ║
 ║     │   └────────────────────────────────────────────────┘     │         ║
 ║     │                                                          │         ║
 ║     │   Step 3: Obtain clean predictions                       │         ║
 ║     │     z_t = z̃_t  (denoised video latent)                   │         ║
 ║     │     s_t = s̃_t  (denoised 3D latent, for next frame)     │         ║
 ║     │                                                          │         ║
 ║     │   Step 4: Update KV-cache & decode                       │         ║
 ║     │     KV_cache.append(z_t, s_t)                            │         ║
 ║     │     Frame_t = VAE_decode(z_t)  → stream output           │         ║
 ║     │                                                          │         ║
 ║     └──────────────────────────┬───────────────────────────────┘         ║
 ║                                │                                         ║
 ║                     ┌──────────▼──────────┐                              ║
 ║                     │   Output Video      │                              ║
 ║                     │  (streaming, T fps) │                              ║
 ║                     └─────────────────────┘                              ║
 ╚═══════════════════════════════════════════════════════════════════════════╝
```

### 3.3 单 DiT Block 内部信息流（Shallow Block）

```
                Shallow Block (layer_idx < N_fusion) 内部详细流程

 Input tokens at frame t:
   video_tokens [B, N_v, D]    struct_tokens [B, N_s, D]    audio_tokens [B, N_a, D]
        │                            │                             │
        ▼                            │                             │
 ┌──────────────────┐                │                             │
 │ Spatial Self-Attn│ (video only,   │                             │
 │ (per-frame)      │  保留原始      │                             │
 └────────┬─────────┘  video DiT     │                             │
          │             能力)         │                             │
          ▼                          ▼                             ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                    Cross-Modal Attention (Gated)                         │
 │                                                                         │
 │   ┌──────────┐  struct_to_video  ┌──────────┐  video_to_struct          │
 │   │  video   │◄════════════════►│  struct  │                           │
 │   │          │  (bidirectional)   │          │                           │
 │   └─────┬────┘                   └─────┬────┘                           │
 │         │                              │                                │
 │         │  audio_to_video              │  audio_to_struct               │
 │         │◄───────────────────┐         │◄──────────────────┐            │
 │         │                    │         │                   │            │
 │         │              ┌─────┴────┐    │             ┌─────┴────┐       │
 │         │              │  audio   │    │             │  audio   │       │
 │         │              │ (cond.)  │    │             │ (cond.)  │       │
 │         │              └──────────┘    │             └──────────┘       │
 │         │                              │                                │
 │   Gating:                        Gating:                                │
 │   v += σ(gate_s2v)·v_from_s     s += σ(gate_v2s)·s_from_v              │
 │   v += σ(gate_a2v)·v_from_a     s += σ(gate_a2s)·s_from_a              │
 │   (gates init=0, 渐进打开)       (gates init=0, 渐进打开)                 │
 └─────────┬──────────────────────────────┬────────────────────────────────┘
           │                              │
           ▼                              ▼
 ┌──────────────────┐          ┌──────────────────┐
 │ Temporal Causal   │          │ Temporal Causal   │
 │ Attn (video)      │          │ Attn (struct)     │
 │ attend t' ≤ t     │          │ attend t' ≤ t     │
 │ + KV-cache        │          │ + KV-cache        │
 └────────┬─────────┘          └────────┬─────────┘
          │                             │
          ▼                             ▼
 ┌──────────────────┐          ┌──────────────────┐
 │ AdaLN(σ_v) + FFN │          │ AdaLN(σ_s) + FFN │
 │ (video-specific)  │          │ (struct-specific) │
 └────────┬─────────┘          └────────┬─────────┘
          │                             │
          ▼                             ▼
   video_tokens_out              struct_tokens_out
```

---

## 4. 趋势验证与相关工作分析

### 4.1 最高度相关：Geometry Forcing（July 2025, NeurIPS 2025）

**直接验证了 "3D + video diffusion 融合" 的核心思路。**

- **方法**：将 video diffusion 中间特征与预训练几何基础模型（VGGT）的特征进行对齐
- **融合方式**：Angular + scale feature alignment loss
- **结果**：FVD 从 364 降至 243（RealEstate10K），证明 3D awareness 显著提升长序列一致性
- **与本方案的区别**：Geometry Forcing 用 feature alignment loss（隐式对齐），我们用 explicit 3D latent tokens 参与 attention（显式融合）；它聚焦 camera-conditioned world generation，我们聚焦 avatar。我们的 3D tokens 在 Diffusion Forcing 框架下也是被联合去噪的，不只是 conditioning
- **启示**：3D+video fusion 已被顶会验证，我们在此基础上更进一步

### 4.2 多模态 Audio-Video 融合架构对比

#### OVI（Character.AI, Sept 2025）⭐⭐⭐ —— 架构设计最相似，KV-Cache 参考来源

- **架构**：Twin-backbone cross-modal fusion（5B video + 5B audio + 1B fusion）
- **融合机制**：Blockwise bidirectional cross-attention + Scaled-RoPE for temporal alignment
- **开源**：`character-ai/Ovi`（GitHub，仅推理代码，无训练脚本）
- **关键洞察**：两个模态使用相同架构，通过 blockwise attention 融合；Scaled-RoPE 确保 audio-video 时间对齐
- **参考价值**：KV-Cache 实现方案的核心参考；cross-attention 融合设计启发了本方案的 shallow fusion 架构

#### LTX-2（Lightricks, Jan 2026）⭐⭐⭐

- **架构**：Asymmetric dual-stream DiT（14B video + 5B audio）
- **关键洞察**：**非对称容量分配**（video >> audio），cross-modality AdaLN
- **参考价值**：Asymmetric capacity + dual AdaLN 设计

#### OmniTalker（NeurIPS 2025）⭐⭐⭐

- **架构**：Dual-branch DiT（audio + video）
- **关键洞察**：**Shallow layers = cross-modal fusion；Deep layers = independent processing**
- **参考价值**：直接启发了我们的 shallow fusion + deep specialization 设计

#### UniVerse-1（Sept 2025）⭐⭐

- **架构**：Stitching of Experts（SoE）—— 融合预训练 Wan2.1 + Ace-step
- **关键洞察**：不需要从头训练，通过 MLP 连接两个 pretrained 模型
- **参考价值**：替代的轻量级融合方案（MLP vs attention）

#### MIDAS（Aug 2025）⭐⭐

- **架构**：LLM-based autoregressive（Qwen2.5-3B）+ diffusion head
- **关键洞察**：多模态 tokens 作为 LLM 输入，64× 压缩 autoencoder
- **参考价值**：验证了 multimodal token-based 控制对 avatar 生成的有效性

### 4.3 架构对比总结

| Model | 融合策略 | 训练方式 | 模态平衡 | 与本方案相似度 |
|-------|---------|---------|---------|--------------|
| **OVI** | Twin-backbone + blockwise cross-attn | From scratch | Symmetric | ⭐⭐⭐ 最高 → **KV-Cache 参考** |
| **LTX-2** | Dual-stream + cross-attn + AdaLN | From scratch | Asymmetric | ⭐⭐⭐ 高 |
| **OmniTalker** | Early fusion + late specialization | From scratch | Dual-branch | ⭐⭐⭐ 高 |
| **UniVerse-1** | Stitching of Experts (MLP) | Pretrained fusion | Symmetric | ⭐⭐ 中 |
| **MIDAS** | LLM + multimodal tokens | LLM-based | Token-based | ⭐⭐ 中 |

### 4.4 趋势对齐评估 ✅

本方案对齐 2024-2025 年三大主流趋势：

1. **3D-aware video generation**（Geometry Forcing, FantasyWorld, WorldForge, DSG-World, Aether 等）：大量工作验证 3D awareness 提升视频一致性，Geometry Forcing 证明 FVD 提升 33%+
2. **多模态双向 attention 融合**（OVI, LTX-2, OmniTalker）：2025 年 audio-video 领域的主流架构，一致地优于 sequential/cascaded pipelines
3. **Diffusion Forcing for autoregressive generation**（Geometry Forcing 直接使用）：新兴标准，无需蒸馏，clean training paradigm

### 4.5 其他 3D-Aware World Model 参考

| 项目 | 3D 表征 | 与 video diffusion 融合方式 | 时间 |
|------|--------|---------------------------|------|
| FantasyWorld | 3D Gaussian | Gaussian-guided video diffusion | 2025 |
| WorldForge | Geometry-guided latent | Geometric conditioning | 2025 |
| DSG-World | 3D Gaussian Splatting | Gaussian-guided diffusion | 2025 |
| Aether | Unified geometric-latent | Joint prediction | 2025 |
| MagicWorld | Scene decomposition | Decomposed conditioning | 2025 |

---

## 5. 隐式 3D Latent 表征选型

### 5.1 候选方案对比

| 模型 / 方法 | 表征形式 | 维度 | 优势 | 劣势 | 开源 |
|------------|---------|------|------|------|------|
| **LivePortrait** | Implicit 3D keypoints + exp/pose | ~66-128 dim | 大规模训练、表征力强、motion/appearance 解耦 | 主要面向 reenactment | ✅ |
| **EMOCA / DECA** | FLAME coefficients | ~159 dim | 语义明确、可解释性强 | 表征力有限 | ✅ |
| **SadTalker ExpNet** | Learned 3DMM motion | ~64 dim | 端到端、audio-aligned | 模型容量有限 | ✅ |
| **MegaPortraits** | Volumetric features | ~512 dim | 极强表征力 | 维度高、计算大 | ❌ |
| **X-Portrait** | Appearance/motion decomposition | ~256 dim | 设计简洁 | 开源不完整 | 部分 |
| **Tri-plane (EG3D)** | 3 × H × W feature planes | ~3×32³ | 3D-aware | 维度太高 | ✅ |

### 5.2 推荐方案：LivePortrait Motion Latent 为主，FLAME 为辅

**选择理由**：维度适中（~128 dim）可作为低维 token 高效参与 attention；开源且经充分验证；天然支持 motion/appearance 解耦。FLAME 参数作为辅助软监督，不限制 latent 表征力。

```python
class Implicit3DEncoder(nn.Module):
    """初始化自 LivePortrait motion extractor"""
    def __init__(self, pretrained_path, latent_dim=128):
        super().__init__()
        self.backbone = load_liveportrait_motion_extractor(pretrained_path)
        self.projector = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )
    
    def forward(self, image):
        implicit_3d = self.backbone(image)  # [B, backbone_dim]
        return self.projector(implicit_3d)  # [B, latent_dim]
```

### 5.3 其他值得关注的备选

- **EMOPortraits**（NeurIPS 2024）：cross-driving 兼容的 expression latent
- **Real3D-Portrait**（ICLR 2024）：3D-aware motion encoder
- **GAIA**（Tencent, 2024）：global/local motion 分解

---

## 6. 核心技术设计：多模态 Diffusion Forcing + Causal 生成

本节详细阐述方案的两个核心技术支柱：(1) Diffusion Forcing 如何实现多模态联合自回归生成，(2) 多模态融合架构中 causal attention 的具体实现。

### 6.1 Diffusion Forcing 原理与多模态扩展

#### 6.1.1 标准 Diffusion Forcing 回顾

标准 Diffusion Forcing 处理 1D token 序列 $\{x_1, x_2, ..., x_T\}$，训练时为每个 token $x_t$ 独立采样噪声水平 $\sigma_t \sim p(\sigma)$：

$$\tilde{x}_t = (1 - \sigma_t) \cdot x_t + \sigma_t \cdot \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

模型在给定混合噪声序列的条件下预测每个 token 的去噪方向（velocity）：

$$v_\theta(\tilde{x}_t | \{\tilde{x}_{t'}\}_{t' \neq t}, \{\sigma_{t'}\}_{t'=1}^T)$$

**关键**：通过 causal mask，$x_t$ 只能看到 $t' \leq t$ 的 tokens。当过去 tokens 已去噪（$\sigma_{t'}$ 小），当前 token 正在去噪（$\sigma_t$ 中等），未来 tokens 为纯噪声（$\sigma_{t''}$ 大），自回归生成自然涌现。

#### 6.1.2 多模态扩展：Video + 3D 联合去噪

本方案的核心扩展：在每个时间步 $t$ 存在**两组** tokens 需要去噪——video latent $z_t$ 和 3D structure latent $s_t$——各有独立的噪声水平：

$$\tilde{z}_t = (1 - \sigma_v^{(t)}) \cdot z_t + \sigma_v^{(t)} \cdot \epsilon_v, \quad \sigma_v^{(t)} \sim \mathcal{U}(0, 1.0)$$
$$\tilde{s}_t = (1 - \sigma_s^{(t)}) \cdot s_t + \sigma_s^{(t)} \cdot \epsilon_s, \quad \sigma_s^{(t)} \sim \mathcal{U}(0, 0.7)$$

$\sigma_s$ 的上界为 0.7（而非 1.0），使得 **3D tokens 在训练时总是比 video tokens 更"干净"**。这创造了隐式的"先结构后外观"生成顺序：

- 在联合 attention 中，低噪声的 3D tokens 自然成为 structural anchor
- Video tokens attend to 已部分去噪的 3D tokens 获取结构引导
- 不需要显式的两阶段 pipeline，但效果类似

Audio $a_t$ 始终为 clean（$\sigma_a = 0$），作为 conditioning-only 信号。

#### 6.1.3 联合去噪的 Flow Matching 形式

模型预测两个 velocity field：

$$v_\theta^{video}(\tilde{z}_t, \tilde{s}_t, a_t, r | \sigma_v^{(t)}, \sigma_s^{(t)}) \quad \text{(video velocity)}$$
$$v_\theta^{struct}(\tilde{z}_t, \tilde{s}_t, a_t, r | \sigma_v^{(t)}, \sigma_s^{(t)}) \quad \text{(structure velocity)}$$

两个 velocity 共享同一个 backbone（MOVA-Lite DiT），通过 Dual AdaLN 分别接收 $\sigma_v$ 和 $\sigma_s$ 的条件信息，最终由各自独立的 output head 输出。

训练 loss：
```python
L_video  = E[|| v_θ^video  - (z_t - ε_v) ||²]   # flow matching v-prediction
L_struct = E[|| v_θ^struct - (s_t - ε_s) ||²]   # flow matching v-prediction

L_total = L_video + λ_s·L_struct + λ_f·L_flame + λ_lip·L_lip_sync
```

#### 6.1.4 完整训练伪代码

```python
def diffusion_forcing_training_step(model, batch):
    """
    基于 MOVA 训练基础设施改造的多模态 Diffusion Forcing 训练步。
    MOVA 原始: 标准 diffusion on video+audio, 统一 σ
    本方案:   Diffusion Forcing on video+3D, 独立 σ per token
    """
    B, T = batch['video_latents'].shape[:2]
    
    # ======= 核心改造点 1: Per-token 独立噪声 (替代 MOVA 的统一噪声) =======
    # Video: 标准均匀采样
    sigma_v = torch.rand(B, T, device=device)          # [B, T], 每帧独立
    # 3D Structure: 偏小噪声（隐式先验：3D 先去噪完成）
    sigma_s = torch.rand(B, T, device=device) * 0.7    # [B, T], 上界 0.7
    
    # 为每帧独立加噪
    noisy_video = []
    noisy_struct = []
    for t in range(T):
        eps_v = torch.randn_like(batch['video_latents'][:, t])
        eps_s = torch.randn_like(batch['struct_latents'][:, t])
        noisy_video.append(
            (1 - sigma_v[:, t:t+1, None, None]) * batch['video_latents'][:, t]
            + sigma_v[:, t:t+1, None, None] * eps_v
        )
        noisy_struct.append(
            (1 - sigma_s[:, t:t+1]) * batch['struct_latents'][:, t]
            + sigma_s[:, t:t+1] * eps_s
        )
    noisy_video = torch.stack(noisy_video, dim=1)    # [B, T, C_v, H, W]
    noisy_struct = torch.stack(noisy_struct, dim=1)   # [B, T, D_s]
    
    # ======= 核心改造点 2: 联合去噪 (MOVA-Lite backbone + Diffusion Forcing mask) =======
    pred_v_video, pred_v_struct = model(
        noisy_video=noisy_video,
        noisy_struct=noisy_struct,
        audio=batch['audio_features'],       # [B, T, D_a], clean
        ref=batch['ref_features'],           # [B, D_ref], clean
        sigma_v=sigma_v,                     # [B, T], per-frame
        sigma_s=sigma_s,                     # [B, T], per-frame
        causal_mask=True,                    # 时间维度 causal
    )
    
    # ======= Loss 计算 =======
    loss_video = flow_matching_loss(pred_v_video, batch['video_latents'], sigma_v)
    loss_struct = flow_matching_loss(pred_v_struct, batch['struct_latents'], sigma_s)
    loss_flame = flame_alignment_loss(pred_v_struct, batch['flame_params'])
    loss_lip = lip_sync_contrastive_loss(
        pred_v_video, batch['audio_features'], batch['mouth_masks']
    )
    
    loss = loss_video + 0.5 * loss_struct + 0.1 * loss_flame + 0.3 * loss_lip
    return loss
```

### 6.2 Causal Attention 在多模态 DiT 中的实现

#### 6.2.1 Attention Mask 设计

本方案的 attention mask 需要同时处理两个维度的约束：**时间维度（causal）** 和 **模态维度（selective）**。

```
时间维度 Causal Mask (所有模态共享):

          t=0    t=1    t=2    t=3
  t=0  [  ✓  ] [  ✗  ] [  ✗  ] [  ✗  ]
  t=1  [  ✓  ] [  ✓  ] [  ✗  ] [  ✗  ]
  t=2  [  ✓  ] [  ✓  ] [  ✓  ] [  ✗  ]
  t=3  [  ✓  ] [  ✓  ] [  ✓  ] [  ✓  ]

模态维度 Attention Pattern (同一时间步 t 内):

Shallow Blocks (layer < N_fusion):
                REF     AUDIO    STRUCT    VIDEO
  REF         [ self ] [  ✗  ] [  ✗  ]  [  ✗  ]    ← REF 不更新
  AUDIO       [ ✓   ] [ self ] [  ✗  ]  [  ✗  ]    ← Audio 不更新
  STRUCT      [ ✓   ] [ ✓←  ] [ self ]  [ ✓↔ ]    ← 3D 更新：video→3D, audio→3D
  VIDEO       [ ✓   ] [ ✓←  ] [ ✓↔  ]  [ self]    ← Video 更新：3D→video, audio→video

  ✓↔ = 双向 cross-attention (bidirectional)
  ✓←  = 单向 cross-attention (conditioning → target)

Deep Blocks (layer ≥ N_fusion):
                REF     AUDIO    STRUCT    VIDEO
  STRUCT      [ ✓   ] [  ✗  ] [ self ]  [  ✗  ]    ← 独立处理
  VIDEO       [ ✓   ] [  ✗  ] [  ✗  ]  [ self]    ← 独立处理
```

#### 6.2.2 KV-Cache 实现细节

在自回归推理时，causal attention 通过 KV-cache 高效实现。关键：**各模态维护独立的 KV-cache**：

```python
class MultiModalKVCache:
    """
    基于 OVI KV-cache 设计参考的多模态实现：
    OVI 原始: video_kv_cache + audio_kv_cache
    本方案:   video_kv_cache + struct_kv_cache + ref_kv_cache + audio_kv_cache
    """
    def __init__(self, num_layers, max_seq_len):
        # 每层、每模态独立的 K, V 缓存
        self.video_cache = {}   # {layer_idx: (K, V)} where K,V ∈ [B, T*N_v, D]
        self.struct_cache = {}  # {layer_idx: (K, V)} where K,V ∈ [B, T*N_s, D]
        self.ref_cache = {}     # {layer_idx: (K, V)} where K,V ∈ [B, N_ref, D] (固定)
        self.audio_cache = {}   # {layer_idx: (K, V)} where K,V ∈ [B, T*N_a, D]
    
    def update(self, layer_idx, video_kv, struct_kv, audio_kv, t):
        """追加当前帧 t 的 KV 到缓存"""
        if layer_idx not in self.video_cache:
            self.video_cache[layer_idx] = video_kv
            self.struct_cache[layer_idx] = struct_kv
            self.audio_cache[layer_idx] = audio_kv
        else:
            self.video_cache[layer_idx] = torch.cat(
                [self.video_cache[layer_idx], video_kv], dim=1
            )
            # ... 同理 struct, audio
    
    def get_for_temporal_attn(self, layer_idx, modality):
        """获取 temporal causal attention 的历史 KV"""
        if modality == 'video':
            return self.video_cache.get(layer_idx, None)
        elif modality == 'struct':
            return self.struct_cache.get(layer_idx, None)
    
    def get_for_cross_modal_attn(self, layer_idx, source_modality):
        """获取 cross-modal attention 中 source 模态的当前帧 KV"""
        # Cross-modal 只看同一帧的其他模态，不需要历史
        # 所以直接用当前帧的 tokens，不从 cache 取
        pass
```

#### 6.2.3 Temporal Causal Attention with KV-Cache

```python
class TemporalCausalAttention(nn.Module):
    """
    基于 MOVA temporal attention 的 causal 改造（KV-Cache 设计参考 OVI）。
    关键改造: MOVA 使用 full temporal attention，我们改为 strict causal + KV-Cache。
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.rope = ScaledRoPE(dim // num_heads)  # MOVA 的 Aligned-RoPE 保留
    
    def forward(self, x, modality, timestep, kv_cache=None):
        """
        x:          [B, N_tokens, D]  (当前帧的 tokens)
        modality:   'video' or 'struct'
        timestep:   当前帧索引
        kv_cache:   MultiModalKVCache 实例
        """
        # 1. 应用 Aligned-RoPE (与 MOVA 一致)
        x = self.rope(x, modality, timestep)
        
        # 2. 计算当前帧的 Q, K, V
        q = self.q_proj(x)
        k_curr = self.k_proj(x)
        v_curr = self.v_proj(x)
        
        # 3. 拼接历史 KV (causal: 只看过去帧)
        if kv_cache is not None:
            past_kv = kv_cache.get_for_temporal_attn(self.layer_idx, modality)
            if past_kv is not None:
                k_past, v_past = past_kv
                k = torch.cat([k_past, k_curr], dim=1)  # [B, T*N + N, D]
                v = torch.cat([v_past, v_curr], dim=1)
            else:
                k, v = k_curr, v_curr
        else:
            # 训练时: 使用 causal mask 代替 KV-cache
            k, v = k_curr, v_curr  # 这里简化，实际用 full sequence + causal mask
        
        # 4. Attention
        out = scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # 5. 更新 KV-cache
        if kv_cache is not None:
            kv_cache.update(self.layer_idx, k_curr, v_curr, modality, timestep)
        
        return self.out_proj(out)
```

#### 6.2.4 Diffusion Forcing 特有的噪声调制

```python
class DualAdaLNZero(nn.Module):
    """
    双 AdaLN-Zero: 不同模态接收不同噪声水平的条件信号。
    
    基于 MOVA 的 AdaLN 改造:
    MOVA 原始: 统一的 σ 对所有模态
    本方案:   video 用 σ_v, struct 用 σ_s, 各有独立 scale/shift
    """
    def __init__(self, dim):
        super().__init__()
        # Video: conditioned on σ_v + timestep
        self.video_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),  # scale, shift, gate × 2
        )
        # Structure: conditioned on σ_s + timestep
        self.struct_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        # 噪声水平编码器
        self.sigma_v_embed = TimestepEmbedder(dim)
        self.sigma_s_embed = TimestepEmbedder(dim)
    
    def forward(self, video_tokens, struct_tokens, sigma_v, sigma_s, timestep_emb):
        # Video modulation
        c_v = timestep_emb + self.sigma_v_embed(sigma_v)
        scale_v, shift_v, gate_v, _, _, _ = self.video_adaln(c_v).chunk(6, dim=-1)
        video_out = gate_v * (scale_v * layer_norm(video_tokens) + shift_v)
        
        # Structure modulation  
        c_s = timestep_emb + self.sigma_s_embed(sigma_s)
        scale_s, shift_s, gate_s, _, _, _ = self.struct_adaln(c_s).chunk(6, dim=-1)
        struct_out = gate_s * (scale_s * layer_norm(struct_tokens) + shift_s)
        
        return video_out, struct_out
```

### 6.3 Audio 参与机制

#### 6.3.1 角色定位

Audio = **Conditioning-only 模态**，不参与去噪。与 OVI（audio 也被生成）、MOVA（audio 也被生成）和 OmniTalker（audio 从 text 生成）不同，我们的 audio 是给定输入。这极大简化了训练。

三模态互补关系：
- Audio → **temporal dynamics**（何时动、怎么动）
- 3D structure → **spatial constraints**（运动的几何约束）
- Video → **appearance details**（最终的视觉呈现）

#### 6.3.2 Audio Encoder Pipeline

```
Raw audio (16kHz) → HuBERT-Large (frozen, 60K hours pretrained)
  → [B, T_audio, 1024] features (50Hz)
  → Temporal pooling (4:1 → align with video fps ~12.5fps)
  → Linear(1024, D_model) projection
  → [B, T_video, N_a, D_model] (N_a = 4 tokens per video frame)
  + Scaled-RoPE temporal embedding (与 video/struct 共享时间基准)
  + Learnable modality type embedding (E_audio)
```

#### 6.3.3 双路径 Audio 驱动

```python
# Shallow blocks 中:

# 路径 1: Audio → Video (直接驱动口型/表情的像素级表现)
v_from_a = audio_to_video_cross_attn(q=video_tokens, kv=audio_tokens)
video = video + sigmoid(gate_a2v) * v_from_a

# 路径 2: Audio → 3D Structure → Video (通过 3D 中介的间接驱动)
s_from_a = audio_to_struct_cross_attn(q=struct_tokens, kv=audio_tokens)
struct = struct + sigmoid(gate_a2s) * s_from_a
# 然后 struct 通过 struct→video cross-attention 传递给 video
```

双路径的好处：即使一条路径失效（如 3D 预测不准），另一条路径仍可提供 lip-sync 信号。

#### 6.3.4 Lip-sync Contrastive Loss

```python
def lip_sync_loss(video_features, audio_features, mouth_mask):
    mouth_feat = extract_mouth_region(video_features, mouth_mask)
    pos_sim = cosine_similarity(mouth_feat, audio_features)      # 同帧
    neg_sim = cosine_similarity(mouth_feat, audio_features.roll(  # 错帧
        shifts=random.randint(2, 10), dims=1))
    return -torch.log(exp(pos_sim/τ) / (exp(pos_sim/τ) + exp(neg_sim/τ))).mean()
```

---

## 7. 基于 MOVA-Lite 的模型架构与改造方案

### 7.1 Base Codebase 选型：MOVA vs OVI vs LTX-2

#### 三个候选方案对比

| 维度 | MOVA (OpenMOSS) | OVI (character-ai) | LTX-2 (Lightricks) |
|------|-----------------|---------------------|---------------------|
| **模型规模** | 32B total / 18B active (MoE) | ~2.35B (video+audio) | 19B (14B video + 5B audio) |
| **架构** | 非对称双塔 + Bridge CrossAttn | Twin-backbone DiT + cross-attn | 非对称双流 48 layers |
| **Video 基座** | Wan 2.2 (14B, **可缩减**) | 自研 DiT (~2B) | 自研 DiT (14B) |
| **Audio 集成** | ✅ 1.3B audio DiT + Bridge CrossAttn | ✅ 5B audio backbone | ✅ 5B audio stream |
| **Text Encoder** | UMT5 | — | Gemma |
| **KV-Cache** | ❌ 无 | ✅ 已实现 | ❌ 无 |
| **训练脚本** | ✅ **完整** (LoRA: 单卡/8卡 FSDP) | ❌ **无** (仅推理) | ✅ **最完整** (LoRA + Full FT) |
| **训练基础设施** | ✅ training loop, FSDP, checkpoint, data pipeline | ❌ 无任何训练代码 | ✅ 完整 monorepo |
| **数据 Pipeline** | ✅ 有 (video_audio_dataset.py) | ❌ 无 | ✅ 有 (ltx-trainer) |
| **Diffusion Forcing** | ❌ 标准 Diffusion | ❌ 标准 Flow Matching | ❌ 标准 Flow Matching |
| **模型可缩减性** | ✅ 改 config 即可缩减层数/维度/去 MoE | ⚠️ 原生 2B，无需缩减但也无法扩展 | ⚠️ 14B 太大，缩减丢失权重 |
| **显存需求 (训练)** | LoRA: ~18GB/GPU (360p) | 未知（无训练脚本） | LoRA: 80GB+; Full FT: 多卡 |
| **开源协议** | Apache 2.0 | Apache 2.0 | Apache 2.0 |

#### 核心权衡：训练基础设施 vs 架构接近度

**OVI 的优势**：架构最接近目标（原生 ~2B、已有 KV-Cache、flow matching）。
**OVI 的致命缺陷**：**完全没有训练代码**。仅提供推理脚本，无 training loop、无 FSDP 配置、无数据加载管线。从零编写一个稳定的 DiT 分布式训练 pipeline（包括 FSDP 封装、梯度 checkpointing、混合精度、数据加载、评估、保存恢复）至少需要 **2-4 周纯工程时间**，且调试风险不可忽略。

**MOVA 的优势**：**完整的训练基础设施**——虽然提供的是 LoRA fine-tuning 脚本而非预训练脚本，但这些脚本包含了完整的训练工程组件：

| MOVA 训练脚本提供的组件 | 说明 | 复用价值 |
|------------------------|------|---------|
| Training Loop | 前向/反向/梯度累积，完整的训练循环 | LoRA→Full FT 只需去掉 peft wrapper |
| FSDP 分布式训练 | 8 GPU × 50GB/卡，已验证配置 | 直接复用 |
| Data Pipeline | video_audio_dataset.py，标准数据加载 | 扩展为三模态 |
| Checkpointing | 模型保存/恢复/断点续训 | 直接复用 |
| Config 管理 | 完整配置文件（LoRA rank/alpha、optimizer、offload）| 修改参数即可 |
| 启动脚本 | 单卡/多卡训练脚本模板 | 直接复用 |
| Aligned RoPE | Video/Audio 不同采样率的时间对齐 | 直接复用，改为 Video/3D |

**MOVA 的局限**：模型过大（14B Video + MoE），无 KV-Cache，无 Diffusion Forcing。但这些都是可以解决的：

1. **模型缩减**：Wan 2.2 架构通过修改 config 即可缩减（减层数 48→28、减 hidden dim、去 MoE routing），得到 ~2B 的 "MOVA-Lite"。缩减后丢失预训练权重，但保留完整的代码结构
2. **KV-Cache**：参考 OVI 的实现方案添加，工作量约 1 周
3. **Diffusion Forcing**：修改 training loop 中的 noise scheduling，从 shared σ → per-token σ_v/σ_s

#### 推荐策略：MOVA-Lite（以 MOVA 为主体 base，缩减到 ~2B）

```
最终推荐: "MOVA-Lite" 策略 — 以 MOVA codebase 为主体

核心思路:
  ┌──────────────────────────────────────────────────────────────────────┐
  │  代码主体 ← MOVA (训练 loop + FSDP + 数据管线 + checkpoint 全保留)     │
  │  模型缩减 ← Wan 2.2 14B → ~2B dense (减层/减维/去 MoE)               │
  │  Bridge CrossAttn ← MOVA 最有价值的设计，直接复用为 shallow fusion     │
  │  KV-Cache ← 参考 OVI 实现方案新增                                    │
  │  核心创新 ← 本方案独有 (3D latent + Diffusion Forcing + Causal)        │
  └──────────────────────────────────────────────────────────────────────┘

为什么优于 "以 OVI 为主体 + 移植 MOVA 训练脚本" 的旧方案：
  1. 省去训练代码移植的适配工作 — 不再需要把 MOVA trainer 适配到 OVI model API
  2. 省去从零写训练基础设施的 2-4 周 — OVI 无训练代码是硬伤
  3. Bridge CrossAttn 可直接复用 — MOVA 的核心融合机制与本方案的 shallow fusion 高度匹配
  4. 缩减模型是配置层面操作 — 代码改动小，风险可控
  5. KV-Cache 可后加 — 参考 OVI 实现，比从零写训练 pipeline 简单得多

开发周期对比：
  MOVA-Lite 路线: ~3 周 (缩减模型 1 周 + 架构改造 1 周 + DF+KV 1 周)
  OVI + 自写训练: ~5 周 (写训练基础设施 2 周 + 架构改造 1 周 + DF 1 周 + 调试 1 周)
```

#### 为什么不直接用 MOVA 做 LoRA 微调？

虽然 MOVA 有现成的 LoRA 训练脚本，但 LoRA 微调 **不能实现** 本方案的核心创新：

| 本方案核心改动 | LoRA 能否实现？ | 原因 |
|---------------|----------------|------|
| Diffusion Forcing (per-token 独立噪声) | ❌ | 需改 training loop 的 noise scheduling，不是权重问题 |
| Causal attention mask | ❌ | 需改 attention mask 逻辑，不是权重问题 |
| 新增 3D structure stream | ❌ | 需新增完整的 backbone 分支，超出 LoRA 范围 |
| KV-Cache 推理 | ❌ | 需改推理 pipeline，MOVA 无此基础设施 |
| Audio 从生成模态变为 conditioning | ❌ | 需根本性改变 audio tower 的角色 |
| 模型缩减到 ~2B | ❌ | 需要修改模型架构定义，不是权重适配问题 |

**结论**：必须以 MOVA 的训练基础设施为工程骨架，但在模型架构层面做深度改造（缩减 + 替换 + 新增），而非简单的 LoRA 微调。

### 7.1.1 实施路径：MOVA → MOVA-Lite 改造流程

```
Week 0-1: MOVA 代码库理解 + 模型缩减
  ├── Clone MOVA + OVI repos
  ├── MOVA 代码分析:
  │   ├── 理解 Wan 2.2 DiT 架构定义（层数、维度、MoE 路由）
  │   ├── 理解 Bridge CrossAttention 实现细节
  │   ├── 理解训练 loop (mova/trainer/) + FSDP 配置
  │   └── 理解 video_audio_dataset.py 数据管线
  ├── 模型缩减 (config 层面):
  │   ├── Wan 2.2 14B → ~2B: 层数 48→28, hidden dim 缩减, 去除 MoE routing
  │   ├── Audio tower 1.3B → 保留架构但降维（后续替换为 3D stream）
  │   └── 验证: 缩减后模型能跑通 forward pass
  ├── 训练脚本适配:
  │   ├── LoRA scripts → Full fine-tuning (去掉 peft wrapper)
  │   └── 验证: FSDP 8卡训练 loop 正常运行
  └── OVI 代码分析:
      └── 理解 KV-Cache 实现 + Scaled-RoPE (后续移植参考)

Week 1-2: 架构改造
  ├── 替换 Audio tower → 3D Structure stream (~200M)
  │   ├── 保留 MOVA Bridge CrossAttn 机制，只换输入端
  │   └── LivePortrait encoder → StructTokenProjector → Bridge CrossAttn
  ├── 添加 HuBERT audio conditioning 支路 (shallow blocks)
  ├── 实现 Shallow Fusion + Deep Specialization split
  │   ├── 前 8 层: 保留 cross-modal attention (改为 video↔3D 双向 + audio 单向)
  │   └── 后 20 层: 移除 cross-attention，各模态独立
  ├── 数据管线扩展:
  │   └── video_audio_dataset.py → 三模态 (video + 3D pseudo label + audio)
  └── 验证: forward pass 通过，loss 计算正确

Week 2-3: Diffusion Forcing + KV-Cache 集成
  ├── 修改 noise scheduling → per-token 独立 (σ_v, σ_s per frame)
  ├── Full attention → Causal attention mask (修改 temporal attention)
  ├── 新增 KV-Cache (参考 OVI 实现):
  │   ├── 各模态独立 KV-Cache (video, struct, ref, audio)
  │   └── Causal mask → KV-Cache reuse for streaming inference
  └── 验证: 在 HDTF 子集上训练收敛 + 自回归推理正常
```

### 7.2 MOVA → MOVA-Lite 的改造清单

```
MOVA 原始架构:
  Video backbone (Wan 2.2, 14B, MoE) + Audio backbone (1.3B DiT)
  → Bridge CrossAttention (双向融合)
  → 标准 Diffusion training (统一 σ)
  → 同时生成 audio + video
  → Full temporal attention, 无 KV-Cache

本方案改造 (5 个核心改动):

改动 0: 模型缩减 (MOVA → MOVA-Lite)
  - Wan 2.2 Video: 14B MoE → ~2B dense
  - 具体: 层数 48→28, hidden dim 缩减, 去除 MoE expert routing
  - 丢失预训练权重，但保留完整的模型代码和训练基础设施
  - Audio tower: 1.3B → 保留结构但后续替换
  - 训练脚本: LoRA → Full FT (去掉 peft wrapper, 保留 FSDP/data pipeline/checkpoint)

改动 1: Audio backbone → 3D Structure backbone
  - 将 MOVA 的 audio backbone 替换为轻量级 3D structure stream
  - 参数量: 1.3B audio → ~200M 3D stream (大幅减少)
  - Token 数: MOVA audio tokens → 8 structure tokens per frame
  - MOVA 的 Bridge CrossAttention 机制完全保留，只换了一侧的模态
  - 这是 MOVA 最大的设计优势：Bridge CrossAttn 天然适配模态替换

改动 2: 标准 Diffusion → Diffusion Forcing
  - MOVA: 所有 tokens 共享同一 σ ~ p(σ)
  - 本方案: video σ_v ~ U(0,1.0), struct σ_s ~ U(0,0.7), 每帧独立
  - 代码改动: 修改训练 loop 中的 noise sampling + AdaLN conditioning
  - MOVA 训练 loop 结构保持不变，只改噪声采样逻辑

改动 3: Full Attention → Causal + Shallow/Deep Split
  - MOVA: 所有层 full temporal attention + 所有层 Bridge CrossAttention
  - 本方案:
    - Temporal: full → causal (修改 attention mask)
    - Cross-modal: Bridge CrossAttn 仅保留在前 N_fusion 层 (OmniTalker insight)
    - 深层: 移除 Bridge CrossAttn，各模态独立
  - MOVA 的 Bridge CrossAttn 代码可直接复用于 shallow blocks

改动 4: 新增 Audio Conditioning 支路 + KV-Cache
  - MOVA 中 audio 是被生成的模态
  - 本方案: audio 变为 conditioning-only 输入
  - 新增: HuBERT encoder + audio cross-attention (shallow blocks only)
  - 新增: KV-Cache (参考 OVI 实现，各模态独立缓存)
  - 不增加训练复杂度 (audio 不参与去噪)
```

### 7.3 改造后的架构规格

```
Base: MOVA-Lite Video Backbone (Wan 2.2 缩减版, ~2B dense DiT, 28 blocks)
      替换 MOVA Audio Backbone 为 3D Structure Stream
      训练基础设施: MOVA (FSDP, data pipeline, checkpoint, training loop)
      KV-Cache: 参考 OVI 实现方案新增

3D Structure Stream:
  - Implicit3DEncoder (LivePortrait, ~15M, frozen/LoRA)
  - StructTokenProjector (128 → 8 tokens × D_model, ~2.5M)
  - Structure-side AdaLN + FFN (per block, ~56M total)

Audio Conditioning:
  - HuBERT-Large (~95M, frozen)
  - AudioProjection (~1.2M)
  - Audio cross-attention in shallow blocks (~32M)

Cross-Modal Attention (Shallow Blocks 0~7):
  - 复用 MOVA Bridge CrossAttention 实现，改为 shallow-only
  - 改为: video↔struct (双向), audio→video (单向), audio→struct (单向)
  - 新增 gated residual (gates init=0)
  - 每个 shallow block: ~16M additional params
  - 8 blocks total: ~128M

Deep Blocks (8~27):
  - 移除 cross-attention (原 MOVA 全层 Bridge CrossAttn → 仅保留 shallow)
  - 保留: spatial self-attn + temporal causal attn + dual FFN/AdaLN
  - 每 block 仅增加 struct-side FFN+AdaLN: ~2M × 20 = ~40M

Output Heads:
  - Video head: 保留 MOVA 原始 (Wan 2.2 decoder head)
  - Structure head: 新增 MLP(D_model → 256 → D_s), ~5M

总参数量:
  - MOVA-Lite video backbone (缩减后 ~2B, LoRA rank=64 微调): ~2B
  - 3D Structure Stream: ~74M
  - Audio Conditioning: ~96M
  - Cross-Modal 新增: ~128M
  - Deep blocks 扩展: ~40M
  - 其他: ~12M
  Total: ~2.35B
  实际训练参数 (LoRA rank=64 on video backbone + all new modules): ~380M
```

### 7.4 架构图

```
                    ┌───────────────┐
                    │  Audio Stream  │
                    │ (HuBERT, ❄️)   │  ← Conditioning-only
                    └──────┬────────┘
                           │ audio_tokens (4/frame)
  ┌──────────────┐         │         ┌──────────────────┐
  │ Ref Image    │         │         │ Prev frames KV   │
  │ (DINO/CLIP,❄️)│         │         │ cache (causal)   │
  └──────┬───────┘         │         └────────┬─────────┘
         │ ref (16 tok)    │                  │
         ▼                 ▼                  ▼
  ┌──────────────────────────────────────────────────┐
  │   MOVA-Lite Multi-Modal Causal DiT (2.35B)          │
  │                                                    │
  │  ┌──────────────────────────────────────────────┐  │
  │  │  Shallow Blocks (0-7): CROSS-MODAL FUSION     │  │
  │  │    [Spatial Attn] → [Cross-Modal Attn] →      │  │
  │  │    [Temporal Causal Attn] → [Dual FFN]        │  │
  │  │                                                │  │
  │  │    Video ↔ 3D: 双向 (MOVA Bridge CrossAttn)    │  │
  │  │    Audio → Video: 单向                         │  │
  │  │    Audio → 3D: 单向                            │  │
  │  └──────────────────────────────────────────────┘  │
  │                         │                          │
  │  ┌──────────────────────────────────────────────┐  │
  │  │  Deep Blocks (8-27): INDEPENDENT              │  │
  │  │    [Spatial Attn] →                            │  │
  │  │    [Temporal Causal Attn] → [Dual FFN]        │  │
  │  │    各模态独立处理，专注自身任务                    │  │
  │  └──────────────────────────────────────────────┘  │
  └──────────────┬──────────────────┬──────────────────┘
                 │                  │
                 ▼                  ▼
          ┌─────────────┐   ┌──────────────┐
          │ Video Head  │   │ Struct Head  │
          │ → v_video   │   │ → v_struct   │
          └──────┬──────┘   └──────┬───────┘
                 │                  │
                 ▼                  ▼
          VAE Decode          3D latent s_t
          → Frame_t          (→ next frame)
```

### 7.5 为什么 3D 分支不会限制 Video 质量？

五重保护机制：

1. **Shallow Fusion Only**（OmniTalker 验证）：3D 仅在前 8/28 层参与融合，影响限于 structural layout 层面
2. **Gated Residual**：sigmoid gate 初始化为 0，训练初期完全不依赖 3D
3. **Asymmetric Token Count**（LTX-2 验证）：Video 256 tokens vs 3D 8 tokens，video 信息量占绝对主导
4. **独立 Noise Schedule**：video 和 3D 的 AdaLN 独立调制，即使 3D 全噪声 video 仍可正常
5. **独立 Loss Head + FFN**：3D 梯度不直接传播到 video FFN 权重

---

## 8. 数据集需求与使用方式

### 8.1 数据需求总览

本方案需要三类数据支撑不同训练阶段，核心约束是：每条样本必须同时包含**高质量面部视频**和**对齐的音频**，以便提取 video latent、3D structure latent、audio features 三元组。

| 训练阶段 | 数据类型 | 用途 | 最低数据量 | 推荐数据量 |
|----------|---------|------|-----------|-----------|
| Stage 1: 3D Latent Preparation | 面部视频 (有音频) | 提取 3D pseudo label | 50 小时 | 200+ 小时 |
| Stage 2: MOVA-Lite Causal Adaptation | 通用视频 | 保留 video DiT 能力 | 500 小时 | 2000+ 小时 |
| Stage 3: Multi-Modal DF Training | **面部视频 + 对齐音频** | 核心训练 | 200 小时 | 500+ 小时 |
| Evaluation | 面部视频 + 对齐音频 | 定量评估 | 标准 test set | 多 benchmark |

### 8.2 训练数据集详细信息

#### 8.2.1 核心训练集

**① HDTF（High-Definition Talking Face）⭐⭐⭐ 必选**

| 维度 | 信息 |
|------|------|
| 规模 | ~362 clips, ~15.8 小时, 300+ 个说话人 |
| 分辨率 | 720P / 1080P (裁剪后 512×512) |
| 音频 | ✅ 对齐语音，高质量 |
| 来源 | YouTube 公开演讲/访谈，CC BY 4.0 |
| 获取 | `github.com/MRzzm/HDTF` (YouTube URL + 时间戳) |
| 优势 | Talking head 领域最通用的 benchmark 数据集；几乎所有 SOTA 方法均使用；音频质量高、画面清晰 |
| 劣势 | 数据量较小（仅 ~16h），说话人多样性有限，多为英语 |
| 本方案用途 | **Stage 1 + Stage 3 核心训练 + 评估**。用于提取三元组 {video_latent, struct_latent, audio_feat}；Phase 0 验证首先在 HDTF 子集上进行 |

**② CelebV-HQ ⭐⭐⭐ 必选**

| 维度 | 信息 |
|------|------|
| 规模 | 35,666 clips, 15,653 identities, ~65 小时 |
| 分辨率 | ≥ 512×512 |
| 时长 | 3-20 秒/clip |
| 音频 | ✅ 包含原始音频 |
| 标注 | 83 个面部属性（40 外观 + 35 动作 + 8 情绪） |
| 来源 | YouTube，ECCV 2022 |
| 获取 | `github.com/CelebV-HQ/CelebV-HQ` |
| 优势 | 身份多样性极高（15K+ IDs）；丰富的属性标注可用于条件化训练；动态变化丰富 |
| 劣势 | 部分 clip 较短（3s）；非纯 talking head（包含非说话场景） |
| 本方案用途 | **Stage 3 主训练数据**，与 HDTF 联合提供多样化面部动态。利用属性标注可做条件化实验（如特定情绪/动作下的生成质量分析） |

**③ VFHQ（Video Face High Quality）⭐⭐ 推荐**

| 维度 | 信息 |
|------|------|
| 规模 | 16,000+ high-quality clips |
| 分辨率 | 远高于 VoxCeleb（高清以上），裁剪后 512×512 |
| 来源 | 访谈场景，多样化头部姿态和眼动 |
| 获取 | `liangbinxie.github.io/projects/vfhq/` |
| 优势 | 画质极高；头部姿态多样（不仅限于正面）；眼动丰富 |
| 劣势 | 非所有 clip 有清晰音频 |
| 本方案用途 | **Stage 3 训练数据（辅助）**，提升模型对大角度头部运动的鲁棒性 |

**④ TalkVid ⭐⭐ 推荐（大规模扩展）**

| 维度 | 信息 |
|------|------|
| 规模 | 7,729 说话人, **1,244 小时** HD/4K 视频 |
| 语言 | 15 种语言（英/中/阿拉伯/德/法/韩/日等） |
| 多样性 | 年龄 0-60+, 多种族, 性别均衡 |
| 来源 | YouTube, 2025 年发布 |
| 获取 | `github.com/FreedomIntelligence/TalkVid` (HuggingFace 元数据) |
| 优势 | 截至 2025 年最大的开源 talking head 数据集；种族/语言/年龄多样性极佳 |
| 子集 | TalkVid-Core: 160 小时高纯度均衡子集 |
| 本方案用途 | **Stage 3 大规模训练**（如 500h+ 需求时使用）。TalkVid-Core 160h 子集已足够覆盖主训练所需 |

#### 8.2.2 辅助/预训练数据

**⑤ VoxCeleb2（大规模预训练辅助）**

| 维度 | 信息 |
|------|------|
| 规模 | 6,112 说话人, 100 万+ utterances |
| 分辨率 | 中等（非高清为主） |
| 用途 | **仅用于 Stage 2 MOVA-Lite causal adaptation** 的大规模通用面部视频 |
| 注意 | 画质参差不齐；不建议直接用于 Stage 3 核心训练 |

**⑥ MEAD（Multi-view Emotional Audio-visual Dataset）**

| 维度 | 信息 |
|------|------|
| 规模 | 60 说话人（43 可用），8 种情绪 × 3 种强度 |
| 场景 | 实验室受控环境 |
| 用途 | **评估 + 情绪控制实验**。受控环境下验证不同情绪的生成质量 |

**⑦ 通用视频数据（Stage 2 专用）**

| 数据集 | 规模 | 用途 |
|--------|------|------|
| Panda-70M 子集 | 抽取 5-10K clips | Stage 2 causal adaptation，保留 video DiT 通用能力 |
| WebVid-10M 子集 | 抽取 5-10K clips | 同上，备选 |

### 8.3 数据集使用方案

#### 推荐配置（8 卡 A100，500h 目标）

```
Stage 1 (3D Latent Preparation):
  数据: HDTF (16h) + CelebV-HQ (65h) + VFHQ (部分)
  处理: 全量提取 3D pseudo label
  工具: LivePortrait + EMOCA

Stage 2 (MOVA-Lite Causal Adaptation):
  数据: Panda-70M 子集 (5K clips) + VoxCeleb2 子集 (10K clips)
  说明: 仅需让 MOVA-Lite 适应 causal mask，不需要面部特化数据

Stage 3 (核心训练):
  Primary:   HDTF (~16h) + CelebV-HQ (~65h)       = ~81h   ← 必选
  Secondary: VFHQ (~50h) + TalkVid-Core (~160h)    = ~210h  ← 推荐
  Total: ~291h
  
  如需扩展至 500h+:
  Extended:  TalkVid 全量 (~1244h) 中按质量筛选   ← 可选
```

#### 最小配置（4 卡 A100，学术验证）

```
Stage 1: HDTF (16h) 全量
Stage 2: 跳过 (直接在 HDTF 上 causal adaptation)
Stage 3: HDTF (16h) + CelebV-HQ (65h) = ~81h
评估:   HDTF test set + CelebV-HQ test set
```

### 8.4 数据预处理 Pipeline

所有训练数据需要经过统一的预处理 pipeline，生成可直接加载的 feature cache：

```
Raw Video + Audio
     │
     ├───────────────────────────────────────────────────────────────┐
     │                                                               │
     ▼                                                               ▼
┌─────────────────────┐                                   ┌──────────────────┐
│ 视频预处理           │                                   │ 音频预处理        │
│                     │                                   │                  │
│ 1. 人脸检测 & 跟踪  │                                   │ 1. 音频提取      │
│    (RetinaFace)     │                                   │    (ffmpeg)      │
│ 2. 人脸对齐 & 裁剪  │                                   │ 2. 重采样 16kHz  │
│    (512×512)        │                                   │ 3. VAD 静音检测  │
│ 3. 质量过滤         │                                   │    (Silero VAD)  │
│    - 模糊检测       │                                   │ 4. 音频-视频     │
│    - 遮挡检测       │                                   │    时间对齐验证   │
│    - 多人脸剔除     │                                   └────────┬─────────┘
│ 4. 统一 FPS (25fps) │                                            │
└────────┬────────────┘                                            │
         │                                                         │
         ▼                                                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Feature Pre-Extraction (离线，一次性)                  │
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ Video VAE Enc. ❄️ │  │ LivePortrait     │  │ HuBERT-Large ❄️      │  │
│  │ (per-frame)      │  │ Motion Ext. ❄️    │  │ (per-audio-chunk)   │  │
│  │                  │  │                  │  │                     │  │
│  │ → z_t            │  │ → s_t            │  │ → a_t               │  │
│  │ [C, H/8, W/8]   │  │ [D_s=128]        │  │ [1024] @ 50Hz       │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬──────────┘  │
│           │                     │                        │             │
│  ┌──────────────────┐  ┌──────────────────┐              │             │
│  │ EMOCA ❄️          │  │ DINOv2/CLIP ❄️   │              │             │
│  │ (FLAME pseudo)   │  │ (ref image feat) │              │             │
│  │ → flame_params   │  │ → ref_tokens     │              │             │
│  │ [159]            │  │ [N_ref, D]       │              │             │
│  └────────┬─────────┘  └────────┬─────────┘              │             │
│           │                     │                        │             │
└───────────┼─────────────────────┼────────────────────────┼─────────────┘
            │                     │                        │
            ▼                     ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  存储格式: 按 clip 组织的 .safetensors / .npz 文件                       │
│                                                                         │
│  clip_0001/                                                             │
│    ├── video_latents.safetensors    # [T, C, H/8, W/8], float16        │
│    ├── struct_latents.safetensors   # [T, D_s], float32                │
│    ├── audio_features.safetensors   # [T_audio, 1024], float16         │
│    ├── flame_params.safetensors     # [T, 159], float32                │
│    ├── ref_features.safetensors     # [N_ref, D], float16              │
│    └── metadata.json                # fps, duration, identity, etc.    │
│                                                                         │
│  预估存储需求 (500h, 25fps):                                            │
│    video_latents: ~3.2 TB (主要开销)                                    │
│    struct_latents: ~16 GB                                               │
│    audio_features: ~35 GB                                               │
│    flame_params:   ~28 GB                                               │
│    ref_features:   ~2 GB                                                │
│    Total: ~3.3 TB                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 预处理脚本要点

```python
class DataPreprocessor:
    """统一的数据预处理 Pipeline"""
    
    def __init__(self):
        self.face_detector = RetinaFace()          # 人脸检测
        self.video_vae = load_vae(frozen=True)      # Video VAE encoder
        self.liveportrait = load_liveportrait(frozen=True)  # 3D latent
        self.emoca = load_emoca(frozen=True)         # FLAME pseudo labels
        self.hubert = load_hubert_large(frozen=True) # Audio features
        self.ref_encoder = load_dinov2(frozen=True)  # Reference image
    
    def process_clip(self, video_path, audio_path):
        # 1. 视频预处理
        frames = extract_frames(video_path, target_fps=25)
        faces = self.face_detector.detect_and_crop(frames, size=512)
        
        # 2. 质量过滤
        if not quality_check(faces):  # 模糊/遮挡/多人脸
            return None
        
        # 3. Feature extraction (batch processing on GPU)
        video_latents = self.video_vae.encode(faces)          # [T, C, H/8, W/8]
        struct_latents = self.liveportrait.extract(faces)      # [T, D_s]
        flame_params = self.emoca.extract(faces)               # [T, 159]
        
        # 4. Audio features
        audio = load_audio(audio_path, sr=16000)
        audio_features = self.hubert.extract(audio)            # [T_audio, 1024]
        
        # 5. Reference image (first frame)
        ref_features = self.ref_encoder(faces[0])              # [N_ref, D]
        
        # 6. 时间对齐验证
        assert_temporal_alignment(video_latents, audio_features, fps=25)
        
        return {
            'video_latents': video_latents,
            'struct_latents': struct_latents,
            'audio_features': audio_features,
            'flame_params': flame_params,
            'ref_features': ref_features,
        }
```

### 8.5 数据质量控制

#### 过滤标准

```
硬过滤（直接丢弃）:
  - 分辨率 < 256×256（裁剪后人脸区域）
  - 检测到多张人脸（非单人 talking head）
  - 音频-视频时间偏移 > 100ms
  - 视频时长 < 2s
  - 人脸被遮挡 > 30%

软过滤（降低采样权重）:
  - 模糊度得分 < 阈值 (Laplacian variance)
  - 头部偏转角 > 45° (极端侧面)
  - 音频 SNR < 10dB
  - 唇动幅度极小（非说话状态）
```

#### 数据增强

```python
class TrainingAugmentation:
    """训练时的在线数据增强"""
    
    def __call__(self, sample):
        # 1. 时间增强
        if random.random() < 0.3:
            sample = random_temporal_crop(sample, min_len=16)
        if random.random() < 0.1:
            sample = temporal_speed_change(sample, factor=(0.8, 1.2))
        
        # 2. 空间增强 (仅 video, 不影响 3D latent)
        if random.random() < 0.2:
            sample['video'] = color_jitter(sample['video'], strength=0.1)
        if random.random() < 0.1:
            sample['video'] = random_horizontal_flip(sample['video'])
            sample['struct'] = flip_struct_latent(sample['struct'])
        
        # 3. Audio 增强
        if random.random() < 0.15:
            sample['audio'] = add_background_noise(sample['audio'], snr=15)
        
        # 4. Reference 增强
        if random.random() < 0.2:
            # 使用同一 clip 的随机帧作为 reference（而非总用第一帧）
            ref_idx = random.randint(0, len(sample['video']) - 1)
            sample['ref'] = extract_ref(sample['video'][ref_idx])
        
        return sample
```

### 8.6 DataLoader 设计

```python
class MultiModalTalkingHeadDataset(torch.utils.data.Dataset):
    """
    多模态训练数据集。
    加载预提取的 feature cache，训练时无需在线 encode。
    """
    def __init__(self, data_root, clip_length=32, split='train'):
        self.clips = load_clip_index(data_root, split)
        self.clip_length = clip_length
        self.augment = TrainingAugmentation()
    
    def __getitem__(self, idx):
        clip_dir = self.clips[idx]
        
        # 加载预提取 features
        video_lat = load_safetensors(clip_dir / 'video_latents.safetensors')
        struct_lat = load_safetensors(clip_dir / 'struct_latents.safetensors')
        audio_feat = load_safetensors(clip_dir / 'audio_features.safetensors')
        flame = load_safetensors(clip_dir / 'flame_params.safetensors')
        ref = load_safetensors(clip_dir / 'ref_features.safetensors')
        
        # 随机裁剪 clip_length 帧
        T = video_lat.shape[0]
        start = random.randint(0, max(0, T - self.clip_length))
        end = start + self.clip_length
        
        sample = {
            'video_latents':  video_lat[start:end],      # [clip_len, C, H/8, W/8]
            'struct_latents': struct_lat[start:end],      # [clip_len, D_s]
            'audio_features': align_audio(audio_feat, start, end, fps=25),
            'flame_params':   flame[start:end],           # [clip_len, 159]
            'ref_features':   ref,                        # [N_ref, D]
        }
        
        sample = self.augment(sample)
        return sample
    
    def __len__(self):
        return len(self.clips)


# DataLoader 配置
dataloader = DataLoader(
    dataset,
    batch_size=2,              # 2 clips per GPU
    num_workers=8,             # 预加载
    pin_memory=True,
    persistent_workers=True,
    shuffle=True,
    drop_last=True,
)
# Effective batch size: 2 clips × 32 frames × 8 GPUs = 512 frames/step
```

### 8.7 评估数据集与 Benchmark

| 数据集 | 用途 | Test Set 规模 | 评估指标 |
|--------|------|--------------|---------|
| **HDTF** | 主要 benchmark | 标准 test split (~50 clips) | FVD, FID, Sync-C, Sync-D, ACD |
| **CelebV-HQ** | 跨身份泛化 | 标准 test split (~5K clips) | FVD, FID, ACD |
| **VFHQ** | 跨姿态鲁棒性 | 标准 test split | FVD, FID |
| **VoxCeleb2** | 大规模多样性 | 随机 200 clips | FVD, FID |
| **MEAD** | 情绪控制 | 标准 test split | 情绪分类准确率, Sync-D |

#### 评估指标说明

| 指标 | 全称 | 衡量维度 | 目标 |
|------|------|---------|------|
| FVD | Fréchet Video Distance | 视频生成质量 + 时间一致性 | ↓ |
| FID | Fréchet Inception Distance | 单帧图像质量 | ↓ |
| ACD | Average Content Distance (ArcFace) | 身份保持 | ↓ |
| APD | Average Pose Distance | 姿态准确性 | ↓ |
| Sync-C | Lip Sync Confidence | 唇音同步（SyncNet 置信度）| ↑ |
| Sync-D | Lip Sync Distance | 唇音同步（SyncNet 距离）| ↓ |
| CSIM | Cosine Similarity (identity) | 身份一致性 | ↑ |

#### 长序列一致性专项评估

```
标准评估: 32 frames (~1.3s)
中等长度: 128 frames (~5s)
长序列:   256 frames (~10s)    ← 关键差异化评估
超长序列: 512 frames (~20s)    ← 验证 3D 先验的一致性优势

评估方式:
  1. ACD 随帧数增长的衰减曲线（衰减越慢 = 一致性越好）
  2. FVD 随序列长度的增长曲线
  3. 人工评估 (MOS): 5 分制，关注 identity drift / jitter / lip-sync
```

### 8.8 数据相关风险与缓解

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|---------|
| YouTube 视频下架导致数据不完整 | 高 | 中 | 尽早下载并本地备份；优先使用提供完整下载的数据集（CelebV-HQ, TalkVid HF） |
| LivePortrait 提取的 3D latent 质量不稳定 | 中 | 高 | 对比 EMOCA FLAME 参数做一致性过滤；丢弃偏差过大的帧 |
| 音频-视频对齐偏差 | 中 | 高 | 预处理阶段强制对齐验证 (< 40ms tolerance)；训练时加入时间抖动增强鲁棒性 |
| 训练数据种族/性别偏差 | 中 | 中 | 使用 TalkVid（多样性最佳）；分层采样确保均衡 |
| 存储空间不足 | 低 | 高 | 预提取 features 需 ~3.3TB/500h；可用 float16 压缩至 ~2TB |

---

## 9. 训练策略

### Stage 1: 3D Latent Space Preparation（1-2 天，单卡 A100）

```
目标: 准备训练数据的 3D pseudo labels
方法:
  1. 对 HDTF / VFHQ / CelebV-HQ 数据集:
     - LivePortrait motion extractor → implicit 3D latent
     - EMOCA → FLAME 参数（辅助监督）
  2. 存储为预计算 features
  3. 训练 Structure Token Projector（MLP）

数据量: ~50-100 小时面部视频
输出: {video_latents, struct_latents, audio_features} 三元组
```

### Stage 2: MOVA-Lite Causal Adaptation（3-5 天，8 卡 A100）

```
目标: 将 MOVA-Lite 的 video backbone 从 full attention 改为 causal
方法:
  - 使用缩减后的 MOVA-Lite (~2B) 模型
  - 修改 temporal attention mask: full → causal
  - 在通用视频数据上从零训练（缩减后无可用预训练权重）
  - 不引入 3D 或音频条件（仅训练基础 causal video generation 能力）

关键: 保留 MOVA 的 Aligned-RoPE；此阶段同步实现 KV-cache (参考 OVI)
Loss: 标准 flow matching loss
数据: Panda-70M 子集或 WebVid-10M 子集
```

### Stage 3: Multi-Modal Diffusion Forcing Training（7-10 天，8 卡 A100）⭐ 核心

```
目标: 在 MOVA-Lite 基础上添加 3D stream 和 Diffusion Forcing 训练
方法:
  - 冻结 MOVA-Lite video backbone 大部分参数
  - LoRA (rank=64) on MOVA-Lite spatial + temporal attention
  - 完全训练: Bridge CrossAttention layers (MOVA 原有的，重新初始化为 shallow-only)
  - 完全训练: Structure Token Projector + Structure Head
  - 完全训练: Audio projection + audio cross-attention
  - LoRA (rank=32) on FFN

关键改造:
  - 替换 MOVA noise sampling → Diffusion Forcing per-token sampling
  - 替换 MOVA audio backbone → 3D structure stream
  - 添加 shallow/deep split (移除 deep blocks 的 Bridge CrossAttn)
  - 添加 audio conditioning 支路

损失函数:
  L_total = L_video + 0.5·L_struct + 0.1·L_flame + 0.3·L_lip_sync

关键超参:
  σ_v ~ U(0, 1.0), σ_s ~ U(0, 0.7)
  Fusion depth: 8 layers (ablation: 4/8/12)
  Learning rate: 1e-4 (new modules), 5e-5 (LoRA)

数据: HDTF + VFHQ + CelebV-HQ (~500 小时)
Batch: 2 clips × 32 frames per GPU, effective batch = 16
```

### Stage 4（可选）: 推理加速

```
候选: Consistency Distillation / Progressive Distillation
目标: 20 steps → 4 steps
预期: ~22-33 FPS with KV-cache + FP16 + TensorRT
```

---

## 10. 推理 Pipeline 与延迟分析

### 9.1 标准推理（20 步去噪，学术评估）

```
Per-frame: Audio(~2ms) + DiT×20(~200ms) + VAE(~3ms) ≈ 205ms → ~5 FPS
```

### 9.2 加速推理（4 步去噪，Stage 4 后）

```
Per-frame: Audio(~2ms) + DiT×4(~40ms) + VAE(~3ms) ≈ 45ms → ~22 FPS
+ TensorRT: ~30ms → ~33 FPS
```

### 9.3 对比

| Method | Steps | FPS | 需要蒸馏? | 长序列一致性 |
|--------|-------|-----|----------|------------|
| Hallo2 | 50 | ~2 | 否 | 中等 |
| Self-Forcing | 4 | ~30 | 是(大量) | 中等 |
| CausVid | 4 | ~25 | 是(大量) | 较好 |
| **本方案 (20步)** | 20 | ~5 | **否** | **好 (3D 锚定)** |
| **本方案 (4步)** | 4 | ~22-33 | 轻量级 | **好 (3D 锚定)** |

---

## 11. 与现有方法的差异化定位

```
                     方法论创新性
                      ↑
              高      │  本方案 ★
  (Diffusion Forcing  │  (多模态联合 DF
   + 3D + MOVA-Lite)  │   + 隐式 3D 先验
                      │   + Shallow Fusion)
                      │
                      │  Geometry Forcing ◆
                      │  (3D align, 非联合生成)
              中      │
                      │  OVI ◆          OmniTalker ◆
                      │  (audio-video    (audio-video
                      │   联合生成)        双分支)
                      │  Self-Forcing     CausVid
                      │       ◆              ◆
              低      │  Hallo2/3
                      │       ◆
                      ├──────────────────────────────────→
                     低              中              高
                               Avatar 任务表现

核心差异化:
1. vs Geometry Forcing: 显式多模态联合生成 (非 feature alignment)，聚焦 avatar
2. vs OVI/LTX-2:       3D 结构先验 (非 audio)，Diffusion Forcing (非标准 flow matching)
3. vs OmniTalker:      3D awareness 增强一致性，Diffusion Forcing
4. vs Self-Forcing:    无需蒸馏 + 3D 一致性先验
5. vs Hallo2/3:        原生自回归 + 结构先验
```

---

## 12. 风险评估

### 关键实验验证点

| 需要验证的假设 | 验证方法 | Fallback |
|---------------|---------|---------|
| 多模态 Diffusion Forcing 收敛 | 200M 小模型 + HDTF 子集 | 退回串行方案 |
| 3D latent 提升长序列一致性 | Ablation: 有/无 3D stream | 强调其他贡献 |
| 双向 attention 优于单向 | Ablation: 双向/单向/无 | 退回单向 |
| Shallow fusion depth 最佳值 | Ablation: depth=4/8/12 | 全层融合 |
| LivePortrait 优于 FLAME | 直接对比 | 用 FLAME |
| MOVA-Lite base 可适配 Diffusion Forcing | Stage 2 验证 | 改用 Open-Sora/Latte |

### 风险矩阵

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 多模态 DF 训练不稳定 | **低** (5+ 论文验证) | 高 | 小模型先验证 |
| 推理速度不达标 | 高 | 中 | 先不 claim 实时 |
| MOVA-Lite→DF 改造复杂度 | 中 | 中 | 只改 noise sampling + mask |
| Fusion depth 敏感 | 中 | 低 | 系统 ablation |
| 计算资源不足 | 低 | 高 | 缩小模型 / 更激进 LoRA |

---

## 13. 实施路线图

### 10 周开发计划（基于 MOVA-Lite）

```
Week 1-2: MOVA 代码库理解 + 模型缩减 + 环境搭建
  - Clone MOVA + OVI repos
  - 理解 MOVA 的 Wan 2.2 DiT 架构、Bridge CrossAttention、Aligned-RoPE
  - 理解 MOVA 训练 loop (mova/trainer/) + FSDP 配置 + 数据管线
  - 缩减 Wan 2.2: 14B MoE → ~2B dense (减层/减维/去 MoE)
  - 训练脚本: LoRA → Full FT (去掉 peft wrapper)
  - 验证: 缩减后模型跑通 forward pass + FSDP 训练 loop 正常
  - 分析 OVI KV-Cache 实现，理解 Scaled-RoPE + KV-cache 机制
  - 研读 Diffusion Forcing 论文，理解 per-token noise sampling

Week 3-4: Stage 2 — Causal Adaptation
  - 修改 MOVA-Lite temporal attention: full → causal
  - 验证 causal video generation 质量
  - 集成 LivePortrait encoder，提取 3D latent 数据集
  - 预计算 HDTF/VFHQ/CelebV-HQ 的 {video, 3D, audio} features

Week 5-6: Stage 3 核心改造 — 3D Stream + Diffusion Forcing
  - 将 MOVA audio backbone 替换为 3D structure stream
  - 保留 MOVA Bridge CrossAttn 机制，只换输入端模态
  - 实现 Diffusion Forcing per-token noise sampling (替换 MOVA 统一 σ)
  - 实现 Dual AdaLN-Zero (σ_v, σ_s 分别调制)
  - 在 HDTF 子集验证 video+3D 双模态 DF 训练收敛

Week 7-8: Audio Conditioning + Shallow/Deep Split + KV-Cache
  - 集成 HuBERT encoder
  - 添加 audio cross-attention (仅 shallow blocks)
  - 实现 shallow fusion + deep specialization (移除 deep cross-attn)
  - 实现 gated residual + lip-sync loss
  - 新增 KV-Cache (参考 OVI 实现)
  - 完整 Stage 3 训练

Week 9-10: 评估 + Ablation
  - 定量评估: FVD, FID, ACD, APD, Sync-C/D
  - Ablation: 有/无 3D stream
  - Ablation: 双向/单向 attention
  - Ablation: fusion depth (4/8/12)
  - Ablation: σ_s range (0.5/0.7/1.0)
  - 长序列一致性: 256, 512 frames
```

### 总时间: ~9-11 周
### 总 GPU 消耗: ~6000-10000 A100 GPU-hours

---

## 14. 论文 Framing

### Title 候选

- **StructFlow: Multi-Modal Diffusion Forcing with Implicit 3D Priors for Autoregressive Talking Head Generation**
- **DualForce: Joint 3D-Video Diffusion Forcing for Consistent Real-Time Avatar Generation**
- **Avatar-DF: Diffusion Forcing Meets 3D Structure for Native Autoregressive Talking Head Synthesis**

### Abstract 草案

> We present [Name], a framework for autoregressive talking head video generation that jointly generates implicit 3D structure latents and video latents through multi-modal Diffusion Forcing. Unlike existing approaches that require large-scale distillation (Self-Forcing, CausVid) for autoregressive diffusion, our method trains natively without distillation by leveraging per-token independent noise levels. Building upon a scaled DiT backbone with MOVA-style Bridge CrossAttention, we introduce a shallow-fusion design where 3D structure tokens and video tokens exchange information via bidirectional cross-attention in early layers, while deeper layers specialize independently. An asymmetric noise schedule (σ_struct ∈ [0, 0.7] vs σ_video ∈ [0, 1.0]) enables implicit "structure-first" generation without explicit two-stage pipelines. Audio drives temporal dynamics through dual-path conditioning (audio→video and audio→3D→video). At inference time, no explicit 3D input is required — both streams are generated autoregressively from audio and a reference image via KV-Cache. Experiments on [benchmarks] demonstrate state-of-the-art long-sequence consistency while maintaining competitive generation quality, all without distillation overhead.

### 投稿目标

- **首选**: NeurIPS 2026 (DDL: ~May 2026) 或 CVPR 2027
- **备选**: ICLR 2027, ECCV 2026

### 核心 Contributions

1. **首次将 Diffusion Forcing 扩展到多模态视频生成**，实现 video + 3D latent 联合自回归生成
2. **Shallow Fusion + Deep Specialization 架构**（借鉴 MOVA Bridge CrossAttn / OmniTalker），3D 浅层结构引导 + 深层独立特化
3. **Asymmetric Noise Schedule**，隐式"先结构后外观"生成顺序
4. **在 avatar 场景验证**，同时提升长序列一致性和消除蒸馏依赖

---

## 15. 关键参考文献

### 14.1 核心方法论

| Paper | 关键技术 | 关系 | 重要性 |
|-------|---------|------|--------|
| **Diffusion Forcing** (Chen et al., 2024, ICML Oral) | Per-token 独立噪声 | 训练范式基础 | ⭐⭐⭐ |
| **Geometry Forcing** (NeurIPS 2025) | 3D + video diffusion | 最直接验证工作 | ⭐⭐⭐ |
| **OVI** (Character.AI, 2025) | Twin-backbone cross-attn | **KV-Cache 参考** | ⭐⭐⭐ |
| **MOVA** (Zong et al., 2024) | 多模态双向 attention | **Base codebase (MOVA-Lite)** | ⭐⭐⭐ |
| **LivePortrait** (Guo et al., 2024) | 隐式 3D keypoints | 3D encoder 来源 | ⭐⭐⭐ |

### 14.2 多模态融合

| Paper | 关键技术 | 关系 |
|-------|---------|------|
| **LTX-2** (Lightricks, 2026) | Asymmetric dual-stream + AdaLN | Dual AdaLN 参考 |
| **OmniTalker** (NeurIPS 2025) | Shallow fusion + deep specialization | 融合深度策略 |
| **UniVerse-1** (2025) | Stitching of Experts | 替代融合参考 |
| **MIDAS** (2025) | LLM + multimodal tokens | Avatar multimodal 参考 |
| **Transfusion** (Meta, 2024) | Text+Image 联合生成 | 多模态联合参考 |

### 14.3 自回归视频扩散

| Paper | 关键技术 | 关系 |
|-------|---------|------|
| **Self-Forcing** (Yin et al., 2024) | AR 视频扩散 + self-correction | 对标方法 |
| **CausVid** (Yin et al., 2025) | Causal video generation | 对标方法 |
| **Open-Sora** (Zheng et al., 2024) | STDiT, 开源 | 备选 base |
| **CogVideoX** (Yang et al., 2024) | 3D causal VAE | 备选 base |

### 14.4 Audio-Driven Avatar

| Paper | 关键技术 | 关系 |
|-------|---------|------|
| **Hallo2/3** (2024-2025) | Audio-driven portrait diffusion | Audio conditioning 参考 |
| **Live Avatar** (Dec 2025) | TPP + RSFM 实时系统 | 系统优化参考 |
| **GAIA** (Tencent, 2024) | Global/local motion 解耦 | Motion 分解参考 |
| **EMOPortraits** (NeurIPS 2024) | Cross-driving expression | 备选 3D latent |

### 14.5 3D-Aware 表征

| Paper | 关键技术 | 关系 |
|-------|---------|------|
| **FantasyWorld** (2025) | 3D Gaussian video diffusion | 3D-aware 参考 |
| **WorldForge** (2025) | Geometry-guided latent | World model 参考 |
| **FLAME** (Li et al., 2017) | Parametric face model | 辅助监督基础 |
| **EMOCA** (Danecek et al., 2022) | Emotion-aware 3D recon | Pseudo label 工具 |

---

## Appendix A: 版本修订记录

### A.1 v3.3 → v3.4 变更

1. **Section 7 全面重写: Base Codebase 从 OVI 改为 MOVA-Lite**
   - 重新评估发现：OVI 完全没有训练代码是致命缺陷，从零写分布式训练 pipeline 需 2-4 周
   - MOVA 的 LoRA 训练脚本虽非预训练脚本，但包含完整训练基础设施（training loop、FSDP、data pipeline、checkpoint）
   - 新策略："MOVA-Lite" — 以 MOVA codebase 为主体，缩减 Wan 2.2 到 ~2B dense，复用 Bridge CrossAttn
   - KV-Cache 参考 OVI 实现方案后加，比从零写训练 pipeline 简单得多
   - 开发周期从 ~5 周缩短到 ~3 周
2. **Section 7.2 重写**: 改造清单从 "OVI → 本方案" 改为 "MOVA → MOVA-Lite"，新增"改动 0: 模型缩减"
3. **Section 7.3 更新**: 架构规格 base 从 "OVI Video Backbone" 改为 "MOVA-Lite Video Backbone (Wan 2.2 缩减版)"
4. **全文 OVI 引用同步更新**: 所有代码注释、流程图、风险表、开发计划、Abstract 中的 "OVI-based" → "MOVA-Lite"
5. **Section 13 开发计划重写**: 基于 MOVA-Lite 的 10 周开发计划，Week 1-2 聚焦模型缩减而非 OVI 理解

### A.2 v3.2 → v3.3 变更

1. **Section 7.1 重写: Base Codebase 选型扩展为 OVI vs MOVA vs LTX-2 三方对比**
   - 新增 MOVA（OpenMOSS, 32B MoE）和 LTX-2（Lightricks, 19B）的完整分析
   - 明确 MOVA 训练基础设施的价值（LoRA 训练、数据 Pipeline、FSDP 分布式）
   - 分析 LoRA 微调无法实现本方案核心创新的 5 个原因
   - 推荐 "MOVA-Informed OVI" Hybrid 策略：MOVA 训练基础设施 + OVI 规模架构
   - 新增 Section 7.1.1: 具体代码移植路径（3 周时间线）
2. **新增 Section 3.4: 项目架构图**（参考 MOVA 风格设计）
3. **章节编号调整**：相关引用同步更新

### A.3 v3.1 → v3.2 变更

1. **新增 Section 8: 数据集需求与使用方式**：完整的数据集选型、使用方案、预处理 pipeline、质量控制、评估 benchmark 说明
   - 训练数据：HDTF（必选）、CelebV-HQ（必选）、VFHQ（推荐）、TalkVid（大规模扩展）
   - 辅助数据：VoxCeleb2（Stage 2）、MEAD（评估）、Panda-70M 子集（Stage 2）
   - 预处理 pipeline：从原始视频到预提取 feature cache 的完整流程
   - 数据增强策略：时间/空间/音频/Reference 四维增强
   - DataLoader 设计：多模态 safetensors 加载 + 在线增强
   - 评估 benchmark：7 项指标 + 长序列专项评估方案
   - 存储需求预估：~3.3TB / 500h
2. **章节编号调整**：原 Section 8-14 顺延为 Section 9-15

### A.4 v3.0 → v3.1 变更

1. **文档结构重组**：版本记录移至附录，前置背景/问题/动机/创新的清晰介绍
2. **新增训练与推理流程图**：科研级别的 ASCII 可视化（Section 3），展示所有模态信息从 input 到 output 的完整流向
3. **Base Codebase 选定 OVI**：初始版本选择 OVI（character-ai/Ovi）作为 base，因 OVI 已有 video DiT + cross-attention + KV-cache（v3.4 已修正为 MOVA-Lite）
4. **详细的多模态 Diffusion Forcing + Causal 设计**：新增 Section 6，含 attention mask 设计、KV-cache 实现、Dual AdaLN、per-token noise sampling 等实现级细节
5. **全文同步更新**：所有涉及 base model、改造路径、训练策略的描述统一调整

### A.5 v2.0 → v3.0 变更

1. **Related Work 大幅扩展**：新增 Geometry Forcing、LTX-2、OVI、OmniTalker、UniVerse-1、MIDAS 等 6 项工作
2. **架构优化**：借鉴 OVI blockwise cross-attention、OmniTalker shallow/deep split、LTX-2 asymmetric design、MOVA Bridge CrossAttn
3. **Audio 机制细化**：明确 conditioning-only，HuBERT + Aligned-RoPE + lip-sync loss
4. **风险评估更新**：多模态融合风险降低（5+ 论文验证）

### A.6 v1.0 → v2.0 变更

1. **3D 表征重构**：FLAME → 隐式 3D latent（LivePortrait-style）
2. **训练范式**：明确 Diffusion Forcing（无蒸馏）
3. **融合架构**：串行 pipeline → MOVA 式双向 attention

---

*Document Version: v3.4*
*Date: 2025-02-20*
*Base Codebase: MOVA-Lite (OpenMOSS/MOVA → Wan 2.2 缩减版 ~2B) | KV-Cache 参考: OVI (character-ai/Ovi)*
