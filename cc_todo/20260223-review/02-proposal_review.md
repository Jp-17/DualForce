# proposal.md 方案设计审查报告

> 审查日期: 2026-02-23
> 审查对象: cc_core_files/proposal.md (v3.4)
> 审查方法: 结合 code_research.md 审查结果 + MOVA 源码验证 + 学术可行性分析
> 审查结论: 方案整体设计合理、方向正确，但存在若干技术细节不精确、可行性风险低估、以及架构设计中的逻辑待商榷之处

---

## 一、总体评价

proposal.md 是一份高水平的研究方案文档。它的核心创新——**将 Diffusion Forcing 扩展到 video + 3D latent 联合自回归生成**——是一个有前景的研究方向。文档对相关工作的调研全面（Geometry Forcing、OVI、LTX-2、OmniTalker 等），趋势判断准确，差异化定位清晰。

但作为一份即将进入实施阶段的方案，仍然有一些需要深入讨论的问题。

---

## 二、核心创新的可行性分析

### 2.1 多模态 Diffusion Forcing ⚠️ 理论可行但实证不足

**proposal 的声明**: "首次将 Diffusion Forcing 扩展到多模态视频生成"

**审查分析**:

Diffusion Forcing (Chen et al., 2024, ICML Oral) 的原始工作处理的是**单模态**的 token 序列（如视频帧序列），通过给每个 token 独立的噪声水平实现自回归生成。proposal 将其扩展到两种模态（video + 3D struct）的联合去噪，每帧存在两组 tokens 各有独立噪声。

**逻辑上的关键问题**:

1. **训练信号的信噪比**: 当 σ_v(t) 很大（接近 1.0）而 σ_s(t) 很小（接近 0）时，video tokens 几乎是纯噪声，而 3D tokens 几乎是干净的。此时模型在 shallow fusion blocks 中做 cross-attention，video tokens（纯噪声）attend to 3D tokens（干净信号），这在理论上是合理的——3D tokens 可以引导 video 去噪。但反过来，3D tokens attend to 纯噪声的 video tokens 能获得什么有用信息？这种情况下双向 attention 中 video→3D 方向的梯度信号质量很差。

2. **与 MOVA 的差异被低估**: MOVA 的 FlowMatchPairScheduler 虽然支持 `dual_sigma_shift`（不同模态可以有不同的 sigma 分布），但它是在**同一时间步内的全局 sigma**，而非 **per-frame 独立 sigma**。Diffusion Forcing 的核心是 per-token 独立噪声，这需要根本性改变训练 loop 的噪声采样逻辑和 loss 计算方式。proposal 在 Section 6.1.4 给出了伪代码，但对训练稳定性的讨论不够充分。

3. **Geometry Forcing 的参考价值有限**: proposal 引用 Geometry Forcing (NeurIPS 2025) 作为 "3D + video diffusion 融合" 的验证。但 Geometry Forcing 使用的是 **feature alignment loss**（将 video diffusion 的中间特征与 VGGT 几何模型对齐），而非在 attention 中直接做跨模态交互。两者的技术路线差异很大，Geometry Forcing 的成功不能直接证明 attention-level 的多模态 Diffusion Forcing 也会成功。

**建议**:
- 在 Phase 4 训练前，先在一个极小模型（如 dim=256, layers=4）上验证多模态 DF 的收敛性
- 考虑加入 video→3D 方向 cross-attention 的 adaptive weighting，当 video tokens 噪声很大时降低其作为 KV 的权重

---

### 2.2 Asymmetric Noise Schedule (σ_s ∈ [0, 0.7]) ⚠️ 隐含假设需验证

**proposal 的声明**: σ_s 的上界为 0.7，使 3D tokens "总是比 video tokens 更干净"，创造隐式的 "先结构后外观" 生成顺序。

**审查分析**:

1. **"总是更干净" 不成立**: 由于两者都是独立均匀采样，σ_v ~ U(0,1.0) 和 σ_s ~ U(0,0.7)，存在大量情况下 σ_v < σ_s。例如 σ_v=0.1, σ_s=0.6 时，video 反而更干净。正确的说法是 "**平均而言/期望上** 3D tokens 更干净"。

2. **0.7 这个阈值缺乏依据**: 为什么是 0.7 而非 0.5 或 0.9？proposal 没有给出理论推导或实验依据。这个值直接影响 3D stream 被训练去处理的噪声范围，过低可能导致 3D stream 无法处理高噪声情况（推理时初始化为纯噪声时可能表现差），过高则"先结构"效果减弱。

3. **推理时的问题**: 推理时 3D stream 从纯噪声（σ_s = 1.0）开始，但训练时从未见过 σ_s > 0.7 的情况。这个 **训练-推理分布不匹配** 是一个严重的隐患。proposal 的 ablation 计划中列出了 σ_s=0.5 和 σ_s=1.0 的对比，这是好的，但对这个 train-test mismatch 问题没有讨论。

**建议**:
- 考虑使用截断正态分布而非均匀分布，中心在较低值但尾巴可以覆盖到 1.0
- 或者在推理时使用 σ_s_max=0.7 作为去噪起点而非 1.0，即 3D stream 从 70% 噪声而非 100% 噪声开始去噪
- 在文档中明确讨论 train-test mismatch 并给出缓解方案

---

### 2.3 Shallow Fusion + Deep Specialization ✅ 设计合理

这是 proposal 中设计最合理的部分。理由：

1. OmniTalker (NeurIPS 2025) 已经验证了 early fusion + late specialization 优于 full fusion
2. MOVA checkpoint 实际使用 "full" 策略（所有层都做 cross-attention），但 DualForce 的场景不同（3D latent vs audio latent，token 数量差异大），shallow fusion 更合适
3. Gated residual (gate init=0) 的设计来自 Flamingo 等成熟工作，确保训练初期不破坏预训练表征

**唯一值得商榷的点**:

proposal 选择 8/28 层做 fusion（约 28.6%），而 OmniTalker 验证的可能是不同的比例。MOVA 的 `shallow_focus` 策略实际代码中是 "first ~1/3 of layers (up to 10 layers)"。因此 8/20 = 40% 可能比 "shallow" 略多。建议在 ablation 中系统测试 4/8/12 层。

---

## 三、架构设计审查

### 3.1 模型缩减: 5120→1536 ⚠️ 激进但有风险

**实际情况**: MOVA-360p 的 video DiT 是 dim=5120, layers=40（约 14B 参数），DualForce 计划缩减到 dim=1536, layers=20（约 617M 参数）。

**分析**:

1. **参数量下降约 96%**（14B → ~0.6B for video backbone alone）。这不是"缩减"，而是几乎从零构建一个新模型。
2. **缩减后无法使用 MOVA 预训练权重**: dim 和 layers 都变了，权重无法迁移。这意味着必须从随机初始化开始训练。
3. **风险**: 一个 ~0.6B 的 video DiT 的生成能力是否足够？作为参考，OVI 的 video backbone 约 2B。dim=1536 对应 head_dim=128 × 12 heads，attention 的表达能力可能受限。

**但也有合理性**:
- 8 卡 A100 的算力限制确实无法训练 14B 模型
- 目标是 talking head（有限域），不是通用视频生成
- OVI 在 2B 模型上已经展现了不错的 video+audio 生成质量

**建议**:
- 考虑更温和的缩减方案，如 dim=2048, layers=24（约 1.5B），仍在 8 卡可训练范围内
- 或者在 Stage 2 使用知识蒸馏从 MOVA-14B 到 MOVA-Lite，而非完全从零训练

---

### 3.2 Audio 从生成模态变为 Conditioning-only ✅ 简化合理

**proposal 的决策**: MOVA 中 audio 是被生成的模态（有 audio DiT + audio VAE），DualForce 将 audio 改为 conditioning-only（使用 HuBERT 编码后作为 cross-attention 的 KV）。

**分析**: 这是一个正确的简化。在 talking head 任务中，audio 是**给定输入**而非需要生成的。将 audio 从被去噪的模态转变为纯 conditioning 信号，极大简化了训练（不需要 audio 的 flow matching loss）。

**需要注意的技术细节**:
- MOVA 的 bridge 对 audio 和 video 做双向 cross-attention（audio→video 和 video→audio），DualForce 中 audio 变为 conditioning-only 后，audio→video 是合理的，但不再有 video→audio。这改变了 bridge 的信息流模式。
- HuBERT 的 50Hz 输出（每秒 50 个 token）与 video 的 25fps（每秒 25 帧）不是整数倍关系。proposal 提到 "Temporal pooling (4:1 → align with video fps ~12.5fps)"，但 50/4=12.5fps ≠ 25fps，时间对齐似乎有误。如果 video 是 25fps，audio pooling 应该是 2:1（50/2=25fps），或者每帧获取 2 个 audio token。

---

### 3.3 3D Latent 的维度设计 ⚠️ 需要更深思考

**proposal 的设计**: LivePortrait motion feature (~128 dim) → StructTokenProjector → 8 tokens × D_model per frame

**分析**:

1. **信息瓶颈**: 128 维的 motion feature 要扩展为 8 × 1536 = 12288 维的 token 序列。这是一个 96x 的放大。StructTokenProjector 本质上是一个 MLP，它能否从 128 维中可靠地生成 12288 维的有意义表征？

2. **实际实现**: 从 branch 上的 `wan_struct_dit.py` 代码来看，StructTokenProjector 支持 `n_tokens_per_frame=1`（简单投影）和 `>1`（扩展投影）两种模式。如果 n_tokens_per_frame=1，则每帧只有 1 个 1536 维的 token 参与 attention，而 video 有 256 个 tokens/frame（按 proposal 估计），信息量极度不对称。

3. **对比 MOVA audio**: MOVA 的 audio DiT 接收 128 维的 DAC latent，通过 patch_size=[1] 的 Conv1d 投影到 1536 维。每个 audio token 对应一段音频。DualForce 的 3D struct 只是一帧的 128 维向量，信息密度远低于 audio。

**建议**:
- 认真考虑 n_tokens_per_frame 的选择。1 个 token/frame 可能信息不足，8 个可能过多（128 维展不开）。4 tokens/frame 或 2 tokens/frame 可能更合适。
- 或者考虑使用 LivePortrait 的更高维表征（如中间层特征），而非最终的 128 维 motion vector。

---

### 3.4 KV-Cache 设计 ✅ 方向正确但细节需验证

proposal 中 KV-Cache 的设计参考 OVI，每层每模态独立缓存。这在概念上是正确的，但有两个潜在问题：

1. **显存增长**: 自回归生成长序列时，KV-cache 随帧数线性增长。对于 dim=1536, heads=12, 假设 video 256 tokens/frame + struct 8 tokens/frame + audio 4 tokens/frame，每层缓存 268 tokens × 2 (K, V) × 1536 × 2 bytes (fp16) ≈ 1.6MB/层/帧。20 层 × 512 帧 = ~16GB 的 KV-cache。这在 A100 80GB 上可行但会压缩 batch size。

2. **推理时 KV-cache 的噪声状态问题**: 自回归推理中，当前帧需要 K 步去噪。在这 K 步中，当前帧的 KV 会随着去噪进展而变化，但 **历史帧的 KV-cache 应该保存的是去噪完成后的 clean 状态**。这意味着每帧去噪完成后才将最终 KV 追加到 cache，而非每一步去噪都更新 cache。proposal 的伪代码（Section 6.2.2-6.2.3）对此描述不够清晰。

---

## 四、训练策略审查

### 4.1 Stage 2: Causal Adaptation ⚠️ 必要性和方法需讨论

**proposal 的计划**: 先在通用视频数据上训练 MOVA-Lite 的 causal attention 能力（3-5 天，8 卡）。

**分析**:

1. **从零训练 video DiT 的基础视频生成能力需要的数据量和计算量远超 3-5 天**: 即使是小模型（~0.6B），学会基本的视频生成也需要大量数据。Panda-70M 的 5K clips 子集可能远远不够。
2. **Stage 2 的数据选择矛盾**: 文档一方面说 "仅需让 MOVA-Lite 适应 causal mask，不需要面部特化数据"，但另一方面如果模型是从零训练的（没有预训练权重），那 Stage 2 实际上是在**训练一个新的视频生成模型**，而非"适应 causal mask"。
3. **考虑跳过 Stage 2**: 如果最终目标只是 talking head，是否可以直接在 talking head 数据上从零训练？节省 Stage 2 的时间和数据准备成本。

---

### 4.2 Loss 权重 ⚠️ 缺乏理论/实验依据

```
L_total = L_video + 0.5·L_struct + 0.1·L_flame + 0.3·L_lip_sync
```

这些权重（1.0, 0.5, 0.1, 0.3）看起来是经验设定，没有给出选择依据。特别是：

- **L_struct 的权重 0.5**: 如果 3D struct 是核心创新，为什么它的权重只有 video 的一半？这可能导致 3D stream 训练不充分。
- **L_flame 的权重 0.1**: 如果 FLAME 是辅助监督（"decaying weight"），0.1 可能合适，但 decay schedule 未定义。
- **L_lip_sync 的权重 0.3**: Lip-sync contrastive loss 的 0.3 相对较高。如果 lip-sync loss 的尺度（scale）与 MSE loss 差异很大，可能需要先归一化。

**建议**: 对每个 loss component 的典型数值范围做预实验，然后基于梯度大小比例来设定权重，或使用自动 loss weighting (如 uncertainty weighting, GradNorm 等)。

---

### 4.3 训练参数量 ⚠️ 估算需修正

proposal Section 7.3 写 "实际训练参数 (LoRA rank=64 on video backbone + all new modules): ~380M"

但 plan.md 和 DualForce config 都明确了 `use_lora=False`（全参数训练）。如果是全参数训练，实际训练的参数量约为：
- Video DiT: ~617M
- Struct DiT: ~617M (与 video 同配置)
- Bridge: ~200M
- Audio conditioning: ~34M
- Other: ~9M
- Total trainable: **~1.5B**

这比 proposal 估计的 380M 大约 4x。全参数训练 1.5B 模型在 8 卡 A100 上是可行的（每卡 80GB），但需要激进的 gradient checkpointing 和 FSDP，训练速度会受影响。

---

## 五、Related Work 和差异化定位审查

### 5.1 Related Work ✅ 全面且准确

proposal 对 2024-2025 年主要工作的覆盖非常全面。特别值得肯定的是：
- Geometry Forcing 的精准定位（最直接的验证工作）
- OVI 的 KV-Cache 参考价值
- OmniTalker 的 shallow/deep split 洞察
- LTX-2 的 asymmetric design 参考

### 5.2 差异化定位 ⚠️ 部分声明可能过于乐观

1. **"首次将 Diffusion Forcing 扩展到多模态视频生成"**: 需要小心这个 claim。Transfusion (Meta, 2024) 已经在 text+image 上做了联合生成，虽然不是严格的 Diffusion Forcing，但相关。而且 MOVA 本身的 FlowMatchPairScheduler 已经支持 dual sigma shift，从某种意义上说已经接近 per-modality 的独立噪声。

2. **"3D 锚定" 带来的长序列一致性**: 这是方案的核心差异化声明，但目前缺乏直接的实验验证。3D latent 提供的 structural anchor 效果需要在长序列（256+ 帧）上与无 3D 的 baseline 做对比才能确认。

3. **与 Self-Forcing/CausVid 的对比**: proposal 正确指出了 "无需蒸馏" 这一优势，但在视频质量和速度上，Self-Forcing 和 CausVid 有大量工程优化（consistency training, progressive distillation），DualForce 在这些维度上可能不占优势。

---

## 六、技术细节问题

### 6.1 Audio Temporal Alignment 算错

Section 6.3.2 写:
```
Raw audio (16kHz) → HuBERT-Large (frozen, 60K hours pretrained)
  → [B, T_audio, 1024] features (50Hz)
  → Temporal pooling (4:1 → align with video fps ~12.5fps)
```

如果 video fps = 25fps，那么 4:1 pooling 后的 audio fps = 50/4 = 12.5fps ≠ 25fps。这会导致 audio tokens 与 video frames 不对齐。正确的做法应该是 2:1 pooling（50/2 = 25fps，每帧 1 个 audio token）或者不做 pooling 而是每帧对应 2 个 audio tokens。

proposal 后面又写 "N_a = 4 tokens per video frame"，如果 50Hz/25fps = 2 tokens/frame 才对。除非 N_a=4 是指扩展后的 tokens 数。这里的对齐逻辑需要理清。

### 6.2 Lip-sync Loss 的 mouth_mask 来源未明确

```python
def lip_sync_loss(video_features, audio_features, mouth_mask):
    mouth_feat = extract_mouth_region(video_features, mouth_mask)
```

训练时 `mouth_mask` 从哪里来？如果来自 video latent space（VAE 编码后），那需要在 latent space 中有对应的 mouth region 信息，这不是 trivial 的——VAE 编码后的 spatial dimensions 被压缩了 8x，mouth 区域可能只有几个 pixel。如果来自 pixel space，则需要额外的 face landmark detection。这个实现细节对 lip-sync loss 的有效性至关重要。

### 6.3 DiT Block 中 Spatial vs Temporal Attention 的顺序

proposal Section 3.3 的 Shallow Block 内部流程是:
```
Spatial Self-Attn → Cross-Modal Attn → Temporal Causal Attn → FFN
```

但 MOVA 的实际 DiT Block 结构（wan_video_dit.py 中的 DiTBlock）是:
```
Self-Attention → Cross-Attention (text) → FFN
```

MOVA 的 self-attention 同时处理 spatial 和 temporal dimensions（3D attention over all tokens in the sequence），不是分离的 spatial-first-then-temporal 模式。proposal 的分离设计是一个新的架构选择，需要明确这是有意为之还是对 MOVA 架构的误解。

如果是有意分离，需要确认：
- 分离后每个 attention 的 token 数量是多少？Spatial: H'×W' per frame? Temporal: T frames per spatial position?
- 这种分离是否与 MOVA 的 3D RoPE 兼容？

---

## 七、文档风格和一致性

### 7.1 版本号管理 ✅

v3.4 的版本记录详尽，修改历史清晰。

### 7.2 自洽性 ⚠️

- Section 7 (base codebase 选型) 中引用了 "48 layers" 的说法（"层数 48→28"），但实际 checkpoint 是 40 层。虽然 plan.md 已修正为 "40→20"，但 proposal 自身的 Section 7.2 仍写 "层数 48→28"。
- Section 7.3 写 "28 blocks"，与 plan.md 的 "20 blocks" 矛盾。

---

## 八、总结与建议

### 核心优势
1. 创新方向明确，将三个前沿方向统一（3D-aware + multi-modal fusion + Diffusion Forcing）
2. 工程基础选择合理（基于 MOVA 训练基础设施）
3. 降级方案（fallback）设计周全

### 核心风险
1. 多模态 Diffusion Forcing 的训练稳定性未经验证
2. σ_s 的 train-test distribution mismatch
3. 从零训练 ~1.5B 模型的数据量和计算量可能不足
4. 3D latent (128 dim) 的信息瓶颈

### 建议优先级

| 优先级 | 建议 | 影响 |
|--------|------|------|
| P0 | 修正 audio temporal alignment (4:1 → 2:1 pooling) | 直接影响训练正确性 |
| P0 | 明确 σ_s 的 train-test mismatch 缓解策略 | 直接影响推理质量 |
| P1 | 统一文档中的层数 (48→40, 28→20) | 避免实施时混淆 |
| P1 | 考虑更温和的模型缩减方案 (如 dim=2048) | 影响生成质量 |
| P2 | 验证 lip-sync loss 在 latent space 中的可行性 | 影响唇音同步质量 |
| P2 | 在 proposal 中讨论 spatial-temporal attention 分离的设计动机 | 影响架构合理性 |
| P3 | 对 loss 权重做自动 weighting 实验 | 影响训练效率 |
