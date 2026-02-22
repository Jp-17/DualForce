# plan.md 执行计划审查报告

> 审查日期: 2026-02-23
> 审查对象: cc_core_files/plan.md (v1.0)
> 审查方法: 结合 code_research.md 审查 + proposal.md 审查 + dataset.md 内容 + 仓库实际状态 + git 历史分析
> 审查结论: plan.md 在修正 proposal 错误方面做了有价值的工作，Phase 分解合理，但对项目实际状态的描述与仓库现实不一致，部分 Phase 的工作量低估，且与 proposal 存在若干未对齐之处

---

## 一、总体评价

plan.md 是一份实操导向的执行计划。相比 proposal.md 的学术方案设计，plan.md 更关注"怎么做"——它列出了具体的文件修改清单、Phase 分解、milestone 定义。文档的一个重要贡献是 **Section 3: Proposal Corrections**，基于 checkpoint 实际配置修正了 proposal 中的多处错误。

但 plan.md 也存在一些问题：

1. **Phase 1 标记为 "✅ CODE COMPLETE" 但代码实际不在 main 分支** —— 代码在 `20260221-cc-1st` 分支上，且 plan 说明了 "如果显示有执行的部分，可能是之前被舍弃的执行部分"。这意味着 Phase 1 的代码质量需要重新审视。
2. **部分 Phase 的工作量被低估**，特别是 Phase 2（causal pretraining from scratch）。
3. **plan 与 proposal 之间存在未对齐的参数值**。

---

## 二、Section 3: Proposal Corrections 审查

plan.md 的 Proposal Corrections 是其最有价值的部分。逐条审查：

### ✅ 正确的修正

| 修正项 | plan 修正 | 审查确认 |
|--------|----------|---------|
| "Wan 2.2 14B, 48 layers, MoE" → "40 layers, dim=5120, no MoE" | ✅ 与 checkpoint 一致 | **重要修正**。proposal 中多处写 "48 layers"、"MoE" 是错误的，MOVA-360p 实际是 40 层的 dense model |
| "MOVA has video_dit + video_dit_2" | ✅ 实际存在两个 video DiT | 正确发现，两个 DiT 的 config.json 完全相同 |
| FlowMatchPairScheduler 支持 independent sigma | ✅ 代码确认 | `dual_sigma_shift` 策略已存在 |
| Bridge 使用 "full" 策略 | ✅ checkpoint 确认 `interaction_strategy="full"` | 重要发现，影响 DualForce 的 shallow_focus 决策 |
| patch_size=(1,2,2) 而非 (2,2,2) | ✅ | 无时间维度 downsampling |
| Audio at 48kHz/960 hop = 50Hz | ✅ checkpoint 确认 encoder_rates=[2,3,4,5,8] | 重要修正 |
| in_dim=36 = 16+4+16 | ✅ | 输入通道构成正确 |

### ⚠️ 修正不完整或引入新问题

1. **"Need to build Diffusion Forcing from scratch" → "FlowMatchPairScheduler already supports independent sigma"**

   这条修正过于乐观。FlowMatchPairScheduler 的 `dual_sigma_shift` 是 **per-sample** 的（整个样本一个 sigma），而 Diffusion Forcing 需要的是 **per-token/per-frame** 的独立 sigma。两者差异如下：

   ```
   MOVA FlowMatchPairScheduler:
     sample σ_visual ~ p(σ), σ_audio ~ p(σ)  ← 整个序列统一

   Diffusion Forcing (需要的):
     for each frame t:
       σ_v(t) ~ p(σ), σ_s(t) ~ p(σ)  ← 每帧独立
   ```

   虽然有了 per-sample 的基础设施，扩展到 per-frame 并非 trivial。需要修改：
   - noise 采样逻辑（从 scalar → [B, T] tensor）
   - loss 计算（每帧独立 weighting）
   - AdaLN conditioning（从 global timestep → per-frame timestep tensor）

   plan 将此描述为 "Extend to per-frame (not just per-sample)" 是准确的，但低估了工作量。

2. **"Keep 3D dim=1536 for bridge compatibility"**

   这是一个合理的设计决策，但需要注意：MOVA bridge 的 `audio_hidden_dim=1536` 是因为 audio DiT 的 dim=1536。如果 3D struct stream 也用 dim=1536，那两者在 bridge 中是同维度的，可以直接复用 bridge 的 projection 层。但 MOVA bridge 实际做的是 **5120 ↔ 1536** 的跨维度 attention（通过 projection），DualForce 如果 video=1536, struct=1536，则变成 **1536 ↔ 1536** 的同维度 attention，projection 层的作用会不同。

---

## 三、Architecture Mapping 审查

### 3.1 MOVA → DualForce 映射 ✅ 概念正确

```
video_dit (5120, 40L) → video_dit (1536, 20L, causal+KV-cache)
video_dit_2 (5120, 40L) → [REMOVED]
audio_dit (1536, 30L) → struct_dit (1536, 20L, 3D stream)
Bridge (full, all layers) → Bridge (shallow_focus, video<->3D)
```

这个映射逻辑清晰。但有几个需要讨论的点：

1. **audio_dit 30L → struct_dit 20L**: 为什么 struct_dit 的层数要与 video_dit 对齐（都是 20 层）？MOVA 中 audio_dit 是 30 层而 video_dit 是 40 层，并非对称设计。DualForce 的 struct 信息量远小于 video（128 dim vs 36ch spatial），也许 struct_dit 不需要 20 层那么深。减少到 10-15 层可以节省大量参数和计算。

2. **Audio VAE → [REMOVED, replaced by HuBERT conditioning]**: 正确。但需要确认 DualForce 是否还需要 text_encoder (UMT5)。proposal 的训练流程图中有 text prompt，但 talking head 的典型 use case 是 "audio + reference image → video"，text prompt 的作用是什么？如果只是描述说话人（"a man talking"），这个信息可能冗余。

### 3.2 Key Files to Modify ✅ 合理完整

文件修改清单覆盖了关键路径。但缺少几个文件：

- `mova/diffusion/models/__init__.py` — 需要注册新的 struct_dit 模型
- `mova/diffusion/pipelines/__init__.py` — 需要注册新的 DualForce pipeline
- `mova/engine/trainer/accelerate/lora_utils.py` — 如果后续 Stage 3 使用 LoRA，需要适配新模块
- 新增文件 `mova/diffusion/models/audio_conditioning.py` — HuBERT 集成

---

## 四、Execution Plan 审查

### 4.1 Phase 0: Environment & Data (Week 1-2) ✅ 合理

数据集下载优先级（HDTF > CelebV-HQ > VFHQ）与 dataset.md 一致。预处理脚本的 7 步 pipeline 设计合理。

**补充建议**:
- 预处理步骤中应包含 **LivePortrait 环境安装和兼容性验证** —— LivePortrait 的依赖可能与 MOVA 的依赖冲突
- 应该在 Phase 0 就验证 LivePortrait 对 HDTF 数据的 3D latent 提取质量，而不是等到后面

### 4.2 Phase 1: MOVA-Lite Backbone (Week 2-3) ⚠️ 已标记完成但需重新审视

Phase 1 标记了 `✅ CODE COMPLETE`，但这些代码在 `20260221-cc-1st` 分支上，且用户已表示 "可能是之前被舍弃的执行部分，在文档中保持未执行状态"。

**从 git 历史和 .pyc 缓存文件的分析来看，这些代码确实曾经被编写过**:

- `wan_struct_dit.py` — 3D Structure DiT（304 行）
- `kv_cache.py` — MultiModalKVCache（252 行）
- `audio_conditioning.py` — AudioProjector, DualAdaLNZero（361 行）
- `dualforce_train.py` — 训练 pipeline（608 行）
- `pipeline_dualforce.py` — 推理 pipeline（397 行）
- `diffusion_forcing.py` — Per-frame sigma scheduler（259 行）
- `dualforce_dataset.py` — 数据集类
- 以及 7 个预处理脚本、6 个评估脚本、7 个 ablation 配置

**这些代码的质量和适用性需要重新评估**:

1. **代码是否与当前最新的分析结论一致？** —— 这些代码是在 2026-02-21 基于 proposal v3.4 编写的。如果后续对 proposal 做了调整（如本审查建议的修改），代码需要同步更新。
2. **代码是否经过 GPU 验证？** —— plan 中明确标注了 "Verify forward pass — needs GPU" 和 "Verify KV-cache matches non-cached output — needs GPU" 仍为未完成状态。
3. **代码的结构设计是否合理？** —— 从 branch 上的 config 文件来看，DualForce config 的设计与 proposal 基本一致（dim=1536, layers=20, shallow_focus, σ_s_max=0.7 等），但存在 proposal 审查中指出的问题。

**建议处理方式**:
- 在 plan.md 中将 Phase 1 的所有项目标记为 `未执行`
- 保留 branch 上的代码作为参考，但重新开始时需要基于最新的分析结论做调整
- 特别关注 code_research 审查中指出的 DAC 配置错误是否影响了已编写的代码

### 4.3 Phase 2: Causal Pretraining (Week 3-4) ❌ 工作量严重低估

这是 plan.md 中问题最大的 Phase。

**plan 的描述**:
```
- Prepare general video data (VoxCeleb2 + Panda-70M subsets)
- Train MOVA-Lite with causal attention (video-only)
- Verify loss convergence
- Save mova_lite_causal_base.pt
Milestone: Causal video generation produces coherent frames
```

**问题分析**:

1. **"从零训练"被轻描淡写**: 由于 MOVA-Lite 的 dim 和 layers 与 MOVA-360p 完全不同，无法迁移预训练权重。这意味着 Phase 2 实际上是在**从随机初始化训练一个全新的视频生成模型**。要在 1-2 周内让一个 ~617M 的 DiT 产生 "coherent frames"，所需的数据量和训练步数远超 VoxCeleb2 + Panda-70M 的 15K clips 子集。

2. **对比参考**: Open-Sora（~700M 参数）的训练在 64 GPU 上跑了数天，使用了百万级视频。即使 MOVA-Lite 更小、数据集更小，在 8 GPU 上 1-2 周也很可能不够达到"coherent frames"的 milestone。

3. **备选方案未讨论**: 如果 Phase 2 收敛缓慢，是否有 fallback？例如：
   - 使用更小的模型（dim=512, layers=8）先做概念验证？
   - 跳过 Phase 2，直接在 Stage 3 的 talking head 数据上联合训练？
   - 使用知识蒸馏从 MOVA-360p 初始化 MOVA-Lite？

**建议**:
- 重新估算 Phase 2 的时间（可能需要 2-4 周而非 1-2 周）
- 或者考虑一个更务实的替代方案：
  a) 使用 MOVA-360p 的部分权重初始化 MOVA-Lite（如 head_dim 兼容的部分权重）
  b) 从一个开源的小型视频 DiT（如 Open-Sora 的某个 checkpoint）迁移部分权重
  c) 跳过通用视频预训练，直接在 talking head 数据上训练

### 4.4 Phase 3: Multi-Modal Architecture (Week 4-6) ✅ 范围合理

Phase 3 的任务列表合理：创建 Struct DiT、LivePortrait wrapper、Bridge adaptation、HuBERT 集成、DualAdaLNZero、gated residual。

**需要补充的任务**:
- 解决 audio temporal alignment 问题（HuBERT 50Hz → video 25fps 的对齐）
- 定义 struct_dit 的 n_tokens_per_frame（1 vs 4 vs 8）
- 定义 σ_s 的训练-推理 mismatch 缓解策略

### 4.5 Phase 4: Diffusion Forcing Training (Week 6-8) ⚠️ 训练时间可能不足

在 ~81h 数据（HDTF + CelebV-HQ）上训练 ~1.5B 参数的模型，2 周时间是否足够？

- 81h × 25fps = 7.29M 帧
- batch_size=2/GPU × 32 frames/clip × 8 GPUs = 512 frames/step
- 7.29M / 512 ≈ 14,200 steps/epoch
- 2 周 × 14h/天 × 3600s ≈ 1M seconds; 假设 ~1.5s/step → ~670K steps → ~47 epochs

47 epochs 可能足够（视频生成通常不需要太多 epochs），但这取决于模型在 Phase 2 后的基线质量。如果 Phase 2 的 causal pretraining 不充分，Phase 4 可能需要更长时间来同时学习基础视频生成和多模态融合。

### 4.6 Phase 5: Evaluation & Ablation (Week 8-10) ✅ 合理

5 个 ablation study 的设计与 proposal 一致：有/无 3D stream、双向/单向 attention、fusion depth (4/8/12)、σ_s range (0.5/0.7/1.0)、长序列一致性。

**补充建议**:
- 增加一个 ablation: **n_tokens_per_frame for struct** (1/4/8)
- 增加一个 ablation: **有/无 FLAME alignment loss** (验证 FLAME 的辅助监督价值)
- 增加一个 ablation: **有/无 lip-sync loss** (验证 contrastive loss 的独立贡献)

---

## 五、Plan 与 Proposal 的不一致之处

| 维度 | plan.md | proposal.md | 差异分析 |
|------|---------|-------------|---------|
| 总层数 | 20 | 28 (Section 7.2) | plan 修正后的值更合理 |
| 总参数量 | ~1.5B (from config) | ~2.35B (Section 7.3) | 差异 ~800M，主要因 proposal 可能包含了 frozen 模块 |
| 训练方式 | Full FT (use_lora=False) | LoRA rank=64 + new modules | **重大矛盾**: plan 用全参训练，proposal 用 LoRA。需统一 |
| 训练总时间 | ~12 weeks (6 Phases) | ~10 weeks (Section 13) | plan 更保守（多 2 周），合理 |
| audio_fps | 25.0 (bridge config) | 50Hz → 4:1 pooling → 12.5fps | 对齐方式不同，需要统一 |
| struct_dit layers | 20 (same as video) | 不明确 | plan 更具体 |
| N_fusion (shallow blocks) | 未明确指定 | 8/28 = ~29% | plan 应明确 |

**最关键的矛盾: Full FT vs LoRA**

plan 的 DualForce config 中 `use_lora=False`，train_modules 包含 "video_dit", "struct_dit" 等所有模块。而 proposal Section 9 (Stage 3) 写 "冻结 MOVA-Lite video backbone 大部分参数, LoRA (rank=64) on MOVA-Lite spatial + temporal attention"。

这是一个根本性的训练策略差异：
- **Full FT**: 所有 ~1.5B 参数都训练，需要更多数据但有更高的上限
- **LoRA**: 只训练 ~380M 参数（rank=64 的 LoRA + 新模块），需要更少数据但上限受限

两种策略各有优劣，但不能同时成立。需要明确选择一种。

**建议**: 鉴于模型是从零训练的（无预训练权重可冻结），Full FT 是唯一合理的选择。LoRA 只有在对预训练模型做 fine-tune 时才有意义。因此 plan.md 的 `use_lora=False` 是正确的，proposal 中的 LoRA 训练策略需要修正。

---

## 六、Plan 与 Dataset.md 的对齐分析

### 6.1 数据使用方案 ✅ 基本一致

plan 的数据使用与 dataset.md 的推荐配置一致：

| 阶段 | plan | dataset.md | 一致性 |
|------|------|-----------|--------|
| Phase 0 data | HDTF + CelebV-HQ + VFHQ | Priority 1: HDTF, Priority 2: CelebV-HQ, Priority 3: VFHQ | ✅ |
| Phase 2 data | VoxCeleb2 + Panda-70M subsets | Stage 2: Panda-70M (5K) + VoxCeleb2 (10K) | ✅ |
| Phase 4 data | HDTF + CelebV-HQ (~81h) | Stage 3 Primary: HDTF + CelebV-HQ = ~81h | ✅ |

### 6.2 遗漏的数据考虑

1. **TalkVid 在 plan 中完全没有出现**: dataset.md 强烈推荐 TalkVid-Core (160h) 作为扩展数据，但 plan 的 Phase 4 只使用了 ~81h。如果 81h 不够（很可能），需要有扩展数据的后备方案。

2. **评估数据集在 plan Phase 5 中未具体列出**: dataset.md 列出了 HDTF test split, CelebV-HQ test split, VFHQ test split, VoxCeleb2 random 200, MEAD test split。plan 应明确使用哪些。

3. **存储需求**: dataset.md 估算最小配置（HDTF + CelebV-HQ）需要 ~800GB 存储。plan 中未提及存储规划，这在 8 卡 A100 环境下可能是一个实际约束。

---

## 七、仓库实际状态 vs Plan 描述

### 7.1 main 分支现状

当前 main 分支仅包含：
- 原始 MOVA 代码（未修改）
- `cc_core_files/` 下的文档（code_research.md, proposal.md, plan.md, dataset.md）
- `cc_todo/` 下的工作记录文档
- 一些 `.pyc` 缓存文件（来自之前在 `20260221-cc-1st` 分支上的开发）

### 7.2 `20260221-cc-1st` 分支的代码

这个分支包含了之前实现的 DualForce 代码，涵盖：
- 核心模型文件（wan_struct_dit.py, kv_cache.py, audio_conditioning.py）
- 训练/推理 pipeline（dualforce_train.py, pipeline_dualforce.py）
- Diffusion Forcing scheduler（diffusion_forcing.py）
- 数据集类（dualforce_dataset.py）
- 预处理脚本（01-07）
- 评估脚本（FVD, FID, identity, sync, pose）
- 配置文件和 ablation configs
- 下载脚本和验证脚本

**这���代码是有价值的参考**，即使最终决定重新实现，也可以从中借鉴设计思路和避免已知的 bug。

### 7.3 建议

- 在重新开始实施前，仔细审查 `20260221-cc-1st` 分支的代码质量
- 特别关注以下已知的代码问题点：
  - struct_dit 的 StructTokenProjector 的 n_tokens_per_frame 设定
  - audio temporal alignment 的实现
  - DualAdaLNZero 的 per-frame sigma 处理是否正确
  - KV-cache 的 clean-state-only 追加逻辑

---

## 八、Plan.md 整体建议

### 8.1 结构性建议

1. **添加 Phase 0.5: 小模型概念验证**
   在 Phase 1 (backbone) 和 Phase 2 (pretraining) 之间增加一个快速验证阶段：
   - 使用极小模型 (dim=256, layers=4) + HDTF 子集 (100 clips)
   - 验证多模态 Diffusion Forcing 的训练收敛性
   - 验证 σ_s 的不同范围对训练稳定性的影响
   - 预计 1-2 天即可完成
   - 这能在投入大量时间做 Phase 2-4 之前快速发现潜在问题

2. **明确 Phase 2 的"从零训练"策略**
   - 选项 A: 通用视频预训练（当前方案）—— 需要增加时间预算到 3-4 周
   - 选项 B: 跳过通用预训练，直接在 talking head 数据上训练 —— 节省时间但可能损失通用能力
   - 选项 C: 使用知识蒸馏从 MOVA-360p 初始化 —— 技术上复杂但效果可能最好

   建议在 plan 中明确选择并给出理由。

3. **添加显式的 Checkpoint / Go-No-Go 决策点**
   - Phase 2 完成后: 如果 video loss 未降到阈值 X 以下，触发 fallback 方案
   - Phase 4 的 5K steps: 如果 video+struct loss 不同时下降，检查训练配置
   - Phase 4 的 20K steps: 生成样本的人工评估，决定是否继续或调整

### 8.2 参数建议

| 参数 | 当前 plan 值 | 建议值 | 理由 |
|------|------------|--------|------|
| struct_dit layers | 20 | 10-15 | 3D latent (128d) 信息量远小于 video，不需要同等深度 |
| n_tokens_per_frame (struct) | 未明确 | 2-4 | 1 个太少（信息不足），8 个太多（128d 展不开） |
| audio pooling ratio | 4:1 (→12.5fps) | 2:1 (→25fps) | 对齐 video fps |
| Phase 2 duration | 1-2 weeks | 3-4 weeks | 从零训练需要更多时间 |
| σ_s inference start | 1.0 (隐含) | 0.7 | 匹配训练分布 |

### 8.3 风险管理建议

plan 的 Risk Mitigation 表虽然列出了风险，但缺少具体的 mitigation action timeline。建议：

| 风险 | 触发条件 | 具体 Action | 决策时间点 |
|------|---------|-------------|-----------|
| Phase 2 不收敛 | loss > X at 20K steps | 切换到 Option B (直接 talking head) | Phase 2 Day 7 |
| 3D latent 质量差 | FLAME cross-check > 50% 帧被丢弃 | 切换到 FLAME-only 表征 | Phase 0 Stage 1 完成后 |
| OOM | 任何配置下 > 80GB/GPU | 减 batch_size 1, 减 clip_length 16 | Phase 1 forward pass 验证时 |
| 多模态 DF 不稳定 | loss 震荡或 NaN | 移除 per-frame 噪声, 退回 per-sample 噪声 | Phase 4 Day 3 |

---

## 九、与 Dataset.md 的交叉审查

### 9.1 Dataset.md 自身的问题

dataset.md 整体质量高，但有几个值得注意的点：

1. **HuBERT 对非英语的表征质量**: dataset.md 正确指出了 TalkVid 的 15 种语言中非英语音频的 HuBERT 表征质量需要验证。HuBERT-Large 主要在英语数据上训练，对中文、阿拉伯语等的特征质量可能下降。这可能限制 TalkVid 多语言数据的使用价值。

2. **VFHQ 的音频缺失**: dataset.md 注意到 "非所有 clip 有清晰音频"。对 DualForce 来说，audio 是 conditioning 输入，**没有清晰音频的 clip 无法用于 Stage 3 训练**。这意味着 VFHQ 的实际可用数据量可能远小于标称的 ~50h。

3. **预处理存储需求**: dataset.md 估算 500h 的 feature cache 需要 ~3.3TB，其中 video_latents 占 ~3.2TB。这个估算可能偏高——如果使用 float16 存储（而非 float32），可以压缩约 50%。

### 9.2 Dataset.md 与 Plan 的配合

dataset.md 的下载优先级（HDTF → CelebV-HQ → VoxCeleb2/Panda-70M → VFHQ → TalkVid → MEAD）与 plan 的 Phase 时序（Phase 0 data → Phase 2 data → Phase 4 data → Phase 5 eval）完全对齐，这说明两份文档的作者有良好的协调。

---

## 十、总结

### Plan.md 的核心优势
1. 对 Proposal 错误的修正有价值且大部分准确
2. Phase 分解逻辑清晰，milestone 定义具体
3. 文件修改清单实用
4. 与 Dataset.md 协调良好

### Plan.md 的核心问题
1. Phase 2 工作量严重低估（从零训练 video DiT 不是 1-2 周能完成的）
2. Full FT vs LoRA 的训练策略与 Proposal 矛盾
3. 缺少小模型快速验证阶段（Phase 0.5）
4. 缺少显式的 Go-No-Go 决策点
5. 部分参数（struct_dit layers, n_tokens_per_frame, audio pooling ratio）需要重新审视
6. TalkVid 扩展数据未纳入计划

### 整体建议优先级

| 优先级 | 行动项 |
|--------|-------|
| **P0** | 统一 Full FT vs LoRA 的训练策略（建议 Full FT） |
| **P0** | 重新评估 Phase 2 的时间预算（至少 3-4 周或考虑替代方案） |
| **P0** | 修正 audio temporal alignment (2:1 而非 4:1 pooling) |
| **P1** | 增加 Phase 0.5: 小模型概念验证 |
| **P1** | 明确 struct_dit 的 layers 和 n_tokens_per_frame |
| **P1** | 讨论 σ_s 的 train-test mismatch 缓解策略 |
| **P2** | 增加 Go-No-Go 决策点 |
| **P2** | 纳入 TalkVid-Core 作为扩展数据计划 |
| **P3** | 审查 20260221-cc-1st 分支代码的可复用性 |
