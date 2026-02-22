# code_research.md 调研分析审查报告

> 审查日期: 2026-02-23
> 审查对象: cc_core_files/code_research.md
> 审查方法: 逐项对比 MOVA 仓库源代码 + checkpoint 配置文件 + git 历史
> 审查结论: 整体可信度较高，v2 修正基本准确，但仍有若干遗漏和不精确之处

---

## 一、总体评价

code_research.md 是一份质量较高的代码调研文档。它做到了关键的一点：**在 v2 中基于 checkpoint config 对初始的代码默认值分析进行了修正**，这是正确的做法。因为 MOVA 的代码默认值（写在函数签名中的默认参数）与实际部署的 MOVA-360p 模型的配置是不同的——checkpoint 中存储的是实际训练使用的超参数，而代码中的默认值只是占位/兼容值。

但文档中仍然存在**"修正了一半"的问题**——某些地方顶部表格用了 checkpoint 值，但正文的详细描述仍然残留着旧的代码默认值，造成自相矛盾。

---

## 二、逐节审查

### 2.1 Section 1: Directory Structure ✅ 基本正确

文档列出的目录结构与实际仓库一致。但有一个遗漏：

- **遗漏**: 未提及 `mova/diffusion/pipelines/mova_lora.py`，这是单独的 LoRA 推理 pipeline
- **遗漏**: 未提及 `scripts/eval/` 目录，其中包含 FVD、FID、identity、sync、pose 等评估脚本
- **遗漏**: 未提及 `scripts/download/` 目录
- **遗漏**: 未提及 `mova/engine/optimizers/` 目录

**影响程度**: 低。这些遗漏不影响核心架构理解。

---

### 2.2 Section 2.1: Video DiT (WanModel) ⚠️ 有矛盾

**顶部修正表（正确）**:

| 参数 | 文档 v2 修正值 | checkpoint 实际值 | 一致性 |
|------|--------------|------------------|--------|
| dim | 5120 | 5120 | ✅ |
| ffn_dim | 13824 | 13824 | ✅ |
| num_heads | 40 | 40 | ✅ |
| num_layers | 40 | 40 | ✅ |
| patch_size | (1,2,2) | [1,2,2] | ✅ |
| in_dim | 36 | 36 | ✅ |
| out_dim | 16 | 16 | ✅ |
| text_dim | 4096 | 4096 | ✅ |
| freq_dim | 256 | 256 | ✅ |
| require_vae_embedding | true | true | ✅ |
| require_clip_embedding | false | false | ✅ |

**但正文 DiT Block 描述仍用旧值** ❌:

```
文档正文 (第120-141行):
  SelfAttention(dim=3072, heads=24)     ← 应为 dim=5120, heads=40
  CrossAttention(dim=3072, heads=24)    ← 应为 dim=5120, heads=40
  FFN: Linear(3072→12288)              ← 应为 5120→13824

  Forward Flow:
  Patchify: Conv3d(16, 3072, ...)      ← 应为 Conv3d(36, 5120, ...)
  "Process through 30 DiT blocks"      ← 应为 40 DiT blocks
  Unpatchify: reshape to [B, 16, T/4+1, H/8, W/8] ← T 维度不再有 /4 因为 patch_size temporal=1
```

**这是文档中最严重的问题**: 顶部做了修正，但详细架构描述全部仍是旧值。读者��果直接看正文会被误导。

**关于 Unpatchify 的细节**: 由于 `patch_size=(1,2,2)` 而非 `(2,2,2)`，时间维度不做 downsampling。因此 output shape 应为 `[B, 16, T, H/2/8, W/2/8]` 而非 `[B, 16, T/4+1, H/8, W/8]`。实际上 VAE 的时间 stride 是 4，所以 latent 的 T 已经是压缩过的了，video DiT 的 `patch_size=(1,2,2)` 仅在空间上做 2x downsample。这一点文档没有说清楚。

**额外遗漏**:
- 文档未提及 `has_image_input=false`（checkpoint 配置），而正文又提到 "Optional image input handling (first 257 tokens for DINO/CLIP)"，这两者矛盾
- 文档未提及 `seperated_timestep` 参数（用于双阶段去噪的独立时间步嵌入）

---

### 2.3 Section 2.2: Audio DiT (WanAudioModel) ⚠️ 未完全修正

**文档值 vs checkpoint 实际值**:

| 参数 | 文档正文值 | checkpoint 实际值 | 问题 |
|------|----------|------------------|------|
| dim | 1536 | 1536 | ✅ |
| ffn_dim | 6144 | **8960** | ❌ 文档正文未修正 |
| num_heads | 12 | 12 | ✅ |
| num_layers | 30 | 30 | ✅ |
| patch_size | (2,1,1) | **[1]** | ❌ 正文未修正 |
| in_dim | 128 | 128 | ✅ |
| out_dim | 128 | 128 | ✅ |

虽然顶部 v2 修正表已经列出了 ffn_dim=8960 和 patch_size=[1]，但正文 Section 2.2 的 Config 块仍写着 `ffn_dim = 6144` 和 `patch_size = (2,1,1)`。

**额外发现**:
- checkpoint 中 audio_dit 的 vae_type 为 "dac"，文档虽提到了 DAC 但未在 Audio DiT 配置中明确标注
- checkpoint 中 audio_dit 的 encoder_rates 是 `[2,3,4,5,8]`（总 stride=960），与代码默认值 `[2,4,8,8]`（总 stride=512）完全不同。这意味着 hop_length=960 而非 2048 或 512。文档顶部表格写的 hop_length=960 是正确的，但 Section 2.4 的 DAC 详细描述中仍写 `stride [2,4,8,8] = total stride 256` ❌

---

### 2.4 Section 2.3: Bridge CrossAttention ⚠️ 未完全修正

**文档正文 vs checkpoint**:

| 参数 | 文档正文值 | checkpoint 实际值 | 问题 |
|------|----------|------------------|------|
| visual_layers | 30 | **40** | ❌ |
| audio_layers | 30 | 30 | ✅ |
| visual_hidden_dim | 3072 | **5120** | ❌ |
| audio_hidden_dim | 1536 | 1536 | ✅ |
| audio_fps | 44100/2048 ≈ 21.5 | **50.0** | ❌ |
| interaction_strategy | "shallow_focus" | **"full"** | ❌ |
| apply_cross_rope | 未提及 | **true** | 遗漏 |
| head_dim | 128 | 128 | ✅ |

这里问题同样严重：顶部表格写了修正值，但正文的 Config 块和交互策略描述全部没改。

**交互策略描述的评价**: 文档对 5 种交互策略的描述基本准确，这部分通过代码验证了。CrossModalInteractionController 中确实存在 shallow_focus、distributed、progressive、custom、full 五种策略。但需要注意的是，**实际 MOVA-360p 使用的是 "full" 策略**（所有 40 个 video 层和 30 个 audio 层都参与交互），而非默认的 "shallow_focus"。这对 DualForce 的设计决策有重要影响——MOVA 选择了最重的融合方案。

**Aligned-RoPE 描述**:
文档提到 `build_aligned_freqs` 函数做 audio-video 时间对齐，这在代码中确实存在。但实际的 checkpoint 配置表明 `apply_cross_rope=true`，文档未提及此配置项。

---

### 2.5 Section 2.4: Audio VAE (DAC) ❌ 严重不准确

文档中的 DAC 配置与 checkpoint 实际配置差异巨大：

| 参数 | 文档值 | checkpoint 实际值 | 差距 |
|------|-------|------------------|------|
| encoder_dim | 64 | **128** | 2x |
| encoder_rates | [2,4,8,8] | **[2,3,4,5,8]** | 完全不同 |
| decoder_dim | 1536 | **2048** | +34% |
| decoder_rates | 未明确 | **[8,5,4,3,2]** | - |
| sample_rate | 44100 | **48000** | 不同 |
| hop_length | 2048 (从 stride 512 × sample_rate 推算?) | **960** (= 2×3×4×5×8) | 完全不同 |
| latent_dim | 128 (正确！但推导过程错误) | 128 | ✅ 巧合正确 |
| continuous | 未提及 | **true** | 遗漏 |

**文档的推导错误**: 文档写 "Encoder: Conv1d chain, stride [2,4,8,8] = total stride 256"，但实际的 encoder_rates 是 [2,3,4,5,8]，total stride = 960，不是 256 也不是 512。文档还写 "Hop length: 2048 samples"，但实际 hop_length 应该等于 total stride = 960。

这部分的错误源于文档使用了代码中的**默认值**（`dac_vae.py` 的 `__init__` 默认参数），而没有检查 checkpoint 中的实际配置。代码中 DAC 的默认 `encoder_dim=64` 和 `encoder_rates=[2,4,8,8]` 仅仅是模块的默认构造参数，MOVA-360p 在训练时使用了完全不同的配置。

---

### 2.6 Section 2.5: Video VAE ✅ 基本正确

- z_dim=16 ✅
- temporal stride=4 ✅ (from config)
- spatial stride=8 ✅

---

### 2.7 Section 3: Training Infrastructure ✅ 基本正确

训练 loop、noise scheduling、training forward pass、data pipeline 的描述都与代码吻合。几个值得注意的点：

1. **FlowMatchPairScheduler 的 shift=5**: checkpoint 中 `shift=5`，而非代码默认的 `shift=3`。文档写的是 `shift=3.0`，不准确。
2. **LoRA target modules**: 文档写 `["q", "k", "v", "o", "to_q", "to_k", "to_v", "proj"]`，实际代码中还有 `"to_out"`。
3. **视频分辨率**: 8 GPU 配置实际用的是 height=352, width=640（非文档中的 352×640...这个倒是一致的）
4. **num_frames=193**: 8 GPU 配置中的帧数是 193，文档未提及这个关键参数
5. **sample_rate=48000**: 训练配置中的音频采样率是 48000，与 checkpoint 一致，但文档在多处仍写 44100

---

### 2.8 Section 4: Inference Pipeline ✅ 基本正确

两阶段去噪（video_dit → video_dit_2 切换基于 boundary_ratio）的描述准确。Reference image processing 和 text encoding 的描述也正确。

---

### 2.9 Section 5: Key Dependencies ✅ 正确

---

### 2.10 Section 6: Implications for DualForce ⚠️ 基于旧值的推断

由于正文多处使用了旧的代码默认值，Section 6 中的某些推断需要调整：

1. **"WanModel: shrink config (30->20 layers, 3072->1536 dim)"**: 应为 **40->20 layers, 5120->1536 dim**。plan.md 已经修正了这一点。
2. **"bridge audio_fps = 44100/2048 ≈ 21.5"**: 应为 **50.0 Hz**。
3. **"模型大约 ~7B"**: 文档顶部已修正为 ~14B video backbone，但 Section 6 中的文字可能仍然暗示较小的模型。

---

## 三、关键发现汇总

### 3.1 确认正确的关键信息

1. ✅ MOVA 采用双塔架构（Video DiT + Audio DiT + Bridge）
2. ✅ 存在 video_dit 和 video_dit_2 两个 video DiT（不同权重，同架构）
3. ✅ Bridge 支持多种交互策略（shallow_focus, distributed, progressive, custom, full）
4. ✅ Flow matching 使用 v-prediction (noise - sample)
5. ✅ FlowMatchPairScheduler 支持独立的 visual/audio sigma
6. ✅ 训练基础设施完整（FSDP、gradient checkpointing、FP8 offload、LoRA 等）
7. ✅ 数据管线结构描述正确
8. ✅ in_dim=36 的输入构造（16 VAE + 4 mask + 16 cond）正确

### 3.2 需要修正的关键错误

1. ❌ 正文中 DiT Block 描述使用旧值（dim=3072, heads=24, layers=30）
2. ❌ Audio DiT 正文中 ffn_dim=6144 和 patch_size=(2,1,1) 未修正
3. ❌ DAC Audio VAE 的配置几乎全部是代码默认值而非 checkpoint 值
4. ❌ Bridge 正文中 visual_layers=30, visual_hidden_dim=3072 未修正
5. ❌ 多处 sample_rate 写 44100 应为 48000
6. ❌ FlowMatchScheduler shift=3.0 应为 shift=5（按 checkpoint）

### 3.3 遗漏信息

1. 📋 未提及 checkpoint 中 scheduler 的 `extra_one_step=true` 和 `shift=5`
2. 📋 未提及 bridge 的 `apply_cross_rope=true`（实际开启了跨模态 RoPE）
3. 📋 未提及 DAC 的 `continuous=true`（连续 latent 模式，非 VQ 离散模式）
4. 📋 未提及 text encoder 的详细配置（UMT5, d_model=4096, 24 layers, 64 heads）
5. 📋 未提及 Context Parallel (cp_size=4) 这个关键的分布式策略
6. 📋 未提及 `eval/` 目录下已有的评估脚本

---

## 四、对后续工作的影响

### 4.1 对 proposal.md 的影响

code_research.md 中的不准确值已部分传播到 proposal.md，需要检查 proposal 中是否引用了旧值。

### 4.2 对 plan.md 的影响

plan.md 在 "Proposal Corrections" 一节中已经做了一些修正（如 40->20 layers, 5120->1536 dim），说明 plan 的作者意识到了 code_research 的问题。但 plan 中是否覆盖了所有需修正的点，需要在 plan 审查中进一步检查。

### 4.3 对实际开发的影响

- DAC 配置的错误如果传导到代码实现中，可能导致 audio token 数量计算错误
- Bridge 实际使用 "full" 策略这一事实，意味着 MOVA 团队选择了最重的融合方案，DualForce 改为 "shallow_focus" 虽然有理论支撑（OmniTalker），但也意味着偏离了经验证的配置，这是一个需要关注的风险点

---

## 五、建议

1. **统一正文和修正表**: 将 Section 2.1-2.4 的正文描述全部更新为 checkpoint 实际值
2. **重写 Section 2.4 (DAC)**: 基于 checkpoint config.json 完全重写
3. **补充遗漏信息**: 特别是 `apply_cross_rope=true`、`continuous=true`、Context Parallel
4. **标注信息来源**: 在每个配置项后标注 "(from checkpoint)" 或 "(from code default)" 以避免混淆
