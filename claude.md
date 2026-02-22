# DualForce 项目 Claude Code 指引

## 项目背景

**DualForce** 是一个基于 MOVA（OpenMOSS 开源的视频-音频同步生成基础模型）进行深度改造的研究项目，目标是实现 **3D-Aware Native Autoregressive Diffusion for Real-Time Avatar Video Generation**。

### 核心问题

> 如何在不引入显式 3D 输入的前提下，让自回归视频扩散模型获得 3D 结构感知能力，从而同时提升长序列一致性、降低训练复杂度？

### 核心思路

在 Diffusion Forcing 框架下，将隐式 3D 结构 latent 作为与 video latent 平行的模态，通过浅层双向 attention 融合 + 深层独立特化实现联合自回归生成。推理时无需任何显式 3D 输入，仅需参考图片和音频即可自回归生成 talking head 视频。

### 五个核心创新

1. **多模态 Diffusion Forcing**：首次将 Diffusion Forcing 扩展到 video + 3D latent 联合生成，消除蒸馏依赖
2. **Shallow Fusion + Deep Specialization**：浅层（0~N_fusion-1）双向跨模态 attention + 深层（N_fusion~L-1）独立特化
3. **Asymmetric Noise Schedule**：σ_s ~ U(0,0.7) vs σ_v ~ U(0,1.0)，隐式"先结构后外观"生成
4. **Audio Dual-path Conditioning**：audio→video + audio→3D→video 双路径驱动
5. **Inference-time 3D-free**：训练时 3D 监督，推理时从噪声联合生成

### 技术基础

- **Base Codebase**: MOVA-Lite（将 MOVA 的 Wan 2.2 14B MoE 缩减为 ~2B dense DiT）
- **3D 表征**: LivePortrait Motion Latent（隐式 3D keypoints，~128 dim）
- **Audio Encoder**: HuBERT-Large（frozen，50Hz tokens）
- **训练范式**: Diffusion Forcing + Flow Matching v-prediction
- **KV-Cache 参考**: OVI（Character.AI）

### 实施路径（MOVA → MOVA-Lite 改造）

5 个核心改动：
- **改动 0**: 模型缩减 — Wan 2.2 14B MoE → ~2B dense（减层/减维/去 MoE）
- **改动 1**: Audio backbone → 3D Structure backbone（复用 Bridge CrossAttention）
- **改动 2**: 标准 Diffusion → Diffusion Forcing（per-token 独立 σ_v/σ_s）
- **改动 3**: Full Attention → Causal + Shallow/Deep Split
- **改动 4**: 新增 Audio Conditioning 支路 + KV-Cache

### 训练阶段

- **Stage 1**: 3D Latent Space Preparation（提取 3D pseudo labels）
- **Stage 2**: MOVA-Lite Causal Adaptation（full → causal attention）
- **Stage 3**: Multi-Modal Diffusion Forcing Training（核心训练）
- **Stage 4**（可选）: 推理加速（Consistency Distillation）

详细方案请参阅 `cc_core_files/proposal.md`。

---

## 当前仓库结构

当前仓库为 MOVA 原始代码库，DualForce 改造尚未开始。

```
项目根目录/
├── mova/                               # MOVA 核心包
│   ├── diffusion/
│   │   ├── models/
│   │   │   ├── wan_video_dit.py        # Video DiT（Wan 2.2 架构）
│   │   │   ├── wan_audio_dit.py        # Audio DiT
│   │   │   ├── interactionv2.py        # DualTowerConditionalBridge（双向融合）
│   │   │   └── dac_vae.py              # DAC 音频 VAE
│   │   ├── pipelines/
│   │   │   ├── pipeline_mova.py        # MOVA 推理流水线
│   │   │   ├── mova_train.py           # MOVA 训练流水线
│   │   │   └── mova_lora.py            # MOVA LoRA 微调
│   │   └── schedulers/
│   │       ├── flow_match.py           # 标准 FlowMatchScheduler
│   │       └── flow_match_pair.py      # Video/Audio 独立噪声调度
│   ├── datasets/
│   │   └── video_audio_dataset.py      # MOVA 视频-音频数据集
│   └── engine/                         # 训练基础设施（FSDP, Accelerate, LoRA）
├── configs/                            # 训练配置（LoRA 单卡/多卡）
├── scripts/
│   ├── inference_single.py             # MOVA 推理脚本
│   └── training_scripts/               # MOVA 训练启动脚本
├── workflow/                           # Streamlit 前端 UI
├── checkpoints/                        # 模型权重（MOVA-360p）
├── cc_core_files/                      # 项目核心文档
│   ├── proposal.md                     # DualForce 完整技术方案（v3.4）
│   ├── code_research.md                # MOVA 代码库分析
│   ├── dataset.md                      # 数据集信息
│   └── plan.md                         # 执行计划
├── 20260221-cc-1st/                    # 历史工作会话记录（仅供参考，代码已撤销）
├── claude.md                           # 本文件
└── progress.md                         # 任务进度记录
```

---

## 当前项目状态

项目处于**开发前期**阶段，DualForce 的代码改造尚未开始。当前仅完成了：
- 研究方案设计（`cc_core_files/proposal.md`，v3.4）
- MOVA 代码库分析（`cc_core_files/code_research.md`）
- 数据集选型与规划（`cc_core_files/dataset.md`）
- 执行计划制定（`cc_core_files/plan.md`）

### 待开展工作（按 proposal 中的 10 周计划）

1. **Week 1-2**: MOVA 代码库理解 + 模型缩减 + 环境搭建
2. **Week 3-4**: Stage 2 — Causal Adaptation
3. **Week 5-6**: Stage 3 核心改造 — 3D Stream + Diffusion Forcing
4. **Week 7-8**: Audio Conditioning + Shallow/Deep Split + KV-Cache
5. **Week 9-10**: 评估 + Ablation

---

## 任务执行规范

### 1. progress.md 维护

每次任务执行时：
- **任务开始前**：查阅 `progress.md` 了解之前的工作内容、经验和待解决问题
- **任务执行中**：记录当前正在做的事情
- **任务完成后**：立即更新 `progress.md`，包含：
  - 日期和时间
  - 任务内容描述
  - 任务结果和效果
  - 遇到的问题及解决方法

### 2. claude.md 维护

- 任务完成后检查 `claude.md` 是否存在过时内容
- 如有过时内容，根据最新的任务结果或已调整的项目实现进行更新
- 多次遇到的问题应沉淀为经验记录到本文件的"经验沉淀"部分

### 3. Git 提交规范

- 每完成一个任务或小模块后，及时执行 `git add & commit & push`
- **Git 帐户**：jpagkr@163.com / Jp-17
- **commit 消息**：主要使用中文
- **文件/文件夹命名**：使用英文
- **md 文档内容**：主要使用中文
- **md 文档名称**：以文档产出日期开头（如 `20260223-xxx.md`）

### 4. 代码修改注意事项

- 修改 MOVA 原有文件时，保持向后兼容（MOVA 原有推理不受影响）
- 新增 DualForce 代码使用 MOVA 的 registry 注册机制（MMEngine-based）
- MOVA 使用非对称双塔架构：Video DiT（大）+ Audio DiT（小）+ Bridge CrossAttention（双向融合）
- MOVA 训练基础设施完整可复用：AccelerateTrainer、FSDP 配置、数据管线、Checkpoint 管理

---

## 经验沉淀

（随项目推进持续积累）

---

## 关键参考文档

| 文档 | 路径 | 说明 |
|------|------|------|
| 研究方案（v3.4） | `cc_core_files/proposal.md` | DualForce 完整技术方案，包含架构设计、训练策略、数据集需求 |
| 代码分析 | `cc_core_files/code_research.md` | MOVA 代码库深度分析 |
| 数据集信息 | `cc_core_files/dataset.md` | 数据集选型与下载指南 |
| 执行计划 | `cc_core_files/plan.md` | 项目执行计划 |
| 任务进度 | `progress.md` | 持续更新的任务进度记录 |
