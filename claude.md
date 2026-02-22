# DualForce 项目 Claude Code 指引

## 项目背景

**DualForce** 是一个基于 MOVA（OpenMOSS 开源的视频-音频同步生成基础模型）进行深度改造的研究项目，目标是实现 **3D-Aware Native Autoregressive Diffusion for Real-Time Avatar Video Generation**。

### 核心思路

在 Diffusion Forcing 框架下，将隐式 3D 结构 latent 作为与 video latent 平行的模态，通过浅层双向 attention 融合 + 深层独立特化实现联合自回归生成。推理时无需任何显式 3D 输入，仅需参考图片和音频即可自回归生成 talking head 视频。

### 五个核心创新

1. **多模态 Diffusion Forcing**：首次将 Diffusion Forcing 扩展到 video + 3D latent 联合生成，消除蒸馏依赖
2. **Shallow Fusion + Deep Specialization**：浅层（0-7）双向跨模态 attention + 深层（8-27）独立特化
3. **Asymmetric Noise Schedule**：σ_s ~ U(0,0.7) vs σ_v ~ U(0,1.0)，隐式"先结构后外观"生成
4. **Audio Dual-path Conditioning**：audio→video + audio→3D→video 双路径驱动
5. **Inference-time 3D-free**：训练时 3D 监督，推理时从噪声联合生成

### 技术基础

- **Base Codebase**: MOVA-Lite（将 MOVA 的 Wan 2.2 14B MoE 缩减为 ~2B dense DiT）
- **3D 表征**: LivePortrait Motion Latent（隐式 3D keypoints，~128 dim）
- **Audio Encoder**: HuBERT-Large（frozen，50Hz tokens）
- **训练范式**: Diffusion Forcing + Flow Matching v-prediction
- **KV-Cache 参考**: OVI��Character.AI）

详细方案请参阅 `cc_core_files/proposal.md`。

---

## 项目结构概览

```
项目根目录/
├── mova/                               # 核心包
│   ├── diffusion/
│   │   ├── models/
│   │   │   ├── wan_video_dit.py        # Video DiT（已添加 block-causal attention）
│   │   │   ├── wan_struct_dit.py       # 3D Structure DiT（DualForce 新增）
│   │   │   ├── wan_audio_dit.py        # Audio DiT（MOVA 原有）
│   │   │   ├── interactionv2.py        # Bridge CrossAttention（双向融合）
│   │   │   ├── kv_cache.py             # KV-Cache（DualForce 新增）
│   │   │   └── audio_conditioning.py   # 音频条件化 + DualAdaLNZero（DualForce 新增）
│   │   ├── pipelines/
│   │   │   ├── dualforce_train.py      # DualForce 训练流水线
│   │   │   ├── pipeline_dualforce.py   # DualForce 推理流水线
│   │   │   └── pipeline_mova.py        # MOVA 原有推理
│   │   └── schedulers/
│   │       ├── diffusion_forcing.py    # Diffusion Forcing 调度器（DualForce 新增）
│   │       └── flow_match.py           # 标准 Flow Matching
│   ├── datasets/
│   │   └── dualforce_dataset.py        # 多模态数据集
│   └── engine/                         # 训练基础设施（FSDP, Accelerate）
├── configs/dualforce/                  # DualForce 训练/消融配置
├── scripts/
│   ├── preprocess/                     # 7步数据预处理流水线
│   ├── download/                       # 数据集下载脚本
│   ├── eval/                           # 评估流水线（FVD/FID/ACD/Sync/APD）
│   ├── verify_dualforce.py             # GPU 验证脚本
│   └── training_scripts/               # 训练启动脚本
├── cc_core_files/                      # 项目核心文档（proposal、代码分析等）
├── checkpoints/                        # 模型权重
├── claude.md                           # 本文件
└── progress.md                         # 任务进度记录
```

---

## 当前项目状态

### 已完成的代码工作（2026-02-21）

所有 Phase 1-5 的代码已编写完成（约 12,747 行新增代码，51 个文件变更），包括：
- 核心双塔架构（MOVA-Lite Video DiT + Structure DiT + Bridge CrossAttention）
- Diffusion Forcing 调度器（per-frame 独立 sigma）
- Block-Causal Attention + KV-Cache
- 音频条件化模块（AudioConditioningModule + DualAdaLNZero）
- 训练损失（L_video, L_struct, L_flame, L_lip_sync）
- 数据预处理流水线（7步）+ 数据集下载脚本
- 评估流水线（5个指标）+ 批量生成脚本
- 7组消融实验配置
- 训练/推理脚本 + FSDP 配置

### 待完成（需要 GPU）

1. **P0**: 运行 `verify_dualforce.py` 验证前向传播 + KV-cache 一致性 + 内存估算
2. **P1**: 下载数据集（HDTF + CelebV-HQ）+ 运行预处理流水线
3. **P2**: Phase 2 causal 视频预训练 + Phase 4 多模态训练
4. **P3**: 评估 + 消融实验 + 论文撰写

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
- 新增 DualForce 代码使用 registry 注册机制
- MOVA-360p 真实架构参数与代码默认值有差异（见下方经验沉淀），注意区分
- 训练脚本中使用 `torch.autocast` 时应动态检测设备类型，不要硬编码 `"cuda"`

---

## 经验沉淀

### MOVA checkpoint 参数差异

MOVA-360p 的实际参数与代码默认值存在重大差异，**必须以 checkpoint 配置为准**：

| 参数 | 代码默认值 | 实际 checkpoint 值 |
|------|-----------|------------------|
| dim | 3072 | **5120** |
| 层数 | 30 | **40** |
| patch_size | (2,2,2) | **(1,2,2)** — 无时间下采样 |
| bridge 策略 | shallow_focus | **full**（所有层） |
| audio 采样率 | 44.1kHz/2048 hop | **48kHz/960 hop = 50Hz** |
| in_dim | — | **36** = 16(VAE) + 4(mask) + 16(first_frame) |

### bitsandbytes 依赖问题

`mova/engine/optimizers/__init__.py` 无条件导入了 `bitsandbytes`（虽然它只在 `[train]` 可选依赖中声明）。即使仅做推理也需要安装 `bitsandbytes`，否则报 `ModuleNotFoundError`。

### 硬编码设备类型

在 `torch.autocast` 中不要硬编码 `"cuda"`，应使用动态设备检测：
```python
device_type = "cuda" if torch.cuda.is_available() else "cpu"
with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    ...
```

### Loss placeholder dtype

创建 loss 占位符张量时，确保 dtype 与模型输出一致（通常为 bfloat16），避免混合精度训练中的类型不匹配。

---

## 关键参考文档

| 文档 | 路径 | 说明 |
|------|------|------|
| 研究方案（v3.4） | `cc_core_files/proposal.md` | DualForce 完整技术方案 |
| 代码分析 | `cc_core_files/code_research.md` | MOVA 代码库深度分析 |
| 数据集信息 | `cc_core_files/dataset.md` | 数据集选型与下载指南 |
| 执行计划 | `cc_core_files/plan.md` | 项目执行计划 |
| 历史工作记录 | `20260221-cc-1st/` | 2026-02-21 首次工作会话记录 |
| 任务进度 | `progress.md` | 持续更新的任务进度记录 |
