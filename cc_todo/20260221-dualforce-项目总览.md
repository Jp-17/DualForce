# DualForce 项目总览：已完成工作、当前状态与待办事项

> 更新日期：2026-02-21
> 项目：DualForce - 基于3D结构感知的自回归扩散模型，用于说话人头像视频生成
> 基础：MOVA (OpenMOSS) 代码库的 fork

---

## 一、项目目标

在 MOVA 的双塔（视频+音频）扩散架构基础上，构建 DualForce 模型：

1. **Diffusion Forcing**：每帧独立噪声水平，实现原生自回归生成（无需蒸馏）
2. **隐式3D结构潜变量**：用 LivePortrait 运动特征替代音频塔，保持面部结构一致性
3. **浅层融合架构**：视频↔3D结构在浅层双向注意力，深层独立处理

最终目标：完成实验并撰写论文投稿（NeurIPS 2026 / CVPR 2027 / ECCV 2026）

---

## 二、已完成的工作

### Phase 0：环境搭建与基线验证

| 任务 | 状态 | 说明 |
|------|------|------|
| MOVA conda 环境搭建 | ✅ 完成 | Python 3.13, PyTorch 2.10+cu124 |
| MOVA-360p 权重下载 | ✅ 完成 | ~50GB, 位于 `autodl-tmp/checkpoints/MOVA-360p/` |
| MOVA 推理测试 | ✅ 完成 | RTX 4090D, `--offload group` 模式，成功生成 `single_person.mp4` |
| MOVA 代码库分析 | ✅ 完成 | 详见 `cc_todo/20260221-mova-codebase-analysis.md` |
| 执行计划制定 | ✅ 完成 | 详见 `cc_todo/20260221-dualforce-execution-plan.md` |

**关键发现**：MOVA-360p 实际架构与代码默认值差异巨大（dim=5120 非 3072, 40层非30层, patch_size=(1,2,2) 非 (2,2,2), bridge 策略 "full" 非 "shallow_focus", audio 采样率 48kHz 非 44.1kHz）。所有设计决策已基于实际 checkpoint 配置修正。

### Phase 1：DualForce 核心架构（代码完成，待GPU验证）

| 组件 | 文件 | 说明 |
|------|------|------|
| MOVA-Lite 视频 DiT | `wan_video_dit.py`（修改） | dim=1536, 20层, heads=12, 含 block-causal 注意力 |
| 3D 结构 DiT | `models/wan_struct_dit.py`（新建） | 替代音频塔, MLP tokenizer, 1D RoPE |
| Diffusion Forcing 调度器 | `schedulers/diffusion_forcing.py`（新建） | 每帧独立 sigma, 滑动窗口推理 |
| DualForce 训练流水线 | `pipelines/dualforce_train.py`（新建） | 双塔前向+bridge, 多模态损失 |
| DualForce 推理流水线 | `pipelines/pipeline_dualforce.py`（新建） | 滑动窗口自回归, CFG, 双塔推理 |
| KV-Cache | `models/kv_cache.py`（新建） | MultiModalKVCache + CachedSelfAttention |
| Block-Causal 注意力 | `wan_video_dit.py`（修改） | `_build_block_causal_mask`, 帧 t 只能看到 ≤t |
| 工厂函数 | `dualforce_train.py` 底部 | `DualForceTrain_from_pretrained` 注册到 DIFFUSION_PIPELINES |
| 训练配置 | `configs/dualforce/dualforce_train_8gpu.py` | 完整训练超参数 |
| FSDP 配置 | `configs/dualforce/accelerate/fsdp_8gpu.yaml` | 8-GPU 分布式训练 |
| 训练脚本 | `scripts/training_scripts/dualforce_train.py` | Accelerate 启动器 |
| 训练 Shell | `scripts/training_scripts/dualforce_train_8gpu.sh` | Bash 一键启动 |
| 推理脚本 | `scripts/dualforce_inference.py` | CLI 工具，输入参考图+prompt |

### Phase 3：多模态架构扩展（代码完成）

| 组件 | 文件 | 说明 |
|------|------|------|
| 音频条件化模块 | `models/audio_conditioning.py`（新建） | AudioProjector + 门控交叉注意力 |
| DualAdaLNZero | `models/audio_conditioning.py` | 每帧 sigma → 正弦编码 → MLP → 6*dim 调制向量 |
| 门控残差连接 | `audio_conditioning.py` | sigmoid(gate) * attn_out, gate 初始化为 0 |
| 浅层音频融合 | `dualforce_train.py`（更新） | 前6层注入音频条件 |

**设计决策**：
- 音频条件化使用零初始化门控，训练初期无影响，逐步学习
- 仅在浅层（前6层）注入音频信息，深层不受音频干扰

### Phase 4：训练损失函数（代码完成）

| 损失 | 权重 | 说明 |
|------|------|------|
| L_video（视频流匹配） | 1.0 | MSE(v_pred, v_target) |
| L_struct（结构流匹配） | 0.5 | MSE(s_pred, s_target) |
| L_flame（FLAME 对齐） | 0.1 | 结构 x0 估计 → 线性投影 → MSE(pred_flame, gt_flame) |
| L_lip_sync（唇音同步） | 0.3 | 对比损失，256维共享嵌入空间 |

### Phase 5：评估流水线与消融实验（代码完成）

**5大评估指标**：

| 指标 | 脚本 | 模型/方法 |
|------|------|----------|
| FVD（视频质量） | `scripts/eval/compute_fvd.py` | R3D-18 视频特征 |
| FID（帧质量） | `scripts/eval/compute_fid.py` | Inception-v3 帧特征 |
| ACD/CSIM（身份保持） | `scripts/eval/compute_identity.py` | ArcFace (insightface) |
| Sync-C/Sync-D（唇音同步） | `scripts/eval/compute_sync.py` | SyncNet |
| APD（头部姿态） | `scripts/eval/compute_pose.py` | MediaPipe FaceMesh |

**评估编排器**：`scripts/eval/run_eval.py` — 支持端到端（生成+评估）和纯指标模式

**7组消融实验配置**：

| 消融 | 配置文件 |
|------|---------|
| 无结构流 | `configs/dualforce/ablations/no_struct_stream.py` |
| 单向 Bridge | `configs/dualforce/ablations/unidirectional_bridge.py` |
| 融合深度=4 | `configs/dualforce/ablations/fusion_depth_4.py` |
| 融合深度=12 | `configs/dualforce/ablations/fusion_depth_12.py` |
| sigma_s=0.5 | `configs/dualforce/ablations/sigma_s_0.5.py` |
| sigma_s=1.0 | `configs/dualforce/ablations/sigma_s_1.0.py` |
| 长序列评估 | `configs/dualforce/ablations/long_sequence_eval.py` |

### 数据相关（脚本完成，待实际运行）

| 组件 | 文件 | 说明 |
|------|------|------|
| 多模态数据集 | `mova/datasets/dualforce_dataset.py` | 加载 .safetensors 预提取特征 |
| 预处理流水线（7步） | `scripts/preprocess/01-07_*.py` | 人脸检测→裁剪→质量过滤→5种特征提取→metadata |
| HDTF 下载脚本 | `scripts/download/download_hdtf.py` | yt-dlp 并行下载+切片 |
| CelebV-HQ 下载脚本 | `scripts/download/download_celebvhq.py` | YouTube 下载 |
| 环境安装脚本 | `scripts/setup_env.sh` | 一键安装所有依赖 |
| GPU 验证脚本 | `scripts/verify_dualforce.py` | 6项测试（前向、KV-cache、内存） |
| MOVA 推理验证脚本 | `scripts/verify_mova_inference.py` | checkpoint 检查+推理 |

---

## 三、已获得的结果

### 代码层面

- **51个文件变更，新增 12,747 行代码**（`git diff --stat 7803632..HEAD`）
- **14次 Git 提交**，从 `a14eb9a`（分析文档）到 `d42f720`（最终状态更新）
- MOVA-360p 推理验证通过，确认基线权重可用

### 架构设计

- **MOVA-Lite 参数量估算**：
  - 视频 DiT (dim=1536, 20层): ~617M
  - 结构 DiT (同规格): ~617M
  - Bridge (20层交互): ~200M
  - 音频条件化 + DualAdaLNZero: ~43M
  - **可训练总计: ~1.5B**
  - 冻结模块 (Video VAE + Text Encoder): ~3.2B

### 文档

- `20260221-mova-codebase-analysis.md` — MOVA 架构深度分析（含 checkpoint 实际值修正）
- `20260221-dualforce-execution-plan.md` — 6阶段执行计划
- `20260221-dualforce-progress-log.md` — 详细进度跟踪（8个 session 记录）
- `2026-02-21-MOVA-环境搭建与推理测试.md` — 环境搭建和 MOVA 推理记录

---

## 四、待完成的工作

### 🔴 紧急（需要 GPU 环境）

| 优先级 | 任务 | 预计输入 | 验证标准 |
|--------|------|---------|---------|
| P0 | 运行 `verify_dualforce.py` 验证前向传播 | 随机输入 | 输出 shape 正确，loss 可计算 |
| P0 | KV-cache 一致性验证 | 对比 cached vs non-cached | 输出差异 < 1e-5 |
| P0 | 全流水线内存估算 | 20层双塔 | <50GB VRAM (8x A100) |

### 🟡 数据准备

| 优先级 | 任务 | 数据量 | 备注 |
|--------|------|--------|------|
| P1 | 下载 HDTF 数据集 | ~362 clips, ~16h | YouTube 链接可能失效，尽快执行 |
| P1 | 下载 CelebV-HQ 数据集 | ~35K clips, ~65h | 核心训练数据 |
| P2 | 下载 VFHQ-512 数据集 | ~15K clips | 已有预处理好的 512×512 版本 |
| P2 | 下载 VoxCeleb2 / Panda-70M 子集 | 5-10K clips | Phase 2 causal pretraining 用 |
| P1 | 运行预处理流水线 | HDTF 先行 | 验证端到端：原始视频 → safetensors 特征 |
| P1 | 安装 LivePortrait + EMOCA | — | 结构潜变量和 FLAME 参数提取的前置依赖 |

### 🟢 训练阶段

| 阶段 | 任务 | 预计周期 | 依赖 |
|------|------|---------|------|
| Phase 2 | Causal 视频预训练（纯视频，causal attention） | 3-5天 (8xA100) | 通用视频数据 |
| Phase 4 | 多模态 Diffusion Forcing 训练 | 1-2周 (8xA100) | HDTF + CelebV-HQ 预处理完成 |
| Phase 4 | 监控训练里程碑 (1K/5K/20K/50K步) | 持续 | 训练启动后 |

### 🔵 评估与论文

| 任务 | 说明 | 依赖 |
|------|------|------|
| 运行5指标评估 | FVD, FID, ACD/CSIM, Sync-C/D, APD | 训练完成的模型 |
| 运行7组消融实验 | 每组单独训练+评估 | 基线模型训练完成 |
| 与 baseline 对比 | Hallo2/3, SadTalker, EMO 等 | 评估脚本 + 对比数据 |
| 长序列一致性测试 | 32/128/256/512 帧 | 训练完成的模型 |
| 撰写论文 | NeurIPS/CVPR 格式 | 全部实验结果 |

---

## 五、代码文件清单

### 新建文件（DualForce 专属）

```
configs/dualforce/
├── dualforce_train_8gpu.py                    # 训练配置
├── accelerate/fsdp_8gpu.yaml                  # FSDP 分布式配置
└── ablations/                                 # 7组消融实验配置
    ├── no_struct_stream.py
    ├── unidirectional_bridge.py
    ├── fusion_depth_4.py
    ├── fusion_depth_12.py
    ├── sigma_s_0.5.py
    ├── sigma_s_1.0.py
    └── long_sequence_eval.py

mova/diffusion/models/
├── wan_struct_dit.py                          # 3D 结构 DiT
├── kv_cache.py                                # KV-Cache
└── audio_conditioning.py                      # 音频条件化 + DualAdaLNZero

mova/diffusion/schedulers/
└── diffusion_forcing.py                       # Diffusion Forcing 调度器

mova/diffusion/pipelines/
├── dualforce_train.py                         # 训练流水线
└── pipeline_dualforce.py                      # 推理流水线

mova/datasets/
└── dualforce_dataset.py                       # 多模态数据集

scripts/
├── dualforce_inference.py                     # 推理 CLI
├── verify_dualforce.py                        # GPU 验证（6项测试）
├── verify_mova_inference.py                   # MOVA 推理验证
├── setup_env.sh                               # 环境安装
├── training_scripts/
│   ├── dualforce_train.py                     # 训练入口
│   └── dualforce_train_8gpu.sh                # 8-GPU 启动脚本
├── preprocess/
│   ├── 01_face_detect_crop.py                 # 人脸检测与裁剪
│   ├── 02_quality_filter.py                   # 质量过滤
│   ├── 03_extract_video_latents.py            # Video VAE 编码
│   ├── 04_extract_struct_latents.py           # LivePortrait 提取
│   ├── 05_extract_flame_params.py             # EMOCA FLAME 提取
│   ├── 06_extract_audio_features.py           # HuBERT 特征提取
│   ├── 07_build_metadata.py                   # 元数据构建
│   └── run_pipeline.sh                        # 流水线一键运行
├── download/
│   ├── download_hdtf.py                       # HDTF 下载
│   └── download_celebvhq.py                   # CelebV-HQ 下载
└── eval/
    ├── compute_fvd.py                         # FVD 指标
    ├── compute_fid.py                         # FID 指标
    ├── compute_identity.py                    # ACD/CSIM 指标
    ├── compute_sync.py                        # Sync-C/D 指标
    ├── compute_pose.py                        # APD 指标
    ├── batch_generate.py                      # 批量生成
    └── run_eval.py                            # 评估编排器
```

### 修改的 MOVA 原有文件

| 文件 | 修改内容 |
|------|---------|
| `mova/diffusion/models/wan_video_dit.py` | 添加 block-causal 注意力掩码, `causal_temporal` 配置 |
| `mova/diffusion/models/__init__.py` | 注册 WanStructModel, KV-Cache, AudioConditioning |
| `mova/diffusion/schedulers/__init__.py` | 注册 DiffusionForcingScheduler |
| `mova/diffusion/pipelines/__init__.py` | 注册 DualForceTrain, DualForceInference |
| `mova/datasets/__init__.py` | 注册 DualForceDataset |
| `mova/engine/trainer/accelerate/accelerate_trainer.py` | batch-key 无关化, DualForce 指标支持 |

---

## 六、Git 提交历史

| 提交 | 说明 |
|------|------|
| `d42f720` | 更新进度日志：所有 pre-GPU 代码完成 |
| `dcf39f1` | 添加 7 组消融实验配置 (Phase 5.2) |
| `e13d88d` | 实现 FLAME 对齐和唇音同步对比损失 |
| `160d2e2` | 添加批量生成脚本，清理未使用的导入 |
| `5a9d66b` | 重写推理流水线：双塔 + CFG + 音频条件化 |
| `7056b56` | 添加评估流水线：FVD, FID, ACD/CSIM, Sync-C/D, APD |
| `9b5bcd7` | 添加音频条件化模块和 per-frame AdaLN (Phase 3) |
| `5f404cf` | 添加 CelebV-HQ 下载脚本和环境安装脚本 |
| `98aefc6` | 修复 bug，添加验证脚本和 HDTF 下载工具 |
| `b4d5899` | 添加 FSDP 配置和启动脚本；Phase 1 代码完成 |
| `222b2ed` | 添加训练/推理脚本、工厂函数、训练器泛化 |
| `7567c46` | 添加数据集、预处理流水线、causal 注意力、KV-cache |
| `f505ce6` | 添加 DualForce 核心架构：struct DiT, DF 调度器, 训练流水线 |
| `a14eb9a` | 添加项目分析和执行计划文档 |

---

## 七、风险与注意事项

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 从头训练不收敛 | 中 | 高 | 渐进式训练，先 causal pretrain 再多模态 |
| 3D 潜变量质量不稳 | 中 | 高 | FLAME 交叉校验过滤，软对齐损失 |
| YouTube 视频下架 | 高 | 中 | 尽快下载，优先使用直接下载数据集 |
| 8x A100 OOM | 低 | 中 | 梯度检查点 + CPU offload 已配置 |
| LivePortrait/EMOCA 兼容性问题 | 中 | 中 | 已准备 MediaPipe fallback |

---

## 八、下一步行动

**当务之急（按顺序执行）**：

1. **在 GPU 服务器上运行 `scripts/verify_dualforce.py`** — 验证全部架构代码
2. **运行 `scripts/verify_mova_inference.py`** — 确认 MOVA 基线权重可用
3. **执行 HDTF 数据集下载** — YouTube 链接时效性强
4. **安装 LivePortrait + EMOCA** — 结构特征提取前置
5. **对 5-10 个样本运行完整预处理流水线** — 端到端验证
6. **启动 Phase 2 causal 视频预训练** — 数据准备好后立即开始

---

*文档生成日期: 2026-02-21*
