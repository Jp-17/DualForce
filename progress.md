# DualForce 项目进度记录

> 项目：DualForce - 3D-Aware Native Autoregressive Diffusion for Real-Time Avatar Video Generation
> 基于：MOVA (OpenMOSS) 代码库
> 开始日期：2026-02-21

---

## 2026-02-21 — 项目启动与代码开发

### 任务内容
在单日内完成了 DualForce 项目从代码分析到核心架构实现的全部预 GPU 工作。

### 完成事项
1. **MOVA 代码库分析**：深入研究 Wan 2.2 DiT 架构、Bridge CrossAttention、训练基础设施
2. **环境搭建**：在 RTX 4090D (24GB) 上创建 conda 环境 `mova`（Python 3.13），安装 PyTorch 2.10 + CUDA 12.4
3. **MOVA 推理验证**：成功使用 `--offload group` 策略完成 MOVA-360p 推理（仅需 12GB 显存）
4. **核心架构代码**（Phase 1）：WanStructModel、DiffusionForcingScheduler、DualForceTrain、Block-Causal Attention、KV-Cache
5. **数据处理**：7步预处理流水线 + HDTF/CelebV-HQ 下载脚本
6. **音频条件化**（Phase 3）：AudioConditioningModule、DualAdaLNZero、门控交叉注意力
7. **训练损失**（Phase 4）：L_video、L_struct、L_flame、L_lip_sync
8. **评估流水线**（Phase 5）：FVD、FID、ACD/CSIM、Sync-C/D、APD + 7组消融配置
9. **推理流水线**：滑动窗口自回归 + CFG + 双塔前向传播

### 结果
- 约 12,747 行新增代码，51 个文件变更，14 次功能提交
- 所有代码层面的工作已完成，等待 GPU 环境进行验证

### 遇到的问题与解决
1. **MOVA checkpoint 参数与代码默认值不符**：发现 dim=5120/40层（非默认的 3072/30层）→ 以 checkpoint 实际配置为准
2. **bitsandbytes 导入问题**：MOVA 无条件导入了训练依赖 → 即使推理也需安装 bitsandbytes
3. **torch.autocast 硬编码 "cuda"**：在非 CUDA 环境报错 → 改为动态设备检测

---

## 2026-02-23 — 项目文档规范化

### 任务内容
创建 `claude.md` 和 `progress.md`，建立项目文档规范。

### 完成事项
1. 创建 `claude.md`：包含项目背景、结构概览、当前状态、任务执行规范、经验沉淀
2. 创建 `progress.md`：整合历史工作记录，建立持续更新的进度跟踪机制
3. 整理 `20260221-cc-1st/` 中的历史工作记录

### 结果
- 建立了完整的项目文档规范体系
- 后续每次任务将按规范更新 progress.md

### 备注
- 历史详细工作记录参见 `20260221-cc-1st/20260221-dualforce-progress-log.md`
- 仓库现状总结参见 `20260221-cc-1st/20260223-工作记录整理与仓库现状总结.md`

---

## 待完成任务（按优先级）

| 优先级 | 任务 | 依赖 |
|--------|------|------|
| P0 | 运行 `verify_dualforce.py` 验证前向传播、KV-cache 一致性、内存估算 | GPU 环境（8×A100） |
| P1 | 下载 HDTF + CelebV-HQ 数据集 | 网络 |
| P1 | 安装 LivePortrait + EMOCA，运行完整预处理流水线 | 数据集 + GPU |
| P2 | Phase 2: Causal 视频预训练（3-5天，8×A100） | 数据 + GPU |
| P2 | Phase 4: 多模态 Diffusion Forcing 训练（1-2周，8×A100） | Phase 2 |
| P3 | 评估 + 消融实验 + 论文撰写 | 训练完成 |

---

*本文件持续更新，记录每次任务的执行详情。*
