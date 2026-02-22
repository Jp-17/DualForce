# DualForce 项目 Claude Code 指引

## 项目背景

**DualForce** 是一个基于 MOVA（OpenMOSS 开源的视频-音频同步生成基础模型）进行深度改造的研究项目，目标是实现 **3D-Aware Native Autoregressive Diffusion for Real-Time Avatar Video Generation**。

核心思路是在 Diffusion Forcing 框架下，将隐式 3D 结构 latent 作为与 video latent 平行的模态，通过浅层双向 attention 融合 + 深层独立特化实现联合自回归生成。推理时无需任何显式 3D 输入，仅需参考图片和音频即可自回归生成 talking head 视频。

> 项目技术细节详见 `cc_core_files/proposal.md`；执行计划详见 `cc_core_files/plan.md`。

---

## 当前项目状态

项目处于**计划优化阶段**，尚未进入正式代码改造/项目执行。当前正在进一步优化和修正以下项目文档：

- 研究方案设计 — `cc_core_files/proposal.md`
- MOVA 代码库分析 — `cc_core_files/code_research.md`
- 数据集选型与规划 — `cc_core_files/dataset.md`
- 执行计划制定 — `cc_core_files/plan.md`

待以上文档完全确定后，方可正式开始代码改造和项目执行。

### 项目规划与执行进展

- **项目详细规划**参考 `cc_core_files/proposal.md`
- **项目执行计划**参考 `cc_core_files/plan.md`（按 plan.md 进行规划和执行，各阶段任务完成后在 plan.md 中打勾标记）

---

## 仓库结构

当前仓库为 MOVA 原始代码库，DualForce 改造尚未开始。

```
项目根目录/
├── mova/                               # MOVA 核心包
│   ├── diffusion/                      # 扩散模型（models / pipelines / schedulers）
│   ├── datasets/                       # 数据集
│   └── engine/                         # 训练基础设施（FSDP, Accelerate, LoRA）
├── configs/                            # 训练配置
├── scripts/                            # 脚本（推理 / 训练 / 数据处理等）
├── workflow/                           # Streamlit 前端 UI
├── checkpoints/                        # 模型权重
├── data/                               # 数据集存放（raw / processed / generated 等）
├── results/                            # 实���结果（可视化图表等）
├── cc_core_files/                      # 项目核心文档
│   ├── proposal.md                     # DualForce 研究方案
│   ├── plan.md                         # 执行计划
│   ├── code_research.md                # MOVA 代码库分析
│   ├── dataset.md                      # 数据集信息
│   ├── scripts.md                      # 脚本功能索引
│   ├── data.md                         # 数据管理说明
│   └── results.md                      # 实验结果说明
├── 20260221-cc-1st/                    # 历史工作会话记录（仅供参考，代码已撤销）
├── claude.md                           # 本文件
└── progress.md                         # 任务进度记录
```

---

## 任务执行规范

### 1. progress.md 维护

每次任务执行时：
- **任务开始前**：查阅 `progress.md` 了解之前的工作内容、经验和待解决问题
- **任务执行中**：记录当前正在做的事情
- **任务完成后**：立即更新 `progress.md`，包含：
  - 日期和时间（格式：`## YYYY-MM-DD HH:MM — 任务标题`）
  - 任务内容描述
  - 任务结果和效果
  - 遇到的问题及解决方法

注意：`progress.md` 只记录已完成的任务进展，不记录待开展工作（待开展工作由 `plan.md` 管理）。

### 2. plan.md 联动

如果当前执行的任务恰好对应 `cc_core_files/plan.md` 中的某项任务，则在 plan.md 的对应位置记录该任务的执行情况：
- 完成状态（已完成 / 进行中 / 待完成）
- 已做到什么程度
- 接下来还需要做什么

### 3. claude.md 维护

- 任务完成后检查 `claude.md` 是否存在过时内容
- 如有过时内容，根据最新的任务结果或已调整的项目实现进行更新
- 多次遇到的问题应沉淀为经验记录到本文件的"经验沉淀"部分

### 4. Git 提交规范

- 每完成一个任务或小模块后，及时执行 `git add & commit & push`
- **Git 帐户**：jpagkr@163.com / Jp-17
- **commit 消息**：主要使用中文
- **文件/文件夹命名**：使用英文
- **md 文档内容**：主要使用中文
- **md 文档名称**：以文档产出日期开头（如 `20260223-xxx.md`）

### 5. 项目资产管理

#### 脚本（scripts/）

数据处理、分析、项目运行、测试等脚本放在 `scripts/` 下的合适位置。每次新增或修改脚本时，**必须**在 `cc_core_files/scripts.md` 中记录：
- 脚本的功能用途
- 创建时间
- 使用方式（命令示例）
- 存储位置

#### 数据（data/）

下载的数据集放在 `data/` 下的合适位置，按数据类型做好分级管理，例如：
- `data/raw/` — 原始下载数据
- `data/processed/` — 预处理后的数据
- `data/generated/` — 模型生成的数据

每次新增数据时，**必须**在 `cc_core_files/data.md` 中记录：
- 数据的来源和含义
- 处理流程
- 存储位置和格式

#### 实验结果（results/）

代码实验分析的结果（可视化图表等）放在 `results/` 下的合适位置。每次新增结果时，**必须**在 `cc_core_files/results.md` 中记录：
- 产生方式（用什么脚本/命令生成）
- 实验目的
- 结果分析与结论

---

## 经验沉淀

（随项目推进持续积累）

---

## 关键参考文档

| 文档 | 路径 | 说明 |
|------|------|------|
| 研究方案 | `cc_core_files/proposal.md` | DualForce 完整技术方案 |
| 执行计划 | `cc_core_files/plan.md` | 项目执行计划（按此规划和执行） |
| 代码分析 | `cc_core_files/code_research.md` | MOVA 代码库分析 |
| 数据集信息 | `cc_core_files/dataset.md` | 数据集选型与规划 |
| 脚本索引 | `cc_core_files/scripts.md` | 脚本功能、用法、位置 |
| 数据管理 | `cc_core_files/data.md` | 数据来源、处理、位置 |
| 实验结果 | `cc_core_files/results.md` | 实验结果与分析 |
| 任务进度 | `progress.md` | 每次任务的执行详情记录 |
