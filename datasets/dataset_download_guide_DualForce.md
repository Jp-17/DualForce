# DualForce 项目数据集下载与使用指南

> 本文档基于 `20260220_DualForce_v3.4.md` 方案，系统梳理项目所需全部数据集的下载方式、所需部分、与项目的匹配程度，以及具体使用建议。

---

## 目录

1. [数据集总览与匹配度评估](#1-数据集总览与匹配度评估)
2. [核心训练集（必选）](#2-核心训练集必选)
3. [推荐训练集（强烈建议）](#3-推荐训练集强烈建议)
4. [辅助/预训练数据集](#4-辅助预训练数据集)
5. [评估专用数据集](#5-评估专用数据集)
6. [下载优先级与存储规划](#6-下载优先级与存储规划)
7. [数据预处理流程](#7-数据预处理流程)
8. [常见问题与注意事项](#8-常见问题与注意事项)

---

## 1. 数据集总览与匹配度评估

| 数据集 | 匹配度 | 必要性 | 训练阶段 | 数据量 | 下载难度 |
|--------|--------|--------|---------|--------|---------|
| **HDTF** | ⭐⭐⭐⭐⭐ | 必选 | Stage 1 + 3 + 评估 | ~16h, 362 clips | 中（需从 YouTube 下载） |
| **CelebV-HQ** | ⭐⭐⭐⭐⭐ | 必选 | Stage 3 核心训练 | ~65h, 35,666 clips | 中（需从 YouTube 下载） |
| **VFHQ** | ⭐⭐⭐⭐ | 推荐 | Stage 3 辅助训练 | ~16,000 clips | 低（百度网盘/OneDrive 直接下载） |
| **TalkVid** | ⭐⭐⭐⭐ | 推荐（大规模） | Stage 3 扩展训练 | 1,244h, 7,729 人 | 中（YouTube + HuggingFace） |
| **VoxCeleb2** | ⭐⭐⭐ | 可选 | Stage 2 预训练 | 100万+ utterances | 中（需申请访问） |
| **MEAD** | ⭐⭐⭐ | 可选 | 评估 + 情绪实验 | 60 人, 8 种情绪 | 中（需申请访问） |
| **Panda-70M** | ⭐⭐ | 可选 | Stage 2 causal adapt. | 子集 5-10K clips | 低（Google Drive/HuggingFace） |
| **WebVid-10M** | ⭐⭐ | 可选（备选） | Stage 2 causal adapt. | 子集 5-10K clips | 中 |

### 匹配度评价标准

匹配度评估基于以下维度：

- **数据类型匹配**：是否为高清面部说话视频 + 对齐音频（DualForce 核心需求）
- **三元组可提取性**：能否提取 {video_latent, struct_latent, audio_feat} 三元组
- **质量与分辨率**：是否满足 ≥512×512 的训练要求
- **多样性贡献**：身份、语言、情绪、姿态的多样性
- **社区验证**：是否为 talking head 领域公认的标准数据集

---

## 2. 核心训练集（必选）

### 2.1 HDTF（High-Definition Talking Face）

**匹配度：⭐⭐⭐⭐⭐（完美匹配）**

HDTF 是 talking head 领域最通用的 benchmark 数据集，几乎所有 SOTA 方法（Hallo2/3、SadTalker、OVI 等）均使用此数据集进行训练和评估。对 DualForce 而言，HDTF 同时覆盖 Stage 1（3D Latent 提取）、Stage 3（核心训练）和评估三个阶段，是不可或缺的基础数据。

#### 基本信息

| 项目 | 详情 |
|------|------|
| 规模 | ~362 clips, ~15.8 小时, 300+ 说话人 |
| 分辨率 | 720P / 1080P（裁剪后 512×512） |
| 音频 | 高质量对齐语音 |
| 许可 | CC BY 4.0 |
| GitHub | `https://github.com/MRzzm/HDTF` |

#### 下载步骤

1. **克隆仓库获取元数据**：
   ```bash
   git clone https://github.com/MRzzm/HDTF.git
   cd HDTF/HDTF_dataset
   ```

2. **安装依赖**：
   ```bash
   pip install yt-dlp opencv-python
   # 确保已安装 ffmpeg
   ```

3. **从 YouTube 下载原始视频**：
   仓库中包含三组数据的元数据文件（`RD_video_url.txt`、`WDA_video_url.txt`、`WRA_video_url.txt`），每个文件包含 YouTube URL。使用 `yt-dlp` 按 `xx_resolution.txt` 中指定的分辨率下载：
   ```bash
   # 示例：下载 RD 组的视频
   while IFS= read -r line; do
     name=$(echo "$line" | cut -d' ' -f1)
     url=$(echo "$line" | cut -d' ' -f2)
     yt-dlp -f "bestvideo[height<=1080]+bestaudio" -o "${name}.%(ext)s" "$url"
   done < RD_video_url.txt
   ```

4. **转换格式并切分 clip**：
   - 将视频转为 mp4 格式并转为逐行扫描（progressive）
   - 按 `xx_annotion_time.txt` 中的时间戳切分为 talking head 片段
   - 命名规则：`video_name_clip_index.mp4`（如 `Radio11_0.mp4`）

5. **人脸裁剪**（两种方式）：
   - **方式 A（简单）**：使用 `xx_crop_wh.txt` 中的固定窗口裁剪
   - **方式 B（推荐）**：检测人脸关键点 → 计算最优裁剪窗口 → 按 `xx_crop_ratio.txt` 放大 → 裁剪 → 统一 resize 到 512×512

#### 需要下载的部分

- 全部三组（RD、WDA、WRA）的视频和元数据
- 建议全量下载（~362 clips），用于 Stage 1 + Stage 3 + 评估

#### 已知问题

- 部分 YouTube 视频可能已下架（Issue #28 报告了 9 个无效 URL）
- 若干裁剪参数可能有误，需对照检查
- **建议**：尽早下载并本地备份，避免后续视频下架

#### DualForce 中的使用方式

- **Stage 1**：全量提取 3D pseudo label（LivePortrait + EMOCA）
- **Stage 3**：作为核心训练数据的一部分（~16h）
- **评估**：标准 test split（~50 clips），计算 FVD、FID、Sync-C、Sync-D、ACD

---

### 2.2 CelebV-HQ

**匹配度：⭐⭐⭐⭐⭐（完美匹配）**

CelebV-HQ 提供 15,653 个不同身份的面部视频，身份多样性远超 HDTF。丰富的属性标注（83 个面部属性）可用于条件化训练实验。与 HDTF 联合使用是 DualForce Stage 3 核心训练的标准配置。

#### 基本信息

| 项目 | 详情 |
|------|------|
| 规模 | 35,666 clips, 15,653 identities, ~65 小时 |
| 分辨率 | ≥ 512×512 |
| 时长 | 3-20 秒/clip |
| 标注 | 83 个面部属性（40 外观 + 35 动作 + 8 情绪） |
| 许可 | CC BY 4.0（仅限非商业研究） |
| GitHub | `https://github.com/CelebV-HQ/CelebV-HQ` |

#### 下载步骤

1. **克隆仓库**：
   ```bash
   git clone https://github.com/CelebV-HQ/CelebV-HQ.git
   cd CelebV-HQ
   ```

2. **安装依赖**：
   ```bash
   pip install youtube_dl opencv-python
   ```

3. **运行下载脚本**：
   ```bash
   python download_and_process.py
   ```
   脚本会自动从 YouTube 下载视频并处理。可在代码中修改下载目录路径。

4. **人脸裁剪**（两种方式与 HDTF 类似）：
   - 使用元数据中的参考分辨率固定窗口裁剪
   - 或下载最高分辨率 → 人脸关键点检测 → 动态裁剪

#### 需要下载的部分

- 全部 35,666 clips 的视频
- `celebvhq_info.json` 标注文件（包含属性标注、边界框、时间戳）
- 建议全量下载以最大化身份多样性

#### DualForce 中的使用方式

- **Stage 1**：提取 3D pseudo label
- **Stage 3**：主训练数据（~65h），与 HDTF 联合组成 ~81h 的核心训练集
- **评估**：标准 test split（~5K clips），评估跨身份泛化能力
- **条件化实验**：利用 83 个属性标注分析特定情绪/动作下的生成质量

---

## 3. 推荐训练集（强烈建议）

### 3.1 VFHQ（Video Face High Quality）

**匹配度：⭐⭐⭐⭐（高度匹配）**

VFHQ 的核心价值在于画质极高且头部姿态多样（包含大角度侧面），这是 HDTF 和 CelebV-HQ 所欠缺的。对 DualForce 的 3D structure stream 来说，大角度头部运动的数据尤为重要，因为这正是 3D 先验发挥最大作用的场景。

#### 基本信息

| 项目 | 详情 |
|------|------|
| 规模 | 15,204 clips（原始）/ 15,381 clips（512×512 版） |
| 分辨率 | 高清以上（远高于 VoxCeleb） |
| 场景 | 访谈场景，多样化头部姿态和眼动 |
| 许可 | 仅限非商业研究 |
| 项目页 | `https://liangbinxie.github.io/projects/vfhq/` |

#### 下载步骤

VFHQ 提供**直接下载链接**（无需从 YouTube 爬取），是所有数据集中下载最便捷的。

**百度网盘下载（推荐国内用户）**：
- **VFHQ_zips**（2.8 TB）：原始分辨率视频，15,204 clips
- **VFHQ-512**（1.2 TB）：预处理好的 512×512 版本，15,381 clips ← **推荐直接下载此版本**
- **meta_info**（170 MB）：元数据，含视频 ID 和人脸关键点
- **VFHQ-Test**（2.37 GB）：论文中的测试集（100 clips）

**OneDrive 下载（备选）**：
- 处理脚本和 meta_info 也可通过 OneDrive 获取

#### 需要下载的部分

- **推荐**：下载 `VFHQ-512`（1.2 TB），已经是 512×512 预处理版本，可直接用于训练
- **必须**：下载 `meta_info` 以获取关键点信息
- **可选**：`VFHQ_zips` 原始版本（如需更高分辨率实验）

#### DualForce 中的使用方式

- **Stage 1**：提取 3D pseudo label（大角度姿态数据对 LivePortrait 提取质量很有价值）
- **Stage 3**：辅助训练数据（~50h），提升模型对大角度头部运动的鲁棒性
- **评估**：标准 test split，评估跨姿态鲁棒性

---

### 3.2 TalkVid

**匹配度：⭐⭐⭐⭐（高度匹配）**

TalkVid 是截至 2025 年最大的开源 talking head 数据集，种族/语言/年龄多样性极佳。其 Core 子集（160h）已足够满足 DualForce 大规模训练需求。对于需要 500h+ 训练数据的场景，TalkVid 是不可替代的扩展来源。

#### 基本信息

| 项目 | 详情 |
|------|------|
| 规模 | 7,729 说话人, 1,244+ 小时 HD/4K 视频 |
| 语言 | 15 种（英/中/阿/德/法/韩/日等） |
| 多样性 | 年龄 0-60+, 多种族, 性别均衡 |
| 子集 | TalkVid-Core: 160h 高纯度均衡子集 |
| 许可 | CC-BY-NC-4.0 |
| GitHub | `https://github.com/FreedomIntelligence/TalkVid` |
| HuggingFace | `https://huggingface.co/datasets/FreedomIntelligence/TalkVid` |

#### 下载步骤

1. **安装环境**：
   ```bash
   conda create -n talkvid python=3.10
   conda activate talkvid
   pip install -r requirements.txt
   conda install -c conda-forge 'ffmpeg<7' -y
   conda install torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
   ```

2. **获取元数据**：
   - 从 HuggingFace 下载 Parquet 格式的元数据文件
   - 或克隆 GitHub 仓库：
   ```bash
   git clone https://github.com/FreedomIntelligence/TalkVid.git
   ```

3. **下载视频**：
   ```bash
   cd data_pipeline/0_video_download
   python download_clips.py --input input.json --output output --limit 50
   ```
   注意：视频从 YouTube 下载，需要稳定的网络访问。

#### 需要下载的部分

- **推荐**：TalkVid-Core 子集（160h），这是经过质量筛选的高纯度数据，已满足 DualForce 的扩展训练需求
- **可选**：全量数据（1,244h），当需要更大规模训练或进行多样性实验时使用
- **必须**：TalkVid-Bench（500 clips）用于评估

#### DualForce 中的使用方式

- **Stage 3**：大规模扩展训练。与 HDTF + CelebV-HQ + VFHQ 联合使用达到 ~291h 的推荐配置
- 当需要扩展至 500h+ 时，从 TalkVid 全量数据中按质量筛选补充
- 多语言多样性有助于提升模型泛化能力

---

## 4. 辅助/预训练数据集

### 4.1 VoxCeleb2

**匹配度：⭐⭐⭐（中度匹配）**

VoxCeleb2 规模庞大（6,112 说话人，100 万+ utterances），但画质参差不齐，不适合直接用于 Stage 3 核心训练。仅建议在 Stage 2（MOVA-Lite Causal Adaptation）中使用其子集，提供大规模面部视频帮助模型适应 causal attention mask。

#### 基本信息

| 项目 | 详情 |
|------|------|
| 规模 | 6,112 说话人, 100 万+ utterances |
| 分辨率 | 中等（非高清为主） |
| 大小 | ~119 GB |
| 许可 | CC BY-SA 4.0 |
| 官网 | `https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html` |

#### 下载方式

- **官方渠道**：需在官网提交表单申请密码，然后使用 wget 下载
- **HuggingFace**：`https://huggingface.co/datasets/Reverb/voxceleb2`（备选）
- **注意**：官方源可能不再提供直接下载，可通过 torrent 或 HuggingFace 获取

#### 需要下载的部分

- 仅需下载子集（~10K clips）用于 Stage 2
- 不需要全量数据
- 建议优先下载 dev set 中画质较高的部分

#### DualForce 中的使用方式

- **仅 Stage 2**：MOVA-Lite Causal Adaptation，让模型适应 causal attention mask
- 不建议用于 Stage 3 核心训练（画质不够）

---

### 4.2 Panda-70M

**匹配度：⭐⭐（低度匹配）**

Panda-70M 是通用视频数据集（非面部专用），与 DualForce 的 talking head 任务匹配度较低。仅在 Stage 2 中使用其子集，让 MOVA-Lite 保留通用视频生成能力。

#### 基本信息

| 项目 | 详情 |
|------|------|
| 规模 | 70M 视频（含字幕） |
| 子集 | 2M / 10M / 全量 |
| 大小 | 全量 ~36 TB；2M 子集 ~1.6 TB |
| 许可 | 仅限非商业研究 |
| GitHub | `https://github.com/snap-research/Panda-70M` |
| HuggingFace | `https://huggingface.co/datasets/multimodalart/panda-70m` |

#### 下载方式

- **Google Drive**：仓库提供 CSV 元数据文件的直接下载链接
- **HuggingFace**：备选下载源
- 视频需根据元数据中的 URL 自行下载

#### 需要下载的部分

- **仅需 5-10K clips 的子集**，不需要全量数据
- 建议从 2M 子集的元数据中筛选适合的 clip
- 优先选择包含人物的视频内容

#### DualForce 中的使用方式

- **仅 Stage 2**：与 VoxCeleb2 子集联合使用，让 MOVA-Lite 适应 causal mask 同时保留通用能力
- Stage 2 仅需训练基础 causal video generation 能力，不需要面部特化数据

---

### 4.3 WebVid-10M（备选）

**匹配度：⭐⭐（低度匹配）**

作为 Panda-70M 的备选方案。如果 Panda-70M 下载困难，可改用 WebVid-10M 的子集。

#### 基本信息

| 项目 | 详情 |
|------|------|
| 规模 | 10M 视频-文本对 |
| 官网 | `https://maxbain.com/webvid-dataset/` |
| 许可 | 仅限研究用途 |

#### 需要下载的部分

- 仅需 5-10K clips 的子集
- 与 Panda-70M 二选一即可

---

## 5. 评估专用数据集

### 5.1 MEAD（Multi-view Emotional Audio-visual Dataset）

**匹配度：⭐⭐⭐（中度匹配，评估价值高）**

MEAD 的核心价值在于受控实验室环境 + 8 种情绪 × 3 种强度的标注，非常适合评估 DualForce 在不同情绪条件下的生成质量。但由于规模较小且环境单一，不适合作为训练数据。

#### 基本信息

| 项目 | 详情 |
|------|------|
| 规模 | 60 说话人（43 可用），8 种情绪 × 3 种强度 |
| 场景 | 实验室受控环境，7 个摄像机角度 |
| 许可 | 需签署使用协议 |
| 官网 | `https://wywu.github.io/projects/MEAD/MEAD.html` |
| GitHub | `https://github.com/uniBruce/Mead` |

#### 下载方式

- 访问官方项目页面，提交使用协议申请
- 获批后通过提供的链接下载

#### 需要下载的部分

- 标准 test split 用于评估
- 正面视角（frontal view）数据即可
- 不需要全部 7 个角度的数据

#### DualForce 中的使用方式

- **评估**：在受控条件下验证不同情绪的生成质量
- **情绪控制实验**：测试 DualForce 的 3D latent 对情绪变化的建模能力
- 评估指标：情绪分类准确率、Sync-D

---

## 6. 下载优先级与存储规划

### 6.1 推荐下载顺序

按照 DualForce 开发时间线，建议按以下顺序下载：

**第一批（Week 1-2，开发初期即需）**：
1. **HDTF** ← 最优先，Phase 0 验证首先在 HDTF 子集上进行
2. **CelebV-HQ** ← 与 HDTF 联合组成核心训练集

**第二批（Week 3-4，Stage 2 准备）**：
3. **VoxCeleb2 子集** 或 **Panda-70M 子集** ← Stage 2 causal adaptation 使用

**第三批（Week 5-6，扩展训练数据）**：
4. **VFHQ** ← 补充大角度姿态数据
5. **TalkVid-Core** ← 大规模扩展

**第四批（Week 9-10，评估阶段）**：
6. **MEAD** ← 情绪控制评估

### 6.2 存储需求估算

| 数据集 | 原始视频大小 | 预处理后 Feature Cache | 说明 |
|--------|-------------|----------------------|------|
| HDTF | ~50 GB | ~100 GB | 16h × ~6 GB/h |
| CelebV-HQ | ~200 GB | ~430 GB | 65h × ~6.6 GB/h |
| VFHQ-512 | 1.2 TB | ~330 GB | 50h（可用部分）× ~6.6 GB/h |
| TalkVid-Core | ~500 GB | ~1 TB | 160h × ~6.6 GB/h |
| VoxCeleb2 子集 | ~30 GB | ~60 GB | 仅 Stage 2 |
| Panda-70M 子集 | ~50 GB | ~100 GB | 仅 Stage 2 |
| MEAD | ~20 GB | — | 仅评估 |

**最小配置**（HDTF + CelebV-HQ）：
- 原始数据：~250 GB
- Feature cache：~530 GB
- 合计：~800 GB

**推荐配置**（+ VFHQ + TalkVid-Core）：
- 原始数据：~2 TB
- Feature cache：~2 TB
- 合计：~4 TB

**完整配置**（全部数据集）：
- 合计：~6-8 TB

### 6.3 各训练阶段数据配置方案

#### 方案 A：最小配置（4 卡 A100，学术验证）

```
Stage 1: HDTF (16h) 全量
Stage 2: 跳过（直接在 HDTF 上 causal adaptation）
Stage 3: HDTF (16h) + CelebV-HQ (65h) = ~81h
评估:   HDTF test set + CelebV-HQ test set
```

#### 方案 B：推荐配置（8 卡 A100，500h 目标）

```
Stage 1: HDTF (16h) + CelebV-HQ (65h) + VFHQ (部分) → 全量提取 3D pseudo label
Stage 2: Panda-70M 子集 (5K clips) + VoxCeleb2 子集 (10K clips)
Stage 3:
  Primary:   HDTF (~16h) + CelebV-HQ (~65h)       = ~81h   ← 必选
  Secondary: VFHQ (~50h) + TalkVid-Core (~160h)    = ~210h  ← 推荐
  Total: ~291h
评估: HDTF + CelebV-HQ + VFHQ + MEAD test sets
```

---

## 7. 数据预处理流程

所有数据集下载完成后，需要经过统一的预处理 pipeline 生成可直接加载的 feature cache。

### 7.1 预处理步骤概览

```
原始视频 + 音频
  → 人脸检测 & 跟踪 (RetinaFace)
  → 人脸对齐 & 裁剪 (512×512)
  → 质量过滤 (模糊/遮挡/多人脸)
  → 统一 FPS (25fps)
  → Feature 离线提取:
      - Video VAE Encoder → video_latents [T, C, H/8, W/8]
      - LivePortrait Motion Extractor → struct_latents [T, D_s=128]
      - HuBERT-Large → audio_features [T_audio, 1024]
      - EMOCA → flame_params [T, 159]
      - DINOv2/CLIP → ref_features [N_ref, D]
  → 保存为 .safetensors 格式
```

### 7.2 质量过滤标准

**硬过滤（直接丢弃）**：
- 分辨率 < 256×256（裁剪后人脸区域）
- 检测到多张人脸（非单人 talking head）
- 音频-视频时间偏移 > 100ms
- 视频时长 < 2s
- 人脸被遮挡 > 30%

**软过滤（降低采样权重）**：
- 模糊度得分低（Laplacian variance 检测）
- 头部偏转角 > 45°
- 音频 SNR < 10dB
- 唇动幅度极小（非说话状态）

### 7.3 各数据集的特殊注意事项

| 数据集 | 特殊处理 |
|--------|---------|
| HDTF | 需自行编写 YouTube 下载 + 切分脚本；注意检查已失效的 URL |
| CelebV-HQ | 部分 clip 很短（3s），可能需要过滤 < 2s 的片段；包含非说话场景需额外过滤 |
| VFHQ | 512 版本已预处理好，但非所有 clip 有清晰音频，需对音频质量做额外检查 |
| TalkVid | 15 种语言的音频特征可能存在差异，HuBERT 对非英语的表征质量需验证 |
| VoxCeleb2 | 画质参差不齐，需严格过滤低分辨率视频 |

---

## 8. 常见问题与注意事项

### Q1: YouTube 视频下架怎么办？

HDTF、CelebV-HQ、TalkVid 的视频源均为 YouTube，随时可能下架。建议：
- **尽早下载并本地备份**
- 优先使用提供完整文件下载的数据集（如 VFHQ 的百度网盘/OneDrive 链接）
- 社区中可能有已下载的备份版本，可在相关论坛/群组中寻找

### Q2: 存储空间不足怎么办？

- Feature cache 使用 float16 压缩（video_latents、audio_features、ref_features 均可用 float16）
- 500h 数据的 feature cache 可从 ~3.3 TB 压缩到 ~2 TB
- 如果空间紧张，优先使用"最小配置"方案（~800 GB）

### Q3: 3D Latent 质量不稳定怎么办？

LivePortrait 提取的 3D latent 在极端姿态或遮挡场景下可能不稳定。缓解策略：
- 对比 EMOCA 的 FLAME 参数做一致性过滤
- 丢弃两者偏差过大的帧
- 在训练中使用 FLAME alignment loss 作为辅助约束

### Q4: 音频-视频对齐问题如何处理？

- 预处理阶段强制对齐验证（< 40ms 容忍度）
- 训练时加入时间抖动增强鲁棒性（随机偏移 ±2 帧）
- 使用 Silero VAD 检测静音段，确保音频与唇动区间匹配

### Q5: 不同数据集之间的分辨率/FPS 不一致？

统一预处理标准：
- 分辨率：全部裁剪并 resize 到 512×512
- FPS：全部统一为 25fps
- 音频采样率：全部重采样到 16kHz

---

*文档生成日期: 2026-02-21*
*对应项目文档: 20260220_DualForce_v3.4.md*
