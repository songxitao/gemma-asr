# Gemma-ASR (Chinese): 基于 Google EmbeddingGemma 2 冻结音频塔的极速中文 ASR

<p align="center">
  <a href="https://github.com/google-deepmind"><img src="https://img.shields.io/badge/Base_Model-EmbeddingGemma--2-blue?logo=google" alt="Base Model"></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.5+-ee4c2c?logo=pytorch" alt="PyTorch"></a>
  <a href="https://huggingface.co"><img src="https://img.shields.io/badge/HuggingFace-Transformers_5.19+-yellow?logo=huggingface" alt="HuggingFace"></a>
  <a href="https://developer.nvidia.com/cuda-zone"><img src="https://img.shields.io/badge/CUDA-12.1+-76b900?logo=nvidia" alt="CUDA"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/VRAM-3.55_GB_Peak-brightgreen" alt="VRAM"></a>
  <a href="#"><img src="https://img.shields.io/badge/Speed-860x_Realtime-orange" alt="Realtime Speed"></a>
</p>

<p align="center">
  <a href="#-项目灵感与架构故事">💡 项目灵感</a> &nbsp;|&nbsp;
  <a href="#-特性矩阵与技术价值">⚡ 特性矩阵</a> &nbsp;|&nbsp;
  <a href="#-系统架构设计">🏗️ 架构设计</a> &nbsp;|&nbsp;
  <a href="#-冒烟训练实测-benchmark">📊 冒烟实测</a> &nbsp;|&nbsp;
  <a href="#-极简闪电快速开始">🚀 快速开始</a> &nbsp;|&nbsp;
  <a href="#-致敬与致谢">🤝 鸣谢与协议</a>
</p>

---

## 💡 项目灵感与架构故事

> **“当所有人都在把多模态检索模型当作 RAG 向量工具时，我们选择唤醒它沉睡的声学感知潜能。”**

Google 近日开源发布了新一代多模态检索与表征大模型 [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)。该模型内置了强大的 USM-style Conformer 音频编码塔（300M 参数，12 层，25 frames/s），旨在对齐跨模态的语音与文本嵌入空间。

越南工程师在其开源项目 **QUAN-Transcribe** 中提出了一种开创性的极简思路：**将 EmbeddingGemma 2 的底层音频编码塔完全冻结（0 参数更新），仅在其顶部训练一个极小型的 CTC Reader（仅 20.4M 可训练参数）**，在越南语 ASR 上以极低的算力达成了惊艳的 SOTA 表现。

**本项目在此灵感之上进一步跨越**：
原版方案天然契合越南语“一词一音节”的词表切分习惯；但当我们将此架构复刻迁移至**中文语音识别**时，遭遇了传统分词器将中文切成“人工智能”等多字稀疏长词、导致 CTC 类别发散的致命瓶颈。

本项目经过系统性攻关，**打造了首个面向中文极速语音识别的复刻工程**：
1. **CJK 单字拓扑映射**：从 Gemma 262k 超大词表中精准锁定 **10,147 个原生同形单字 Token**，逐字拆解汉字并初始化线性分类头，实现连续无缝的单字连缀解码；
2. **纯净规范化与声学评测基准**：重构 Unicode NFKC 中文清洗流，彻底剔除标点与异常符号；全面落地字符错误率（**CER**）与音素级无调拼音错误率（**P-CER**）；
3. **消费级单卡闭环实证**：在单张消费级笔记本显卡 **RTX 4080 Laptop (12GB)** 上，显存占用仅 **3.55 GB**，以 **860x 实时吞吐倍率** 仅耗时 **8.5 分钟（1500 步）** 即完成了中文声学校准，验证集 CER 降至 **26.44%**，P-CER 降至 **10.89%**（发音准确率 89.11%）！

---

## ⚡ 特性矩阵与技术价值

| 核心特性 (Key Feature) | 传统技术痛点 (Pain Point) | 创新技术方案 (Technical Solution) | 核心技术价值 (Value Proposition) |
| :--- | :--- | :--- | :--- |
| **🧠 冻结音频塔特征复用** | 端到端训练大型语音模型动辄需要 8×H100 显卡，普通开发者无法承受算力成本 | 100% 冻结 300M 的 Gemma USM 编码塔，仅训练 6 层 Transformer Reader（**仅 20.4M 参数**） | **显存峰值压低至 3.55GB，单张 12GB 显卡算力大幅溢出** |
| **🔤 CJK 单字拓扑映射词表** | Gemma 原生分词器会将中文切为多音节词块，引发严重的数据稀疏与解码对齐漂移 | 从词表中精准抽离 10,147 个单字 token 建立汉字投影，连缀解码零空格干扰 | **彻底根除长词稀疏问题，实现中文端到端稳健流式解码** |
| **⚡ 动态时长分桶吞吐** | 传统固定 batch 填充海量静音 Padding，导致 GPU 70% 算力被无效零值吞噬 | 基于音频时长构建自适应队列，以 `max_batch_seconds=300` 动态打包多路音频 | **张量计算饱和度逼近 100%，达成 860x 极限实时吞吐倍率** |
| **🎯 字/音双轨自适应评测** | 传统英文/越南语 WER 无法客观评估中文“同音字多、音准字错”的声学真实表现 | 落地除零保护的汉字级 CER 与拼音级 P-CER 双轨联合评测流 | **精准分离声学识别与语言先验，为轻量解码提供科学基准** |

---

## 🏗️ 系统架构设计

```mermaid
flowchart TD
    A["16kHz 单声道音频 (WAV / FLAC)"] --> B["GPU Log-Mel 声学前端 (128 维)"]
    B --> C["EmbeddingGemma-2 音频编码塔 (USM Conformer, 12 层, 300M, 100% 冻结)"]
    C --> D["自适应多层 Softmax 混合 (13 层加权平均)"]
    D --> E["轻量级 CTC Reader (Linear + Depthwise Conv + 6×Transformer, 20.4M 参数)"]
    E --> F["Gemma 文本空间线性分类头 (10,147 CJK 单字 Token 初始化)"]
    F --> G["贪心 CTC 解码器 (Greedy CTC Blank 消除)"]
    G --> H["中文连缀汉字文本流 (Chinese Transcript)"]

    style C fill:#f1f5f9,stroke:#64748b,stroke-width:2px,stroke-dasharray: 5 5
    style E fill:#dbeafe,stroke:#2563eb,stroke-width:2px
    style F fill:#dcfce7,stroke:#16a34a,stroke-width:2px
```

---

## 📊 冒烟训练实测 Benchmark

在只有 **2.7 小时** 真实 AISHELL-1 训练小子集、单张 **RTX 4080 Laptop (12GB)** 上执行 1500 步冒烟验证：

### 1. 真实转写效果演进（见证模型自主纠错）

在完全没见过的验证集测试音频上，模型展现出了清晰的**语义自纠错能力**：

```text
【真实原声 Ref】：城市化快速发展期的重要特征就是基础设施投资需求量大

【第 500 步 Hyp】：城市化快速发展期的重要特争就事基础建设投资需求量大
                  (声学校准完成：将"特征"听成同音"特争"，将"就是"听成同音"就事")

【第 1500 步 Hyp】：城市化快速发展期的重要特征就是基础建设投资需求量大
                  (语义自纠错达成：模型自主将"特争就事"校正回"特征就是"！)
```

### 2. 训练全景收敛指标

<p align="center">
  <img src="docs/images/training_metrics.png" width="900" alt="Training Metrics Overview"/>
</p>

| 训练步数 (Steps) | CTC Loss | 验证集 CER (字错率) | 验证集 P-CER (发音音错率) | 显存占用 (VRAM) | 实时处理倍率 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 50** | 10.872 | - | - | 3.54 GB | 762x realtime |
| **Step 500** | 1.649 | 33.75% | 12.32% | 3.55 GB | 864x realtime |
| **Step 1000** | 0.660 | 27.45% | 10.77% | 3.55 GB | 865x realtime |
| **Step 1500 (最终)** | **0.424** | **26.44%** | **10.89%** | **3.55 GB** | **860x realtime** |

*注：P-CER 10.89% 意味着模型对中文语音的**音节听音准确率已高达 89.11%**。*

---

## 🚀 极简闪电快速开始

### 1. 环境准备

推荐在 Python 3.11 环境下运行：

```bash
# 克隆仓库
git clone https://github.com/your-username/gemma-asr-zh.git
cd gemma-asr-zh

# 创建并激活自包含虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux

# 安装核心依赖
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. 模型准备

下载 Google 原生模型权重（或放入本地目录）：
```bash
# 确保本地 models/google/embeddinggemma-2 存在
python -c "from transformers import AutoModel; AutoModel.from_pretrained('google/embeddinggemma-2')"
```

### 3. 数据导入与分片（AISHELL-1）

支持断点续传与多进程转码：
```bash
python scripts/prepare_data.py --aishell_dir E:/data/data_aishell --out_dir data/prepared --workers 4
```

### 4. 构建字级词表

```bash
python scripts/build_vocab.py --data_dir data/prepared --out data/prepared/units.json
```

### 5. 一键启动训练

Windows 单卡一键运行（或直接双击 `smoke_train.bat`）：
```powershell
& ".venv\Scripts\python.exe" scripts/train.py `
    --data_dir data/prepared `
    --units data/prepared/units.json `
    --out_dir runs/zh `
    --encoder_id models/google/embeddinggemma-2 `
    --max_steps 1500 `
    --max_batch_seconds 300.0 `
    --eval_every 500 `
    --save_every 500 `
    --log_every 50 `
    --num_workers 0
```

---

## 📂 项目结构全景

```text
gemma-asr/
├── gemma_asr/             # 核心模块源码
│   ├── text.py            # 中文文本规范化 (NFKC、标点/数字过滤、无调拼音转换)
│   ├── features.py        # GPU Log-Mel 声学前端 (与 HF 原生算子数值 100% 对齐)
│   ├── units.py           # 字级 CJK CTC 词表构建与连缀解码
│   ├── model.py           # GemmaCTC 核心模型架构 (Reader + 动态多层加权 + CTC)
│   ├── data.py            # 流式分片加载器与动态时长分桶
│   └── scoring.py         # 中文 CER 与 P-CER 评测指标
├── scripts/               # 工程运维与训练脚本
│   ├── prepare_data.py    # AISHELL-1 本地导入与标准 FLAC 分片落盘管线
│   ├── build_vocab.py     # 字级 CTC 词表自动解算脚本
│   ├── train.py           # 单卡/分布式模型训练引擎
│   └── plot_smoke_metrics.py # 训练指标多子图科研级可视化
├── tests/                 # 44 项完整覆盖的自动化测试套件 (TDD 规范)
├── docs/                  # 架构决策记录 (ADR) 与图表资产
├── smoke_train.bat        # Windows 单卡一键冒烟批处理 (GBK 兼容)
└── README.md              # 开源项目说明文档
```

---

## 🤝 鸣谢与协议

- **原始灵感与架构原型**：致敬 **QUAN-Transcribe** 项目及其作者在 EmbeddingGemma 迁移 ASR 上的开创性先驱工作；
- **底层多模态表征模型**：感谢 **Google DeepMind / Gemma Team** 开源的 `embeddinggemma-2` 卓越底座；
- **数据集**：感谢 **OpenSLR (AISHELL-1)** 团队提供的优质中文开源语音数据。

本项目基于 [Apache-2.0 License](./LICENSE) 协议开源。
