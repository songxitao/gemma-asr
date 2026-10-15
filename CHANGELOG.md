# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- **工单 10**：接入可选轻量标点恢复引擎（`--punc`），基于 CT-Transformer 延迟导入。
- **工单 12**：长音频 VAD 切分、CTC 帧级时间戳提取与 SRT/VTT 字幕输出支持。
- **工单 13**：多场景数据扩充，引入 WenetSpeech 强噪声/多方言语料提升通用泛化鲁棒性。
- **工单 14**：可选轻量 N-gram 语言模型解码（抹平 3.3% 同音字异形误差）。

---

## [0.2.0] - 2026-10-15

### Added
- **全量 28,000 步训练权重**: 在 AISHELL-1 全量 150.9h 训练集上完成 28,000 步稳定收敛，锁定最佳 checkpoint `runs/zh_full/best.pt`（净权重约 80MB）。
- **科研级收敛图表**: 编写 `scripts/plot_full_metrics.py`，生成高分辨率双子图 `assets/training_curve.png`，直观呈现 Loss、学习率与 CER/P-CER 周期拐点。
- **公平基准打擂框架**: 新增 `scripts/benchmark.py` 与跨环境执行器 `scripts/run_funclip_worker.py`，支持逐句单推（`bs1`）与大批吞吐（`batch`）两种标准评测模式。
- **自动化契约测试**: 新增 `tests/test_benchmark.py`，覆盖 7 项契约测试（指标计算、编辑距离、表格格式化、外部引擎优雅 fallback 等），全库测试扩充至 66 项。

### Changed
- **README 重构与透明化**: 完整呈现 28,000 步训练动态、盲测成果、三方天梯榜实测数据、详细测试硬件环境与部署显存规格。
- **路径去敏与开源泛化**: 将基准评测和 worker 调度脚本中的绝对路径全面改造为环境变量读取（`FUNCLIP_PYTHON`, `SENSEVOICE_PATH`, `PARAFORMER_PATH`）与 CLI 参数透传，杜绝硬编码泄露。

### Performance
- **750x ~ 1,190x 实时吞吐**: 非自回归全句并行解码，单卡 RTX 4080 Laptop 下全量 10.0 小时盲测仅耗时 30.25 秒（RTF = 0.00084）。
- **三方同台速度突破**: 在 AISHELL-1 前 1,000 句实测中，Gemma-ASR 推理耗时 6.04 秒（751x），比同台 SenseVoiceSmall 快 10.6 倍，比 SeACo-Paraformer 快 20.2 倍。
- **声学拼音高保真**: 盲测集声学拼音错误率（P-CER）仅 1.80% ~ 2.29%，声学特征解码正确率达 98.20%。

---

## [0.1.0] - 2026-10-14

### Added
- **中文 ASR 基础工程落地**: 完成 Google EmbeddingGemma 2 Conformer 音频塔（300M）物理绝对冻结，外挂 6 层轻量 CTC Transformer Reader（21.7M）。
- **数据管线与分片引擎**: 完整适配 AISHELL-1 数据清洗、Mel-filterbank 声学特征提取与 Parquet 流式分片存储。
- **离线转写与独立评测 CLI**: 交付 `scripts/transcribe.py` 与 `scripts/evaluate.py`，无空格连续汉字输出，配套 Windows GBK 批处理脚本。
- **冒烟测试全链路验证**: 4,000 步单卡冒烟验证通过，全自动化测试套件初步成型。
