# Spec: 中文 ASR MVP（EmbeddingGemma 2 冻结音频塔 + 字级 CTC Reader）

**Status:** ready-for-agent
**Branch:** `zh`（禁止在 `main` 上改动，`main` 保留为越南语原版对照）
**Supersedes:** `charter.md` 中「SenseVoice-CTC (24MB 极速头)」架构锁、`HANDOFF.md` 中「Gemma 字词混合词表」「冻结 10 层 + 解冻 2 层」方案（理由见 Further Notes）。

---

## Problem Statement

尖子想把越南语项目 QUAN-Transcribe（冻结 EmbeddingGemma 2 音频编码器 + 小型 CTC reader，8×H100 训练 2 小时达到越南语 SOTA）复刻成**中文** ASR，并发布到 GitHub。

当前仓库仍是越南语流水线：文本规范化只接受越南语/拉丁字母（中文句子会被全部丢弃）、词表按 Gemma 原生分词（中文会被切成「人工智能」这类多字词，数据稀疏）、评测指标是按空格切分的 WER 与「去声调 WER」（对中文无意义）、数据源全是越南语 HF 数据集、训练默认参数面向 8×H100 Linux（NCCL、10 个 worker、每卡 1200 秒 batch）。

本地只有一张 **RTX 4080 Laptop (12GB)**，C 盘仅剩约 20GB，必须在单卡、Windows、E 盘存储的前提下跑通。

## Solution

在不改动声学前端与模型主体的前提下，把**文本侧**与**数据侧**整体中文化，使尖子能够：

1. 用 AISHELL-1（178h，OpenSLR 公开下载）一条命令生成训练分片；
2. 构建一个**字级**中文 CTC 词表（每个汉字映射到 Gemma 词表里已存在的单字 token，分类头仍用 Gemma 文本 embedding 初始化）；
3. 在 4080 Laptop 上以**编码器全冻结**模式训练，日志输出中文 **CER** 与**拼音 CER（P-CER）**；
4. 用训练好的 checkpoint 转写任意长度的中文音频，输出无空格的简体汉字。

MVP 验收目标（沿用 ADR-001 / charter 的数字）：AISHELL-1 test 纯 CTC 贪心解码 **CER ≤ 15%**、**P-CER ≤ 5%**；推理 ≥ 200× 实时。

## User Stories

1. 作为尖子，我希望中文转写文本经过统一规范化（NFKC 全半角统一、去标点、去空格），这样模型不必学习书写变体。
2. 作为尖子，我希望含阿拉伯数字的句子在 MVP 阶段被直接丢弃，这样不会因为「2026」读法不确定而污染标注。
3. 作为尖子，我希望含拉丁字母或其他非汉字字符的句子在 MVP 阶段被丢弃，这样词表保持纯汉字。
4. 作为尖子，我希望 AISHELL 原始标注中的词间空格被去掉，这样训练目标是连续汉字串。
5. 作为尖子，我希望规范化函数对模型输出（假设文本）永不拒绝、只做宽松清洗，这样评测时不会丢句。
6. 作为尖子，我希望 CTC 词表是**单字级**的，每个汉字恰好一个 unit，这样 178h 数据也能覆盖充分。
7. 作为尖子，我希望每个汉字 unit 对应 Gemma 词表中**已有的单字 token ID**，这样分类头可以用 Gemma 文本 embedding 初始化，保留原项目最关键的设计。
8. 作为尖子，我希望 Gemma 中不存在单字 token 的罕见字统一映射为 `<unk>`，并在构建词表时打印覆盖率，这样我能判断词表是否够用。
9. 作为尖子，我希望词表构建时能按最低频次过滤（默认出现 ≥ 2 次），这样噪声字不进入词表。
10. 作为尖子，我希望词表文件以 UTF-8 存储、记录的模型路径是**本地 E 盘模型目录**，这样 Windows 上不会出现 GBK 乱码、也不会触发向 C 盘 HF 缓存下载。
11. 作为尖子，我希望 unit 序列解码回文本时直接拼接汉字、不插入空格，这样输出就是可读的中文。
12. 作为尖子，我希望评测输出 **CER**（逐字编辑距离），这样能和 AISHELL 公开榜单直接对比。
13. 作为尖子，我希望评测额外输出 **P-CER**（转成不带声调拼音后逐音节比较），这样能区分「听错音」和「同音字选错」。
14. 作为尖子，我希望训练日志里不再出现越南语专用的「WER(no tone)」，而是 `CER / P-CER`，这样指标含义清晰。
15. 作为尖子，我希望最优 checkpoint 按**平均 valid CER** 选择，这样选模标准符合中文习惯。
16. 作为尖子，我希望有一个数据导入入口，能读取本地解压后的 AISHELL-1 目录（wav + 官方 transcript），输出与现有流水线完全相同格式的 parquet 分片（FLAC 16k 单声道、text、duration、source），这样训练、评测代码无需感知数据来源。
17. 作为尖子，我希望 AISHELL-1 的 train / dev / test 分别落到 train / valid / test 三个 split，这样训练中能自动评估 dev。
18. 作为尖子，我希望数据导入可断点续跑、支持只转换少量文件的调试模式，这样能先用几分钟验证管线。
19. 作为尖子，我希望数据导入结束打印每个 split 的保留小时数与各类丢弃原因计数，这样我知道过滤是否过严。
20. 作为尖子，我希望所有原始数据、分片、checkpoint、HF 缓存都位于 E 盘，这样 C 盘不会爆。
21. 作为尖子，我希望训练脚本在 Windows 单卡、未经 torchrun 启动时能直接运行，这样不需要 NCCL。
22. 作为尖子，我希望训练默认参数适配 12GB：每步 padded 音频 ≤ 300 秒、音频切片 ≤ 15 秒、编码器全冻结，这样不会 OOM。
23. 作为尖子，我希望 DataLoader 的 worker 数在 Windows 上默认取保守值，并且脚本在 Windows spawn 模式下不会重复执行主逻辑，这样训练能稳定启动。
24. 作为尖子，我希望能用一个「冒烟训练」配置（几百步、小数据）在 30 分钟内看到 CTC loss 明显下降、valid CER 低于 100%，这样能尽早验证可行性。
25. 作为尖子，我希望训练日志打印实时倍率与显存峰值，这样我能评估 4080 的真实吞吐。
26. 作为尖子，我希望能用 checkpoint 转写 `example_audio.wav` 这类长音频，长音频按静音点切成 ≤ 30 秒片段批量解码，这样能直接体验效果。
27. 作为尖子，我希望独立的评测入口能在 AISHELL-1 test 上输出 CER / P-CER 表格，并可导出 ref/hyp 对照，这样便于误差分析。
28. 作为尖子，我希望核心纯函数（规范化、词表编解码、CER/P-CER）有 pytest 单元测试，这样 Flash 模型改代码时有回归保护。
29. 作为尖子，我希望 README 改写为中文项目说明（保留对原越南语项目的致谢），这样发布到 GitHub 时定位清晰。
30. 作为尖子，我希望新增依赖（pyarrow、pytest、pypinyin）在安装前先征得我同意，这样环境变更可控。
31. 作为尖子，我希望声学前端（log-mel 特征）保持不动，这样已对齐到 1e-5 的数值不被破坏。
32. 作为尖子，我希望模型主体结构（layer mix、reader、文本空间分类头、CTC）保持与原版一致，这样中文结果可以和越南语结果横向比较。

## Implementation Decisions

- **文本规范化模块（`gemma_asr.text`）整体替换为中文实现**，越南语声调相关逻辑全部删除。对外保留两个函数名与语义：
  - `normalize_text(text) -> str | None`：NFKC → 去所有标点与空白 → 若含数字或非 CJK 统一汉字（U+4E00–U+9FFF）字符则返回 `None` → 返回连续汉字串。
  - `normalize_hypothesis(text) -> str`：同样清洗但永不返回 `None`（非汉字字符直接删除）。
  - 新增 `to_pinyin(text) -> list[str]`：不带声调的逐字拼音，供 P-CER 使用。
  - 繁简转换不在 MVP 范围（AISHELL 本身是简体）。
- **词表模块（`gemma_asr.units`）改为字级**：
  - `tokenize` 不再调用 Gemma 分词器对整句分词，而是把规范化后的字符串逐字拆开，每个字查 Gemma 词表中**同形单字 token** 的 ID（`tokenizer.convert_tokens_to_ids`；查不到或得到 unk ID 视为缺失）。已实测：Gemma 262,144 词表中有 10,147 个单字 CJK token，常用字全覆盖。
  - unit 编号约定不变：0 = blank，1 = `<unk>`，2.. = 保留的 Gemma 单字 token。
  - `decode` 直接把 unit 对应的汉字拼接，不插空格。
  - `save` / `load` 显式使用 UTF-8。
  - `build` 打印：不同字数、保留字数、无 Gemma 单字 token 的字数、字频覆盖率。
  - 模型 ID 一律传**本地模型目录**（E 盘的 embeddinggemma-2 目录），不使用 HF 仓库名，避免写 C 盘缓存。
- **评测模块（`gemma_asr.scoring`）**：`ErrorCounts` 字段改为 `char_errors / ref_chars / pinyin_errors / ref_pinyin / utts`；`summary()` 返回 `cer`、`pcer`、`utts`、`ref_chars`。CER 用 jiwer 的字符级编辑距离；P-CER 把拼音音节当作「词」用 jiwer 词级编辑距离。保留 `all_reduce` 以兼容多卡。
- **训练脚本**：评测日志改为打印 `CER / P-CER`；`best.pt` 按平均 valid CER 选择；默认值改为 `out_dir=runs/zh`、`max_batch_seconds=300`、`num_workers=2`、`encoder_id=<本地模型目录>`；非 torchrun 启动时不初始化分布式；记录 `torch.cuda.max_memory_allocated()`。`fused=True` 的 AdamW 若在当前 torch 版本报错则回退为非 fused。
- **数据导入**：新增 AISHELL-1 本地导入入口，读取 OpenSLR 解压目录（`data_aishell/wav/{train,dev,test}/<spk>/*.wav` + `data_aishell/transcript/aishell_transcript_v0.8.txt`），复用现有分片写出逻辑（FLAC 16k、`audio/text/duration/source` 四列、`.stats.json` 断点续跑、`manifest.json` 汇总），`source="aishell1"`，dev → `valid`。默认时长过滤 `0.5 ~ 15` 秒。不依赖 HF Hub。
- **词表构建脚本**沿用现有入口，只需确认它读取 `train` split 的 `text` 列并调用新的字级 `Units.build`。
- **转写脚本**：输出为中文字串；长音频切分逻辑不变。
- **不变部分**：log-mel 特征模块禁止修改；模型结构、`GemmaCTC` 前向、`greedy_decode` 不改；编码器默认全冻结（`unfreeze_top_k=0`）。

## Testing Decisions

- **好测试的标准**：只测公开函数的外部行为（输入文本 → 输出文本 / ID 序列 / 指标数值），不测内部实现细节。
- **测试接缝（尽量高、尽量少）**：
  1. `normalize_text` / `normalize_hypothesis` / `to_pinyin`：纯函数，覆盖标点、全角、空格、数字拒绝、英文拒绝、空串。
  2. `Units`：用真实本地 Gemma tokenizer 构建一个小词表，测 `encode → decode` 往返一致、罕见字映射为 1、unit 0/1 保留、`save → load` 往返（含中文路径与 UTF-8）。
  3. `score_batch` → `ErrorCounts.summary()`：用手算样例断言 CER 与 P-CER 数值（例如同音字替换 CER > 0 而 P-CER = 0）。
  4. 数据导入：用 2～3 个临时生成的 wav + 伪 transcript 组成迷你 AISHELL 目录，断言输出 parquet 行数、列名、duration、text 已规范化。
- **端到端验证（非 pytest，人工/脚本）**：冒烟训练（几百步）loss 下降 + valid CER < 100%；转写 `example_audio.wav` 输出中文。
- **测试框架**：pytest，测试放在仓库根目录 `tests/`。仓库目前无测试先例，这是第一批测试。重型模型加载（音频塔）不进入单元测试。

## Out of Scope

- ITN / TN（数字、英文读法转换）、标点恢复、繁简转换。
- 外部语言模型（KenLM / n-gram beam search）—— 属于 ADR-001 Phase 3。
- 解冻编码器层、激活检查点、梯度累加等显存优化（全冻结在 12GB 内已足够，见 Further Notes）。
- 流式/分块推理（原代码 reader 为双向注意力，不存在流式实现）。
- AISHELL-1 以外的数据集（WenetSpeech、KeSpeech、Common Voice zh 等）与云端大规模训练。
- 任何外部纠错模型（MacBERT、LLM 等）。
- 复用 SenseVoice 的 CTC 头权重。
- RAG / 向量检索功能。

## Further Notes

- **为什么不复用 SenseVoice-CTC 头**：CTC 头只是「某个编码器输出空间 → 词表」的线性映射。SenseVoice 头学的是 SenseVoice 编码器的特征分布，与 Gemma USM Conformer 的输出坐标系无关，接上等同随机初始化。原项目本身就只训练 reader + 分类头（约 22M 参数），成本极低。SenseVoice 可复用的只有其字表文件。
- **为什么是字级而不是 Gemma 原生分词**：实测 Gemma 把「人工智能」「语音」「识别」等切为单 token，CTC 类别会变成稀疏的多音节词；越南语之所以可行，是因为 Gemma 对越南语基本一音节一 token。字级 CTC 是中文 ASR 标准做法（AISHELL / WeNet 约 4k 字）。
- **显存依据**：原 README 实测单张 A6000、编码器全冻结、600 秒/batch 时显存 9.5GB；4080 Laptop 12GB 取 300 秒/batch 留足余量。
- **探针结论需降级**：Phase 0 相似度探针中「近音干扰」0.7097 高于「正样本」0.6976，只能说明句向量含声学信息，不能证明可做 ASR。真正的可行性判据是冒烟训练的 CER 曲线。
- **外部参照**：越南语原版仅用 39h 数据、单张 A6000 训练 20 分钟即得 FLEURS WER 21%。中文 AISHELL 竞品基线：Paraformer ≈ 1.9% CER、SenseVoice-small ≈ 3% CER。本项目卖点是轻量、极速与「同一编码器同时产出音频 embedding 与转写」，而非刷榜。
- **环境**：项目虚拟环境 Python 3.11 + torch 2.5.1+cu121 + transformers 5.19.0。当前**缺失** pyarrow、pytest、pypinyin，安装前必须征得尖子同意。所有 `.py` / `.md` / `.json` 读写显式 `encoding='utf-8'`；`.bat` 用 GBK。
