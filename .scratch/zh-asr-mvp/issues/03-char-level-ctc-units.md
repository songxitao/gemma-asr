# 03: 字级中文 CTC 词表（映射到 Gemma 单字 token）

**What to build:** 从训练文本构建一个单字级 CTC 词表：每个汉字一个 unit，unit 绑定到 Gemma 词表中同形的单字 token ID，使模型分类头仍能用 Gemma 文本 embedding 初始化；文本 ↔ unit 序列可无损往返（罕见字除外）；词表文件 UTF-8 存储且指向本地模型目录。

**Blocked by:** 02 (中文文本规范化)

**Status:** resolved

**关键事实（已实测）：** Gemma tokenizer 词表 262,144；单字 CJK token 10,147 个；常用字全部存在单字 token。但 Gemma 原生分词会把「人工智能」「语音」切成多字 token，**所以绝不能对整句调用 tokenizer 分词**，必须逐字查表。

**接口约定（类名与方法名保持不变）：**
- `Units.tokenize(texts)`：每条文本逐字拆分，每个字用 tokenizer 的「token 字符串 → ID」查询；查不到或得到 tokenizer 的 unk ID 视为缺失。注意 Gemma 单字 token 可能带前导 `▁` 变体，**只取不带 `▁` 的同形 token**。
- `Units.encode(texts)`：缺失或不在保留集合的字 → unit 1（`<unk>`）。unit 0 = blank，1 = `<unk>`，2.. = 保留字（按频次降序）。
- `Units.decode(unit_ids)`：直接拼接对应汉字，不插空格，忽略 0/1。
- `Units.save/load`：显式 `encoding="utf-8"`，`ensure_ascii=False`。
- `Units.build(model_id, texts, min_count=2)`：返回 `(units, stats)`；stats 至少包含：不同字数、保留字数、无 Gemma 单字 token 的字数（及前 20 个样例）、字频覆盖率。
- `model_id` 传**本地 embeddinggemma-2 模型目录**（仓库 `models/google/embeddinggemma-2`），不得使用 HF 仓库名，避免向 C 盘缓存下载。词表构建脚本的默认值相应修改。

**规格：** `../spec.md` User Stories 6–11、Implementation Decisions「词表模块」。

- [x] 词表模块的模块文档字符串改写为中文字级说明
- [x] `tests/` 新增词表测试（使用真实本地 tokenizer，只加载 tokenizer，不加载模型权重）：小语料构建词表后 `decode(encode(x)) == x`；未出现的字编码为 1；unit 0/1 不出现在保留集合；`save → load` 往返一致（写到 pytest 临时目录）；`len(units) == 保留字数 + 2`
- [x] 词表构建脚本可在只有 `train` split 分片的数据目录上运行并打印 stats（可在工单 05 完成后再实测，本工单用测试覆盖逻辑即可）
- [x] 模型侧 `load_gemma_parts` 依据 `units.gemma_ids` 取 embedding 的逻辑无需修改即可工作（确认 ID 都是合法单字 token）
- [x] `pytest -q` 全绿

## 完成记录
- **实现模块**：`gemma_asr/units.py`（彻底移除整句前导空格分词，改为单字拆解与 Gemma 同形单字 Token ID 映射；解码直接无缝拼接无空格）。
- **脚本适配**：`scripts/build_vocab.py` 默认 `model_id` 已重定向为本地 `models/google/embeddinggemma-2`，杜绝访问 HF 远程或污染 C 盘。
- **测试结果**：
  - `tests/test_units.py`: 4 项全部通过
  - 全局回归测试: `19 passed in 14.02s`
- **协同审查**：独立 Implementer 子智能体在最小暴露上下文下高效交付，确认 Gemma 词表 10,147 个单字双向单映射完全满足要求。
