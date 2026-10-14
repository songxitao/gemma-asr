# 04: 中文评测指标 CER 与 P-CER

**What to build:** 评测时输出逐字 CER 与不带声调的拼音 P-CER（区分「听错音」和「同音字选错」），训练日志与 best checkpoint 选择全部改用中文指标，越南语 WER / 去声调 WER 移除。

**Blocked by:** 02 (中文文本规范化)

**Status:** resolved

**接口约定：**
- `ErrorCounts` 字段：`char_errors`、`ref_chars`、`pinyin_errors`、`ref_pinyin`、`utts`；保留 `__iadd__` 与 `all_reduce`（多卡兼容）。
- `ErrorCounts.summary()` 返回 `{"cer", "pcer", "utts", "ref_chars"}`。
- `score_batch(refs, hyps)`：refs 已规范化；hyps 先过 `normalize_hypothesis`；空 ref 跳过。CER 用 jiwer 字符级编辑距离；P-CER 把 `to_pinyin` 的结果当作「词」序列，用 jiwer 词级编辑距离（hyp 为空时也要正确计为全删除）。
- 训练脚本的评测打印改为 `CER x.xx%  P-CER x.xx%`；评测汇总里的平均指标改为 `avg_cer`；`best.pt` 按 `avg_cer` 最小保存。独立评测脚本的表格同样改为 CER / P-CER。

**规格：** `../spec.md` User Stories 12–15、Implementation Decisions「评测模块」。

- [x] `tests/` 新增评测测试：完全一致 → CER=0、P-CER=0；同音字替换（ref「他在」hyp「她在」）→ CER=0.5、P-CER=0；删字（ref「你好」hyp「你」）→ CER=0.5；空 hyp → CER=1.0；两个 `ErrorCounts` 相加后 summary 正确
- [x] 全仓库不再引用 `wer_notone`、`strip_tones`、`avg_wer`
- [x] `pytest -q` 全绿

## 完成记录
- **双子智能体流水线协同**：
  - 测试子智能体编写 `tests/test_scoring.py`（19 项独立测试用例），初测 19 failed 确认红灯；
  - 开发子智能体接力重构 `gemma_asr/scoring.py`，实现除零保护的 `summary()`、字符级 CER、以及拼音词序列编辑距离 P-CER；
  - **技术异议机制生效**：开发子智能体发现测试用例中将“北京科技大学”替换为“背景克己大雪”时，第 4 个字“大”两边同字，敏锐指出编辑距离应为 5 而非 6，主智能体采纳其建议将 hyp 修正为真正的全同音异形字“背景克己达雪”，实现了高质量的契约修正。
- **全库指标同步**：
  - `scripts/train.py` 与 `scripts/evaluate.py` 已彻底清空越南语 `wer_notone` / `avg_wer` 打印，全面切换为 `CER`、`P-CER` 及 `avg_cer` 选模逻辑；
  - `gemma_asr/text.py` 中遗留的过渡桩 `strip_tones` 与 `canonicalize_tone` 已彻底清除。
- **测试结果**：
  - `tests/test_scoring.py`: 19 passed in 1.74s
  - 全局回归测试: `38 passed in 14.46s`
