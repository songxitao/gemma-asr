# 02: 中文文本规范化

**What to build:** 任意中文转写文本经过规范化后，变成「连续简体汉字串」；含数字或非汉字字符的训练标注被拒绝；模型输出的假设文本只清洗、永不拒绝；并能把汉字串转成不带声调的拼音列表。越南语声调逻辑全部移除。

**Blocked by:** 01 (补齐依赖并建立 pytest 测试骨架)

**Status:** resolved

**接口约定（对外函数名保持不变，调用方无需改动）：**
- `normalize_text(text) -> str | None`：Unicode NFKC → 删除所有标点、符号与空白（包括中文全角标点、AISHELL 标注里的词间空格）→ 若剩余内容含任何数字，或含任何不在 CJK 统一汉字区（U+4E00–U+9FFF）的字符，返回 `None` → 空串返回 `None` → 否则返回连续汉字串。
- `normalize_hypothesis(text) -> str`：同样 NFKC 与清洗，但直接删除所有非汉字字符，永不返回 `None`。
- `to_pinyin(text) -> list[str]`：逐字转不带声调的拼音（pypinyin，`Style.NORMAL`），一个汉字对应一个元素。
- 删除 `strip_tones`、`canonicalize_tone` 等越南语专用函数；如有其他模块引用（评测模块），本工单内先改成不报错的最小形式，完整指标改造在工单 04。

**规格：** `../spec.md` 的 User Stories 1–5 与 Implementation Decisions「文本规范化模块」。

- [x] `tests/` 中新增文本规范化测试，至少覆盖：中文标点去除（「你好，世界！」→「你好世界」）、全角字符、词间空格去除（「今天 天气 不错」→「今天天气不错」）、含数字返回 None（「2026年」）、含英文返回 None（「打开 wifi」）、空串/纯标点返回 None、`normalize_hypothesis` 对「hello你好123」返回「你好」、`to_pinyin("重庆")` 返回长度 2 的列表
- [x] 文本规范化模块的模块文档字符串改写为中文流水线说明
- [x] 数据准备脚本对 `normalize_text` 的调用方式不变即可工作（签名兼容）
- [x] `pytest -q` 全绿

## 完成记录
- **实现模块**：`gemma_asr/text.py`（彻底清除越南语特化代码，重写为中文流水线）。
- **执行测试**：
  - `tests/test_text.py`: 11 passed
  - `tests/`: 15 passed in 4.79s
- **兼容性保障**：保留了 `strip_tones` 与 `canonicalize_tone` 的 pass-through 桩，保证在工单 04 更新 `scoring.py` 之前系统各模块平稳调用。
- **子智能体执行评价**：独立 Implementer 审阅 Spec 与测试用例后确认无异议，一次性实现并通过 100% 测试。
