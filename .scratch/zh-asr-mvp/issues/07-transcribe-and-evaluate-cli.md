# 07: 中文转写与独立评测入口

**What to build:** 用冒烟（或正式）checkpoint，尖子可以转写任意长度的中文音频并得到无空格简体汉字；也可以在 AISHELL-1 test 上一条命令输出 CER / P-CER 表格并导出 ref/hyp 对照。

**Blocked by:** 06 (Windows 单卡训练适配与冒烟训练)

**Status:** ready-for-agent

**要求：**
- 转写脚本：加载 checkpoint 中记录的词表与本地模型目录；长音频按现有静音点切分逻辑（≤ 30 秒）批量解码；输出拼接后的中文字串；输出文件（如有）使用 UTF-8。
- 评测脚本：单卡无 torchrun 可运行；表格列为 `source / CER / P-CER / utts / hours`；`--dump` 导出 jsonl（UTF-8，`ensure_ascii=False`）。
- 两个脚本都不得触发向 C 盘 HF 缓存下载。
- 各提供一个 GBK 编码的 `.bat` 包装（可选参数透传）。

- [ ] 用工单 06 的 checkpoint 转写仓库内 `example_audio.wav` 与 `example_slice.wav`，输出为中文（质量不作要求），把输出前 100 字写进完成记录
- [ ] 在冒烟子集的 valid/test 上运行评测脚本，表格与 `--dump` 文件正常生成
- [ ] `pytest -q` 全绿
