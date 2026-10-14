# 09: 中文 README 与项目约束文档同步

**What to build:** 仓库对外呈现为「基于 EmbeddingGemma 2 的中文极速 ASR」：README 改为中文项目说明并附真实结果；内部约束文档与本规格一致，不再出现已废弃的 SenseVoice-CTC 头与字词混合词表方案。

**Blocked by:** 08 (AISHELL-1 全量首版训练与测试集报告)

**Status:** ready-for-agent

**要求：**
- README：中文；项目定位（轻量、极速、同一编码器兼得音频 embedding 与转写）；架构图（文字版即可）；环境安装（Windows + 项目虚拟环境）；AISHELL-1 数据准备 → 词表 → 训练 → 评测 → 转写的完整命令；工单 06/08 的真实结果表（CER / P-CER / 实时倍率 / 显存）；局限（无 ITN、无标点、无 LM、非流式）；致谢原越南语项目 QUAN-Transcribe 与 Google EmbeddingGemma 2；不得写入未经实测的数字。
- `charter.md`、`findings.md`、`docs/ADR-001-CHINESE-ASR-MVP.md`、`HANDOFF.md`：**修改前先向尖子展示拟改内容并获得同意**（charter 默认只读）。拟改要点：移除「SenseVoice-CTC 头」架构锁，改为「字级 Gemma 单字 token 分类头」；移除「冻结 10 层 + 解冻 2 层」；把 Phase 0 探针结论降级为「句向量含声学信息，可行性以冒烟训练为准」。

- [ ] README 中文版完成，所有命令在 Windows 下可直接复制执行
- [ ] 约束文档修改经尖子确认后落地
- [ ] 全仓库搜索不再把 SenseVoice-CTC 头描述为当前架构
