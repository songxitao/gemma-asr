# 08: AISHELL-1 全量首版训练与测试集报告

**What to build:** 在全量 AISHELL-1（约 178h）上完成首版训练，得到 `best.pt`，并在 AISHELL-1 test 上给出正式 CER / P-CER，对照 MVP 验收线（CER ≤ 15%，P-CER ≤ 5%）。

**Blocked by:** 06 (Windows 单卡训练适配与冒烟训练)

**Status:** ready-for-agent

**分工说明：** 全量训练耗时数小时级，**由尖子在本机启动长任务**；Flash 负责准备命令、`.bat` 脚本、监控方式与结果整理，不要在一次会话里阻塞等待训练结束。

**要求：**
- 全量导入命令与词表构建命令（基于工单 05、03）。
- 训练参数基于工单 06 冒烟中验证过的 `max_batch_seconds`；`max_steps` 按「约 15–20 个 epoch」估算并写明估算过程（178h × epoch 数 ÷ 每步音频秒数）；`eval_every` / `save_every` 取能在 1 小时内至少评估一次的值；支持 `--resume` 续训。
- GBK 编码的一键全量训练 `.bat`，日志重定向到 E 盘文件。
- 训练结束后用工单 07 的评测入口跑 AISHELL-1 test。

- [ ] 「完成记录」中写明：实际命令、总步数、训练时长、实时倍率、显存峰值、dev 曲线（每次评估的 CER / P-CER）、test 最终 CER / P-CER、10 条 ref/hyp 样例
- [ ] 结论一句话：是否达到 MVP 验收线；若未达到，列出 P-CER 与 CER 的差距（同音字问题占比），供尖子决定是否进入 Phase 3（n-gram LM）或扩数据
