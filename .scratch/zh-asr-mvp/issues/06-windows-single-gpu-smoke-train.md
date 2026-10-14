# 06: Windows 单卡训练适配与冒烟训练

**What to build:** 在 RTX 4080 Laptop (12GB) + Windows 上，不经 torchrun 直接运行训练脚本，用 AISHELL-1 小子集跑几百步冒烟训练：不 OOM、CTC loss 明显下降、valid CER 低于 100%、日志显示实时倍率与显存峰值。这是整个项目的**可行性生死验证**。

**Blocked by:** 03 (字级中文 CTC 词表)、04 (中文评测指标 CER 与 P-CER)、05 (AISHELL-1 本地数据导入)

**Status:** ready-for-agent

**需要的改动：**
- 训练脚本默认值：`out_dir=runs/zh`，`max_batch_seconds=300`，`num_workers=2`，`encoder_id` 默认指向仓库内本地 embeddinggemma-2 模型目录，`unfreeze_top_k=0`（编码器全冻结）。
- 未设置 `RANK` 环境变量时不初始化分布式（现有逻辑基本满足，确认 Windows 下不会触碰 NCCL）。
- 若 `AdamW(fused=True)` 在当前环境报错，回退为非 fused。
- 每 `log_every` 步额外打印 `torch.cuda.max_memory_allocated()`（GB）。
- 确认 Windows spawn 模式下 DataLoader worker 能正常启动（主逻辑在 `if __name__ == "__main__":` 内；`TrainStream` 可被 pickle）。
- 确认 log-mel 特征模块从本地模型目录加载特征参数，**禁止修改 log-mel 模块的数值逻辑**。
- 设置 `HF_HOME` / `TRANSFORMERS_CACHE` 等到 E 盘，或确认全程只读本地目录，不产生 C 盘缓存。

**冒烟流程（Flash 负责执行并记录结果）：**
1. 用工单 05 的调试参数导入一个小子集（例如 train 约 5–10 小时、valid 约 0.5 小时）到 E 盘数据目录。
2. 用工单 03 的词表构建脚本生成词表，记录 stats。
3. 冒烟训练约 1500 步、`eval_every=500`、`warmup_steps=200`、`max_batch_seconds=300`。
4. 若 OOM，按 300 → 200 → 150 递减 `max_batch_seconds` 重试，并记录。

- [ ] 冒烟训练在 Windows 单卡完成，无 OOM、无 worker 崩溃
- [ ] 本工单末尾「完成记录」包含：词表 stats、每 500 步的 train loss、valid CER / P-CER、ref/hyp 示例、实时倍率、显存峰值、最终使用的 `max_batch_seconds`
- [ ] valid CER 随步数下降且最后一次评估 < 100%（若不满足，停下并在完成记录里写明现象，交给尖子判断，不要擅自改模型结构）
- [ ] 提供一个 GBK 编码的 `.bat` 一键冒烟脚本（调用项目虚拟环境 Python）
- [ ] `pytest -q` 全绿
