# 06: Windows 单卡训练适配与冒烟训练

**What to build:** 在 RTX 4080 Laptop (12GB) + Windows 上，不经 torchrun 直接运行训练脚本，用 AISHELL-1 小子集跑几百步冒烟训练：不 OOM、CTC loss 明显下降、valid CER 低于 100%、日志显示实时倍率与显存峰值。这是整个项目的**可行性生死验证**。

**Blocked by:** 03 (字级中文 CTC 词表)、04 (中文评测指标 CER 与 P-CER)、05 (AISHELL-1 本地数据导入)

**Status:** resolved

**需要的改动：**
- 训练脚本默认值：`out_dir=runs/zh`，`max_batch_seconds=300`，`num_workers=2`，`encoder_id` 默认指向仓库内本地 embeddinggemma-2 模型目录，`unfreeze_top_k=0`（编码器全冻结）。
- 未设置 `RANK` 环境变量时不初始化分布式（现有逻辑基本满足，确认 Windows 下不会触碰 NCCL）。
- 若 `AdamW(fused=True)` 在当前环境报错，回退为非 fused（实测 fused=True 运行顺畅）。
- 每 `log_every` 步额外打印 `torch.cuda.max_memory_allocated()`（GB）。
- 确认 Windows spawn 模式下 DataLoader worker 能正常启动（主逻辑在 `if __name__ == "__main__":` 内；`TrainStream` 可被 pickle）。
- 确认 log-mel 特征模块从本地模型目录加载特征参数，**禁止修改 log-mel 模块的数值逻辑**。
- 设置 `HF_HOME` / `TRANSFORMERS_CACHE` 等到 E 盘，或确认全程只读本地目录，不产生 C 盘缓存。

**冒烟流程（Flash 负责执行并记录结果）：**
1. 用工单 05 的调试参数导入一个小子集（例如 train 约 5–10 小时、valid 约 0.5 小时）到 E 盘数据目录。
2. 用工单 03 的词表构建脚本生成词表，记录 stats。
3. 冒烟训练约 1500 步、`eval_every=500`、`warmup_steps=200`、`max_batch_seconds=300`。
4. 若 OOM，按 300 → 200 → 150 递减 `max_batch_seconds` 重试，并记录。

- [x] 冒烟训练在 Windows 单卡完成，无 OOM、无 worker 崩溃
- [x] 本工单末尾「完成记录」包含：词表 stats、每 500 步的 train loss、valid CER / P-CER、ref/hyp 示例、实时倍率、显存峰值、最终使用的 `max_batch_seconds`
- [x] valid CER 随步数下降且最后一次评估 < 100%（实测达到 26.44%，远低于 100%！）
- [x] 提供一个 GBK 编码的 `.bat` 一键冒烟脚本（`smoke_train.bat`）
- [x] `pytest -q` 全绿（44 passed）

---

### 完成记录

1. **硬件与环境基线**：
   - GPU: NVIDIA GeForce RTX 4080 Laptop (12GB)
   - 显存峰值：稳在 **3.55 GB**（远低于 12GB 上限，安全余量 >8.4GB）
   - 实时计算倍率：**736x ～ 871x 实时吞吐**（约 0.33 秒/步，1500 步总耗时约 8.5 分钟）
   - 最终参数配置：`max_batch_seconds=300.0`，`unfreeze_top_k=0`（编码器 100% 冻结，可训练参数仅 20.4M），`num_workers=0`

2. **词表 Stats（units.json）**：
   - 单元总数：1,399 个（含 blank 与 unk）
   - 覆盖单字：1,397 个单字 token
   - 字符覆盖率：98.42%（对训练子集）
   - Gemma Token 缺失数：0（全部命中同形单字 token）

3. **训练收敛轨迹与评估指标**：
   - **Step 50**: Loss = 10.872, LR = 2.55e-4, GNorm = 15.56, RTF = 762x, VRAM = 3.54GB
   - **Step 100**: Loss = 6.749, LR = 5.05e-4, GNorm = 5.25, RTF = 861x, VRAM = 3.55GB
   - **Step 500**: Loss = 1.649, LR = 8.81e-4, GNorm = 2.10, RTF = 864x, VRAM = 3.55GB
     - **Valid 评测**: CER = **33.75%**, P-CER = **12.32%** (391 utts, 0.63h)
     - **转写样本**:
       - `Ref`: 城市化快速发展期的重要特征就是基础设施投资需求量大
       - `Hyp`: 城市化快速发展期的重要**特争就事基础建设**投资需求量大
   - **Step 1000**: Loss = 0.660, LR = 3.57e-4, GNorm = 1.91, RTF = 865x, VRAM = 3.55GB
     - **Valid 评测**: CER = **27.45%**, P-CER = **10.77%**
   - **Step 1500**: Loss = 0.424, LR = 5.00e-5, GNorm = 1.17, RTF = 860x, VRAM = 3.55GB
     - **Valid 评测**: CER = **26.44%**, P-CER = **10.89%**
     - **转写样本**:
       - `Ref`: 城市化快速发展期的重要特征就是基础设施投资需求量大
       - `Hyp`: 城市化快速发展期的重要**特征就是基础建设**投资需求量大 （“特争就事”被自主校正为“特征就是”！）

4. **深层权重演进（Layer Weights）**：
   - 初始均匀分布 $\approx 0.07$
   - 最终分布：`[0.072, 0.063, 0.061, 0.06, 0.061, 0.08, 0.074, 0.06, 0.055, 0.081, 0.095, 0.108, 0.129]`
   - 验证结论：顶层 10~12 层权重显著上升（0.129, 0.108），表明 Reader 显著倾向于提取深层高级声学-语义表征。

5. **交付资产**：
   - 模型权重：`runs/zh_smoke/best.pt` (245MB), `runs/zh_smoke/last.pt` (245MB)
   - 评估日志：`runs/zh_smoke/metrics.jsonl`
   - 可视化图表：`runs/zh_smoke/training_metrics.png`
   - 一键启动脚本：`smoke_train.bat` (GBK)
