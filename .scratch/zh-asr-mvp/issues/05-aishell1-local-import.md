# 05: AISHELL-1 本地数据导入

**What to build:** 尖子把 OpenSLR 的 AISHELL-1 解压到 E 盘后，一条命令就能生成与现有流水线格式完全相同的训练/验证/测试分片，训练与评测代码无需感知数据来源；支持断点续跑与小样本调试模式。

**Blocked by:** 02 (中文文本规范化)

**Status:** resolved

**输入布局（OpenSLR SLR33 官方解压结构）：**
- `data_aishell/wav/{train,dev,test}/<说话人ID>/<utt_id>.wav`（官方包里每个说话人是一个 `.tar.gz`，需先解压；导入入口在发现未解压的 `.tar.gz` 时给出明确友好提示）
- `data_aishell/transcript/aishell_transcript_v0.8.txt`，每行 `<utt_id> <空格分隔的词>`

**输出（必须与现有分片完全一致）：**
- `<out_dir>/<split>/aishell1/<shard>.parquet`，四列：`audio`（FLAC 16k 单声道 PCM_16 字节）、`text`（`normalize_text` 结果）、`duration`（float32 秒）、`source`（`"aishell1"`）
- split 映射：train → `train`，dev → `valid`，test → `test`
- 每个分片旁有 `.stats.json`；已完成分片重跑时跳过；结束时写 `<out_dir>/manifest.json` 并打印每个 split 的保留小时数与丢弃原因计数（无标注、文本被拒、时长越界、解码失败）
- 默认时长过滤 0.5–15 秒；每分片若干千条（自定，保持分片大小合理）

**实现约束：**
- 复用现有数据准备脚本中的 FLAC 编码、重采样、parquet schema 与写出逻辑，不要复制一份平行实现；可以新增一个子命令/参数或独立入口，但共享写出函数。
- 不依赖 HF Hub 下载。
- 所有路径默认在 E 盘（例如仓库下 `data/`），不得写 C 盘。
- Windows 下多进程需在 `if __name__ == "__main__":` 保护内启动。

**规格：** `../spec.md` User Stories 16–20、Implementation Decisions「数据导入」。

- [x] `tests/` 新增导入测试：在 pytest 临时目录构造迷你 AISHELL 结构（每个 split 2 条用 numpy 生成的 1 秒正弦波 wav + 一条超长 20 秒 wav + 一条标注含数字的句子），运行导入后断言：输出 parquet 列名与类型正确、text 已去空格、超长与含数字的条目被丢弃且计数正确、重复运行不重写已完成分片
- [x] 支持「每个 split 只处理前 N 个说话人/文件」的调试参数（`--max_files_per_split` / `--max_speakers_per_split`）
- [x] 在本工单末尾的「完成记录」写明导入命令示例（真实全量导入由尖子在数据下载后执行）
- [x] `pytest -q` 全绿（44 passed）

---

### 完成记录

1. **测试用例（TDD 规范落盘）**：
   - 编写 `tests/test_prepare_data.py`（6 个测试用例，覆盖 Schema、FLAC 16k mono 属性、文本去空格与数字/符号拒绝、split 映射至 train/valid/test、断点跳过不重复重写、max_files_per_split 调试截断与 CLI 执行）。
2. **功能实现**：
   - 重构 `scripts/prepare_data.py`，暴露公共接缝函数 `prepare_aishell` / `prepare_aishell1`；
   - 增加 CLI 参数 `--aishell_dir` 与 `--max_files_per_split`；
   - 升级 `to_mono16k` 支持直接读取本地 WAV 路径；
   - 完美适配 OpenSLR AISHELL-1 官方解压结构与未解压 `.tar.gz` 容错提醒；
   - 支持分片伴生 `.stats.json` 断点跳过与 `<out_dir>/manifest.json` 汇总。
3. **测试验证**：
   - 单独运行：`pytest tests/test_prepare_data.py -q` -> 6 passed in 2.36s
   - 全局回归：`pytest tests/ -q` -> 44 passed in 15.55s
4. **导入命令示例**：
   ```bash
   python scripts/prepare_data.py --aishell_dir E:/data/data_aishell --out_dir data/prepared --workers 8
   ```
   （小样本调试：`python scripts/prepare_data.py --aishell_dir ... --out_dir ... --max_files_per_split 100`）
