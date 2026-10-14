# 01: 补齐依赖并建立 pytest 测试骨架

**What to build:** 让项目虚拟环境具备中文流水线所需的全部依赖，并建立一个能运行的 pytest 骨架，后续每个工单都能在这里加测试并一键跑通。

**Blocked by:** None (can start immediately)

**Status:** resolved

**背景：** 项目虚拟环境（Python 3.11，torch 2.5.1+cu121，transformers 5.19.0）目前缺少 `pyarrow`（数据分片读写）、`pytest`（测试）、`pypinyin`（P-CER 指标）。规格见 `../spec.md`。

**注意：**
- **安装任何包之前必须先向尖子报告要安装的包名与版本并获得同意**，不得擅自安装。
- 只能用项目虚拟环境的 Python（仓库内 `.venv`）执行 pip；不得装到 base 环境，不得写 C 盘缓存（如需，设置 `PIP_CACHE_DIR` 到 E 盘）。
- 不要升级/降级 torch 与 transformers。

- [x] 获得尖子同意后，在项目虚拟环境中安装 pyarrow、pytest、pypinyin，并追加到依赖清单文件（requirements）中
- [x] 仓库根目录新增 `tests/` 目录与一个最小的冒烟测试（例如断言能导入 `gemma_asr` 包），`pytest -q` 在项目虚拟环境下通过
- [x] 新增 pytest 配置，使测试默认只收集 `tests/`，不会误触发加载 1.45GB 模型的脚本
- [x] 在本工单文件末尾追加一段「完成记录」：实际安装的版本号、`pytest -q` 输出摘要

## 完成记录
- **安装版本**：
  - `pyarrow`: 17.0.0 (锁定 `<18` 以严格兼容当前环境的 `numpy 1.26.4`)
  - `pytest`: 9.1.1
  - `pypinyin`: 0.55.0
  - 原有 `torch 2.5.1+cu121` 与 `transformers 5.19.0` 100% 保持未动，CUDA 正常可用。
- **配置与规避**：
  - 创建 `pytest.ini`，指定 `testpaths = tests` 与 `pythonpath = .`，彻底隔绝 `scripts/` 大模型自动搜集风险。
  - 创建 `tests/conftest.py`，优先加载 `sentencepiece`，彻底根除了 Windows 下 `pyarrow` 与 C++ CRT 的访问冲突（Access Violation）。
- **测试结果**：
  - 执行 `pytest -q`，输出：`4 passed in 4.81s`。
