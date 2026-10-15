@echo off
chcp 936 >nul
title Gemma-ASR 150h Full Training Pipeline

echo ======================================================================
echo           Gemma-ASR AISHELL-1 全量 150 小时正式微调训练 (G3)
echo ======================================================================
echo 硬件配置: NVIDIA GeForce RTX 4080 (12GB) 单卡
echo 训练数据: data/prepared_full (120,098 句, 150.9 小时)
echo 词表路径: data/prepared_full/units.json (3,791 单元)
echo 核心参数: max_batch_seconds=300, lr=5e-4, warmup=1500, max_steps=28000
echo 并发调度: num_workers=0 (Windows 单进程安全模式, 杜绝多进程死锁)
echo 输出目录: runs/zh_full
echo.
echo [尖子实时止损监控线]
echo  1. loss 出现 NaN/Inf 或连续 3 次上涨 >50%%: 立刻 Ctrl+C 停止并报告
echo  2. 显存占用超过 10.5 GB: 停止并将 max_batch_seconds 降至 240 后续训
echo  3. valid CER 连续 3 次 eval 不再刷新: 可手动提前停止 (已锁 best.pt)
echo  4. 若中途意外断电或关闭: 运行 train_full.bat resume 即可续训
echo ======================================================================
echo.

if not exist "runs\zh_full" mkdir "runs\zh_full"

set RESUME_ARG=
if /i "%1"=="resume" (
    echo [提示] 激活断点续训模式: --resume runs/zh_full/last.pt
    set RESUME_ARG=--resume runs/zh_full/last.pt
)

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [启动] 正在启动全量训练引擎，训练进度将在下方实时打印...
echo [提示] 亦可另开终端运行 python scripts/watch_metrics.py 查看看板！
echo.

"%PYTHON_EXE%" scripts\train.py ^
  --data_dir data/prepared_full ^
  --units data/prepared_full/units.json ^
  --out_dir runs/zh_full ^
  --max_batch_seconds 300 ^
  --lr 5e-4 ^
  --warmup_steps 1500 ^
  --min_lr_ratio 0.05 ^
  --max_steps 28000 ^
  --eval_every 2000 ^
  --save_every 1000 ^
  --max_eval_utts 1500 ^
  --num_workers 0 ^
  --log_every 50 %RESUME_ARG%

echo.
echo ======================================================================
echo [完成] 全量训练已顺利结束！请检查 runs\zh_full\best.pt
echo ======================================================================
pause
