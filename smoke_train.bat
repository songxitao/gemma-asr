@echo off
chcp 936 >nul
title Gemma-ASR 中文冒烟训练 (4080 单卡 1500 步)
echo ========================================================
echo   Gemma-ASR 中文极速 ASR - RTX 4080 Laptop 单卡冒烟训练
echo ========================================================
echo.
echo 正在启动 1500 步冒烟训练...
echo 观察要点：显存稳定性、Loss 平稳下降、CER 逐步收敛。
echo.

"E:\project\gemma-asr\.venv\Scripts\python.exe" scripts\train.py ^
    --data_dir data/prepared ^
    --units data/prepared/units.json ^
    --out_dir runs/zh_smoke ^
    --encoder_id models/google/embeddinggemma-2 ^
    --max_steps 1500 ^
    --max_batch_seconds 300.0 ^
    --eval_every 500 ^
    --save_every 500 ^
    --log_every 50 ^
    --num_workers 0

echo.
echo ========================================================
echo   冒烟训练已完成！请查看 runs/zh_smoke 目录结果与日志。
echo ========================================================
pause
