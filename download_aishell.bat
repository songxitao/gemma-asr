@echo off
chcp 936 >nul
title AISHELL-1 全量中文数据集高速下载器
echo ========================================================
echo   AISHELL-1 全量中文数据集高速下载 (Aria2c + 国内源)
echo   落地目录: E:\project\gemma-asr\data\raw\aishell
echo ========================================================
echo.

"E:\project\gemma-asr\.venv\Scripts\python.exe" "E:\project\gemma-asr\scripts\download_aishell.py" --clean --extract

echo.
echo ========================================================
echo   处理完成！按任意键退出...
echo ========================================================
pause
