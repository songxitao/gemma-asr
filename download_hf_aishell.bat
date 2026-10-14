@echo off
chcp 936 >nul
title AISHELL-1 HF-Mirror 极速下载器 (Parquet)
echo ========================================================
echo   AISHELL-1 HF-Mirror 镜像极速下载 (Aria2c + Parquet)
echo   仓库: carlot/AIShell (免解压 14 万个零碎小文件)
echo   落地目录: E:\project\gemma-asr\data\raw\aishell_hf
echo ========================================================
echo.
echo 模式说明:
echo   1. 输入 quick : 仅下载第 1 个测试集分片 (约 360MB，用于秒速验证)
echo   2. 输入 test  : 下载全部测试集 (3 个分片，约 1.1GB)
echo   3. 输入 all   : 下载全量 178 小时数据集 (45 个分片，约 18.9GB)
echo.

set /p mode="请输入模式 [quick/test/all] (默认 quick): "
if "%mode%"=="" set mode=quick

"E:\project\gemma-asr\.venv\Scripts\python.exe" "E:\project\gemma-asr\scripts\download_hf_aishell.py" --split %mode%

echo.
echo ========================================================
echo   处理完成！按任意键退出...
echo ========================================================
pause
