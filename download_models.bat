@echo off
chcp 936 >nul
title EmbeddingGemma 2 模型极速下载器 (Aria2c + HF-Mirror)
color 0A

echo ======================================================================
echo           EmbeddingGemma 2 极速镜像下载器 (Aria2 + HF-Mirror)
echo ======================================================================
echo.
echo  本脚本将依次下载：
echo    1. [原版] google/embeddinggemma-2 (Transformers/PyTorch, 约 1.45 GB)
echo    2. [量化版] unsloth/embeddinggemma-2-GGUF (Q8_0 / Q4_K_XL + mmproj, 约 1.2 GB)
echo.
echo  落地目录：%~dp0models\
echo  下载引擎：D:\download\Aria2\aria2c.exe (16线程断点续传)
echo  镜像源：  https://hf-mirror.com
echo.
echo ======================================================================
echo.

set "PYTHON_EXE=D:\program files\Miniconda\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [错误] 未在系统找到 Miniconda Python: %PYTHON_EXE%
    echo 尝试使用环境变量中的 python...
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" "%~dp0scripts\download_hf_aria2.py" --stage all --dest "%~dp0models"

echo.
echo ======================================================================
echo  下载流程执行完毕，请查看上方输出信息。
echo ======================================================================
echo.
pause
