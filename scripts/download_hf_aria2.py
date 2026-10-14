# -*- coding: utf-8 -*-
"""
Hugging Face 镜像站 Aria2c 高速并行下载脚本
支持 google/embeddinggemma-2 原版与 unsloth/embeddinggemma-2-GGUF 量化版
"""

import os
import sys
import json
import subprocess
import urllib.request
import argparse
from pathlib import Path

DEFAULT_ARIA2 = r"D:\download\Aria2\aria2c.exe"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_DIR = str(PROJECT_ROOT / "models")
HF_MIRROR = "https://hf-mirror.com"

# 待下载的仓库定义
REPOS = {
    "original": {
        "repo_id": "google/embeddinggemma-2",
        "desc": "Google 原版 EmbeddingGemma 2 (Transformers/PyTorch 格式，用于训练与 Phase 0 探针)",
        "filter": lambda fn: not fn.startswith(".git"),
    },
    "quantized": {
        "repo_id": "unsloth/embeddinggemma-2-GGUF",
        "desc": "Unsloth GGUF 量化版 (含 Q8_0 / Q4_K_XL 文本模型与 mmproj 多模态投影，用于端侧极速推理)",
        # 默认下载最核心的常用量化组合：Q8_0 与 Q4_K_XL 以及对应的多模态投影 mmproj
        "filter": lambda fn: fn in [
            "README.md",
            "embeddinggemma-2-Q8_0.gguf",
            "embeddinggemma-2-UD-Q4_K_XL.gguf",
            "mmproj-Q8_0.gguf",
            "mmproj-BF16.gguf",
        ],
    },
}

def get_repo_files(repo_id: str, mirror_url: str):
    """通过镜像站 API 获取文件列表"""
    api_url = f"{mirror_url}/api/models/{repo_id}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            siblings = data.get("siblings", [])
            return [s["rfilename"] for s in siblings if "rfilename" in s]
    except Exception as e:
        print(f"[错误] 无法获取仓库文件列表 {repo_id}: {e}")
        return []

def download_file_with_aria2(aria2_bin: str, file_url: str, out_dir: str, file_name: str):
    """调用 aria2c 执行多线程断点续传下载"""
    target_path = Path(out_dir) / file_name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        aria2_bin,
        "-x", "16",                     # 单任务最大连接数
        "-s", "16",                     # 单服务器最大连接数
        "-k", "1M",                     # 分块大小
        "-c",                           # 断点续传
        "--console-log-level=warn",      # 日志级别
        "--summary-interval=2",          # 进度刷新间隔
        "-d", str(target_path.parent),   # 目标目录
        "-o", target_path.name,          # 保存文件名
        file_url
    ]
    ret = subprocess.run(cmd)
    return ret.returncode == 0

def download_repo(repo_key: str, aria2_bin: str, base_out_dir: str, mirror_url: str):
    repo_info = REPOS[repo_key]
    repo_id = repo_info["repo_id"]
    desc = repo_info["desc"]
    filter_fn = repo_info["filter"]

    target_dir = Path(base_out_dir) / repo_id.replace("/", os.sep)
    target_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"📦 开始处理: [{repo_key.upper()}] {repo_id}")
    print(f"ℹ️  说明: {desc}")
    print(f"📂 落地路径: {target_dir}")
    print("=" * 70)

    all_files = get_repo_files(repo_id, mirror_url)
    if not all_files:
        print(f"[警告] 仓库 {repo_id} 未检索到文件或网络连接失败，跳过。")
        return False

    targets = [f for f in all_files if filter_fn(f)]
    print(f"🎯 待下载文件清单 (共 {len(targets)} 个文件):")
    for f in targets:
        print(f"   • {f}")
    print("-" * 70)

    success_count = 0
    for idx, rfn in enumerate(targets, 1):
        file_url = f"{mirror_url}/{repo_id}/resolve/main/{rfn}"
        dest_file = target_dir / rfn.replace("/", os.sep)

        # 检查是否已有完全匹配的文件（如果不存在 .aria2 临时文件且文件存在）
        aria2_temp = dest_file.with_name(dest_file.name + ".aria2")
        if dest_file.exists() and not aria2_temp.exists() and dest_file.stat().st_size > 0:
            print(f"[{idx}/{len(targets)}] ⏩ 已存在且无未完标记，跳过: {rfn}")
            success_count += 1
            continue

        print(f"[{idx}/{len(targets)}] 🚀 下载中: {rfn}")
        ok = download_file_with_aria2(aria2_bin, file_url, str(target_dir), rfn)
        if ok:
            success_count += 1
            print(f"[{idx}/{len(targets)}] ✅ 下载完成: {rfn}")
        else:
            print(f"[{idx}/{len(targets)}] ❌ 下载失败: {rfn}")

    print(f"\n🎉 仓库 [{repo_id}] 下载任务完成！成功: {success_count}/{len(targets)}")
    return success_count == len(targets)

def main():
    parser = argparse.ArgumentParser(description="EmbeddingGemma 2 HF-Mirror Aria2 下载工具")
    parser.add_argument("--stage", choices=["all", "original", "quantized"], default="all",
                        help="选择下载阶段：original (仅原版), quantized (仅量化), all (原版+量化)")
    parser.add_argument("--dest", default=DEFAULT_BASE_DIR, help="基础输出目录，默认 E:\\models")
    parser.add_argument("--aria2", default=DEFAULT_ARIA2, help="aria2c.exe 完整路径")
    parser.add_argument("--mirror", default=HF_MIRROR, help="镜像站基础地址")
    args = parser.parse_args()

    if not os.path.exists(args.aria2):
        print(f"[严重错误] 未找到 aria2c.exe: {args.aria2}")
        sys.exit(1)

    print("=" * 70)
    print("🚀 EmbeddingGemma 2 极速多线程镜像下载器 (Aria2c + HF-Mirror)")
    print(f"⚙️  Aria2 执行路径: {args.aria2}")
    print(f"🌐 镜像站地址: {args.mirror}")
    print(f"📁 根输出目录: {args.dest}")
    print(f"📌 执行任务模式: {args.stage}")
    print("=" * 70)

    if args.stage in ["all", "original"]:
        download_repo("original", args.aria2, args.dest, args.mirror)

    if args.stage in ["all", "quantized"]:
        download_repo("quantized", args.aria2, args.dest, args.mirror)

    print("\n" + "=" * 70)
    print("✨ 全部指定任务已执行完毕！")
    print(f"📂 原版落地路径: {os.path.join(args.dest, 'google', 'embeddinggemma-2')}")
    print(f"📂 量化版落地路径: {os.path.join(args.dest, 'unsloth', 'embeddinggemma-2-GGUF')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
