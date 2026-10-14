# -*- coding: utf-8 -*-
"""通过 Hugging Face 镜像站 (hf-mirror.com) + Aria2c 极速下载 AISHELL-1 Parquet 数据集

数据集仓库: carlot/AIShell (全量标准 Parquet 格式，总计 18.91 GB，45 个分片)
相比解压 14 万个散碎 wav，Parquet 大文件多线程吞吐更快，完美契合现有数据预处理管线。
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

DEFAULT_ARIA2 = r"D:\download\Aria2\aria2c.exe"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = str(PROJECT_ROOT / "data" / "raw" / "aishell_hf")
HF_MIRROR = "https://hf-mirror.com"
REPO_ID = "carlot/AIShell"


def get_repo_files(mirror_url: str, repo_id: str):
    """从镜像 API 获取仓库文件清单"""
    api_url = f"{mirror_url}/api/datasets/{repo_id}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            siblings = data.get("siblings", [])
            return [s["rfilename"] for s in siblings if "rfilename" in s]
    except Exception as e:
        print(f"❌ 无法连接镜像 API {api_url}: {e}")
        return []


def download_file_aria2(aria2_bin: str, url: str, out_dir: Path, fn: str) -> bool:
    dest = out_dir / fn.replace("/", os.sep)
    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        aria2_bin,
        "-x", "16",
        "-s", "16",
        "-k", "2M",
        "-c",
        "--console-log-level=warn",
        "--summary-interval=3",
        "-d", str(dest.parent),
        "-o", dest.name,
        url,
    ]
    ret = subprocess.run(cmd)
    return ret.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Hugging Face 镜像 AISHELL-1 高速下载器")
    parser.add_argument("--split", choices=["all", "test", "dev", "train", "quick"], default="quick",
                        help="下载模式: quick(仅1个test分片测试,约360M), test(全部测试集,约1.1G), all(全量18.9G)")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="E 盘落地目录")
    parser.add_argument("--aria2", default=DEFAULT_ARIA2, help="aria2c 路径")
    parser.add_argument("--mirror", default=HF_MIRROR, help="镜像站基础地址")
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 AISHELL-1 Hugging Face 镜像极速下载器 (Parquet 格式)")
    print(f"🌐 镜像站: {args.mirror}")
    print(f"📦 目标仓库: {REPO_ID}")
    print(f"📁 落地目录: {args.out_dir}")
    print(f"🎯 选定模式: {args.split}")
    print("=" * 70)

    if not os.path.exists(args.aria2):
        print(f"❌ 未找到 aria2c: {args.aria2}")
        sys.exit(1)

    print("🔍 正在检索镜像站文件元数据...")
    all_files = get_repo_files(args.mirror, REPO_ID)
    if not all_files:
        print("❌ 无法获取文件列表，请检查网络。")
        sys.exit(1)

    # 过滤待下载的文件
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    
    if args.split == "quick":
        # 仅取第 1 个 test 分片 (约 360MB)，用于秒速验证数据管线
        targets = [f for f in parquet_files if "test" in f][:1]
    elif args.split == "test":
        targets = [f for f in parquet_files if "test" in f]
    elif args.split == "dev":
        targets = [f for f in parquet_files if "dev" in f]
    elif args.split == "train":
        targets = [f for f in parquet_files if "train" in f]
    else:  # all
        targets = parquet_files

    print(f"🎯 待下载分片数: {len(targets)} 个文件")
    out_path = Path(args.out_dir)

    for idx, fn in enumerate(targets, 1):
        url = f"{args.mirror}/datasets/{REPO_ID}/resolve/main/{fn}"
        dest = out_path / fn.replace("/", os.sep)
        temp = dest.with_name(dest.name + ".aria2")

        if dest.exists() and not temp.exists() and dest.stat().st_size > 1024 * 1024:
            print(f"[{idx}/{len(targets)}] ⏩ 已存在且完整，跳过: {fn} ({dest.stat().st_size / (1024**2):.1f} MB)")
            continue

        print(f"\n[{idx}/{len(targets)}] 🚀 下载中: {fn}")
        ok = download_file_aria2(args.aria2, url, out_path, fn)
        if not ok:
            print(f"❌ 下载失败: {fn}")
            sys.exit(1)
        print(f"[{idx}/{len(targets)}] ✅ 下载完成: {fn}")

    print("\n🎉 下载任务顺利完成！")


if __name__ == "__main__":
    main()
