# -*- coding: utf-8 -*-
"""AISHELL-1 (OpenSLR 33) 官方数据集高速断点续传下载工具

支持国内高速镜像源（MagicData / 上海交大）与 Hugging Face 镜像（hf-mirror.com），
默认使用本地 Aria2 (D:\\download\\Aria2\\aria2c.exe) 16 线程断点续传。
落地目录严格锁死在 E 盘，杜绝向 C 盘写入。
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

# 默认配置
DEFAULT_ARIA2 = r"D:\download\Aria2\aria2c.exe"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = str(PROJECT_ROOT / "data" / "raw" / "aishell")

# AISHELL-1 数据源地址映射 (总计约 15.58 GB)
SOURCES = {
    "magicdata": {
        "name": "Magic Data 国内高速镜像 (推荐，速度快)",
        "files": {
            "data_aishell.tgz": "https://openslr.magicdatatech.com/resources/33/data_aishell.tgz",
            "resource_aishell.tgz": "https://openslr.magicdatatech.com/resources/33/resource_aishell.tgz",
        },
    },
    "sjtug": {
        "name": "上海交通大学开源镜像站",
        "files": {
            "data_aishell.tgz": "https://mirrors.sjtug.sjtu.edu.cn/openslr/33/data_aishell.tgz",
            "resource_aishell.tgz": "https://mirrors.sjtug.sjtu.edu.cn/openslr/33/resource_aishell.tgz",
        },
    },
    "openslr": {
        "name": "OpenSLR 官方源 (海外)",
        "files": {
            "data_aishell.tgz": "https://www.openslr.org/resources/33/data_aishell.tgz",
            "resource_aishell.tgz": "https://www.openslr.org/resources/33/resource_aishell.tgz",
        },
    },
}

IDM_LEFTOVER_DIR = r"C:\Users\song\AppData\Roaming\IDM\DwnlData\song\data_aishell_177"


def clean_idm_leftovers():
    """清理历史未下完的 IDM 残余分块，释放 C 盘空间。"""
    if os.path.exists(IDM_LEFTOVER_DIR):
        print(f"🧹 检测到 IDM 历史残余目录: {IDM_LEFTOVER_DIR}")
        try:
            total_size = sum(
                f.stat().st_size for f in Path(IDM_LEFTOVER_DIR).rglob("*") if f.is_file()
            )
            shutil.rmtree(IDM_LEFTOVER_DIR)
            print(f"✅ 成功清理 IDM 残余文件，释放 C 盘空间: {total_size / (1024**2):.2f} MB")
        except Exception as e:
            print(f"⚠️ 清理残余失败 (可能被占用): {e}")
    else:
        print("✨ 无需清理，IDM 残余目录不存在。")


def download_with_aria2(aria2_bin: str, url: str, out_dir: str, file_name: str) -> bool:
    """使用 Aria2c 执行多线程断点续传。"""
    dest = Path(out_dir) / file_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        aria2_bin,
        "-x", "16",
        "-s", "16",
        "-k", "1M",
        "-c",
        "--console-log-level=warn",
        "--summary-interval=5",
        "-d", str(dest.parent),
        "-o", file_name,
        url,
    ]
    ret = subprocess.run(cmd)
    return ret.returncode == 0


def extract_tar_file(tgz_path: Path, extract_to: Path):
    """解压 tgz 压缩文件。"""
    print(f"📦 开始解压: {tgz_path.name} -> {extract_to} ...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"✅ 解压完成: {tgz_path.name}")


def main():
    parser = argparse.ArgumentParser(description="AISHELL-1 数据集高速断点续传下载工具")
    parser.add_argument("--source", choices=list(SOURCES.keys()), default="magicdata",
                        help="下载镜像源选择 (默认 magicdata)")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="E 盘落地目录")
    parser.add_argument("--aria2", default=DEFAULT_ARIA2, help="aria2c 可执行文件路径")
    parser.add_argument("--clean", action="store_true", help="清理历史 IDM 残留")
    parser.add_argument("--extract", action="store_true", help="下载完成后自动解压")
    args = parser.parse_args()

    print("=" * 70)
    print("🎙️ AISHELL-1 (178小时) 全量中文语音数据集高速下载器")
    print(f"📁 存储目录: {args.out_dir}")
    print(f"🌐 选定源: {SOURCES[args.source]['name']}")
    print(f"⚡ Aria2 路径: {args.aria2}")
    print("=" * 70)

    if args.clean:
        clean_idm_leftovers()

    if not os.path.exists(args.aria2):
        print(f"❌ 严重错误: 未找到 aria2c.exe ({args.aria2})，请确认路径。")
        sys.exit(1)

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    source_cfg = SOURCES[args.source]
    for fn, url in source_cfg["files"].items():
        dest_file = out_path / fn
        aria2_temp = dest_file.with_name(fn + ".aria2")
        
        # 简单校验是否已下完
        if dest_file.exists() and not aria2_temp.exists() and dest_file.stat().st_size > 1024 * 1024:
            print(f"⏩ 文件已存在且无未完标记，跳过: {fn} ({dest_file.stat().st_size / (1024**3):.2f} GB)")
            continue

        print(f"\n🚀 开始下载: {fn}")
        print(f"🔗 URL: {url}")
        ok = download_with_aria2(args.aria2, url, str(out_path), fn)
        if not ok:
            print(f"❌ 下载中断或失败: {fn}")
            sys.exit(1)
        print(f"✅ 下载成功: {fn}")

    print("\n🎉 全部文件下载完毕！")

    if args.extract:
        data_tgz = out_path / "data_aishell.tgz"
        res_tgz = out_path / "resource_aishell.tgz"
        if data_tgz.exists():
            extract_tar_file(data_tgz, out_path)
        if res_tgz.exists():
            extract_tar_file(res_tgz, out_path)
        print("🎉 全部数据集解压就绪！")


if __name__ == "__main__":
    main()
