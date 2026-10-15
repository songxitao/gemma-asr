# -*- coding: utf-8 -*-
"""Plot training curves and validation metrics from runs/zh_smoke/metrics.jsonl."""

import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = PROJECT_ROOT / "runs" / "zh_smoke" / "metrics.jsonl"
OUT_IMG = PROJECT_ROOT / "runs" / "zh_smoke" / "training_metrics.png"
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", str(PROJECT_ROOT / "assets")))

# Configure Matplotlib fonts for Windows Chinese rendering
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    if not METRICS_PATH.exists():
        print(f"Error: {METRICS_PATH} does not exist.")
        return

    steps, losses, lrs, gnorms, mems, rtfs = [], [], [], [], [], []
    eval_steps, cers, pcers = [], [], []

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "loss" in data:
                steps.append(data["step"])
                losses.append(data["loss"])
                lrs.append(data["lr"])
                gnorms.append(data["grad_norm"])
                mems.append(data["gpu_mem_gb"])
                rtfs.append(data["rtf_x"])
            elif "eval" in data:
                eval_steps.append(data["step"])
                ev = data["eval"]["aishell1"]
                cers.append(ev["cer"] * 100.0)
                pcers.append(ev["pcer"] * 100.0)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle("Gemma-ASR 中文 1500 步冒烟训练全景评估报告 (RTX 4080 Laptop)", fontsize=16, fontweight="bold")

    # 1. CTC Loss 曲线
    ax1 = axes[0, 0]
    ax1.plot(steps, losses, color="#2563eb", linewidth=2.2, marker="o", markersize=3, label="CTC Loss")
    ax1.set_title("1. CTC 训练损失收敛曲线 (CTC Loss)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("训练步数 (Steps)")
    ax1.set_ylabel("损失值 (Loss)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")
    # Annotate start and final loss
    ax1.annotate(f"初始: {losses[0]:.2f}", xy=(steps[0], losses[0]), xytext=(steps[0]+50, losses[0]+1),
                 arrowprops=dict(arrowstyle="->", color="gray"))
    ax1.annotate(f"最终: {losses[-1]:.3f}", xy=(steps[-1], losses[-1]), xytext=(steps[-1]-300, losses[-1]+1.5),
                 arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.5), fontweight="bold")

    # 2. 评测指标收敛曲线 (CER & P-CER)
    ax2 = axes[0, 1]
    width = 60
    ax2.plot(eval_steps, cers, color="#dc2626", linewidth=2.5, marker="s", markersize=6, label="字错率 (CER %)")
    ax2.plot(eval_steps, pcers, color="#16a34a", linewidth=2.5, marker="^", markersize=6, label="拼音错率 (P-CER %)")
    for s, c, p in zip(eval_steps, cers, pcers):
        ax2.annotate(f"CER: {c:.2f}%\nP-CER: {p:.2f}%", xy=(s, c), xytext=(s-80, c+3),
                     fontsize=9, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef2f2", edgecolor="#dc2626", alpha=0.8))
    ax2.set_title("2. 验证集错误率下降曲线 (Valid CER & P-CER)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("训练步数 (Steps)")
    ax2.set_ylabel("错误率 (%)")
    ax2.set_ylim(0, 45)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    # 3. 学习率与梯度范数
    ax3 = axes[1, 0]
    color_lr = "#9333ea"
    ax3.plot(steps, [lr * 1e3 for lr in lrs], color=color_lr, linewidth=2, label="学习率 (×10⁻³)")
    ax3.set_xlabel("训练步数 (Steps)")
    ax3.set_ylabel("学习率 (×10⁻³)", color=color_lr)
    ax3.tick_params(axis="y", labelcolor=color_lr)
    ax3.grid(True, linestyle="--", alpha=0.5)

    ax3_twin = ax3.twinx()
    color_gnorm = "#d97706"
    ax3_twin.plot(steps, gnorms, color=color_gnorm, linewidth=1.5, linestyle="--", label="梯度范数 (G-Norm)")
    ax3_twin.set_ylabel("梯度范数 (G-Norm)", color=color_gnorm)
    ax3_twin.tick_params(axis="y", labelcolor=color_gnorm)
    ax3.set_title("3. 学习率退火与梯度稳定性", fontsize=12, fontweight="bold")

    # 4. 显存与吞吐倍率
    ax4 = axes[1, 1]
    color_rtf = "#0891b2"
    ax4.plot(steps, rtfs, color=color_rtf, linewidth=2, label="实时吞吐倍率 (Realtime X)")
    ax4.set_xlabel("训练步数 (Steps)")
    ax4.set_ylabel("实时吞吐倍率 (x)", color=color_rtf)
    ax4.tick_params(axis="y", labelcolor=color_rtf)
    ax4.grid(True, linestyle="--", alpha=0.5)

    ax4_twin = ax4.twinx()
    color_mem = "#475569"
    ax4_twin.plot(steps, mems, color=color_mem, linewidth=2, linestyle=":", label="显存峰值 (GB)")
    ax4_twin.set_ylabel("显存占用 (GB)", color=color_mem)
    ax4_twin.set_ylim(0, 12)
    ax4_twin.axhline(12.0, color="#ef4444", linestyle="--", alpha=0.6, label="4080 显存上限 (12GB)")
    ax4_twin.tick_params(axis="y", labelcolor=color_mem)
    ax4.set_title("4. 算力开销与显存安全线", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=150)
    print(f"Chart saved to {OUT_IMG}")

    if ARTIFACT_DIR.exists():
        dest_art = ARTIFACT_DIR / "training_metrics.png"
        shutil.copy2(OUT_IMG, dest_art)
        print(f"Copied to artifact dir: {dest_art}")


if __name__ == "__main__":
    main()
