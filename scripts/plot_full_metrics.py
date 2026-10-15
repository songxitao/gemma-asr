# -*- coding: utf-8 -*-
"""
生成全量 28,000 步训练收敛图表 (Loss, Learning Rate, CER, P-CER)
导出至 assets/training_curve.png
"""
import os
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

# 避免中文乱码，优先使用系统常见中文字体，fallback 到无衬线
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

METRICS_PATH = "runs/zh_full/metrics.jsonl"
OUTPUT_DIR = "assets"
OUTPUT_IMG = os.path.join(OUTPUT_DIR, "training_curve.png")


def main():
    if not os.path.exists(METRICS_PATH):
        print(f"Error: {METRICS_PATH} not found.")
        return

    steps_train = []
    losses = []
    lrs = []
    
    steps_eval = []
    cers = []
    pcers = []

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            step = d["step"]
            if "loss" in d and step >= 100:  # 排除前 100 步初始高噪点便于观察细节
                steps_train.append(step)
                losses.append(d["loss"])
                lrs.append(d["lr"])
            if "eval" in d and "aishell1" in d["eval"]:
                steps_eval.append(step)
                cers.append(d["eval"]["aishell1"]["cer"] * 100)
                pcers.append(d["eval"]["aishell1"]["pcer"] * 100)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), dpi=300)

    # ----------------------------------------------------
    # Subplot 1: Loss & Learning Rate
    # ----------------------------------------------------
    color_loss = "#1f77b4"
    ax1.set_title("Gemma-ASR 28,000-Step Training Dynamics (AISHELL-1 150h)", fontsize=13, pad=10, fontweight="bold")
    ax1.set_xlabel("Training Steps", fontsize=11)
    ax1.set_ylabel("CTC Loss", color=color_loss, fontsize=11)
    line1 = ax1.plot(steps_train, losses, color=color_loss, alpha=0.85, linewidth=1.5, label="CTC Loss")
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax1_twin = ax1.twinx()
    color_lr = "#ff7f0e"
    ax1_twin.set_ylabel("Learning Rate", color=color_lr, fontsize=11)
    line2 = ax1_twin.plot(steps_train, lrs, color=color_lr, linestyle="--", linewidth=1.5, label="Cosine LR")
    ax1_twin.tick_params(axis='y', labelcolor=color_lr)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    # ----------------------------------------------------
    # Subplot 2: Validation CER & P-CER
    # ----------------------------------------------------
    color_cer = "#d62728"
    color_pcer = "#2ca02c"
    ax2.set_title("Validation Convergence: CER & Pinyin-CER (Speaker-Independent Valid Set)", fontsize=13, pad=10, fontweight="bold")
    ax2.set_xlabel("Training Steps", fontsize=11)
    ax2.set_ylabel("Error Rate (%)", fontsize=11)
    
    p_cer = ax2.plot(steps_eval, cers, color=color_cer, marker="o", markersize=6, linewidth=2.0, label="Character Error Rate (CER)")
    p_pcer = ax2.plot(steps_eval, pcers, color=color_pcer, marker="s", markersize=5, linewidth=1.8, label="Pinyin Error Rate (P-CER)")
    ax2.grid(True, linestyle="--", alpha=0.4)

    # 关键里程碑标注
    best_idx = cers.index(min(cers))
    best_step = steps_eval[best_idx]
    best_cer = cers[best_idx]
    best_pcer = pcers[best_idx]
    ax2.annotate(
        f"Best CER: {best_cer:.2f}%\n(P-CER: {best_pcer:.2f}%)",
        xy=(best_step, best_cer),
        xytext=(best_step - 4500, best_cer + 4.5),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff2f2", ec="#d62728", lw=1)
    )

    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    plt.close()
    print(f"✅ 训练收敛图已生成: {OUTPUT_IMG}")


if __name__ == "__main__":
    main()
