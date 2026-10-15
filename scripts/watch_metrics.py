# -*- coding: utf-8 -*-
"""实时监控训练 metrics.jsonl 的轻量小脚本（不引入新依赖）。

用法：
    python scripts/watch_metrics.py
    python scripts/watch_metrics.py --metrics runs/zh_full/metrics.jsonl
    python scripts/watch_metrics.py --follow
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="训练指标监控器")
    ap.add_argument(
        "--metrics",
        default="runs/zh_full/metrics.jsonl",
        help="metrics.jsonl 路径（默认：runs/zh_full/metrics.jsonl）",
    )
    ap.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="持续刷新监控（类似 tail -f）",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="刷新间隔（秒，默认 5.0）",
    )
    return ap.parse_args()


def read_metrics(path: str):
    if not os.path.exists(path):
        return [], []
    steps = []
    evals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "eval" in rec:
                eval_data = rec["eval"]
                avg_cer = eval_data.get("avg_cer", 0.0)
                # 尝试从具体 source（如 aishell1）提取 pcer
                pcer = 0.0
                for k, v in eval_data.items():
                    if isinstance(v, dict) and "pcer" in v:
                        pcer = v["pcer"]
                        break
                evals.append({
                    "step": rec.get("step", 0),
                    "cer": avg_cer,
                    "pcer": pcer,
                })
            elif "loss" in rec:
                steps.append(rec)
    return steps, evals


def render_dashboard(metrics_path: str):
    steps, evals = read_metrics(metrics_path)
    if not steps and not evals:
        print(f"[watch_metrics] 暂无指标数据，等待日志写入: {metrics_path}")
        return

    latest_step = steps[-1] if steps else {}
    print("=" * 72)
    print(f"📊 Gemma-ASR 训练实时监控看板 [{Path(metrics_path).name}]")
    print("=" * 72)

    if latest_step:
        step_num = latest_step.get("step", 0)
        loss = latest_step.get("loss", 0.0)
        lr = latest_step.get("lr", 0.0)
        gnorm = latest_step.get("gnorm", 0.0)
        mem = latest_step.get("gpu_mem_gb", latest_step.get("mem", 0.0))
        rtf = latest_step.get("rtf_x", 0.0)
        s_step = latest_step.get("s_per_step", 0.0)
        print(f"当前步数 (Step):   {step_num:<6}  | CTC Loss:     {loss:.4f}")
        print(f"当前学习率 (LR):   {lr:.2e}    | 梯度范数:     {gnorm:.2f}")
        print(f"单步耗时:          {s_step:.3f}s     | 实时吞吐:     {rtf:.1f}x 实时")
        print(f"显存占用 (VRAM):   {mem:.2f} GB")
    print("-" * 72)

    if evals:
        print(f"{'Step':<8} | {'CER (%)':<10} | {'P-CER (%)':<10} | {'Time':<18} | {'状态'}")
        print("-" * 72)
        best_cer = float("inf")
        for ev in evals:
            s = ev.get("step", 0)
            cer = ev.get("cer", ev.get("eval_cer", 0.0)) * 100.0
            pcer = ev.get("pcer", ev.get("eval_pcer", 0.0)) * 100.0
            tm = ev.get("time", "")
            if tm and "T" in tm:
                tm = tm.split("T")[1][:8]
            is_best = cer < best_cer
            if is_best:
                best_cer = cer
                badge = "★ 刷新最优 (best.pt)"
            else:
                badge = ""
            print(f"{s:<8} | {cer:<10.2f} | {pcer:<10.2f} | {tm:<18} | {badge}")
        print("-" * 72)
        print(f"🏆 当前最优验证集 CER: {best_cer:.2f}%")
    else:
        print("暂无评估点数据（待触发首个 eval 周期）")
    print("=" * 72)


def main():
    args = parse_args()
    if not args.follow:
        render_dashboard(args.metrics)
        return

    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            render_dashboard(args.metrics)
            print(f"\n[按 Ctrl+C 退出监控] 每 {args.interval} 秒自动刷新一次...")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[watch_metrics] 监控已停止。")


if __name__ == "__main__":
    main()
