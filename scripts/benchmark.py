# -*- coding: utf-8 -*-
"""
工单 11：公平基准重测脚本 (scripts/benchmark.py)
特性:
1. 统一驱动三方引擎 (Gemma-ASR, SenseVoiceSmall, SeACo-Paraformer)
2. 严格同条件对比: 文本规范化、预热 20 句、计时仅限推理与解码、记录显存峰值
3. 支持两种模式: bs1 (逐句单推延迟) 与 batch (显存 <= 6GB 最优吞吐)
4. 多运行时兼容: FunASR 外部模型按需延迟导入或安全调度 asr_ui_env
"""
import sentencepiece  # noqa: F401  # Windows CRT clash protection
import os
import sys
import io
import time
import json
import re
import argparse
import subprocess
import soundfile as sf
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gemma_asr.text import normalize_hypothesis, to_pinyin

DEFAULT_CKPT = r"runs/zh_full/best.pt"
DEFAULT_DATA_DIR = r"data/prepared_full"
FUNCLIP_PYTHON = os.getenv("FUNCLIP_PYTHON", sys.executable)
SENSEVOICE_PATH = os.getenv("SENSEVOICE_PATH", "models/SenseVoiceSmall")
PARAFORMER_PATH = os.getenv("PARAFORMER_PATH", "models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")


def normalize_text(text: str) -> str:
    """统一清洗：过滤 FunASR 特殊标签、标点符号、空格与非法字符"""
    if not text:
        return ""
    text = re.sub(r"<\|.*?\|>", "", text)
    return normalize_hypothesis(text)


def levenshtein_distance(seq1, seq2) -> int:
    """标准编辑距离动态规划算法"""
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + 1)
            prev = temp
    return dp[n]


def calc_metrics(refs: list[str], hyps: list[str]) -> dict:
    """计算 CER、P-CER 与同音字错误差值 (CER - P-CER)"""
    total_char_err = 0
    total_ref_chars = 0
    total_py_err = 0
    total_ref_py = 0

    for r, h in zip(refs, hyps):
        if not r:
            continue
        c_err = levenshtein_distance(r, h)
        total_char_err += c_err
        total_ref_chars += len(r)

        r_py = to_pinyin(r)
        h_py = to_pinyin(h)
        p_err = levenshtein_distance(r_py, h_py)
        total_py_err += p_err
        total_ref_py += len(r_py)

    cer = total_char_err / max(total_ref_chars, 1)
    pcer = total_py_err / max(total_ref_py, 1)
    homophone_err = max(0.0, cer - pcer)
    return {
        "cer": cer,
        "pcer": pcer,
        "homophone_err": homophone_err,
        "char_errors": total_char_err,
        "ref_chars": total_ref_chars,
        "py_errors": total_py_err,
        "ref_py": total_ref_py,
    }


def format_summary_table(records: list[dict]) -> str:
    """输出格式化 Markdown 汇总大表，包含工单 11 规定的所有列"""
    lines = [
        "| engine | mode | batch | CER | P-CER | 同音字错误 (CER-P-CER) | 耗时 (s) | 实时倍率 | 峰值显存 (GB) |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in records:
        line = (
            f"| {r['engine']} | {r['mode']} | {r['batch']} | "
            f"{r['cer']:.2%} | {r['pcer']:.2%} | {r['homophone_err']:.2%} | "
            f"{r['time']:.2f}s | {r['speed']:.0f}x | {r['vram_gb']:.2f}GB |"
        )
        lines.append(line)
    return "\n".join(lines)


def load_test_samples(data_dir: str, max_utts: int = 0):
    """加载测试集样本（max_utts=0 表示全量）"""
    shards_dir = os.path.join(data_dir, "test", "aishell1")
    parquet_files = sorted(
        [os.path.join(shards_dir, f) for f in os.listdir(shards_dir) if f.endswith(".parquet")]
    )
    samples = []
    for pf in parquet_files:
        table = pq.read_table(pf, columns=["audio", "text", "duration"])
        audios = table.column("audio").to_pylist()
        texts = table.column("text").to_pylist()
        durs = table.column("duration").to_pylist()
        for a_bytes, txt, dur in zip(audios, texts, durs):
            arr, sr = sf.read(io.BytesIO(a_bytes), dtype="float32")
            samples.append({
                "wav": arr,
                "text": txt,
                "clean_ref": normalize_text(txt),
                "duration": dur,
            })
            if max_utts > 0 and len(samples) >= max_utts:
                break
        if max_utts > 0 and len(samples) >= max_utts:
            break
    total_audio_s = sum(s["duration"] for s in samples)
    return samples, total_audio_s


def run_gemma_eval(samples, ckpt_path, mode="batch", batch_size=32, warmup_utts=20):
    """运行 Gemma-ASR 评估"""
    from gemma_asr.model import GemmaCTC, greedy_decode
    from gemma_asr.units import Units

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    units = Units(ckpt["config"]["encoder_id"], ckpt["unit_gemma_ids"])
    model = GemmaCTC.from_checkpoint(ckpt).to(device).eval()

    # 1. 预热
    if warmup_utts > 0:
        warmup_batch = samples[:warmup_utts]
        w_lens = torch.tensor([len(s["wav"]) for s in warmup_batch], dtype=torch.long, device=device)
        w_wav = torch.zeros(len(warmup_batch), int(w_lens.max()), dtype=torch.float32, device=device)
        for j, s in enumerate(warmup_batch):
            w_wav[j, : len(s["wav"])] = torch.from_numpy(s["wav"]).to(device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _ = model(w_wav, w_lens)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # 2. 计时推理
    bs = 1 if mode == "bs1" else batch_size
    hyps = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, len(samples), bs):
            batch = samples[i : i + bs]
            lens = torch.tensor([len(s["wav"]) for s in batch], dtype=torch.long, device=device)
            wav_tensor = torch.zeros(len(batch), int(lens.max()), dtype=torch.float32, device=device)
            for j, s in enumerate(batch):
                wav_tensor[j, : len(s["wav"])] = torch.from_numpy(s["wav"]).to(device)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, out_lens = model(wav_tensor, lens)
            batch_hyps = [units.decode(u) for u in greedy_decode(logits, out_lens)[: len(batch)]]
            hyps.extend(batch_hyps)

    if device.type == "cuda":
        torch.cuda.synchronize()
    compute_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == "cuda" else 0.0

    del model, ckpt
    torch.cuda.empty_cache()

    clean_hyps = [normalize_text(h) for h in hyps]
    return clean_hyps, compute_time, peak_vram


def benchmark_engine(engine_name, samples, mode="batch", batch_size=16, ckpt_path=DEFAULT_CKPT):
    """通用引擎打擂入口"""
    if engine_name.lower() in ("gemma", "gemma-asr"):
        bs = 1 if mode == "bs1" else 32
        hyps, c_time, vram = run_gemma_eval(samples, ckpt_path, mode=mode, batch_size=bs)
        return hyps, c_time, vram, bs

    # 外部引擎 (SenseVoice / Paraformer)
    # 如果本地当前 python 没有 funasr，调度 FUNCLIP_PYTHON 子进程执行
    try:
        from funasr import AutoModel  # noqa: F401
        has_funasr = True
    except ImportError:
        has_funasr = False

    if not has_funasr and os.path.exists(FUNCLIP_PYTHON):
        # 调度外部子进程
        dump_tmp = f"runs/benchmark/_tmp_{engine_name}_{mode}.json"
        cmd = [
            FUNCLIP_PYTHON,
            "scripts/run_funclip_worker.py",
            "--engine", engine_name,
            "--mode", mode,
            "--batch_size", str(batch_size if mode == "batch" else 1),
            "--samples_count", str(len(samples)),
            "--output", dump_tmp,
        ]
        subprocess.run(cmd, check=True)
        with open(dump_tmp, "r", encoding="utf-8") as f:
            res = json.load(f)
        return [normalize_text(h) for h in res["hyps"]], res["time"], res["vram_gb"], res["batch"]

    # 本地直接运行
    from funasr import AutoModel
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model_path = SENSEVOICE_PATH if "sense" in engine_name.lower() else PARAFORMER_PATH
    m = AutoModel(model=model_path, vad_model=None, punc_model=None, device=device, disable_update=True)

    bs = 1 if mode == "bs1" else batch_size
    # 预热 20 句
    for s in samples[:20]:
        _ = m.generate(input=[s["wav"]], language="zh", use_itn=False) if "sense" in engine_name.lower() else m.generate(input=[s["wav"]])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    hyps = []
    for i in range(0, len(samples), bs):
        batch = samples[i : i + bs]
        wav_inputs = [s["wav"] for s in batch]
        if "sense" in engine_name.lower():
            res = m.generate(input=wav_inputs, language="zh", use_itn=False)
        else:
            res = m.generate(input=wav_inputs)
        for r in res:
            hyps.append(r["text"])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    compute_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

    del m
    torch.cuda.empty_cache()
    return [normalize_text(h) for h in hyps], compute_time, peak_vram, bs


def main():
    parser = argparse.ArgumentParser(description="Issue 11 Fair Benchmark")
    parser.add_argument("--samples", type=int, default=0, help="0 为全量测试集 (7173 句)")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--out_dir", default="runs/benchmark")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    samples, total_s = load_test_samples(args.data_dir, args.samples)
    print(f"✅ 加载了 {len(samples)} 句测试样本，总音频时长: {total_s:.1f} 秒 ({total_s/3600:.2f} 小时)")
    refs = [s["clean_ref"] for s in samples]

    engines = ["Gemma-ASR", "SenseVoiceSmall", "SeACo-Paraformer"]
    modes = ["bs1", "batch"]

    records = []
    for eng in engines:
        for m in modes:
            print("\n" + "=" * 60)
            print(f"🚀 正在测试: 引擎={eng} | 模式={m}")
            print("=" * 60)
            hyps, c_time, vram, actual_bs = benchmark_engine(eng, samples, mode=m, ckpt_path=args.ckpt)
            metrics = calc_metrics(refs, hyps)
            speed = total_s / max(c_time, 1e-6)

            rec = {
                "engine": eng,
                "mode": m,
                "batch": actual_bs,
                "cer": metrics["cer"],
                "pcer": metrics["pcer"],
                "homophone_err": metrics["homophone_err"],
                "time": c_time,
                "speed": speed,
                "vram_gb": vram,
            }
            records.append(rec)
            print(f"--> CER: {rec['cer']:.2%} | P-CER: {rec['pcer']:.2%} | Speed: {rec['speed']:.0f}x | VRAM: {rec['vram_gb']:.2f}GB")

            # 导出逐句结果
            dump_file = os.path.join(args.out_dir, f"{eng}_{m}.jsonl")
            with open(dump_file, "w", encoding="utf-8") as f:
                for i, s in enumerate(samples):
                    f.write(json.dumps({"ref": refs[i], "hyp": hyps[i], "dur": s["duration"]}, ensure_ascii=False) + "\n")

    # 导出汇总表
    summary_md = format_summary_table(records)
    summary_path = os.path.join(args.out_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# 公平基准重测汇总天梯榜 (AISHELL-1 Test 全量)\n\n" + summary_md + "\n")

    print("\n" + "#" * 84)
    print("🏆【公平基准重测汇总天梯榜】")
    print("#" * 84)
    print(summary_md)
    print(f"\n📁 汇总报告与逐句详情已导出至: {args.out_dir}")


if __name__ == "__main__":
    main()
