# -*- coding: utf-8 -*-
"""
在 asr_ui_env 环境下独立运行 SenseVoice 或 Paraformer 推理的 Worker 脚本。
用于 benchmark.py 进行隔离进程调用，避免模型间显存与上下文污染。
"""
import os
import sys
import io
import time
import json
import argparse
import soundfile as sf
import pyarrow.parquet as pq
import torch
from funasr import AutoModel

SENSEVOICE_PATH = os.getenv("SENSEVOICE_PATH", "models/SenseVoiceSmall")
PARAFORMER_PATH = os.getenv("PARAFORMER_PATH", "models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
DEFAULT_DATA_DIR = r"data/prepared_full"


def load_test_samples(data_dir: str, num_samples: int):
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
            samples.append({"wav": arr, "text": txt, "duration": dur})
            if num_samples > 0 and len(samples) >= num_samples:
                break
        if num_samples > 0 and len(samples) >= num_samples:
            break
    total_audio_s = sum(s["duration"] for s in samples)
    return samples, total_audio_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--mode", default="batch", choices=["bs1", "batch"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--samples_count", type=int, default=0)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model_path", default=None, help="自定义模型目录")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    samples, total_s = load_test_samples(args.data_dir, args.samples_count)

    is_sense = "sense" in args.engine.lower()
    default_path = SENSEVOICE_PATH if is_sense else PARAFORMER_PATH
    model_path = args.model_path or default_path

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    m = AutoModel(model=model_path, vad_model=None, punc_model=None, device=device, disable_update=True)

    # 预热 20 句
    for s in samples[:20]:
        _ = m.generate(input=[s["wav"]], language="zh", use_itn=False) if is_sense else m.generate(input=[s["wav"]])

    bs = 1 if args.mode == "bs1" else args.batch_size
    hyps = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    for i in range(0, len(samples), bs):
        batch = samples[i : i + bs]
        wav_inputs = [s["wav"] for s in batch]
        if is_sense:
            res = m.generate(input=wav_inputs, language="zh", use_itn=False)
        else:
            res = m.generate(input=wav_inputs)
        for r in res:
            hyps.append(r["text"])

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    compute_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "hyps": hyps,
            "time": compute_time,
            "vram_gb": peak_vram,
            "batch": bs,
        }, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
