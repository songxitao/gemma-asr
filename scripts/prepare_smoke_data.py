# -*- coding: utf-8 -*-
"""Prepare smoke training and validation shards from the available AISHELL Parquet slice.

Reads: data/raw/aishell_hf/data/test-00000-of-00003-25094929c3ce9a28.parquet (2392 utts)
Writes:
    data/prepared/train/aishell1/shard_0000.parquet (2000 utts)
    data/prepared/valid/aishell1/shard_0000.parquet (391 utts)
    and accompanying .stats.json and manifest.json
"""

import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import soxr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gemma_asr.text import normalize_text

SR = 16000
SCHEMA = pa.schema([
    ("audio", pa.binary()),
    ("text", pa.string()),
    ("duration", pa.float32()),
    ("source", pa.string()),
])


def to_mono16k(audio_bytes: bytes) -> tuple[bytes, float]:
    x, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = soxr.resample(x, sr, SR, quality="HQ")
    dur = float(len(x)) / SR
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue(), dur


def main():
    src_path = PROJECT_ROOT / "data" / "raw" / "aishell_hf" / "data" / "test-00000-of-00003-25094929c3ce9a28.parquet"
    if not src_path.exists():
        print(f"Error: source file not found at {src_path}")
        sys.exit(1)

    out_base = PROJECT_ROOT / "data" / "prepared"
    train_dir = out_base / "train" / "aishell1"
    valid_dir = out_base / "valid" / "aishell1"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading input from {src_path}...")
    pf = pq.ParquetFile(src_path)

    items = []
    skipped_text = 0
    for batch in pf.iter_batches(batch_size=256, columns=["audio", "transcription"]):
        audios = batch.column("audio").to_pylist()
        texts = batch.column("transcription").to_pylist()
        for a, t in zip(audios, texts):
            norm = normalize_text(t or "")
            if norm is None:
                skipped_text += 1
                continue
            flac_bytes, dur = to_mono16k(a["bytes"])
            if dur < 0.5 or dur > 15.0:
                continue
            items.append({
                "audio": flac_bytes,
                "text": norm,
                "duration": float(dur),
                "source": "aishell1",
            })

    print(f"Total valid items: {len(items)}, skipped text: {skipped_text}")

    train_items = items[:2000]
    valid_items = items[2000:]

    splits = [
        ("train", train_items, train_dir / "shard_0000.parquet"),
        ("valid", valid_items, valid_dir / "shard_0000.parquet"),
    ]

    manifest = {}

    for split_name, split_data, shard_path in splits:
        print(f"Writing {split_name} shard ({len(split_data)} items) to {shard_path}...")
        col_audio = [x["audio"] for x in split_data]
        col_text = [x["text"] for x in split_data]
        col_dur = [x["duration"] for x in split_data]
        col_src = [x["source"] for x in split_data]

        table = pa.table(
            [col_audio, col_text, col_dur, col_src],
            schema=SCHEMA,
        )
        pq.write_table(table, shard_path, compression="none", row_group_size=512)

        total_hours = sum(col_dur) / 3600.0
        stats = {
            "in": len(split_data),
            "out": len(split_data),
            "hours": round(total_hours, 4),
            "skip_text": 0,
            "skip_duration": 0,
            "skip_decode": 0,
        }
        stats_path = str(shard_path) + ".stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        manifest.setdefault(split_name, {})["aishell1"] = stats
        print(f"  {split_name}: {len(split_data)} utts, {total_hours:.2f} hours")

    manifest_path = out_base / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
