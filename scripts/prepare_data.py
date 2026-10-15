#!/usr/bin/env python
"""Download Vietnamese speech datasets from the HF Hub and convert them to uniform training shards.

Output layout:
    <out_dir>/<split>/<source>/<shard>.parquet   columns: audio (FLAC 16k mono), text, duration, source
    <out_dir>/manifest.json                      hours / utterances per split and source

Every input parquet file becomes one output shard, so the conversion is parallel and resumable
(finished shards are skipped on re-run).

Example:
    python scripts/prepare_data.py --out_dir data/prepared --raw_dir data/raw --workers 96
"""

try:
    import sentencepiece  # noqa: F401
except ImportError:
    pass

import argparse
import fnmatch
import glob
import io
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import soxr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gemma_asr.text import normalize_text  # noqa: E402

# split name -> glob patterns inside the HF repo
SOURCES = {
    # ~500h, podcasts/books/travel..., all three regional accents. Gated (auto-approve): accept terms on HF.
    "bud500": dict(
        repo="linhtran92/viet_bud500", text="transcription",
        splits={"train": ["data/train-*"], "valid": ["data/validation-*"], "test": ["data/test-*"]},
    ),
    # ~1000h of YouTube speech (made for TTS, punctuated text). Gated (manual approval, can take days).
    "vivoice": dict(repo="capleaf/viVoice", text="text", splits={"train": ["data/train-*"]}),
    # VLSP 2020 (VinAI) ~100h
    "vlsp2020": dict(repo="doof-ferb/vlsp2020_vinai_100h", text="transcription", splits={"train": ["data/train-*"]}),
    # FPT Open Speech Data ~30h (mp3)
    "fpt_fosd": dict(repo="doof-ferb/fpt_fosd", text="transcription", splits={"train": ["data/train-*"]}),
    # InfoRe 1 (~25h single speaker) and InfoRe 2 audiobooks (~415h)
    "infore1": dict(repo="doof-ferb/infore1_25hours", text="transcription", splits={"train": ["data/train-*"]}),
    "infore2": dict(repo="doof-ferb/infore2_audiobooks", text="transcription", splits={"train": ["data/train-*"]}),
    # LSVSC ~100h, labelled dialect/topic
    "lsvsc": dict(
        repo="doof-ferb/LSVSC", text="transcription",
        splits={"train": ["data/train-*"], "valid": ["data/validation-*"], "test": ["data/test-*"]},
    ),
    # ViMD: 63 provincial dialects
    "vimd": dict(
        repo="nguyendv02/ViMD_Dataset", text="text",
        splits={"train": ["data/train-*"], "valid": ["data/valid-*"], "test": ["data/test-*"]},
    ),
    # Common Voice 17 (Vietnamese)
    "cv17": dict(
        repo="fixie-ai/common_voice_17_0", text="sentence",
        splits={"train": ["vi/train/*/*.parquet"], "valid": ["vi/validation/*/*.parquet"],
                "test": ["vi/test/*/*.parquet"]},
    ),
    # FLEURS vi_vn (read Wikipedia sentences)
    "fleurs": dict(
        repo="google/fleurs", text="raw_transcription",
        splits={"train": ["parquet-data/vi_vn/train-*"], "valid": ["parquet-data/vi_vn/validation-*"],
                "test": ["parquet-data/vi_vn/test-*"]},
    ),
    # GigaSpeech 2 Vietnamese dev/test (YouTube, hard). Eval only.
    "gigaspeech2": dict(
        repo="doof-ferb/gigaspeech2_vie", text="transcription",
        splits={"valid": ["data/validation-*"], "test": ["data/test-*"]},
    ),
    # AISHELL-1 via Hugging Face parquet shards (~178h)
    "aishell1": dict(
        repo="SpeechColab/aishell", text="transcription", default=False,
        splits={
            "train": ["data/train-*.parquet", "train-*.parquet", "*train-*.parquet"],
            "valid": ["data/validation-*.parquet", "validation-*.parquet", "*validation-*.parquet"],
            "test": ["data/test-*.parquet", "test-*.parquet", "*test-*.parquet"],
        },
    ),
}
DEFAULT_SOURCES = [k for k, v in SOURCES.items() if v.get("default", True)]
SR = 16000
ROW_GROUP = 512


def download(source: str, splits: list[str], raw_dir: str, max_workers: int) -> None:
    from huggingface_hub import snapshot_download

    cfg = SOURCES[source]
    patterns = [p for s in splits for p in cfg["splits"].get(s, [])]
    if not patterns:
        return
    print(f"[download] {source}: {cfg['repo']} {patterns}", flush=True)
    snapshot_download(
        repo_id=cfg["repo"], repo_type="dataset", allow_patterns=patterns,
        local_dir=os.path.join(raw_dir, source), max_workers=max_workers,
    )


def local_files(source: str, split: str, raw_dir: str) -> list[str]:
    root = os.path.join(raw_dir, source)
    if not os.path.exists(root) and os.path.exists(raw_dir):
        root = raw_dir
    files = []
    for pat in SOURCES[source]["splits"].get(split, []):
        files += [f for f in glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)
                  if fnmatch.fnmatch(os.path.relpath(f, root).replace("\\", "/"), pat)
                  or fnmatch.fnmatch(os.path.basename(f), pat)]
    return sorted(set(files))


def to_mono16k(audio_source: bytes | str | os.PathLike) -> np.ndarray:
    if isinstance(audio_source, (str, os.PathLike)):
        x, sr = sf.read(str(audio_source), dtype="float32", always_2d=True)
    else:
        x, sr = sf.read(io.BytesIO(audio_source), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = soxr.resample(x, sr, SR, quality="HQ")
    return x


def encode_flac(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1.0, 1.0), SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


SCHEMA = pa.schema([("audio", pa.binary()), ("text", pa.string()), ("duration", pa.float32()),
                    ("source", pa.string())])


def process_file(job: tuple) -> dict:
    source, split, in_path, out_path, text_col, min_dur, max_dur, delete_raw = job
    stats_path = out_path + ".stats.json"
    if os.path.exists(out_path) and os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)
    st = Counter()
    hours = 0.0
    tmp = out_path + ".tmp"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = pq.ParquetWriter(tmp, SCHEMA, compression="none")  # FLAC is already compressed
    rows = {k: [] for k in SCHEMA.names}

    def flush():
        if rows["audio"]:
            writer.write_table(pa.table(rows, schema=SCHEMA), row_group_size=ROW_GROUP)
            for v in rows.values():
                v.clear()

    pf = pq.ParquetFile(in_path)
    for batch in pf.iter_batches(batch_size=64, columns=["audio", text_col]):
        for audio, text in zip(batch.column("audio").to_pylist(), batch.column(text_col).to_pylist()):
            st["in"] += 1
            norm = normalize_text(text or "")
            if norm is None:
                st["skip_text"] += 1
                continue
            b = audio.get("bytes") if isinstance(audio, dict) else None
            if not b:
                st["skip_no_audio"] += 1
                continue
            try:
                x = to_mono16k(b)
            except Exception:
                st["skip_decode"] += 1
                continue
            dur = len(x) / SR
            if (min_dur is not None and dur < min_dur) or (max_dur is not None and dur > max_dur):
                st["skip_duration"] += 1
                continue
            rows["audio"].append(encode_flac(x))
            rows["text"].append(norm)
            rows["duration"].append(dur)
            rows["source"].append(source)
            st["out"] += 1
            hours += dur / 3600
            if len(rows["audio"]) >= ROW_GROUP:
                flush()
    flush()
    writer.close()
    os.replace(tmp, out_path)
    result = {"source": source, "split": split, "file": out_path, "hours": hours, **st}
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    if delete_raw:
        os.remove(in_path)
    return result


def process_aishell_shard(job: tuple) -> dict:
    source, split, out_path, items, min_dur, max_dur = job
    stats_path = out_path + ".stats.json"
    if os.path.exists(out_path) and os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)

    st = Counter()
    hours = 0.0
    tmp = out_path + ".tmp"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = pq.ParquetWriter(tmp, SCHEMA, compression="none")  # FLAC is already compressed
    rows = {k: [] for k in SCHEMA.names}

    def flush():
        if rows["audio"]:
            writer.write_table(pa.table(rows, schema=SCHEMA), row_group_size=ROW_GROUP)
            for v in rows.values():
                v.clear()

    for wav_path, utt_id, text in items:
        st["in"] += 1
        if text is None:
            st["skip_no_transcript"] += 1
            continue
        norm = normalize_text(text)
        if norm is None:
            st["skip_text"] += 1
            continue
        try:
            x = to_mono16k(wav_path)
        except Exception:
            st["skip_decode"] += 1
            continue
        dur = len(x) / SR
        if dur < min_dur or dur > max_dur:
            st["skip_duration"] += 1
            continue
        rows["audio"].append(encode_flac(x))
        rows["text"].append(norm)
        rows["duration"].append(np.float32(dur))
        rows["source"].append(source)
        st["out"] += 1
        hours += dur / 3600.0
        if len(rows["audio"]) >= ROW_GROUP:
            flush()

    flush()
    writer.close()
    os.replace(tmp, out_path)
    result = {"source": source, "split": split, "file": out_path, "hours": hours, **st}
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    return result


def prepare_aishell(
    aishell_dir: str | Path,
    out_dir: str | Path,
    min_dur: float = 0.5,
    max_dur: float = 15.0,
    workers: int = 1,
    max_files_per_split: int = 0,
    max_speakers_per_split: int = 0,
    shard_size: int = 2000,
) -> dict:
    """Convert uncompressed AISHELL-1 dataset into uniform Parquet training shards.

    Split mapping:
        train -> train
        dev   -> valid
        test  -> test

    Output layout:
        <out_dir>/<split>/aishell1/shard_XXXXX.parquet
        <out_dir>/<split>/aishell1/shard_XXXXX.parquet.stats.json
        <out_dir>/manifest.json
    """
    aishell_path = Path(aishell_dir).resolve()
    out_path = Path(out_dir).resolve()
    os.makedirs(out_path, exist_ok=True)

    if (aishell_path / "data_aishell").is_dir() and not (aishell_path / "wav").is_dir():
        aishell_path = aishell_path / "data_aishell"

    wav_root = aishell_path / "wav" if (aishell_path / "wav").is_dir() else aishell_path

    # Check for unextracted .tar.gz archives
    tar_files = list(wav_root.rglob("*.tar.gz"))
    if tar_files:
        print(f"[aishell] Notice: found {len(tar_files)} unextracted .tar.gz archives in {wav_root}. "
              "Please make sure speaker archives are extracted before processing.", flush=True)

    # Locate transcript file
    transcript_file = None
    candidate_transcripts = [
        aishell_path / "transcript" / "aishell_transcript_v0.8.txt",
        aishell_path / "aishell_transcript_v0.8.txt",
    ]
    for cand in candidate_transcripts:
        if cand.is_file():
            transcript_file = cand
            break
    if transcript_file is None:
        globbed = list(aishell_path.rglob("aishell_transcript*.txt"))
        if globbed:
            transcript_file = globbed[0]
        else:
            globbed_any = list(aishell_path.rglob("*transcript*.txt"))
            if globbed_any:
                transcript_file = globbed_any[0]

    transcripts = {}
    if transcript_file and transcript_file.is_file():
        print(f"[aishell] Loading transcript from {transcript_file}", flush=True)
        with open(transcript_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                utt_id = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                transcripts[utt_id] = text
    else:
        print(f"[aishell] Warning: No transcript file found under {aishell_path}!", flush=True)

    split_targets = [("train", "train"), ("dev", "valid"), ("test", "test")]
    jobs = []

    for raw_split, out_split in split_targets:
        split_dir = wav_root / raw_split
        if not split_dir.is_dir() and raw_split == "dev" and (wav_root / "valid").is_dir():
            split_dir = wav_root / "valid"

        if not split_dir.is_dir():
            continue

        spk_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        if max_speakers_per_split > 0 and spk_dirs:
            spk_dirs = spk_dirs[:max_speakers_per_split]

        wav_files = []
        if spk_dirs:
            for spk in spk_dirs:
                wav_files.extend(sorted(spk.glob("*.wav")))
                if max_files_per_split > 0 and len(wav_files) >= max_files_per_split:
                    break
        else:
            wav_files = sorted(split_dir.glob("*.wav"))

        if max_files_per_split > 0:
            wav_files = wav_files[:max_files_per_split]

        if not wav_files:
            continue

        # Chunk into shards
        for shard_idx, i in enumerate(range(0, len(wav_files), shard_size)):
            chunk = wav_files[i : i + shard_size]
            items = [(str(w), w.stem, transcripts.get(w.stem)) for w in chunk]
            shard_name = f"shard_{shard_idx:05d}.parquet"
            out_file = str(out_path / out_split / "aishell1" / shard_name)
            jobs.append(("aishell1", out_split, out_file, items, min_dur, max_dur))

    print(f"[aishell] Processing {len(jobs)} shard jobs across splits with {workers} worker(s)...", flush=True)

    results = []
    if workers <= 1:
        for job in jobs:
            results.append(process_aishell_shard(job))
    else:
        with Pool(workers) as pool:
            for r in pool.imap_unordered(process_aishell_shard, jobs):
                results.append(r)

    manifest = {}
    for _, out_split in split_targets:
        manifest.setdefault(out_split, {}).setdefault(
            "aishell1",
            Counter({"hours": 0.0, "in": 0, "out": 0, "skip_text": 0, "skip_duration": 0, "skip_decode": 0}),
        )

    for r in results:
        m = manifest.setdefault(r["split"], {}).setdefault(r["source"], Counter())
        for k, v in r.items():
            if isinstance(v, (int, float)):
                m[k] += v

    manifest_file = out_path / "manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                old = json.load(f)
            for split, d in old.items():
                for src, v in d.items():
                    manifest.setdefault(split, {}).setdefault(src, v)
        except Exception:
            pass

    clean_manifest = {}
    for split, d in manifest.items():
        clean_manifest[split] = {}
        for src, v in d.items():
            clean_manifest[split][src] = dict(v) if isinstance(v, Counter) else v

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(clean_manifest, f, indent=1)

    print("\nsplit        source         hours     utts   dropped(text/dur/decode)")
    for split, d in sorted(clean_manifest.items()):
        for src, v in sorted(d.items()):
            print(f"{split:12s} {src:12s} {v['hours']:8.4f} {v.get('out', 0):8d}   "
                  f"{v.get('skip_text', 0)}/{v.get('skip_duration', 0)}/{v.get('skip_decode', 0)}")
        print(f"{split:12s} {'TOTAL':12s} {sum(v['hours'] for v in d.values()):8.4f}")

    return clean_manifest


prepare_aishell1 = prepare_aishell


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aishell_dir", default=None, help="path to local uncompressed AISHELL-1 dataset directory")
    ap.add_argument("--out_dir", default="data/prepared")
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES, choices=list(SOURCES))
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--download_workers", type=int, default=16)
    ap.add_argument("--min_dur", type=float, default=0.5)
    ap.add_argument("--max_dur", type=float, default=None)
    ap.add_argument("--max_files_per_source", type=int, default=0, help="debug: only convert N files per split")
    ap.add_argument("--max_files_per_split", type=int, default=0, help="debug: only convert N files per split")
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--delete_raw", action="store_true", help="delete each raw parquet after converting it")
    args = ap.parse_args()

    if args.aishell_dir:
        max_dur = args.max_dur if args.max_dur is not None else 15.0
        max_files = args.max_files_per_split or args.max_files_per_source or 0
        prepare_aishell(
            aishell_dir=args.aishell_dir,
            out_dir=args.out_dir,
            min_dur=args.min_dur,
            max_dur=max_dur,
            workers=args.workers,
            max_files_per_split=max_files,
        )
        return

    max_dur = args.max_dur if args.max_dur is not None else 30.0

    if not args.skip_download:
        for s in args.sources:
            try:
                download(s, args.splits, args.raw_dir, args.download_workers)
            except Exception as e:  # gated / no access: keep going with the rest
                print(f"[download] !! {s} failed: {type(e).__name__}: {str(e)[:300]}", flush=True)

    jobs = []
    for s in args.sources:
        cfg = SOURCES[s]
        for split in args.splits:
            files = local_files(s, split, args.raw_dir)
            max_files = args.max_files_per_split or args.max_files_per_source or 0
            if max_files:
                files = files[: max_files]
            for f in files:
                name = os.path.splitext(os.path.basename(f))[0]
                out = os.path.join(args.out_dir, split, s, f"{name}.parquet")
                jobs.append((s, split, f, out, cfg["text"], args.min_dur, max_dur, args.delete_raw))
    # big files first for better load balancing
    jobs.sort(key=lambda j: -os.path.getsize(j[2]) if os.path.exists(j[2]) else 0)
    print(f"[convert] {len(jobs)} files with {args.workers} workers", flush=True)

    results, t0 = [], time.time()
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process_file, jobs), 1):
            results.append(r)
            if i % 10 == 0 or i == len(jobs):
                h = sum(x["hours"] for x in results)
                print(f"[convert] {i}/{len(jobs)} files, {h:.1f}h kept, {time.time() - t0:.0f}s", flush=True)

    manifest = {}
    for r in results:
        m = manifest.setdefault(r["split"], {}).setdefault(r["source"], Counter())
        for k, v in r.items():
            if isinstance(v, (int, float)):
                m[k] += v
    # merge with an existing manifest (e.g. sources prepared in an earlier run)
    path = os.path.join(args.out_dir, "manifest.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            old = json.load(f)
        for split, d in old.items():
            for src, v in d.items():
                manifest.setdefault(split, {}).setdefault(src, v)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print("\nsplit        source         hours     utts   dropped(text/dur/decode)")
    for split, d in sorted(manifest.items()):
        for src, v in sorted(d.items()):
            print(f"{split:12s} {src:12s} {v['hours']:8.1f} {v.get('out', 0):8d}   "
                  f"{v.get('skip_text', 0)}/{v.get('skip_duration', 0)}/{v.get('skip_decode', 0)}")
        print(f"{split:12s} {'TOTAL':12s} {sum(v['hours'] for v in d.values()):8.1f}")


if __name__ == "__main__":
    main()
