# -*- coding: utf-8 -*-
"""Unit tests for AISHELL-1 local data preparation and sharding (Ticket 05).

Covers:
- Standard Parquet schema (audio, text, duration, source) and FLAC 16k mono PCM_16 encoding
- Split mapping: train -> train, dev -> valid, test -> test
- Strict text normalization (removing word spaces, filtering numbers / non-CJK)
- Audio duration filtering (>15s rejected, <0.5s rejected, normal 1.0s kept)
- Resumability and breakpoint skipping (existing shard + .stats.json skipped on re-run)
- Manifest.json generation with retained hours and drop counters
- Debug limit parameter (max_files_per_split)
- CLI invocation via scripts/prepare_data.py --aishell_dir ... --out_dir ...
"""

# Windows CRT conflict prevention: import sentencepiece before pyarrow
try:
    import sentencepiece  # noqa: F401
except ImportError:
    pass

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_prepare_function():
    """Retrieve the public seam for AISHELL-1 data preparation from scripts.prepare_data."""
    try:
        from scripts.prepare_data import prepare_aishell
        return prepare_aishell
    except ImportError:
        try:
            from scripts.prepare_data import prepare_aishell1
            return prepare_aishell1
        except ImportError:
            return None


def generate_sine_wave(duration_s: float, sample_rate: int = 16000, freq: float = 440.0) -> np.ndarray:
    """Generate a pure sine wave numpy array for synthetic test audio."""
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    return audio.astype(np.float32)


@pytest.fixture
def mini_aishell_dataset(tmp_path):
    """Constructs a deterministic minimal AISHELL-1 dataset directory.

    Structure:
        data_aishell/
            wav/
                train/S0001/
                    BAC009S0001W0121.wav  (1.0s, valid)
                    BAC009S0001W0122.wav  (1.0s, valid)
                    BAC009S0001W0123.wav  (20.0s, duration > 15s, should be dropped)
                    BAC009S0001W0124.wav  (1.0s, transcript has numbers, should be dropped)
                dev/S0724/
                    BAC009S0724W0121.wav  (1.0s, valid)
                    BAC009S0724W0122.wav  (1.0s, valid)
                    BAC009S0724W0123.wav  (20.0s, duration > 15s, should be dropped)
                    BAC009S0724W0124.wav  (1.0s, transcript has numbers, should be dropped)
                test/S0764/
                    BAC009S0764W0121.wav  (1.0s, valid)
                    BAC009S0764W0122.wav  (1.0s, valid)
                    BAC009S0764W0123.wav  (20.0s, duration > 15s, should be dropped)
                    BAC009S0764W0124.wav  (1.0s, transcript has numbers, should be dropped)
            transcript/
                aishell_transcript_v0.8.txt
    """
    aishell_dir = tmp_path / "data_aishell"
    wav_root = aishell_dir / "wav"
    transcript_dir = aishell_dir / "transcript"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    splits_config = {
        "train": {
            "spk": "S0001",
            "utts": [
                ("BAC009S0001W0121", 1.0, "甚至 出现 交易 几乎 停滞 的 情况"),
                ("BAC009S0001W0122", 1.0, "标杆 房企 必然 调整 市场 战略"),
                ("BAC009S0001W0123", 20.0, "广州 拟 出台 住房 租赁 意见 实施 细则 超过 十五 秒 长度"),
                ("BAC009S0001W0124", 1.0, "昨天 降温 ２０２６ 年 １０ 月 １５ 日"),
            ],
        },
        "dev": {
            "spk": "S0724",
            "utts": [
                ("BAC009S0724W0121", 1.0, "推进 供给侧 结构性 改革 取得 突破"),
                ("BAC009S0724W0122", 1.0, "积极 稳妥 降低 实体 经济 杠杆率"),
                ("BAC009S0724W0123", 20.0, "建设 现代 经济 体系 超长 音频 样本 录制"),
                ("BAC009S0724W0124", 1.0, "投资 预算 增加 ５０ 亿元"),
            ],
        },
        "test": {
            "spk": "S0764",
            "utts": [
                ("BAC009S0764W0121", 1.0, "发展 普惠 金融 助力 乡村 振兴"),
                ("BAC009S0764W0122", 1.0, "坚持 绿色 低碳 循环 发展 理念"),
                ("BAC009S0764W0123", 20.0, "加快 建设 现代化 基础设施 体系 样本 超长"),
                ("BAC009S0764W0124", 1.0, "产值 预计 增长 ８ 个 百分点"),
            ],
        },
    }

    transcript_lines = []
    for split, conf in splits_config.items():
        spk_dir = wav_root / split / conf["spk"]
        spk_dir.mkdir(parents=True, exist_ok=True)
        for utt_id, dur_s, text in conf["utts"]:
            wav_path = spk_dir / f"{utt_id}.wav"
            audio_data = generate_sine_wave(dur_s, sample_rate=16000)
            sf.write(str(wav_path), audio_data, 16000, format="WAV", subtype="PCM_16")
            transcript_lines.append(f"{utt_id} {text}")

    transcript_file = transcript_dir / "aishell_transcript_v0.8.txt"
    transcript_file.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")

    return aishell_dir


class TestPrepareAishellData:
    """Test suite for AISHELL-1 local data conversion and sharding pipeline."""

    def test_output_schema_and_flac_format(self, mini_aishell_dataset, tmp_path):
        """Verify generated Parquet files match the project schema and audio is FLAC 16k mono."""
        prepare_fn = get_prepare_function()
        if prepare_fn is None:
            raise NotImplementedError(
                "Public seam 'prepare_aishell' (or 'prepare_aishell1') is not implemented in scripts/prepare_data.py"
            )

        out_dir = tmp_path / "prepared"
        prepare_fn(aishell_dir=mini_aishell_dataset, out_dir=out_dir, workers=1)

        train_shard_dir = out_dir / "train" / "aishell1"
        assert train_shard_dir.exists(), f"Expected directory {train_shard_dir} does not exist"

        parquet_files = sorted(train_shard_dir.glob("*.parquet"))
        assert len(parquet_files) >= 1, f"No parquet shards found in {train_shard_dir}"

        table = pq.read_table(parquet_files[0])
        # Public schema verification
        assert table.column_names == ["audio", "text", "duration", "source"]
        assert table.schema.field("audio").type == pa.binary()
        assert table.schema.field("text").type == pa.string()
        assert table.schema.field("duration").type == pa.float32()
        assert table.schema.field("source").type == pa.string()

        # All rows must belong to source "aishell1"
        assert set(table.column("source").to_pylist()) == {"aishell1"}

        # Inspect FLAC audio bytes
        audio_bytes = table.column("audio")[0].as_py()
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0

        info = sf.info(io.BytesIO(audio_bytes))
        assert info.format == "FLAC"
        assert info.subtype == "PCM_16"
        assert info.samplerate == 16000
        assert info.channels == 1

        wav, sr = sf.read(io.BytesIO(audio_bytes))
        assert sr == 16000
        assert wav.ndim == 1
        recorded_dur = table.column("duration")[0].as_py()
        assert len(wav) / 16000 == pytest.approx(recorded_dur, abs=0.01)
        assert recorded_dur == pytest.approx(1.0, abs=0.05)

    def test_text_normalization_and_filtering(self, mini_aishell_dataset, tmp_path):
        """Verify strict normalization (spaces stripped) and rejection of numbers / long audio."""
        prepare_fn = get_prepare_function()
        if prepare_fn is None:
            raise NotImplementedError(
                "Public seam 'prepare_aishell' (or 'prepare_aishell1') is not implemented in scripts/prepare_data.py"
            )

        out_dir = tmp_path / "prepared"
        prepare_fn(aishell_dir=mini_aishell_dataset, out_dir=out_dir, min_dur=0.5, max_dur=15.0, workers=1)

        # Check train split texts against independent ground-truth literals
        train_shards = sorted((out_dir / "train" / "aishell1").glob("*.parquet"))
        assert len(train_shards) >= 1

        texts = []
        for f in train_shards:
            texts.extend(pq.read_table(f, columns=["text"]).column("text").to_pylist())

        # Exact expected normalized texts:
        # 1. "甚至 出现 交易 几乎 停滞 的 情况" -> "甚至出现交易几乎停滞的情况" (spaces removed)
        # 2. "标杆 房企 必然 调整 市场 战略" -> "标杆房企必然调整市场战略"
        # 3. 20.0s audio dropped by duration > 15.0s
        # 4. Text with digits "２０２６" dropped by normalize_text returning None
        expected_texts = ["甚至出现交易几乎停滞的情况", "标杆房企必然调整市场战略"]
        assert sorted(texts) == sorted(expected_texts)

    def test_split_mapping_and_manifest(self, mini_aishell_dataset, tmp_path):
        """Verify split mapping (train->train, dev->valid, test->test) and manifest.json output."""
        prepare_fn = get_prepare_function()
        if prepare_fn is None:
            raise NotImplementedError(
                "Public seam 'prepare_aishell' (or 'prepare_aishell1') is not implemented in scripts/prepare_data.py"
            )

        out_dir = tmp_path / "prepared"
        prepare_fn(aishell_dir=mini_aishell_dataset, out_dir=out_dir, min_dur=0.5, max_dur=15.0, workers=1)

        # Directory structure checks: dev MUST be mapped to valid
        assert (out_dir / "train" / "aishell1").exists()
        assert (out_dir / "valid" / "aishell1").exists(), "dev split must be mapped to 'valid'"
        assert (out_dir / "test" / "aishell1").exists()
        assert not (out_dir / "dev").exists(), "raw 'dev' directory should not exist in prepared output"

        manifest_file = out_dir / "manifest.json"
        assert manifest_file.exists(), f"manifest.json missing at {manifest_file}"

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert "train" in manifest
        assert "valid" in manifest
        assert "test" in manifest
        assert "dev" not in manifest

        # Check each split's independent deterministic counters
        for split_name in ["train", "valid", "test"]:
            assert "aishell1" in manifest[split_name]
            st = manifest[split_name]["aishell1"]
            assert st["out"] == 2, f"Expected 2 kept utterances in {split_name}, got {st.get('out')}"
            assert st["skip_duration"] == 1, f"Expected 1 dropped for duration in {split_name}"
            assert st["skip_text"] == 1, f"Expected 1 dropped for text normalization in {split_name}"
            assert st["in"] == 4, f"Expected 4 input utterances in {split_name}"
            # 2 utterances * 1.0s = 2.0s / 3600 = ~0.0005556 hours
            assert st["hours"] == pytest.approx(2.0 / 3600.0, rel=1e-2)

    def test_resumability_and_shard_skipping(self, mini_aishell_dataset, tmp_path):
        """Verify breakpoint resumability: existing shards with .stats.json are not re-processed."""
        prepare_fn = get_prepare_function()
        if prepare_fn is None:
            raise NotImplementedError(
                "Public seam 'prepare_aishell' (or 'prepare_aishell1') is not implemented in scripts/prepare_data.py"
            )

        out_dir = tmp_path / "prepared"
        # First execution: generate shards
        prepare_fn(aishell_dir=mini_aishell_dataset, out_dir=out_dir, workers=1)

        parquet_files = list(out_dir.glob("*/*/*.parquet"))
        assert len(parquet_files) >= 3  # at least one for train, valid, test

        # Check stats files exist next to each shard
        for p in parquet_files:
            stats_p = Path(str(p) + ".stats.json")
            assert stats_p.exists(), f"Expected companion stats file {stats_p} does not exist"

        # Record file mtimes
        mtimes_before = {str(p): p.stat().st_mtime_ns for p in parquet_files}

        # Second execution: should skip existing shards without rewriting them
        prepare_fn(aishell_dir=mini_aishell_dataset, out_dir=out_dir, workers=1)

        mtimes_after = {str(p): p.stat().st_mtime_ns for p in parquet_files}
        assert mtimes_before == mtimes_after, "Existing shards were rewritten instead of being skipped"

    def test_debug_limit_max_files(self, mini_aishell_dataset, tmp_path):
        """Verify the debug parameter (max_files_per_split) limits the processed files."""
        prepare_fn = get_prepare_function()
        if prepare_fn is None:
            raise NotImplementedError(
                "Public seam 'prepare_aishell' (or 'prepare_aishell1') is not implemented in scripts/prepare_data.py"
            )

        out_dir = tmp_path / "prepared_debug"
        # Only process at most 1 file per split
        prepare_fn(aishell_dir=mini_aishell_dataset, out_dir=out_dir, max_files_per_split=1, workers=1)

        manifest_file = out_dir / "manifest.json"
        assert manifest_file.exists()

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for split_name in ["train", "valid", "test"]:
            st = manifest[split_name]["aishell1"]
            assert st["in"] <= 1, f"Expected at most 1 input processed with debug limit, got {st['in']}"

    def test_cli_execution(self, mini_aishell_dataset, tmp_path):
        """Verify command-line interface execution (scripts/prepare_data.py --aishell_dir ...)."""
        out_dir = tmp_path / "prepared_cli"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_data.py"),
            "--aishell_dir",
            str(mini_aishell_dataset),
            "--out_dir",
            str(out_dir),
            "--workers",
            "1",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        assert result.returncode == 0, (
            f"CLI execution failed with returncode {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        manifest_file = out_dir / "manifest.json"
        assert manifest_file.exists(), f"CLI execution did not produce manifest at {manifest_file}"
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert "valid" in manifest
        assert manifest["valid"]["aishell1"]["out"] == 2
