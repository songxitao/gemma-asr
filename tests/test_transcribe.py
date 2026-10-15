"""Tests for Chinese speech transcription script and offline decoding contract (Ticket 07).

Tested contracts (Seams):
1. `split_quiet` audio segmentation:
   - Short audio (< chunk_seconds) returns single chunk intact.
   - Long audio (> chunk_seconds) splits near quiet points, chunks <= chunk_seconds.
   - Extremely short trailing fragments (< 0.1s) are dropped.
2. Chinese continuous text stitching:
   - Decoded chunks for Chinese must be seamlessly joined without spaces (no " ".join).
3. CLI argument parsing contract:
   - Missing required args (--ckpt or audio) exits with non-zero code.
   - Default values for chunk_seconds (30.0) and batch_seconds (1200.0).
   - Optional --output / -o file export flag.
4. Offline execution & UTF-8 output:
   - Runs on CPU or single GPU without CUDA hard crashes.
   - Output file (when specified) is written with UTF-8 encoding without GBK mojibake.
   - Never triggers remote HuggingFace hub downloads.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts.transcribe import split_quiet
import scripts.transcribe as transcribe_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_wav_file(tmp_path):
    """Creates a temporary 1-second 16kHz mono WAV file."""
    path = tmp_path / "test_1s.wav"
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
    sf.write(str(path), samples, 16000)
    return str(path)


@pytest.fixture
def dummy_checkpoint(tmp_path):
    """Creates a lightweight dummy checkpoint with Chinese vocab metadata."""
    ckpt_path = tmp_path / "dummy_ckpt.pt"
    payload = {
        "config": {
            "encoder_id": "models/google/embeddinggemma-2",
            "unfreeze_top_k": 0,
        },
        "unit_gemma_ids": [235284, 235285, 235286],  # Dummy token IDs
        "model": {},
    }
    torch.save(payload, str(ckpt_path))
    return str(ckpt_path)


# ---------------------------------------------------------------------------
# 1. Pure function tests: split_quiet audio chunking
# ---------------------------------------------------------------------------

class TestSplitQuiet:
    """Verifies silence-aware audio chunking logic at boundary seams."""

    def test_short_audio_returns_single_chunk(self):
        sr = 16000
        # 5 seconds of audio, chunk_s = 30s
        x = np.random.randn(sr * 5).astype(np.float32)
        chunks = split_quiet(x, chunk_s=30.0)

        assert len(chunks) == 1
        assert len(chunks[0]) == len(x)
        np.testing.assert_array_equal(chunks[0], x)

    def test_long_audio_split_near_quiet_point(self):
        sr = 16000
        # 40 seconds total: loud everywhere except silence between 24s and 26s
        x = np.ones(sr * 40, dtype=np.float32) * 0.8
        x[sr * 24 : sr * 26] = 0.0

        chunks = split_quiet(x, chunk_s=30.0, search_s=8.0)

        # Must split into at least 2 chunks
        assert len(chunks) >= 2

        # Each chunk must respect max chunk_s
        for c in chunks:
            assert len(c) <= int(30.0 * sr)

        # The first cut point should fall inside the quiet window [24s, 26s]
        cut_point = len(chunks[0])
        assert int(23.5 * sr) <= cut_point <= int(26.5 * sr)

    def test_tiny_tail_fragment_discarded(self):
        sr = 16000
        # Audio shorter than 0.1s (SR // 10 = 1600 samples), e.g. 0.05s
        x = np.ones(int(sr * 0.05), dtype=np.float32)
        chunks = split_quiet(x, chunk_s=30.0)

        # Micro fragment (< 0.1s) must be discarded by filter
        assert len(chunks) == 0


# ---------------------------------------------------------------------------
# 2. Chinese continuous text stitching contract (No space insertion)
# ---------------------------------------------------------------------------

class TestChineseTextStitching:
    """Chinese transcription must seamlessly stitch characters without spaces."""

    def test_multi_chunk_transcription_has_no_inserted_spaces(
        self, dummy_wav_file, dummy_checkpoint, monkeypatch, capsys
    ):
        """Simulate decoding multiple chunks of Chinese speech.

        Decoded chunk texts: ['北京科技大学', '坐落于海淀区', '科研实力雄厚']
        Contract: The joined result MUST be '北京科技大学坐落于海淀区科研实力雄厚'.
        Anti-pattern: Vietnamese-style ' '.join producing '北京科技大学 坐落于海淀区 科研实力雄厚'.
        """
        # Mock greedy decoding output from Units
        mock_texts = ["北京科技大学", "坐落于海淀区", "科研实力雄厚"]

        class DummyModel(torch.nn.Module):
            def forward(self, wav, lens):
                return torch.zeros(len(wav), 10, 5), torch.tensor([10] * len(wav))

            def float(self):
                return self

        mock_units = MagicMock()
        mock_units.decode.side_effect = mock_texts

        # Patch heavy model & units creation
        monkeypatch.setattr(transcribe_mod, "load_audio", lambda p: np.zeros(16000 * 3, dtype=np.float32))
        monkeypatch.setattr(transcribe_mod, "split_quiet", lambda x, c, s=5.0: [x[:16000], x[16000:32000], x[32000:]])
        monkeypatch.setattr(transcribe_mod.GemmaCTC, "from_checkpoint", lambda ckpt: DummyModel())
        monkeypatch.setattr(transcribe_mod, "Units", lambda model_id, gemma_ids: mock_units)
        monkeypatch.setattr(transcribe_mod, "greedy_decode", lambda logits, lens: [0, 1, 2])

        monkeypatch.setattr(
            sys, "argv", ["transcribe.py", "--ckpt", dummy_checkpoint, dummy_wav_file]
        )

        transcribe_mod.main()

        out = capsys.readouterr().out
        expected_stitched = "北京科技大学坐落于海淀区科研实力雄厚"
        assert expected_stitched in out, f"Expected continuous text '{expected_stitched}', got output:\n{out}"
        assert "北京科技大学 坐落于海淀区" not in out, "Detected unwanted space between Chinese chunks!"


# ---------------------------------------------------------------------------
# 3. CLI argument parsing contract
# ---------------------------------------------------------------------------

class TestTranscribeCliContract:
    """Verifies command-line interface arguments and error handling."""

    def test_missing_ckpt_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["transcribe.py", "dummy.wav"])
        with pytest.raises(SystemExit) as exc_info:
            transcribe_mod.main()
        assert exc_info.value.code != 0

    def test_missing_audio_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["transcribe.py", "--ckpt", "model.pt"])
        with pytest.raises(SystemExit) as exc_info:
            transcribe_mod.main()
        assert exc_info.value.code != 0

    def test_output_file_argument_supported(
        self, dummy_wav_file, dummy_checkpoint, tmp_path, monkeypatch
    ):
        """Ticket 07 requirement: '输出文件（如有）使用 UTF-8'.

        The CLI should accept --output (or -o) to write transcriptions to a file.
        """
        output_txt = tmp_path / "output_transcription.txt"

        class DummyModel(torch.nn.Module):
            def forward(self, wav, lens):
                return torch.zeros(len(wav), 5, 4), torch.tensor([5] * len(wav))

            def float(self):
                return self

        mock_units = MagicMock()
        mock_units.decode.return_value = "人工智能语音识别"

        monkeypatch.setattr(transcribe_mod, "load_audio", lambda p: np.zeros(16000, dtype=np.float32))
        monkeypatch.setattr(transcribe_mod.GemmaCTC, "from_checkpoint", lambda ckpt: DummyModel())
        monkeypatch.setattr(transcribe_mod, "Units", lambda model_id, gemma_ids: mock_units)
        monkeypatch.setattr(transcribe_mod, "greedy_decode", lambda logits, lens: [0])

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "transcribe.py",
                "--ckpt",
                dummy_checkpoint,
                dummy_wav_file,
                "--output",
                str(output_txt),
            ],
        )

        transcribe_mod.main()

        # Output file must exist and contain the Chinese text in valid UTF-8
        assert output_txt.exists(), "Output file was not created when --output was specified!"
        content = output_txt.read_text(encoding="utf-8")
        assert "人工智能语音识别" in content


# ---------------------------------------------------------------------------
# 4. Device compatibility and offline safety
# ---------------------------------------------------------------------------

class TestTranscribeDeviceAndSafety:
    """Verifies CPU compatibility and that no remote downloads are initiated."""

    def test_runs_cleanly_on_cpu(self, dummy_wav_file, dummy_checkpoint, monkeypatch, capsys):
        """When CUDA is not available, execution falls back to CPU gracefully."""
        class DummyModel(torch.nn.Module):
            def forward(self, wav, lens):
                return torch.zeros(len(wav), 4, 3), torch.tensor([4] * len(wav))

            def float(self):
                return self

        mock_units = MagicMock()
        mock_units.decode.return_value = "测试成功"

        import itertools
        time_seq = itertools.count(start=1.0, step=0.05)
        monkeypatch.setattr(transcribe_mod.time, "time", lambda: next(time_seq))

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(transcribe_mod, "load_audio", lambda p: np.zeros(16000, dtype=np.float32))
        monkeypatch.setattr(transcribe_mod.GemmaCTC, "from_checkpoint", lambda ckpt: DummyModel())
        monkeypatch.setattr(transcribe_mod, "Units", lambda model_id, gemma_ids: mock_units)
        monkeypatch.setattr(transcribe_mod, "greedy_decode", lambda logits, lens: [0])

        monkeypatch.setattr(
            sys, "argv", ["transcribe.py", "--ckpt", dummy_checkpoint, dummy_wav_file]
        )

        transcribe_mod.main()
        out = capsys.readouterr().out
        assert "测试成功" in out

    def test_offline_checkpoint_uses_local_encoder_id(
        self, dummy_wav_file, dummy_checkpoint, monkeypatch
    ):
        """Units must be initialized with local encoder path, never querying remote HF."""
        created_encoder_id = []

        def fake_units_init(encoder_id, gemma_ids):
            created_encoder_id.append(encoder_id)
            m = MagicMock()
            m.decode.return_value = "测试"
            return m

        import itertools
        time_seq = itertools.count(start=1.0, step=0.05)
        monkeypatch.setattr(transcribe_mod.time, "time", lambda: next(time_seq))

        class DummyModel(torch.nn.Module):
            def forward(self, wav, lens):
                return torch.zeros(len(wav), 4, 3), torch.tensor([4] * len(wav))

            def float(self):
                return self

        monkeypatch.setattr(transcribe_mod, "load_audio", lambda p: np.zeros(16000, dtype=np.float32))
        monkeypatch.setattr(transcribe_mod.GemmaCTC, "from_checkpoint", lambda ckpt: DummyModel())
        monkeypatch.setattr(transcribe_mod, "Units", fake_units_init)
        monkeypatch.setattr(transcribe_mod, "greedy_decode", lambda logits, lens: [0])

        monkeypatch.setattr(
            sys, "argv", ["transcribe.py", "--ckpt", dummy_checkpoint, dummy_wav_file]
        )

        transcribe_mod.main()
        assert len(created_encoder_id) == 1
        # Must match what is recorded in the checkpoint
        assert created_encoder_id[0] == "models/google/embeddinggemma-2"
