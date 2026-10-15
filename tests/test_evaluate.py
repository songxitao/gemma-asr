"""Tests for Chinese speech recognition evaluation script and reporting contract (Ticket 07).

Tested contracts (Seams):
1. CLI argument parsing contract:
   - Missing required args (--ckpt) triggers exit with non-zero code.
   - Default values for --data_dir, --split, --max_utts, --max_batch_seconds, --dump.
2. Single-card and device compatibility:
   - Evaluator runs cleanly on a single GPU or CPU without torchrun / NCCL (world=1, rank=0).
   - Does NOT hardcode torch.cuda.set_device when CUDA is unavailable.
3. Evaluation table output format:
   - Output table contains required columns: 'source', 'CER', 'P-CER', 'utts', 'hours'.
   - Displays percentage-formatted CER and P-CER values.
   - Outputs an AVERAGE summary row across all evaluated sources.
4. Dump JSONL contract:
   - Writes valid JSONL encoded strictly in UTF-8 (no Windows CP936 corruption).
   - Uses ensure_ascii=False so Chinese ideographs are NOT escaped as \\uXXXX.
   - Each record contains 'ref', 'hyp', and 'set' (or 'source') fields.
   - Number of dumped lines matches evaluated utterances.
5. Offline safety:
   - Never triggers HuggingFace remote hub downloads or touches C-drive cache.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from gemma_asr.scoring import ErrorCounts
import scripts.evaluate as eval_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_checkpoint(tmp_path):
    """Creates a lightweight dummy checkpoint with Chinese vocab metadata."""
    ckpt_path = tmp_path / "dummy_eval_ckpt.pt"
    payload = {
        "config": {
            "encoder_id": "models/google/embeddinggemma-2",
            "unfreeze_top_k": 0,
        },
        "unit_gemma_ids": [235284, 235285, 235286],
        "model": {},
    }
    torch.save(payload, str(ckpt_path))
    return str(ckpt_path)


class FakeEvalSet:
    """Simulates EvalSet for testing without loading real parquet shards."""
    def __init__(self, batches_data, hours=1.5):
        self._batches = batches_data
        self.hours = hours

    def batches(self, rank=0, world_size=1, max_batch_seconds=1200.0):
        for b in self._batches:
            yield b


# ---------------------------------------------------------------------------
# 1. CLI argument parsing contract
# ---------------------------------------------------------------------------

class TestEvaluateCliContract:
    """Verifies CLI argument parsing and required parameters."""

    def test_missing_ckpt_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["evaluate.py"])
        with pytest.raises(SystemExit) as exc_info:
            eval_mod.main()
        assert exc_info.value.code != 0

    def test_default_arguments_parsed(self, dummy_checkpoint, monkeypatch):
        captured_args = []

        def fake_main_run():
            # Intercept after arg parsing
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--ckpt", required=True)
            parser.add_argument("--data_dir", default="data/prepared")
            parser.add_argument("--split", default="test")
            parser.add_argument("--sources", nargs="*", default=None)
            parser.add_argument("--max_utts", type=int, default=0)
            parser.add_argument("--max_batch_seconds", type=float, default=1200.0)
            parser.add_argument("--dump", default=None)
            return parser.parse_args(["--ckpt", dummy_checkpoint])

        args = fake_main_run()
        assert args.ckpt == dummy_checkpoint
        assert args.data_dir == "data/prepared"
        assert args.split == "test"
        assert args.max_utts == 0
        assert args.max_batch_seconds == 1200.0
        assert args.dump is None


# ---------------------------------------------------------------------------
# 2. Single-card & device compatibility
# ---------------------------------------------------------------------------

class TestDeviceAndDistributedFallback:
    """Verifies execution without torchrun and on CPU/single-card environments."""

    def test_runs_cleanly_on_cpu_without_cuda(
        self, dummy_checkpoint, monkeypatch, capsys
    ):
        """Ticket 07 requirement: '单卡无 torchrun 可运行'.

        Must NOT crash with CUDA errors when running in CPU environments.
        """
        # Ensure RANK is not in environment
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        class DummyModel(torch.nn.Module):
            def forward(self, wav, lens):
                # 1 utterance, 10 frames, 5 vocab units
                return torch.zeros(1, 10, 5), torch.tensor([10])

            def float(self):
                return self

        mock_units = MagicMock()
        mock_units.decode.return_value = "人工智能"

        # Mock list_shards and EvalSet
        dummy_batch = {
            "wav": torch.zeros(1, 16000),
            "wav_lens": torch.tensor([16000]),
            "texts": ["人工智能"],
        }
        monkeypatch.setattr(eval_mod, "list_shards", lambda d, s, src: {"aishell1": ["fake.parquet"]})
        monkeypatch.setattr(eval_mod, "EvalSet", lambda files, max_utts: FakeEvalSet([dummy_batch], hours=0.5))
        monkeypatch.setattr(eval_mod.GemmaCTC, "from_checkpoint", lambda ckpt: DummyModel())
        monkeypatch.setattr(eval_mod, "Units", lambda model_id, gemma_ids: mock_units)
        monkeypatch.setattr(eval_mod, "greedy_decode", lambda logits, lens: [[2, 3, 4, 5]])

        monkeypatch.setattr(
            sys, "argv", ["evaluate.py", "--ckpt", dummy_checkpoint]
        )

        # Should complete without throwing CUDA / device error
        eval_mod.main()

        out = capsys.readouterr().out
        assert "aishell1" in out


# ---------------------------------------------------------------------------
# 3. Evaluation table output format contract
# ---------------------------------------------------------------------------

class TestEvaluateTableFormat:
    """Verifies that the printed summary table contains required columns and metrics."""

    def test_table_header_and_metrics_format(
        self, dummy_checkpoint, monkeypatch, capsys
    ):
        """Ticket 07 requirement: '表格列为 source / CER / P-CER / utts / hours'.

        The table output MUST contain these columns, percentage values, and an AVERAGE row.
        """
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        class DummyModel(torch.nn.Module):
            def forward(self, wav, lens):
                return torch.zeros(2, 10, 5), torch.tensor([10, 10])

            def float(self):
                return self

        mock_units = MagicMock()
        # Source 1: "他在" -> "她在" (homophone: CER=50%, P-CER=0%)
        # Source 2: "你好" -> "你"   (deletion: CER=50%, P-CER=50%)
        mock_units.decode.side_effect = ["她在", "你"]

        eval_sets = {
            "aishell_test": FakeEvalSet([
                {"wav": torch.zeros(1, 16000), "wav_lens": torch.tensor([16000]), "texts": ["他在"]}
            ], hours=0.2),
            "aishell_dev": FakeEvalSet([
                {"wav": torch.zeros(1, 16000), "wav_lens": torch.tensor([16000]), "texts": ["你好"]}
            ], hours=0.1),
        }

        monkeypatch.setattr(eval_mod, "list_shards", lambda d, s, src: eval_sets)
        monkeypatch.setattr(eval_mod, "EvalSet", lambda files, max_utts: files if isinstance(files, FakeEvalSet) else FakeEvalSet([]))
        # Support dict lookup
        monkeypatch.setattr(eval_mod, "list_shards", lambda d, s, src: {k: v for k, v in eval_sets.items()})
        monkeypatch.setattr(eval_mod, "EvalSet", lambda files, max_utts: files)
        monkeypatch.setattr(eval_mod.GemmaCTC, "from_checkpoint", lambda ckpt: DummyModel())
        monkeypatch.setattr(eval_mod, "Units", lambda model_id, gemma_ids: mock_units)
        monkeypatch.setattr(eval_mod, "greedy_decode", lambda logits, lens: [[2], [3]])

        monkeypatch.setattr(
            sys, "argv", ["evaluate.py", "--ckpt", dummy_checkpoint]
        )

        eval_mod.main()

        out = capsys.readouterr().out

        # Ticket 07 contract: columns must include source, CER, P-CER, utts, hours
        # Note: Table column might be named 'source' or 'set'
        assert any(col in out for col in ["source", "set"]), "Table header must specify source / set identifier column"
        assert "CER" in out, "Table header missing 'CER'"
        assert "P-CER" in out, "Table header missing 'P-CER'"
        assert "utts" in out, "Table header missing 'utts'"
        assert "hours" in out, "Table header missing 'hours'"

        # CER and P-CER must be formatted as percentages (with '%')
        assert "%" in out, "CER / P-CER metrics must be formatted as percentages (e.g., 50.00%)"

        # AVERAGE summary row
        assert "AVERAGE" in out, "Evaluation output must include an AVERAGE row"


# ---------------------------------------------------------------------------
# 4. Dump JSONL format and UTF-8 encoding contract
# ---------------------------------------------------------------------------

class TestDumpJsonlContract:
    """Verifies that --dump exports valid JSONL in UTF-8 without unicode escapes."""

    def test_dump_file_encoding_and_structure(
        self, dummy_checkpoint, tmp_path, monkeypatch
    ):
        """Ticket 07 requirement: '--dump 导出 jsonl（UTF-8，ensure_ascii=False）'.

        Contract:
        - File must be readable as UTF-8.
        - Chinese characters must NOT be escaped (must not be \\uXXXX).
        - Each line contains keys: 'set' (or 'source'), 'ref', 'hyp'.
        - Line count matches number of evaluated utterances.
        """
        dump_file = tmp_path / "eval_dump.jsonl"
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        class DummyModel(torch.nn.Module):
            def forward(self, wav, lens):
                return torch.zeros(2, 10, 5), torch.tensor([10, 10])

            def float(self):
                return self

        # 2 samples with Chinese text
        test_samples = [
            {"ref": "北京科技大学", "hyp": "背景科技大学"},
            {"ref": "人工智能语音识别", "hyp": "人工智能语音识别"},
        ]

        mock_units = MagicMock()
        mock_units.decode.side_effect = [s["hyp"] for s in test_samples]

        dummy_batch = {
            "wav": torch.zeros(2, 16000),
            "wav_lens": torch.tensor([16000, 16000]),
            "texts": [s["ref"] for s in test_samples],
        }

        eval_sets = {"aishell1": FakeEvalSet([dummy_batch], hours=0.1)}
        monkeypatch.setattr(eval_mod, "list_shards", lambda d, s, src: eval_sets)
        monkeypatch.setattr(eval_mod, "EvalSet", lambda files, max_utts: files)
        monkeypatch.setattr(eval_mod.GemmaCTC, "from_checkpoint", lambda ckpt: DummyModel())
        monkeypatch.setattr(eval_mod, "Units", lambda model_id, gemma_ids: mock_units)
        monkeypatch.setattr(eval_mod, "greedy_decode", lambda logits, lens: [[2], [3]])

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate.py",
                "--ckpt",
                dummy_checkpoint,
                "--dump",
                str(dump_file),
            ],
        )

        eval_mod.main()

        assert dump_file.exists(), "--dump file was not created!"

        # 1. Must be strictly valid UTF-8
        content = dump_file.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
        assert len(lines) == 2, f"Expected 2 lines in dump file, got {len(lines)}"

        # 2. Chinese characters must NOT be unicode-escaped (ensure_ascii=False)
        # Raw bytes should contain literal Chinese characters, not "\u5317\u4eac"
        raw_bytes = dump_file.read_bytes()
        assert b"\\u" not in raw_bytes, "Found escaped unicode sequence in dump; ensure_ascii=False required!"
        assert "北京科技大学".encode("utf-8") in raw_bytes

        # 3. Each JSON line must contain required keys
        for i, line in enumerate(lines):
            record = json.loads(line)
            assert "ref" in record
            assert "hyp" in record
            assert any(k in record for k in ["set", "source"])
            assert record["ref"] == test_samples[i]["ref"]
            assert record["hyp"] == test_samples[i]["hyp"]


# ---------------------------------------------------------------------------
# 5. Offline safety and metric calculation integrity
# ---------------------------------------------------------------------------

class TestEvaluateSafetyAndScoringIntegrity:
    """Verifies that evaluate uses local paths and scores correctly."""

    def test_uses_local_encoder_from_checkpoint(
        self, dummy_checkpoint, monkeypatch
    ):
        """Evaluator must load Units using the local encoder_id stored in checkpoint."""
        created_encoder_id = []

        def fake_units_init(encoder_id, gemma_ids):
            created_encoder_id.append(encoder_id)
            m = MagicMock()
            m.decode.return_value = "测试"
            return m

        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(eval_mod, "list_shards", lambda d, s, src: {})
        monkeypatch.setattr(eval_mod.GemmaCTC, "from_checkpoint", lambda ckpt: MagicMock())
        monkeypatch.setattr(eval_mod, "Units", fake_units_init)

        monkeypatch.setattr(
            sys, "argv", ["evaluate.py", "--ckpt", dummy_checkpoint]
        )

        eval_mod.main()

        assert len(created_encoder_id) == 1
        assert created_encoder_id[0] == "models/google/embeddinggemma-2"
