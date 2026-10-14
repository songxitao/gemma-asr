"""Unit tests for Character-level CTC Units mapping to Gemma Tokens (Ticket 03)."""

import json
from pathlib import Path
import pytest
from gemma_asr.units import BLANK, UNK, Units

LOCAL_MODEL_ID = str(Path(__file__).resolve().parent.parent / "models" / "google" / "embeddinggemma-2")


@pytest.fixture(scope="module")
def gemma_model_path():
    """Ensure local EmbeddingGemma 2 tokenizer directory exists."""
    p = Path(LOCAL_MODEL_ID)
    if not (p / "tokenizer.json").exists():
        pytest.skip(f"Local tokenizer not found at {LOCAL_MODEL_ID}")
    return str(p)


class TestUnits:
    def test_build_and_coverage(self, gemma_model_path):
        corpus = [
            "今天天气真不错",
            "我们一起去公园散步吧",
            "人工智能语音识别系统",
            "今天天气确实很好",
        ]
        units, stats = Units.build(gemma_model_path, corpus, min_count=1)

        # Check special units: 0 is blank, 1 is unk
        assert len(units) == len(units.gemma_ids) + 2
        assert stats["distinct_tokens"] > 0
        assert stats["kept_tokens"] > 0
        assert stats["token_coverage"] > 0.99

        # Ensure all kept units are mapped to legitimate Gemma single character IDs
        for gid in units.gemma_ids:
            decoded = units.tokenizer.decode([gid]).strip()
            # Must be a single character CJK ideograph, NOT multi-word like "人工智能"
            assert len(decoded) == 1, f"Expected single character, got: {decoded}"

    def test_encode_and_decode_roundtrip(self, gemma_model_path):
        corpus = ["甚至出现交易几乎停滞的情况", "标杆房企必然调整市场战略"]
        units, _ = Units.build(gemma_model_path, corpus, min_count=1)

        encoded = units.encode(["标杆房企"])[0]
        # Unit IDs must be >= 2 for mapped in-vocab characters
        assert all(u >= 2 for u in encoded)
        assert len(encoded) == 4

        # Decode must join characters directly without inserting spaces
        decoded = units.decode(encoded)
        assert decoded == "标杆房企"

    def test_unknown_and_special_tokens(self, gemma_model_path):
        corpus = ["你好世界"]
        units, _ = Units.build(gemma_model_path, corpus, min_count=1)

        # '天' is not in the built vocab, so it should map to UNK (1)
        encoded = units.encode(["你好天"])[0]
        assert encoded[0] >= 2
        assert encoded[1] >= 2
        assert encoded[2] == UNK

        # Blank (0) and UNK (1) should be dropped during decode
        decoded = units.decode([BLANK, encoded[0], UNK, encoded[1], BLANK])
        assert decoded == "你好"

    def test_save_and_load_utf8(self, gemma_model_path, tmp_path):
        corpus = ["测试汉字编解码保存与加载"]
        units, _ = Units.build(gemma_model_path, corpus, min_count=1)

        save_path = tmp_path / "units_zh.json"
        units.save(str(save_path))

        # Verify JSON file is valid UTF-8
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "gemma_ids" in data
        assert len(data["gemma_ids"]) == len(units.gemma_ids)

        loaded_units = Units.load(str(save_path))
        assert len(loaded_units) == len(units)
        assert loaded_units.gemma_ids == units.gemma_ids
        assert loaded_units.decode(loaded_units.encode(["测试"])[0]) == "测试"
