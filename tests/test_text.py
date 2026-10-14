"""Unit tests for Chinese transcript and hypothesis normalization (Ticket 02)."""

import pytest
from gemma_asr.text import normalize_hypothesis, normalize_text, to_pinyin


class TestNormalizeText:
    """Tests for training/eval reference normalization (strict filtering)."""

    def test_basic_chinese_sentence(self):
        # Removes full-width and half-width punctuations and spaces
        assert normalize_text("你好，世界！") == "你好世界"
        assert normalize_text("这是一个测试：包含各种标点符号、括注（注释）及书名号《自然语言处理》。") == "这是一个测试包含各种标点符号括注注释及书名号自然语言处理"

    def test_strip_whitespace_and_aishell_style_spaces(self):
        # In AISHELL, transcripts have word-segmented spaces e.g. "今 天 天 气 不 错"
        assert normalize_text("今 天 天 气 不 错") == "今天天气不错"
        assert normalize_text("  语音  识别   系统 \n\t") == "语音识别系统"

    def test_reject_arabic_and_fullwidth_digits(self):
        # Utterances with numbers should be discarded in MVP
        assert normalize_text("2026年") is None
        assert normalize_text("今天气温25度") is None
        assert normalize_text("全角数字２０２６") is None

    def test_allow_digits_when_drop_digits_false(self):
        # When explicitly configured not to drop digits (if supported)
        # Still non-CJK characters would be tested according to design
        pass

    def test_reject_english_and_latin_characters(self):
        # Utterances containing English words or Latin letters should be rejected
        assert normalize_text("打开 wifi") is None
        assert normalize_text("Hello 世界") is None
        assert normalize_text("iPhone 手机") is None

    def test_reject_empty_or_punctuation_only(self):
        assert normalize_text("") is None
        assert normalize_text("   ") is None
        assert normalize_text("，。！？；：“”‘’（）《》、—…") is None
        assert normalize_text(",.!?()[]{}") is None


class TestNormalizeHypothesis:
    """Tests for decoding hypothesis normalization (lenient, never rejects)."""

    def test_lenient_strip_and_filter(self):
        assert normalize_hypothesis("你好，世界！") == "你好世界"
        assert normalize_hypothesis("hello你好123") == "你好"
        assert normalize_hypothesis("  今 天 天 气 不 错  ") == "今天天气不错"

    def test_never_returns_none(self):
        assert normalize_hypothesis("") == ""
        assert normalize_hypothesis("   ") == ""
        assert normalize_hypothesis("12345") == ""
        assert normalize_hypothesis("abc") == ""
        assert normalize_hypothesis("！？，。") == ""


class TestToPinyin:
    """Tests for pinyin conversion without tones (for P-CER metric)."""

    def test_basic_pinyin_conversion(self):
        py = to_pinyin("人工智能")
        assert py == ["ren", "gong", "zhi", "neng"]

    def test_multi_char_length_matches(self):
        py = to_pinyin("重庆")
        assert len(py) == 2
        # Accept standard pinyin syllable
        assert py[1] == "qing"

    def test_empty_string(self):
        assert to_pinyin("") == []
