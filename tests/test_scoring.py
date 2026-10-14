"""Unit tests for Chinese speech recognition evaluation metrics CER and P-CER (Ticket 04).

Covers:
- ErrorCounts dataclass fields (char_errors, ref_chars, pinyin_errors, ref_pinyin, utts)
- ErrorCounts.summary() format and zero-division safety
- ErrorCounts.__iadd__ in-place accumulation
- ErrorCounts.all_reduce distributed fallback
- score_batch exact match (CER=0, P-CER=0)
- score_batch homophone substitution (CER>0, P-CER=0)
- score_batch edits (deletion, insertion, non-homophone substitution)
- score_batch empty hypothesis (full deletion: CER=1.0, P-CER=1.0)
- score_batch hypothesis normalization (whitespace, punctuation, English stripping)
- score_batch empty reference filtering (skipped utterances)
- Multi-utterance batch scoring and accumulation
"""

import pytest
from gemma_asr.scoring import ErrorCounts, score_batch


class TestErrorCounts:
    """Tests for ErrorCounts container, summary metrics, and accumulation."""

    def test_default_init(self):
        counts = ErrorCounts()
        assert counts.char_errors == 0
        assert counts.ref_chars == 0
        assert counts.pinyin_errors == 0
        assert counts.ref_pinyin == 0
        assert counts.utts == 0

    def test_summary_keys_and_values(self):
        counts = ErrorCounts(
            char_errors=2,
            ref_chars=10,
            pinyin_errors=1,
            ref_pinyin=10,
            utts=3,
        )
        summary = counts.summary()

        # Contract: {"cer", "pcer", "utts", "ref_chars"}
        assert set(summary.keys()) == {"cer", "pcer", "utts", "ref_chars"}
        assert summary["cer"] == pytest.approx(0.2)
        assert summary["pcer"] == pytest.approx(0.1)
        assert summary["utts"] == 3
        assert summary["ref_chars"] == 10

        # Vietnamese metrics must be removed
        assert "wer" not in summary
        assert "wer_notone" not in summary
        assert "ref_words" not in summary

    def test_summary_zero_division_safety(self):
        # Empty counts should not raise ZeroDivisionError
        counts = ErrorCounts()
        summary = counts.summary()
        assert summary["cer"] == 0.0
        assert summary["pcer"] == 0.0
        assert summary["utts"] == 0
        assert summary["ref_chars"] == 0

    def test_iadd_accumulation(self):
        c1 = ErrorCounts(
            char_errors=1,
            ref_chars=6,
            pinyin_errors=0,
            ref_pinyin=6,
            utts=1,
        )
        c2 = ErrorCounts(
            char_errors=2,
            ref_chars=4,
            pinyin_errors=1,
            ref_pinyin=4,
            utts=1,
        )
        c1 += c2

        assert c1.char_errors == 3
        assert c1.ref_chars == 10
        assert c1.pinyin_errors == 1
        assert c1.ref_pinyin == 10
        assert c1.utts == 2

        summary = c1.summary()
        assert summary["cer"] == pytest.approx(0.3)
        assert summary["pcer"] == pytest.approx(0.1)
        assert summary["utts"] == 2
        assert summary["ref_chars"] == 10

    def test_all_reduce_uninitialized(self):
        # When dist is not initialized, all_reduce should return self without error
        counts = ErrorCounts(char_errors=3, ref_chars=10, pinyin_errors=1, ref_pinyin=10, utts=2)
        reduced = counts.all_reduce(device="cpu")
        assert reduced.char_errors == 3
        assert reduced.ref_chars == 10
        assert reduced.pinyin_errors == 1
        assert reduced.ref_pinyin == 10
        assert reduced.utts == 2


class TestScoreBatchBasic:
    """Core phonetic and character metric contract tests for score_batch."""

    def test_exact_match(self):
        # Identical characters -> CER = 0, P-CER = 0
        refs = ["今天天气真好"]
        hyps = ["今天天气真好"]
        counts = score_batch(refs, hyps)

        assert counts.char_errors == 0
        assert counts.ref_chars == 6
        assert counts.pinyin_errors == 0
        assert counts.ref_pinyin == 6
        assert counts.utts == 1

        summary = counts.summary()
        assert summary["cer"] == pytest.approx(0.0)
        assert summary["pcer"] == pytest.approx(0.0)

    def test_homophone_substitution(self):
        # Ticket 04 requirement:
        # ref="他在" hyp="她在" -> CER = 0.5 (1 sub / 2 chars), P-CER = 0.0 (both are ["ta", "zai"])
        refs = ["他在"]
        hyps = ["她在"]
        counts = score_batch(refs, hyps)

        assert counts.char_errors == 1
        assert counts.ref_chars == 2
        assert counts.pinyin_errors == 0
        assert counts.ref_pinyin == 2
        assert counts.utts == 1

        summary = counts.summary()
        assert summary["cer"] == pytest.approx(0.5)
        assert summary["pcer"] == pytest.approx(0.0)

    def test_multi_char_homophone_substitution(self):
        # All homophones: "北京科技大学" vs "背景克己达雪"
        # Both pinyin sequences: ["bei", "jing", "ke", "ji", "da", "xue"]
        refs = ["北京科技大学"]
        hyps = ["背景克己达雪"]
        counts = score_batch(refs, hyps)

        assert counts.char_errors == 6
        assert counts.ref_chars == 6
        assert counts.pinyin_errors == 0
        assert counts.ref_pinyin == 6

        summary = counts.summary()
        assert summary["cer"] == pytest.approx(1.0)
        assert summary["pcer"] == pytest.approx(0.0)

    def test_deletion(self):
        # Ticket 04 requirement:
        # ref="你好" hyp="你" -> CER = 0.5, P-CER = 0.5
        refs = ["你好"]
        hyps = ["你"]
        counts = score_batch(refs, hyps)

        assert counts.char_errors == 1
        assert counts.ref_chars == 2
        assert counts.pinyin_errors == 1
        assert counts.ref_pinyin == 2

        summary = counts.summary()
        assert summary["cer"] == pytest.approx(0.5)
        assert summary["pcer"] == pytest.approx(0.5)

    def test_insertion(self):
        # Insertion: ref="你好" hyp="你好啊" -> 1 insertion / 2 ref units
        refs = ["你好"]
        hyps = ["你好啊"]
        counts = score_batch(refs, hyps)

        assert counts.char_errors == 1
        assert counts.ref_chars == 2
        assert counts.pinyin_errors == 1
        assert counts.ref_pinyin == 2

        summary = counts.summary()
        assert summary["cer"] == pytest.approx(0.5)
        assert summary["pcer"] == pytest.approx(0.5)

    def test_non_homophone_substitution(self):
        # Substitution with different sound: ref="中国" hyp="美国"
        refs = ["中国"]
        hyps = ["美国"]
        counts = score_batch(refs, hyps)

        assert counts.char_errors == 1
        assert counts.ref_chars == 2
        assert counts.pinyin_errors == 1
        assert counts.ref_pinyin == 2

        summary = counts.summary()
        assert summary["cer"] == pytest.approx(0.5)
        assert summary["pcer"] == pytest.approx(0.5)


class TestScoreBatchEdgeCases:
    """Edge cases: empty hypothesis, noise hypothesis, empty reference skipping."""

    def test_empty_hyp_is_full_deletion(self):
        # Ticket 04 requirement:
        # Empty hypothesis counts as complete deletion: CER = 1.0, P-CER = 1.0
        refs = ["语音识别"]
        hyps = [""]
        counts = score_batch(refs, hyps)

        assert counts.char_errors == 4
        assert counts.ref_chars == 4
        assert counts.pinyin_errors == 4
        assert counts.ref_pinyin == 4
        assert counts.utts == 1

        summary = counts.summary()
        assert summary["cer"] == pytest.approx(1.0)
        assert summary["pcer"] == pytest.approx(1.0)

    def test_hyp_normalized_to_empty(self):
        # Non-CJK noise in hyp is stripped by normalize_hypothesis -> becomes empty string
        refs = ["语音识别"]
        hyps = ["  123 ... abc !? "]
        counts = score_batch(refs, hyps)

        assert counts.char_errors == 4
        assert counts.ref_chars == 4
        assert counts.pinyin_errors == 4
        assert counts.ref_pinyin == 4

        summary = counts.summary()
        assert summary["cer"] == pytest.approx(1.0)
        assert summary["pcer"] == pytest.approx(1.0)

    def test_hyp_with_punctuation_and_whitespace(self):
        # Model output containing spaces and punctuation should be normalized leniently
        refs = ["深度学习"]
        hyps = [" 深 度 学 习 ， ！ "]
        counts = score_batch(refs, hyps)

        assert counts.char_errors == 0
        assert counts.ref_chars == 4
        assert counts.pinyin_errors == 0
        assert counts.ref_pinyin == 4

        summary = counts.summary()
        assert summary["cer"] == pytest.approx(0.0)
        assert summary["pcer"] == pytest.approx(0.0)

    def test_skip_empty_refs(self):
        # Empty or whitespace-only refs should be skipped
        refs = ["", "   ", "人工智能", ""]
        hyps = ["忽略一", "忽略二", "人工智能", "忽略三"]
        counts = score_batch(refs, hyps)

        assert counts.utts == 1
        assert counts.char_errors == 0
        assert counts.ref_chars == 4
        assert counts.pinyin_errors == 0
        assert counts.ref_pinyin == 4

    def test_all_empty_refs(self):
        refs = ["", "  \t\n  "]
        hyps = ["测试", "样例"]
        counts = score_batch(refs, hyps)

        assert counts.utts == 0
        assert counts.char_errors == 0
        assert counts.ref_chars == 0
        assert counts.pinyin_errors == 0
        assert counts.ref_pinyin == 0

        summary = counts.summary()
        assert summary["cer"] == 0.0
        assert summary["pcer"] == 0.0
        assert summary["utts"] == 0

    def test_empty_batch(self):
        counts = score_batch([], [])
        assert counts.utts == 0
        assert counts.char_errors == 0
        assert counts.ref_chars == 0
        assert counts.pinyin_errors == 0
        assert counts.ref_pinyin == 0


class TestScoreBatchMultiUtterance:
    """Batch scoring and multi-batch accumulation verification."""

    def test_mixed_multi_utterance_batch(self):
        refs = [
            "他在",      # 2 chars: homophone sub -> 1 char_err, 0 pinyin_err
            "你好",      # 2 chars: deletion -> 1 char_err, 1 pinyin_err
            "世界",      # 2 chars: exact match -> 0 char_err, 0 pinyin_err
            "  ",        # empty ref -> skipped
            "语音",      # 2 chars: empty hyp -> 2 char_err, 2 pinyin_err
        ]
        hyps = [
            "她在",
            "你",
            "世界",
            "跳过",
            "",
        ]
        counts = score_batch(refs, hyps)

        # 4 valid utterances, total ref_chars = 2 + 2 + 2 + 2 = 8
        assert counts.utts == 4
        assert counts.ref_chars == 8
        assert counts.char_errors == 1 + 1 + 0 + 2  # 4
        assert counts.ref_pinyin == 8
        assert counts.pinyin_errors == 0 + 1 + 0 + 2  # 3

        summary = counts.summary()
        assert summary["cer"] == pytest.approx(4 / 8)    # 0.5
        assert summary["pcer"] == pytest.approx(3 / 8)   # 0.375
        assert summary["utts"] == 4
        assert summary["ref_chars"] == 8

    def test_accumulate_batches_with_iadd(self):
        # Batch 1: homophone
        batch1 = score_batch(["他在"], ["她在"])
        # Batch 2: deletion
        batch2 = score_batch(["你好"], ["你"])

        total = ErrorCounts()
        total += batch1
        total += batch2

        assert total.utts == 2
        assert total.ref_chars == 4
        assert total.char_errors == 2
        assert total.ref_pinyin == 4
        assert total.pinyin_errors == 1

        summary = total.summary()
        assert summary["cer"] == pytest.approx(2 / 4)  # 0.5
        assert summary["pcer"] == pytest.approx(1 / 4)  # 0.25
        assert summary["utts"] == 2
        assert summary["ref_chars"] == 4
