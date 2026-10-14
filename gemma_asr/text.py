"""Chinese transcript and hypothesis normalization pipeline.

Every transcript (training, evaluation, and decoding) goes through text normalization:

- normalize_text(text, drop_digits=True): Strict normalization for training and evaluation
  reference transcripts:
  * Unicode NFKC normalization (full-width characters/digits to standard half-width).
  * Strips all punctuation, symbols, whitespace (including full-width and word-segmented spaces).
  * Rejects (returns None) if any digits exist (when drop_digits=True) or any characters
    outside CJK Unified Ideographs (U+4E00 - U+9FFF) remain.
  * Rejects (returns None) empty or punctuation-only strings.
  * Returns a continuous sequence of Chinese characters.

- normalize_hypothesis(text): Lenient normalization for ASR hypothesis / model output:
  * Unicode NFKC normalization.
  * Strips all non-CJK characters directly.
  * Never rejects (returns empty string "" if no valid Chinese characters exist).

- to_pinyin(text): Converts Chinese text to tone-less pinyin syllable list (Style.NORMAL),
  used for Pinyin-CER (P-CER) evaluation.
"""

import unicodedata
from pypinyin import Style, lazy_pinyin


def normalize_text(text: str, drop_digits: bool = True) -> str | None:
    """Normalize reference transcript to continuous CJK characters.

    Returns None if the utterance should be discarded (e.g. contains digits, Latin/English,
    foreign scripts, or is empty after stripping punctuation and whitespace).
    """
    if not text:
        return None

    # Step 1: Unicode NFKC normalization
    s = unicodedata.normalize("NFKC", text)

    # Step 2: Remove punctuation, symbols, whitespace, and control characters
    # P* (punctuation), S* (symbol), Z* (separator), C* (other/control) or whitespace
    cleaned = "".join(
        c for c in s
        if not (unicodedata.category(c)[0] in ("P", "S", "Z", "C") or c.isspace())
    )
    if not cleaned:
        return None

    # Step 3: Digits rejection
    if drop_digits and any(c.isdigit() for c in cleaned):
        return None

    # Step 4: Validate all remaining characters are CJK Unified Ideographs (U+4E00 - U+9FFF)
    for c in cleaned:
        if not ("\u4e00" <= c <= "\u9fff"):
            if not drop_digits and c.isdigit():
                continue
            return None

    return cleaned


def normalize_hypothesis(text: str) -> str:
    """Lenient normalization for model output / decoding hypothesis (never rejects)."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    return "".join(c for c in s if "\u4e00" <= c <= "\u9fff")


def to_pinyin(text: str) -> list[str]:
    """Convert Chinese text to a list of tone-less pinyin syllables (one per character)."""
    if not text:
        return []
    # Filter to pure CJK characters to ensure 1-to-1 character-to-syllable mapping
    chars = "".join(c for c in text if "\u4e00" <= c <= "\u9fff")
    if not chars:
        return []
    return lazy_pinyin(chars, style=Style.NORMAL)


