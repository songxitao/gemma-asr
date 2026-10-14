"""CER / P-CER (Pinyin CER), accumulated as counts so they can be summed across GPUs."""

from dataclasses import dataclass, fields

import jiwer
import torch
import torch.distributed as dist

from .text import normalize_hypothesis, to_pinyin


@dataclass
class ErrorCounts:
    char_errors: int = 0
    ref_chars: int = 0
    pinyin_errors: int = 0
    ref_pinyin: int = 0
    utts: int = 0

    def __iadd__(self, other: "ErrorCounts") -> "ErrorCounts":
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))
        return self

    def all_reduce(self, device) -> "ErrorCounts":
        if not (dist.is_available() and dist.is_initialized()):
            return self
        t = torch.tensor([getattr(self, f.name) for f in fields(self)], dtype=torch.long, device=device)
        dist.all_reduce(t)
        return ErrorCounts(*t.tolist())

    def summary(self) -> dict:
        return {
            "cer": self.char_errors / self.ref_chars if self.ref_chars > 0 else 0.0,
            "pcer": self.pinyin_errors / self.ref_pinyin if self.ref_pinyin > 0 else 0.0,
            "utts": self.utts,
            "ref_chars": self.ref_chars,
        }


def _char_errors(refs: list[str], hyps: list[str]) -> tuple[int, int]:
    o = jiwer.process_characters(refs, hyps)
    return o.substitutions + o.deletions + o.insertions, o.substitutions + o.deletions + o.hits


def _pinyin_errors(refs: list[str], hyps: list[str]) -> tuple[int, int]:
    ref_pinyins = [" ".join(to_pinyin(r)) for r in refs]
    hyp_pinyins = [" ".join(to_pinyin(h)) for h in hyps]
    o = jiwer.process_words(ref_pinyins, hyp_pinyins)
    return o.substitutions + o.deletions + o.insertions, o.substitutions + o.deletions + o.hits


def score_batch(refs: list[str], hyps: list[str]) -> ErrorCounts:
    """refs are already normalized; hyps get lenient normalization. Empty refs are skipped."""
    pairs = [(r.strip(), normalize_hypothesis(h)) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return ErrorCounts()
    refs = [r for r, _ in pairs]
    hyps = [h for _, h in pairs]
    ce, rc = _char_errors(refs, hyps)
    pe, rp = _pinyin_errors(refs, hyps)
    return ErrorCounts(ce, rc, pe, rp, len(pairs))
