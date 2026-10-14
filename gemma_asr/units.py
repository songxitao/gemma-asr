"""CTC 输出单元词表：从中文训练文本构建的单字级词表，每个字映射到 Gemma 词表中的同形单字 token ID。

每个汉字对应 Gemma 词表中已有的单字 token，使得模型分类头仍能利用 Gemma 文本 embedding 进行初始化。
绝不进行整句分词，而是逐字拆解查表，避免分词器将中文切成多字词 token。

单元编号约定：
    0 = CTC blank
    1 = <unk> (未在 Gemma 词表中出现或训练集中频次低于 min_count 的汉字)
    2.. = 保留的 Gemma 单字 token（按频次降序排列）
"""

import json
from collections import Counter
from collections.abc import Iterable
from typing import Optional

from transformers import AutoTokenizer

BLANK, UNK = 0, 1
DEFAULT_MODEL_ID = "models/google/embeddinggemma-2"


class Units:
    def __init__(self, model_id: str, gemma_ids: list[int], tokenizer: Optional[AutoTokenizer] = None):
        self.model_id = model_id
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_id)
        self.gemma_ids = list(gemma_ids)
        self.gemma_to_unit = {g: i + 2 for i, g in enumerate(self.gemma_ids)}
        self._vocab = self.tokenizer.get_vocab()
        self.unk_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 3
        self.unit_to_char = {
            i + 2: self.tokenizer.convert_ids_to_tokens(gid)
            for i, gid in enumerate(self.gemma_ids)
        }

    def __len__(self) -> int:
        return len(self.gemma_ids) + 2

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        """逐字拆分文本并查询每个字符在 Gemma 词表中的同形单字 token ID。

        查不到或得到 tokenizer 的 unk ID 视为缺失，填入 unk_id。
        """
        if isinstance(texts, str):
            texts = [texts]
        res = []
        for t in texts:
            ids = []
            for ch in t:
                gid = self._vocab.get(ch)
                if gid is None or gid == self.unk_id:
                    ids.append(self.unk_id)
                else:
                    ids.append(gid)
            res.append(ids)
        return res

    def encode(self, texts: list[str]) -> list[list[int]]:
        """将文本编码为 unit ID 序列。缺失或不在保留集合中的字符映射为 UNK (1)。"""
        return [[self.gemma_to_unit.get(g, UNK) for g in ids] for ids in self.tokenize(texts)]

    def decode(self, unit_ids: list[int]) -> str:
        """将 unit 序列解码为汉字文本。

        直接拼接对应汉字，不插入空格，忽略 blank (0) 与 <unk> (1)。
        """
        chars = []
        for u in unit_ids:
            u_int = int(u)
            if u_int in self.unit_to_char:
                chars.append(self.unit_to_char[u_int])
        return "".join(chars)

    def save(self, path: str, extra: dict | None = None) -> None:
        """显式使用 utf-8 与 ensure_ascii=False 保存词表元数据。"""
        payload = {"model_id": self.model_id, "gemma_ids": self.gemma_ids, **(extra or {})}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Units":
        """显式使用 utf-8 加载词表元数据。"""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["model_id"], d["gemma_ids"])

    @classmethod
    def build(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        texts: Iterable[str] = (),
        min_count: int = 2,
        batch_size: int = 10_000,
    ) -> tuple["Units", dict]:
        """从训练文本统计单字频次并构建单字级 CTC 词表。

        每个汉字映射到 Gemma 词表中的单字 token ID；频次低于 min_count 或无单字 token 的字不保留。
        返回 (Units 实例, 统计信息字典)。
        """
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        vocab = tokenizer.get_vocab()
        unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 3

        counts: Counter[str] = Counter()
        for t in texts:
            for ch in t:
                if ch.isspace():
                    continue
                counts[ch] += 1

        total_tokens = sum(counts.values())
        missing_chars = []
        kept_gemma_ids = []
        covered_tokens = 0

        for ch, freq in counts.most_common():
            gid = vocab.get(ch)
            if gid is None or gid == unk_id:
                missing_chars.append(ch)
                continue
            if freq >= min_count:
                kept_gemma_ids.append(gid)
                covered_tokens += freq

        coverage = covered_tokens / max(total_tokens, 1)

        stats = {
            "distinct_tokens": len(counts),
            "kept_tokens": len(kept_gemma_ids),
            "total_tokens": total_tokens,
            "token_coverage": coverage,
            "missing_gemma_tokens": len(missing_chars),
            "missing_gemma_samples": missing_chars[:20],
            "missing_tokens": len(missing_chars),
            "missing_samples": missing_chars[:20],
        }

        return cls(model_id, kept_gemma_ids, tokenizer=tokenizer), stats
