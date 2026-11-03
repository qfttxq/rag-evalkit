"""语料体检：在调检索参数之前，先看切出来的文本本身什么样。

一个反复出现的教训：**检索效果差，多数时候不是检索的问题。**

如果切出来的片段有一半没有标点、或者彼此近重复，那么：
- 嵌入模型拿到的是一串没有停顿信号的字符，语义边界模糊；
- top-k 里挤满同义重复片段，看起来"召回很多"，实际只覆盖了一个点。

这两种情况调权重、换重排模型都不会有明显改善。先量出来，再决定要不要动检索。

本模块只做度量，不改文本。改写请在你的流水线上做——这里给出的是判断依据。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

SENTENCE_END = "。！？!?"
CJK_PUNCT = "。、，！？：；（）《》〈〉「」『』【】—…·"
ASCII_PUNCT = ",?!:;"
WHITESPACE = re.compile(r"\s+")
CJK = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class CorpusReport:
    n: int
    no_punct_ratio: float
    duplicate_ratio: float
    near_duplicate_pairs: int
    length_p50: float
    length_p95: float
    length_max: int
    ascii_punct_ratio: float
    cjk_ratio: float
    whitespace_only: int


def has_sentence_end(text: str) -> bool:
    return any(char in SENTENCE_END for char in text)


def is_unpunctuated(text: str, min_len: int = 20) -> bool:
    """长度达标却一个句末标点都没有。

    min_len 用来排除"标题""短标签"这类本来就不该有标点的片段——
    把它们算进无标点率会让数字虚高，也会让你误判问题规模。
    """
    if len(text) < min_len:
        return False
    return not has_sentence_end(text)


def ascii_punct_ratio(text: str) -> float:
    """半角标点占全部标点的比例。

    中文语料里出现半角标点，通常说明经过了多次转手
    （复制粘贴、表格导出、程序拼接），这类片段往往还带着其他脏数据。
    """
    ascii_count = sum(1 for char in text if char in ASCII_PUNCT)
    cjk_count = sum(1 for char in text if char in CJK_PUNCT)
    total = ascii_count + cjk_count
    return ascii_count / total if total else 0.0


def ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    """字符 n-gram Jaccard 相似度，用来找近重复片段。"""

    def grams(text: str) -> set[str]:
        compact = WHITESPACE.sub("", text)
        if len(compact) < n:
            return {compact} if compact else set()
        return {compact[i : i + n] for i in range(len(compact) - n + 1)}

    left, right = grams(a), grams(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def audit(
    texts: Sequence[str],
    min_len: int = 20,
    near_dup_threshold: float = 0.9,
    max_near_dup_pairs: int = 200_000,
) -> CorpusReport:
    """对一批片段做体检。

    near_dup_threshold 默认 0.9，这个值只适合长片段。

    实测：25 字左右的中文片段，**只改一个字**，3-gram Jaccard 就从 1.0 掉到 0.77。
    也就是说短片段上 0.9 几乎检不出任何近重复。判断标准应该跟着片段的
    典型长度走——几百字的文档段落用 0.9，几十字的问答对用 0.75 上下。

    near_dup_threshold 低于阈值通常是"讨论同一主题的不同段落"，不该当重复删掉。

    近重复检测是 O(n^2)，超过 max_near_dup_pairs 会跳过并返回 0，
    避免在大语料上把 CLI 卡死。大规模场景请先做 shingling 分桶或 MinHash。
    """
    if not texts:
        raise ValueError("语料为空，无法体检")

    n = len(texts)
    no_punct = sum(1 for text in texts if is_unpunctuated(text, min_len))

    normalized = [WHITESPACE.sub("", text) for text in texts]
    counted = Counter(normalized)
    duplicate_slots = sum(count - 1 for count in counted.values() if count > 1)

    lengths = sorted(len(text) for text in texts)
    length_p50 = _quantile(lengths, 50)
    length_p95 = _quantile(lengths, 95)

    punct_ratios = [ascii_punct_ratio(text) for text in texts]
    cjk_chars = sum(len(CJK.findall(text)) for text in texts)
    total_chars = sum(len(text) for text in texts)

    near_pairs = 0
    unique_texts = list(dict.fromkeys(texts))
    estimated_pairs = len(unique_texts) * (len(unique_texts) - 1) // 2
    if estimated_pairs <= max_near_dup_pairs:
        for i in range(len(unique_texts)):
            for j in range(i + 1, len(unique_texts)):
                if ngram_jaccard(unique_texts[i], unique_texts[j]) >= near_dup_threshold:
                    near_pairs += 1

    return CorpusReport(
        n=n,
        no_punct_ratio=no_punct / n,
        duplicate_ratio=duplicate_slots / n,
        near_duplicate_pairs=near_pairs,
        length_p50=length_p50,
        length_p95=length_p95,
        length_max=lengths[-1],
        ascii_punct_ratio=sum(punct_ratios) / n,
        cjk_ratio=cjk_chars / total_chars if total_chars else 0.0,
        whitespace_only=sum(1 for text in texts if not text.strip()),
    )


def _quantile(sorted_values: Sequence[int], p: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight)
