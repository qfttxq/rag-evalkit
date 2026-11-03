"""rag-evalkit：RAG 检索评测与语料体检，零第三方依赖。

三个命令：

    rag-evalkit eval runs.jsonl           # 召回率、MRR、延迟分位
    rag-evalkit compare a.jsonl b.jsonl   # 逐题配对对比（查"假实验"）
    rag-evalkit audit chunks.jsonl        # 语料体检（调参前先跑这个）

设计原则：只度量，不替你下结论。指标口径写在 docs/metric-definitions.md，
不一致时以文档为准，改口径需要同时改文档和测试。
"""

from __future__ import annotations

from .corpus import CorpusReport, audit, ngram_jaccard
from .metrics import (
    EvalResult,
    PairedDiff,
    Run,
    evaluate,
    first_gold_rank,
    mrr,
    paired_compare,
    percentile,
    recall_at_k,
)
from .report import compare_markdown, corpus_markdown, eval_markdown

__version__ = "0.1.0"

__all__ = [
    "CorpusReport",
    "EvalResult",
    "PairedDiff",
    "Run",
    "audit",
    "compare_markdown",
    "corpus_markdown",
    "eval_markdown",
    "evaluate",
    "first_gold_rank",
    "mrr",
    "ngram_jaccard",
    "paired_compare",
    "percentile",
    "recall_at_k",
]
