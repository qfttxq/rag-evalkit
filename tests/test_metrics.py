"""指标口径测试。

这些测试保护的是"数字能不能信"，不是"代码跑不跑得通"。
如果有人为了好看去改分母、跳过失败题，这里的断言会先拦住。
"""

from __future__ import annotations

import pytest

from rag_evalkit.metrics import (
    Run,
    evaluate,
    first_gold_rank,
    mrr,
    paired_compare,
    percentile,
    recall_at_k,
)


def test_first_gold_rank_returns_one_based_position():
    assert first_gold_rank(["a", "b", "c"], ["c"]) == 3


def test_first_gold_rank_returns_none_when_missing():
    assert first_gold_rank(["a", "b"], ["z"]) is None


def test_first_gold_rank_uses_earliest_of_multiple_gold():
    assert first_gold_rank(["a", "b", "c"], ["c", "a"]) == 1


def test_empty_gold_is_rejected():
    with pytest.raises(ValueError):
        first_gold_rank(["a"], [])


def test_recall_keeps_unrecalled_in_denominator():
    """4 题里 2 题召回在 @1，答案必须是 0.5 —— 而不是剔除未召回的 1.0。"""
    assert recall_at_k([1, 1, None, None], 1) == 0.5


def test_recall_at_k_boundaries():
    ranks = [1, 2, 3, 4]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 3) == 0.75
    assert recall_at_k(ranks, 4) == 1.0


def test_mrr_penalises_misses():
    assert mrr([1]) == 1.0
    assert mrr([None]) == 0.0
    assert mrr([1, None]) == 0.5


def test_mrr_uses_reciprocal_rank():
    assert mrr([2]) == 0.5
    assert mrr([4]) == 0.25


def test_percentile_interpolates_and_handles_single_value():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5.0


def test_percentile_rejects_out_of_range():
    with pytest.raises(ValueError):
        percentile([1, 2], 0)
    with pytest.raises(ValueError):
        percentile([1, 2], 101)


def test_error_run_is_counted_as_miss():
    """检索报错的题即使 gold 就在第一位，也算未召回。"""
    run = Run(qid="q1", gold_ids=("c1",), retrieved_ids=("c1",), error="timeout")
    assert run.rank is None


def test_evaluate_reports_error_rate_and_ids():
    runs = [
        Run(qid="q1", gold_ids=("c1",), retrieved_ids=("c1",)),
        Run(qid="q2", gold_ids=("c2",), retrieved_ids=(), error="HTTP 500"),
    ]
    result = evaluate(runs)
    assert result.n == 2
    assert result.recall[1] == 0.5
    assert result.errors == ("q2",)
    assert result.error_rate == 0.5
    assert result.unrecalled == ("q2",)


def test_evaluate_rejects_empty_input():
    with pytest.raises(ValueError):
        evaluate([])


def test_latency_percentiles_are_computed():
    runs = [Run(qid=f"q{i}", gold_ids=("c1",), retrieved_ids=("c1",), latency_ms=ms)
            for i, ms in enumerate([100, 200, 300, 400])]
    result = evaluate(runs)
    assert result.latency["p50"] == 250
    assert result.latency["p95"] == 385


def test_paired_compare_flags_identical_runs():
    """逐题完全一致 → identical=True，用来识别"参数没生效"的假实验。"""
    baseline = [
        Run(qid="q1", gold_ids=("c1",), retrieved_ids=("c2", "c1")),
        Run(qid="q2", gold_ids=("c2",), retrieved_ids=("c1", "c2")),
    ]
    candidate = [
        Run(qid="q1", gold_ids=("c1",), retrieved_ids=("c2", "c1")),
        Run(qid="q2", gold_ids=("c2",), retrieved_ids=("c1", "c2")),
    ]
    diff = paired_compare(baseline, candidate, k=2)
    assert diff.identical is True
    assert diff.improved == ()
    assert diff.regressed == ()
    assert diff.unchanged == ("q1", "q2")


def test_paired_compare_detects_direction():
    baseline = [
        Run(qid="q1", gold_ids=("c1",), retrieved_ids=("c2", "c1")),  # rank 2
        Run(qid="q2", gold_ids=("c2",), retrieved_ids=("c2", "c1")),  # rank 1
    ]
    candidate = [
        Run(qid="q1", gold_ids=("c1",), retrieved_ids=("c1", "c2")),  # rank 1 变好
        Run(qid="q2", gold_ids=("c2",), retrieved_ids=("c1", "c2")),  # rank 2 变差
    ]
    diff = paired_compare(baseline, candidate, k=1)
    assert diff.improved == ("q1",)
    assert diff.regressed == ("q2",)
    assert diff.identical is False
    assert diff.baseline_recall == 0.5
    assert diff.candidate_recall == 0.5


def test_paired_compare_requires_shared_qids():
    baseline = [Run(qid="q1", gold_ids=("c1",), retrieved_ids=("c1",))]
    candidate = [Run(qid="q9", gold_ids=("c1",), retrieved_ids=("c1",))]
    with pytest.raises(ValueError):
        paired_compare(baseline, candidate)
