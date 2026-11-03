"""检索评测指标。

三个口径上的决定，直接决定数字能不能信：

1. **未召回的题目记 0 并计入分母，不剔除。**
   剔除会让分母变小，指标虚高。只看"召回的题表现多好"是一种自欺。

2. **检索返回空、调用报错，都算一条结果（rank=None）。**
   跳过失败题等于替系统隐瞒。报错率应该单独统计，而不是从分母里消失。

3. **同时报 @1 和 @5。**
   只看 @1 会把"排序不够好"误判成"召回不到"——两者的修法完全不同：
   前者要调重排，后者要动分段和嵌入。只测一个数会让你优化错方向。

指标定义见 docs/metric-definitions.md，改口径请先改文档。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

Rank = int | None


def first_gold_rank(retrieved: Sequence[str], gold: Iterable[str]) -> Rank:
    """gold 在检索结果中首次出现的 1-based 位置；未出现返回 None。"""
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("gold_ids 为空：这道题没有标准答案，算出来的指标没有意义")
    for position, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold_set:
            return position
    return None


def recall_at_k(ranks: Sequence[Rank], k: int) -> float:
    """Recall@k：top-k 内至少命中一个 gold 的题目占比。

    分母固定为题目总数，未召回的题按 0 计入。
    """
    if k < 1:
        raise ValueError(f"k 必须 >= 1，收到 {k}")
    if not ranks:
        raise ValueError("没有题目，无法计算召回率")
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / len(ranks)


def mrr(ranks: Sequence[Rank]) -> float:
    """MRR：mean(1 / 首次命中排名)，未召回记 0（不剔除）。"""
    if not ranks:
        raise ValueError("没有题目，无法计算 MRR")
    return sum(1.0 / rank for rank in ranks if rank is not None) / len(ranks)


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数，p ∈ (0, 100]。

    延迟这类指标 P95 比均值有信息量：均值会被少数极慢的请求拉平，
    而用户感受到的正是那几条慢的。
    """
    if not values:
        raise ValueError("没有样本，无法计算分位数")
    if not 0 < p <= 100:
        raise ValueError(f"p 必须在 (0, 100] 之间，收到 {p}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * p / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * weight)


@dataclass(frozen=True)
class Run:
    """一次检索运行的单题结果。

    error 非空表示这次检索失败——仍然计入分母，rank 视为 None。
    """

    qid: str
    gold_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    latency_ms: float | None = None
    error: str | None = None

    @property
    def rank(self) -> Rank:
        if self.error:
            return None
        return first_gold_rank(self.retrieved_ids, self.gold_ids)


@dataclass(frozen=True)
class EvalResult:
    n: int
    recall: Mapping[int, float]
    mrr: float
    latency: Mapping[str, float]
    unrecalled: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def error_rate(self) -> float:
        return len(self.errors) / self.n if self.n else 0.0


def evaluate(runs: Sequence[Run], ks: Sequence[int] = (1, 3, 5, 10)) -> EvalResult:
    if not runs:
        raise ValueError("没有运行记录，无法评测")

    ranks = [run.rank for run in runs]
    latencies = [run.latency_ms for run in runs if run.latency_ms is not None]

    latency_stats: dict[str, float] = {}
    if latencies:
        latency_stats = {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "mean": sum(latencies) / len(latencies),
        }

    return EvalResult(
        n=len(runs),
        recall={k: recall_at_k(ranks, k) for k in ks},
        mrr=mrr(ranks),
        latency=latency_stats,
        unrecalled=tuple(run.qid for run, rank in zip(runs, ranks) if rank is None),
        errors=tuple(run.qid for run in runs if run.error),
    )


@dataclass(frozen=True)
class PairedDiff:
    """两组运行的逐题对比。

    `identical=True` 是一个很强的信号：改了参数但每一题的排名完全没变，
    说明改动大概率没有真正生效（配置没被读到、走了缓存、平台侧忽略了该参数）。
    这种"假实验"比做错实验更费时间，因为它让你以为自己在迭代。
    """

    n: int
    k: int
    baseline_recall: float
    candidate_recall: float
    identical: bool
    improved: tuple[str, ...]
    regressed: tuple[str, ...]
    unchanged: tuple[str, ...]


def paired_compare(
    baseline: Sequence[Run],
    candidate: Sequence[Run],
    k: int = 5,
) -> PairedDiff:
    base_by_qid = {run.qid: run for run in baseline}
    cand_by_qid = {run.qid: run for run in candidate}
    shared = [qid for qid in base_by_qid if qid in cand_by_qid]
    if not shared:
        raise ValueError("两组运行没有共同的 qid，无法配对比较")

    base_ranks = [base_by_qid[qid].rank for qid in shared]
    cand_ranks = [cand_by_qid[qid].rank for qid in shared]

    improved: list[str] = []
    regressed: list[str] = []
    unchanged: list[str] = []

    for qid, before, after in zip(shared, base_ranks, cand_ranks):
        before_rank = before if before is not None else float("inf")
        after_rank = after if after is not None else float("inf")
        if after_rank < before_rank:
            improved.append(qid)
        elif after_rank > before_rank:
            regressed.append(qid)
        else:
            unchanged.append(qid)

    return PairedDiff(
        n=len(shared),
        k=k,
        baseline_recall=recall_at_k(base_ranks, k),
        candidate_recall=recall_at_k(cand_ranks, k),
        identical=base_ranks == cand_ranks,
        improved=tuple(improved),
        regressed=tuple(regressed),
        unchanged=tuple(unchanged),
    )
