"""把评测结果渲染成 Markdown。

报告是给人看的，所以要满足两件事：
1. **结论在前，表格在后**——先给一句"这次能不能用"，再给支撑数字。
2. **不达标的地方写清楚**，不要用"基本符合预期"这类话盖过去。
   交付物里最值钱的部分往往是"已知限制"那一节。
"""

from __future__ import annotations

from .corpus import CorpusReport
from .metrics import EvalResult, PairedDiff


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def eval_markdown(result: EvalResult, title: str = "检索评测报告") -> str:
    lines: list[str] = [f"# {title}", ""]

    ranked = sorted(result.recall.items())
    low_k, low_value = ranked[0]
    high_k, high_value = ranked[-1]

    lines.append(
        f"**{result.n} 道题**：Recall@{low_k} 为 {_pct(low_value)}，"
        f"Recall@{high_k} 为 {_pct(high_value)}，MRR {result.mrr:.3f}。"
    )
    lines.append("")

    if high_value - low_value >= 0.15:
        lines.append(
            f"> 注意 @{low_k} 与 @{high_k} 相差超过 15 个百分点。"
            f"这通常说明**排序**是短板而不是**召回**——"
            f"答案已经在候选集里，只是没排到前面。修法是加/换重排，不是重新分段。"
        )
        lines.append("")

    lines.append("## 召回")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    for k, value in ranked:
        lines.append(f"| Recall@{k} | {_pct(value)} |")
    lines.append(f"| MRR | {result.mrr:.3f} |")
    lines.append(f"| 未召回题数 | {len(result.unrecalled)} |")
    lines.append("")

    if result.latency:
        lines.append("## 延迟")
        lines.append("")
        lines.append("| 指标 | 毫秒 |")
        lines.append("|---|---|")
        for name in ("p50", "p95", "mean"):
            if name in result.latency:
                lines.append(f"| {name.upper() if name != 'mean' else '均值'} | {result.latency[name]:.0f} |")
        lines.append("")

    if result.errors:
        lines.append("## 失败请求")
        lines.append("")
        lines.append(f"共 {len(result.errors)} 条（已计入分母，rank 记为未召回）：")
        lines.append("")
        for qid in result.errors:
            lines.append(f"- `{qid}`")
        lines.append("")

    if result.unrecalled:
        lines.append("## 未召回的题")
        lines.append("")
        lines.append("这些题在两个方向上值得单独看：题目本身写得不清楚，或语料里确实没有答案。")
        lines.append("")
        for qid in result.unrecalled:
            lines.append(f"- `{qid}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def compare_markdown(diff: PairedDiff) -> str:
    lines: list[str] = ["# 配对对比", ""]
    lines.append(
        f"共 {diff.n} 道共同题目，Recall@{diff.k}："
        f"{_pct(diff.baseline_recall)} → {_pct(diff.candidate_recall)}。"
    )
    lines.append("")

    if diff.identical:
        lines.append(
            "> **每一题的排名完全一致。** 这通常不是「改进幅度太小」，"
            "而是改动根本没有生效：配置没被读到、结果走了缓存，或平台侧忽略了该参数。"
            "先验证参数确实传到了，再谈效果。"
        )
        lines.append("")

    lines.append("| 类别 | 题数 |")
    lines.append("|---|---|")
    lines.append(f"| 变好 | {len(diff.improved)} |")
    lines.append(f"| 变差 | {len(diff.regressed)} |")
    lines.append(f"| 不变 | {len(diff.unchanged)} |")
    lines.append("")

    if diff.regressed:
        lines.append("## 变差的题")
        lines.append("")
        lines.append("改动之前先看这一节——只看总体提升会被少数退化题掩盖。")
        lines.append("")
        for qid in diff.regressed:
            lines.append(f"- `{qid}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def corpus_markdown(report: CorpusReport, title: str = "语料体检报告") -> str:
    lines: list[str] = [f"# {title}", ""]
    lines.append(f"共 **{report.n}** 个片段。")
    lines.append("")

    verdict: list[str] = []
    if report.no_punct_ratio >= 0.2:
        verdict.append("无标点片段比例偏高")
    if report.duplicate_ratio >= 0.05:
        verdict.append("完全重复片段偏多")
    if report.length_p95 > 4 * max(report.length_p50, 1):
        verdict.append("长度分布悬殊")

    if verdict:
        lines.append(f"> **先修语料，再调检索。** 检出问题：{'、'.join(verdict)}。")
        lines.append("")
        lines.append(
            "语料没有语义边界时，嵌入模型拿到的是一串无停顿字符，"
            "任何检索参数都救不回来。"
        )
    else:
        lines.append("> 语料基本健康，可以进入检索调优。")
    lines.append("")

    lines.append("| 指标 | 数值 | 说明 |")
    lines.append("|---|---|---|")
    lines.append(
        f"| 无标点片段 | {_pct(report.no_punct_ratio)} | 长度 ≥20 且无句末标点 |"
    )
    lines.append(f"| 完全重复 | {_pct(report.duplicate_ratio)} | 去空白后完全相同 |")
    lines.append(f"| 近重复对 | {report.near_duplicate_pairs} | 3-gram Jaccard ≥ 0.9 |")
    lines.append(f"| 长度 P50 | {report.length_p50:.0f} 字 | |")
    lines.append(f"| 长度 P95 | {report.length_p95:.0f} 字 | |")
    lines.append(f"| 最长片段 | {report.length_max} 字 | 过长通常会稀释嵌入质量 |")
    lines.append(f"| 半角标点占比 | {_pct(report.ascii_punct_ratio)} | 高说明文本经过多次转手 |")
    lines.append(f"| 中文占比 | {_pct(report.cjk_ratio)} | |")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"
