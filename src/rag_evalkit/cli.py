"""命令行入口。

输出默认是 Markdown（直接贴进交付报告），加 --json 输出机器可读结果。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .corpus import audit
from .metrics import Run, evaluate, paired_compare
from .report import compare_markdown, corpus_markdown, eval_markdown


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"找不到文件：{path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} 不是合法 JSON（{exc.msg}）")
    if not rows:
        raise SystemExit(f"{path} 里没有可用数据")
    return rows


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"找不到文件：{path}")
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line.strip()]


def _load_runs(path: Path) -> list[Run]:
    runs: list[Run] = []
    for index, row in enumerate(_read_jsonl(path), start=1):
        qid = row.get("qid")
        if not qid:
            raise SystemExit(f"{path} 第 {index} 条缺少 qid")
        runs.append(
            Run(
                qid=str(qid),
                gold_ids=tuple(row.get("gold_ids") or ()),
                retrieved_ids=tuple(row.get("retrieved_ids") or ()),
                latency_ms=row.get("latency_ms"),
                error=row.get("error"),
            )
        )
    return runs


def _parse_ks(value: str) -> tuple[int, ...]:
    try:
        ks = tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError:
        raise SystemExit(f"--ks 需要逗号分隔的整数，收到 {value!r}")
    if not ks or any(k < 1 for k in ks):
        raise SystemExit("--ks 必须包含至少一个 >= 1 的整数")
    return ks


def _emit(text: str, as_json: dict[str, Any] | None) -> None:
    if as_json is not None:
        print(json.dumps(as_json, ensure_ascii=False, indent=2))
    else:
        print(text, end="")


def cmd_eval(args: argparse.Namespace) -> int:
    runs = _load_runs(Path(args.runs))
    ks = _parse_ks(args.ks)
    result = evaluate(runs, ks)
    payload = (
        {
            "n": result.n,
            "recall": {str(k): v for k, v in result.recall.items()},
            "mrr": result.mrr,
            "latency": result.latency,
            "unrecalled": list(result.unrecalled),
            "errors": list(result.errors),
            "error_rate": result.error_rate,
        }
        if args.json
        else None
    )
    _emit(eval_markdown(result), payload)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = _load_runs(Path(args.baseline))
    candidate = _load_runs(Path(args.candidate))
    diff = paired_compare(baseline, candidate, k=args.k)
    payload = (
        {
            "n": diff.n,
            "k": diff.k,
            "baseline_recall": diff.baseline_recall,
            "candidate_recall": diff.candidate_recall,
            "identical": diff.identical,
            "improved": list(diff.improved),
            "regressed": list(diff.regressed),
            "unchanged": list(diff.unchanged),
        }
        if args.json
        else None
    )
    _emit(compare_markdown(diff), payload)
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    path = Path(args.source)
    if args.plain:
        texts = _read_lines(path)
    else:
        texts = [str(row.get(args.field, "")) for row in _read_jsonl(path)]
        texts = [text for text in texts if text.strip()]
    if not texts:
        raise SystemExit("没有读到任何文本片段")
    report = audit(texts, min_len=args.min_len, near_dup_threshold=args.near_dup)
    payload = (
        {
            "n": report.n,
            "no_punct_ratio": report.no_punct_ratio,
            "duplicate_ratio": report.duplicate_ratio,
            "near_duplicate_pairs": report.near_duplicate_pairs,
            "length_p50": report.length_p50,
            "length_p95": report.length_p95,
            "length_max": report.length_max,
            "ascii_punct_ratio": report.ascii_punct_ratio,
            "cjk_ratio": report.cjk_ratio,
            "whitespace_only": report.whitespace_only,
        }
        if args.json
        else None
    )
    _emit(corpus_markdown(report), payload)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-evalkit",
        description="RAG 检索评测与语料体检（零依赖）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval", help="计算召回率、MRR、延迟分位")
    eval_parser.add_argument("runs", help="运行结果 JSONL")
    eval_parser.add_argument("--ks", default="1,3,5,10", help="要统计的 k，逗号分隔")
    eval_parser.add_argument("--json", action="store_true", help="输出 JSON")
    eval_parser.set_defaults(func=cmd_eval)

    compare_parser = subparsers.add_parser("compare", help="两组运行逐题配对对比")
    compare_parser.add_argument("baseline", help="基线运行 JSONL")
    compare_parser.add_argument("candidate", help="改动后运行 JSONL")
    compare_parser.add_argument("--k", type=int, default=5, help="对比时使用的 k")
    compare_parser.add_argument("--json", action="store_true", help="输出 JSON")
    compare_parser.set_defaults(func=cmd_compare)

    audit_parser = subparsers.add_parser("audit", help="语料体检")
    audit_parser.add_argument("source", help="片段 JSONL，或纯文本文件（配 --plain）")
    audit_parser.add_argument("--field", default="text", help="JSONL 中存放文本的字段名")
    audit_parser.add_argument("--plain", action="store_true", help="按纯文本逐行读取")
    audit_parser.add_argument("--min-len", type=int, default=20, help="判定无标点的最小长度")
    audit_parser.add_argument(
        "--near-dup", type=float, default=0.9, help="近重复的 Jaccard 阈值"
    )
    audit_parser.add_argument("--json", action="store_true", help="输出 JSON")
    audit_parser.set_defaults(func=cmd_audit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
