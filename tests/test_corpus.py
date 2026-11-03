"""语料体检测试。

重点在两处容易判错的地方：
- 短片段（标题、标签）不该被算成"无标点"；
- 重复检测要区分完全重复和近重复，前者按槽位算，后者按对数算。
"""

from __future__ import annotations

import pytest

from rag_evalkit.corpus import audit, is_unpunctuated, ngram_jaccard


def test_short_text_is_not_flagged():
    """标题类短片段本来就没有句末标点，算进去会让问题规模虚高。"""
    assert is_unpunctuated("准备活动") is False


def test_long_text_without_sentence_end_is_flagged():
    text = "今天我们先做热身然后进入主题练习最后放松身体"
    assert is_unpunctuated(text) is True


def test_text_with_sentence_end_is_clean():
    assert is_unpunctuated("今天我们先做热身，然后进入主题练习。") is False


def test_ngram_jaccard_identical_text_is_one():
    assert ngram_jaccard("今天我们练习呼吸", "今天我们练习呼吸") == 1.0


def test_ngram_jaccard_unrelated_text_is_low():
    score = ngram_jaccard("今天我们练习呼吸控制", "服务器部署需要检查端口映射")
    assert score < 0.1


def test_audit_detects_exact_duplicates():
    texts = ["同样的内容完全一致这是第一段", "同样的内容完全一致这是第一段", "另一段不同的内容在这里"]
    report = audit(texts)
    assert report.duplicate_ratio == pytest.approx(1 / 3)
    assert report.n == 3


def test_audit_counts_unpunctuated_ratio():
    texts = [
        "这一段的长度足够但没有出现任何句末标点符号所以我们把它算作无标点片段",
        "这一段是正常的，有标点。",
    ]
    report = audit(texts, min_len=10)
    assert report.no_punct_ratio == pytest.approx(0.5)


def test_audit_reports_length_distribution():
    report = audit(["短一点的内容在这里", "长很多的内容" * 10])
    assert report.length_p50 <= report.length_p95
    assert report.length_max == len("长很多的内容" * 10)


def test_audit_detects_near_duplicates():
    """25 字的片段只差一个字，3-gram Jaccard 只有约 0.77。

    所以近重复阈值必须按语料长度校准——默认 0.9 是给长片段用的，
    短片段上会漏判。这里用 0.75 才能检出。
    """
    base = "练习之前先做三分钟的热身活动让身体充分舒展避免受伤"
    near = "练习之前先做三分钟的热身活动让身体充分舒展以免受伤"
    report = audit([base, near], near_dup_threshold=0.75)
    assert report.near_duplicate_pairs == 1
    assert ngram_jaccard(base, near) < 0.8


def test_audit_rejects_empty_corpus():
    with pytest.raises(ValueError):
        audit([])
