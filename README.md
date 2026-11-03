# rag-evalkit

RAG 检索评测与语料体检。**零第三方依赖**，只用 Python 标准库。

给你的检索链路一个能写进验收报告的数字：Recall@k、MRR、延迟分位、切分质量，
以及一个多数评测脚本不提供的东西——**两组运行之间的逐题配对对比**，用来识别"改了参数但其实没生效"的假实验。

> Evaluation and corpus audit for RAG pipelines. Zero dependencies. Computes Recall@k, MRR,
> latency percentiles, chunk-quality metrics, and per-question paired comparison between runs.

```
$ rag-evalkit eval runs.jsonl
**20 道题**：Recall@1 为 40.0%，Recall@10 为 80.0%，MRR 0.520。

> 注意 @1 与 @10 相差超过 15 个百分点。这通常说明**排序**是短板而不是**召回**——
答案已经在候选集里，只是没排到前面。修法是加/换重排，不是重新分段。
```

---

## 为什么写这个

做 RAG 项目时反复遇到同样的三件事，市面上的评测脚本都没解决：

**一、指标算出来偏乐观。** 很多脚本会把"检索报错""返回空"的题目剔除，分母变小，
指标就上去了。或者只统计"召回的题里排得多好"，未召回的当不存在。
这类数字在验收会上站不住——甲方随便抽一道没答上的题就能戳穿。

**二、只看一个 k，会优化错方向。** 我的第一个项目 `Recall@1` 只有 63.3%，
看着像"召回不行"，但 `@5` 是 93.3%、`@20` 是 98.3%。答案明明在候选集里，只是排序没排上来。
如果当时照着 @1 去改分段和嵌入模型，会浪费一周，而且效果不会动——**该修的是重排**。

**三、改了半天，其实参数没生效。** 我在一个平台上做过三组检索权重实验
（无权重 / 语义优先 / 关键词优先），每组的逐题排名**完全一致**。
一开始以为是"权重影响不大"，后来发现是平台侧根本没读到这个参数。
`compare` 命令的 `identical` 标志就是为了抓这种情况——**假实验比做错实验更贵**，
因为它让你以为自己在迭代。

---

## 安装

不需要装任何依赖：

```bash
git clone https://github.com/<your-name>/rag-evalkit.git
cd rag-evalkit
pip install -e .
```

不装也能跑：

```bash
PYTHONPATH=src python -m rag_evalkit eval examples/runs_baseline.jsonl
```

要求 Python ≥ 3.10。

---

## 三个命令

### 1. `eval` —— 算指标

```bash
rag-evalkit eval examples/runs_baseline.jsonl --ks 1,3,5,10
```

输出 Markdown（可直接贴进交付文档），加 `--json` 输出机器可读结果。

### 2. `compare` —— 逐题配对对比

```bash
rag-evalkit compare examples/runs_baseline.jsonl examples/runs_rerank.jsonl --k 5
```

```
共 20 道共同题目，Recall@5：70.0% → 80.0%。

| 类别 | 题数 |
|---|---|
| 变好 | 8 |
| 变差 | 0 |
| 不变 | 12 |
```

只看总体数字会掩盖退化：曾有一次改动整体涨了 2 个点，但有 3 道题从 @1 掉到 @8。
那 3 道恰好是客户最常问的。`compare` 会把它们单独列出来。

如果输出 `identical`，说明**每一题排名都没变**——先去查参数有没有真的传进去，
别急着调下一组。

### 3. `audit` —— 调参之前先跑这个

```bash
rag-evalkit audit examples/chunks.jsonl              # JSONL，字段名默认 text
rag-evalkit audit plain.txt --plain                  # 纯文本，一行一段
```

```
| 指标 | 数值 | 说明 |
|---|---|---|
| 无标点片段 | 48.7% | 长度 ≥20 且无句末标点 |
| 完全重复 | 6.2% | 去空白后完全相同 |
| 近重复对 | 31 | 3-gram Jaccard ≥ 0.9 |
| 长度 P95 | 412 字 | |
```

（上面是真实项目修复前的数字，不是示例数据。）

**这一条最省钱：** 我在一个音频转写项目里发现 48.7% 的句子没有标点。
在这种语料上调检索参数、换嵌入模型都是白费——模型拿到的是一串没有语义边界的字符。
先把标点重建做好，`Recall@5` 直接从 71% 提到 93.3%。**语料问题的修法在语料，不在检索。**

---

## 数据结构

`eval` 和 `compare` 读 JSONL，一行一道题：

```json
{
  "qid": "q01",
  "gold_ids": ["c001"],
  "retrieved_ids": ["c001", "c025", "c035", "..."],
  "latency_ms": 279
}
```

- `gold_ids` 可以有多个（一题多标准答案），命中其中任意一个即算召回
- `retrieved_ids` 按相关性降序，只放 top-k 就够
- `latency_ms` 可选
- 检索失败时加 `"error": "HTTP 500"` —— **仍然计入分母**，不会被静默跳过

`audit` 读 `{"id": "...", "text": "..."}`，或用 `--plain` 读纯文本。

---

## 三条口径（这些是刻意的，不是疏忽）

**1. 未召回的题记 0，计入分母。**
剔除未召回的题，`Recall@k` 会变成"命中题里的准确率"，这个数字没有意义。

**2. 报错、返回空，都算一条结果。**
失败率应该单独统计（`error_rate`），而不是从分母里消失。隐瞒失败比失败本身更糟。

**3. 同时报 @1 和 @5，差值大于 15 个点会提示。**
两者差距大意味着"排序问题"；两者都低才是"召回问题"。修法完全不同。

详细定义见 [docs/metric-definitions.md](docs/metric-definitions.md)。

---

## 代码里的几个刻意选择

- **零依赖**：评测工具不该让你先装 20 个包。连 `pandas` 都没用，`statistics` 足够
- **纯函数优先**：`recall_at_k` / `mrr` / `percentile` 都只吃列表返回数字，不碰文件、不碰网络，好测
- **`Run` / `EvalResult` 用 frozen dataclass**：字段名写错会立刻报错，而不是在报告里出现一个 `None`
- **CLI 默认输出 Markdown**：交付文档里要的就是表格，不用再转一遍
- **测试保护的是口径**：`test_metrics.py` 里断言"未召回必须计入分母"，
  防止后来的人为了数字好看改掉分母

---

## 目录

```
rag-evalkit/
├── src/rag_evalkit/
│   ├── metrics.py      # Recall@k / MRR / 分位数 / 配对对比
│   ├── corpus.py       # 无标点率 / 重复率 / 长度分布 / 近重复
│   ├── report.py       # Markdown 渲染
│   └── cli.py          # 命令行
├── tests/              # 27 个测试，全部围绕口径正确性
├── examples/           # 合成数据，用于跑通命令
└── docs/metric-definitions.md
```

跑测试：

```bash
pip install -e ".[dev]"
pytest -q
```

---

## 不做什么

明确边界，避免被当成万能工具：

- **不做检索**。它只评测，不提供向量库或检索实现
- **不跑模型**。忠实性、答案质量这类需要 LLM 判断的指标不在范围内（我用自己的另一套脚本做，涉及模型调用和人工复核，不适合放进零依赖库）
- **不改写语料**。`audit` 只度量。清洗属于你的流水线，这里给的是"该不该洗"的判断依据
- **近重复检测是 O(n²)**。一万条以上建议先做 MinHash 分桶，否则会很慢

---

## 许可

MIT
