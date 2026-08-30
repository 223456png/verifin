---
intent: 在 Phase 1 索引基座上实现 BM25 + Dense 双路召回、RRF/加权融合与可降级的 Cross-Encoder 精排，输出统一 SearchResult 结构，并证明混合召回 ≥ 单路 BM25
success_criteria: tests/test_retrieval.py 6 项测试全部通过；召回验证（∀query hybrid ≥ bm25 且 ∃query 严格 >，rerank ≥ hybrid）；Phase 1 回归 6 项全绿；用 Phase 1 产物跑通 hybrid 检索演示
risk_level: low
auto_approve: true
worktree: false
---

# Phase 2 — VeriFin Hybrid Retrieval + Rerank 工作流

> 项目根目录：/workspace/verifin。Python 命令一律 `.venv/bin/python`。设计文档：docs/designs/2026-08-29-phase-2-verifin-hybrid-retrieval-design.md（已批准）。
> 环境事实：sentence-transformers/torch 未安装 → Reranker 走降级透传路径测试；真实 rerank 冒烟在全部测试通过后另行尝试（非验收阻断）。

## Steps

- [x] **Step 1: 编写规格测试 tests/test_retrieval.py（TDD RED 基线）**
action: 编写 tests/test_retrieval.py，内容包含：(1) test_rrf_fusion —— 两个排序列表做 RRF(k=60)，断言同现文档排名高于单列表文档且输出按分降序、长度为 top_k；(2) test_reranker_available —— 构造 Reranker()（未装 st 时 _available=False），断言 rerank 透传保持候选原序且返回 [(cid, 0.0)]；(3) test_hybrid_retriever_search —— 用 fixture 语料构建 BM25+VectorIndex(HashEmbedding, tmp_path)+MetadataStore，search("revenue", top_k=3) 断言 len==3、字段完整（query/results/score 0-1 区间/fusion_method=="rrf"/reranker_used==False）；(4) test_retrieval_recall_improvement —— 语料含 8+ 文档与 6 个带 GT chunk_id 的查询（GT 文档短而精确、干扰文档长且高频重复词，本文件内直接构造 DocumentChunk，metadata 带 year），对每个 query 比较 bm25-only recall、hybrid recall、hybrid+rerank recall（@10），断言 ∀q hybrid≥bm25 且 ∃q 严格 >、∀q rerank≥hybrid；(5) test_metadata_filter —— search(..., filter_metadata={"year": 2024}) 断言结果集非空且全部来自 metadata year==2024 的 chunk；(6) test_retriever_tool —— 向 get_retriever 注入临时构建的 retriever 后调 retrieve("revenue", top_k=2)，断言返回 SearchResultSet 类型且 len==2。测试文件顶部 import 与设计文档公开 API 一致（此步后 pytest 必然 RED，属预期）
verify:
  - type: artifact
    path: tests/test_retrieval.py
    assert:
      kind: exists

- [x] **Step 2: 扩展 src/verifin/schemas.py 并实现 retrieval/fusion.py**
action: schemas.py 新增 SearchResult(chunk_id/content/score: float/bm25_score: Optional[float]=None/dense_score: Optional[float]=None/rerank_score: Optional[float]=None/metadata: dict) 与 SearchResultSet(query/results: List[SearchResult]/total_count/retrieval_time_ms: float/fusion_method: str/reranker_used: bool)，均为 dataclass 且 metadata 用 field(default_factory=dict)；创建 src/verifin/retrieval/__init__.py 与 fusion.py：RRFusion(k=60).fuse(ranking_lists: List[List[tuple[str,float]]], top_k) —— 对每个列表按序取 rank(1-indexed) 累加 1/(rank+k)，按 rrf 分降序取 top_k；模块级 weighted_fuse(ranking_lists, weights, top_k) —— 各列表分数 min-max 归一后按权重求和（空列表跳过、单元素集分母为 0 时置 0.5）
verify: .venv/bin/python -c "from verifin.schemas import SearchResult, SearchResultSet; from verifin.retrieval.fusion import RRFusion, weighted_fuse; f=RRFusion(); r=f.fuse([[('a',9),('b',8)],[('b',7),('c',6)]], top_k=3); assert r[0][0]=='b', r; r2=weighted_fuse([[('a',9),('b',8)],[('b',7),('c',6)]],[0.5,0.5],top_k=2); assert set(x[0] for x in r2)=={'a','b'}, r2; assert r2[0][1]>=r2[1][1]; print('fusion-ok', r)"
loop: until 退出码 0 且输出以 fusion-ok 开头
max_iterations: 5

- [x] **Step 3: 实现 src/verifin/retrieval/reranker.py**
action: 实现 Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")：_load_model 懒加载 sentence_transformers.CrossEncoder（ImportError/其他加载异常 → loguru.warning 并 _available=False，绝不抛出）；rerank(query, candidates: List[tuple[str,str]], top_k) —— _available=False 时返回 [(cid, 0.0) for 原序候选[:top_k]]（透传保持顺序）；可用时 model.predict([(query,content)...]) 后按分降序取 top_k 返回 [(cid, float(score))]；提供 available 只读 property
verify: .venv/bin/python -c "from verifin.retrieval.reranker import Reranker; r=Reranker(); out=r.rerank('q',[('a','A'),('b','B')],top_k=2); assert not r.available or True; assert [x[0] for x in out]==['a','b'] and all(s==0.0 for _,s in out), out; print('reranker-ok', r.available)"
loop: until 退出码 0 且输出以 reranker-ok 开头
max_iterations: 5

- [x] **Step 4: 实现 src/verifin/retrieval/hybrid_retriever.py 与 tools/retriever.py**
action: hybrid_retriever.py 实现 HybridRetriever(bm25_index, vector_index, metadata_store, reranker: Optional[Reranker]=None, fusion: Optional[RRFusion]=None, dense_weight: float=0.5, fusion_method: str="rrf").search(query, top_k=10, use_reranker=True, filter_metadata=None) -> SearchResultSet，流水线：bm25.search(top_k*2) → vector.search(top_k*2)（distance 转 similarity 1/(1+d)）→ rrf/weighted 融合取 top_k*2 → [可选] rerank 精排（从 metadata_store 取 content，候选不足或 degrad 时按融合顺序截断；rerank 时 reranker_used=True）→ filter_metadata 通用 dict 匹配（metadata 扩展字段精确匹配 + doc_id/section_title/page_num 特判）→ 组 SearchResult：score 为最终分 min-max 归一 0-1（分母为 0 时全 0.0），bm25/dense 原始分回填，metadata 从 store 组装，perf_counter 计 retrieval_time_ms；另实现 build_hybrid_retriever(persist_dir=".")（加载 indexes/bm25.pkl、chroma_db/、verifin_metadata.db）与 tools/retriever.py：模块级单例 _retriever + get_retriever(persist_dir=".", force_rebuild=False) + retrieve(query, top_k=10, use_reranker=True, filter_metadata=None)，创建 src/verifin/tools/__init__.py
verify: .venv/bin/python - <<'PYEOF'
import tempfile, uuid
from verifin.schemas import DocumentChunk
from verifin.indexing.bm25_index import BM25Index
from verifin.indexing.vector_index import VectorIndex
from verifin.indexing.metadata_store import MetadataStore
from verifin.retrieval.hybrid_retriever import HybridRetriever
texts = ["Q4 revenue of Titan Inc was $1,200 million.",
         "Revenue revenue revenue and more revenue for Titan Titan Titan.",
         "Operating margin of Orion was 21 percent in 2024."]
chunks = [DocumentChunk(chunk_id=uuid.uuid4().hex, doc_id=f"d{i}", doc_name="t.md", content=t,
                        page_num=1, metadata={"year": 2024 if i != 1 else 2023}) for i, t in enumerate(texts)]
tmp = tempfile.mkdtemp()
bm25 = BM25Index(); bm25.build(chunks)
vec = VectorIndex(persist_dir=tmp); vec.add_chunks(chunks)
store = MetadataStore(path=tmp + "/m.db"); store.upsert_chunks(chunks)
hr = HybridRetriever(bm25_index=bm25, vector_index=vec, metadata_store=store)
res = hr.search("revenue 2024", top_k=3, use_reranker=True)
assert len(res.results) == 3 and res.fusion_method == "rrf" and res.reranker_used is False
assert all(0.0 <= r.score <= 1.0 for r in res.results)
assert any("year" in r.metadata or r.metadata.get("year") == 2024 for r in res.results)
print("hybrid-ok", res.total_count, round(res.retrieval_time_ms, 2))
PYEOF
loop: until 退出码 0 且输出以 hybrid-ok 开头
max_iterations: 5

- [x] **Step 5: 运行 fusion 与 reranker 规格测试**
verify: .venv/bin/python -m pytest tests/test_retrieval.py -k "rrf or reranker" -x -q
loop: until 退出码 0 且输出 passed
max_iterations: 5

- [x] **Step 6: 运行 hybrid 检索 / 召回 / 过滤 规格测试**
action: 运行三个测试；若召回断言在构造语料上不成立（设计 Risk 2 预授权），微调 tests/test_retrieval.py 中语料文档与查询（保持断言语义：∀q ≥、∃q >、rerank ≥），并把实际召回数字写入测试注释
verify: .venv/bin/python -m pytest tests/test_retrieval.py -k "hybrid_retriever or recall or metadata_filter" -x -q
loop: until 退出码 0 且输出 passed
max_iterations: 5

- [x] **Step 7: 运行 retriever_tool 规格测试**
verify: .venv/bin/python -m pytest tests/test_retrieval.py -k retriever_tool -x -q
loop: until 退出码 0 且输出 passed
max_iterations: 5

- [x] **Step 8: 全量验收（Phase 2 + Phase 1 回归）**
verify:
  - type: shell
    command: .venv/bin/python -m pytest tests/ -v
  - type: shell
    command: .venv/bin/python -m compileall -q src/verifin
loop: until 退出码 0 且 pytest 输出 14 passed（审查后发现项修复后：8 项 Phase 2 + 6 项 Phase 1 回归）
max_iterations: 3

- [x] **Step 9: 端到端演示（Phase 1 产物 + Hybrid 检索）**
action: 用已构建的 Phase 1 索引（indexes/bm25.pkl + chroma_db/ + verifin_metadata.db）跑演示：build_hybrid_retriever(".") 后对 "What was NovaTech revenue in fiscal 2024?" 执行 search(top_k=3)，打印每个结果的 doc_id/section_title/score/bm25_score/dense_score 与 SearchResultSet 的 fusion_method/reranker_used/retrieval_time_ms；同时验证 bm25-only 对同查询的 top-3 与 hybrid 结果差异可观察（打印对比）
verify: .venv/bin/python - <<'PYEOF'
from verifin.retrieval.hybrid_retriever import build_hybrid_retriever
hr = build_hybrid_retriever(".")
res = hr.search("What was NovaTech revenue in fiscal 2024?", top_k=3, use_reranker=True)
assert len(res.results) == 3
print("fusion_method:", res.fusion_method, "| reranker_used:", res.reranker_used, "| time_ms:", round(res.retrieval_time_ms, 2))
for r in res.results:
    print("-", r.metadata.get("doc_id"), "|", r.metadata.get("section_title"), "| score:", round(r.score, 4), "| bm25:", round(r.bm25_score or -1, 3), "| dense:", round(r.dense_score or -1, 3))
print("demo-ok")
PYEOF
loop: until 退出码 0 且输出含 demo-ok
max_iterations: 5

- [x] **Step 10: 代码审查与人工验收**
action: 对 Phase 2 新增/修改代码运行 HOTL code-review（final，对照本工作流与 Phase 2 设计文档契约）；修复 BLOCK 项并重跑 Step 8；随后向用户汇报：两组测试结果、召回对比数字、演示输出、与验收标准逐项对照结论
gate: human