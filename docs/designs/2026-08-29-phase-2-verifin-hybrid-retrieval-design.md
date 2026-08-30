---
design_type: phase
created_at: 2026-08-29
---

# Phase 2 — VeriFin Hybrid Retrieval + Rerank 设计

## Intent Contract

```yaml
intent: 在 Phase 1 索引基座上实现 BM25 + Dense 双路召回、RRF/加权融合与可降级的 Cross-Encoder 精排，输出统一 SearchResult 结构，并证明混合召回 ≥ 单路 BM25
constraints:
  - 默认运行路径不依赖网络与重型依赖（sentence-transformers/FlagEmbedding 全部走 optional extra；reranker 不可用时降级为保持候选顺序的透传）
  - Phase 1 公开 API 与 6 项测试不得破坏（回归全绿）
  - RRF 公式与标准一致（k=60）；最终 score 做 min-max 归一化到 0-1，原始分保留在独立字段
  - 检索层不 import LangGraph；tools/retriever 为纯函数单例，Phase 3 再包装为 Tool
  - embedding 模式是构建期属性（由 build_index.py --embedding 决定），检索器不携带 embedding_mode 参数
success_criteria:
  - tests/test_retrieval.py 6 项测试全部通过（RRF / reranker 状态 / 混合检索 / 召回对比 / 元数据过滤 / 工具函数）
  - 召回验证：全部查询 hybrid ≥ bm25-only 且至少一个查询严格 >；rerank 层召回 ≥ 融合层
  - Phase 1 回归 6 项测试全绿
risk_level: low
```

## Verification Contract

```yaml
verify_steps:
  - run tests: cd /workspace/verifin && .venv/bin/python -m pytest tests/ -v
  - check: RRF 融合排序正确；reranker._available 与当前环境一致性；HybridRetriever 返回 top_k 条且字段完整；召回对比断言成立；year 过滤后结果全来自目标年份；retrieve() 返回 SearchResultSet
  - run demo: 用 Phase 1 产物跑一次 hybrid 检索演示（bm25/dense/rrf/rerank 各层分数可见）
  - confirm: 两组测试全绿（6+6），演示输出中 fusion_method 与 reranker_used 字段符合预期
```

## Governance Contract

```yaml
approval_gates:
  - 本设计文档经人工批准后方可开始实现（HOTL 硬门）
  - 实现完成后运行 code-review 流程（final），BLOCK 项修复后方可视为完成
rollback: 新增代码集中于 src/verifin/retrieval/、src/verifin/tools/ 与 tests/test_retrieval.py，回滚 = 删除上述新增文件并还原 schemas.py/pyproject.toml 改动；索引产物可再生
ownership: 用户为项目 owner 并批准设计；AI agent 负责实现、测试与验证
```

## Scope

| In | Out |
|---|---|
| SearchResult / SearchResultSet 数据类 | 结构化证据抽取（Phase 4） |
| RRFusion（rrf + weighted 两种融合） | 图谱增强检索 |
| Reranker 封装（懒加载 + 降级透传） | FlagEmbedding 的 BGE Reranker 实装（仅留 extra 入口，本 Phase 不验证） |
| HybridRetriever（BM25+Dense+融合+Rerank+元数据过滤） | 查询改写 / 多跳检索规划（Phase 3/5） |
| tools/retriever.py 纯函数单例工具 | LangGraph Tool 包装（Phase 3） |
| tests/test_retrieval.py 6 项测试 + 构造语料 | 真实 FinQA 数据集评测（Phase 7） |
| embedding 工厂补全（hash/bge 命名归一） | 语义分块实验（本 Phase 不做，若做作为 Phase 2 实验开关另立设计） |

## Decisions

| # | 决策 | 选择 | 被否决的替代方案 |
|---|---|---|---|
| 1 | 最终分语义 | score 为 min-max 归一化到 0-1 的综合分；bm25/dense/rerank 原始分保存在独立字段 | 直接用 logit 当 score（量纲不定，展示与阈值均不可用） |
| 2 | 融合算法 | RRF (k=60) 为默认；同时实现 weighted（各列表 min-max 归一后加权，dense_weight 默认 0.5） | 仅 RRF 而保留 dense_weight 死参数；仅加权（丢失排名稳健性） |
| 3 | dense 分数方向 | VectorIndex 保持返回 Chroma distance；检索器内转换为相似度 1/(1+d) 供加权融合，RRF 只用排名 | 改 VectorIndex.search 签名（破坏 Phase 1 API） |
| 4 | Reranker 默认模型 | cross-encoder/ms-marco-MiniLM-L-6-v2（~90MB，语义精排足够） | 默认 BGE Reranker（FlagEmbedding+torch 重装，仅留 [bge-reranker] extra） |
| 5 | 依赖定位 | sentence-transformers → `[rerank]` extra；FlagEmbedding → `[bge-reranker]` extra；core 依赖不增 | 规格原案的 st 进 core（torch 数 GB，破坏 Phase 1 零重依赖契约） |
| 6 | embedding_mode | 构建期属性：get_embedding_function("hash"/"bge") 复用 Phase 1 HashEmbedding；bge 走 chromadb onnx 路径 | HybridRetriever 携带 embedding_mode（查询期无法换 embedding，属接口毒性）；hash(text)%1024 退化实现（进程随机化+无判别力） |
| 7 | 召回断言 | 离线模式：∀query hybrid ≥ bm25 且 ∃query 严格 >；rerank ≥ hybrid（降级透传时相等）；bge 模式再断言整体严格 > | 无条件断言 Hybrid > BM25（hash-dense 为词法信号，构造上不保证，测试会成 flaky） |
| 8 | 召回测试语料 | 测试内构造 8 篇文档 + 6 查询带 GT chunk 标注；干扰文档设计为长文本+高频重复词，GT 文档短而精确 | 依赖 data/ 样例数据跑召回（字段模板雷同，两路几乎同序，无法体现差异） |
| 9 | 元数据过滤 | filter_metadata 为通用 dict 匹配：metadata 扩展字段精确匹配 + doc_id/section_title/page_num 特判 | 写死 year 字段过滤（Phase 1 chunk 尚无 year，公司/年份抽取归 Phase 4；功能仍以测试语料 year 字段验收） |
| 10 | 检索器组装 | 提供 build_hybrid_retriever(persist_dir) 工厂：加载 indexes/bm25.pkl + chroma_db/ + verifin_metadata.db；tools.retriever 用模块级单例 | 每次调用重建索引（资源浪费） |

## Surface

**公开 API（src/verifin）**：`schemas.py` 新增 `SearchResult`（chunk_id/content/score/bm25_score/dense_score/rerank_score/metadata）与 `SearchResultSet`（query/results/total_count/retrieval_time_ms/fusion_method/reranker_used）；`retrieval/fusion.py` 提供 `RRFusion.fuse(ranking_lists, top_k) -> list[tuple[str, float]]` 与 `weighted_fuse(ranking_lists, weights, top_k)`；`retrieval/reranker.py` 提供 `Reranker(model_name)` 懒加载类，`rerank(query, candidates, top_k)` 在不可用时按原序透传零分；`retrieval/hybrid_retriever.py` 提供 `HybridRetriever(bm25_index, vector_index, metadata_store, reranker, fusion, dense_weight, fusion_method).search(query, top_k, use_reranker, filter_metadata) -> SearchResultSet` 与 `build_hybrid_retriever(persist_dir)`；`tools/retriever.py` 提供 `retrieve(query, top_k=10, use_reranker=True, filter_metadata=None)` 与 `get_retriever(persist_dir=...)` 单例。

**流水线（HybridRetriever.search）**：BM25 召回 top_k*2 → Dense 召回 top_k*2（距离转相似度）→ 融合（rrf/weighted）取 top_k*2 → [可选] 取 metadata 中 content 交 Reranker 精排到 top_k（得分覆盖 score，标记 reranker_used）→ filter_metadata 过滤 → 计算检索耗时与 min-max 归一化 → 组装 SearchResultSet。rerank 候选不足或不可用时直接以融合顺序截断。

**动过的文件**：新增 `src/verifin/retrieval/{__init__,fusion,reranker,hybrid_retriever}.py`、`src/verifin/tools/{__init__,retriever}.py`、`tests/test_retrieval.py`；修改 `src/verifin/schemas.py`（新增两 dataclass）、`src/verifin/embedding.py`（工厂命名归一，行为不变）、`pyproject.toml`（新增 [rerank]/[bge-reranker] extras）；`docs/designs/` 新增本设计文档。

## Risks & Open Questions

1. **重依赖验证缺口（低）**：本沙箱不预装 sentence-transformers/torch，真实 CrossEncoder 精排路径以「降级透传 + `_available=False`」测试；执行期将**尝试**安装 [rerank] extra 做真实路径冒烟（下载量 ~1GB，失败则如实报告），不属于验收阻断项。
2. **召回升言的可重复性（低）**：离线 hash-dense 是确定性信号，测试语料固定则可复现；若个别断言在 CI 漂移，以「构造语料微调 + 记录实际召回数字」处理，不放松断言语义。
3. **score 归一化边界**：候选集全同分（如整体 0.0）时 min-max 分母为 0 → 处理为 score 全 0.0，避免除零。
4. **开放问题**：真实数据集下 dense（bge）对召回的实际增益与 RRF 的 top_k*2 候选池大小，留 Phase 7 消融；weighted 模式的 weight 建议值以消融校准，本 Phase 先给 0.5 默认。