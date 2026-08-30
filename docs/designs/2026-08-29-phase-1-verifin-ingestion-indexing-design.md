---
design_type: phase
created_at: 2026-08-29
---

# Phase 1 — VeriFin 数据 Ingestion + 索引构建 设计

## Intent Contract

```yaml
intent: 为 VeriFin 搭建数据基座：PDF/文本解析 → 按标题层级分块 → 构建 BM25 索引、ChromaDB 向量索引与 SQLite 元数据存储，并用测试与端到端构建脚本验收
constraints:
  - 默认运行路径不依赖网络（embedding 可插拔，默认本地确定性实现；sentence-transformers 为可选依赖）
  - 每个 chunk ≤ 512 tokens，且保留标题层级（section_title / section_level）与页码
  - 仅做数字 PDF 文本抽取（pymupdf/pypdf），不做 OCR
  - Python ≥ 3.10，src 布局，包名 verifin（项目更名，原 PEV）
  - 不破坏用户数据的原始语义（解析只抽取文本，不改写内容）
success_criteria:
  - tests/test_ingestion.py 中 6 个规格测试全部通过
  - scripts/build_index.py 在 data/ 样例数据集上完整跑通（退出码 0），产出 BM25 持久化文件、Chromadb 持久化目录、SQLite 元数据库
risk_level: low
```

## Verification Contract

```yaml
verify_steps:
  - run tests: cd /workspace/verifin && python -m pytest tests/test_ingestion.py -v
  - check: 解析输出的文本非空；chunk 按标题正确切分且估算 token 数 ≤ 512；BM25 搜索返回 top_k 条；向量搜索返回正确 chunk_id；构建脚本日志中 BM25 chunk 数 == Chroma collection count
  - confirm: 测试全绿（6 passed），build_index.py 正常退出且 manifest 记录与索引实体数量一致
```

## Governance Contract

```yaml
approval_gates:
  - 本设计文档经人工批准后方可开始实现（HOTL 硬门）
  - 实现完成后运行 code-review 流程，发现的问题修复后方可视为完成
rollback: 全新项目，回滚 = 删除 /workspace/verifin 即可；索引产物（bm25 持久化、chroma 目录、sqlite 文件）均为可再生衍生物，已列入 .gitignore
ownership: 用户为项目 owner 并批准设计；AI agent 负责实现、测试与验证
```

## Scope

| In | Out |
|---|---|
| 文本/Markdown 解析（含标题识别） | OCR / 扫描件还原 |
| PDF 解析（pymupdf 主、pypdf 回退），文本 + 简单表格文本 | 复杂表格结构还原（Phase 4） |
| 标题层级分块 + 512 token 上限 + 段落级二次切分 | 语义分割 / embedding 驱动的切分 |
| BM25Okapi 索引（build/search/持久化） | 多路召回融合与 Rerank（Phase 2） |
| ChromaDB 持久化向量索引（可插拔 embedding） | 倒数图 / 图谱增强检索（Phase 2） |
| SQLite 元数据存储（写入/查询/过滤） | 证据 schema、四要素校验与 PoT 计算器（Phase 4） |
| 样例合成数据（无需 Kaggle 凭证即跑通） | Kaggle 自动下载器（无凭证环境） |
| 6 个规格测试 + 辅助测试 | 基准评测与 Agent 行为指标（Phase 7） |

## Decisions

| # | 决策 | 选择 | 被否决的替代方案 |
|---|---|---|---|
| 1 | 项目命名 | **VeriFin**，包名 `verifin` | 保持 PEV（缩写无辨识度，与已有项目易撞名） |
| 2 | PDF 解析库 | pymupdf 为主、pypdf 降级回退 | 仅用 pypdf（文本抽取质量弱、无表格能力） |
| 3 | token 估算 | `ceil(词数 × 1.3)` 的轻量估算函数，避免依赖重型 tokenizer | 引入 tiktoken（对金融英文文档收益低且增依赖）；字符/1.5（对英文偏松） |
| 4 | 分块策略 | 任意级别 Markdown 标题触发切分（保留祖先标题链），超 512 token 的节按段落二次切分 | 仅按 ##/### 切分（# 章节会被整段吞掉，超长）；不做二次切分（产生超限 chunk，违反验收标准） |
| 5 | 默认 embedding | 自研确定性哈希 embedding（256 维、离线、可测）；sentence-transformers(BGE) 走可选依赖 `[bge]` 经工厂注入 | 硬依赖 sentence-transformers/torch（安装数 GB，阻塞沙箱与 CI） |
| 6 | 元数据存储 | stdlib `sqlite3`，chunks 表（chunk_id 主键 + 元数据列 + JSON 扩展字段） | 纯 JSON 文件（过滤查询需全量扫描）；引入 SQLAlchemy（重） |
| 7 | BM25 持久化 | `pickle` 保存索引三元组（corpus/chunk_ids/BM25Okapi）+ 构建 manifest JSON | 仅内存（重启即失）；重写自有稀疏索引（重复造轮子） |
| 8 | 数据加载器 | 容错解析 ICAIF-24 常见字段（pre_text/post_text/table/html_text/text）及裸 txt/md/jsonl | 每种数据集写死一种格式的专用 loader（脆弱、难扩展） |
| 9 | 样例数据 | 内置约 10 篇合成财报节选（含标题层级与年份口径示例，专为四要素演示设计） | 依赖真实 Kaggle 数据跑通流程（本环境无凭证、无网络保证） |

## Surface

**公开 API（src/verifin）**：`schemas.py` 定义 `DocumentChunk` dataclass（chunk_id/doc_id/doc_name/content/page_num/section_title/section_level/metadata）与解析中间类型 `ParsedPage`；`ingestion/parser.py` 提供 `parse_file(path) -> list[ParsedPage]`（分页文本 + 每页标题行）；`ingestion/chunker.py` 提供 `chunk_documents(pages) -> list[DocumentChunk]`；`ingestion/loader.py` 提供 `load_dataset(data_dir, limit) -> list[ParsedDocument]`；`indexing/bm25_index.py` 提供 `BM25Index.build/search/save/load`；`indexing/vector_index.py` 提供 `VectorIndex.add_chunks/search/count`（基于 ChromaDB PersistentClient，collection 名 `verifin_chunks`）；`indexing/metadata_store.py` 提供 `MetadataStore.upsert_chunks/get/filter/close`；`embedding.py` 提供 `HashEmbedding` 与 `get_embedding_function(name)` 工厂。

**存储布局**：运行时产物都落在运行目录（被 .gitignore 排除）：`indexes/bm25.pkl`、`indexes/manifest.json`、`chroma_db/`（Chroma 持久化目录）、`verifin_metadata.db`（SQLite）。样例数据位于 `data/finqa/sample_1.jsonl`。

**组件要点**：chunker 维护「标题栈」以保留祖先标题链（如 `Financial Results > Operating Margins`）；对超过 512 token 的节按段落累计切分，切出的延续片记录 `"is_continuation": true`。BM25 索引同时保存 chunk 内容，为 Phase 2 的稀疏/稠密融合与 RRF 提供统一入口。元数据表包含 `doc_id/doc_name/page_num/section_title/section_level/metadata_json/content`，支撑「检索后按公司、年份过滤」的后续需求（与 Phase 4 证据存储同库扩展）。

**动过的文件**：`pyproject.toml`、`src/verifin/{__init__,schemas,embedding}.py`、`src/verifin/ingestion/{__init__,parser,chunker,loader}.py`、`src/verifin/indexing/{__init__,bm25_index,vector_index,metadata_store}.py`、`scripts/build_index.py`、`tests/test_ingestion.py`、`data/finqa/sample_1.jsonl`、`.gitignore`、两份设计文档。

## Risks & Open Questions

1. **依赖安装风险（中）**：chromadb 依赖链（onnxruntime 等）在沙箱可能安装失败或很慢。缓解：核心测试对 Chroma 用临时持久化路径；若沙箱网络彻底不可用，向量索引测试将改用 mock 或延迟到本机执行，并如实向用户报告（不开绿灯盲标通过）。
2. **token 估算偏差（低）**：词数 ×1.3 与真实 BPE token 有偏差。本 Phase 仅需满足 512 上限；Phase 2 引入 BGE tokenizer 做精确计算并在文档中标注口径。
3. **chromadb API 漂移（低）**：0.5.x 之后 `get_or_create_collection` 与 `embedding_function` 的签名有变化，实现时以实际安装版本为准，隔离在 vector_index 内部。
4. **开放问题**：真实 ICAIF-24 数据的字段结构以落地后的首个文件为准微调 loader；`metadata` 中是否预置「公司名/年份」抽取（决定放 Phase 4 Extractor，Phase 1 仅透传数据集自带字段）。