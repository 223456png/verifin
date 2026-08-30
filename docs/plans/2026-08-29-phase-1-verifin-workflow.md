---
intent: 为 VeriFin 搭建数据基座：PDF/文本解析 → 按标题层级分块 → 构建 BM25 索引、ChromaDB 向量索引与 SQLite 元数据存储，并用测试与端到端构建脚本验收
success_criteria: tests/test_ingestion.py 中 6 个规格测试全部通过；scripts/build_index.py 在 data/finqa 样例数据上端到端跑通（退出码 0），产出 indexes/bm25.pkl、indexes/manifest.json、chroma_db/ 与 verifin_metadata.db
risk_level: low
auto_approve: true
worktree: false
---

# Phase 1 — VeriFin Ingestion + Indexing 工作流

> 项目根目录：/workspace/verifin。环境事实：网络可用（代理 egress）；uv 0.7.8 与 pyenv Python 3.12.13 已就绪。
> 所有 Python 命令均通过 `.venv/bin/python` 执行；token 估算函数为 `ceil(词数 × 1.3)`，上限 512。

## Steps

- [x] **Step 1: 初始化项目骨架**
action: 在 /workspace/verifin 下执行 `git init -b main`（不修改任何 git config、不做 commit）；写入 `.gitignore`（排除 .venv/、indexes/、chroma_db/、*.db、__pycache__/、*.pyc、.pytest_cache/）；写入 `pyproject.toml`（项目名 verifin、requires-python >=3.10、依赖 pypdf/pymupdf/chromadb/rank-bm25/pydantic/loguru、dev extra 含 pytest、[bge] extra 含 sentence-transformers、setuptools src 布局、packages 指向 src/verifin）；创建目录 src/verifin/ingestion、src/verifin/indexing、tests、data/finqa、scripts、docs/plans，并在 src/verifin、src/verifin/ingestion、src/verifin/indexing 下写入空的 `__init__.py`
verify:
  - type: artifact
    path: pyproject.toml
    assert:
      kind: exists
  - type: artifact
    path: src/verifin/ingestion/__init__.py
    assert:
      kind: exists
  - type: artifact
    path: src/verifin/indexing/__init__.py
    assert:
      kind: exists

- [x] **Step 2: 创建 Python 3.12 虚拟环境并安装依赖**
action: 在 /workspace/verifin 下执行 `uv venv --python 3.12 .venv`，然后执行 `uv pip install --python .venv/bin/python -e ".[dev]"`；若 uv 解析失败则执行 `.venv/bin/python -m pip install -e ".[dev]"`
verify: .venv/bin/python -c "import chromadb, rank_bm25, pypdf, pydantic, loguru, pytest; import importlib.util, importlib; m = importlib.import_module('pymupdf') if importlib.util.find_spec('pymupdf') else importlib.import_module('fitz'); print('deps-ok', m.__name__)"
loop: until 确认输出以 deps-ok 开头
max_iterations: 3

- [x] **Step 3: 编写规格测试 tests/test_ingestion.py（TDD RED 基线）**
action: 编写 tests/test_ingestion.py，内容包含：(1) test_parser_pdf —— pytest fixture 用 pymupdf 在 tmp_path 生成含多行文本的示例 PDF，parse_file 后断言输出非空且含文本；(2) test_chunker_by_heading —— 输入带 #/##/### 标题的文本，断言 chunk 数、section_title 与 section_level 正确；(3) test_chunker_max_tokens —— 输入超长文本，断言每个 chunk 的估算 token 数 ≤ 512；(4) test_bm25_index_build —— build 后断言索引与 chunk_ids 非空；(5) test_bm25_search —— 断言返回结果数 == top_k；(6) test_vector_index_add_search —— 用 HashEmbedding（离线确定性 256 维）构建 VectorIndex（临时目录 PersistentClient），add_chunks 后 search 断言返回正确 chunk_id。测试文件顶部 from verifin... 导入保持与设计文档 API 一致（此步后 pytest 必然 RED，属预期）
verify:
  - type: artifact
    path: tests/test_ingestion.py
    assert:
      kind: exists

- [x] **Step 4: 实现 src/verifin/schemas.py、embedding.py 与 ingestion/parser.py**
action: schemas.py 定义 DocumentChunk dataclass（chunk_id/doc_id/doc_name/content/page_num: Optional[int]/section_title: Optional[str]/section_level: Optional[int]/metadata: dict）与 ParsedPage、ParsedDocument dataclass；embedding.py 实现 HashEmbedding（chromadb EmbeddingFunction 接口，SHA256 哈希投影为 256 维 L2 归一，无网络）与 get_embedding_function(name)；parser.py 实现 parse_file(path) -> list[ParsedPage]：.pdf 优先 import pymupdf（回退 import fitz）逐页抽取文本并识别页内行首 Markdown 标题，其余扩展名按 UTF-8 文本读取并按空行切页，同时实现 parse_documents(docs) 批处理入口
verify: .venv/bin/python -m pytest tests/test_ingestion.py -k parser -x -q
loop: until 退出码 0 且输出 passed
max_iterations: 5

- [x] **Step 5: 实现 src/verifin/ingestion/chunker.py**
action: 实现 chunk_documents(pages, doc_id, doc_name, metadata) -> list[DocumentChunk]：用正则识别任意级别 Markdown 标题（#~######）触发切分；维护标题栈保留祖先标题链（形如 "Financial Results > Operating Margins" 存入 section_title，section_level 取末段层级）；estimate_tokens(text) = ceil(词数 × 1.3)，MAX_TOKENS = 512；超过上限的节按段落累计二次切分，延续片 metadata 记录 "is_continuation": true；chunk_id 用 uuid4().hex；页码透传自 ParsedPage
verify: .venv/bin/python -m pytest tests/test_ingestion.py -k chunker -x -q
loop: until 退出码 0 且输出 passed
max_iterations: 5

- [x] **Step 6: 实现 src/verifin/indexing/bm25_index.py**
action: 实现 BM25Index：build(chunks) 以 rank_bm25.BM25Okapi 构建（_tokenize 用小写字母数字词正则分词）；search(query, top_k=10) 返回 [(chunk_id, score)] 按分降序取 top_k，空索引返回 []；save(path)/load(path) 用 pickle 持久化 (corpus, chunk_ids, index)；__len__ 返回 chunk 数
verify: .venv/bin/python -m pytest tests/test_ingestion.py -k bm25 -x -q
loop: until 退出码 0 且输出 passed
max_iterations: 5

- [x] **Step 7: 实现 src/verifin/indexing/metadata_store.py 与 vector_index.py**
action: metadata_store.py 用 stdlib sqlite3 实现 MetadataStore(path="verifin_metadata.db")：表 chunks(chunk_id TEXT PRIMARY KEY, doc_id, doc_name, page_num INTEGER, section_title, section_level INTEGER, metadata_json TEXT, content TEXT)，提供 upsert_chunks/get/filter(doc_id, page_num, section_title)/close/__len__；vector_index.py 实现 VectorIndex(persist_dir="chroma_db", collection_name="verifin_chunks", embedding=None)：基于 chromadb.PersistentClient 的 get_or_create_collection，默认 HashEmbedding，add_chunks 以 chunk_id 为 id、content 为 document、metadata 存 doc_id/page_num/section_title/section_level 与 metadata_json，search(query, top_k) 返回 [(chunk_id, distance)]，count() 返回集合数量
verify: .venv/bin/python -m pytest tests/test_ingestion.py -k vector_index -x -q
loop: until 退出码 0 且输出 passed
max_iterations: 5

- [x] **Step 8: 实现 src/verifin/ingestion/loader.py 并编写样例数据**
action: loader.py 实现 load_dataset(data_dir, limit=None) -> list[ParsedDocument]：扫描目录下 .json/.jsonl/.txt/.md；jsonl/json 逐条容错解析 ICAIF-24 常见字段（html_text/pre_text+post_text+table_ori/text/content），取到文本后按换行合成 Markdown；裸 txt/md 直接读取；ParsedDocument 含 doc_id/doc_name/raw_text/format/metadata，doc_id 取记录 id 或文件名 hash。编写 data/finqa/sample_1.jsonl：>=10 条合成财报节选文档，使用虚构公司名（如 NovaTech/BlueRiver Capital），每条含 Markdown 标题层级（## Financial Results、### Revenue、### Operating Margin 等）与不同年份/口径（GAAP vs Non-GAAP）示例，字段采用 `{"id": ..., "html_text": ...}` 的 ICAIF 风格
verify: .venv/bin/python -c "from verifin.ingestion.loader import load_dataset; docs = load_dataset('data/finqa'); assert len(docs) >= 10; print('docs-ok', len(docs))"
loop: until 退出码 0 且输出以 docs-ok 开头
max_iterations: 5

- [x] **Step 9: 实现 scripts/build_index.py 并端到端构建**
action: 编写脚本：argparse 参数 --data-dir（默认 data/finqa）、--limit（默认 None）、--persist-dir（默认 . 即项目根，供 indexes/chroma_db/verifin_metadata.db 落盘）、--rebuild（重建时清空旧索引）；流程 = load_dataset → parse_documents → chunk_documents → BM25Index.build + save(indexes/bm25.pkl) → MetadataStore.upsert_chunks → VectorIndex.add_chunks → 写 indexes/manifest.json（chunk 总数、文档数、构建时间、embedding 类型、版本）；loguru 打点每步数量；结束后自检 BM25 数量 == 元数据行数 == Chroma count 并打印，不一致则退出码 1。执行 `.venv/bin/python scripts/build_index.py --data-dir data/finqa`
verify: .venv/bin/python scripts/build_index.py --data-dir data/finqa --rebuild && test -f indexes/bm25.pkl && test -f indexes/manifest.json
loop: until 退出码 0
max_iterations: 5

- [x] **Step 10: 全量验收**
action: 在 /workspace/verifin 下执行完整测试套件，确认 6 个规格测试全部通过且无 error；同时执行 `.venv/bin/python -m compileall -q src/verifin` 确认无语法问题
verify:
  - type: shell
    command: .venv/bin/python -m pytest tests/test_ingestion.py -v
  - type: shell
    command: .venv/bin/python -m compileall -q src/verifin
loop: until 退出码 0 且 pytest 输出 6 passed
max_iterations: 3

- [x] **Step 11: 代码审查与人工验收**
action: 对本 Phase 全部新增代码运行 HOTL code-review 流程（对照 Phase 1 设计文档的 Intent/Verification 契约与验收标准）；修复审查发现的阻断问题并重跑 Step 10；最后向用户汇报：测试结果、端到端构建产物清单（indexes/bm25.pkl、indexes/manifest.json、chroma_db/、verifin_metadata.db）、与设计契约逐项对照结论
gate: human