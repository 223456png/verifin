#!/usr/bin/env python3
"""构建 VeriFin 索引（Phase 1 入口）。

用法：
    python scripts/build_index.py --dataset finqa --data-dir ./data/finqa

流程：加载 → 解析 → 分块 → BM25 + ChromaDB 向量索引 + SQLite 元数据 → 一致性自检 → manifest。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from verifin import __version__
from verifin.embedding import get_embedding_function
from verifin.indexing.bm25_index import BM25Index
from verifin.indexing.metadata_store import MetadataStore
from verifin.indexing.vector_index import VectorIndex
from verifin.ingestion.chunker import chunk_documents
from verifin.ingestion.loader import load_dataset
from verifin.ingestion.parser import parse_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default=None, help="数据集名称（写入 manifest，不影响加载行为）"
    )
    parser.add_argument("--data-dir", default="data/finqa", help="原始数据目录")
    parser.add_argument("--limit", type=int, default=None, help="最多加载 N 个文档")
    parser.add_argument(
        "--embedding", default="hash", choices=["hash", "bge"],
        help="向量索引 embedding（hash=离线默认；bge=需 [bge] 依赖）",
    )
    parser.add_argument(
        "--persist-dir", default=".",
        help="索引产物根目录（indexes/、chroma_db/、verifin_metadata.db 落于此）",
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="重建前清空旧的索引产物"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    persist = Path(args.persist_dir)
    indexes_dir = persist / "indexes"
    bm25_path = indexes_dir / "bm25.pkl"
    manifest_path = indexes_dir / "manifest.json"
    chroma_dir = persist / "chroma_db"
    db_path = persist / "verifin_metadata.db"

    if args.rebuild:
        logger.info("--rebuild：清空旧索引产物")
        shutil.rmtree(chroma_dir, ignore_errors=True)
        shutil.rmtree(indexes_dir, ignore_errors=True)
        indexes_dir.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            db_path.unlink()
    elif any(
        path.exists() for path in (chroma_dir, bm25_path, db_path)
    ):
        # chunk_id 每次构建都会重新生成，重复构建只会累积索引并触发一致性自检失败，
        # 因此检测到既有产物时快速失败并给出明确指引
        logger.error(
            "检测到既有索引产物（chroma_db/ indexes/ verifin_metadata.db）且未指定 --rebuild。"
            "重复构建会导致索引累积，请加 --rebuild 重建，或手动删除上述产物后重试"
        )
        return 1
    indexes_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据集
    documents = load_dataset(args.data_dir, limit=args.limit)
    if not documents:
        logger.error("数据集为空: {}", args.data_dir)
        return 1

    # 2. 解析文档
    parsed = parse_documents(documents)
    total_pages = sum(len(pages) for _, pages in parsed)
    logger.info("解析完成：{} 文档 / {} 页", len(documents), total_pages)

    # 3. 分块
    chunks = []
    for doc, pages in parsed:
        chunks.extend(
            chunk_documents(
                pages, doc_id=doc.doc_id, doc_name=doc.doc_name, metadata=doc.metadata
            )
        )
    logger.info(
        "分块完成：{} chunks / {} 文档 / {} 页", len(chunks), len(documents), total_pages
    )
    if not chunks:
        logger.error("分块结果为空")
        return 1

    # 4. 构建 BM25 + 向量索引 + 元数据存储
    bm25 = BM25Index()
    bm25.build(chunks)
    bm25.save(bm25_path)
    logger.info("BM25 索引已保存: {}", bm25_path)

    with MetadataStore(path=db_path) as store:
        store.upsert_chunks(chunks)
        metadata_count = len(store)
    logger.info("元数据已写入: {}", db_path)

    vector = VectorIndex(
        persist_dir=str(chroma_dir), embedding=get_embedding_function(args.embedding)
    )
    vector.add_chunks(chunks)
    logger.info("向量索引已写入: {}", chroma_dir)

    # 5. 一致性自检 + manifest
    counts = {"bm25": len(bm25), "metadata": metadata_count, "vector": vector.count()}
    logger.info("索引计数自检: {}", counts)
    if len(set(counts.values())) != 1:
        logger.error("索引一致性自检失败: {}", counts)
        return 1

    manifest = {
        "version": __version__,
        "dataset": args.dataset or args.data_dir,
        "embedding": args.embedding,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "doc_count": len(documents),
        "chunk_count": len(chunks),
        "indexes": {
            "bm25": "indexes/bm25.pkl",
            "vector": "chroma_db",
            "metadata": "verifin_metadata.db",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("构建完成，manifest 已写入 {}", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())