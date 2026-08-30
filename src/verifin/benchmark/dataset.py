"""评测数据集加载器（FinQA / ConvFinQA / TAT-QA）+ 内置合成语料回退。

设计要点（Phase 7 §五「容错」）：
- 真实数据集按 ICAIF-24 / FinQA 常用字段**容错解析**（question/exe_ans/qa 等字段名
  在不同版本间漂移，逐一尝试），解析不到即降级内置合成语料；
- 内置合成语料（20 个样本 + 23 个文档 chunk）使「无网络 / Kaggle 数据不可用」
  时端到端流水线依旧可跑通——面试演示零外部依赖；
- 合成样本按用途分组：单轮直接问答（含 % / 金额 / 无指标）、故意埋入的
  **增长类多跳题**（暴露计算工具未接入主循环的缺口）、**冲突样本**（新闻稿 vs
  年报大差异 → 重规划 → 权威口径）、**多轮组**（指代消解依赖偏好继承）。

统一样本格式::

    {
        "query": str,
        "ground_truth": str,           # "$12,000 million" / "15%" 等
        "gold_evidence": [{"doc_id": str, "chunk_id": str}],
        "conversation_id": Optional[str],  # 多轮同组共享
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Union

from loguru import logger

from verifin.schemas import DocumentChunk

PathLike = Union[str, Path]

_JSON_SUFFIXES = {".json", ".jsonl"}


class Dataset:
    """评测数据集最小协议：indexable + 可选语料构建。"""

    name: str = "base"

    def __init__(self, queries: List[dict]) -> None:
        self.queries = queries

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict:
        return self.queries[idx]

    def get_batch(self, indices: List[int]) -> List[dict]:
        return [self.queries[i] for i in indices]

    def build_chunks(self) -> List[DocumentChunk]:
        """返回可索引语料 chunk（真实数据集由 build_index.py 外部构建，返回空）。"""
        return []


# ---------------------------------------------------------------------------
# 内置合成语料（真实数据不可用时的离线展示回退）
# ---------------------------------------------------------------------------

def synth_chunks() -> List[DocumentChunk]:
    """构建合成金融语料（23 个 chunk）。

    4 家公司 × (年份 × 指标) 的标准财报片段；另含：
    - nt-news-revenue-2024：新闻稿口径（与年报大差异 → 冲突仲裁/重规划演示）；
    - nt-consolidated-revenue-2024：合并报表权威口径（重规划后的答案）；
    - helios-capex-2024：无指标命名的干扰文档（metrics 兜底路径演示）。
    """
    rows = [
        # (company, code, year, metric label, value)
        ("NovaTech", "nt", 2023, "revenue", "$10,000 million"),
        ("NovaTech", "nt", 2024, "revenue", "$12,000 million"),
        ("NovaTech", "nt", 2023, "operating margin", "13%"),
        ("NovaTech", "nt", 2024, "operating margin", "15%"),
        ("NovaTech", "nt", 2023, "gross margin", "40%"),
        ("NovaTech", "nt", 2024, "gross margin", "42%"),
        ("NovaTech", "nt", 2023, "net profit", "$1,500 million"),
        ("NovaTech", "nt", 2024, "net profit", "$1,800 million"),
        ("Orion", "or", 2023, "revenue", "$8,000 million"),
        ("Orion", "or", 2024, "revenue", "$8,500 million"),
        ("Orion", "or", 2024, "operating margin", "12%"),
        ("Orion", "or", 2024, "gross margin", "35%"),
        ("Orion", "or", 2024, "net profit", "$1,020 million"),
        ("Vertex", "vt", 2023, "revenue", "$19,000 million"),
        ("Vertex", "vt", 2024, "revenue", "$20,000 million"),
        ("Vertex", "vt", 2024, "operating margin", "18%"),
        ("Vertex", "vt", 2024, "gross margin", "44%"),
        ("Quantum", "qm", 2023, "revenue", "$4,500 million"),
        ("Quantum", "qm", 2024, "revenue", "$5,000 million"),
        ("Quantum", "qm", 2024, "operating margin", "9%"),
        ("Quantum", "qm", 2024, "gross margin", "30%"),
    ]
    chunks: List[DocumentChunk] = []
    for company, code, year, label, value in rows:
        key = label.replace(" ", "_")
        chunk_id = f"{code}-fy{year}-{key}"
        content = (
            f"{company} {label} in {year} was {value} "
            "according to the consolidated financial statements."
        )
        chunks.append(DocumentChunk(
            chunk_id=chunk_id, doc_id=chunk_id, doc_name=f"{code}-fy{year}.txt",
            content=content, metadata={"source_type": "filing", "year": year},
        ))
    # 新闻稿冲突口径（与年报 12,000 大差异 → major 冲突 → 重规划）
    news_id = "nt-news-revenue-2024"
    chunks.append(DocumentChunk(
        chunk_id=news_id, doc_id=news_id, doc_name="nt-press-release.txt",
        content="NovaTech press release reported revenue of $11,000 million in 2024.",
        metadata={"source_type": "news", "year": 2024},
    ))
    # 合并报表权威口径（重规划 consolidation_refinement 的命中目标）
    consolidated_id = "nt-consolidated-revenue-2024"
    chunks.append(DocumentChunk(
        chunk_id=consolidated_id, doc_id=consolidated_id, doc_name="nt-10k.txt",
        content=(
            "NovaTech 2024 consolidated revenue per Form 10-K was $12,000 million, "
            "audited under US GAAP."
        ),
        metadata={"source_type": "filing", "year": 2024},
    ))
    # 干扰文档：capex 不在指标词典内（无 metric 的兜底路径）
    helios_id = "helios-capex-2024"
    chunks.append(DocumentChunk(
        chunk_id=helios_id, doc_id=helios_id, doc_name="helios-news.txt",
        content="Helios Energy capex in 2024 was $900 million.",
        metadata={"source_type": "news", "year": 2024},
    ))
    return chunks


def _sample(query: str, ground_truth: str, chunk_id: str,
            conversation_id: Optional[str] = None) -> dict:
    return {
        "query": query,
        "ground_truth": ground_truth,
        "gold_evidence": [{"doc_id": chunk_id, "chunk_id": chunk_id}],
        "conversation_id": conversation_id,
    }


_SYNTHETIC_SAMPLES: List[dict] = [
    # ---- 单轮直接问答 ----
    _sample("What was NovaTech revenue in 2024?", "$12,000 million", "nt-fy2024-revenue"),
    _sample("What was NovaTech revenue in 2023?", "$10,000 million", "nt-fy2023-revenue"),
    _sample("What was Orion revenue in 2024?", "$8,500 million", "or-fy2024-revenue"),
    _sample("What was Vertex revenue in 2024?", "$20,000 million", "vt-fy2024-revenue"),
    _sample("What was Quantum revenue in 2024?", "$5,000 million", "qm-fy2024-revenue"),
    _sample("What was Quantum revenue in 2023?", "$4,500 million", "qm-fy2023-revenue"),
    _sample("What was NovaTech operating margin in 2024?", "15%", "nt-fy2024-operating_margin"),
    _sample("What was NovaTech gross margin in 2023?", "40%", "nt-fy2023-gross_margin"),
    _sample("What was NovaTech net profit in 2024?", "$1,800 million", "nt-fy2024-profit"),
    _sample("What was Quantum operating margin in 2024?", "9%", "qm-fy2024-operating_margin"),
    _sample("What was Vertex gross margin in 2024?", "44%", "vt-fy2024-gross_margin"),
    # 显式偏好指令（只看年报）
    _sample("只看年报，What was NovaTech revenue in 2024?", "$12,000 million", "nt-fy2024-revenue"),
    # 增长类多跳题：ground_truth 是计算产物，语料中无现成数值
    # → 暴露「PoT 计算工具未接入主循环」的缺口（错误分类 calculation_error）
    _sample("What was the percentage growth in NovaTech revenue from 2023 to 2024?",
            "20.0%", "nt-fy2024-revenue"),
    _sample("How much did Orion revenue increase from 2023 to 2024?",
            "6.25%", "or-fy2024-revenue"),
    # 指标词典外指标（capex）：无 metric claim 也应收敛到正确数值
    _sample("What was Helios Energy capex in 2024?", "$900 million", "helios-capex-2024"),
    # ---- 多轮组 A：指代消解依赖偏好继承（NovaTech）----
    _sample("What was NovaTech revenue in 2024?", "$12,000 million",
            "nt-fy2024-revenue", conversation_id="mt-a"),
    _sample("What about its gross margin?", "42%",
            "nt-fy2024-gross_margin", conversation_id="mt-a"),
    _sample("What about its operating margin?", "15%",
            "nt-fy2024-operating_margin", conversation_id="mt-a"),
    # ---- 多轮组 B：实体 + 指标继承（Orion）----
    _sample("What was Orion revenue in 2024?", "$8,500 million",
            "or-fy2024-revenue", conversation_id="mt-b"),
    _sample("And net profit?", "$1,020 million",
            "or-fy2024-profit", conversation_id="mt-b"),
]


class SyntheticDataset(Dataset):
    """内置合成评测集（离线可跑通的兜底语料）。"""

    name = "synthetic"
    is_synthetic = True  # 报告诚实声明用（语料来源标记）

    def __init__(self) -> None:
        super().__init__([dict(sample) for sample in _SYNTHETIC_SAMPLES])

    def build_chunks(self) -> List[DocumentChunk]:
        return synth_chunks()


# ---------------------------------------------------------------------------
# 真实数据集加载（容错解析）
# ---------------------------------------------------------------------------

def _read_records(path: Path) -> List[dict]:
    """JSON / JSONL 容错读取（整文件 JSON 或逐行 JSONL）。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, dict):
            return [data]
        return [item for item in data if isinstance(item, dict)]
    except json.JSONDecodeError:
        records: List[dict] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
            except json.JSONDecodeError:
                continue
        return records


def _find_data_file(data_dir: Path, split: str, prefix: str = "") -> Optional[Path]:
    """按候选命名找数据文件：<prefix><split>.json(l) → <split>.json(l) → 任意 json 文件。"""
    if not data_dir.is_dir():
        return None
    for suffix in _JSON_SUFFIXES:
        for name in (f"{prefix}{split}{suffix}", f"{split}{suffix}"):
            candidate = data_dir / name
            if candidate.is_file():
                return candidate
    return next(
        (path for path in sorted(data_dir.iterdir())
         if path.is_file() and path.suffix.lower() in _JSON_SUFFIXES),
        None,
    )


def _string_field(record: dict, *names: str) -> Optional[str]:
    """按字段名顺序取出第一个非空字符串值（不同数据集版本字段名漂移的容错）。"""
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return None


def _gold_answer_type(record: dict, answer: str) -> str:
    """FinQA 答案类型标注（Phase 8 多口径评测）：

    - ``boolean``：program 含大于/与/或比较，或答案本身为 yes/no；
    - ``extractive``：答案数值可在 gold 文档全文（pre_text+post_text）中
      以常见写法找到（单位归一 + 1% 容差）；
    - ``derived``：其余（program 算术推导，答案不出现在原文）。
    """
    qa = record.get("qa") if isinstance(record.get("qa"), dict) else {}
    program = str(qa.get("program") or qa.get("program_re") or "").lower()
    if (
        str(answer).strip().lower() in {"yes", "no"}
        or re.search(r"\b(greater|and|or)\s*\(", program)
    ):
        return "boolean"
    gold_lines = []
    for field in ("pre_text", "post_text"):
        value = record.get(field)
        if isinstance(value, list):
            gold_lines.extend(str(item) for item in value if item)
        elif value:
            gold_lines.append(str(value))
    try:
        from verifin.benchmark.metrics import answer_in_text  # 延迟导入避免环

        if answer_in_text(answer, "\n".join(gold_lines)):
            return "extractive"
    except ImportError:  # pragma: no cover - 导入必然可用
        pass
    return "derived"


def _parse_qa_record(record: dict, idx: int) -> Optional[dict]:
    """FinQA/ConvFinQA/TAT-QA 单条记录 → 统一样本格式（无法解析返回 None）。"""
    qa = record.get("qa") if isinstance(record.get("qa"), dict) else {}
    query = _string_field(qa, "question") or _string_field(record, "question", "query")
    answer = (
        _string_field(qa, "exe_ans", "answer", "ans")
        or _string_field(record, "exe_ans", "answer", "ans")
    )
    if not query or answer is None:
        return None
    doc_id = str(record.get("id") or record.get("doc_id") or f"{idx}")
    conversation_id = None
    for key in ("conversation_id", "dialog_id", "conv_id"):
        if record.get(key) is not None:
            conversation_id = str(record[key])
            break
    return {
        "query": query,
        "ground_truth": answer,
        "gold_evidence": [{"doc_id": doc_id, "chunk_id": None}],
        "conversation_id": conversation_id,
        "gold_answer_type": _gold_answer_type(record, answer),
    }


class FinQADataset(Dataset):
    """FinQA（ICAIF-24）评测集；解析不到任何样本时降级合成语料。"""

    name = "finqa"

    def __init__(self, data_dir: PathLike, split: str = "test") -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.is_synthetic = False
        super().__init__(self._load())

    def _load(self) -> List[dict]:
        path = _find_data_file(self.data_dir, self.split, prefix="finqa_")
        queries: List[dict] = []
        if path is not None:
            for idx, record in enumerate(_read_records(path)):
                sample = _parse_qa_record(record, idx)
                if sample is not None:
                    queries.append(sample)
        if not queries:
            logger.warning(
                "FinQA 数据不可用（{}），降级为内置合成语料（20 个样本）",
                self.data_dir,
            )
            self.is_synthetic = True
            return [dict(sample) for sample in _SYNTHETIC_SAMPLES]
        return queries

    def build_chunks(self) -> List[DocumentChunk]:
        if self.is_synthetic:
            return synth_chunks()
        # 真实语料由 scripts/build_index.py 预构建索引，此处不重复
        return []


class ConvFinQADataset(Dataset):
    """ConvFinQA 多轮对话评测集（按 conversation 聚合排序，保证同组轮次连续）。

    官方仓库数据位于 ``data/`` 子目录（dev_turn.json 等）：
    id 形如 ``Single_MRO/2007/page_134.pdf-1_0``（尾缀 ``_N`` 为轮次号），
    对话组 = 去掉尾缀的 doc id，与文档级文件（dev.json）的 id 一致。
    """

    name = "convfinqa"

    def __init__(self, data_dir: PathLike, split: str = "test") -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.is_synthetic = False
        super().__init__(self._load())

    def _load(self) -> List[dict]:
        path = self._find_turn_file() or _find_data_file(
            self.data_dir, self.split, prefix="convfinqa_"
        )
        queries: List[dict] = []
        if path is not None:
            for idx, record in enumerate(_read_records(path)):
                # 两种形态：qa_0/qa_1/...（同一 doc 内嵌多轮，如 Double_*）
                # 或 qa（单轮记录，多轮靠 id 尾缀 _N 区分）
                qa_keys = sorted(
                    (
                        key for key in record
                        if isinstance(key, str)
                        and re.fullmatch(r"qa(_\d+)?", key)
                        and isinstance(record.get(key), dict)
                    ),
                    key=lambda key: (0, 0) if key == "qa" else (1, int(key[3:])),
                )
                doc_id = str(record.get("id") or record.get("doc_id") or f"{idx}")
                base = re.sub(r"_\d+$", "", doc_id)
                if not qa_keys:
                    continue
                for turn, key in enumerate(qa_keys, start=1):
                    turn_record = dict(record)
                    turn_record["qa"] = record.get(key)
                    sample = _parse_qa_record(turn_record, idx)
                    if sample is None:
                        continue
                    if len(qa_keys) > 1:
                        # 内嵌多轮：qa_0=第 1 轮，qa_1=第 2 轮（"its ..." 指代依赖历史）
                        sample["conversation_id"] = base
                        sample["_turn"] = turn
                    elif base != doc_id:
                        # 单轮记录 + 尾缀 _N：轮次文件逐轮展开
                        sample["conversation_id"] = base
                    sample["gold_evidence"][0]["doc_id"] = base
                    queries.append(sample)
        if not queries:
            logger.warning("ConvFinQA 数据不可用，降级为内置合成语料")
            self.is_synthetic = True
            return [dict(sample) for sample in _SYNTHETIC_SAMPLES]
        # 稳定排序：同 conversation 轮次连续（多轮评估的连续性契约；
        # 同组内的轮次顺序保持文件原始顺序：qa_0 → qa_1 → … / _0 → _1 → …）
        return sorted(
            queries,
            key=lambda item: (
                item["conversation_id"] or "",
                item.get("_turn") or 0,
            ),
        )

    def _find_turn_file(self) -> Optional[Path]:
        """优先轮次文件（data/<split>_turn.json 或 <split>_turn.json），多轮评测用。"""
        candidates = [
            self.data_dir / "data" / f"{self.split}_turn.json",
            self.data_dir / f"{self.split}_turn.json",
            self.data_dir / "data" / f"{self.split}_turn.jsonl",
            self.data_dir / f"{self.split}_turn.jsonl",
        ]
        return next((path for path in candidates if path.is_file()), None)

    def build_chunks(self) -> List[DocumentChunk]:
        if self.is_synthetic:
            return synth_chunks()
        return []


class TATQADataset(Dataset):
    """TAT-QA 表格问答评测集（字段容错解析；降级合成语料）。"""

    name = "tatqa"

    def __init__(self, data_dir: PathLike, split: str = "test") -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.is_synthetic = False
        super().__init__(self._load())

    def _load(self) -> List[dict]:
        path = _find_data_file(self.data_dir, self.split, prefix="tatqa_")
        queries: List[dict] = []
        if path is not None:
            for idx, record in enumerate(_read_records(path)):
                sample = _parse_qa_record(record, idx)
                if sample is not None:
                    queries.append(sample)
        if not queries:
            logger.warning("TAT-QA 数据不可用，降级为内置合成语料")
            self.is_synthetic = True
            return [dict(sample) for sample in _SYNTHETIC_SAMPLES]
        return queries

    def build_chunks(self) -> List[DocumentChunk]:
        if self.is_synthetic:
            return synth_chunks()
        return []


_DATASET_CLASSES = {
    "finqa": FinQADataset,
    "convfinqa": ConvFinQADataset,
    "tatqa": TATQADataset,
    "synthetic": SyntheticDataset,
}


def load_dataset(name: str, data_dir: PathLike, split: str = "test") -> Dataset:
    """数据集工厂：``finqa`` / ``convfinqa`` / ``tatqa`` / ``synthetic`` /
    ``synthetic_multihop``（多跳合成基准，见 benchmark/multihop_synth.py）。"""
    if name == "synthetic_multihop":
        # 延迟导入：multihop_synth 依赖本模块的 synth_chunks（避免环）
        from verifin.benchmark.multihop_synth import MultihopSynthDataset

        return MultihopSynthDataset()  # type: ignore[return-value]
    if name not in _DATASET_CLASSES:
        raise ValueError(
            f"Unknown dataset: {name}（可用: {', '.join(_DATASET_CLASSES)}, synthetic_multihop）"
        )
    if name == "synthetic":
        return SyntheticDataset()
    return _DATASET_CLASSES[name](Path(data_dir), split)