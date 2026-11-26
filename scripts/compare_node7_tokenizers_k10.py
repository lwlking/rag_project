"""在冻结 Chroma 副本上共享向量候选，离线对比 char 与 mixed 的前十块。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation_hashing import sha256_text_file

DATASET_ID = "zh_public_minibench_v0.1"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIMENSIONS = 2048
FINAL_K = 10
VECTOR_K = 20
BM25_K = 20
RRF_K = 60


def digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_frozen_inputs(
    questions: list[dict[str, Any]],
    query_vectors: dict[str, list[float]],
    documents: list[Any],
    catalog: dict[str, dict[str, str]],
) -> None:
    """在检索前拒绝错题、缺缓存或任何 ID、来源、正文不一致的语料。"""
    question_ids: set[str] = set()
    expected_queries: set[str] = set()
    for question in questions:
        question_id = question.get("question_id")
        query = question.get("query")
        if not question_id or question_id in question_ids:
            raise ValueError(f"评测题号重复或为空：{question_id}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"评测问题为空：{question_id}")
        question_ids.add(question_id)
        expected_queries.add(query)
    missing_queries = expected_queries - query_vectors.keys()
    if missing_queries:
        raise ValueError(f"缺少缓存向量：{sorted(missing_queries)[:3]}")
    extra_queries = query_vectors.keys() - expected_queries
    if extra_queries:
        raise ValueError(f"缓存包含本轮题集之外的问题：{sorted(extra_queries)[:3]}")

    actual: dict[str, dict[str, str]] = {}
    for document in documents:
        metadata = document.metadata or {}
        chunk_id = metadata.get("chunk_id")
        if not chunk_id or chunk_id in actual:
            raise ValueError(f"实际语料切块 ID 为空或重复：{chunk_id}")
        if not metadata.get("document_id"):
            raise ValueError(f"实际语料缺少来源文档 ID：{chunk_id}")
        actual[str(chunk_id)] = {
            "document_id": str(metadata["document_id"]),
            "text": document.page_content,
        }
    if actual != catalog:
        missing = sorted(catalog.keys() - actual.keys())
        unexpected = sorted(actual.keys() - catalog.keys())
        differing = sorted(
            chunk_id for chunk_id in actual.keys() & catalog.keys()
            if actual[chunk_id] != catalog[chunk_id]
        )
        raise ValueError(
            "Chroma 与冻结语料不一致："
            f"缺少切块={missing[:3]}，多余切块={unexpected[:3]}，来源或正文不符={differing[:3]}"
        )
    if not catalog:
        raise ValueError("冻结切块目录为空")


class SharedVectorSearch:
    """每个原始问题只执行一次有序向量召回，给两臂重放同一结果。"""

    def __init__(self, search: Callable[[str, int], list[Any]], vector_k: int) -> None:
        self.search = search
        self.vector_k = vector_k
        self.documents: dict[str, list[Any]] = {}
        self.elapsed_ms: dict[str, float] = {}

    def __call__(self, query: str, limit: int) -> list[Any]:
        if limit != self.vector_k:
            raise ValueError(f"向量候选数必须固定为 {self.vector_k}")
        if query not in self.documents:
            started = time.perf_counter()
            documents = self.search(query, self.vector_k)
            self.elapsed_ms[query] = round((time.perf_counter() - started) * 1000, 3)
            self.documents[query] = list(documents)
        return list(self.documents[query])


def _score_question(
    question: dict[str, Any],
    chunk_ids: list[str],
    catalog: dict[str, dict[str, str]],
    item_to_document: dict[str, str],
    final_k: int,
) -> dict[str, float]:
    from scripts.run_node7_end_to_end import retrieval_metrics

    metrics = retrieval_metrics(question, chunk_ids, catalog, item_to_document, final_k)
    if metrics is None:
        raise ValueError(f"可答题未生成检索指标：{question.get('question_id')}")
    return metrics


def score_ranked_documents(
    question: dict[str, Any],
    ranked_documents: list[Any],
    catalog: dict[str, dict[str, str]],
    item_to_document: dict[str, str],
    final_k: int = FINAL_K,
    score_fn: Callable[..., dict[str, float]] | None = None,
    diagnostic_requirements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """只评分实际返回的前 K 块；第 K+1 块不进入上下文，也不补齐短列表。"""
    if final_k < 1:
        raise ValueError("最终候选数量必须大于0")
    selected = ranked_documents[:final_k]
    ids = [str((document.metadata or {}).get("chunk_id", "")) for document in selected]
    if any(not chunk_id for chunk_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"检索结果包含空或重复切块 ID：{question.get('question_id')}")
    if set(ids) - catalog.keys():
        raise ValueError(f"检索结果包含未知冻结切块：{question.get('question_id')}")

    metrics = None
    if question["answerable"]:
        scoring = score_fn or _score_question
        metrics = scoring(question, ids, catalog, item_to_document, final_k)
    scope_document_present = None
    if not question["answerable"]:
        item_id = question.get("scope_document_id")
        if not item_id or item_id not in item_to_document:
            raise ValueError(f"拒答题缺少已锁定的目标资料：{question.get('question_id')}")
        target_document = item_to_document[item_id]
        scope_document_present = any(
            catalog[chunk_id]["document_id"] == target_document for chunk_id in ids
        )

    row: dict[str, Any] = {
        "question_id": question["question_id"],
        "question_type": question.get("question_type"),
        "evaluation_suite": question["evaluation_suite"],
        "answerable": bool(question["answerable"]),
        "ranked_chunk_ids": ids,
        "sources": [
            {
                "chunk_id": chunk_id,
                "document_id": catalog[chunk_id]["document_id"],
                "source": (document.metadata or {}).get("source"),
            }
            for chunk_id, document in zip(ids, selected)
        ],
        "metrics": metrics,
        "scope_document_present": scope_document_present,
    }
    if diagnostic_requirements:
        from scripts.audit_node7_evidence import match_requirement

        selected_catalog_rows = [
            {"chunk_id": chunk_id, **catalog[chunk_id]} for chunk_id in ids
        ]
        row["diagnostic_evidence_hits"] = {
            requirement["id"]: bool(match_requirement(
                requirement, selected_catalog_rows, item_to_document,
            ))
            for requirement in diagnostic_requirements
        }
        row["diagnostic_complete"] = all(row["diagnostic_evidence_hits"].values())
    return row


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] * (1 - (position - lower)) + ordered[upper] * (position - lower), 3)


def _summarize(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows.values() if row["answerable"]]
    refusals = [row for row in rows.values() if not row["answerable"]]
    suites: dict[str, Any] = {}
    for suite in sorted({row["evaluation_suite"] for row in rows.values()}):
        suite_rows = [row for row in answerable if row["evaluation_suite"] == suite]
        metric_names = sorted({
            key for row in suite_rows for key in (row["metrics"] or {})
            if key != "first_relevant_rank"
        })
        complete_key = (
            f"complete_evidence_at_{FINAL_K}" if suite == "acceptance_23"
            else f"complete_evidence_hit_at_{FINAL_K}"
        )
        suites[suite] = {
            "answerable_count": len(suite_rows),
            "complete_evidence_count": sum(
                bool(row["metrics"].get(complete_key)) for row in suite_rows
            ),
            "mean_metrics": {
                key: round(statistics.fmean(row["metrics"][key] for row in suite_rows
                                            if key in row["metrics"]), 6)
                for key in metric_names
                if any(key in row["metrics"] for row in suite_rows)
            },
        }
    return {
        "question_count": len(rows),
        "answerable_count": len(answerable),
        "refusal_count": len(refusals),
        "refusal_scope_targets_present": sum(
            row["scope_document_present"] is True for row in refusals
        ),
        "suites": suites,
        "mean_context_tokens": round(statistics.fmean(
            row["estimated_context_tokens"] for row in rows.values()
        ), 3) if rows else 0.0,
        "retrieval_elapsed_ms": {
            "p50": _percentile([row["estimated_retrieval_elapsed_ms"] for row in rows.values()], 0.5),
            "p95": _percentile([row["estimated_retrieval_elapsed_ms"] for row in rows.values()], 0.95),
        },
        "latency_note": "共享向量查询耗时计入两臂；BM25/RRF 耗时单独记录，便于配对比较。",
    }


def _default_retriever_factory(**kwargs: Any) -> Any:
    from retrieval.hybrid_retriever import HybridRetrieverService

    return HybridRetrieverService(**kwargs)


def compare_tokenizers(
    questions: list[dict[str, Any]],
    documents: list[Any],
    catalog: dict[str, dict[str, str]],
    item_to_document: dict[str, str],
    vector_search: Callable[[str, int], list[Any]],
    *,
    count_tokens: Callable[[str], int] | None = None,
    score_fn: Callable[..., dict[str, float]] | None = None,
    retriever_factory: Callable[..., Any] | None = None,
    diagnostic_cases: dict[str, list[dict[str, Any]]] | None = None,
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在相同题目、切块和向量候选上，只改变 BM25 分词器。"""
    ids = [question.get("question_id") for question in questions]
    if not questions or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("评测题集为空、题号为空或题号重复")
    if not catalog or len({doc.metadata.get("chunk_id") for doc in documents}) != len(documents):
        raise ValueError("实际语料为空或存在重复切块 ID")
    if retriever_factory is None:
        retriever_factory = _default_retriever_factory
    if count_tokens is None:
        from ingestion.chunking import TokenCounter

        count_tokens = TokenCounter().count

    settings = {
        "vector_k": VECTOR_K,
        "bm25_k": BM25_K,
        "rrf_k": RRF_K,
        "final_k": FINAL_K,
        "retrieval_strategy": "hybrid_rrf",
        "query_resolution_strategy": "original",
        "rerank_strategy": "none",
        "evidence_expansion": "none",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
    }
    shared_vector = SharedVectorSearch(vector_search, VECTOR_K)
    for question in questions:
        shared_vector(question["query"], VECTOR_K)

    retrievers = {
        tokenizer: retriever_factory(
            get_vector_docs=shared_vector,
            get_all_docs=lambda: documents,
            get_document_count=lambda: len(documents),
            vector_k=VECTOR_K,
            bm25_k=BM25_K,
            final_k=FINAL_K,
            rrf_k=RRF_K,
            strategy_id="hybrid_rrf",
            tokenizer_id=tokenizer,
        )
        for tokenizer in ("char", "mixed")
    }
    for retriever in retrievers.values():
        warm_bm25 = getattr(retriever, "_ensure_bm25", None)
        if callable(warm_bm25):
            warm_bm25()

    arms: dict[str, dict[str, Any]] = {}
    for tokenizer, retriever in retrievers.items():
        rows: dict[str, dict[str, Any]] = {}
        for question in questions:
            query = question["query"]
            started = time.perf_counter()
            candidates = retriever.get_candidates(query, FINAL_K)
            bm25_rrf_ms = round((time.perf_counter() - started) * 1000, 3)
            row = score_ranked_documents(
                question, candidates, catalog, item_to_document,
                score_fn=score_fn,
                diagnostic_requirements=(diagnostic_cases or {}).get(question["question_id"]),
            )
            row.update({
                "estimated_context_tokens": sum(
                    count_tokens(catalog[chunk_id]["text"]) for chunk_id in row["ranked_chunk_ids"]
                ),
                "shared_vector_elapsed_ms": shared_vector.elapsed_ms[query],
                "bm25_rrf_elapsed_ms": bm25_rrf_ms,
                "estimated_retrieval_elapsed_ms": round(
                    shared_vector.elapsed_ms[query] + bm25_rrf_ms, 3,
                ),
            })
            rows[question["question_id"]] = row
        arms[tokenizer] = {
            "tokenizer": tokenizer,
            "identity": identity or {},
            "settings": settings,
            "rows": rows,
            "summary": _summarize(rows),
        }

    return {
        "dataset_id": DATASET_ID,
        "identity": identity or {},
        "settings": settings,
        "dense_candidates": {
            query: [str(doc.metadata["chunk_id"]) for doc in docs]
            for query, docs in shared_vector.documents.items()
        },
        "arms": arms,
    }


def paired_metric_deltas(
    left: dict[str, Any],
    right: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    """固定输入或参数不同的两份报告不得计算 A/B 差值。"""
    if left.get("identity") != right.get("identity") or left.get("settings") != right.get("settings"):
        raise ValueError("固定实验参数不一致，禁止计算配对差值")
    left_rows, right_rows = left.get("rows", {}), right.get("rows", {})
    if set(left_rows) != set(right_rows):
        raise ValueError("两份报告的题目集合不一致，禁止计算配对差值")
    per_question: dict[str, float] = {}
    for question_id in left_rows:
        old = (left_rows[question_id].get("metrics") or {}).get(metric)
        new = (right_rows[question_id].get("metrics") or {}).get(metric)
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            per_question[question_id] = round(float(new) - float(old), 6)
    return {
        "metric": metric,
        "compared_count": len(per_question),
        "improved": [key for key, value in per_question.items() if value > 0],
        "regressed": [key for key, value in per_question.items() if value < 0],
        "unchanged": [key for key, value in per_question.items() if value == 0],
        "per_question": per_question,
    }


def _check_runtime_config(config: Any, storage_root: Path, chroma_root: Path) -> None:
    required_env = ("RAG_STORAGE_ROOT", "CHROMA_PERSIST_DIRECTORY")
    missing = [key for key in required_env if not os.getenv(key)]
    if missing:
        raise ValueError(f"必须显式设置隔离评测路径：{', '.join(missing)}")
    storage_root, chroma_root = storage_root.resolve(), chroma_root.resolve()
    if not chroma_root.is_relative_to(storage_root):
        raise ValueError("Chroma 副本必须位于隔离 RAG_STORAGE_ROOT 内")
    expected = {
        "vector_store_backend": "chroma",
        "chroma_mode": "local",
        "collection_name": "rag_text_embedding_v4_2048",
        "chroma_hnsw_ef_search": 200,
        "embedding_model_name": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "retrieval_strategy": "hybrid_rrf",
        "query_resolution_strategy": "original",
        "rerank_strategy": "none",
        "evidence_expansion": "none",
        "hybrid_vector_k": VECTOR_K,
        "hybrid_bm25_k": BM25_K,
        "hybrid_final_k": FINAL_K,
        "hybrid_rrf_k": RRF_K,
        "retrieval_source_filter": "",
    }
    mismatches = {
        key: (getattr(config, key), value)
        for key, value in expected.items() if getattr(config, key) != value
    }
    if Path(config.persist_directory).resolve() != chroma_root:
        mismatches["persist_directory"] = (config.persist_directory, str(chroma_root))
    if mismatches:
        raise ValueError(f"当前配置与冻结检索实验不符：{mismatches}")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    from core import config
    from infra.vector_store import VectorStoreService
    from infra.vector_stores.chroma import ChromaVectorStoreAdapter
    from ingestion.document_model import stable_document_id
    from ingestion.storage import StorageLayout
    from scripts.evaluate_embedded_retrieval import load_chunk_catalog
    from scripts.run_node7_end_to_end import load_questions_and_vectors
    from scripts.run_node6_retrieval_baseline import CachedQuestionEmbeddings
    from ingestion.chunking import TokenCounter

    storage_root = Path(os.environ.get("RAG_STORAGE_ROOT", ""))
    chroma_root = Path(os.environ.get("CHROMA_PERSIST_DIRECTORY", ""))
    _check_runtime_config(config, storage_root, chroma_root)

    bundle = ROOT / "data" / "eval" / DATASET_ID
    evaluation = StorageLayout.from_env(ROOT).evaluation(DATASET_ID)
    annotation_dir = bundle / "annotations"
    questions, vectors = load_questions_and_vectors(bundle, evaluation)
    question_hashes: dict[str, str] = {}
    for filename in ("retrieval_questions.jsonl", "acceptance_questions_v1.jsonl"):
        question_hashes[filename] = sha256_text_file(annotation_dir / filename)
    if (len(questions) != 78
            or sum(q["evaluation_suite"] == "retrieval_55" and q["answerable"] for q in questions) != 55
            or sum(q["evaluation_suite"] == "acceptance_23" and q["answerable"] for q in questions) != 19
            or sum(not q["answerable"] for q in questions) != 4):
        raise ValueError("冻结题集应包含55道旧可答题、19道新增可答题和4道拒答题")

    manifest = read_jsonl(bundle / "manifest.jsonl")
    item_to_document = {
        row["item_id"]: stable_document_id(row["sha256"]) for row in manifest
    }
    catalog_root = evaluation.chunks / "structure_aware_v1"
    catalog = load_chunk_catalog(catalog_root)
    catalog_digest = digest(catalog)
    chunks_digest = hashlib.sha256()
    for path in sorted(catalog_root.rglob("chunks.jsonl")):
        chunks_digest.update(path.relative_to(catalog_root).as_posix().encode("utf-8"))
        chunks_digest.update(sha256_text_file(path).encode("ascii"))

    adapter = ChromaVectorStoreAdapter(CachedQuestionEmbeddings(vectors))
    service = VectorStoreService(adapter=adapter)
    documents = service.get_all_documents()
    validate_frozen_inputs(questions, vectors, documents, catalog)
    if service.count() != len(catalog):
        raise ValueError(f"Chroma 切块数不符：{service.count()}/{len(catalog)}")
    for question in questions:
        for item_id in (
            question.get("relevant_document_ids", [])
            + [item for group in question.get("relevant_document_groups", []) for item in group]
            + [item for evidence in question.get("evidence", []) for item in evidence.get("document_ids", [])]
        ):
            if item_id not in item_to_document:
                raise ValueError(f"题目引用的来源不在冻结清单中：{question['question_id']} / {item_id}")
        if not question["answerable"] and question.get("scope_document_id") not in item_to_document:
            raise ValueError(f"拒答题的目标资料不在冻结清单中：{question['question_id']}")

    audit_path = annotation_dir / "node7_evidence_audit_v1.json"
    audit_spec = json.loads(audit_path.read_text(encoding="utf-8"))
    # 锁定先前的 8 题问题级诊断；排除 12 道已到位题和 omni-05 上游缺证题。
    diagnostic_ids = {
        "omni-09-01",
        "ret-te-2013-01",
        "wb-328-01-02",
        "wb-37-01-02",
        "wb-cross-03",
        "wb-cross-04",
        "accept-v1-x01",
        "accept-v1-x04",
    }
    if not diagnostic_ids <= {question["question_id"] for question in questions}:
        raise ValueError("冻结题集中缺少先前核定的8道漏证据诊断题")
    if not diagnostic_ids <= audit_spec.get("evidence_cases", {}).keys():
        raise ValueError("证据审计附表缺少先前核定的漏证据题")
    diagnostic_cases = {
        question_id: case["requirements"]
        for question_id, case in audit_spec["evidence_cases"].items()
        if question_id in diagnostic_ids
    }
    identity = {
        "protocol": "node7-tokenizer-ab-k10-v1",
        "dataset_id": DATASET_ID,
        "question_sha256": question_hashes,
        "manifest_sha256": sha256_text_file(bundle / "manifest.jsonl"),
        "catalog_sha256": catalog_digest,
        "chunks_sha256": chunks_digest.hexdigest(),
        "query_vectors_sha256": digest(vectors),
        "diagnostic_spec_sha256": sha256_text_file(audit_path),
        "collection": config.collection_name,
        "chroma_copy": str(chroma_root),
        "retrieval_code_sha256": {
            name: sha256_text_file(ROOT / name)
            for name in (
                "retrieval/hybrid_retriever.py",
                "retrieval/tokenizers.py",
                "scripts/evaluate_node6_acceptance.py",
                "scripts/evaluate_embedded_retrieval.py",
                Path(__file__).relative_to(ROOT).as_posix(),
            )
        },
    }
    report = compare_tokenizers(
        questions,
        documents,
        catalog,
        item_to_document,
        service.get_vector_docs,
        count_tokens=TokenCounter().count,
        diagnostic_cases=diagnostic_cases,
        identity=identity,
    )
    metric_names = sorted({
        key
        for arm in report["arms"].values()
        for row in arm["rows"].values()
        for key in (row["metrics"] or {})
        if key != "first_relevant_rank"
    })
    report["paired_deltas"] = {
        metric: paired_metric_deltas(
            report["arms"]["char"], report["arms"]["mixed"], metric,
        )
        for metric in metric_names
    }
    report["diagnostic_summary"] = {
        tokenizer: {
            "covered_questions": sum("diagnostic_complete" in row for row in arm["rows"].values()),
            "requirements_completed": sum(
                sum(row.get("diagnostic_evidence_hits", {}).values())
                for row in arm["rows"].values()
            ),
            "complete_question_ids": [
                key for key, row in arm["rows"].items() if row.get("diagnostic_complete")
            ],
        }
        for tokenizer, arm in report["arms"].items()
    }
    report["model_requests_this_run"] = 0
    report["scoring_note"] = (
        "拒答题不进入可答题检索分母；前10块的检索指标不代表回答正确率。"
        "上下文token使用cl100k_base估算；原8题另按预先核定的最小原文证据诊断。"
    )
    fingerprint = digest(identity)[:12]
    output = evaluation.reports / f"node7_tokenizers_k10_{fingerprint}.json"
    _write_json_atomic(output, report)
    print(json.dumps({
        "report": str(output),
        "identity": fingerprint,
        "summary": {
            tokenizer: arm["summary"] for tokenizer, arm in report["arms"].items()
        },
        "diagnostic_summary": report["diagnostic_summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
