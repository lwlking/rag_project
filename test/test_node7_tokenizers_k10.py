"""配对 @10 分词评测的冻结输入、边界与可比性检查。"""

import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.compare_node7_tokenizers_k10 import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    _check_runtime_config,
    compare_tokenizers,
    paired_metric_deltas,
    score_ranked_documents,
    validate_frozen_inputs,
)
from retrieval.embedding_cache import EmbeddingCache
from scripts.evaluation_hashing import sha256_text_file
from scripts.run_node7_end_to_end import load_questions_and_vectors


class FakeDocument:
    def __init__(self, chunk_id: str, document_id: str, text: str) -> None:
        self.page_content = text
        self.id = None
        self.metadata = {
            "chunk_id": chunk_id, "document_id": document_id, "source": f"{document_id}.md",
        }


class TokenizerK10Tests(unittest.TestCase):
    def document(self, chunk_id: str, document_id: str, text: str) -> FakeDocument:
        return FakeDocument(chunk_id, document_id, text)

    def acceptance_question(self, question_id: str = "q1", query: str = "查询") -> dict:
        return {
            "question_id": question_id,
            "query": query,
            "answerable": True,
            "evaluation_suite": "acceptance_23",
            "relevant_document_groups": [["target"]],
            "evidence": [{
                "document_ids": ["target"],
                "quote": "唯一证据",
                "locator": {"type": "line", "line": 1},
            }],
        }

    def test_rank_ten_evidence_counts_but_rank_eleven_does_not(self) -> None:
        distractors = [self.document(f"c{i}", f"d{i}", f"干扰内容{i}") for i in range(1, 10)]
        target = self.document("target-10", "doc-target", "唯一证据")
        mapping = {"target": "doc-target"}
        catalog_at_ten = {doc.metadata["chunk_id"]: {
            "document_id": doc.metadata["document_id"], "text": doc.page_content,
        } for doc in [*distractors, target]}

        at_ten = score_ranked_documents(
            self.acceptance_question(), [*distractors, target],
            catalog_at_ten, mapping, final_k=10,
        )
        self.assertEqual(at_ten["metrics"]["complete_evidence_at_10"], 1)

        after_ten = self.document("target-11", "doc-target", "唯一证据")
        ranked = [*distractors, self.document("c10", "d10", "干扰内容10"), after_ten]
        catalog_at_eleven = {doc.metadata["chunk_id"]: {
            "document_id": doc.metadata["document_id"], "text": doc.page_content,
        } for doc in ranked}
        at_eleven = score_ranked_documents(
            self.acceptance_question(), ranked, catalog_at_eleven, mapping, final_k=10,
        )
        self.assertEqual(at_eleven["metrics"]["complete_evidence_at_10"], 0)
        self.assertEqual(len(at_eleven["ranked_chunk_ids"]), 10)

    def test_short_retrieval_result_is_not_padded(self) -> None:
        docs = [self.document("c1", "d1", "内容")]
        row = score_ranked_documents(
            self.acceptance_question(), docs,
            {"c1": {"document_id": "d1", "text": "内容"}},
            {"target": "d1"}, final_k=10,
            score_fn=lambda *_args: {},
        )
        self.assertEqual(row["ranked_chunk_ids"], ["c1"])

    def test_missing_query_vector_fails_before_retrieval(self) -> None:
        question = self.acceptance_question()
        doc = self.document("c1", "d1", "内容")
        with self.assertRaisesRegex(ValueError, "缺少缓存向量"):
            validate_frozen_inputs([question], {}, [doc], {
                "c1": {"document_id": "d1", "text": "内容"},
            })

    def test_missing_sqlite_cache_fails_without_creating_or_refilling_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle, evaluation, _questions = self.make_loader_inputs(Path(directory))
            with self.assertRaisesRegex(ValueError, "问题向量缓存不存在，不允许在线补齐"):
                load_questions_and_vectors(bundle, evaluation)
            self.assertFalse((evaluation.embeddings / f"{EMBEDDING_MODEL}-{EMBEDDING_DIMENSIONS}"
                              / "question_embeddings.sqlite3").exists())

    def test_stale_sqlite_cache_fails_instead_of_requesting_a_new_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle, evaluation, questions = self.make_loader_inputs(Path(directory))
            cache_root = evaluation.embeddings / f"{EMBEDDING_MODEL}-{EMBEDDING_DIMENSIONS}"
            cache_root.mkdir(parents=True)
            retrieval_cache = cache_root / "question_embeddings.sqlite3"
            acceptance_cache = cache_root / "acceptance_v1_embeddings.sqlite3"
            with EmbeddingCache(retrieval_cache) as cache:
                retrieval_question = questions[0]
                cache.put_many([(retrieval_question["question_id"], "旧问题",
                                 [0.0] * EMBEDDING_DIMENSIONS, None)],
                               EMBEDDING_MODEL, EMBEDDING_DIMENSIONS)
            with EmbeddingCache(acceptance_cache):
                pass
            with self.assertRaisesRegex(ValueError, "问题向量缓存不匹配，不允许在线补齐"):
                load_questions_and_vectors(bundle, evaluation)

    def make_loader_inputs(self, root: Path):
        from scripts.run_chunking_benchmark import write_jsonl

        bundle = root / "bundle"
        annotation_dir = bundle / "annotations"
        annotation_dir.mkdir(parents=True)
        retrieval_question = {
            "question_id": "retrieval-q1", "query": "检索查询", "answerable": True,
            "relevant_document_ids": ["target"], "evidence": [],
        }
        acceptance_question = {
            "question_id": "acceptance-q1", "query": "验收查询", "answerable": True,
            "relevant_document_groups": [["target"]], "evidence": [{
                "document_ids": ["target"], "quote": "唯一证据",
                "locator": {"type": "line", "line": 1},
            }],
        }
        write_jsonl(annotation_dir / "retrieval_questions.jsonl", [retrieval_question])
        acceptance_path = annotation_dir / "acceptance_questions_v1.jsonl"
        write_jsonl(acceptance_path, [acceptance_question])
        lock = {"questions_sha256": sha256_text_file(acceptance_path)}
        (annotation_dir / "acceptance_v1.lock.json").write_text(
            __import__("json").dumps(lock), encoding="utf-8",
        )
        evaluation = SimpleNamespace(embeddings=root / "evaluation" / "embeddings")
        return bundle, evaluation, [retrieval_question, acceptance_question]

    def test_duplicate_question_ids_fail_before_retrieval(self) -> None:
        question = self.acceptance_question()
        doc = self.document("c1", "d1", "内容")
        with self.assertRaisesRegex(ValueError, "题号重复"):
            validate_frozen_inputs([question, dict(question)], {"查询": [0.0]}, [doc], {
                "c1": {"document_id": "d1", "text": "内容"},
            })

    def test_corpus_id_or_text_mismatch_fails(self) -> None:
        doc = self.document("c1", "d1", "实际内容")
        for catalog in (
            {},
            {"c1": {"document_id": "d1", "text": "不同内容"}},
            {"other": {"document_id": "d1", "text": "实际内容"}},
        ):
            with self.subTest(catalog=catalog):
                with self.assertRaisesRegex(ValueError, "冻结语料不一致"):
                    validate_frozen_inputs([], {}, [doc], catalog)

    def test_both_tokenizers_share_one_ordered_vector_recall_per_query(self) -> None:
        question = self.acceptance_question()
        docs = [self.document(f"c{i}", f"d{i}", f"查询 证据{i}") for i in range(1, 4)]
        catalog = {doc.metadata["chunk_id"]: {
            "document_id": doc.metadata["document_id"], "text": doc.page_content,
        } for doc in docs}
        dense = Mock(return_value=docs)

        report = compare_tokenizers(
            [question], docs, catalog, {"target": "d1"}, dense,
            count_tokens=lambda text: len(text),
            score_fn=lambda *_args: {},
            retriever_factory=self.fake_retriever_factory,
        )

        dense.assert_called_once_with("查询", 20)
        self.assertEqual(report["dense_candidates"]["查询"],
                         [doc.metadata["chunk_id"] for doc in docs])
        self.assertLessEqual(len(report["arms"]["char"]["rows"]["q1"]["ranked_chunk_ids"]), 10)
        self.assertEqual(report["settings"]["vector_k"], 20)
        self.assertEqual(report["settings"]["rrf_k"], 60)

    def test_refusals_are_recorded_but_excluded_from_answerable_denominator(self) -> None:
        answerable = self.acceptance_question()
        refusal = {
            "question_id": "r1", "query": "缺项查询", "answerable": False,
            "evaluation_suite": "acceptance_23", "scope_document_id": "target",
        }
        docs = [self.document("c1", "doc-target", "内容")]
        catalog = {"c1": {"document_id": "doc-target", "text": "内容"}}
        report = compare_tokenizers(
            [answerable, refusal], docs, catalog, {"target": "doc-target"},
            lambda _query, _limit: docs, count_tokens=lambda text: len(text),
            score_fn=lambda *_args: {}, retriever_factory=self.fake_retriever_factory,
        )

        arm = report["arms"]["char"]
        self.assertEqual(arm["summary"]["answerable_count"], 1)
        self.assertEqual(arm["summary"]["refusal_count"], 1)
        self.assertIsNone(arm["rows"]["r1"]["metrics"])
        self.assertTrue(arm["rows"]["r1"]["scope_document_present"])

    def test_each_arm_builds_bm25_with_its_selected_tokenizer(self) -> None:
        from retrieval.hybrid_retriever import HybridRetrieverService
        from retrieval.tokenizers import TOKENIZERS

        question = self.acceptance_question()
        docs = [self.document(f"c{i}", f"d{i}", f"查询 证据{i}") for i in range(1, 4)]
        catalog = {doc.metadata["chunk_id"]: {
            "document_id": doc.metadata["document_id"], "text": doc.page_content,
        } for doc in docs}
        retrievers = {}

        def factory(**kwargs):
            retriever = HybridRetrieverService(**kwargs)
            retrievers[kwargs["tokenizer_id"]] = retriever
            return retriever

        compare_tokenizers(
            [question], docs, catalog, {"target": "d1"},
            lambda _query, _limit: docs, count_tokens=len,
            score_fn=lambda *_args: {}, retriever_factory=factory,
        )

        self.assertIs(retrievers["char"]._bm25_retriever.preprocess_func, TOKENIZERS["char"])
        self.assertIs(retrievers["mixed"]._bm25_retriever.preprocess_func, TOKENIZERS["mixed"])

    def fake_retriever_factory(
        self, *, get_vector_docs, get_all_docs, get_document_count,
        vector_k, bm25_k, final_k, rrf_k, strategy_id, tokenizer_id,
    ):
        class Retriever:
            def get_candidates(self, query, limit):
                return get_vector_docs(query, vector_k)[:limit]

        return Retriever()

    def test_fixed_settings_mismatch_prevents_paired_delta(self) -> None:
        left = {
            "identity": {"dataset_sha256": "same", "catalog_sha256": "same"},
            "settings": {"vector_k": 20, "final_k": 10},
            "rows": {"q1": {"metrics": {"evidence_recall_at_10": 1.0}}},
        }
        right = {
            "identity": {"dataset_sha256": "same", "catalog_sha256": "same"},
            "settings": {"vector_k": 20, "final_k": 6},
            "rows": {"q1": {"metrics": {"evidence_recall_at_10": 0.0}}},
        }
        with self.assertRaisesRegex(ValueError, "固定实验参数不一致"):
            paired_metric_deltas(left, right, "evidence_recall_at_10")

    def test_runtime_config_accepts_equivalent_windows_path_separators(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory)
            chroma = storage / "chroma_db"
            values = {
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
                "hybrid_vector_k": 20,
                "hybrid_bm25_k": 20,
                "hybrid_final_k": 10,
                "hybrid_rrf_k": 60,
                "retrieval_source_filter": "",
                "persist_directory": str(chroma).replace("\\", "/"),
            }
            with unittest.mock.patch.dict("os.environ", {
                "RAG_STORAGE_ROOT": str(storage),
                "CHROMA_PERSIST_DIRECTORY": str(chroma),
            }):
                _check_runtime_config(SimpleNamespace(**values), storage, chroma)


if __name__ == "__main__":
    unittest.main()
