from __future__ import annotations

import json
import os
import sys
import pathlib
import pytest
from unittest.mock import MagicMock, patch

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from documents import DOCUMENTS
from embedder import LocalEmbedder, VertexAIEmbedder, TextEmbeddingResponse
from vector_store import FAISSVectorStore, SearchResult
from query_expander import VertexAIQueryExpander
from retriever import RAGRetriever, StrategyResult, ComparisonReport


@pytest.fixture(scope="module")
def local_embedder() -> LocalEmbedder:
    return LocalEmbedder()


@pytest.fixture(scope="module")
def vertex_embedder() -> VertexAIEmbedder:
    return VertexAIEmbedder.from_pretrained()


@pytest.fixture(scope="module")
def expander() -> VertexAIQueryExpander:
    return VertexAIQueryExpander()


@pytest.fixture(scope="module")
def populated_store(vertex_embedder) -> FAISSVectorStore:
    store = FAISSVectorStore(dimension=vertex_embedder.dimension)
    texts   = [doc["text"] for doc in DOCUMENTS]
    vectors = vertex_embedder.embed_batch(texts)
    store.add(DOCUMENTS, vectors)
    return store


@pytest.fixture(scope="module")
def retriever(vertex_embedder, expander) -> RAGRetriever:
    r = RAGRetriever(top_k=3, embedder=vertex_embedder, expander=expander)
    r.ingest(DOCUMENTS)
    return r


class TestLocalEmbedder:
    def test_embed_returns_list(self, local_embedder):
        result = local_embedder.embed("Test query about peak load.")
        assert isinstance(result, list), "embed() should return a list"

    def test_embed_correct_dimension(self, local_embedder):
        result = local_embedder.embed("Test query.")
        assert len(result) == local_embedder.dimension, (
            f"Expected {local_embedder.dimension} dims, got {len(result)}"
        )

    def test_embed_values_are_floats(self, local_embedder):
        result = local_embedder.embed("Cloud auto-scaling policies.")
        assert all(isinstance(v, float) for v in result), \
            "All vector values must be floats"

    def test_embed_batch_length(self, local_embedder):
        texts = ["Peak load handling.", "Caching strategy.", "Database scaling."]
        results = local_embedder.embed_batch(texts)
        assert len(results) == len(texts), \
            "embed_batch must return same number of vectors as inputs"

    def test_embed_batch_correct_dimensions(self, local_embedder):
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        results = local_embedder.embed_batch(texts)
        for vec in results:
            assert len(vec) == local_embedder.dimension

    def test_different_texts_produce_different_vectors(self, local_embedder):
        vec1 = local_embedder.embed("Peak load traffic spike handling.")
        vec2 = local_embedder.embed("Database connection pooling strategy.")
        assert vec1 != vec2, "Different texts should produce different vectors"

    def test_similar_texts_produce_close_vectors(self, local_embedder):
        import numpy as np
        vec1 = local_embedder.embed("How does the system handle peak load?")
        vec2 = local_embedder.embed("How does the platform manage high traffic spikes?")
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        cosine_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        assert cosine_sim > 0.45, \
            f"Similar texts should have cosine similarity > 0.45, got {cosine_sim:.4f}"


class TestVertexAIEmbedder:
    def test_from_pretrained_returns_instance(self):
        embedder = VertexAIEmbedder.from_pretrained()
        assert isinstance(embedder, VertexAIEmbedder)

    def test_get_embeddings_returns_response_objects(self, vertex_embedder):
        responses = vertex_embedder.get_embeddings(["Test query."])
        assert len(responses) == 1
        assert isinstance(responses[0], TextEmbeddingResponse)

    def test_response_has_values(self, vertex_embedder):
        responses = vertex_embedder.get_embeddings(["Auto-scaling policies."])
        assert len(responses[0].values) > 0, "Response .values must not be empty"

    def test_response_correct_dimension(self, vertex_embedder):
        responses = vertex_embedder.get_embeddings(["Caching strategy."])
        assert len(responses[0].values) == vertex_embedder.dimension

    def test_response_preserves_text(self, vertex_embedder):
        text = "How does the database scale under load?"
        responses = vertex_embedder.get_embeddings([text])
        assert responses[0].text == text

    def test_batch_get_embeddings(self, vertex_embedder):
        texts = ["Peak load.", "Caching.", "Fault tolerance."]
        responses = vertex_embedder.get_embeddings(texts)
        assert len(responses) == 3

    def test_mock_sdk_init_does_not_raise(self):
        embedder = VertexAIEmbedder()
        try:
            embedder._mock_sdk.init(project="test-project", location="us-central1")
        except Exception as e:
            pytest.fail(f"Mocked vertexai.init() raised an exception: {e}")

    def test_embed_shortcut_returns_list(self, vertex_embedder):
        vec = vertex_embedder.embed("Test query")
        assert isinstance(vec, list)
        assert len(vec) == vertex_embedder.dimension

class TestFAISSVectorStore:

    def test_add_increases_document_count(self, vertex_embedder):
        store = FAISSVectorStore(dimension=vertex_embedder.dimension)
        assert store.total_documents == 0
        texts   = [doc["text"] for doc in DOCUMENTS[:3]]
        vectors = vertex_embedder.embed_batch(texts)
        store.add(DOCUMENTS[:3], vectors)
        assert store.total_documents == 3

    def test_search_returns_correct_top_k(self, populated_store, vertex_embedder):
        query_vec = vertex_embedder.embed("How does the system handle peak load?")
        results   = populated_store.search(query_vec, top_k=3)
        assert len(results) == 3

    def test_search_returns_search_result_objects(self, populated_store, vertex_embedder):
        query_vec = vertex_embedder.embed("Caching strategy.")
        results   = populated_store.search(query_vec, top_k=3)
        for r in results:
            assert isinstance(r, SearchResult)

    def test_search_results_have_valid_scores(self, populated_store, vertex_embedder):
        query_vec = vertex_embedder.embed("Database scaling.")
        results   = populated_store.search(query_vec, top_k=3)
        for r in results:
            assert 0.0 <= r.score <= 1.0, \
                f"Score {r.score} is outside [0, 1]"

    def test_search_results_are_ranked(self, populated_store, vertex_embedder):
        query_vec = vertex_embedder.embed("Error handling and fault tolerance.")
        results   = populated_store.search(query_vec, top_k=3)
        scores    = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), \
            "Results must be sorted by score descending"

    def test_search_results_have_metadata(self, populated_store, vertex_embedder):
        query_vec = vertex_embedder.embed("Auto-scaling policies.")
        results   = populated_store.search(query_vec, top_k=1)
        r = results[0]
        assert r.doc_id, "doc_id must not be empty"
        assert r.topic,  "topic must not be empty"
        assert r.text,   "text must not be empty"

    def test_top_k_clamped_to_store_size(self, vertex_embedder):
        store   = FAISSVectorStore(dimension=vertex_embedder.dimension)
        texts   = [doc["text"] for doc in DOCUMENTS[:2]]
        vectors = vertex_embedder.embed_batch(texts)
        store.add(DOCUMENTS[:2], vectors)
        query_vec = vertex_embedder.embed("Peak load.")
        results   = store.search(query_vec, top_k=10)  # more than 2 docs
        assert len(results) <= 2, "top_k must be clamped to available docs"

    def test_mismatched_docs_vectors_raises(self, vertex_embedder):
        store   = FAISSVectorStore(dimension=vertex_embedder.dimension)
        vectors = vertex_embedder.embed_batch([DOCUMENTS[0]["text"]])
        with pytest.raises(ValueError, match="same length"):
            store.add(DOCUMENTS[:3], vectors)  # 3 docs, 1 vector → error

    def test_search_empty_store_raises(self, vertex_embedder):
        store     = FAISSVectorStore(dimension=vertex_embedder.dimension)
        query_vec = vertex_embedder.embed("Test.")
        with pytest.raises(RuntimeError, match="empty"):
            store.search(query_vec, top_k=3)

    def test_save_and_load(self, populated_store, vertex_embedder, tmp_path):
        save_dir = str(tmp_path / "test_index")
        populated_store.save(save_dir)

        reloaded = FAISSVectorStore(dimension=vertex_embedder.dimension)
        reloaded.load(save_dir)

        assert reloaded.total_documents == populated_store.total_documents

        query_vec = vertex_embedder.embed("Peak load handling.")
        original_results = populated_store.search(query_vec, top_k=3)
        reloaded_results = reloaded.search(query_vec, top_k=3)

        assert [r.doc_id for r in original_results] == \
               [r.doc_id for r in reloaded_results], \
               "Reloaded store must return same results as original"

class TestVertexAIQueryExpander:

    def test_expand_returns_string(self, expander):
        result = expander.expand("How does the system handle peak load?")
        assert isinstance(result, str)

    def test_expand_returns_non_empty_string(self, expander):
        result = expander.expand("How does the system handle peak load?")
        assert len(result.strip()) > 0, "Expanded query must not be empty"

    def test_expansion_is_longer_than_input(self, expander):
        raw      = "How does the system handle peak load?"
        expanded = expander.expand(raw)
        assert len(expanded) > len(raw), \
            "Expanded query should be longer than raw query"

    def test_expansion_contains_relevant_terms(self, expander):
        expanded = expander.expand("How does the system handle peak load?").lower()
        relevant_terms = ["load", "scal", "traffic", "balanc"]
        matched = [t for t in relevant_terms if t in expanded]
        assert len(matched) >= 2, \
            f"Expected relevant terms in expansion, matched: {matched}"

    def test_caching_query_expansion(self, expander):
        expanded = expander.expand("What is the caching strategy?").lower()
        assert any(term in expanded for term in ["cache", "redis", "ttl", "evict"])

    def test_database_query_expansion(self, expander):
        expanded = expander.expand("How does database scaling work?").lower()
        assert any(term in expanded for term in ["replica", "shard", "pool", "query"])

    def test_error_handling_query_expansion(self, expander):
        expanded = expander.expand("What happens when a service failure occurs?").lower()
        assert any(term in expanded for term in ["retry", "fault", "circuit", "error"])

    def test_fallback_for_unknown_query(self, expander):
        result = expander.expand("Something completely random and unrelated xyz123")
        assert isinstance(result, str)
        assert len(result.strip()) > 0, "Fallback expansion must not be empty"

    def test_generate_content_returns_mock_with_text(self, expander):
        response = expander.generate_content("Query: Peak load handling?")
        assert hasattr(response, "text"), "Response must have a .text attribute"
        assert isinstance(response.text, str)
        assert len(response.text) > 0

class TestRAGRetriever:

    def test_retriever_ingests_documents(self, retriever):
        assert retriever.store.total_documents == len(DOCUMENTS), \
            f"Expected {len(DOCUMENTS)} docs, got {retriever.store.total_documents}"

    def test_strategy_a_returns_strategy_result(self, retriever):
        result = retriever.strategy_a("How does the system handle peak load?")
        assert isinstance(result, StrategyResult)
        assert result.strategy == "A"

    def test_strategy_b_returns_strategy_result(self, retriever):
        result = retriever.strategy_b("How does the system handle peak load?")
        assert isinstance(result, StrategyResult)
        assert result.strategy == "B"

    def test_strategy_a_top_k_count(self, retriever):
        result = retriever.strategy_a("Caching strategy.")
        assert len(result.results) == retriever.top_k

    def test_strategy_b_top_k_count(self, retriever):
        result = retriever.strategy_b("Caching strategy.")
        assert len(result.results) == retriever.top_k

    def test_strategy_a_uses_raw_query(self, retriever):
        query  = "How does auto-scaling work?"
        result = retriever.strategy_a(query)
        assert result.search_query == query, \
            "Strategy A must search with the raw unmodified query"

    def test_strategy_b_uses_expanded_query(self, retriever):
        query  = "How does auto-scaling work?"
        result = retriever.strategy_b(query)
        assert result.search_query != query, \
            "Strategy B must use the expanded query, not the raw input"

    def test_strategy_b_expanded_query_longer(self, retriever):
        query  = "How does the system handle peak load?"
        result = retriever.strategy_b(query)
        assert len(result.search_query) > len(query), \
            "Strategy B expanded query must be richer and longer"

    def test_strategy_b_different_results_from_a(self, retriever):
        queries = [
            "How does the system handle peak load?",
            "What caching and database scaling strategies are in use?",
            "How does the platform manage failures and retries?",
        ]
        found_difference = False
        for query in queries:
            result_a = retriever.strategy_a(query)
            result_b = retriever.strategy_b(query)
            ids_a = [r.doc_id for r in result_a.results]
            ids_b = [r.doc_id for r in result_b.results]
            if ids_a != ids_b:
                found_difference = True
                break

        assert found_difference, (
            "Strategy B must return at least one different result "
            "compared to Strategy A across the benchmark queries"
        )

    def test_strategy_a_latency_recorded(self, retriever):
        result = retriever.strategy_a("Peak load handling.")
        assert result.latency_ms > 0, "Latency must be a positive number"

    def test_strategy_b_latency_recorded(self, retriever):
        result = retriever.strategy_b("Peak load handling.")
        assert result.latency_ms > 0, "Latency must be a positive number"

    def test_compare_returns_comparison_report(self, retriever):
        report = retriever.compare("How does the system handle peak load?")
        assert isinstance(report, ComparisonReport)

    def test_comparison_report_has_both_strategies(self, retriever):
        report = retriever.compare("Caching strategy.")
        assert report.strategy_a is not None
        assert report.strategy_b is not None

    def test_comparison_winner_is_valid_string(self, retriever):
        report = retriever.compare("Database scaling.")
        winner = report.winner()
        assert isinstance(winner, str)
        assert len(winner) > 0

    def test_benchmark_returns_correct_number_of_reports(self, retriever):
        queries = [
            "How does the system handle peak load?",
            "What is the caching strategy?",
            "How does fault tolerance work?",
        ]
        reports = retriever.benchmark(queries)
        assert len(reports) == len(queries)

    def test_ingest_required_before_search(self, vertex_embedder, expander):
        fresh = RAGRetriever(top_k=3, embedder=vertex_embedder, expander=expander)
        with pytest.raises(RuntimeError, match="ingest"):
            fresh.strategy_a("Peak load.")


class TestIntegration:

    def test_full_pipeline_produces_json_serialisable_output(self, retriever):
        report = retriever.compare("How does the system handle peak load?")
        try:
            serialised = json.dumps(report.to_dict())
        except (TypeError, ValueError) as e:
            pytest.fail(f"ComparisonReport.to_dict() is not JSON-serialisable: {e}")
        assert len(serialised) > 0

    def test_full_pipeline_markdown_output(self, retriever):
        report = retriever.compare("Caching and database scaling.")
        md = report.to_markdown()
        assert isinstance(md, str)
        assert "Strategy A" in md
        assert "Strategy B" in md

    def test_all_benchmark_queries_complete(self, retriever):
        queries = [
            "How does the system handle peak load and traffic spikes?",
            "What caching and database scaling strategies are used?",
            "How does the platform manage failures and fault tolerance?",
        ]
        reports = retriever.benchmark(queries)
        assert len(reports) == 3
        for report in reports:
            assert len(report.strategy_a.results) == 3
            assert len(report.strategy_b.results) == 3

    def test_top_ranked_result_is_relevant(self, retriever):
        result  = retriever.strategy_b("How does the system handle peak load?")
        top_doc = result.results[0]
        relevant_topics = [
            "Peak Load Handling",
            "Auto-Scaling Policies",
            "Error Handling and Fault Tolerance",
        ]
        assert top_doc.topic in relevant_topics, (
            f"Top result topic '{top_doc.topic}' is not relevant to peak load query"
        )