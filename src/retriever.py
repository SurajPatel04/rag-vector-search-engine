from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from embedder import VertexAIEmbedder
from query_expander import VertexAIQueryExpander
from vector_store import FAISSVectorStore, SearchResult


@dataclass
class StrategyResult:
    strategy: str
    raw_query: str
    search_query: str
    results: List[SearchResult]
    latency_ms: float

    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy,
            "raw_query": self.raw_query,
            "search_query": self.search_query,
            "latency_ms": round(self.latency_ms, 2),
            "results": [r.to_dict() for r in self.results],
        }


@dataclass
class ComparisonReport:
    query: str
    strategy_a: StrategyResult
    strategy_b: StrategyResult

    def overlap(self) -> List[str]:
        ids_a = {r.doc_id for r in self.strategy_a.results}
        ids_b = {r.doc_id for r in self.strategy_b.results}
        return sorted(ids_a & ids_b)

    def unique_to_b(self) -> List[str]:
        ids_a = {r.doc_id for r in self.strategy_a.results}
        ids_b = {r.doc_id for r in self.strategy_b.results}
        return sorted(ids_b - ids_a)

    def avg_score_a(self) -> float:
        if not self.strategy_a.results:
            return 0.0
        return round(sum(r.score for r in self.strategy_a.results) / len(self.strategy_a.results), 4)

    def avg_score_b(self) -> float:
        if not self.strategy_b.results:
            return 0.0
        return round(sum(r.score for r in self.strategy_b.results) / len(self.strategy_b.results), 4)

    def winner(self) -> str:
        if self.avg_score_b() > self.avg_score_a():
            return "B (AI-Enhanced)"
        elif self.avg_score_a() > self.avg_score_b():
            return "A (Raw Vector)"
        return "Tie"

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "strategy_a": self.strategy_a.to_dict(),
            "strategy_b": self.strategy_b.to_dict(),
            "analysis": {
                "avg_score_a": self.avg_score_a(),
                "avg_score_b": self.avg_score_b(),
                "winner": self.winner(),
                "overlapping_docs": self.overlap(),
                "unique_to_strategy_b": self.unique_to_b(),
            },
        }

    def to_markdown(self) -> str:
        lines = [
            f"## Query: `{self.query}`",
            "",
            "### Strategy A — Raw Vector Search",
            f"- **Search query:** {self.strategy_a.search_query}",
            f"- **Latency:** {self.strategy_a.latency_ms:.1f} ms",
            f"- **Avg Score:** {self.avg_score_a()}",
            "",
            "| Rank | Doc ID  | Topic | Score |",
            "|------|---------|-------|-------|",
        ]
        for r in self.strategy_a.results:
            lines.append(f"| {r.rank} | {r.doc_id} | {r.topic} | {r.score:.4f} |")

        lines += [
            "",
            "### Strategy B — AI-Enhanced Retrieval",
            f"- **Expanded query:** {self.strategy_b.search_query[:120]}...",
            f"- **Latency:** {self.strategy_b.latency_ms:.1f} ms",
            f"- **Avg Score:** {self.avg_score_b()}",
            "",
            "| Rank | Doc ID  | Topic | Score |",
            "|------|---------|-------|-------|",
        ]
        for r in self.strategy_b.results:
            lines.append(f"| {r.rank} | {r.doc_id} | {r.topic} | {r.score:.4f} |")

        lines += [
            "",
            "### Analysis",
            f"- **Winner:** {self.winner()}",
            f"- **Overlapping docs:** {self.overlap() or 'None'}",
            f"- **New docs found by B:** {self.unique_to_b() or 'None'}",
            "",
            "---",
            "",
        ]
        return "\n".join(lines)


class RAGRetriever:
    def __init__(
        self,
        top_k: int = 3,
        embedder: Optional[VertexAIEmbedder] = None,
        expander: Optional[VertexAIQueryExpander] = None,
    ) -> None:
        print("\n[RAGRetriever] Initialising pipeline components...")
        self.top_k    = top_k
        self.embedder = embedder or VertexAIEmbedder.from_pretrained()
        self.expander = expander or VertexAIQueryExpander()
        self.store    = FAISSVectorStore(dimension=self.embedder.dimension)
        self._ingested = False
        print(f"[RAGRetriever] Ready — top_k={self.top_k}\n")

    def ingest(self, documents: List[Dict]) -> None:
        print(f"[RAGRetriever] Ingesting {len(documents)} documents...")
        texts   = [doc["text"] for doc in documents]
        vectors = self.embedder.embed_batch(texts)
        self.store.add(documents, vectors)
        self._ingested = True
        print(f"[RAGRetriever] Ingestion complete. Index size: {self.store.total_documents}\n")

    def strategy_a(self, query: str) -> StrategyResult:
        self._check_ingested()
        t0 = time.perf_counter()
        query_vector = self.embedder.embed(query)
        results = self.store.search(query_vector, top_k=self.top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        return StrategyResult(
            strategy="A",
            raw_query=query,
            search_query=query,
            results=results,
            latency_ms=latency_ms,
        )

    def strategy_b(self, query: str) -> StrategyResult:
        self._check_ingested()
        t0 = time.perf_counter()
        expanded_query = self.expander.expand(query)
        query_vector = self.embedder.embed(expanded_query)
        results = self.store.search(query_vector, top_k=self.top_k)
        latency_ms = (time.perf_counter() - t0) * 1000

        return StrategyResult(
            strategy="B",
            raw_query=query,
            search_query=expanded_query,
            results=results,
            latency_ms=latency_ms,
        )

    def compare(self, query: str) -> ComparisonReport:
        result_a = self.strategy_a(query)
        result_b = self.strategy_b(query)
        return ComparisonReport(query=query, strategy_a=result_a, strategy_b=result_b)

    def benchmark(self, queries: List[str]) -> List[ComparisonReport]:
        reports = []
        print(f"[RAGRetriever] Running benchmark on {len(queries)} queries...\n")
        for i, query in enumerate(queries, start=1):
            print(f"  [{i}/{len(queries)}] {query}")
            report = self.compare(query)
            reports.append(report)
            print(f"          Winner → {report.winner()} | "
                  f"Avg A={report.avg_score_a()} | Avg B={report.avg_score_b()}")
        print("\n[RAGRetriever] Benchmark complete.\n")
        return reports

    def _check_ingested(self) -> None:
        if not self._ingested:
            raise RuntimeError("No documents ingested. Call retriever.ingest(DOCUMENTS) first.")

    def __repr__(self) -> str:
        return (
            f"RAGRetriever(top_k={self.top_k}, "
            f"indexed_docs={self.store.total_documents}, "
            f"ingested={self._ingested})"
        )