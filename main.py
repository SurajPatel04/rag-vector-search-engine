"""
main.py
-------
Entry point for the RAG pipeline benchmark.

Runs 3 complex queries through both strategies and saves:
  - retrieval_benchmark.md   (human-readable markdown report)
  - retrieval_benchmark.json (structured JSON output)

Usage
-----
    python main.py
"""

from __future__ import annotations

import json
import os
import sys
import pathlib

# Make sure src/ is on the path
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from documents import DOCUMENTS
from embedder import VertexAIEmbedder
from query_expander import VertexAIQueryExpander
from retriever import RAGRetriever, ComparisonReport


# ---------------------------------------------------------------------------
# 1.  The 3 benchmark queries (complex, as required by the assessment)
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES = [
    "How does the system handle peak load and traffic spikes?",
    "What caching and database scaling strategies are used to ensure low latency?",
    "How does the platform manage failures, retries, and fault tolerance across services?",
]


# ---------------------------------------------------------------------------
# 2.  Helpers to format output
# ---------------------------------------------------------------------------

def build_json_report(reports: list[ComparisonReport]) -> list[dict]:
    """
    Build the structured JSON output matching the assessment format:
    {
        "query": "...",
        "strategy_a": ["chunk_id", ...],
        "strategy_b": ["chunk_id", ...],
        "scores_a":   [0.91, ...],
        "scores_b":   [0.95, ...],
    }
    Plus extra detail for completeness.
    """
    output = []
    for report in reports:
        output.append({
            "query": report.query,
            "strategy_a": [r.doc_id for r in report.strategy_a.results],
            "strategy_b": [r.doc_id for r in report.strategy_b.results],
            "scores_a":   [round(r.score, 4) for r in report.strategy_a.results],
            "scores_b":   [round(r.score, 4) for r in report.strategy_b.results],
            "topics_a":   [r.topic for r in report.strategy_a.results],
            "topics_b":   [r.topic for r in report.strategy_b.results],
            "analysis": {
                "avg_score_a":          report.avg_score_a(),
                "avg_score_b":          report.avg_score_b(),
                "winner":               report.winner(),
                "overlapping_docs":     report.overlap(),
                "unique_to_strategy_b": report.unique_to_b(),
                "latency_a_ms":         round(report.strategy_a.latency_ms, 2),
                "latency_b_ms":         round(report.strategy_b.latency_ms, 2),
            },
        })
    return output


def build_markdown_report(reports: list[ComparisonReport], json_data: list[dict]) -> str:
    """Build the full retrieval_benchmark.md content."""
    lines = [
        "# Retrieval Benchmark: Strategy A vs Strategy B",
        "",
        "## Overview",
        "",
        "| | Strategy A | Strategy B |",
        "|---|---|---|",
        "| **Name** | Raw Vector Search | AI-Enhanced Retrieval |",
        "| **Query used** | Raw user input | Expanded by mocked GenerativeModel |",
        "| **Embedding** | Embed raw query | Embed expanded query |",
        "| **Search** | FAISS cosine similarity | FAISS cosine similarity |",
        "| **Similarity metric** | Cosine (Inner Product on unit vectors) | Same |",
        "",
        "---",
        "",
        "## Why Cosine Similarity over Euclidean?",
        "",
        "Text embeddings encode **semantic direction**, not magnitude.",
        "Two sentences with the same meaning but different lengths still",
        "point in the same direction in vector space — cosine similarity",
        "captures this perfectly, while Euclidean distance would penalise",
        "length differences incorrectly.",
        "",
        "Because `sentence-transformers` (and Vertex AI gecko) return",
        "**L2-normalised** unit vectors, cosine similarity reduces to a",
        "simple dot product — which FAISS `IndexFlatIP` implements natively",
        "and very efficiently.",
        "",
        "---",
        "",
        "## Benchmark Results",
        "",
    ]

    # Per-query sections
    for i, (report, jdata) in enumerate(zip(reports, json_data), start=1):
        lines += [
            f"### Query {i}: `{report.query}`",
            "",
            "#### JSON Output",
            "",
            "```json",
            json.dumps(jdata, indent=2),
            "```",
            "",
            "#### Strategy A — Raw Vector Search",
            "",
            f"- **Search query:** `{report.strategy_a.search_query}`",
            f"- **Latency:** {report.strategy_a.latency_ms:.1f} ms",
            f"- **Avg similarity score:** {report.avg_score_a()}",
            "",
            "| Rank | Doc ID | Topic | Score | Text Preview |",
            "|------|--------|-------|-------|--------------|",
        ]
        for r in report.strategy_a.results:
            preview = r.text[:80].replace("|", "-") + "..."
            lines.append(f"| {r.rank} | {r.doc_id} | {r.topic} | {r.score:.4f} | {preview} |")

        lines += [
            "",
            "#### Strategy B — AI-Enhanced Retrieval",
            "",
            f"- **Expanded query:** `{report.strategy_b.search_query[:120]}...`",
            f"- **Latency:** {report.strategy_b.latency_ms:.1f} ms",
            f"- **Avg similarity score:** {report.avg_score_b()}",
            "",
            "| Rank | Doc ID | Topic | Score | Text Preview |",
            "|------|--------|-------|-------|--------------|",
        ]
        for r in report.strategy_b.results:
            preview = r.text[:80].replace("|", "-") + "..."
            lines.append(f"| {r.rank} | {r.doc_id} | {r.topic} | {r.score:.4f} | {preview} |")

        lines += [
            "",
            "#### Analysis",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Winner | **{report.winner()}** |",
            f"| Avg Score A | {report.avg_score_a()} |",
            f"| Avg Score B | {report.avg_score_b()} |",
            f"| Overlapping docs | {report.overlap() or 'None'} |",
            f"| New docs found by B | {report.unique_to_b() or 'None'} |",
            "",
            "---",
            "",
        ]

    # Summary table
    lines += [
        "## Summary Table",
        "",
        "| Query | Avg Score A | Avg Score B | Winner |",
        "|-------|-------------|-------------|--------|",
    ]
    for report in reports:
        short_q = report.query[:55] + "..."
        lines.append(
            f"| {short_q} | {report.avg_score_a()} | {report.avg_score_b()} | {report.winner()} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Production Migration: Vertex AI Matching Engine",
        "",
        "To migrate this pipeline from local FAISS to **Vertex AI Vector Search**",
        "(Matching Engine) in production:",
        "",
        "1. **Create a Matching Engine Index** in GCP with the same dimension (768 for gecko).",
        "2. **Replace `FAISSVectorStore.add()`** with a call to upload embeddings to the index.",
        "3. **Replace `FAISSVectorStore.search()`** with `endpoint.find_neighbors()` from",
        "   `google.cloud.aiplatform`.",
        "4. **Replace `VertexAIEmbedder`** mock with the real",
        "   `TextEmbeddingModel.from_pretrained('textembedding-gecko@003')`.",
        "5. **Replace `VertexAIQueryExpander`** mock with the real",
        "   `GenerativeModel('gemini-1.5-pro')`.",
        "6. The `RAGRetriever`, `StrategyResult`, and `ComparisonReport` classes",
        "   remain **unchanged** — only the injected dependencies swap out.",
        "",
        "This design follows the **Dependency Injection** pattern, making the",
        "local ↔ cloud swap a one-line config change.",
        "",
        "---",
        "",
        "*Generated by main.py — context-aware-retrieval-engine*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3.  Main runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  Context-Aware Retrieval Engine — Benchmark Runner")
    print("=" * 60 + "\n")

    # -- Step 1: Build pipeline --
    retriever = RAGRetriever(top_k=3)

    # -- Step 2: Ingest all documents --
    retriever.ingest(DOCUMENTS)

    # -- Step 3: Run benchmark --
    reports = retriever.benchmark(BENCHMARK_QUERIES)

    # -- Step 4: Build outputs --
    json_data = build_json_report(reports)
    markdown   = build_markdown_report(reports, json_data)

    # -- Step 5: Save files --
    output_dir = ROOT
    md_path    = output_dir / "retrieval_benchmark.md"
    json_path  = output_dir / "retrieval_benchmark.json"

    with open(md_path, "w") as f:
        f.write(markdown)

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # -- Step 6: Print JSON to console --
    print("\n" + "=" * 60)
    print("  BENCHMARK JSON OUTPUT")
    print("=" * 60)
    print(json.dumps(json_data, indent=2))

    print("\n" + "=" * 60)
    print("  FILES SAVED")
    print("=" * 60)
    print(f"  ✅ {md_path}")
    print(f"  ✅ {json_path}")
    print("\n✅ Benchmark complete.\n")


if __name__ == "__main__":
    main()