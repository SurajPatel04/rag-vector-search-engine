# Context-Aware Retrieval Engine

A local **Retrieval Augmented Generation (RAG)** pipeline that ingests technical documents, generates embeddings, and benchmarks two semantic retrieval strategies side-by-side.

---

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌────────────────┐
│  documents.py│────▶│  VertexAIEmbedder│────▶│ FAISSVectorStore│
│  (10 chunks) │     │  (mocked gecko)  │     │  (cosine / IP) │
└──────────────┘     └──────────────────┘     └───────┬────────┘
                                                      │
                     ┌──────────────────┐              │
  User Query ───────▶│   RAGRetriever   │◀─────────────┘
                     │                  │
                     │  Strategy A ─── embed raw query ──────▶ FAISS search
                     │  Strategy B ─── expand ──▶ embed ────▶ FAISS search
                     │                  │
                     └──────┬───────────┘
                            │
                     ┌──────▼───────────┐
                     │ ComparisonReport  │
                     │  (JSON + Table)   │
                     └──────────────────┘
```

### Retrieval Strategies

| | Strategy A (Raw Vector Search) | Strategy B (AI-Enhanced Retrieval) |
|---|---|---|
| **Query** | Raw user input | Expanded by mocked `GenerativeModel` |
| **Embedding** | Embed raw query directly | Embed semantically enriched query |
| **Search** | FAISS cosine similarity | FAISS cosine similarity |

---

## Project Structure

```
├── main.py                    # Entry point — runs benchmark, generates reports
├── requirements.txt           # Python dependencies
├── retrieval_benchmark.md     # Generated comparison report (Strategy A vs B)
├── retrieval_benchmark.json   # Generated structured JSON output
│
├── src/
│   ├── __init__.py
│   ├── embedder.py            # LocalEmbedder + mocked VertexAI TextEmbeddingModel
│   ├── vector_store.py        # FAISS-backed vector store with cosine similarity
│   ├── query_expander.py      # Mocked VertexAI GenerativeModel for query expansion
│   └── retriever.py           # RAGRetriever — orchestrates ingestion + both strategies
│
├── data/
│   ├── __init__.py
│   └── documents.py           # 10 technical paragraphs (the knowledge base)
│
└── tests/
    ├── __init__.py
    └── test_pipeline.py       # Pytest suite — 40+ tests across 6 categories
```

---

## Setup

```bash
# Clone the repository
git clone <repo-url>
cd Teleport-assigment

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the Benchmark

```bash
python main.py
```

This will:
1. Load 10 technical documents from `data/documents.py`
2. Embed all documents using the mocked Vertex AI `TextEmbeddingModel`
3. Run 3 complex queries through both Strategy A and Strategy B
4. Print the JSON comparison to the console
5. Save `retrieval_benchmark.md` and `retrieval_benchmark.json` to the project root

### Run Tests

```bash
pytest tests/test_pipeline.py -v
```

Test coverage includes:
- **Embedder** — vector shape, dimensions, batch processing, semantic similarity
- **VertexAI Mock** — SDK interface compatibility, `from_pretrained()`, response objects
- **Vector Store** — indexing, search ranking, top-k clamping, save/load persistence
- **Query Expander** — expansion quality, relevant term matching, fallback handling
- **Retriever** — Strategy A vs B, latency recording, ingestion guards
- **Integration** — end-to-end pipeline, JSON serialisation, markdown output

---

## Similarity Metric: Cosine vs Euclidean

This pipeline uses **cosine similarity** (implemented as inner product on L2-normalised vectors) rather than Euclidean distance. The rationale:

| Factor | Cosine Similarity | Euclidean Distance |
|---|---|---|
| **What it measures** | Angular direction between vectors | Absolute distance between points |
| **Sensitivity to magnitude** | Invariant — only cares about direction | Sensitive — penalises length differences |
| **Semantic fit** | Two sentences with the same meaning but different lengths score high | Same sentences may score lower due to magnitude differences |
| **FAISS implementation** | `IndexFlatIP` (inner product) — native, very fast | `IndexFlatL2` — also fast, but semantically less appropriate |
| **Normalisation** | `sentence-transformers` returns L2-normalised vectors, so IP ≡ cosine | Requires no normalisation, but gives worse semantic results |

**Bottom line:** Text embeddings encode *semantic direction*, not magnitude. Cosine similarity captures this correctly; Euclidean distance would penalise irrelevant length differences.

---

## Mocking Strategy

Both GCP SDK components are mocked using `unittest.mock.MagicMock`:

### TextEmbeddingModel (embedder.py)
- `vertexai.init()` → mocked, no-op
- `TextEmbeddingModel.from_pretrained("textembedding-gecko@003")` → returns `VertexAIEmbedder`
- `.get_embeddings()` → delegates to local `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim)

### GenerativeModel (query_expander.py)
- `GenerativeModel("gemini-1.5-pro")` → returns mocked model
- `.generate_content(prompt)` → routes to a local rule-based expander that maps keyword patterns to semantically rich expansions

The mocks replicate the **exact GCP SDK interface**, so the codebase is production-ready with a one-line swap.

---

## Production Migration: Vertex AI Matching Engine

To migrate from local FAISS to **Vertex AI Vector Search** (Matching Engine):

| Component | Local (current) | Production (GCP) |
|---|---|---|
| **Embedder** | `sentence-transformers` (all-MiniLM-L6-v2) | `TextEmbeddingModel.from_pretrained("textembedding-gecko@003")` |
| **Vector Store** | FAISS `IndexFlatIP` | Vertex AI Matching Engine `endpoint.find_neighbors()` |
| **Query Expander** | Rule-based mock | `GenerativeModel("gemini-1.5-pro")` |
| **Retriever** | `RAGRetriever` | **Unchanged** — dependency injection |

The `RAGRetriever`, `StrategyResult`, and `ComparisonReport` classes remain **unchanged** — only the injected dependencies swap out. This follows the **Dependency Injection** pattern, making the local ↔ cloud transition a configuration change rather than a code rewrite.

---

## Benchmark Results Summary

| Query | Avg Score A | Avg Score B | Winner |
|---|---|---|---|
| Peak load & traffic spikes | 0.4726 | 0.5578 | **Strategy B** |
| Caching & database scaling | 0.5515 | 0.6109 | **Strategy B** |
| Failures, retries & fault tolerance | 0.4922 | 0.5208 | **Strategy B** |

Strategy B (AI-Enhanced Retrieval) consistently outperforms Strategy A by producing higher similarity scores and discovering relevant documents that raw vector search misses. See [`retrieval_benchmark.md`](retrieval_benchmark.md) for the full comparison report.
