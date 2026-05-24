# Retrieval Benchmark: Strategy A vs Strategy B

## Overview

| | Strategy A | Strategy B |
|---|---|---|
| **Name** | Raw Vector Search | AI-Enhanced Retrieval |
| **Query used** | Raw user input | Expanded by mocked GenerativeModel |
| **Embedding** | Embed raw query | Embed expanded query |
| **Search** | FAISS cosine similarity | FAISS cosine similarity |
| **Similarity metric** | Cosine (Inner Product on unit vectors) | Same |

---

## Why Cosine Similarity over Euclidean?

Text embeddings encode **semantic direction**, not magnitude.
Two sentences with the same meaning but different lengths still
point in the same direction in vector space — cosine similarity
captures this perfectly, while Euclidean distance would penalise
length differences incorrectly.

Because `sentence-transformers` (and Vertex AI gecko) return
**L2-normalised** unit vectors, cosine similarity reduces to a
simple dot product — which FAISS `IndexFlatIP` implements natively
and very efficiently.

---

## Benchmark Results

### Query 1: `How does the system handle peak load and traffic spikes?`

#### JSON Output

```json
{
  "query": "How does the system handle peak load and traffic spikes?",
  "strategy_a": [
    "doc_001",
    "doc_002",
    "doc_009"
  ],
  "strategy_b": [
    "doc_001",
    "doc_002",
    "doc_004"
  ],
  "scores_a": [
    0.6173,
    0.4648,
    0.3356
  ],
  "scores_b": [
    0.7484,
    0.5231,
    0.4019
  ],
  "topics_a": [
    "Peak Load Handling",
    "Auto-Scaling Policies",
    "Data Pipeline and ETL"
  ],
  "topics_b": [
    "Peak Load Handling",
    "Auto-Scaling Policies",
    "Database Scaling"
  ],
  "analysis": {
    "avg_score_a": 0.4726,
    "avg_score_b": 0.5578,
    "winner": "B (AI-Enhanced)",
    "overlapping_docs": [
      "doc_001",
      "doc_002"
    ],
    "unique_to_strategy_b": [
      "doc_004"
    ],
    "latency_a_ms": 36.0,
    "latency_b_ms": 26.32
  }
}
```

#### Strategy A — Raw Vector Search

- **Search query:** `How does the system handle peak load and traffic spikes?`
- **Latency:** 36.0 ms
- **Avg similarity score:** 0.4726

| Rank | Doc ID | Topic | Score | Text Preview |
|------|--------|-------|-------|--------------|
| 1 | doc_001 | Peak Load Handling | 0.6173 | The system handles peak load through a combination of horizontal scaling and int... |
| 2 | doc_002 | Auto-Scaling Policies | 0.4648 | Auto-scaling policies are governed by a rule engine that monitors CPU utilizatio... |
| 3 | doc_009 | Data Pipeline and ETL | 0.3356 | The data pipeline ingests raw events from operational systems, transforms them i... |

#### Strategy B — AI-Enhanced Retrieval

- **Expanded query:** `Describe the mechanisms for handling high traffic spikes and peak load conditions, including load balancing strategies, ...`
- **Latency:** 26.3 ms
- **Avg similarity score:** 0.5578

| Rank | Doc ID | Topic | Score | Text Preview |
|------|--------|-------|-------|--------------|
| 1 | doc_001 | Peak Load Handling | 0.7484 | The system handles peak load through a combination of horizontal scaling and int... |
| 2 | doc_002 | Auto-Scaling Policies | 0.5231 | Auto-scaling policies are governed by a rule engine that monitors CPU utilizatio... |
| 3 | doc_004 | Database Scaling | 0.4019 | Database scaling is achieved through a combination of read replicas, connection ... |

#### Analysis

| Metric | Value |
|--------|-------|
| Winner | **B (AI-Enhanced)** |
| Avg Score A | 0.4726 |
| Avg Score B | 0.5578 |
| Overlapping docs | ['doc_001', 'doc_002'] |
| New docs found by B | ['doc_004'] |

---

### Query 2: `What caching and database scaling strategies are used to ensure low latency?`

#### JSON Output

```json
{
  "query": "What caching and database scaling strategies are used to ensure low latency?",
  "strategy_a": [
    "doc_003",
    "doc_004",
    "doc_002"
  ],
  "strategy_b": [
    "doc_003",
    "doc_004",
    "doc_006"
  ],
  "scores_a": [
    0.5887,
    0.5708,
    0.4948
  ],
  "scores_b": [
    0.8607,
    0.518,
    0.4541
  ],
  "topics_a": [
    "Caching Strategies",
    "Database Scaling",
    "Auto-Scaling Policies"
  ],
  "topics_b": [
    "Caching Strategies",
    "Database Scaling",
    "Message Queue and Async Processing"
  ],
  "analysis": {
    "avg_score_a": 0.5515,
    "avg_score_b": 0.6109,
    "winner": "B (AI-Enhanced)",
    "overlapping_docs": [
      "doc_003",
      "doc_004"
    ],
    "unique_to_strategy_b": [
      "doc_006"
    ],
    "latency_a_ms": 22.37,
    "latency_b_ms": 38.86
  }
}
```

#### Strategy A — Raw Vector Search

- **Search query:** `What caching and database scaling strategies are used to ensure low latency?`
- **Latency:** 22.4 ms
- **Avg similarity score:** 0.5515

| Rank | Doc ID | Topic | Score | Text Preview |
|------|--------|-------|-------|--------------|
| 1 | doc_003 | Caching Strategies | 0.5887 | The platform employs a multi-tier caching architecture to minimize latency and r... |
| 2 | doc_004 | Database Scaling | 0.5708 | Database scaling is achieved through a combination of read replicas, connection ... |
| 3 | doc_002 | Auto-Scaling Policies | 0.4948 | Auto-scaling policies are governed by a rule engine that monitors CPU utilizatio... |

#### Strategy B — AI-Enhanced Retrieval

- **Expanded query:** `Describe the multi-tier caching architecture including in-process L1 cache, distributed L2 cache backed by Redis, cache ...`
- **Latency:** 38.9 ms
- **Avg similarity score:** 0.6109

| Rank | Doc ID | Topic | Score | Text Preview |
|------|--------|-------|-------|--------------|
| 1 | doc_003 | Caching Strategies | 0.8607 | The platform employs a multi-tier caching architecture to minimize latency and r... |
| 2 | doc_004 | Database Scaling | 0.5180 | Database scaling is achieved through a combination of read replicas, connection ... |
| 3 | doc_006 | Message Queue and Async Processing | 0.4541 | Asynchronous workloads are decoupled from the synchronous request path using a d... |

#### Analysis

| Metric | Value |
|--------|-------|
| Winner | **B (AI-Enhanced)** |
| Avg Score A | 0.5515 |
| Avg Score B | 0.6109 |
| Overlapping docs | ['doc_003', 'doc_004'] |
| New docs found by B | ['doc_006'] |

---

### Query 3: `How does the platform manage failures, retries, and fault tolerance across services?`

#### JSON Output

```json
{
  "query": "How does the platform manage failures, retries, and fault tolerance across services?",
  "strategy_a": [
    "doc_005",
    "doc_003",
    "doc_008"
  ],
  "strategy_b": [
    "doc_005",
    "doc_001",
    "doc_003"
  ],
  "scores_a": [
    0.6308,
    0.4247,
    0.421
  ],
  "scores_b": [
    0.7018,
    0.4348,
    0.4257
  ],
  "topics_a": [
    "Error Handling and Fault Tolerance",
    "Caching Strategies",
    "Observability and Monitoring"
  ],
  "topics_b": [
    "Error Handling and Fault Tolerance",
    "Peak Load Handling",
    "Caching Strategies"
  ],
  "analysis": {
    "avg_score_a": 0.4922,
    "avg_score_b": 0.5208,
    "winner": "B (AI-Enhanced)",
    "overlapping_docs": [
      "doc_003",
      "doc_005"
    ],
    "unique_to_strategy_b": [
      "doc_001"
    ],
    "latency_a_ms": 18.07,
    "latency_b_ms": 33.26
  }
}
```

#### Strategy A — Raw Vector Search

- **Search query:** `How does the platform manage failures, retries, and fault tolerance across services?`
- **Latency:** 18.1 ms
- **Avg similarity score:** 0.4922

| Rank | Doc ID | Topic | Score | Text Preview |
|------|--------|-------|-------|--------------|
| 1 | doc_005 | Error Handling and Fault Tolerance | 0.6308 | The system implements structured error handling at every layer of the stack. All... |
| 2 | doc_003 | Caching Strategies | 0.4247 | The platform employs a multi-tier caching architecture to minimize latency and r... |
| 3 | doc_008 | Observability and Monitoring | 0.4210 | The observability stack is built on three pillars: metrics, logs, and distribute... |

#### Strategy B — AI-Enhanced Retrieval

- **Expanded query:** `Describe the error handling and fault tolerance mechanisms including exponential backoff with jitter for retries, dead-l...`
- **Latency:** 33.3 ms
- **Avg similarity score:** 0.5208

| Rank | Doc ID | Topic | Score | Text Preview |
|------|--------|-------|-------|--------------|
| 1 | doc_005 | Error Handling and Fault Tolerance | 0.7018 | The system implements structured error handling at every layer of the stack. All... |
| 2 | doc_001 | Peak Load Handling | 0.4348 | The system handles peak load through a combination of horizontal scaling and int... |
| 3 | doc_003 | Caching Strategies | 0.4257 | The platform employs a multi-tier caching architecture to minimize latency and r... |

#### Analysis

| Metric | Value |
|--------|-------|
| Winner | **B (AI-Enhanced)** |
| Avg Score A | 0.4922 |
| Avg Score B | 0.5208 |
| Overlapping docs | ['doc_003', 'doc_005'] |
| New docs found by B | ['doc_001'] |

---

## Summary Table

| Query | Avg Score A | Avg Score B | Winner |
|-------|-------------|-------------|--------|
| How does the system handle peak load and traffic spikes... | 0.4726 | 0.5578 | B (AI-Enhanced) |
| What caching and database scaling strategies are used t... | 0.5515 | 0.6109 | B (AI-Enhanced) |
| How does the platform manage failures, retries, and fau... | 0.4922 | 0.5208 | B (AI-Enhanced) |

---

## Production Migration: Vertex AI Matching Engine

To migrate this pipeline from local FAISS to **Vertex AI Vector Search**
(Matching Engine) in production:

1. **Create a Matching Engine Index** in GCP with the same dimension (768 for gecko).
2. **Replace `FAISSVectorStore.add()`** with a call to upload embeddings to the index.
3. **Replace `FAISSVectorStore.search()`** with `endpoint.find_neighbors()` from
   `google.cloud.aiplatform`.
4. **Replace `VertexAIEmbedder`** mock with the real
   `TextEmbeddingModel.from_pretrained('textembedding-gecko@003')`.
5. **Replace `VertexAIQueryExpander`** mock with the real
   `GenerativeModel('gemini-1.5-pro')`.
6. The `RAGRetriever`, `StrategyResult`, and `ComparisonReport` classes
   remain **unchanged** — only the injected dependencies swap out.

This design follows the **Dependency Injection** pattern, making the
local ↔ cloud swap a one-line config change.

---

*Generated by main.py — context-aware-retrieval-engine*