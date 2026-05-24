from __future__ import annotations

import re
from typing import Dict, List
from unittest.mock import MagicMock


EXPANSION_RULES: List[Dict] = [
    {
        "keywords": ["peak load", "peak traffic", "high traffic", "traffic spike"],
        "expansion": (
            "Describe the mechanisms for handling high traffic spikes and peak load conditions, "
            "including load balancing strategies, horizontal scaling, request queuing, "
            "circuit breakers, auto-scaling triggers, and system performance under maximum "
            "concurrent user demand."
        ),
    },
    {
        "keywords": ["auto-scal", "autoscal", "scale out", "scale in", "scaling policy"],
        "expansion": (
            "Explain the auto-scaling policies and rules that govern when the system provisions "
            "or removes compute resources, including CPU thresholds, memory pressure, latency "
            "percentile triggers, cooldown periods, predictive scaling based on historical "
            "traffic patterns, and pre-warming of instances before expected demand spikes."
        ),
    },
    {
        "keywords": ["cache", "caching", "redis", "in-memory", "ttl"],
        "expansion": (
            "Describe the multi-tier caching architecture including in-process L1 cache, "
            "distributed L2 cache backed by Redis, cache TTL settings, write-through and "
            "lazy-eviction invalidation strategies, cache hit rate monitoring, and how "
            "caching reduces downstream database pressure and minimises latency."
        ),
    },
    {
        "keywords": ["database", "db scaling", "read replica", "shard", "connection pool"],
        "expansion": (
            "Explain the database scaling strategies including horizontal sharding by tenant ID, "
            "read replicas for query distribution, synchronous replication for durability, "
            "connection pooling with PgBouncer, query routing based on transaction intent, "
            "and how write operations are isolated to the primary instance."
        ),
    },
    {
        "keywords": ["error", "fault", "failure", "retry", "resilience", "circuit breaker"],
        "expansion": (
            "Describe the error handling and fault tolerance mechanisms including exponential "
            "backoff with jitter for retries, dead-letter queues for unrecoverable errors, "
            "circuit breakers at the API gateway, health endpoint monitoring, liveness probes, "
            "pod removal from load-balancer pools, and weekly chaos engineering tests."
        ),
    },
    {
        "keywords": ["message queue", "kafka", "async", "event", "consumer", "producer"],
        "expansion": (
            "Explain the asynchronous messaging architecture using Apache Kafka, including "
            "partitioned topics, consumer groups with independent offsets, message ordering "
            "guarantees within partitions, event replay for backfill scenarios, replication "
            "factor for fault tolerance, and decoupling of asynchronous workloads from the "
            "synchronous request path."
        ),
    },
    {
        "keywords": ["security", "access control", "jwt", "rbac", "encryption", "secret"],
        "expansion": (
            "Describe the zero-trust security model including JWT authentication, role-based "
            "access control evaluated at the API gateway, secrets management with runtime "
            "injection into pods, TLS 1.3 encryption for data in transit, AES-256 encryption "
            "for data at rest, and immutable security audit log streaming for compliance."
        ),
    },
    {
        "keywords": ["monitor", "observ", "metric", "log", "trace", "alert", "prometheus"],
        "expansion": (
            "Explain the observability stack covering the three pillars: metrics scraped by "
            "Prometheus and visualised in Grafana, structured JSON logs indexed in a centralised "
            "aggregator, distributed tracing with OpenTelemetry for end-to-end request visibility, "
            "and alerting rules versioned in Git as part of the change management process."
        ),
    },
    {
        "keywords": ["pipeline", "etl", "data ingestion", "airflow", "warehouse", "batch"],
        "expansion": (
            "Describe the ETL data pipeline architecture including Apache Airflow DAG scheduling, "
            "containerised task isolation, incremental loads using watermark columns, embedded "
            "data quality checks for null-rate and schema-drift, quarantine procedures for "
            "failing batches, and full pipeline lineage tracking in an internal data catalog."
        ),
    },
    {
        "keywords": ["deploy", "ci/cd", "canary", "rollback", "blue-green", "kubernetes"],
        "expansion": (
            "Explain the CI/CD and deployment strategy including automated build and test pipelines, "
            "canary deployments routing 5% of traffic to new versions, automated rollback on "
            "error-rate increase, blue-green deployment for database schema migrations, "
            "Kubernetes manifests stored as code in Git, and peer-review requirements before "
            "merging to the main branch."
        ),
    },
]

FALLBACK_TEMPLATE = (
    "Provide a detailed technical explanation of: {query}. "
    "Include relevant system components, underlying mechanisms, performance implications, "
    "failure modes, and how this aspect integrates with the broader platform architecture."
)


class _LocalQueryExpander:

    def expand(self, query: str) -> str:
        query_lower = query.lower()
        for rule in EXPANSION_RULES:
            for keyword in rule["keywords"]:
                if keyword in query_lower:
                    return rule["expansion"]
        return FALLBACK_TEMPLATE.format(query=query.strip("?").strip())


class VertexAIQueryExpander:
    GCP_MODEL_NAME = "gemini-2.5-pro"

    def __init__(self) -> None:
        self._mock_vertexai = MagicMock(name="vertexai")
        self._mock_vertexai.init = MagicMock(return_value=None)
        self._local_expander = _LocalQueryExpander()

        self._mock_model = MagicMock(name="GenerativeModel")
        self._mock_model.generate_content.side_effect = self._generate_content
        self._mock_vertexai.generative_models.GenerativeModel.return_value = self._mock_model

        print(
            f"[VertexAIQueryExpander] Mocked '{self.GCP_MODEL_NAME}' "
            "→ running locally via rule-based expander"
        )

    def _generate_content(self, prompt: str) -> MagicMock:
        lines = [l.strip() for l in prompt.strip().splitlines() if l.strip()]
        raw_query = lines[-1] if lines else prompt
        expanded = self._local_expander.expand(raw_query)

        mock_response = MagicMock(name="GenerateContentResponse")
        mock_response.text = expanded
        return mock_response

    def generate_content(self, prompt: str) -> MagicMock:
        return self._generate_content(prompt)

    def expand(self, query: str) -> str:
        prompt = (
            "You are an expert query expansion assistant for a technical RAG system.\n"
            "Rewrite the following query into a richer, more descriptive version that\n"
            "captures the key technical concepts, mechanisms, and related terms.\n"
            "Output only the expanded query with no preamble.\n\n"
            f"Query: {query}"
        )
        response = self.generate_content(prompt)
        return response.text

    def __repr__(self) -> str:
        return f"VertexAIQueryExpander(model={self.GCP_MODEL_NAME!r}, backend='local-rule-based')"