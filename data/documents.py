DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Peak Load Handling",
        "text": (
            "The system handles peak load through a combination of horizontal scaling and "
            "intelligent traffic distribution. When incoming request rates exceed a predefined "
            "threshold — typically 80% of baseline capacity — the orchestration layer "
            "automatically provisions additional compute nodes within seconds. A weighted "
            "round-robin load balancer distributes traffic across all active nodes, ensuring "
            "no single instance becomes a bottleneck. Circuit breakers are engaged at the "
            "API gateway level to shed non-critical traffic during extreme spikes, preserving "
            "core service availability for high-priority workloads."
        ),
    },
    {
        "id": "doc_002",
        "topic": "Auto-Scaling Policies",
        "text": (
            "Auto-scaling policies are governed by a rule engine that monitors CPU utilization, "
            "memory pressure, request queue depth, and custom application metrics published to "
            "the telemetry pipeline. Scale-out events are triggered when the 90th-percentile "
            "latency crosses 200ms for more than 60 consecutive seconds. Scale-in events follow "
            "a conservative cooldown period of 5 minutes to prevent thrashing. Predictive "
            "scaling uses historical traffic patterns to pre-warm instances ahead of known "
            "demand windows, such as scheduled batch jobs or business-hours traffic ramps, "
            "reducing cold-start latency for end users."
        ),
    },
    {
        "id": "doc_003",
        "topic": "Caching Strategies",
        "text": (
            "The platform employs a multi-tier caching architecture to minimize latency and "
            "reduce downstream database pressure. An in-process L1 cache stores frequently "
            "accessed objects in heap memory with a TTL of 30 seconds. A distributed L2 cache "
            "backed by Redis Cluster holds serialized responses for up to 5 minutes, shared "
            "across all service replicas. Cache invalidation follows a write-through strategy "
            "for critical data and a lazy-eviction strategy for read-heavy reference data. "
            "Cache hit rates are monitored in real time; a hit rate below 70% triggers an "
            "automated alert for tuning review."
        ),
    },
    {
        "id": "doc_004",
        "topic": "Database Scaling",
        "text": (
            "Database scaling is achieved through a combination of read replicas, connection "
            "pooling, and sharding. Read-heavy workloads are routed to up to five read replicas "
            "via a query router that inspects transaction intent at the SQL AST level. Write "
            "operations are directed exclusively to the primary instance, which uses synchronous "
            "replication to the nearest replica for durability. For multi-tenant workloads, "
            "horizontal sharding partitions data by tenant ID hash, distributing rows evenly "
            "across shard groups. PgBouncer manages connection pooling in transaction mode, "
            "capping the total open connections to the database engine at 500 regardless of "
            "application concurrency."
        ),
    },
    {
        "id": "doc_005",
        "topic": "Error Handling and Fault Tolerance",
        "text": (
            "The system implements structured error handling at every layer of the stack. "
            "All service-to-service calls are wrapped with exponential backoff and jitter, "
            "retrying transient failures up to three times before propagating an error to the "
            "caller. Unrecoverable errors are classified by error code and routed to a dead-letter "
            "queue for asynchronous inspection. Each microservice exposes a health endpoint that "
            "reports dependency status; the orchestration layer removes unhealthy pods from the "
            "load-balancer pool within 10 seconds of a failed liveness probe. Chaos engineering "
            "tests are run weekly to validate fault-tolerance assumptions under simulated node, "
            "network, and dependency failures."
        ),
    },
    {
        "id": "doc_006",
        "topic": "Message Queue and Async Processing",
        "text": (
            "Asynchronous workloads are decoupled from the synchronous request path using a "
            "distributed message queue built on Apache Kafka. Producers publish events to "
            "partitioned topics, enabling parallel consumption by multiple consumer groups "
            "without message loss. Each consumer group maintains its own offset, allowing "
            "independent replay and reprocessing of historical events for audit or backfill "
            "scenarios. Message ordering is guaranteed within a single partition, making it "
            "suitable for event-sourcing patterns. The Kafka cluster is configured with a "
            "replication factor of three, ensuring no data loss in the event of a single "
            "broker failure."
        ),
    },
    {
        "id": "doc_007",
        "topic": "Security and Access Control",
        "text": (
            "Access control is enforced through a zero-trust security model where every "
            "request, whether internal or external, must present a valid JWT signed by the "
            "platform's identity provider. Role-based access control (RBAC) policies are "
            "evaluated at the API gateway before any request reaches a downstream service. "
            "Secrets such as database credentials and API keys are stored in a centralized "
            "secrets manager and injected into pods at runtime via environment variables, "
            "never baked into container images. All data in transit is encrypted with TLS 1.3, "
            "and data at rest is encrypted using AES-256. Security audit logs are streamed "
            "to an immutable log archive for compliance and forensic purposes."
        ),
    },
    {
        "id": "doc_008",
        "topic": "Observability and Monitoring",
        "text": (
            "The observability stack is built on three pillars: metrics, logs, and distributed "
            "traces. Prometheus scrapes service metrics at 15-second intervals and feeds "
            "Grafana dashboards used by on-call engineers. Structured JSON logs are shipped "
            "to a centralized log aggregator where they are indexed and queryable within "
            "seconds of emission. Distributed tracing via OpenTelemetry propagates trace "
            "context across all microservices, providing end-to-end visibility into request "
            "flows and identifying latency hotspots. Alerting rules are defined as code and "
            "versioned in Git, ensuring that alert thresholds are reviewed as part of the "
            "standard change management process."
        ),
    },
    {
        "id": "doc_009",
        "topic": "Data Pipeline and ETL",
        "text": (
            "The data pipeline ingests raw events from operational systems, transforms them "
            "into analytics-ready schemas, and loads them into a columnar data warehouse for "
            "reporting. The ETL process runs on an Apache Airflow DAG scheduler, with each "
            "task isolated in its own container for reproducibility and failure isolation. "
            "Incremental loads use watermark columns to process only new or updated records, "
            "minimizing compute cost. Data quality checks are embedded as pipeline stages; "
            "any batch failing a null-rate or schema-drift check is quarantined and triggers "
            "a Slack alert to the data engineering team. Full pipeline lineage is tracked "
            "and surfaced in an internal data catalog."
        ),
    },
    {
        "id": "doc_010",
        "topic": "Deployment and CI/CD",
        "text": (
            "The continuous delivery pipeline builds, tests, and deploys every merged pull "
            "request to a staging environment within 8 minutes. Canary deployments route 5% "
            "of production traffic to the new version for 15 minutes while automated smoke "
            "tests and error-rate monitors evaluate stability. If error rates increase by more "
            "than 0.5% relative to the baseline, an automated rollback is triggered without "
            "human intervention. Blue-green deployment is used for database schema migrations "
            "to allow instant cutover and rollback. All deployment configurations are stored "
            "as Kubernetes manifests in Git, and changes require two peer approvals before "
            "merging to the main branch."
        ),
    },
]


if __name__ == "__main__":
    print(f"Total documents loaded: {len(DOCUMENTS)}\n")
    for doc in DOCUMENTS:
        print(f"[{doc['id']}] {doc['topic']}")
        print(f"  {doc['text'][:80]}...")
        print()