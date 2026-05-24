from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List

import faiss
import numpy as np


@dataclass
class SearchResult:
    rank: int
    doc_id: str
    topic: str
    text: str
    score: float

    def __repr__(self) -> str:
        return (
            f"SearchResult(rank={self.rank}, doc_id={self.doc_id!r}, "
            f"score={self.score:.4f}, topic={self.topic!r})"
        )

    def to_dict(self) -> Dict:
        return {
            "rank": self.rank,
            "doc_id": self.doc_id,
            "topic": self.topic,
            "score": round(self.score, 4),
            "text_preview": self.text[:120] + "...",
        }


class FAISSVectorStore:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self._id_map: Dict[int, Dict] = {}
        print(f"[FAISSVectorStore] Initialised — dimension={dimension}, metric=cosine (IP)")

    def add(self, documents: List[Dict], vectors: List[List[float]]) -> None:
        if len(documents) != len(vectors):
            raise ValueError(
                f"documents ({len(documents)}) and vectors ({len(vectors)}) "
                "must have the same length."
            )

        matrix = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(matrix)

        start_pos = self._index.ntotal
        self._index.add(matrix)

        for offset, doc in enumerate(documents):
            position = start_pos + offset
            self._id_map[position] = {
                "doc_id": doc["id"],
                "topic":  doc["topic"],
                "text":   doc["text"],
            }

        print(
            f"[FAISSVectorStore] Added {len(documents)} documents. "
            f"Total indexed: {self._index.ntotal}"
        )

    def search(self, query_vector: List[float], top_k: int = 3) -> List[SearchResult]:
        if self._index.ntotal == 0:
            raise RuntimeError("Vector store is empty. Call .add() first.")

        query = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results: List[SearchResult] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1:
                continue
            meta = self._id_map[int(idx)]
            results.append(
                SearchResult(
                    rank=rank,
                    doc_id=meta["doc_id"],
                    topic=meta["topic"],
                    text=meta["text"],
                    score=float(score),
                )
            )

        return results

    def save(self, directory: str = "index") -> None:
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "faiss.index")
        meta_path  = os.path.join(directory, "metadata.json")

        faiss.write_index(self._index, index_path)
        with open(meta_path, "w") as f:
            json.dump({str(k): v for k, v in self._id_map.items()}, f, indent=2)

        print(f"[FAISSVectorStore] Saved index → {index_path}")

    def load(self, directory: str = "index") -> None:
        index_path = os.path.join(directory, "faiss.index")
        meta_path  = os.path.join(directory, "metadata.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No index found at {index_path}")

        self._index = faiss.read_index(index_path)
        with open(meta_path, "r") as f:
            raw = json.load(f)
        self._id_map = {int(k): v for k, v in raw.items()}

        print(f"[FAISSVectorStore] Loaded {self._index.ntotal} vectors from {directory}/")

    @property
    def total_documents(self) -> int:
        return self._index.ntotal

    def __repr__(self) -> str:
        return (
            f"FAISSVectorStore(dimension={self.dimension}, "
            f"total_documents={self.total_documents})"
        )