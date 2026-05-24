from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock

from sentence_transformers import SentenceTransformer


@dataclass
class TextEmbeddingResponse:
    text: str
    values: List[float] = field(default_factory=list)

    def __repr__(self) -> str:
        preview = self.values[:4]
        return (
            f"TextEmbeddingResponse(text={self.text[:40]!r}..., "
            f"dim={len(self.values)}, values_preview={preview})"
        )


class LocalEmbedder:

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        print(f"[LocalEmbedder] Loading model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self._model.get_embedding_dimension()
        print(f"[LocalEmbedder] Ready — embedding dimension: {self.dimension}")

    def embed(self, text: str) -> List[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]


class VertexAIEmbedder:
    GCP_MODEL_NAME = "textembedding-gecko@003"

    def __init__(self) -> None:
        self._mock_sdk = MagicMock(name="vertexai")
        self._mock_sdk.init = MagicMock(return_value=None)
        self._local_embedder = LocalEmbedder()

        self._mock_sdk.language_models.TextEmbeddingModel.from_pretrained = (
            MagicMock(return_value=self)
        )

        print(
            f"[VertexAIEmbedder] Mocked '{self.GCP_MODEL_NAME}' "
            f"→ running locally via '{self._local_embedder.model_name}'"
        )

    @classmethod
    def from_pretrained(cls, model_name: str = GCP_MODEL_NAME) -> "VertexAIEmbedder":
        instance = cls()
        print(f"[VertexAIEmbedder] from_pretrained('{model_name}') called — mock active.")
        return instance

    def get_embeddings(self, texts: List[str]) -> List[TextEmbeddingResponse]:
        vectors = self._local_embedder.embed_batch(texts)
        return [
            TextEmbeddingResponse(text=text, values=vector)
            for text, vector in zip(texts, vectors)
        ]

    def embed(self, text: str) -> List[float]:
        return self.get_embeddings([text])[0].values

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [r.values for r in self.get_embeddings(texts)]

    @property
    def dimension(self) -> int:
        return self._local_embedder.dimension


def get_embedder(backend: str = "vertex") -> VertexAIEmbedder | LocalEmbedder:
    if backend == "vertex":
        return VertexAIEmbedder()
    elif backend == "local":
        return LocalEmbedder()
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'vertex' or 'local'.")