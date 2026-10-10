"""Vector store for the maintenance knowledge base.

FAISS is used when importable; otherwise a pure-NumPy exact-cosine index takes
over so retrieval still works without the native dependency. Both paths share
the same interface and the same persisted-on-disk format.

Stored chunks always carry their provenance (document, page, chunk id) because
every generated answer must cite real sources.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

INDEX_DIR = settings.storage_dir / "vector_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "chunks.jsonl"


@dataclass
class Chunk:
    chunk_id: int
    document_id: str
    filename: str
    document_title: str | None
    page: int | None
    text: str
    char_start: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchHit:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "document_id": self.chunk.document_id,
            "document_title": self.chunk.document_title,
            "filename": self.chunk.filename,
            "page": self.chunk.page,
            "score": round(self.score, 6),
            "text": self.chunk.text,
        }


class EmbeddingUnavailable(RuntimeError):
    """Raised when no embedding backend can be constructed."""


class VectorStore:
    """Thread-safe in-process vector index with lazy embedding."""

    _instance: "VectorStore | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, embedding_model: str | None = None) -> None:
        self.embedding_model_name = embedding_model or settings.embedding_model
        self._encoder: Any = None
        self._encoder_failed = False
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._index: Any = None
        self._lock = threading.RLock()
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    # -- singleton --------------------------------------------------------
    @classmethod
    def instance(cls) -> "VectorStore":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    # -- embeddings -------------------------------------------------------
    def _get_encoder(self) -> Any:
        if self._encoder is not None or self._encoder_failed:
            return self._encoder
        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model_name)
            logger.info("Loaded embedding model %s", self.embedding_model_name)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Embedding model %s unavailable (%s). Retrieval is disabled; the knowledge "
                "base will report insufficient evidence instead of inventing citations.",
                self.embedding_model_name, exc,
            )
            self._encoder_failed = True
        return self._encoder

    @property
    def embedding_available(self) -> bool:
        return self._get_encoder() is not None

    def embed(self, texts: list[str]) -> np.ndarray:
        encoder = self._get_encoder()
        if encoder is None:
            raise EmbeddingUnavailable(
                f"Embedding model '{self.embedding_model_name}' could not be loaded. "
                "Run: pip install sentence-transformers  (and check network access to "
                "huggingface.co for the first-time weight download)."
            )
        vectors = encoder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)

    # -- write ------------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embed([c.text for c in chunks])
        with self._lock:
            start = len(self._chunks)
            for offset, chunk in enumerate(chunks):
                chunk.chunk_id = start + offset
            self._chunks.extend(chunks)
            self._vectors = (
                vectors if self._vectors is None else np.vstack([self._vectors, vectors])
            )
            self._rebuild_index()
        self._persist()
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            keep = [i for i, c in enumerate(self._chunks) if c.document_id != document_id]
            removed = len(self._chunks) - len(keep)
            if removed == 0:
                return 0
            self._chunks = [self._chunks[i] for i in keep]
            self._vectors = self._vectors[keep] if self._vectors is not None else None
            self._rebuild_index()
        self._persist()
        return removed

    def _rebuild_index(self) -> None:
        try:
            import faiss

            dimension = self._vectors.shape[1]
            index = faiss.IndexFlatIP(dimension)   # vectors are L2-normalised
            index.add(self._vectors)
            self._index = index
        except Exception as exc:  # noqa: BLE001
            logger.warning("FAISS unavailable (%s); using exact NumPy search", exc)
            self._index = None

    # -- read -------------------------------------------------------------
    def search(self, query: str, top_k: int = 5, document_id: str | None = None) -> list[SearchHit]:
        with self._lock:
            if not self._chunks or self._vectors is None:
                return []
            query_vector = self.embed([query])[0]
            candidates = np.arange(len(self._chunks))
            matrix = self._vectors
            if document_id:
                candidates = np.array(
                    [i for i, c in enumerate(self._chunks) if c.document_id == document_id],
                    dtype=int,
                )
                if candidates.size == 0:
                    return []
                matrix = self._vectors[candidates]

            if self._index is not None and document_id is None:
                k = min(top_k, matrix.shape[0])
                scores, indices = self._index.search(query_vector.reshape(1, -1), k)
                pairs = zip(indices[0], scores[0])
            else:
                similarities = matrix @ query_vector
                order = np.argsort(similarities)[::-1][:top_k]
                pairs = ((candidates[i], similarities[i]) for i in order)

            return [
                SearchHit(chunk=self._chunks[i], score=float(s))
                for i, s in pairs
                if 0 <= i < len(self._chunks)
            ]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._chunks)

    def documents(self) -> set[str]:
        with self._lock:
            return {c.document_id for c in self._chunks}

    def chunks_for(self, document_id: str) -> list[Chunk]:
        with self._lock:
            return [c for c in self._chunks if c.document_id == document_id]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "chunks": len(self._chunks),
                "documents": len({c.document_id for c in self._chunks}),
                "embedding_model": self.embedding_model_name,
                "embedding_available": self.embedding_available,
                "backend": "faiss" if self._index is not None else "numpy",
                "dimension": int(self._vectors.shape[1]) if self._vectors is not None else None,
                "index_path": str(INDEX_DIR),
            }

    # -- persistence ------------------------------------------------------
    def _persist(self) -> None:
        try:
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            if self._vectors is not None:
                np.save(INDEX_DIR / "vectors.npy", self._vectors)
            with METADATA_FILE.open("w", encoding="utf-8") as handle:
                for chunk in self._chunks:
                    handle.write(json.dumps(chunk.to_dict()) + "\n")
        except OSError as exc:
            logger.warning("Could not persist the vector index: %s", exc)

    def _load(self) -> None:
        vectors_file = INDEX_DIR / "vectors.npy"
        if not METADATA_FILE.is_file() or not vectors_file.is_file():
            return
        try:
            self._chunks = [
                Chunk(**json.loads(line))
                for line in METADATA_FILE.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self._vectors = np.load(vectors_file)
            self._rebuild_index()
            logger.info("Loaded %d knowledge chunks from disk", len(self._chunks))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ignoring unreadable vector index: %s", exc)
            self._chunks, self._vectors = [], None


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore.instance()
    return _store


def reset_store() -> None:
    """Drop the cached store. Used by tests."""
    global _store
    _store = None
    VectorStore.reset()
    Path(INDEX_DIR / "vectors.npy").unlink(missing_ok=True)
    METADATA_FILE.unlink(missing_ok=True)
