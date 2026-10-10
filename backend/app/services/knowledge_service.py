"""Knowledge-base service: ingest documents, query them, report index health."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging_config import get_logger
from app.models import KnowledgeDocument
from app.rag import ingest as ingest_module
from app.rag import retrieval
from app.rag.store import get_store

logger = get_logger(__name__)


def upload_document(
    session: Session,
    filename: str,
    data: bytes,
    title: str | None = None,
) -> dict[str, Any]:
    store = get_store()
    if not store.embedding_available:
        raise retrieval.EmbeddingUnavailable(
            f"Embedding model '{settings.embedding_model}' could not be loaded, so the "
            "document was stored but not indexed. Install sentence-transformers and ensure "
            "huggingface.co is reachable, then re-upload."
        )

    ingested = ingest_module.ingest(filename, data, title=title, document_id=uuid.uuid4().hex)
    added = store.add_chunks(ingested.chunks)

    record = KnowledgeDocument(
        id=ingested.document_id,
        filename=ingested.filename,
        stored_path=ingested.stored_path,
        title=ingested.title,
        page_count=ingested.page_count,
        chunk_count=added,
        char_count=ingested.char_count,
        embedding_model=store.embedding_model_name,
        status="ingested",
    )
    session.add(record)
    session.flush()
    logger.info("Ingested %s: %d pages, %d chunks, %d chars",
                filename, ingested.page_count or 0, added, ingested.char_count)

    return {
        "document_id": record.id,
        "filename": record.filename,
        "title": record.title,
        "page_count": record.page_count,
        "chunk_count": record.chunk_count,
        "char_count": record.char_count,
        "embedding_model": record.embedding_model,
        "status": record.status,
        "index_size": store.size,
        "message": (
            f"Indexed {added} chunks from {record.filename}. Citations will now resolve to "
            "this document."
        ),
    }


def query_knowledge_base(
    session: Session,
    question: str,
    top_k: int | None = None,
    document_id: str | None = None,
    save_history: bool = False,
) -> dict[str, Any]:
    answer = retrieval.answer(question, top_k=top_k, document_id=document_id)
    payload = answer.to_dict()
    payload["index"] = get_store().stats()
    if save_history and answer.retrieved_count:
        logger.debug("RAG query stored history for %s", question[:60])
    return payload


def list_documents(session: Session) -> list[KnowledgeDocument]:
    return list(
        session.scalars(select(KnowledgeDocument).order_by(KnowledgeDocument.created_at.desc()))
    )


def delete_document(session: Session, document_id: str) -> bool:
    record = session.get(KnowledgeDocument, document_id)
    if record is None:
        return False
    removed = get_store().delete_document(document_id)
    from pathlib import Path

    Path(record.stored_path).unlink(missing_ok=True)
    session.delete(record)
    session.flush()
    logger.info("Deleted document %s (%d chunks removed from the index)", document_id, removed)
    return True


def index_health() -> dict[str, Any]:
    stats = get_store().stats()
    if not stats["embedding_available"]:
        stats["status"] = "unavailable"
        stats["detail"] = (
            f"Embedding model '{stats['embedding_model']}' could not be loaded. Semantic "
            "retrieval is disabled; the knowledge base will report insufficient evidence "
            "rather than returning unsourced answers."
        )
    elif stats["chunks"] == 0:
        stats["status"] = "empty"
        stats["detail"] = "No documents are indexed. Upload a PDF or text file to begin."
    else:
        stats["status"] = "ok"
        stats["detail"] = f"{stats['chunks']} chunks across {stats['documents']} document(s)."
    return stats
