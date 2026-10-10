"""Knowledge base (RAG) endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile, status

from app.api.deps import DbSession, embeddings_unavailable
from app.core.logging_config import get_logger
from app.rag import ingest as ingest_module
from app.rag.store import EmbeddingUnavailable
from app.schemas.api import (
    KnowledgeDocumentRead,
    KnowledgeQueryRequest,
    KnowledgeQueryResponse,
    KnowledgeUploadResponse,
)
from app.services import knowledge_service

logger = get_logger(__name__)
router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post(
    "/upload",
    response_model=KnowledgeUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
)
async def upload(
    session: DbSession,
    file: UploadFile = File(..., description="PDF, TXT, MD, RST, LOG or CSV"),
    title: str | None = Form(default=None),
) -> KnowledgeUploadResponse:
    """Extracts text, chunks it sentence-aware, embeds it and writes to the index."""
    data = await file.read()
    try:
        ingest_module.validate(file.filename or "document.pdf", data)
    except ingest_module.UnsupportedDocument as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    try:
        result = knowledge_service.upload_document(
            session, file.filename or "document", data, title=title
        )
    except ingest_module.UnsupportedDocument as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except EmbeddingUnavailable as exc:
        raise embeddings_unavailable(exc)

    return KnowledgeUploadResponse(**result)


@router.post(
    "/query",
    response_model=KnowledgeQueryResponse,
    summary="Ask a question against the knowledge base",
)
def query(session: DbSession, payload: KnowledgeQueryRequest) -> KnowledgeQueryResponse:
    """Retrieves passages and answers with mandatory source citations.

    Returns ``insufficient_evidence: true`` rather than an unsourced answer when
    nothing relevant is indexed.
    """
    try:
        result = knowledge_service.query_knowledge_base(
            session,
            question=payload.question,
            top_k=payload.top_k,
            document_id=payload.document_id,
            save_history=payload.save_history,
        )
    except EmbeddingUnavailable as exc:
        raise embeddings_unavailable(exc)

    from datetime import datetime, timezone

    return KnowledgeQueryResponse(**result, created_at=datetime.now(timezone.utc))


@router.get("/documents", response_model=list[KnowledgeDocumentRead], summary="Indexed documents")
def documents(session: DbSession) -> list[KnowledgeDocumentRead]:
    return [
        KnowledgeDocumentRead.model_validate(doc)
        for doc in knowledge_service.list_documents(session)
    ]


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a document",
)
def delete_document(document_id: str, session: DbSession) -> Response:
    if not knowledge_service.delete_document(session, document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found"
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
    if not knowledge_service.delete_document(session, document_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{document_id}' not found"
        )


@router.get("/status", summary="Vector index health")
def status_endpoint() -> dict[str, Any]:
    return knowledge_service.index_health()


@router.post("/reindex", summary="Re-embed every stored document")
def reindex(
    session: DbSession,
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    from app.rag.store import get_store

    store = get_store()
    if not store.embedding_available:
        raise embeddings_unavailable(
            EmbeddingUnavailable("Embedding backend unavailable; reindex is not possible.")
        )

    processed = 0
    failed = 0
    for doc in knowledge_service.list_documents(session)[:limit]:
        from pathlib import Path

        path = Path(doc.stored_path)
        if not path.is_file():
            failed += 1
            continue
        try:
            store.delete_document(doc.id)
            reingested = ingest_module.ingest(
                doc.filename, path.read_bytes(), title=doc.title, document_id=doc.id
            )
            store.add_chunks(reingested.chunks)
            doc.chunk_count = len(reingested.chunks)
            doc.status = "ingested"
            processed += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("Reindex failed for %s: %s", doc.filename, exc)
            doc.status = "failed"
            doc.error_message = str(exc)
            failed += 1
    session.flush()
    return {"processed": processed, "failed": failed, "index": store.stats()}
