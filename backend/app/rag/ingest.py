"""PDF/text ingestion with page-aware chunking."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging_config import get_logger
from app.rag.store import Chunk

logger = get_logger(__name__)

MAX_FILE_BYTES = 40 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".rst", ".log", ".csv"}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WHITESPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


class UnsupportedDocument(ValueError):
    pass


@dataclass
class IngestedDocument:
    document_id: str
    filename: str
    title: str | None
    stored_path: str
    page_count: int | None
    char_count: int
    chunks: list[Chunk]

    def summary(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "title": self.title,
            "stored_path": self.stored_path,
            "page_count": self.page_count,
            "char_count": self.char_count,
            "chunk_count": len(self.chunks),
        }


def validate(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UnsupportedDocument(
            f"Unsupported file type '{suffix or filename}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}"
        )
    if not data:
        raise UnsupportedDocument("Uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise UnsupportedDocument(
            f"File is {len(data) / 1e6:.1f} MB; the limit is {MAX_FILE_BYTES / 1e6:.0f} MB."
        )
    return suffix


def _normalise(text: str) -> str:
    text = text.replace("\x00", " ")
    text = _WHITESPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def extract_pages(data: bytes, suffix: str) -> list[tuple[int | None, str]]:
    """Return ``(page_number, text)`` pairs. Page 1-indexed for PDFs, None otherwise."""
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(_as_stream(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise UnsupportedDocument(f"PDF is encrypted and cannot be read: {exc}") from exc
        pages: list[tuple[int | None, str]] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning("Page %d of PDF could not be extracted: %s", index, exc)
                text = ""
            pages.append((index, _normalise(text)))
        return pages

    return [(None, _normalise(data.decode("utf-8", errors="replace")))]


def _as_stream(data: bytes) -> Any:
    import io

    return io.BytesIO(data)


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Sentence-aware chunking with overlap.

    Splits on sentence boundaries where possible so a chunk does not cut a
    procedure step in half, then falls back to hard slices for very long
    unbroken text.
    """
    chunk_size = chunk_size or settings.rag_chunk_size
    overlap = overlap if overlap is not None else settings.rag_chunk_overlap
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            for start in range(0, len(sentence), chunk_size):
                chunks.append(sentence[start : start + chunk_size].strip())
            continue
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current.strip())
            tail = current[-overlap:].strip() if overlap else ""
            current = f"{tail} {sentence}".strip() if tail else sentence
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 20]


def ingest(
    filename: str,
    data: bytes,
    title: str | None = None,
    document_id: str | None = None,
) -> IngestedDocument:
    suffix = validate(filename, data)
    pages = extract_pages(data, suffix)
    resolved_title = title or Path(filename).stem.replace("_", " ").strip()

    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    document_id = document_id or uuid.uuid4().hex
    stored_path = settings.documents_dir / f"{document_id}{suffix}"
    stored_path.write_bytes(data)

    chunks: list[Chunk] = []
    char_start = 0
    raw_text_parts: list[str] = []
    for page_number, page_text in pages:
        raw_text_parts.append(page_text)
        for piece in chunk_text(page_text):
            chunks.append(
                Chunk(
                    chunk_id=-1,   # assigned by the vector store
                    document_id=document_id,
                    filename=filename,
                    document_title=resolved_title,
                    page=page_number,
                    text=piece,
                    char_start=char_start,
                )
            )
        char_start += len(page_text)

    full_text = "\n\n".join(t for t in raw_text_parts if t).strip()
    if not chunks:
        raise UnsupportedDocument(
            "No extractable text found. The file may be a scanned image PDF; OCR is not "
            "configured in FactoryMind."
        )

    return IngestedDocument(
        document_id=document_id,
        filename=filename,
        title=resolved_title,
        stored_path=str(stored_path),
        page_count=len(pages) if pages and pages[0][0] is not None else None,
        char_count=len(full_text),
        chunks=chunks,
    )
