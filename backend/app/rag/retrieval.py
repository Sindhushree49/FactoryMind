"""Answer generation over retrieved chunks, with mandatory citations.

Two engines:

``extractive``
    Always available. Assembles the highest-scoring passages into an answer with
    explicit source references. It does not paraphrase beyond the source text.

``llm_synthesised``
    Only when ``settings.llm_active`` is true. The model receives the retrieved
    passages as its entire evidence base and is instructed to answer only from
    them. If the model returns a source marker that was not in the supplied
    context, that marker is stripped and the answer is downgraded to extractive.

If retrieval returns nothing above threshold, the answer says so. The system
never fills the gap with un-sourced prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.core.logging_config import get_logger
from app.rag.store import SearchHit, get_store

logger = get_logger(__name__)

# Below this cosine similarity a passage is not considered evidence.
RELEVANCE_FLOOR = 0.15

DISCLAIMER = (
    "Answers are generated only from the documents indexed in this knowledge base. "
    "Verify any procedure against the original manufacturer document before work begins."
)

SYSTEM_PROMPT = """You are a maintenance documentation assistant for an industrial \
predictive-maintenance platform.

Rules you must follow:
1. Answer ONLY from the numbered context passages below. They are the entire \
evidence base available to you.
2. Cite every factual claim inline as [n], where n is the passage number.
3. If the passages do not contain the answer, reply exactly with \
"INSUFFICIENT_EVIDENCE" and nothing else. Do not speculate.
4. Never invent document names, section numbers, part numbers, torque values, or \
procedures. If a specific value is not in the passages, say it is not documented.
5. Do not diagnose equipment as safe or unsafe. Provide documented guidance only, \
and defer the final decision to a qualified technician.
6. Be concise and technical. Use bullet points when listing steps."""

NO_EVIDENCE_ANSWER = (
    "The knowledge base does not contain passages relevant enough to answer this "
    "question. Upload the relevant manual or procedure document, then ask again."
)


@dataclass
class Answer:
    question: str
    answer: str
    engine: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    insufficient_evidence: bool = False
    disclaimer: str = DISCLAIMER
    retrieved_count: int = 0
    evidence_scores: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "answer_engine": self.engine,
            "citations": self.citations,
            "retrieved_count": self.retrieved_count,
            "insufficient_evidence": self.insufficient_evidence,
            "disclaimer": self.disclaimer,
            "evidence_scores": self.evidence_scores,
        }


def retrieve(
    question: str,
    top_k: int | None = None,
    document_id: str | None = None,
) -> list[SearchHit]:
    store = get_store()
    if not store.embedding_available:
        logger.warning("Retrieval skipped: no embedding backend available")
        return []
    hits = store.search(question, top_k=top_k or settings.rag_top_k, document_id=document_id)
    return [hit for hit in hits if hit.score >= RELEVANCE_FLOOR]


def answer(question: str, top_k: int | None = None, document_id: str | None = None) -> Answer:
    hits = retrieve(question, top_k=top_k, document_id=document_id)
    if not hits:
        return Answer(
            question=question,
            answer=NO_EVIDENCE_ANSWER,
            engine="extractive",
            insufficient_evidence=True,
            retrieved_count=0,
        )

    citations = [_citation(index, hit) for index, hit in enumerate(hits, start=1)]

    if settings.llm_active:
        try:
            text = _llm_answer(question, hits)
            if text is not None:
                verified, hallucinated = _verify_citations(text, citations)
                if hallucinated:
                    logger.warning(
                        "LLM answer cited %d marker(s) absent from the retrieved context; "
                        "those markers were stripped and the answer was downgraded.",
                        len(hallucinated),
                    )
                return Answer(
                    question=question,
                    answer=verified,
                    engine="extractive" if hallucinated else "llm_synthesised",
                    citations=citations,
                    retrieved_count=len(hits),
                    evidence_scores=_scores(hits),
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM answer generation failed (%s); using extractive fallback", exc)

    return Answer(
        question=question,
        answer=_extractive_answer(question, hits),
        engine="extractive",
        citations=citations,
        retrieved_count=len(hits),
        evidence_scores=_scores(hits),
    )


def _citation(index: int, hit: SearchHit) -> dict[str, Any]:
    page = f", p.{hit.chunk.page}" if hit.chunk.page else ""
    return {
        "index": index,
        "document_id": hit.chunk.document_id,
        "document_title": hit.chunk.document_title,
        "filename": hit.chunk.filename,
        "page": hit.chunk.page,
        "reference": f"{hit.chunk.document_title or hit.chunk.filename}{page}, "
                     f"chunk {hit.chunk.chunk_id}",
        "score": round(hit.score, 4),
    }


def _scores(hits: list[SearchHit]) -> list[dict[str, Any]]:
    return [
        {"reference": f"{h.chunk.document_title or h.chunk.filename}", "score": round(h.score, 4)}
        for h in hits
    ]


def _extractive_answer(question: str, hits: list[SearchHit]) -> str:
    """Assemble the best passages verbatim with source markers."""
    lines = [
        f"No language model is configured, so FactoryMind is returning the most relevant "
        f"passages verbatim rather than synthesising an answer. "
        f"Top {min(len(hits), 3)} matches for: \"{question}\"\n"
    ]
    for index, hit in enumerate(hits[:3], start=1):
        page = f" (page {hit.chunk.page})" if hit.chunk.page else ""
        excerpt = _trim(hit.chunk.text, 700)
        lines.append(
            f"[{index}] {hit.chunk.document_title or hit.chunk.filename}{page} "
            f"- relevance {hit.score:.3f}\n{excerpt}\n"
        )
    lines.append(
        "Set OPENAI_API_KEY and LLM_ENABLED=true in .env to enable synthesised answers "
        "grounded on these same passages."
    )
    return "\n".join(lines)


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


def _llm_answer(question: str, hits: list[SearchHit]) -> str | None:
    """Call the LLM with the retrieved passages as the only context."""
    import re

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
    except ImportError:
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("No LLM client library available; install langchain-openai or openai")
            return None
        client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=45.0,
            max_retries=2,
        )
        context = _context_block(hits)
        response = client.chat.completions.create(
            model=settings.openai_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context passages:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return response.choices[0].message.content or ""

    model = ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.openai_api_key,
        timeout=45,
        max_retries=2,
    )
    context = _context_block(hits)
    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context passages:\n{context}\n\nQuestion: {question}"),
        ]
    )
    text = response.content if isinstance(response.content, str) else str(response.content)
    if re.fullmatch(r"\s*INSUFFICIENT_EVIDENCE\s*", text, flags=re.IGNORECASE):
        return NO_EVIDENCE_ANSWER
    return text


def _context_block(hits: list[SearchHit]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        page = f", page {hit.chunk.page}" if hit.chunk.page else ""
        blocks.append(
            f"[{index}] source: {hit.chunk.document_title or hit.chunk.filename}{page}\n{hit.chunk.text}"
        )
    return "\n\n".join(blocks)


def _verify_citations(text: str, citations: list[dict[str, Any]]) -> tuple[str, list[int]]:
    """Strip citation markers that reference passages not actually supplied."""
    import re

    valid = {c["index"] for c in citations}
    used: list[int] = []
    stripped: list[int] = []

    def replace(match: re.Match[str]) -> str:
        marker = int(match.group(1))
        if marker in valid:
            used.append(marker)
            return f"[{marker}]"
        stripped.append(marker)
        return ""

    cleaned = re.sub(r"\[(\d{1,3})\]", replace, text)
    if stripped:
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned, sorted(set(stripped))
    return cleaned, []
