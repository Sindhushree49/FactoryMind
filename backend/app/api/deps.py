"""Shared FastAPI dependencies and error translation."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.ml.failure_prediction.inference import ModelNotTrained
from app.rag.store import EmbeddingUnavailable
from app.services import ml_service
from app.services.maintenance_service import InvalidTransition

DbSession = Annotated[Session, Depends(get_db)]


def model_not_trained(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": "model_not_trained",
            "message": str(exc),
            "remediation": (
                "This endpoint serves a trained model. Train it first, then retry. "
                "See README 'Model training' for the exact command."
            ),
        },
    )


def invalid_transition(exc: InvalidTransition) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"error": "invalid_state_transition", "message": str(exc)},
    )


def embeddings_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "error": "retrieval_unavailable",
            "message": str(exc),
            "remediation": (
                "Install sentence-transformers and verify huggingface.co is reachable so "
                "the embedding model can be downloaded."
            ),
        },
    )


__all__ = [
    "DbSession",
    "EmbeddingUnavailable",
    "InvalidTransition",
    "ModelNotTrained",
    "embeddings_unavailable",
    "invalid_transition",
    "ml_service",
    "model_not_trained",
]
