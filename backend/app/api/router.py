"""Aggregated API router."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import agents, health, knowledge, machines, operations, predictions, vision

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(machines.router)
api_router.include_router(predictions.router)
api_router.include_router(vision.router)
api_router.include_router(agents.router)
api_router.include_router(knowledge.router)
api_router.include_router(operations.router)

__all__ = ["api_router"]
