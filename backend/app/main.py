"""FactoryMind FastAPI application factory."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.router import api_router
from app.core.config import settings
from app.core.logging_config import configure_logging, get_logger
from app.database.session import create_all

logger = get_logger(__name__)

DESCRIPTION = """
**FactoryMind** - agentic AI for predictive maintenance and industrial intelligence.

The API serves trained models, not estimates. Every endpoint that depends on a
model returns `503 model_not_trained` with a remediation command when that model
has not been trained, rather than a fabricated number.

### Workflow

`data ingestion -> preprocessing -> failure prediction -> anomaly detection ->
RUL estimation -> visual inspection -> agentic root-cause investigation ->
maintenance planning -> verification -> human approval`

### Datasets

| Dataset | Role | Licence |
| --- | --- | --- |
| AI4I 2020 (UCI 601) | Failure prediction, failure-mode classification | CC BY 4.0 |
| NASA C-MAPSS | Remaining Useful Life regression | US Government public domain |
| MVTec AD | Visual anomaly detection | CC BY-NC-SA 4.0 (manual registration) |

Datasets are never merged. Each has its own loader, pipeline and models.

### Units

AI4I temperatures are **Kelvin**, matching the published schema
(nominal air 300 K, process 310 K). Torque is Nm, speed rpm, tool wear minutes.
"""

TAGS_METADATA = [
    {"name": "system", "description": "Service health and dependency status."},
    {"name": "machines", "description": "Machine registry and sensor telemetry."},
    {"name": "ml", "description": "Failure prediction, anomaly detection, RUL, model metrics."},
    {"name": "vision", "description": "MVTec AD visual inspection and defect localisation."},
    {"name": "agents", "description": "Six-agent LangGraph investigation workflow."},
    {"name": "knowledge", "description": "RAG document ingestion, retrieval and citations."},
    {"name": "operations", "description": "Dashboard, maintenance workflow, alerts, digital twin."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings.ensure_directories()
    logger.info(
        "%s %s starting (env=%s, db=%s, llm=%s)",
        settings.app_name, settings.app_version, settings.app_env,
        "sqlite" if settings.is_sqlite else "postgresql",
        "llm" if settings.llm_active else "rule_based",
    )
    from app.ml import datasets as dataset_registry
    from app.ml.bootstrap import ensure_models

    try:
        create_all()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Database initialization failed: %s: %s", type(exc).__name__, exc,
            exc_info=True,
        )
        raise
    else:
        logger.info(
            "database schema ready (%s)", "sqlite" if settings.is_sqlite else "postgresql"
        )

    for key, dataset in dataset_registry.all_statuses().items():
        level = logger.info if dataset.available else logger.warning
        level("dataset %-7s %-8s %s", key, "OK" if dataset.available else "MISSING", dataset.detail)

    try:
        await asyncio.to_thread(ensure_models)
    except Exception as exc:  # noqa: BLE001
        logger.error("Model initialization failed: %s", exc, exc_info=True)

    yield
    logger.info("%s shutting down", settings.app_name)


def create_app() -> FastAPI:
    app = FastAPI(
        title="FactoryMind API",
        description=DESCRIPTION,
        version=settings.app_version,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error on %s %s", request.method, request.url.path)
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "internal_error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "request_id": request_id,
                },
            )
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration_ms}ms"
        logger.info(
            "%s %s -> %s in %.2fms", request.method, request.url.path,
            response.status_code, duration_ms,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "The request body or query parameters failed validation.",
                "details": exc.errors(),
                "path": request.url.path,
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict):
            payload = {**detail, "path": request.url.path}
        else:
            payload = {"error": "http_error", "message": str(detail), "path": request.url.path}
        return JSONResponse(status_code=exc.status_code, content=payload, headers=exc.headers)

    app.include_router(api_router, prefix=settings.api_prefix)

    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/storage", StaticFiles(directory=str(settings.storage_dir)), name="storage")

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, Any]:
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": f"{settings.api_prefix}/health",
        }

    return app


app = create_app()
