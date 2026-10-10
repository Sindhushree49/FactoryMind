"""Application settings for FactoryMind.

All configuration is environment driven through ``backend/.env`` or the
repository-root ``.env``. Secrets are never hardcoded; API keys are read from
the environment only.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Typed application settings."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application ---
    app_name: str = "FactoryMind"
    app_env: str = "development"
    app_version: str = "0.1.0"
    api_prefix: str = "/api"
    log_level: str = "INFO"
    log_json: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # --- Database ---
    database_url: str = "sqlite:///./factorymind.db"
    db_echo: bool = False

    # --- Paths ---
    datasets_dir: Path = REPO_ROOT / "datasets"
    models_dir: Path = REPO_ROOT / "models"
    storage_dir: Path = REPO_ROOT / "storage"

    # --- LLM ---
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    llm_enabled: bool = False
    llm_max_tokens: int = 1200
    llm_temperature: float = 0.0

    # --- RAG ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 5
    rag_chunk_size: int = 900
    rag_chunk_overlap: int = 150

    # --- ML runtime thresholds ---
    anomaly_threshold: float = 0.62
    risk_watch_threshold: float = 0.30
    risk_high_threshold: float = 0.60
    risk_critical_threshold: float = 0.82
    torch_device: str = "cpu"
    random_seed: int = 42

    # --- Computer vision ---
    vision_image_size: int = 256
    vision_patch_size: int = 16
    vision_threshold: float = 0.5

    @field_validator("datasets_dir", "models_dir", "storage_dir", mode="after")
    @classmethod
    def _resolve(cls, value: Path) -> Path:
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: Any) -> Any:
        if isinstance(value, str):
            raw = value.strip().strip("[]")
            return [item.strip().strip("\"'") for item in raw.split(",") if item.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    # --- Derived paths -------------------------------------------------
    @property
    def ai4i_dir(self) -> Path:
        return self.datasets_dir / "ai4i"

    @property
    def cmapss_dir(self) -> Path:
        return self.datasets_dir / "nasa_cmapss" / "cmapss"

    @property
    def mvtec_dir(self) -> Path:
        return self.datasets_dir / "mvtec_ad" / "mvtec_anomaly_detection"

    @property
    def documents_dir(self) -> Path:
        return self.storage_dir / "documents"

    @property
    def reports_dir(self) -> Path:
        return self.storage_dir / "reports"

    @property
    def vision_outputs_dir(self) -> Path:
        return self.storage_dir / "vision"

    @property
    def manifest_path(self) -> Path:
        return self.datasets_dir / "manifests" / "datasets.json"

    # --- Derived runtime flags ------------------------------------------
    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def llm_active(self) -> bool:
        """LLM use requires both the opt-in flag and a real key."""
        return bool(self.llm_enabled and self.openai_api_key)

    def ensure_directories(self) -> None:
        for path in (
            self.models_dir,
            self.documents_dir,
            self.reports_dir,
            self.vision_outputs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
