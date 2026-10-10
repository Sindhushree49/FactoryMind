"""Health endpoint and database bootstrap regression tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect

from app.core.config import settings
from app.database.session import engine
from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_startup_creates_every_table(client: TestClient) -> None:
    assert client.get("/api/health").status_code in (200, 503)

    tables = set(inspect(engine).get_table_names())
    expected = {
        "users",
        "machines",
        "sensor_readings",
        "failure_predictions",
        "anomaly_events",
        "rul_predictions",
        "visual_inspections",
        "agent_investigations",
        "maintenance_recommendations",
        "maintenance_history",
        "knowledge_documents",
        "model_versions",
        "alerts",
    }
    assert expected <= tables, f"missing tables: {sorted(expected - tables)}"


def test_health_reports_model_components(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code in (200, 503)
    payload = response.json()
    names = {component["name"] for component in payload["components"]}
    assert {
        "database",
        "failure_prediction_model",
        "anomaly_detection_model",
        "rul_prediction_model",
        "computer_vision_model",
        "llm_engine",
    } <= names
    assert payload["version"] == settings.app_version


def test_knowledge_documents_endpoint_reads_real_db(client: TestClient) -> None:
    response = client.get("/api/knowledge/documents")

    assert response.status_code == 200
    assert isinstance(response.json(), list)