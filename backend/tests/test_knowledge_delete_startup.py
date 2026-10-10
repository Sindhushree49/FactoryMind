"""Startup + DELETE /knowledge/documents/{id} no-content regression test."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.deps import DbSession
from app.main import app

calls: list[str] = []


def _fake_session() -> None:
    return None


def test_app_imports_and_openapi_generates() -> None:
    assert app.openapi()["paths"]["/api/knowledge/documents/{document_id}"]["delete"]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    import app.api.routes.knowledge as knowledge

    def fake_delete(session: object, document_id: str) -> bool:
        calls.append(document_id)
        return document_id != "missing"

    monkeypatch.setattr(knowledge.knowledge_service, "delete_document", fake_delete)
    app.dependency_overrides[DbSession] = _fake_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_delete_returns_204_with_empty_body(client: TestClient) -> None:
    response = client.delete("/api/knowledge/documents/doc-1")

    assert response.status_code == 204
    assert response.content == b""
    assert calls == ["doc-1"]


def test_delete_unknown_document_returns_404(client: TestClient) -> None:
    response = client.delete("/api/knowledge/documents/missing")

    assert response.status_code == 404
    body = response.json()
    assert body["message"] == "Document 'missing' not found"
    assert body["path"] == "/api/knowledge/documents/missing"