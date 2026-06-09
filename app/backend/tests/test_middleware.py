"""Tests for request-context middleware: API key, rate limit, request id.

Each test imports `main` fresh with a custom env so module-level state
(_request_windows, API_KEY) starts clean.
"""

import importlib
import logging
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


def _fresh_app(monkeypatch, env: dict[str, str]):
    """Import `main` with the given env and return (app, module)."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import main  # noqa: WPS433
    importlib.reload(main)
    main.app.state.rag_engine = MagicMock()
    main.app.state.mongo_db = MagicMock()
    main.app.state.mongo_db.upsert_progress = MagicMock()
    main.app.state.mongo_db.get_progress = MagicMock(return_value=None)
    main.app.state.vector_db = MagicMock()
    main.app.state.log_parser = MagicMock()
    return main.app, main


# ---------------------------------------------------------------------------
# API_KEY
# ---------------------------------------------------------------------------

class TestAPIKey:

    def test_api_unauth_when_key_set_no_header(self, monkeypatch):
        app, _ = _fresh_app(monkeypatch, {"API_KEY": "s3cret", "RATE_LIMIT_PER_MINUTE": "1000"})
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/api/incident/resolve", json={})
            assert response.status_code == 401

    def test_api_get_progress_blocked_without_key(self, monkeypatch):
        """GETs are now also gated — fixes the §1.4 leak."""
        app, _ = _fresh_app(monkeypatch, {"API_KEY": "s3cret", "RATE_LIMIT_PER_MINUTE": "1000"})
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/api/analysis/some-id/progress")
            assert response.status_code == 401

    def test_api_get_context_blocked_without_key(self, monkeypatch):
        app, _ = _fresh_app(monkeypatch, {"API_KEY": "s3cret", "RATE_LIMIT_PER_MINUTE": "1000"})
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/api/analysis/some-id/context")
            assert response.status_code == 401

    def test_api_health_open_even_with_key(self, monkeypatch):
        app, _ = _fresh_app(monkeypatch, {"API_KEY": "s3cret", "RATE_LIMIT_PER_MINUTE": "1000"})
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/api/health")
            # Either 200 (mock app state has dbs) or 503 if not — but must NOT be 401
            assert response.status_code != 401

    def test_api_authorized_passes(self, monkeypatch):
        app, _ = _fresh_app(monkeypatch, {"API_KEY": "s3cret", "RATE_LIMIT_PER_MINUTE": "1000"})
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/incident/resolve",
                json={},  # invalid body
                headers={"X-API-Key": "s3cret"},
            )
            # Auth passed; body validation kicks in
            assert response.status_code in (400, 422)

    def test_api_open_when_key_unset(self, monkeypatch):
        app, _ = _fresh_app(monkeypatch, {"API_KEY": "", "RATE_LIMIT_PER_MINUTE": "1000"})
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/api/incident/resolve", json={})
            # No auth required → reaches body validation
            assert response.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------

class TestRateLimit:

    def test_rate_limit_returns_429(self, monkeypatch):
        app, main_mod = _fresh_app(monkeypatch, {"API_KEY": "", "RATE_LIMIT_PER_MINUTE": "3"})
        with TestClient(app, raise_server_exceptions=False) as client:
            for _ in range(3):
                assert client.get("/api/health").status_code != 429
            assert client.get("/api/health").status_code == 429

    def test_rate_limit_header_includes_request_id(self, monkeypatch):
        app, _ = _fresh_app(monkeypatch, {"API_KEY": "", "RATE_LIMIT_PER_MINUTE": "1"})
        with TestClient(app, raise_server_exceptions=False) as client:
            client.get("/api/health")
            response = client.get("/api/health")
            assert response.status_code == 429
            assert "X-Request-ID" in response.headers


# ---------------------------------------------------------------------------
# Request ID
# ---------------------------------------------------------------------------

class TestRequestId:

    def test_supplied_request_id_echoed_back(self, monkeypatch):
        app, _ = _fresh_app(monkeypatch, {"API_KEY": "", "RATE_LIMIT_PER_MINUTE": "1000"})
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/api/health", headers={"X-Request-ID": "rid-abc"})
            assert response.headers.get("X-Request-ID") == "rid-abc"

    def test_generated_when_missing(self, monkeypatch):
        app, _ = _fresh_app(monkeypatch, {"API_KEY": "", "RATE_LIMIT_PER_MINUTE": "1000"})
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/api/health")
            assert response.headers.get("X-Request-ID")

    def test_request_id_appears_in_log_records(self, monkeypatch, caplog):
        app, main_mod = _fresh_app(monkeypatch, {"API_KEY": "", "RATE_LIMIT_PER_MINUTE": "1000", "LOG_JSON": "0"})

        # Add a handler that reads the per-record request_id field via the filter
        captured: list[str] = []

        class _Capture(logging.Handler):
            def emit(self, record):
                captured.append(getattr(record, "request_id", "MISSING"))

        cap = _Capture()
        cap.addFilter(main_mod._RequestIdFilter())

        with TestClient(app, raise_server_exceptions=False) as client:
            # Attach the capture handler only AFTER startup: lifespan-time
            # records (e.g. MongoDB retry errors when no local mongo exists)
            # legitimately carry request_id "-" and must not be counted.
            logging.getLogger("app").addHandler(cap)
            try:
                # /incident/resolve logs from app.routes inside the request
                # context regardless of whether downstream services are up
                # (success and failure paths both log after the [▶] line).
                response = client.post(
                    "/api/incident/resolve",
                    headers={"X-Request-ID": "rid-zzz"},
                    json={
                        "context": {
                            "dag_id": "d1",
                            "root_cause": "rc",
                            "causal_chain": ["e1"],
                        },
                        "root_cause_expln": "expl",
                    },
                )
            finally:
                logging.getLogger("app").removeHandler(cap)

        if response.status_code == 503 and not captured:
            pytest.skip("RAG engine unavailable at startup; no request-path logging to observe")
        # The middleware must have stamped the supplied request id onto every
        # record emitted within the request context.
        assert any(rid == "rid-zzz" for rid in captured), f"captured={captured}"
