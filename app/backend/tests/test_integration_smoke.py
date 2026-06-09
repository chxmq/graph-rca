import os
import pytest


pytestmark = pytest.mark.integration


def _missing(pkg: str) -> bool:
    try:
        __import__(pkg)
        return False
    except Exception:
        return True


@pytest.mark.skipif(_missing("testcontainers") or _missing("ollama"), reason="integration deps not installed")
def test_integration_smoke_services_wiring():
    """Mongo container roundtrip: connect, init indexes, upsert/get progress."""
    from testcontainers.mongodb import MongoDbContainer

    with MongoDbContainer("mongo:8.0.5") as mongo:
        os.environ["MONGO_URI"] = mongo.get_connection_url()
        from app.database import MongoDBHandler
        handler = MongoDBHandler()
        assert handler.client is not None

        # Round-trip: ensure progress upsert/get works against real Mongo.
        handler.upsert_progress("smoke-id", {"status": "completed", "progress": 100})
        result = handler.get_progress("smoke-id")
        assert result == {"status": "completed", "progress": 100}

        # Re-upsert overwrites; get_progress returns the latest
        handler.upsert_progress("smoke-id", {"status": "failed", "progress": 100})
        assert handler.get_progress("smoke-id") == {"status": "failed", "progress": 100}
