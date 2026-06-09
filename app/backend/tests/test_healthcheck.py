from unittest.mock import MagicMock, patch
import pytest
from app.healthcheck import ServerHealthCheck
from app.database import VectorDatabaseHandler, MongoDBHandler


@pytest.fixture
def real_vector_db():
    with patch("app.database.chromadb.PersistentClient") as MockClient:
        instance = MagicMock()
        instance.heartbeat.return_value = 1
        instance.get_or_create_collection.return_value = MagicMock()
        MockClient.return_value = instance
        yield VectorDatabaseHandler()


@pytest.fixture
def real_mongo_db():
    with patch("pymongo.MongoClient") as MockClient:
        instance = MagicMock()
        instance.server_info.return_value = {"ok": 1.0}
        MockClient.return_value = instance
        handler = MongoDBHandler()
        # Replace the underlying pymongo client so health-check calls succeed
        handler.client = instance
        yield handler


def test_healthcheck_both_healthy(real_vector_db, real_mongo_db):
    status = ServerHealthCheck(vector_db=real_vector_db, mongo_db=real_mongo_db).get_status()
    assert status["chromadb"] is True
    assert status["mongodb"] is True


def test_healthcheck_mongo_unhealthy(real_vector_db, real_mongo_db):
    real_mongo_db.client.server_info.side_effect = Exception("connection refused")
    status = ServerHealthCheck(vector_db=real_vector_db, mongo_db=real_mongo_db).get_status()
    assert status["chromadb"] is True
    assert status["mongodb"] is False


def test_healthcheck_chroma_unhealthy(real_vector_db, real_mongo_db):
    real_vector_db.client.heartbeat.side_effect = Exception("chroma down")
    status = ServerHealthCheck(vector_db=real_vector_db, mongo_db=real_mongo_db).get_status()
    assert status["chromadb"] is False
    assert status["mongodb"] is True


def test_healthcheck_none_services():
    status = ServerHealthCheck(vector_db=None, mongo_db=None).get_status()
    assert status["chromadb"] is False
    assert status["mongodb"] is False
