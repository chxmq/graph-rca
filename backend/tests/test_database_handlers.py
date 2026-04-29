import pytest
from unittest.mock import patch, MagicMock
from app.database import VectorDatabaseHandler, MongoDBHandler, Document
from pymongo.results import InsertOneResult
from chromadb.api.models.Collection import Collection
from datetime import datetime, timezone

@pytest.fixture
def vector_db():
    with patch('chromadb.PersistentClient') as mock_client_cls:
        handler = VectorDatabaseHandler()
        client_instance = mock_client_cls.return_value
        handler.client = client_instance
        mock_collection = MagicMock(spec=Collection)
        client_instance.get_or_create_collection.return_value = mock_collection
        yield handler

@pytest.fixture
def mongo_db():
    with patch('pymongo.MongoClient') as mock_client:
        handler = MongoDBHandler()
        handler.db = MagicMock()
        yield handler

class TestVectorDatabaseHandler:
    def test_initialization(self, vector_db):
        assert vector_db.client is not None
        assert vector_db.ef is not None

    def test_get_query_collection(self, vector_db):
        collection = vector_db._get_query_collection("test")
        vector_db.client.get_or_create_collection.assert_called_with(
            name="test",
            embedding_function=vector_db.ef
        )
        assert collection == vector_db.client.get_or_create_collection.return_value

    def test_add_documents(self, vector_db):
        test_docs = ["doc1", "doc2"]
        test_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        
        vector_db.add_documents(test_docs, test_embeddings)
        
        import hashlib
        expected_ids = [hashlib.sha256(doc.encode()).hexdigest()[:32] for doc in test_docs]
        collection = vector_db._get_write_collection()
        collection.upsert.assert_called_once_with(
            documents=test_docs,
            ids=expected_ids,
            embeddings=test_embeddings,
            metadatas=None,
        )

    def test_query_collection(self, vector_db):
        test_query = ["test query"]
        vector_db.query_collection(test_query, n_results=5)
        
        collection = vector_db._get_query_collection()
        collection.query.assert_called_once_with(
            query_texts=test_query,
            n_results=5
        )

    def test_search_success(self, vector_db):
        mock_results = {
            "documents": [["result1", "result2"]],
            "metadatas": [[{"source": "test1"}, {"source": "test2"}]]
        }
        vector_db._get_query_collection().query.return_value = mock_results
        
        results = vector_db.search("test", "context", top_k=2)
        
        assert len(results) == 2
        assert isinstance(results[0], Document)
        assert results[0].text == "result1"
        assert results[0].metadata["source"] == "test1"

    def test_search_no_results(self, vector_db):
        mock_results = {
            "documents": [[]],  # Empty list of documents in first result
            "metadatas": [[]]
        }
        vector_db._get_query_collection().query.return_value = mock_results
        
        results = vector_db.search("test", "context")
        assert results == []

    def test_search_exception_handling(self, vector_db):
        vector_db._get_query_collection().query.side_effect = Exception("Test error")
        with pytest.raises(Exception):
            vector_db.search("test", "context")

class TestMongoDBHandler:
    def test_initialization_validates_connectivity(self):
        with patch('app.database.MongoClient') as mock_client:
            MongoDBHandler()
            mock_client.return_value.server_info.assert_called_once()

    def test_save_dag(self, mongo_db):
        mock_result = MagicMock(spec=InsertOneResult)
        mock_result.inserted_id = "123"
        mongo_db.db["dags"].insert_one.return_value = mock_result
        
        test_data = {"dag_id": "test_dag", "config": {}}
        result = mongo_db.save_dag(test_data)
        
        mongo_db.db["dags"].insert_one.assert_called_once_with(test_data)
        assert result.inserted_id == "123"

    def test_get_context_with_dag_id(self, mongo_db):
        test_dag_id = "test_dag_123"
        mongo_db.get_context(test_dag_id)
        
        mongo_db.db["contexts"].find_one.assert_called_once_with(
            {"dag_id": test_dag_id}
        )

    def test_get_context_latest(self, mongo_db):
        mongo_db.get_context()
        mongo_db.db["contexts"].find_one.assert_called_once_with(
            sort=[('created_at', -1)]
        )

    def test_get_context_by_analysis_id(self, mongo_db):
        mongo_db.get_context_by_analysis_id("analysis-123")
        mongo_db.db["contexts"].find_one.assert_called_once_with(
            {"analysis_id": "analysis-123"}
        )

    def test_upsert_progress(self, mongo_db):
        mongo_db.upsert_progress("analysis-xyz", {"status": "processing", "progress": 50})
        mongo_db.db["analysis_progress"].update_one.assert_called_once()
        call_args = mongo_db.db["analysis_progress"].update_one.call_args
        assert call_args[0][0] == {"analysis_id": "analysis-xyz"}
        assert call_args[1]["upsert"] is True

    def test_get_progress_found(self, mongo_db):
        mongo_db.db["analysis_progress"].find_one.return_value = {
            "_id": "x",
            "analysis_id": "analysis-xyz",
            "updated_at": "t",
            "status": "completed",
            "progress": 100,
        }
        result = mongo_db.get_progress("analysis-xyz")
        assert result == {"status": "completed", "progress": 100}

    def test_get_progress_not_found(self, mongo_db):
        mongo_db.db["analysis_progress"].find_one.return_value = None
        assert mongo_db.get_progress("missing") is None

    def test_save_context(self, mongo_db):
        mock_result = MagicMock(spec=InsertOneResult)
        mock_result.inserted_id = "context_123"
        mongo_db.db["contexts"].insert_one.return_value = mock_result
        
        test_context = {"dag_id": "test_dag", "created_at": datetime.now(timezone.utc)}
        result = mongo_db.save_context(test_context)
        
        mongo_db.db["contexts"].insert_one.assert_called_once_with(test_context)
        assert result.inserted_id == "context_123"
