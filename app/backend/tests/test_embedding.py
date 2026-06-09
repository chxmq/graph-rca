import pytest
from unittest.mock import patch, Mock, MagicMock
from app.embedding import EmbeddingCreator


@pytest.fixture
def mock_ef():
    """Patch OllamaEmbeddingFunction so tests don't need a live Ollama."""
    with patch("app.embedding.OllamaEmbeddingFunction") as mock_cls:
        mock_instance = MagicMock()
        # Return realistic float lists (not numpy arrays)
        mock_instance.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def embedder(mock_ef):
    return EmbeddingCreator()


class TestEmbeddingCreator:
    def test_create_embedding_returns_list(self, embedder):
        embedding = embedder.create_embedding("test text")
        assert isinstance(embedding, list)
        assert len(embedding) == 768

    def test_create_batch_embeddings_valid(self, embedder):
        results = embedder.create_batch_embeddings(["first", "second"])
        assert len(results) == 2
        assert all(len(e) == 768 for e in results)

    def test_create_batch_embeddings_empty_raises(self, embedder):
        with pytest.raises(ValueError, match="empty"):
            embedder.create_batch_embeddings([])

    def test_create_batch_embeddings_blank_string_raises(self, embedder):
        with pytest.raises(ValueError, match="empty or non-string"):
            embedder.create_batch_embeddings([""])

    def test_create_batch_embeddings_non_string_raises(self, embedder):
        with pytest.raises(ValueError, match="empty or non-string"):
            embedder.create_batch_embeddings([42])

    def test_get_similarity_identical(self, embedder):
        vec = [0.5] * 768
        assert embedder.get_similarity(vec, vec) == pytest.approx(1.0, abs=1e-6)

    def test_get_similarity_different(self, embedder):
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        assert embedder.get_similarity(vec_a, vec_b) == pytest.approx(0.0, abs=1e-6)

    def test_get_similarity_zero_vectors(self, embedder):
        """Zero vectors should return 0.0, not raise ZeroDivisionError."""
        assert embedder.get_similarity([0, 0, 0], [0, 0, 0]) == 0.0

    def test_embedding_backend_failure(self, mock_ef):
        """Errors from the embedding backend should propagate."""
        mock_ef.side_effect = Exception("API failure")
        creator = EmbeddingCreator()
        with pytest.raises(Exception, match="API failure"):
            creator.create_embedding("test")
