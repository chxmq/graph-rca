import logging
import numpy as np
from app.config import OLLAMA_HOST, EMBEDDING_MODEL
from app.database import OllamaEmbeddingFunction

logger = logging.getLogger(__name__)

class EmbeddingCreator:
    def __init__(self):
        self.ef = OllamaEmbeddingFunction(
            url=f"{OLLAMA_HOST}/api/embeddings",
            model_name=EMBEDDING_MODEL
        )
    
    def create_embedding(self, text: str) -> list[float]:
        """Create embedding for a single text"""
        return self.ef([text])[0]
    
    def create_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for multiple texts"""
        # Add input validation
        if not texts:
            raise ValueError("Cannot embed empty texts list")
        
        if any(not isinstance(t, str) or len(t.strip()) == 0 for t in texts):
            raise ValueError("Text list contains empty or non-string values")
        
        try:
            return self.ef(texts)
        except Exception as e:
            # Add error logging
            logger.error("Embedding generation failed: %s", e)
            raise
    
    def get_similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        a = np.array(embedding1, dtype=float)
        b = np.array(embedding2, dtype=float)
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
