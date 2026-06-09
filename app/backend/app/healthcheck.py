import logging
from app.database import VectorDatabaseHandler, MongoDBHandler

logger = logging.getLogger(__name__)


class ServerHealthCheck:
    """
    Health check utility for database services.
    Can be used by FastAPI health endpoints or CLI tools.
    """
    def __init__(self, vector_db: VectorDatabaseHandler | None, mongo_db: MongoDBHandler | None) -> None:
        self.vector_db = vector_db
        self.mongo_db = mongo_db
        
    def check_chroma(self) -> bool:
        """Check if ChromaDB (same client as runtime) is accessible."""
        try:
            if self.vector_db is None:
                return False
            # Exercise the same persistent client used by runtime data path.
            self.vector_db.client.heartbeat()
            self.vector_db.get_collection()
            return True
        except Exception:
            logger.exception("ChromaDB health check failed")
            return False
    
    def check_mongo(self) -> bool:
        """Check if MongoDB is accessible."""
        try:
            if self.mongo_db is None:
                return False
            server_info = self.mongo_db.client.server_info()
            return server_info.get('ok', 0) == 1.0
        except Exception:
            logger.exception("MongoDB health check failed")
            return False
    
    def get_status(self) -> dict:
        """Get status of all services."""
        return {
            "chromadb": self.check_chroma(),
            "mongodb": self.check_mongo()
        }