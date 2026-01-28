import chromadb
import pymongo


class ServerHealthCheck:
    """
    Health check utility for database services.
    Can be used by FastAPI health endpoints or CLI tools.
    """
    def __init__(self) -> None:
        self.chroma_client = chromadb.HttpClient(host='localhost', port=8000)
        self.mongo_client = pymongo.MongoClient('mongodb://admin:password@localhost:27017/')
        
    def check_chroma(self) -> bool:
        """Check if ChromaDB is accessible."""
        try:
            server_info = self.chroma_client.heartbeat()
            return server_info != 0
        except Exception as e:
            return False
    
    def check_mongo(self) -> bool:
        """Check if MongoDB is accessible."""
        try:
            server_info = self.mongo_client.server_info()
            return server_info.get('ok', 0) == 1.0
        except Exception as e:
            return False
    
    def get_status(self) -> dict:
        """Get status of all services."""
        return {
            "chromadb": self.check_chroma(),
            "mongodb": self.check_mongo()
        }