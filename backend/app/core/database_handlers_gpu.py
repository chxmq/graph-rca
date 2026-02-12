"""
Database handlers optimized for GPU servers without Docker.
Uses in-process ChromaDB and connects to local Ollama.
"""

import chromadb

import ollama
from pymongo import MongoClient
from typing import Optional
import os

class OllamaEmbeddingFunction:
    """Embedding function using local Ollama instance"""
    
    def __init__(self, model: str = "llama3.2:3b", host: str = "http://localhost:11434"):
        self.model = model
        self.client = ollama.Client(host=host)
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts"""
        embeddings = []
        for text in input:
            response = self.client.embeddings(model=self.model, prompt=text)
            embeddings.append(response['embedding'])
        return embeddings


class VectorDatabaseHandler:
    """ChromaDB handler using in-process/persistent mode (no server required)"""
    
    def __init__(self, persist_directory: str = "./data/chromadb"):
        """Initialize ChromaDB in persistent mode"""
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )
        
        print(f"✓ ChromaDB initialized (persistent mode: {persist_directory})")
    
    def get_or_create_collection(self, name: str, embedding_function=None):
        """Get or create a collection"""
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function
        )
    
    def delete_collection(self, name: str):
        """Delete a collection"""
        try:
            self.client.delete_collection(name=name)
        except Exception:
            pass


class MongoDBHandler:
    """MongoDB handler for local MongoDB instance"""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/"):
        """Initialize MongoDB connection"""
        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client["graph_rca"]
            print("✓ MongoDB connected")
        except Exception as e:
            print(f"⚠ MongoDB not available: {e}")
            print("  The app will work without MongoDB (limited features)")
            self.client = None
            self.db = None
    
    def get_collection(self, name: str):
        """Get a collection"""
        if self.db is None:
            return None
        return self.db[name]
    
    def insert_one(self, collection_name: str, document: dict):
        """Insert a document"""
        if self.db is None:
            return None
        collection = self.get_collection(collection_name)
        return collection.insert_one(document)
    
    def find_one(self, collection_name: str, query: dict):
        """Find one document"""
        if self.db is None:
            return None
        collection = self.get_collection(collection_name)
        return collection.find_one(query)
    
    def find_many(self, collection_name: str, query: dict, limit: int = 10):
        """Find multiple documents"""
        if self.db is None:
            return []
        collection = self.get_collection(collection_name)
        return list(collection.find(query).limit(limit))
