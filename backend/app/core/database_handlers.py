import chromadb
import pymongo
from pymongo import MongoClient
from chromadb.utils.embedding_functions import EmbeddingFunction
from typing import List
from dataclasses import dataclass
import requests
import os

@dataclass
class Document:
    text: str
    metadata: dict

class OllamaEmbeddingFunction(EmbeddingFunction):
    """Custom Ollama embedding function for ChromaDB"""
    def __init__(self, model_name: str, url: str):
        self.model_name = model_name
        self.url = url
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        embeddings = []
        for text in input:
            try:
                response = requests.post(
                    self.url,
                    json={"model": self.model_name, "prompt": text}
                )
                if response.status_code == 200:
                    embeddings.append(response.json()["embedding"])
                else:
                    # Return zero vector on error
                    embeddings.append([0.0] * 768)  # Default embedding size
            except Exception as e:
                print(f"Error generating embedding: {e}")
                embeddings.append([0.0] * 768)
        return embeddings

class VectorDatabaseHandler:
    def __init__(self):
        try:
            # Use environment variable for persist directory (useful for experiments/testing)
            persist_dir = os.environ.get("CHROMADB_PATH", "/chroma/data")
            
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=chromadb.Settings(
                    anonymized_telemetry=False
                )
            )
                
            self.ef = OllamaEmbeddingFunction(
                model_name="nomic-embed-text",
                url="http://localhost:11434/api/embeddings",
            )
            
        except Exception as e:
            raise ConnectionError(
                f"Failed to initialize ChromaDB at persist directory. "
                f"Error: {str(e)}"
            )
    
    def get_collection(self, name: str = "docs"):
        try:
            collection = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.ef
            )
            return collection
        except Exception as e:
            return None
    
    def add_documents(self, documents: List[str], embeddings: List[List[float]], ids: List[str] = None):
        collection = self.get_collection()
        # Generate unique ID for each document chunk if not provided
        if ids is None:
            ids = [str(hash(doc)) for doc in documents] 
        
        collection.add(
            documents=documents,
            ids=ids,
            embeddings=embeddings
        )
    
    def query_collection(self, query_texts: List[str], n_results: int = 3):
        collection = self.get_collection()
        return collection.query(
            query_texts=query_texts,
            n_results=n_results
        )

    def search(self, query: str, context: str, top_k: int = 5) -> list:
        """Search implementation using text query"""
        try:
            collection = self.get_collection()
            
            if not collection:
                return [Document(text="No documentation available", metadata={"source": "system"})]

            query_text = f"{query}\nContext: {context}"
            
            try:
                results = collection.query(
                    query_texts=[query_text],
                    n_results=top_k,
                    include=["documents", "metadatas"]
                )
            except Exception as e:
                return [Document(text=f"Error executing query: {str(e)}", metadata={"source": "error"})]
            
            if not results:
                return [Document(text="No relevant documentation found", metadata={"source": "system"})]
            
            if not isinstance(results, dict):
                return [Document(text="Unexpected query result format", metadata={"source": "error"})]
            
            if "documents" not in results:
                return [Document(text="No documents found in results", metadata={"source": "system"})]
            
            if not results["documents"] or not results["documents"][0]:
                return [Document(text="Empty documents list", metadata={"source": "system"})]

            return [
                Document(
                    text=doc,
                    metadata=meta if meta else {"source": "unknown"}
                ) for doc, meta in zip(
                    results["documents"][0],
                    results.get("metadatas", [[{"source": "unknown"}]])[0]
                )
            ]
        except Exception as e:
            return [Document(text=f"Error searching documentation: {str(e)}", metadata={"source": "error"})]

class MongoDBHandler:
    def __init__(self):
        self.client = MongoClient('mongodb://admin:password@localhost:27017/')
        self.db = self.client["log_analysis"]
    
    def save_dag(self, dag_data: dict):
        return self.db["dags"].insert_one(dag_data)
    
    def get_context(self, dag_id: str = None):
        """Retrieve context by DAG ID or latest if not specified"""
        if dag_id:
            return self.db["contexts"].find_one({"dag_id": dag_id})
        return self.db["contexts"].find_one(sort=[('timestamp', -1)])
    
    def save_context(self, context_data: dict):
        return self.db["contexts"].insert_one(context_data)