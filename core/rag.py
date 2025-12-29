from typing import List
from models.rag_response_data_models import SummaryResponse, SolutionQuery
from .embedding import EmbeddingCreator
from core.database_handlers import VectorDatabaseHandler, MongoDBHandler
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class RAG_Engine:
    def __init__(self):
        try:
            self.embedder = EmbeddingCreator()
            self.vector_db = VectorDatabaseHandler()
            self.mongo_db = MongoDBHandler()
            self.ollama_client = ollama.Client(host="http://localhost:11435")
        except Exception as e:
            raise ConnectionError(
                f"Failed to initialize RAG engine. "
                f"Make sure all services (ChromaDB, MongoDB, Ollama) are running. "
                f"Error: {str(e)}"
            )
    
    def generate_summary(self, context: List[str]) -> SummaryResponse:
        """Generate summary using LLM"""
        context_text = '\n'.join(context)
        
        prompt = f"""Analyze the following log context and provide a structured summary in JSON format.

Log Context:
{context_text}

Provide your response in this exact JSON structure:
{{
  "summary": ["summary point 1", "summary point 2", "summary point 3"],
  "root_cause": "detailed explanation of the root cause",
  "severity": "Critical|High|Medium|Low"
}}

Make sure to identify the actual root cause from the logs and assess the appropriate severity level."""

        response = self.ollama_client.generate(
            model="llama3.2:3b",
            prompt=prompt,
            format="json",
            options={"temperature": 0.2}
        )
        
        try:
            import json
            parsed = json.loads(response['response'])
            return SummaryResponse(
                summary=parsed.get('summary', ['Unable to generate summary']),
                root_cause_expln=parsed.get('root_cause', 'Unable to identify root cause'),
                severity=parsed.get('severity', 'Unknown')
            )
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback if JSON parsing fails
            return SummaryResponse(
                summary=response['response'].split("\n"),
                root_cause_expln="Unable to parse root cause from LLM response",
                severity="Unknown"
            )
    
    def generate_solution(self, context: str, root_cause: str) -> SolutionQuery:
        """Generate solution using RAG with automatic query"""
        automated_query = f"Provide resolution steps for: {root_cause}"
        
        try:
            # Ensure context is a string
            if not isinstance(context, str):
                if isinstance(context, (list, tuple)):
                    context_str = "\n".join(map(str, context))
                else:
                    context_str = str(context)
            else:
                context_str = context
            
            # Search documentation using context embeddings
            try:
                results = self.vector_db.search(
                    query=automated_query,
                    context=context_str,
                    top_k=5
                )
            except Exception as e:
                results = [Document(text="Error searching documentation", metadata={"source": "error"})]
            
            # Format context for prompt - Using doc.text since that's what our Document objects have
            doc_context = "\n".join([doc.text for doc in results])
            
            try:
                # Format response with sources
                llm_response = self.ollama_client.generate(
                    model="llama3.2:3b",
                    prompt=f"""Based on the following information, provide a structured solution:

Root Cause:
{root_cause}

Context:
{context_str}

Available Documentation:
{doc_context}

Please provide a detailed solution in the following format:

Problem Analysis:
- Briefly describe the identified issue
- Key observations from the context

Recommended Steps:
1. First step with explanation
2. Second step with explanation
3. Additional steps as needed

Additional Recommendations:
- Important considerations
- Preventive measures
- Monitoring suggestions

Please be specific and actionable in your recommendations.""",
                    options={"temperature": 0.1}
                )
                
            except Exception as e:
                return SolutionQuery(
                    context=context_str,
                    query=automated_query,
                    response="Error: Unable to generate solution from LLM",
                    sources=[doc.metadata.get("source", "Unknown") for doc in results]
                )
            
            if not llm_response or 'response' not in llm_response:
                return SolutionQuery(
                    context=context_str,
                    query=automated_query,
                    response="Error: No response received from LLM",
                    sources=[doc.metadata.get("source", "Unknown") for doc in results]
                )
            
            return SolutionQuery(
                context=context_str,
                query=automated_query,
                response=llm_response['response'],
                sources=[doc.metadata.get("source", "Unknown") for doc in results]
            )
            
        except Exception as e:
            error_context = str(context) if isinstance(context, str) else "\n".join(map(str, context)) if isinstance(context, (list, tuple)) else str(context)
            return SolutionQuery(
                context=error_context,
                query=automated_query,
                response=f"Error generating solution: {str(e)}",
                sources=[]
            )
    
    def store_documentation(self, documents: List[str]) -> None:
        """Store documentation in ChromaDB"""
        # Add validation for empty documents
        if not documents:
            raise ValueError("Received empty documents list")
        
        # Add chunk size validation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        
        chunks = text_splitter.split_text("\n\n".join(documents))
        
        # Validate chunks before embedding
        if not chunks:
            raise ValueError("No text chunks created after splitting")
        
        embeddings = self.embedder.create_batch_embeddings(chunks)
        
        self.vector_db.add_documents(
            documents=chunks,
            embeddings=embeddings
        )