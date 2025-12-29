#!/bin/bash

# Script to start graph-rca services without docker-compose (using sudo)

echo "Starting graph-rca services with sudo..."

# Create network if it doesn't exist
sudo docker network create rca-network 2>/dev/null || echo "Network rca-network already exists"

# Create data directories
mkdir -p ./data/chroma ./data/db ./data/ollama

echo "Starting ChromaDB..."
sudo docker run -d \
  --name graph-rca-chroma \
  --network rca-network \
  -p 8000:8000 \
  -v $(pwd)/data/chroma:/chroma/data \
  -e ALLOW_RESET=TRUE \
  -e ANONYMIZED_TELEMETRY=FALSE \
  -e IS_PERSISTENT=TRUE \
  --restart unless-stopped \
  chromadb/chroma:latest

echo "Starting MongoDB..."
sudo docker run -d \
  --name graph-rca-mongodb \
  --network rca-network \
  -p 27017:27017 \
  -v $(pwd)/data/db:/data/db \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  --restart unless-stopped \
  mongo:latest

echo "Starting Ollama with GPU support..."
sudo docker run -d \
  --name graph-rca-ollama \
  --network rca-network \
  --gpus all \
  -p 11435:11434 \
  -v $(pwd)/data/ollama:/root/.ollama \
  --restart unless-stopped \
  ollama/ollama

echo ""
echo "All services started successfully!"
echo ""
echo "To check status: sudo docker ps"
echo "To view logs: sudo docker logs <container-name>"
echo ""
echo "Next steps:"
echo "1. Install Ollama models:"
echo "   sudo docker exec -it graph-rca-ollama ollama pull llama3.2:3b"
echo "   sudo docker exec -it graph-rca-ollama ollama pull nomic-embed-text"
echo "2. Run the Streamlit app:"
echo "   streamlit run frontend/app.py --server.port 8550"
