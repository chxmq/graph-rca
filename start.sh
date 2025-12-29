#!/bin/bash

# Script to start graph-rca services without docker-compose (using sudo)

echo "Starting graph-rca services with sudo..."

# Create network if it doesn't exist
sudo docker network create rca-network 2>/dev/null || echo "Network rca-network already exists"

# Create data directories
mkdir -p ./data/chroma ./data/db ./data/ollama

# Check for GPU support (macOS doesn't support --gpus flag)
GPU_FLAG=""
if [[ "$OSTYPE" != "darwin"* ]]; then
    GPU_FLAG="--gpus all"
    echo "Linux detected - Attempting to use GPU for Ollama (will fall back to CPU if unavailable)"
else
    echo "macOS detected - Ollama will run on CPU (GPU not supported in Docker Desktop)"
fi

# Function to remove existing container if it exists
remove_container() {
    local container_name=$1
    if sudo docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
        echo "Removing existing container: ${container_name}"
        sudo docker rm -f "${container_name}" 2>/dev/null || true
    fi
}

echo "Starting ChromaDB..."
remove_container graph-rca-chroma
sudo docker run -d \
  --name graph-rca-chroma \
  --network rca-network \
  -p 8000:8000 \
  -v "$(pwd)/data/chroma:/chroma/data" \
  -e ALLOW_RESET=TRUE \
  -e ANONYMIZED_TELEMETRY=FALSE \
  -e IS_PERSISTENT=TRUE \
  --restart unless-stopped \
  chromadb/chroma:latest

echo "Starting MongoDB..."
remove_container graph-rca-mongodb
sudo docker run -d \
  --name graph-rca-mongodb \
  --network rca-network \
  -p 27017:27017 \
  -v "$(pwd)/data/db:/data/db" \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  --restart unless-stopped \
  mongo:latest

echo "Starting Ollama..."
remove_container graph-rca-ollama
if [ -n "$GPU_FLAG" ]; then
    sudo docker run -d \
      --name graph-rca-ollama \
      --network rca-network \
      $GPU_FLAG \
      -p 11435:11434 \
      -v "$(pwd)/data/ollama:/root/.ollama" \
      --restart unless-stopped \
      ollama/ollama
else
    sudo docker run -d \
      --name graph-rca-ollama \
      --network rca-network \
      -p 11435:11434 \
      -v "$(pwd)/data/ollama:/root/.ollama" \
      --restart unless-stopped \
      ollama/ollama
fi

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
