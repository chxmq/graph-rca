#!/bin/bash

# Graph-RCA Master Startup Script
# This script handles everything: Docker services, backend, and frontend

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║     Graph-RCA Startup Script              ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker is running
echo -e "${BLUE}[1/6]${NC} Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running!${NC}"
    echo "   Please start Docker Desktop and try again."
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker is running"

# Clean up any existing containers first
echo ""
echo -e "${BLUE}[2/6]${NC} Cleaning up old containers..."
docker-compose down > /dev/null 2>&1 || true

# Start Docker services
echo -e "${BLUE}[3/6]${NC} Starting Docker services (MongoDB, ChromaDB, Ollama)..."
docker-compose up -d
echo -e "${GREEN}✓${NC} Docker services started"

# Check/Install Ollama model
echo ""
echo -e "${BLUE}[4/6]${NC} Checking Ollama model..."
if ! docker exec $(docker ps -qf "name=ollama") ollama list | grep -q "llama3.2:3b"; then
    echo -e "${YELLOW}⚠${NC}  Model llama3.2:3b not found. Installing..."
    docker exec -it $(docker ps -qf "name=ollama") ollama pull llama3.2:3b
    echo -e "${GREEN}✓${NC} Model installed"
else
    echo -e "${GREEN}✓${NC} Model llama3.2:3b already installed"
fi

# Setup Python backend
echo ""
echo -e "${BLUE}[5/6]${NC} Setting up Python backend..."
cd backend

# Check if python3.13 is available
if ! command -v python3.13 &> /dev/null; then
    echo -e "${RED}❌ Python 3.13 not found!${NC}"
    echo "   Please install Python 3.13 first."
    echo "   Visit: https://www.python.org/downloads/"
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "   Creating Python 3.13 virtual environment..."
    python3.13 -m venv venv
fi

# Source the venv for dependency installation
source venv/bin/activate

# Always ensure pip is up to date
pip install -q --upgrade pip

# Check if requirements need to be installed/updated
echo "   Checking Python dependencies..."
pip install -q -r requirements.txt
echo -e "${GREEN}✓${NC} Python dependencies installed/verified"

# Deactivate venv
deactivate

cd ..

# Setup Node.js frontend
echo ""
echo -e "${BLUE}[6/6]${NC} Setting up Node.js frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "   Installing Node dependencies..."
    npm install --silent
    echo -e "${GREEN}✓${NC} Node dependencies installed"
else
    echo -e "${GREEN}✓${NC} Node dependencies already installed"
fi

cd ..

# Wait for services to be ready
echo ""
echo "Waiting for services to be ready..."
sleep 3
echo -e "${GREEN}✓${NC} All services ready"

# Print summary
echo ""
echo "╔════════════════════════════════════════════╗"
echo "║          Setup Complete! 🚀                ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Python Environment:${NC}"
echo "  • Version:  Python 3.13"
echo "  • Location: backend/venv (auto-activated by start script)"
echo ""
echo -e "${GREEN}Docker Services:${NC}"
echo "  • MongoDB:  localhost:27017"
echo "  • ChromaDB: localhost:8000"
echo "  • Ollama:   localhost:11435"
echo ""
echo -e "${YELLOW}To start the application:${NC}"
echo ""
echo "  ${BLUE}Terminal 1${NC} - Backend API:"
echo "    cd backend"
echo "    source venv/bin/activate"
echo "    python -m uvicorn main:app --host 0.0.0.0 --port 8010 --reload"
echo ""
echo "  ${BLUE}Terminal 2${NC} - Frontend UI:"
echo "    cd frontend"
echo "    npm run dev"
echo ""
echo -e "${GREEN}URLs:${NC}"
echo "  • Frontend:  http://localhost:5173"
echo "  • Backend:   http://localhost:8010"
echo "  • API Docs:  http://localhost:8010/docs"
echo ""
echo -e "${YELLOW}Quick start (in separate terminals):${NC}"
echo "  ./start-backend.sh"
echo "  ./start-frontend.sh"
echo ""

# Create convenience scripts
cat > start-backend.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/backend"
source venv/bin/activate
echo "Starting backend on http://localhost:8010"
echo "API docs: http://localhost:8010/docs"
python -m uvicorn main:app --host 0.0.0.0 --port 8010 --reload
EOF

cat > start-frontend.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/frontend"
echo "Starting frontend on http://localhost:5173"
npm run dev
EOF

chmod +x start-backend.sh start-frontend.sh

echo -e "${GREEN}✓${NC} Created start-backend.sh and start-frontend.sh"
echo ""
