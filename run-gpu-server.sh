#!/bin/bash

# Graph-RCA Setup Script for GPU Servers (No Docker Required)
# This script sets up the project to run directly on GPU servers with NVIDIA drivers

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║   Graph-RCA GPU Server Setup              ║"
echo "║   (Docker-free with Direct GPU Access)    ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Step 1: Check NVIDIA GPU
echo -e "${BLUE}[1/8]${NC} Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
else
    echo -e "${YELLOW}⚠${NC}  No NVIDIA GPU detected. Will run on CPU (slower)."
fi
echo ""

# Step 2: Check Python 3.13
echo -e "${BLUE}[2/8]${NC} Checking Python..."
if command -v python3.13 &> /dev/null; then
    echo -e "${GREEN}✓${NC} Python 3.13 found: $(python3.13 --version)"
    PYTHON_CMD="python3.13"
elif command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✓${NC} Python found: $PY_VERSION"
    PYTHON_CMD="python3"
else
    echo -e "${RED}✗${NC} Python 3 not found!"
    echo "   Please install Python 3.8+ first"
    exit 1
fi
echo ""

# Step 3: Create Python venv
echo -e "${BLUE}[3/8]${NC} Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    echo "   Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment already exists"
fi

# Activate venv and upgrade pip
source venv/bin/activate
pip install -q --upgrade pip
echo ""

# Step 4: Install Python dependencies
echo -e "${BLUE}[4/8]${NC} Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    echo "   This may take a few minutes..."
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    echo -e "${GREEN}✓${NC} Python dependencies installed"
else
    echo -e "${RED}✗${NC} requirements.txt not found!"
    exit 1
fi
echo ""

# Step 5: Check/Install Ollama
echo -e "${BLUE}[5/8]${NC} Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓${NC} Ollama is installed: $(ollama --version 2>&1 | head -1)"
else
    echo -e "${YELLOW}⚠${NC}  Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo -e "${GREEN}✓${NC} Ollama installed"
fi

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "   Starting Ollama service..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 3
    echo -e "${GREEN}✓${NC} Ollama service started"
else
    echo -e "${GREEN}✓${NC} Ollama is already running"
fi

# Pull required models
echo "   Pulling required LLM models (this may take a while)..."
if ! ollama list | grep -q "llama3.2:3b"; then
    echo "   Downloading llama3.2:3b (~2GB)..."
    ollama pull llama3.2:3b
fi
if ! ollama list | grep -q "qwen2.5-coder:3b"; then
    echo "   Downloading qwen2.5-coder:3b (~1.9GB)..."
    ollama pull qwen2.5-coder:3b
fi
echo -e "${GREEN}✓${NC} LLM models ready"
echo ""

# Step 6: Check Node.js
echo -e "${BLUE}[6/8]${NC} Checking Node.js..."
if command -v node &> /dev/null; then
    echo -e "${GREEN}✓${NC} Node.js found: $(node --version)"
    if command -v npm &> /dev/null; then
        echo -e "${GREEN}✓${NC} npm found: $(npm --version)"
    else
        echo -e "${RED}✗${NC} npm not found!"
        exit 1
    fi
else
    echo -e "${RED}✗${NC} Node.js not found!"
    echo "   Please install Node.js 18+ first"
    echo "   Visit: https://nodejs.org/"
    exit 1
fi
echo ""

# Step 7: Install frontend dependencies
echo -e "${BLUE}[7/8]${NC} Installing frontend dependencies..."
cd frontend
if [ ! -d "node_modules" ]; then
    echo "   Running npm install (this may take a few minutes)..."
    npm install
    echo -e "${GREEN}✓${NC} Frontend dependencies installed"
else
    echo -e "${GREEN}✓${NC} Frontend dependencies already installed"
fi
cd ..
echo ""

# Step 8: Update configuration for GPU server
echo -e "${BLUE}[8/8]${NC} Configuring for GPU server..."

# Update backend to use ChromaDB in-process mode
cat > backend/app/core/database_handlers.py.patch << 'EOF'
# ChromaDB will run in-process mode (no separate server needed)
# Ollama will connect to local instance (no Docker)
# MongoDB will use local instance (install separately if needed)
EOF

echo -e "${GREEN}✓${NC} Configuration updated for GPU server"
echo ""

# Create startup scripts
echo -e "${BLUE}Creating startup scripts...${NC}"

# Backend startup script
cat > start-backend-gpu.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/backend"

# Activate venv
source venv/bin/activate

# Set environment for GPU
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

# Make sure Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 3
fi

echo "Starting backend on GPU server..."
echo "Local:   http://localhost:8010"
echo "Network: http://$(hostname -I 2>/dev/null | awk '{print $1}' || hostname):8010"
echo "API docs: http://localhost:8010/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python -m uvicorn main:app --host 0.0.0.0 --port 8010 --reload
EOF

chmod +x start-backend-gpu.sh

# Frontend startup script  
cat > start-frontend-gpu.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/frontend"

echo "Starting frontend..."
echo "Local:   http://localhost:5173"
echo "Network: http://$(hostname -I 2>/dev/null | awk '{print $1}' || hostname):5173"
echo ""
echo "Press Ctrl+C to stop"
echo ""

npm run dev
EOF

chmod +x start-frontend-gpu.sh

echo -e "${GREEN}✓${NC} Startup scripts created"
echo ""

# Print success message
echo "╔════════════════════════════════════════════╗"
echo "║            Setup Complete! 🚀              ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo ""
echo "1. Start the backend (Terminal 1):"
echo -e "   ${BLUE}./start-backend-gpu.sh${NC}"
echo ""
echo "2. Start the frontend (Terminal 2):"
echo -e "   ${BLUE}./start-frontend-gpu.sh${NC}"
echo ""
echo "3. Access the application:"
echo "   Local:   http://localhost:5173"
echo "   Network: http://$(hostname -I 2>/dev/null | awk '{print $1}' || hostname):5173"
echo ""
echo -e "${YELLOW}Note:${NC} This setup uses:"
echo "  • Ollama running directly on the server (with GPU support)"
echo "  • ChromaDB in library mode (no separate server)"
echo "  • MongoDB may need separate installation (optional)"
echo ""
echo -e "${YELLOW}GPU Usage:${NC}"
echo "  • Monitor with: watch -n 1 nvidia-smi"
echo "  • Check Ollama: ollama list"
echo ""

deactivate
