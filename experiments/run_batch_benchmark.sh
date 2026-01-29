#!/bin/bash
#===============================================================================
# Run Batch Inference Benchmark for GraphRCA Paper
#===============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     GraphRCA Batch Inference Benchmark                       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

# 1. Check dependencies
echo -e "\n${YELLOW}[1/3] Checking dependencies...${NC}"
pip install -q ollama pandas 2>/dev/null || pip install ollama pandas

# 2. Check Ollama connection
echo -e "\n${YELLOW}[2/3] Checking Ollama connection...${NC}"

# Try Docker container first
if curl -s http://localhost:11435/api/version > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Ollama running on port 11435 (Docker container)"
elif curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Ollama running on port 11434 (local)"
    # Update script to use local port
    sed -i.bak 's/11435/11434/g' "$SCRIPT_DIR/batch_inference_benchmark.py"
else
    echo -e "${RED}✗${NC} Ollama not running!"
    echo ""
    echo "Start Ollama using one of:"
    echo "  Docker:  cd $PROJECT_ROOT && docker-compose up -d ollama"
    echo "  Local:   ollama serve"
    echo ""
    echo "Then make sure the model is available:"
    echo "  ollama pull llama3.2:3b"
    exit 1
fi

# 3. Run benchmark
echo -e "\n${YELLOW}[3/3] Running batch inference benchmark...${NC}"
echo ""

cd "$SCRIPT_DIR"
python3 batch_inference_benchmark.py

# Results location
echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  NEXT STEPS                                                  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "1. Copy the LaTeX table from above into your paper"
echo "2. Location: IEEE Paper/access.tex, Section IV-A"
echo "3. Results saved to: $SCRIPT_DIR/batch_benchmark_results.json"
echo ""
