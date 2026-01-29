#!/bin/bash
#===============================================================================
# Standalone Real-Incident Evaluation Script (GPU Server Optimized)
#===============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INCIDENT_DIR="$PROJECT_ROOT/data/real_incidents"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     GraphRCA Real-World Incident Evaluation (GPU)            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

# 1. Setup Environment
echo -e "\n${YELLOW}[1/4] Checking dependencies...${NC}"
pip install -q ollama pandas requests 2>/dev/null || true
echo -e "${GREEN}✓${NC} Dependencies ready."

# 2. Ensure Ollama is running
echo -e "\n${YELLOW}[2/4] Checking Ollama status...${NC}"
if ! pgrep -x "ollama" > /dev/null 2>&1; then
    echo "Starting Ollama service..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 10
fi

# Pull the model if missing
ollama list 2>/dev/null | grep -q "llama3.2:3b" || ollama pull llama3.2:3b
echo -e "${GREEN}✓${NC} Ollama is ready (Model: llama3.2:3b)"

# 3. Synchronize/Generate Logs for Curated Incidents
echo -e "\n${YELLOW}[3/4] Synthesizing evidence (logs.txt) for incidents...${NC}"
cd "$INCIDENT_DIR"
python3 generate_synthetic_logs.py
echo -e "${GREEN}✓${NC} Log synthesis complete."

# 4. Run RAG Accuracy Benchmark
echo -e "\n${YELLOW}[4/4] Running RAG Accuracy Benchmark...${NC}"
cd "$SCRIPT_DIR"
python3 evaluate_rag_accuracy.py

# Check for results
if [ -f "./eval_final_results/real_incident_bench.json" ]; then
    echo -e "\n${GREEN}📊 Results generated in ./eval_final_results/real_incident_bench.json${NC}"
    
    # Generate quick summary report for terminal
    python3 -c "
import json
from pathlib import Path
data = json.load(open('./eval_final_results/real_incident_bench.json'))
s = data['summary']
print('\nFinal Benchmarking Score:')
print(f'  Baseline Accuracy: {s[\"baseline_avg_accuracy\"]*100:.1f}%')
print(f'  RAG Accuracy:      {s[\"rag_avg_accuracy\"]*100:.1f}%')
print(f'  Net Improvement:   {s[\"improvement_pct\"]:.1f}%')
"
else
    echo -e "\n${RED}✗ Benchmark failed to produce results.${NC}"
    exit 1
fi

echo -e "\n${BLUE}Evaluation Complete.${NC}"
