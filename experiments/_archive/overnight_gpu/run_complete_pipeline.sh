#!/bin/bash
# =============================================================================
# GraphRCA Complete Pipeline
# =============================================================================
# Run this script to:
#   1. Expand dataset to 120+ incidents (optional, ~1 hour)
#   2. Run all overnight experiments (4-6 hours)
#
# Usage:
#   ./run_complete_pipeline.sh
#   ./run_complete_pipeline.sh --skip-expansion  # If already expanded
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "GraphRCA Complete Pipeline"
echo "=================================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Parse args
SKIP_EXPANSION=false
for arg in "$@"; do
    case $arg in
        --skip-expansion)
            SKIP_EXPANSION=true
            ;;
    esac
done

# Check prerequisites
echo "Checking prerequisites..."

# Ollama
if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "❌ Ollama not running!"
    echo "   Start with: ollama serve"
    exit 1
fi
echo "✓ Ollama running"

# OpenAI key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set!"
    echo "   export OPENAI_API_KEY='your-key'"
    exit 1
fi
echo "✓ OpenAI API key set"

# Current incident count
INCIDENT_COUNT=$(ls -d data/real_incidents/incident_* 2>/dev/null | wc -l | tr -d ' ')
echo "✓ Current incidents: $INCIDENT_COUNT"

# Step 1: Dataset Expansion
if [ "$SKIP_EXPANSION" = false ] && [ "$INCIDENT_COUNT" -lt 100 ]; then
    echo ""
    echo "=================================================="
    echo "STEP 1: Expanding Dataset"
    echo "=================================================="
    echo "This will add ~60 more incidents (takes ~1 hour)"
    echo ""
    
    cd data/real_incidents/expansion
    python expand_dataset.py
    
    NEW_COUNT=$(ls -d "$PROJECT_ROOT"/data/real_incidents/incident_* 2>/dev/null | wc -l | tr -d ' ')
    echo ""
    echo "✓ Dataset expanded: $INCIDENT_COUNT → $NEW_COUNT incidents"
    cd "$PROJECT_ROOT"
else
    if [ "$SKIP_EXPANSION" = true ]; then
        echo "⏭️  Skipping expansion (--skip-expansion flag)"
    else
        echo "⏭️  Skipping expansion (already have $INCIDENT_COUNT incidents)"
    fi
fi

# Step 2: ChromaDB
echo ""
echo "Checking ChromaDB..."
if ! curl -s http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; then
    echo "Starting ChromaDB in background..."
    chroma run --port 8000 &
    sleep 5
fi
echo "✓ ChromaDB running"

# Step 3: Overnight Experiments
echo ""
echo "=================================================="
echo "STEP 2: Running Overnight Experiments"
echo "=================================================="
echo "This will take 4-6 hours. Safe to leave overnight."
echo "Progress is checkpointed - can resume if interrupted."
echo ""

cd experiments/overnight_gpu
python run_overnight.py

# Done
echo ""
echo "=================================================="
echo "COMPLETE!"
echo "=================================================="
echo "Results in: experiments/overnight_gpu/results/"
echo ""
echo "Next steps:"
echo "  1. Review results/final_summary.json"
echo "  2. Update paper with new metrics"
