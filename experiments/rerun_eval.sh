#!/bin/bash

# Quick Re-run Script for RVCE
# After pulling latest changes, run this to re-evaluate with improved RAG

set -e

echo "🔄 Re-running RAG Evaluation with Improvements..."
echo ""
echo "Changes:"
echo "  - Using semantic embeddings (nomic-embed-text)"
echo "  - Retrieving top-1 most similar incident (not 3)"
echo ""

# Set API key if not already set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set!"
    echo "Run: export OPENAI_API_KEY='your-key'"
    exit 1
fi

# Pull latest changes
echo "[1/3] Pulling latest code..."
git pull origin main

# Download embedding model if needed
echo "[2/3] Ensuring embedding model is available..."
ollama pull nomic-embed-text > /dev/null 2>&1 || true

# Run evaluation
echo "[3/3] Running improved evaluation..."
cd experiments
python3 evaluate_rag_accuracy.py

echo ""
echo "✅ Evaluation complete! Check results:"
echo "   cat experiments/eval_final_results/real_incident_bench.json"
