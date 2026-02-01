#!/bin/bash
# Quick start script for overnight experiments

set -e

echo "=================================================="
echo "GraphRCA Overnight GPU Experiments"
echo "=================================================="

# Check prerequisites
echo "Checking prerequisites..."

# Ollama
if ! curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "❌ Ollama not running! Start with: ollama serve"
    exit 1
fi
echo "✓ Ollama running"

# ChromaDB
if ! curl -s http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; then
    echo "⚠️  ChromaDB not running on port 8000"
    echo "   Starting ChromaDB in background..."
    chroma run --port 8000 &
    sleep 3
fi
echo "✓ ChromaDB running"

# OpenAI key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY not set - will use keyword scoring fallback"
else
    echo "✓ OpenAI API key set"
fi

echo ""
echo "Starting experiments..."
echo "This will take 4-6 hours. Safe to leave overnight."
echo "Progress saved every 5 incidents - can resume if interrupted."
echo ""

cd "$(dirname "$0")"
python run_overnight.py

echo ""
echo "=================================================="
echo "Done! Results in: experiments/overnight_gpu/results/"
echo "=================================================="
