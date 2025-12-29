#!/bin/bash

# Script to stop graph-rca services (using sudo)

echo "Stopping graph-rca services..."

sudo docker stop graph-rca-chroma graph-rca-mongodb graph-rca-ollama 2>/dev/null
sudo docker rm graph-rca-chroma graph-rca-mongodb graph-rca-ollama 2>/dev/null

echo "Services stopped and removed."
echo "Network 'rca-network' is still available for future use."
echo "Data is preserved in ./data directory."
