# Log Analysis & Incident Resolution System

This application provides automated log analysis and incident resolution using AI. It processes log files, generates insights, and provides solutions based on your documentation.

## What This Project Does

This system helps you:
- **Analyze log files** automatically to identify issues and patterns
- **Identify root causes** of incidents from log entries
- **Generate solutions** by combining log analysis with your documentation
- **Visualize causal chains** showing how issues propagate through your system
- **Store analysis results** for future reference and learning

The system uses:
- **Ollama** (local LLM) for log parsing and solution generation
- **ChromaDB** for storing and searching documentation
- **MongoDB** for storing analysis results and DAG structures
- **Streamlit** for the web interface

## Prerequisites

Before you begin, ensure you have:

- **Docker** installed and running
  - For GPU support: `nvidia-container-toolkit` installed
  - Verify with: `docker --version`
- **Python 3.13**
  - Verify with: `python3 --version`
- **NVIDIA GPU** (optional but recommended for faster inference)
  - Verify with: `nvidia-smi` (if you have a GPU)

## Quick Start Guide

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/KTS-o7/graph-rca.git
cd graph-rca

# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Start Services

Choose one of the following methods:

#### Option A: Docker Compose (Recommended)

```bash
# Start all services (ChromaDB, MongoDB, Ollama)
docker-compose up -d

# Verify services are running
docker ps
```

You should see three containers:
- `graph-rca-chroma` (port 8000)
- `graph-rca-mongodb` (port 27017)
- `graph-rca-ollama` (port 11435)

#### Option B: Shell Scripts (For servers requiring sudo)

```bash
# Make scripts executable
chmod +x start.sh stop.sh

# Start all services
./start.sh

# Verify services are running
sudo docker ps
```

### Step 3: Install Ollama Models

The application requires two models. Install them now:

```bash
# Using Docker Compose
docker exec -it graph-rca-ollama ollama pull llama3.2:3b
docker exec -it graph-rca-ollama ollama pull nomic-embed-text

# OR using sudo (if you used shell scripts)
sudo docker exec -it graph-rca-ollama ollama pull llama3.2:3b
sudo docker exec -it graph-rca-ollama ollama pull nomic-embed-text
```

**Note:** This step may take several minutes depending on your internet connection. The models are approximately 2GB total.

Verify models are installed:
```bash
docker exec -it graph-rca-ollama ollama list
# or with sudo: sudo docker exec -it graph-rca-ollama ollama list
```

You should see both `llama3.2:3b` and `nomic-embed-text` in the list.

### Step 4: Run the Application

```bash
# Make sure your virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start the Streamlit application
streamlit run frontend/app.py --server.port 8550
```

The application will open automatically in your browser at `http://localhost:8550`

If it doesn't open automatically, navigate to that URL manually.

## Usage Guide

### 1. Upload Log File

1. Click on the **"Upload Log File"** expander
2. Select your log file (`.log` or `.txt` format)
3. The system will automatically:
   - Parse the log entries
   - Build a causal chain (DAG)
   - Generate a summary
   - Identify the root cause
   - Display severity assessment

**Supported log formats:**
- Standard text logs (`.log`, `.txt`)
- Logs with timestamps, log levels, and messages
- Multi-line log entries

### 2. Add Documentation

1. Click on the **"Add Documentation"** expander
2. Upload one or more documentation files (`.txt` or `.md` format)
3. The system will:
   - Split documents into chunks
   - Generate embeddings
   - Store them in ChromaDB for retrieval

**Tips for documentation:**
- Include troubleshooting guides
- Add runbooks and procedures
- Include system architecture docs
- The more relevant documentation you add, the better the solutions

### 3. Get Automated Solutions

1. Once both log and documentation are processed
2. Click on the **"Automatic Incident Resolution"** expander
3. The system will:
   - Search documentation for relevant solutions
   - Generate step-by-step resolution steps
   - Provide root cause analysis
   - Show reference documents used

## Configuration

### Default Settings

The application uses these default settings (hardcoded for simplicity):

- **MongoDB:**
  - Host: `localhost`
  - Port: `27017`
  - Username: `admin`
  - Password: `password`
  - Database: `log_analysis`

- **ChromaDB:**
  - Host: `localhost`
  - Port: `8000`

- **Ollama:**
  - Host: `localhost`
  - Port: `11435` (mapped from container port 11434)

**To change these settings:** Edit the connection strings in:
- `core/database_handlers.py` (MongoDB and ChromaDB)
- `core/rag.py` (Ollama)
- `utilz/log_parser.py` (Ollama)
- `utilz/database_healthcheck.py` (MongoDB and ChromaDB)

### Port Configuration

The services run on these ports:
- **ChromaDB:** 8000
- **MongoDB:** 27017
- **Ollama:** 11435 (host) → 11434 (container)
- **Streamlit:** 8550

Make sure these ports are available on your system.

## GPU Support

### With GPU (Recommended)

If you have an NVIDIA GPU:

1. Install NVIDIA drivers: `nvidia-smi` should work
2. Install `nvidia-container-toolkit`:
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. Verify GPU is accessible:
   ```bash
   docker exec -it graph-rca-ollama nvidia-smi
   ```

### Without GPU

The application works perfectly on CPU, just slower. No additional configuration needed.

## Troubleshooting

### Services Won't Start

**Problem:** Docker containers fail to start

**Solutions:**
- Check if ports are already in use:
  ```bash
  # Check port 8000 (ChromaDB)
  lsof -i :8000
  # Check port 27017 (MongoDB)
  lsof -i :27017
  # Check port 11435 (Ollama)
  lsof -i :11435
  ```
- Stop conflicting services or change ports in `docker-compose.yaml`
- Check Docker is running: `docker ps`
- View container logs: `docker logs graph-rca-ollama` (or chroma/mongodb)

### Models Not Found

**Problem:** `Error: model 'llama3.2:3b' not found`

**Solutions:**
- Verify models are installed: `docker exec -it graph-rca-ollama ollama list`
- Reinstall models if missing:
  ```bash
  docker exec -it graph-rca-ollama ollama pull llama3.2:3b
  docker exec -it graph-rca-ollama ollama pull nomic-embed-text
  ```
- Check Ollama container is running: `docker ps | grep ollama`

### Connection Errors

**Problem:** `Connection refused` or `Failed to connect`

**Solutions:**
- Verify all services are running: `docker ps`
- Check service health:
  ```bash
  # Test ChromaDB
  curl http://localhost:8000/api/v1/heartbeat
  
  # Test MongoDB (requires mongosh or similar)
  # Test Ollama
  curl http://localhost:11435/api/tags
  ```
- Wait a few seconds after starting services (they need time to initialize)
- Check firewall settings if accessing remotely

### GPU Not Working

**Problem:** GPU not being used (slow performance)

**Solutions:**
- Verify `nvidia-container-toolkit` is installed
- Check GPU is accessible: `docker exec -it graph-rca-ollama nvidia-smi`
- Restart Docker after installing nvidia-container-toolkit
- GPU is optional - CPU will work but slower

### Streamlit Won't Start

**Problem:** `streamlit: command not found`

**Solutions:**
- Ensure virtual environment is activated: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python3 --version` (needs 3.10+)

### Log Parsing Errors

**Problem:** Log file upload fails or parsing errors

**Solutions:**
- Ensure log file is text format (`.log` or `.txt`)
- Check log file is not empty
- Verify log entries have timestamps and messages
- Try a smaller log file first to test

### No Solutions Generated

**Problem:** Solution generation returns errors or empty results

**Solutions:**
- Ensure documentation is uploaded first
- Check documentation files contain relevant troubleshooting info
- Verify ChromaDB is running and has documents: Check container logs
- Try uploading more comprehensive documentation

### Port Already in Use

**Problem:** `Error: port 8550 is already in use`

**Solutions:**
- Find process using port: `lsof -i :8550`
- Kill the process or use different port:
  ```bash
  streamlit run frontend/app.py --server.port 8551
  ```
- Then access at `http://localhost:8551`

## Stopping Services

### Using Docker Compose

```bash
docker-compose down
```

### Using Shell Scripts

```bash
./stop.sh
```

**Note:** Data is preserved in the `./data` directory. Containers will be removed but data persists.

## Remote Access

To access the application running on a remote server:

1. **SSH Port Forwarding:**
   ```bash
   ssh -L 8550:localhost:8550 user@remote-server
   ```

2. **Open in browser:**
   Navigate to `http://localhost:8550` on your local machine

3. **For multiple services** (if needed):
   ```bash
   ssh -L 8550:localhost:8550 -L 8000:localhost:8000 -L 11435:localhost:11435 user@remote-server
   ```

## Project Structure

```
graph-rca/
├── frontend/
│   └── app.py              # Streamlit web interface
├── core/
│   ├── rag.py              # RAG engine for solution generation
│   ├── embedding.py        # Embedding creation
│   └── database_handlers.py # MongoDB and ChromaDB handlers
├── utilz/
│   ├── log_parser.py       # Log parsing with LLM
│   ├── graph_generator.py  # DAG generation from logs
│   ├── context_builder.py  # Context building from DAG
│   └── database_healthcheck.py # Service health checks
├── models/
│   ├── parsing_data_models.py      # Log entry models
│   ├── graph_data_models.py        # DAG models
│   ├── context_data_models.py      # Context models
│   └── rag_response_data_models.py # RAG response models
├── data/                   # Persistent data (gitignored)
│   ├── chroma/            # ChromaDB data
│   ├── db/                # MongoDB data
│   └── ollama/            # Ollama models
├── tests/                 # Unit tests
├── experiments/           # Experimental scripts
├── docker-compose.yaml    # Docker Compose configuration
├── start.sh                # Start Docker containers (with sudo)
├── stop.sh                 # Stop Docker containers (with sudo)
└── requirements.txt      # Python dependencies
```

## Data Persistence

All data is stored in the `./data` directory:
- **ChromaDB data:** `./data/chroma/`
- **MongoDB data:** `./data/db/`
- **Ollama models:** `./data/ollama/`

This directory is gitignored. Your data persists even when containers are stopped.

## Common Workflows

### First Time Setup

1. Clone repository
2. Create virtual environment
3. Install dependencies
4. Start services with `docker-compose up -d`
5. Install Ollama models
6. Run `streamlit run frontend/app.py --server.port 8550`
7. Upload a test log file
8. Upload documentation
9. Generate solutions

### Daily Usage

1. Start services: `docker-compose up -d` (if not running)
2. Run application: `streamlit run frontend/app.py --server.port 8550`
3. Upload logs and documentation as needed
4. Review generated solutions

### Resetting Data

To start fresh (removes all stored data):

```bash
# Stop services
docker-compose down

# Remove data directories
rm -rf ./data/chroma ./data/db ./data/ollama

# Restart services
docker-compose up -d

# Reinstall models
docker exec -it graph-rca-ollama ollama pull llama3.2:3b
docker exec -it graph-rca-ollama ollama pull nomic-embed-text
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
