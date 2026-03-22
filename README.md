# Graph-RCA: Log Analysis & Incident Resolution System

An AI-powered system for automated log analysis and incident resolution. Features a modern React frontend with a FastAPI backend, leveraging LLMs for root cause analysis and solution generation.

## 🎯 Features

- 📊 **Automated log file analysis** with causal chain generation
- 🔍 **Root cause identification** using graph-based analysis
- 📝 **Documentation integration** for context-aware solutions
- 🤖 **AI-powered incident resolution** using RAG (Retrieval Augmented Generation)
- 💎 **Modern glass-morphism UI** built with React + TypeScript + Tailwind
- 🚀 **FastAPI backend** with RESTful endpoints
- 🗄️ **Persistent storage** with MongoDB and ChromaDB

## 📁 Project Structure

```
graph-rca/
├── backend/                        # Python FastAPI backend
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py           # API route handlers
│   │   ├── core/
│   │   │   ├── database_handlers.py     # MongoDB + ChromaDB clients
│   │   │   ├── database_handlers_gpu.py # GPU-accelerated variant
│   │   │   ├── embedding.py             # Embedding utilities
│   │   │   └── rag.py                   # RAG engine
│   │   ├── models/
│   │   │   ├── context_data_models.py
│   │   │   ├── graph_data_models.py
│   │   │   ├── parsing_data_models.py
│   │   │   └── rag_response_data_models.py
│   │   └── utils/
│   │       ├── context_builder.py       # DAG traversal & context extraction
│   │       ├── database_healthcheck.py  # Service health checks
│   │       ├── graph_generator.py       # DAG construction
│   │       └── log_parser.py            # LLM-based log parser
│   ├── tests/                      # Backend unit tests (5 files)
│   ├── main.py                     # FastAPI application entry point
│   └── requirements.txt            # Python dependencies
│
├── frontend/                       # React TypeScript frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── AnalysisHistory.tsx
│   │   │   ├── DocsUploadPanel.tsx
│   │   │   ├── IncidentResolutionPanel.tsx
│   │   │   ├── Layout.tsx
│   │   │   ├── LogUploadPanel.tsx
│   │   │   ├── StatusSidebar.tsx
│   │   │   └── StepTabs.tsx
│   │   ├── api.ts                  # API client
│   │   ├── App.tsx                 # Main application
│   │   ├── main.tsx                # Entry point
│   │   ├── store.ts                # Zustand state (persisted to localStorage)
│   │   └── styles.css
│   └── package.json
│
├── experiments/                    # 9 reproducible experiment scripts
│   ├── 01_batch_inference/
│   ├── 02_scalability/
│   ├── 03_baseline_comparison/
│   ├── 04_doc_ablation/
│   ├── 05_noise_sensitivity/
│   ├── 06_parser_accuracy/
│   ├── 07_multi_judge_validation/
│   ├── 08_rag_real_world/
│   ├── 09_latency_profiling/
│   └── README.md
│
├── results/                        # Raw JSON outputs from experiment runs
├── data/
│   ├── real_incidents/             # 200 annotated production incidents
│   ├── chroma/                     # ChromaDB persistence (auto-created)
│   ├── db/                         # MongoDB persistence (auto-created)
│   └── ollama/                     # Ollama model cache (auto-created)
├── docs/                           # Sample documentation corpus
│   ├── API.md
│   ├── sample_documentation.txt
│   ├── sample_log.log
│   └── sample_log_2.log
│
├── .env.example                    # Environment variable template (copy to .env)
├── docker-compose.yaml             # Docker services (MongoDB, ChromaDB, Ollama)
├── docker-compose.gpu.yaml         # GPU-accelerated deployment variant
├── run.sh                          # One-command setup script
├── run-gpu-server.sh               # Setup for remote GPU servers
├── run_all_experiments.py          # Run all 9 experiments sequentially
├── check_prerequisites.py          # Pre-flight dependency checker
├── start-backend.sh                # Start FastAPI backend
└── start-frontend.sh               # Start Vite dev server
```

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose**
- **Python 3.11+** (3.11, 3.12, or 3.13 — auto-detected by `run.sh`)
- **Node.js 18+**
- **NVIDIA GPU** (optional, for faster LLM inference — Linux only)

### One-Command Setup

> ⚠️ **Before first run:** copy the environment template and fill in your values:
> ```bash
> cp .env.example .env
> # Edit .env — at minimum set MONGO_URI and OLLAMA_HOST
> ```

```bash
git clone https://github.com/KTS-o7/graph-rca.git
cd graph-rca
./run.sh
```

The script will:
- ✓ Check Docker
- ✓ Start all Docker services (MongoDB, ChromaDB, Ollama)
- ✓ Download the LLM model if needed
- ✓ Set up Python virtual environment
- ✓ Install all Python dependencies
- ✓ Install all Node.js dependencies
- ✓ Create convenience start scripts

### Start the Application

After running `./run.sh`, open **two separate terminals**:

**Terminal 1 - Backend:**
```bash
./start-backend.sh
```

**Terminal 2 - Frontend:**
```bash
./start-frontend.sh
```

Then open your browser to `http://localhost:5173`

### What Gets Started

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:5173 | React UI |
| **Backend API** | http://localhost:8010 | FastAPI server |
| **API Docs** | http://localhost:8010/docs | Interactive API documentation |
| MongoDB | localhost:27017 | Document store |
| ChromaDB | localhost:8000 | Vector database |
| Ollama | localhost:11435 | LLM inference |

## 📖 Usage

### Step-by-step workflow:

1. **Analyse Log File**
   - Navigate to the "1. Log Analysis" tab
   - Upload a `.log` or `.txt` file (**max 5 MB, 500 lines** — larger files are truncated)
   - Click "Analyse log"
   - View severity, root cause, and summary

2. **Add Documentation** (Optional but recommended)
   - Go to "2. Documentation" tab
   - Upload runbooks, guides, or service docs (`.txt` or `.md`)
   - Click "Index documentation"

3. **Generate Resolution**
   - Open "3. Incident Resolution" tab
   - Click "Run resolution"
   - Review the generated solution with references

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/log/analyse` | Analyse uploaded log file (returns context + root cause) |
| `POST` | `/api/docs/upload` | Upload documentation files |
| `POST` | `/api/incident/resolve` | Generate incident resolution (requires context from `/analyse`) |

Full API docs: `http://localhost:8010/docs`

## ⚙️ Environment Variables

Create a `.env` file (copy from `.env.example`) and set the following variables before running:

| Variable | Required | Default | Description |
|---|---|---|---|
| `MONGO_URI` | **Yes** | `mongodb://localhost:27017/` | MongoDB connection string. Use `mongodb://admin:changeme@localhost:27017/` for the bundled Docker Compose setup. |
| `OLLAMA_HOST` | **Yes** | `http://localhost:11435` | Ollama service URL. Port 11435 matches the Docker Compose host mapping. |
| `ALLOWED_ORIGINS` | No | `http://localhost:5173,http://localhost:3000` | Comma-separated CORS origins. Override for production deployments. |
| `OPENAI_API_KEY` | For exp 07/08 | — | Required for multi-judge validation experiments with GPT-4o-mini. |
| `GROQ_API_KEY` | For exp 07/08 | — | Required for multi-judge validation experiments with Groq Llama-70B. |

```bash
cp .env.example .env
# Edit .env with your values
```

## 🧪 Running Experiments

To reproduce all paper results:

```bash
# Run all 9 experiments sequentially
python run_all_experiments.py

# Or run a specific experiment
python experiments/01_batch_inference/run_experiment.py
python experiments/07_multi_judge_validation/run_experiment.py
# ... etc
```

Results are saved as JSON in `results/`. Pre-computed outputs are already included for verification without re-execution.

## 🐳 Docker Services

Check running services:
```bash
docker-compose ps
```

View logs:
```bash
docker-compose logs <service-name>
# Examples:
docker-compose logs ollama
docker-compose logs chroma
docker-compose logs mongodb
```

Stop all services:
```bash
docker-compose down
```

## 🛠️ Development

### Backend Development

```bash
cd backend
source venv/bin/activate
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8010
```

### Frontend Development

```bash
cd frontend
npm run dev
```

### Run Backend Tests

```bash
cd backend
source venv/bin/activate
pytest tests/
```

## 🎨 Tech Stack

**Frontend:**
- React 18 + TypeScript
- Vite (build tool)
- Tailwind CSS (styling)
- Zustand (state management with localStorage persistence)
- React Icons

**Backend:**
- FastAPI (Python web framework)
- Pydantic (data validation)
- Ollama (local LLM inference — Llama 3.2 3B, Qwen 2.5-Coder)
- LangChain (text processing & chunking)

**Databases:**
- ChromaDB (vector embeddings, HNSW index)
- MongoDB (DAG structures & analysis contexts)

**Experiment LLM Judges:**
- Qwen3:32b (local via Ollama)
- GPT-4o-mini (OpenAI API)
- Llama-3.1-70B (Groq API)

## 🔍 Troubleshooting

**Backend won't start:**
- Ensure Docker services are running: `docker ps`
- Run the pre-flight checker: `python check_prerequisites.py`
- Check Ollama has the model: `docker exec -it $(docker ps -qf "name=ollama") ollama list`

**Frontend can't connect:**
- Verify backend is running on port 8010
- For CORS errors in production, set the `ALLOWED_ORIGINS` environment variable:
  ```bash
  ALLOWED_ORIGINS=https://yourdomain.com ./start-backend.sh
  ```

**GPU Support (Optional):**
- By default, Ollama runs on CPU (works on all systems)
- For NVIDIA GPU acceleration (Linux only):
  ```bash
  docker-compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d
  ```
- For remote GPU servers, use `./run-gpu-server.sh` instead of `./run.sh`
- Requires: NVIDIA drivers, CUDA 12.x, and nvidia-docker2 installed

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
