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
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── core/           # Database handlers, embeddings, RAG engine
│   │   ├── models/         # Pydantic data models
│   │   └── utils/          # Log parser, graph generator, context builder
│   ├── tests/              # Backend tests
│   ├── main.py             # FastAPI application entry point
│   └── requirements.txt    # Python dependencies
│
├── frontend/               # React TypeScript frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── api.ts         # API client
│   │   ├── store.ts       # State management (Zustand)
│   │   └── App.tsx        # Main application
│   └── package.json       # Node dependencies
│
├── docs/                  # Documentation and samples
├── data/                  # Docker volume mounts (auto-created)
├── docker-compose.yaml    # Docker services (MongoDB, ChromaDB, Ollama)
└── run.sh                 # Master setup script (run this first!)
```

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose**
- **Python 3.13+**
- **Node.js 18+**
- **NVIDIA GPU** (optional, for faster LLM inference - Linux only)

### One-Command Setup

```bash
git clone https://github.com/KTS-o7/graph-rca.git
cd graph-rca
./run.sh
```

That's it! The script will:
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
| MongoDB | localhost:27017 | Database |
| ChromaDB | localhost:8000 | Vector database |
| Ollama | localhost:11435 | LLM inference |

## 📖 Usage

### Step-by-step workflow:

1. **Analyse Log File**
   - Navigate to the "1. Log Analysis" tab
   - Upload a `.log` or `.txt` file
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
| `POST` | `/api/log/analyse` | Analyse uploaded log file |
| `POST` | `/api/docs/upload` | Upload documentation |
| `POST` | `/api/incident/resolve` | Generate incident resolution |

Full API docs: `http://localhost:8010/docs`

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

### Run Tests

```bash
cd backend
source venv/bin/activate
pytest tests/
```

### Stop All Services

```bash
docker-compose down
# Kill backend/frontend with Ctrl+C in their terminals
```

## 🎨 Tech Stack

**Frontend:**
- React 18 + TypeScript
- Vite (build tool)
- Tailwind CSS (styling)
- Zustand (state management)
- React Icons

**Backend:**
- FastAPI (Python web framework)
- Pydantic (data validation)
- Ollama (local LLM inference)
- LangChain (text processing)

**Databases:**
- ChromaDB (vector embeddings)
- MongoDB (structured data)

## 🔍 Troubleshooting

**Backend won't start:**
- Ensure Docker services are running: `docker ps`
- Check Ollama has the model: `docker exec -it $(docker ps -qf "name=ollama") ollama list`

**Frontend can't connect:**
- Verify backend is running on port 8010
- Check CORS settings in `backend/main.py`

**GPU Support (Optional):**
- By default, Ollama runs on CPU (works on all systems)
- For NVIDIA GPU acceleration (Linux only):
  ```bash
  docker-compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d
  ```
- Requires: NVIDIA drivers, CUDA, and nvidia-docker2 installed

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
