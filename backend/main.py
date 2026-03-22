# Fix for ChromaDB SQLite issue on older systems
import sys
import os
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import signal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

# Handle Ctrl+C gracefully
def signal_handler(sig, frame):
    print('\n[!] Shutting down gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

app = FastAPI(
    title="Graph-RCA Backend API",
    description="Log analysis and incident resolution backend",
    version="1.0.0",
)

# CORS: read from env var so any deployment works without code changes.
# e.g. ALLOWED_ORIGINS=https://myrca.example.com,http://localhost:5173
_default_origins = "http://localhost:5173,http://localhost:3000"
origins = os.environ.get("ALLOWED_ORIGINS", _default_origins).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "Graph-RCA Backend API",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)
