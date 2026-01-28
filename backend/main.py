import signal
import sys
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

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
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
