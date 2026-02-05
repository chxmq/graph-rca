#!/usr/bin/env python3
"""
Prerequisites Checker for GraphRCA Experiments

Checks if all required dependencies, models, and data are available before running experiments.

Usage:
    python check_prerequisites.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Required models
REQUIRED_MODELS = {
    "llama3.2:3b": [1, 3, 6, 8, 9],
    "qwen3:32b": [4, 7],
    "nomic-embed-text": [5, 8],
}

# Required data directories
REQUIRED_DATA = {
    "real_incidents": [4, 5, 7, 8],
    "loghub": [6],
}

# Required Python packages
REQUIRED_PACKAGES = {
    "ollama": [1, 3, 4, 5, 6, 7, 8, 9],
    "chromadb": [5, 8, 9],
    "openai": [7],  # Optional
    "groq": [7],    # Optional
}


def check_ollama():
    """Check if Ollama is running and accessible."""
    print("Checking Ollama...")
    try:
        import ollama
        client = ollama.Client(host="http://localhost:11434", timeout=5.0)
        client.list()
        print("  ✓ Ollama is running")
        return True, []
    except ImportError:
        print("  ✗ Ollama Python package not installed")
        print("    Install: pip install ollama")
        return False, ["ollama package"]
    except Exception as e:
        print(f"  ✗ Ollama not accessible: {e}")
        print("    Start Ollama: ollama serve")
        return False, ["ollama service"]


def check_models():
    """Check if required models are available."""
    print("\nChecking Ollama models...")
    missing = []
    
    try:
        import ollama
        client = ollama.Client(host="http://localhost:11434", timeout=10.0)
        try:
            # Handle different versions of ollama python client
            raw_models = client.list().get("models", [])
        except:
            raw_models = client.list()

        available_models = []
        for m in raw_models:
            # Try dict access (newer/older mismatch)
            if isinstance(m, dict):
                name = m.get("name") or m.get("model")
            # Try object attribute checks
            else:
                 name = getattr(m, "name", None) or getattr(m, "model", None)
            
            if name:
                available_models.append(name)
        
        for model_name, exp_nums in REQUIRED_MODELS.items():
            # Loose matching to handle tags like 'latest' if needed, but exact is better for repro
            if model_name in available_models:
                print(f"  ✓ {model_name} (needed for experiments: {exp_nums})")
            else:
                print(f"  ✗ {model_name} NOT FOUND (needed for experiments: {exp_nums})")
                print(f"    Install: ollama pull {model_name}")
                missing.append(model_name)
        
        return len(missing) == 0, missing
    except Exception as e:
        print(f"  ✗ Cannot check models: {e}")
        return False, ["model check failed"]


def check_packages():
    """Check if required Python packages are installed."""
    print("\nChecking Python packages...")
    missing = []
    
    for package, exp_nums in REQUIRED_PACKAGES.items():
        try:
            __import__(package)
            print(f"  ✓ {package} (needed for experiments: {exp_nums})")
        except ImportError:
            optional = package in ["openai", "groq"]
            status = "⚠" if optional else "✗"
            print(f"  {status} {package} NOT INSTALLED (needed for experiments: {exp_nums})")
            if not optional:
                print(f"    Install: pip install {package}")
                missing.append(package)
            else:
                print(f"    Optional - install if running experiment 7 with {package.upper()} judge")
    
    return len(missing) == 0, missing


def check_data_directories():
    """Check if required data directories exist."""
    print("\nChecking data directories...")
    missing = []
    
    for data_name, exp_nums in REQUIRED_DATA.items():
        data_path = PROJECT_ROOT / "data" / data_name
        
        if data_name == "real_incidents":
            if data_path.exists() and len(list(data_path.glob("incident_*"))) > 0:
                count = len(list(data_path.glob("incident_*")))
                print(f"  ✓ {data_name}/ ({count} incidents found, needed for experiments: {exp_nums})")
            else:
                print(f"  ✗ {data_name}/ NOT FOUND (needed for experiments: {exp_nums})")
                print(f"    Expected: {data_path}")
                missing.append(data_name)
        
        elif data_name == "loghub":
            if data_path.exists():
                print(f"  ✓ {data_name}/ (will auto-download if missing, needed for experiment: {exp_nums})")
            else:
                print(f"  ⚠ {data_name}/ not found (will auto-download, needed for experiment: {exp_nums})")
                print(f"    Expected: {data_path}")
    
    return len(missing) == 0, missing


def check_backend_dependencies():
    """Check if backend dependencies are available."""
    print("\nChecking backend dependencies...")
    backend_path = PROJECT_ROOT / "backend"
    
    if not backend_path.exists():
        print("  ✗ Backend directory not found")
        print(f"    Expected: {backend_path}")
        return False, ["backend directory"]
    
    try:
        sys.path.insert(0, str(backend_path))
        from app.utils.graph_generator import GraphGenerator
        print("  ✓ Backend dependencies available (needed for experiment 2)")
        return True, []
    except ImportError as e:
        print(f"  ✗ Backend dependencies not available: {e}")
        print("    Install: cd backend && pip install -r requirements.txt")
        return False, ["backend dependencies"]


def check_api_keys():
    """Check optional API keys."""
    print("\nChecking optional API keys...")
    
    if "OPENAI_API_KEY" in os.environ:
        print("  ✓ OPENAI_API_KEY set (for GPT judge in experiment 7)")
    else:
        print("  ⚠ OPENAI_API_KEY not set (optional, for GPT judge in experiment 7)")
        print("    To set: export OPENAI_API_KEY='your-key-here'")
    
    if "GROQ_API_KEY" in os.environ:
        print("  ✓ GROQ_API_KEY set (for Llama-70B judge in experiment 7)")
    else:
        print("  ⚠ GROQ_API_KEY not set (optional, for Llama-70B judge in experiment 7)")
        print("    To set: export GROQ_API_KEY='your-key-here'")


def main():
    print("="*70)
    print("GraphRCA Experiments - Prerequisites Checker")
    print("="*70)
    
    all_ok = True
    all_missing = []
    
    # Check Ollama
    ok, missing = check_ollama()
    all_ok = all_ok and ok
    all_missing.extend(missing)
    
    # Check models (only if Ollama is working)
    if ok:
        ok, missing = check_models()
        all_ok = all_ok and ok
        all_missing.extend(missing)
    
    # Check packages
    ok, missing = check_packages()
    all_ok = all_ok and ok
    all_missing.extend(missing)
    
    # Check data
    ok, missing = check_data_directories()
    all_ok = all_ok and ok
    all_missing.extend(missing)
    
    # Check backend
    ok, missing = check_backend_dependencies()
    all_ok = all_ok and ok
    all_missing.extend(missing)
    
    # Check API keys (optional)
    check_api_keys()
    
    # Summary
    print("\n" + "="*70)
    if all_ok:
        print("✓ All required prerequisites are met!")
        print("  You can run: python run_all_experiments.py")
    else:
        print("✗ Some prerequisites are missing:")
        for item in set(all_missing):
            print(f"  - {item}")
        print("\n  Please install missing prerequisites before running experiments.")
    print("="*70)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
