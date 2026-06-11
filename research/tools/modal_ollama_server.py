"""
Serverless Ollama on Modal (https://modal.com) for GraphRCA research runs.

Exposes a standard Ollama HTTP API backed by a 24 GB GPU, so every existing
script (regenerate_symptom_logs.py, experiments 07/08) works unchanged by
pointing OLLAMA_HOST at the Modal URL.  Models are cached in a Modal Volume,
so they download once, not per container start.

Setup (one-time, on your laptop):
  pip install modal
  modal setup                      # opens browser to authenticate

Start the server (keeps running while the terminal is open; scales to zero
a few minutes after you stop sending requests, so idle time costs nothing):
  modal serve research/tools/modal_ollama_server.py

Or deploy it persistently (still scales to zero when idle):
  modal deploy research/tools/modal_ollama_server.py

Then, from your laptop:
  export OLLAMA_HOST=https://<your-username>--graphrca-ollama-serve.modal.run
  curl $OLLAMA_HOST/api/tags                       # sanity check
  curl $OLLAMA_HOST/api/pull -d '{"name": "qwen2.5-coder:32b"}'   # ~20 min, once
  python research/tools/regenerate_symptom_logs.py --model qwen2.5-coder:32b --dry-run

Security note: anyone with the URL can use your GPU.  Either treat the URL
as a secret, or enable proxy auth by setting requires_proxy_auth=True below
and exporting MODAL_KEY / MODAL_SECRET (Modal dashboard -> Proxy Auth
Tokens); regenerate_symptom_logs.py forwards those headers automatically.

Cost notes (check current pricing): A10G ~ $1.10/hr, L4 ~ $0.80/hr.  The
free monthly credits (~$30) buy roughly 25-35 GPU-hours.  If qwen2.5-coder:32b
feels tight on 24 GB, switch GPU_TYPE to "L40S" (48 GB, pricier).
"""

import os
import subprocess

import modal

GPU_TYPE = os.environ.get("MODAL_GPU", "A10G")  # A10G / L4 / L40S / A100
OLLAMA_PORT = 11434

app = modal.App("graphrca-ollama")

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "ca-certificates", "zstd", "pciutils")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
)

# Persistent model cache: survives container restarts so `ollama pull`
# happens once per model, not once per session.
models_volume = modal.Volume.from_name("graphrca-ollama-models", create_if_missing=True)


# Models guaranteed available when the server reports ready.  First-time
# downloads land in the volume and are committed, so later cold starts are
# just a volume mount, not a re-download.
REQUIRED_MODELS = ["llama3.2:3b", "qwen2.5-coder:32b", "qwen3:32b", "nomic-embed-text"]


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/root/.ollama": models_volume},
    timeout=60 * 60 * 6,        # allow long batch sessions
    scaledown_window=60 * 10,   # scale to zero after 10 idle minutes
    max_containers=1,           # single instance: no stale-volume split brain
)
@modal.web_server(port=OLLAMA_PORT, startup_timeout=60 * 30)
def serve():
    import time
    import urllib.request

    models_volume.reload()  # pick up models committed by earlier containers

    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"0.0.0.0:{OLLAMA_PORT}"
    # Keep the model loaded between requests within a session; reloading a
    # 32B model per request would dominate runtime.
    env["OLLAMA_KEEP_ALIVE"] = "30m"
    subprocess.Popen(["ollama", "serve"], env=env)

    # Wait for the local API before pulling.
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{OLLAMA_PORT}/api/tags", timeout=2)
            break
        except Exception:
            time.sleep(1)

    pulled_new = False
    have = subprocess.run(["ollama", "list"], capture_output=True, text=True, env=env).stdout
    for model in REQUIRED_MODELS:
        if model.split(":")[0] not in have:
            subprocess.run(["ollama", "pull", model], check=True, env=env)
            pulled_new = True
    if pulled_new:
        models_volume.commit()  # make downloads durable for future containers
