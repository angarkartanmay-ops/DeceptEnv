# DeceptEnv server image — designed for Hugging Face Spaces (sdk: docker).
# The image runs ONLY the env server. Training (TRL/Unsloth) happens
# elsewhere (Colab/laptop) and talks to the Space over HTTP.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Hugging Face Spaces injects the user 'user' (uid 1000); keep matching default.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# --- Server-only requirements (small image, fast cold start) ---------------
# We deliberately avoid torch/transformers in the Spaces image — the env
# server only needs FastAPI + the optional LLM-detective SDKs.
COPY requirements-server.txt ./
RUN pip install --upgrade pip && pip install -r requirements-server.txt

# Copy only what the server needs at runtime.
# README.md + docs/assets/ are required so GET / can render the full PRD card
# with embedded plots inside the HF Spaces "App" iframe.
COPY server/        ./server/
COPY openenv.yaml   ./openenv.yaml
COPY README.md      ./README.md
COPY docs/          ./docs/

USER user
ENV DECEPTENV_HOST=0.0.0.0 \
    DECEPTENV_PORT=7860 \
    DECEPTENV_DETECTIVE=mock

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; \
                 r=urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=3); \
                 sys.exit(0 if r.status==200 else 1)"

CMD ["python", "-m", "server.app"]
