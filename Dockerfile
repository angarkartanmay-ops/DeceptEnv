# Hugging Face Spaces (sdk: docker). Image runs only the env server;
# training happens elsewhere and talks to the Space over HTTP.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Spaces injects uid 1000 named "user".
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements-server.txt ./
RUN pip install --upgrade pip && pip install -r requirements-server.txt

# README.md + docs/ are needed at runtime so the static-asset mount can
# serve the plot images embedded in the SPA.
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
