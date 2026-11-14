# CPU-only image that serves vg/api.py. See docs/SERVING.md.
# Build from the repository root after a trained run exists in runs/baseline:
#   docker build -t visionguard-api .
#   docker run --rm -p 7860:7860 visionguard-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    VG_RUN_DIR=/app/model \
    VG_PRELOAD=1 \
    PORT=7860

WORKDIR /app
# Non-root user; uid 1000 is also what Hugging Face Docker Spaces expect.
RUN useradd --create-home --uid 1000 app

COPY requirements-api.txt .
RUN python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu \
 && python -m pip install -r requirements-api.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY vg/ vg/
# verify_selection() checks the hash of every checkpoint listed in selection.json, so all three are needed.
COPY runs/baseline/selection.json runs/baseline/cnn.pt runs/baseline/frozen.pt runs/baseline/finetuned.pt /app/model/

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '7860'), timeout=4)"
CMD ["sh", "-c", "exec uvicorn vg.api:app --host 0.0.0.0 --port ${PORT} --workers 1"]
