FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-caching)
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Install project as editable package
RUN pip install --no-cache-dir -e .

# Try installing openm-core CLI (official OpenEnv package)
# Falls back gracefully if not yet on PyPI — direct FastAPI impl handles all endpoints
RUN pip install --no-cache-dir openm-core || echo "openm-core not found on PyPI, continuing with direct FastAPI"

EXPOSE 7860

ENV HOST=0.0.0.0
ENV PORT=7860
ENV WORKERS=2
# Enable Gradio web UI for manual testing at /web (localhost:7860/web)
ENV ENABLE_WEB_INTERFACE=true

CMD uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS
