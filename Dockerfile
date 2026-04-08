FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-caching)
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Install project as editable package
RUN pip install --no-cache-dir -e .

EXPOSE 7860

ENV HOST=0.0.0.0
ENV PORT=7860
ENV WORKERS=2

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

CMD uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS
