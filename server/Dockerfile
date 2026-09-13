FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py qvalidate.py questions.py .

# AI-authored question code runs in a qvalidate.py subprocess; don't give it root.
USER nobody

# Fly.io injects PORT; default 8080 matches fly.toml internal_port
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
