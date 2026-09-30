# agent-harness: API server and queue workers.
#   docker build -t agent-harness .
#   docker run --rm agent-harness --help
# Secrets (API keys, model keys, database URL) are provided at run time through
# environment variables or mounted files (file:/run/secrets/...), never baked in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 10001 harness
WORKDIR /opt/agent-harness
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api,postgres]"

USER harness
WORKDIR /home/harness
EXPOSE 8000
ENTRYPOINT ["agent-harness"]
CMD ["--help"]
