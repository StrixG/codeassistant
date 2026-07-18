# Telegram support bot image.
#
# Ships the bot plus its two runtime companions: the mcp_crm.server subprocess
# (needs the `mcp` extra) and the RAG stack (chromadb + sentence-transformers,
# pulled in by the base dependencies). The support_kb index is not baked in —
# the entrypoint builds it on first start into a mounted volume, so a fresh
# clone needs no host-side setup.

FROM python:3.11-slim

# Build toolchain for chromadb / sentence-transformers native wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models

WORKDIR /app

# Install deps first so code edits don't bust the wheel cache.
COPY pyproject.toml README.md ./
# Package tree needed for the metadata build (`packages.find`) and install.
COPY assistant ./assistant
COPY mcp_crm ./mcp_crm
COPY support_assistant ./support_assistant
COPY support_bot ./support_bot

RUN pip install --upgrade pip && pip install ".[bot,mcp]"

# Seed data (faq, product guide, tickets, users). Baked to /app/seed so the
# entrypoint can populate an empty data volume on first run without ever
# overwriting live CRM writes.
COPY data ./seed

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Non-root user owning the mutable paths (data volume, index, model cache).
# Pre-create every volume mount point owned by bot. Docker seeds an empty
# named volume from the image path *including ownership*, so these dirs must
# exist and be bot-owned here or the fresh volume lands root-owned and the
# non-root process can't write to it.
RUN useradd --create-home --uid 10001 bot \
    && mkdir -p /app/data/support /app/.chroma /models \
    && chown -R bot:bot /app /models
USER bot

ENTRYPOINT ["entrypoint.sh"]
