#!/usr/bin/env bash
# Container entrypoint for the Telegram support bot.
#
# Runs three steps before handing off to the bot:
#   1. Seed the data volume from the baked copy on first run (never clobbers
#      live CRM writes — only fills in files the volume is missing).
#   2. Build the support_kb index if it is empty (the slow embedder load only
#      happens on the first ever start, then the index persists in the volume).
#   3. exec the bot as PID 1 so SIGTERM reaches it for a clean shutdown.
set -euo pipefail

DATA_DIR="${SUPPORT_DATA_DIR:-/app/data/support}"
SEED_DIR="/app/seed/support"

# --- 1. seed data volume ------------------------------------------------------
mkdir -p "$DATA_DIR"
if [ -d "$SEED_DIR" ]; then
    for f in "$SEED_DIR"/*; do
        name="$(basename "$f")"
        if [ ! -e "$DATA_DIR/$name" ]; then
            echo "entrypoint: seeding $name into data volume"
            cp "$f" "$DATA_DIR/$name"
        fi
    done
fi

# --- 2. build index if empty --------------------------------------------------
# The bot itself refuses to start on an empty support_kb, so build it here.
# Reuses the app's own count() so the check matches the bot's expectation.
needs_index="$(python - <<'PY'
from dataclasses import replace
from assistant.config import Config
from assistant.core.rag import RagSearcher
cfg = Config.load(require_api_key=False)
cfg = replace(cfg, chroma_collection=cfg.support_chroma_collection)
try:
    empty = RagSearcher(cfg).count() == 0
except Exception:
    empty = True
print("yes" if empty else "no")
PY
)"

if [ "$needs_index" = "yes" ]; then
    echo "entrypoint: support_kb index empty, building…"
    python -m support_assistant.index_support_kb
else
    echo "entrypoint: support_kb index present, skipping build"
fi

# --- 3. run the bot -----------------------------------------------------------
echo "entrypoint: starting bot"
exec python -m support_bot.bot
