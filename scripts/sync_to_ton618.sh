#!/usr/bin/env bash
# Copy the repo (code only, no venv/runs/results) to ton-618 over Tailscale.
# Windows OpenSSH has tar but not rsync, and its default remote shell is cmd.exe.
#   scripts/sync_to_ton618.sh [remote_dir]
set -euo pipefail
HOST="${TON618_HOST:-ton-618}"
DEST="${1:-atc-copilot}"
cd "$(dirname "$0")/.."
COPYFILE_DISABLE=1 tar --exclude .venv --exclude .git --exclude runs --exclude logs \
    --exclude 'results/*.jsonl' --exclude __pycache__ -czf - . |
  ssh "$HOST" "if not exist $DEST mkdir $DEST && tar -xzf - -C $DEST"
echo "synced to $HOST:$DEST"
