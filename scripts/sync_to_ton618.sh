#!/usr/bin/env bash
# Copy the repo (code only, no venv/runs/results) to ton-618 over Tailscale.
# Windows OpenSSH has tar but not rsync, and this box's remote shell is PowerShell,
# so: tar locally -> scp -> extract remotely.
#   scripts/sync_to_ton618.sh [remote_dir]
set -euo pipefail
HOST="${TON618_HOST:-singu@ton-618}"
DEST="${1:-atc-copilot}"
cd "$(dirname "$0")/.."
TARBALL="$(mktemp -t atc-sync).tgz"
trap 'rm -f "$TARBALL"' EXIT
COPYFILE_DISABLE=1 tar --exclude .venv --exclude .git --exclude runs --exclude logs \
    --exclude 'results/*.jsonl' --exclude __pycache__ --exclude .DS_Store -czf "$TARBALL" .
scp -q "$TARBALL" "$HOST:atc-sync.tgz"
ssh "$HOST" "New-Item -Force -ItemType Directory $DEST | Out-Null; tar -xzf atc-sync.tgz -C $DEST; Remove-Item atc-sync.tgz"
echo "synced to $HOST:$DEST"
