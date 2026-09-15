#!/usr/bin/env bash
# Install / refresh the read-only interactive viewer systemd unit on the lab VM.
# Run from the repo root on the server:
#   bash deploy/install-dashboard-interactive.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_SRC="$REPO/deploy/dashboard-interactive.service"
UNIT_DST="/etc/systemd/system/dashboard-interactive.service"

if [[ ! -f "$UNIT_SRC" ]]; then
  echo "Missing $UNIT_SRC" >&2
  exit 1
fi

echo "Repo: $REPO"
echo "Installing $UNIT_DST"

# Patch WorkingDirectory / PYTHONPATH to this clone if it is not the default.
TMP="$(mktemp)"
sed \
  -e "s|^WorkingDirectory=.*|WorkingDirectory=$REPO|" \
  -e "s|^Environment=PYTHONPATH=.*|Environment=PYTHONPATH=$REPO|" \
  "$UNIT_SRC" >"$TMP"

sudo cp "$TMP" "$UNIT_DST"
rm -f "$TMP"
sudo systemctl daemon-reload
sudo systemctl enable dashboard-interactive
sudo systemctl restart dashboard-interactive
sudo systemctl --no-pager --full status dashboard-interactive || true

echo
echo "Checks:"
echo "  systemctl cat dashboard-interactive | grep -E 'ExecStart|DASH_'"
echo "  curl -sI http://127.0.0.1:8052/interactive/ | head"
echo "Expect ExecStart to contain: src.GUI.wsgi:server"
echo "Expect UI: no Working folder / Register / Filter (read-only)."
