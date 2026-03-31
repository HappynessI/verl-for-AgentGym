#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME=${SESSION_NAME:-textcraft_server_new_prefix}
PORT=${PORT:-36001}
HOST=${HOST:-0.0.0.0}
UVICORN_BIN=${UVICORN_BIN:-/Data/wyh/conda_envs/agentenv-textcraft/bin/uvicorn}

if ss -ltnp | grep -q ":${PORT} "; then
  echo "TextCraft server is already listening on port ${PORT}."
  ss -ltnp | grep ":${PORT} "
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "${UVICORN_BIN} agentenv_textcraft:app --host ${HOST} --port ${PORT}"

sleep 2

if ss -ltnp | grep -q ":${PORT} "; then
  echo "TextCraft server started on ${HOST}:${PORT} in tmux session ${SESSION_NAME}."
  ss -ltnp | grep ":${PORT} "
  exit 0
fi

echo "Failed to start TextCraft server on ${HOST}:${PORT}."
exit 1
