#!/bin/sh
set -eu

uvicorn main:app --host 0.0.0.0 --port 8000 &
api_pid=$!

cleanup() {
  kill "$api_pid" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

streamlit run app.py --server.address 0.0.0.0 --server.port 8501