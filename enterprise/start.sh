#!/bin/bash
# Start the Callex Enterprise Platform

ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "=========================="
echo " Callex Enterprise Platform"
echo "=========================="

# Start backend
echo "[1/2] Starting backend on http://localhost:4000 ..."
cd "$ROOT/backend" && node src/index.js &
BACKEND_PID=$!

sleep 2
echo "[2/2] Starting frontend on http://localhost:3000 ..."
cd "$ROOT/frontend" && npm run dev

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT
