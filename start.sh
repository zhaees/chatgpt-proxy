#!/bin/bash
# Start the toolcall-middleware with ChatGPT proxy
# 
# Architecture:
#   chatgpt-http-helper.py (curl_cffi, port 1436) → chatgpt.com
#   proxy.ts (bun, port 1435) → routes chatgpt/* models to helper
#
# Usage: ./start.sh

cd "$(dirname "$0")"

# Kill existing processes
pkill -f "chatgpt-http-helper.py" 2>/dev/null
pkill -f "bun.*proxy.ts" 2>/dev/null
sleep 1

# Start the HTTP helper first (needs time to warm up)
echo "Starting ChatGPT HTTP helper..."
python3 chatgpt-http-helper.py &
HELPER_PID=$!

# Wait for helper to be ready
for i in $(seq 1 30); do
  if curl -s --connect-timeout 2 http://127.0.0.1:1436/health 2>/dev/null | grep -q "ok"; then
    echo "HTTP helper ready ✓"
    break
  fi
  sleep 1
done

# Start the proxy
echo "Starting proxy..."
bun run proxy.ts &
PROXY_PID=$!

echo ""
echo "  toolcall-middleware running"
echo "  → Proxy:  http://127.0.0.1:1435/v1"
echo "  → Helper: http://127.0.0.1:1436"
echo "  → PIDs:   proxy=$PROXY_PID helper=$HELPER_PID"
echo ""

# Wait for either to exit
wait -n $HELPER_PID $PROXY_PID
echo "Process exited, shutting down..."
kill $HELPER_PID $PROXY_PID 2>/dev/null
