#!/bin/bash
# Diagnostic script to verify environment variables during Render build

echo "=== Environment Variable Diagnostic ==="
echo "VITE_API_URL: ${VITE_API_URL:-NOT SET}"
echo "NODE_ENV: ${NODE_ENV:-NOT SET}"
echo ""
echo "All VITE_ prefixed variables:"
env | grep VITE_ || echo "No VITE_ variables found"
echo ""
echo "=== Starting Vite Build ==="
npm run build
