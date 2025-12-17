#!/bin/bash
# Start Webshop environment server for verl training

set -e

WEBSHOP_DIR="/Data/wyh/AgentGym-RL/AgentGym/agentenv-webshop"
PORT=36003

echo "================================"
echo "Starting Webshop Server"
echo "================================"
echo "Port: $PORT"
echo "Directory: $WEBSHOP_DIR"
echo ""

# Check if directory exists
if [ ! -d "$WEBSHOP_DIR" ]; then
    echo "Error: Webshop directory not found at $WEBSHOP_DIR"
    exit 1
fi

# Check if another process is using the port
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "Warning: Port $PORT is already in use"
    echo "Current process:"
    lsof -Pi :$PORT -sTCP:LISTEN
    echo ""
    read -p "Kill existing process and restart? (y/n): " answer
    if [ "$answer" = "y" ]; then
        echo "Killing process on port $PORT..."
        kill -9 $(lsof -t -i:$PORT) || true
        sleep 2
    else
        echo "Exiting..."
        exit 1
    fi
fi

# Change to webshop directory
cd "$WEBSHOP_DIR"

echo "Starting Webshop server..."
echo "Server will be available at: http://127.0.0.1:$PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================"
echo ""

# Start server
python -m agentenv_webshop.launch --host 0.0.0.0 --port $PORT

