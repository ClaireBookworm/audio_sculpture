#!/bin/bash

# Audio Sculpture Production Stop Script

echo "🛑 Stopping Audio Sculpture Generator"
echo "====================================="

# Read PID from file if it exists
if [ -f /tmp/audio_sculpture.pid ]; then
    PID=$(cat /tmp/audio_sculpture.pid)
    echo "Stopping server with PID: $PID"
    kill $PID 2>/dev/null || echo "Process $PID not found"
    rm -f /tmp/audio_sculpture.pid
else
    echo "No PID file found, killing all gunicorn processes..."
fi

# Kill any remaining processes
pkill -f "gunicorn.*server:app" || echo "No gunicorn processes found"
pkill -f "python.*server.py" || echo "No python server processes found"

echo "✅ Server stopped"
echo "====================================="
