#!/bin/bash

# Audio Sculpture Production Startup Script

echo "🎵 Starting Audio Sculpture Generator in Production Mode"
echo "=================================================="

# Create necessary directories
mkdir -p uploads outputs logs

# Set environment variables
export FLASK_ENV=production
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Kill any existing processes
pkill -f "gunicorn.*server:app" || true
pkill -f "python.*server.py" || true

# Wait a moment for processes to stop
sleep 2

# Start the server with Gunicorn
echo "Starting Gunicorn server..."
nohup gunicorn --config gunicorn.conf.py server:app > logs/server.log 2>&1 &

# Get the process ID
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"
echo "Server PID: $SERVER_PID" > /tmp/audio_sculpture.pid

# Wait for server to start
echo "Waiting for server to start..."
sleep 5

# Test if server is running
if curl -f http://localhost:8080 > /dev/null 2>&1; then
    echo "✅ Server is running successfully!"
    echo "🌐 Server URL: http://localhost:8080"
    echo "📝 Logs: tail -f logs/server.log"
    echo "🛑 Stop server: kill $SERVER_PID"
else
    echo "❌ Server failed to start. Check logs/server.log for details."
    exit 1
fi

echo "=================================================="
