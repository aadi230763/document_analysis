#!/bin/bash
echo "🚀 Starting LLM-Powered Document Analysis System on Azure"
echo "============================================================"

# Add backend to Python path
export PYTHONPATH="/home/site/wwwroot/backend:$PYTHONPATH"

# Get port from environment variable
PORT=${PORT:-8000}

echo "📋 System Information:"
echo "   - Server will run on: http://0.0.0.0:$PORT"
echo "   - Hackathon endpoint: POST /hackrx/run"
echo "   - Health check: GET /health"
echo "   - Python path: $PYTHONPATH"

echo ""
echo "🔧 Starting server..."
echo "============================================================"

# Change to backend directory
cd /home/site/wwwroot/backend

# Start the Flask application
python app.py 