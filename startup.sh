#!/bin/bash

echo "🚀 Starting LLM-Powered Document Analysis System on Azure"
echo "============================================================"

# Add backend to Python path
export PYTHONPATH="${PYTHONPATH}:/home/site/wwwroot/backend"

# Get port from environment variable (Azure sets this)
PORT=${PORT:-8000}

echo "📋 System Information:"
echo "   - Server will run on: http://0.0.0.0:${PORT}"
echo "   - Hackathon endpoint: POST /hackrx/run"
echo "   - Health check: GET /health"
echo "   - Python path: ${PYTHONPATH}"

echo ""
echo "🔧 Starting server..."
echo "============================================================"

# Change to the backend directory and start the Flask app
cd /home/site/wwwroot/backend

# Set environment variables for better error handling
export FLASK_ENV=production
export FLASK_DEBUG=0

# Start the Flask app with error handling
python -u app.py 