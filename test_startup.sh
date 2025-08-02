#!/bin/bash

echo "🚀 TEST: Starting LLM-Powered Document Analysis System on Azure"
echo "============================================================"

# Test if we can access the backend directory
if [ -d "/home/site/wwwroot/backend" ]; then
    echo "✅ Backend directory exists"
else
    echo "❌ Backend directory not found"
    echo "Current directory: $(pwd)"
    echo "Contents: $(ls -la)"
    exit 1
fi

# Test if we can access the app.py file
if [ -f "/home/site/wwwroot/backend/app.py" ]; then
    echo "✅ app.py file exists"
else
    echo "❌ app.py file not found"
    echo "Backend contents: $(ls -la /home/site/wwwroot/backend/)"
    exit 1
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/home/site/wwwroot/backend"
echo "✅ PYTHONPATH set to: ${PYTHONPATH}"

# Get port
PORT=${PORT:-8000}
echo "✅ PORT set to: ${PORT}"

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
echo "✅ Changed to backend directory: $(pwd)"

# Test Python import
python -c "import sys; print('✅ Python path:', sys.path)"
python -c "import app; print('✅ Flask app imported successfully')"

# Start the server
python app.py 