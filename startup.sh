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

# Check if we can import the app
echo "🔍 Testing imports..."
python -c "
try:
    import app
    print('✅ App imports successfully')
except ImportError as e:
    print(f'⚠️  Import warning: {e}')
    print('🔄 Continuing with startup...')
except Exception as e:
    print(f'❌ Critical error: {e}')
    exit(1)
"

# Start the Flask application
echo "🚀 Starting Flask application..."
python app.py 