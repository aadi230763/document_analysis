#!/bin/bash

echo "🔍 DEBUG: Startup script is being executed!"
echo "=========================================="
echo "Timestamp: $(date)"
echo "Current directory: $(pwd)"
echo "Current user: $(whoami)"
echo "Environment variables:"
echo "  PORT: $PORT"
echo "  PYTHONPATH: $PYTHONPATH"
echo "=========================================="

# Test if we can write to stdout
echo "✅ Can write to stdout"

# Test if we can access the backend directory
if [ -d "/home/site/wwwroot/backend" ]; then
    echo "✅ Backend directory exists"
    ls -la /home/site/wwwroot/backend/
else
    echo "❌ Backend directory not found"
    echo "Contents of /home/site/wwwroot/:"
    ls -la /home/site/wwwroot/
fi

# Test if we can access the startup.sh file
if [ -f "/home/site/wwwroot/startup.sh" ]; then
    echo "✅ startup.sh file exists"
    echo "Permissions: $(ls -la /home/site/wwwroot/startup.sh)"
else
    echo "❌ startup.sh file not found"
fi

# Test Python
echo "Testing Python..."
python --version
python -c "print('✅ Python is working')"

echo "🔍 DEBUG: Script completed"
echo "=========================================="

# Keep the script running for a few seconds so we can see the output
sleep 5 