#!/usr/bin/env python3
"""
Entry point for Azure App Service deployment
"""
import os
import sys

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import the Flask app from backend
from backend.app import app

if __name__ == '__main__':
    # Get port from environment variable (for production) or use 5001
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False) 