#!/usr/bin/env python3
"""
Startup script for Azure App Service
This file is used by Azure to start the Flask application
"""

import os
import sys
from pathlib import Path

def main():
    print("🚀 Starting LLM-Powered Document Analysis System on Azure")
    print("=" * 60)
    
    # Add backend to Python path
    backend_path = Path("/home/site/wwwroot/backend")
    if backend_path.exists():
        sys.path.insert(0, str(backend_path))
        print(f"✅ Added {backend_path} to Python path")
    else:
        print(f"❌ Backend path not found: {backend_path}")
        return 1
    
    # Get port from environment variable
    port = int(os.environ.get('PORT', 8000))
    
    print("📋 System Information:")
    print(f"   - Server will run on: http://0.0.0.0:{port}")
    print(f"   - Hackathon endpoint: POST /hackrx/run")
    print(f"   - Health check: GET /health")
    print(f"   - Python path: {sys.path[:3]}...")
    
    print("\n🔧 Starting server...")
    print("=" * 60)
    
    try:
        # Change to backend directory
        os.chdir(str(backend_path))
        print(f"✅ Changed to directory: {os.getcwd()}")
        
        # Import and run the Flask app
        from app import app
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
