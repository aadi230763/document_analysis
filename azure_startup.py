#!/usr/bin/env python3
"""
Azure Startup Script for LLM-Powered Document Analysis System
This file serves as the entry point for Azure App Service deployment
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def main():
    """Main startup function for Azure"""
    print("🚀 Starting LLM-Powered Document Analysis System on Azure")
    print("=" * 60)
    
    # Get port from environment variable (Azure sets this)
    port = int(os.environ.get('PORT', 8000))
    
    print(f"📋 System Information:")
    print(f"   - Server will run on: http://0.0.0.0:{port}")
    print(f"   - Hackathon endpoint: POST /hackrx/run")
    print(f"   - Health check: GET /health")
    
    print("\n🔧 Starting server...")
    print("=" * 60)
    
    # Import and run the Flask app
    try:
        from app import app
        app.run(host='0.0.0.0', port=port, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 