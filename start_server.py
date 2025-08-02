#!/usr/bin/env python3
"""
Startup script for the LLM-Powered Insurance Document Query System
"""

import os
import sys
from pathlib import Path

def main():
    """Main startup function"""
    print("🚀 Starting LLM-Powered Insurance Document Query System")
    print("=" * 60)
    
    # Add backend to path
    backend_path = Path(__file__).parent / "backend"
    if backend_path.exists():
        sys.path.insert(0, str(backend_path))
        print(f"✅ Added {backend_path} to Python path")
    else:
        print(f"❌ Backend path not found: {backend_path}")
        return 1
    
    # Get port from environment variable (Azure sets this)
    port = int(os.environ.get('PORT', 8000))
    
    print("\n📋 System Information:")
    print(f"   - Server will run on: http://0.0.0.0:{port}")
    print(f"   - Hackathon endpoint: POST /hackrx/run")
    print(f"   - Health check: GET /health")
    
    print("\n🔧 Starting server...")
    print("=" * 60)
    
    # Import and run the Flask app
    try:
        from app import app
        print("✅ Successfully imported Flask app")
        app.run(host='0.0.0.0', port=port, debug=False)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This might be due to missing dependencies or incorrect path")
        return 1
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        sys.exit(exit_code) 
