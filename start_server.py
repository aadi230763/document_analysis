#!/usr/bin/env python3
"""
Startup script for the LLM-Powered Insurance Document Query System
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        ('flask', 'Flask'),
        ('chromadb', 'chromadb'),
        ('sentence_transformers', 'sentence_transformers'),
        ('requests', 'requests'),
        ('cohere', 'cohere'),
        ('PyPDF2', 'PyPDF2'),
        ('docx', 'python-docx')  # python-docx imports as 'docx'
    ]
    
    missing_packages = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r backend/requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_api_keys():
    """Check if API keys are configured"""
    gemini_key = os.getenv("GEMINI_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")
    
    if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY":
        print("⚠️  GEMINI_API_KEY not configured (will use fallback)")
    
    if not cohere_key:
        print("⚠️  COHERE_API_KEY not configured (will use fallback)")
    
    return True

def main():
    """Main startup function"""
    print("🚀 Starting LLM-Powered Insurance Document Query System")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check API keys
    check_api_keys()
    
    # Get port from environment variable (for production) or use 5001
    port = int(os.environ.get('PORT', 5001))
    
    print("\n📋 System Information:")
    print(f"   - Server will run on: http://localhost:{port}")
    print(f"   - Hackathon endpoint: POST /hackrx/run")
    print(f"   - Health check: GET /health")
    
    print("\n🔧 Starting server...")
    print("   Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Import and run the Flask app
    try:
        import sys
        sys.path.insert(0, str(backend_path))
        from app import app
        app.run(host='0.0.0.0', port=port, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 