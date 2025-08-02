#!/usr/bin/env python3
"""
Environment Setup Script for LLM-Powered Document Analysis System
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Setup environment variables and check configurations"""
    print("🚀 Setting up LLM-Powered Document Analysis System")
    print("=" * 60)
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file...")
        create_env_file()
    else:
        print("✅ .env file already exists")
    
    # Check API keys
    check_api_keys()
    
    # Check dependencies
    check_dependencies()
    
    print("\n🎯 Setup Complete!")
    print("Next steps:")
    print("1. Set your API keys in the .env file")
    print("2. Run: python start_server.py")
    print("3. Test with: curl http://localhost:5001/health")

def create_env_file():
    """Create .env file with template"""
    env_content = """# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=gcp-starter

# Configuration
DEBUG_MODE=False
LOG_LEVEL=INFO
ENABLE_AUDIT_LOGGING=True

# Performance
CACHE_SIZE=300
MAX_TOKENS_PER_REQUEST=1200
RESPONSE_TIMEOUT=45
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ .env file created")

def check_api_keys():
    """Check if API keys are configured"""
    print("\n🔑 Checking API Keys:")
    
    keys_to_check = [
        ("GEMINI_API_KEY", "Gemini API"),
        ("COHERE_API_KEY", "Cohere API"),
        ("PINECONE_API_KEY", "Pinecone API")
    ]
    
    for key, name in keys_to_check:
        value = os.getenv(key)
        if value and value != f"your_{key.lower()}_here":
            print(f"✅ {name}: Configured")
        else:
            print(f"⚠️  {name}: Not configured (will use fallback)")

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking Dependencies:")
    
    required_packages = [
        ('flask', 'Flask'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('pinecone_client', 'Pinecone Client'),
        ('cohere', 'Cohere'),
        ('requests', 'Requests'),
        ('PyPDF2', 'PyPDF2'),
        ('docx', 'Python DOCX'),
        ('numpy', 'NumPy')
    ]
    
    missing_packages = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name}: Missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")

if __name__ == "__main__":
    setup_environment() 