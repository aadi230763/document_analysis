#!/usr/bin/env python3
"""
Simple test script to verify Python startup works
"""

import os
import sys
from datetime import datetime

def main():
    print("🧪 SIMPLE TEST: Python startup script is working!")
    print("=" * 50)
    print(f"Timestamp: {datetime.now()}")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Environment variables:")
    print(f"  PORT: {os.environ.get('PORT', 'Not set')}")
    print(f"  PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    # List files in current directory
    print(f"\nFiles in current directory:")
    try:
        for file in os.listdir('.'):
            print(f"  - {file}")
    except Exception as e:
        print(f"  Error listing files: {e}")
    
    # Test if backend directory exists
    backend_path = "/home/site/wwwroot/backend"
    if os.path.exists(backend_path):
        print(f"\n✅ Backend directory exists: {backend_path}")
        try:
            backend_files = os.listdir(backend_path)
            print(f"Backend files: {backend_files[:5]}...")  # Show first 5 files
        except Exception as e:
            print(f"Error listing backend files: {e}")
    else:
        print(f"\n❌ Backend directory not found: {backend_path}")
    
    print("\n🧪 TEST COMPLETED - Keeping alive for 30 seconds...")
    
    # Keep the script running so we can see the output
    import time
    time.sleep(30)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 