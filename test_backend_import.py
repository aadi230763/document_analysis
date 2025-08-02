#!/usr/bin/env python3
"""
Test script to verify backend app.py can be imported without errors
"""

import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

def test_import():
    """Test importing the backend app"""
    try:
        print("🧪 Testing backend app import...")
        from app import app
        print("✅ Successfully imported backend app")
        
        # Test basic Flask app properties
        print(f"✅ Flask app name: {app.name}")
        print(f"✅ Flask app debug: {app.debug}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to import backend app: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_import()
    sys.exit(0 if success else 1) 