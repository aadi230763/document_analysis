#!/usr/bin/env python3
"""
Test script to check if all imports work correctly
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    
    try:
        import os
        print("✅ os")
    except Exception as e:
        print(f"❌ os: {e}")
    
    try:
        from dotenv import load_dotenv
        print("✅ dotenv")
    except Exception as e:
        print(f"❌ dotenv: {e}")
    
    try:
        from flask import Flask, request, jsonify
        print("✅ flask")
    except Exception as e:
        print(f"❌ flask: {e}")
    
    try:
        from chromadb.utils import embedding_functions
        print("✅ chromadb.utils")
    except Exception as e:
        print(f"❌ chromadb.utils: {e}")
    
    try:
        import requests
        print("✅ requests")
    except Exception as e:
        print(f"❌ requests: {e}")
    
    try:
        import cohere
        print("✅ cohere")
    except Exception as e:
        print(f"❌ cohere: {e}")
    
    try:
        from flask_cors import CORS
        print("✅ flask-cors")
    except Exception as e:
        print(f"❌ flask-cors: {e}")
    
    try:
        from PyPDF2 import PdfReader
        print("✅ PyPDF2")
    except Exception as e:
        print(f"❌ PyPDF2: {e}")
    
    try:
        import docx
        print("✅ python-docx")
    except Exception as e:
        print(f"❌ python-docx: {e}")
    
    try:
        import numpy as np
        print("✅ numpy")
    except Exception as e:
        print(f"❌ numpy: {e}")
    
    try:
        import pdfplumber
        print("✅ pdfplumber")
    except Exception as e:
        print(f"❌ pdfplumber: {e}")
    
    try:
        import config
        print("✅ config")
    except Exception as e:
        print(f"❌ config: {e}")
    
    try:
        from utils.medical_terms import get_dynamic_synonyms, parse_demographics, extract_policy_duration
        print("✅ utils.medical_terms")
    except Exception as e:
        print(f"❌ utils.medical_terms: {e}")
    
    try:
        from utils.query_expander import QueryExpander
        print("✅ utils.query_expander")
    except Exception as e:
        print(f"❌ utils.query_expander: {e}")
    
    try:
        from utils.rule_engine import RuleEngine
        print("✅ utils.rule_engine")
    except Exception as e:
        print(f"❌ utils.rule_engine: {e}")

def test_app_creation():
    """Test Flask app creation"""
    print("\nTesting Flask app creation...")
    
    try:
        from app import app
        print("✅ Flask app created successfully")
        return True
    except Exception as e:
        print(f"❌ Flask app creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_imports()
    test_app_creation() 