#!/usr/bin/env python3
"""
Startup script for the Document Q&A API with NLTK setup
"""

import sys
import os
import subprocess

def setup_nltk():
    """Setup NLTK data if not already available"""
    try:
        from nltk.tokenize import sent_tokenize
        test_text = "This is a test sentence."
        sent_tokenize(test_text)
        print("✓ NLTK data is already available")
        return True
    except Exception:
        print("NLTK data not found. Downloading...")
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            print("✓ NLTK data downloaded successfully")
            return True
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")
            print("The system will use fallback tokenization methods.")
            return False

def main():
    """Main startup function"""
    print("Document Q&A API Startup")
    print("=" * 30)
    
    # Setup NLTK
    setup_nltk()
    
    # Change to backend directory
    os.chdir('backend')
    
    # Start the Flask app
    print("Starting Flask server...")
    try:
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 