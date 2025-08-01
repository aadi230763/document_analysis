#!/usr/bin/env python3
"""
Setup script to download NLTK data for the document Q&A API
"""

import nltk
import sys
import os

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    
    try:
        # Download punkt tokenizer
        nltk.download('punkt', quiet=False)
        print("✓ punkt tokenizer downloaded successfully")
        
        # Download other useful NLTK data
        nltk.download('stopwords', quiet=False)
        print("✓ stopwords downloaded successfully")
        
        # Download averaged perceptron tagger for better tokenization
        nltk.download('averaged_perceptron_tagger', quiet=False)
        print("✓ averaged_perceptron_tagger downloaded successfully")
        
        print("All NLTK data downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        return False

def check_nltk_data():
    """Check if NLTK data is available"""
    try:
        from nltk.tokenize import sent_tokenize
        test_text = "This is a test sentence. This is another sentence."
        sentences = sent_tokenize(test_text)
        if len(sentences) == 2:
            print("✓ NLTK data is working correctly")
            return True
        else:
            print("✗ NLTK data is not working correctly")
            return False
    except Exception as e:
        print(f"✗ NLTK data check failed: {e}")
        return False

if __name__ == "__main__":
    print("NLTK Setup for Document Q&A API")
    print("=" * 40)
    
    # Check if NLTK data is already available
    if check_nltk_data():
        print("NLTK data is already available. No setup needed.")
        sys.exit(0)
    
    # Download NLTK data
    if download_nltk_data():
        # Verify the download
        if check_nltk_data():
            print("Setup completed successfully!")
            sys.exit(0)
        else:
            print("Setup completed but verification failed.")
            sys.exit(1)
    else:
        print("Setup failed.")
        sys.exit(1) 