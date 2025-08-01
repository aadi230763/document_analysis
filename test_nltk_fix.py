#!/usr/bin/env python3
"""
Test script to verify NLTK fix for document processing
"""

import sys
import os

# Add backend to path
sys.path.insert(0, 'backend')

def test_nltk_tokenization():
    """Test NLTK tokenization"""
    print("Testing NLTK tokenization...")
    
    try:
        from app import safe_sent_tokenize, fallback_sent_tokenize
        
        test_text = "This is the first sentence. This is the second sentence. And this is the third sentence."
        
        # Test safe tokenization
        sentences = safe_sent_tokenize(test_text)
        print(f"✓ Safe tokenization: {len(sentences)} sentences")
        for i, sent in enumerate(sentences):
            print(f"  {i+1}: {sent}")
        
        # Test fallback tokenization
        fallback_sentences = fallback_sent_tokenize(test_text)
        print(f"✓ Fallback tokenization: {len(fallback_sentences)} sentences")
        
        return True
        
    except Exception as e:
        print(f"✗ Tokenization test failed: {e}")
        return False

def test_document_parsing():
    """Test document parsing with sample text"""
    print("\nTesting document parsing...")
    
    try:
        from app import smart_chunk_text
        
        sample_text = """
        Section 1: Introduction
        This is a sample insurance policy document. It contains multiple sections with important information.
        
        Section 2: Coverage Details
        The policy covers various medical procedures including surgery, hospitalization, and outpatient care.
        There is a 30-day grace period for premium payments.
        
        Section 3: Exclusions
        Pre-existing conditions are not covered for the first 12 months.
        Cosmetic procedures are excluded from coverage.
        """
        
        chunks = smart_chunk_text(sample_text, section_name="Test Document", chunk_type="test")
        print(f"✓ Document parsing: {len(chunks)} chunks created")
        
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"  Chunk {i+1}: {chunk['text'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Document parsing test failed: {e}")
        return False

def main():
    """Main test function"""
    print("NLTK Fix Verification Test")
    print("=" * 40)
    
    # Test tokenization
    if not test_nltk_tokenization():
        print("❌ Tokenization test failed")
        return False
    
    # Test document parsing
    if not test_document_parsing():
        print("❌ Document parsing test failed")
        return False
    
    print("\n✅ All tests passed! NLTK fix is working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 