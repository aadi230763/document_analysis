# Insurance Policy Analysis API

## Project Overview
This API provides intelligent analysis of insurance policy documents using advanced NLP and vector search capabilities. It can answer questions about policy coverage, exclusions, waiting periods, and other policy details by analyzing uploaded insurance documents.

## Features
- **Semantic Search**: Advanced document retrieval using BGE embedding model
- **Multi-Model LLM**: Uses Gemini API with Cohere fallback for robust responses
- **Batch Processing**: Handle multiple queries efficiently
- **Caching**: Performance optimization with intelligent caching
- **PII Protection**: Automatic redaction of sensitive information
- **Audit Logging**: Complete audit trail for compliance

## API Endpoints

### Core Endpoints
1. **POST** `/query` - Single question analysis
2. **POST** `/batch_query` - Multiple questions processing
3. **POST** `/optimized_query` - Enhanced endpoint with all optimizations
4. **POST** `/feedback` - User feedback submission
5. **GET** `/health` - Health check

## Quick Start

### Prerequisites
- Python 3.8+
- Insurance policy documents

### Installation
```bash
cd backend
pip install -r requirements.txt
```

### NLTK Setup (Required)
The system requires NLTK data for sentence tokenization. Run one of these commands:

**Option 1: Automatic setup (recommended)**
```bash
python start_server.py
```

**Option 2: Manual NLTK setup**
```bash
python backend/setup_nltk.py
```

**Option 3: Direct NLTK download**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

### Running the API
```bash
# Option 1: Use the startup script (recommended)
python start_server.py

# Option 2: Run directly
cd backend
python app.py
```

### Testing
```bash
# Test NLTK fix
python test_nltk_fix.py

# Health check
curl http://localhost:5001/health

# Single query
curl -X POST http://localhost:5001/optimized_query \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/document.pdf", "questions": "Is cataract surgery covered?"}'

# Batch query
curl -X POST http://localhost:5001/optimized_query \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/document.pdf", "questions": ["Is cataract covered?", "What is maternity waiting period?"]}'
```

## Example Usage

### Question: "Is cataract surgery covered under my policy?"
**Response:**
```json
{
  "answers": ["The policy covers cataract surgery up to $8,000 with a 20% co-payment."],
  "response_time_ms": 2340,
  "document_url": "https://example.com/document.pdf",
  "questions_processed": 1,
  "chunks_used": 45
}
```

### Batch Questions
**Request:**
```json
{
  "url": "https://example.com/policy.pdf",
  "questions": [
    "Is cataract surgery covered?",
    "What is the maternity waiting period?",
    "Are pre-existing diseases covered?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "Cataract surgery is covered up to $8,000.",
    "Maternity benefits have a 9-month waiting period.",
    "Pre-existing diseases are covered after 2 years."
  ],
  "response_time_ms": 3450,
  "document_url": "https://example.com/policy.pdf",
  "questions_processed": 3,
  "chunks_used": 45
}
```

## Technical Architecture

### Components
- **Flask Backend**: RESTful API server
- **ChromaDB**: Vector database for document storage
- **BGE-Small-EN**: Embedding model for semantic search
- **Gemini API**: Primary LLM for answer generation
- **Cohere API**: Fallback LLM for reliability
- **Hybrid Retrieval**: Dense + Sparse (BM25/TF-IDF) search

### Data Flow
1. User submits question with document URL
2. Document is parsed and cached (if not already cached)
3. Hybrid retrieval finds relevant chunks
4. LLM generates answer from retrieved evidence
5. Response returned with performance metrics

## Performance
- Average response time: 2-5 seconds
- Supports 100+ concurrent requests
- Intelligent caching reduces latency
- Handles 33,000+ document chunks
- Cache hit rate: 75-90%

## Security & Compliance
- PII automatic redaction
- Audit logging for all queries
- No sensitive data in responses
- Rate limiting protection

## Troubleshooting

### Common Issues

#### 1. NLTK Error: "Resource punkt_tab not found"
**Error Message:**
```
Resource punkt_tab not found.
Please use the NLTK Downloader to obtain the resource
```

**Solution:**
```bash
# Run the automatic setup
python start_server.py

# Or manually download NLTK data
python backend/setup_nltk.py

# Or use Python directly
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

#### 2. Document Parsing Errors
**Error:** "Document could not be parsed as PDF, DOCX, EML, or plain text"

**Solutions:**
- Ensure the document URL is accessible
- Check if the document format is supported (PDF, DOCX, EML, TXT)
- Verify the document is not corrupted
- Try a different document to test

#### 3. Slow Response Times
**Solutions:**
- Check cache hit rate: `curl http://localhost:5001/performance`
- Ensure models are preloaded (check startup logs)
- Verify API keys are configured correctly
- Check network connectivity to LLM APIs

#### 4. Memory Issues
**Solutions:**
- Reduce cache sizes in `config.py`
- Restart the application
- Check available system memory
- Consider using a smaller embedding model

### Debug Mode
Enable debug logging for detailed error information:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Documentation
See `OPTIMIZATION_GUIDE.md` for detailed optimization information and `API_DOCUMENTATION.md` for detailed API specifications.

## License
This project is developed for insurance policy analysis and evaluation purposes. 