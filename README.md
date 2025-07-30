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
3. **POST** `/feedback` - User feedback submission
4. **GET** `/health` - Health check

## Quick Start

### Prerequisites
- Python 3.8+
- Insurance policy documents

### Installation
```bash
cd backend
pip install -r requirements.txt
```

### Running the API
```bash
python app.py
```

### Testing
```bash
# Health check
curl http://localhost:5001/health

# Single query
curl -X POST http://localhost:5001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Is cataract surgery covered?"}'

# Batch query
curl -X POST http://localhost:5001/batch_query \
  -H "Content-Type: application/json" \
  -d '{"questions": ["Is cataract covered?", "What is maternity waiting period?"]}'
```

## Example Usage

### Question: "Is cataract surgery covered under my policy?"
**Response:**
```json
{
  "answer": "The policy covers cataract surgery up to $8,000 with a 20% co-payment."
}
```

### Batch Questions
**Request:**
```json
{
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
  "results": [
    {"answer": "Cataract surgery is covered up to $8,000."},
    {"answer": "Maternity benefits have a 9-month waiting period."},
    {"answer": "Pre-existing diseases are covered after 2 years."}
  ]
}
```

## Technical Architecture

### Components
- **Flask Backend**: RESTful API server
- **ChromaDB**: Vector database for document storage
- **BGE-Small-EN**: Embedding model for semantic search
- **Gemini API**: Primary LLM for answer generation
- **Cohere API**: Fallback LLM for reliability

### Data Flow
1. User submits question
2. Query expansion for better retrieval
3. Semantic search in policy documents
4. LLM generates answer from relevant clauses
5. Response returned with evidence

## Performance
- Average response time: 1-3 seconds
- Supports 100+ concurrent requests
- Intelligent caching reduces latency
- Handles 33,000+ document chunks

## Security & Compliance
- PII automatic redaction
- Audit logging for all queries
- No sensitive data in responses
- Rate limiting protection

## Documentation
See `API_DOCUMENTATION.md` for detailed API specifications and examples.

## License
This project is developed for insurance policy analysis and evaluation purposes. 