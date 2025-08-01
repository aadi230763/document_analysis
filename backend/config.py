import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Model Config
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Use HuggingFace model that will be downloaded
LLM_MODEL = BASE_DIR / "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Gemini API Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent")

# ChromaDB Config
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "insurance_policies_enhanced"

# Processing Config - Optimized for accuracy and performance
CHUNK_SIZE = 800  # Optimized for better context preservation
CHUNK_OVERLAP_RATIO = 0.3  # 30% overlap for better context preservation
CHUNK_OVERLAP = int(CHUNK_SIZE * CHUNK_OVERLAP_RATIO)  # 240 characters overlap
MIN_CHUNK_LENGTH = 30  # Lowered for better coverage of short but important clauses

# Retrieval Config - Enhanced for accuracy
BASE_RESULTS = 20  # Increased for better coverage
EXPANDED_RESULTS = 5  # Increased for better expansion
SIMILARITY_THRESHOLD = 0.25  # Lowered for more inclusive matching

# Performance Config - Optimized for hackathon
CACHE_SIZE = 300  # Increased cache size for better performance
CACHE_TTL = 7200  # Increased cache TTL to 2 hours
MAX_TOKENS_PER_REQUEST = 1200  # Increased token limit for better answers
RESPONSE_TIMEOUT = 45  # Increased timeout for complex documents

# Accuracy Config - Enhanced for better results
MIN_CONFIDENCE_THRESHOLD = 0.2  # Lowered for more inclusive matching
MEDICAL_RELEVANCE_THRESHOLD = 0.05  # Lowered for better medical term matching
ENHANCED_SCORING = True

# Deployment Config
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_AUDIT_LOGGING = os.getenv("ENABLE_AUDIT_LOGGING", "True").lower() == "true"

# API Config
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
MAX_REQUESTS_PER_MINUTE = 150  # Increased for hackathon usage
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "False").lower() == "true"

# Medical Terms Config
ENABLE_DYNAMIC_SYNONYMS = True
SYNONYM_CACHE_SIZE = 800  # Increased for better synonym coverage
MEDICAL_ENTITY_EXTRACTION = True

# Rule Engine Config
ENABLE_CONFIDENCE_SCORING = True
ENABLE_MEDICAL_RELEVANCE = True
PATTERN_MATCHING_STRICT = False  # Keep flexible for better matching

# Document Processing Config - New optimizations
MAX_CHUNKS_PER_DOCUMENT = 30  # Increased for better coverage
MIN_CHUNK_CHARACTERS = 20  # Minimum characters for valid chunks
MAX_CHUNK_CHARACTERS = 1000  # Maximum characters per chunk
ENABLE_SECTION_AWARE_CHUNKING = True  # Enable section-based chunking
ENABLE_KEYWORD_ENHANCED_RETRIEVAL = True  # Enable keyword-based enhancement

# LLM Generation Config
DEFAULT_MAX_TOKENS = 200  # Default token limit for responses
DEFAULT_TEMPERATURE = 0.1  # Default temperature for generation
GEMINI_TIMEOUT = 10  # Timeout for Gemini API calls
DOCUMENT_DOWNLOAD_TIMEOUT = 30  # Timeout for document downloads

# Chunk Processing Config
DEFAULT_CHUNK_SIZE = 800  # Default chunk size
DEFAULT_MAX_CHUNKS = 20  # Default max chunks for processing
DEFAULT_TOP_CHUNKS = 15  # Default top chunks for retrieval
DEFAULT_FINAL_CHUNKS = 10  # Default final chunks for grace period questions
DEFAULT_FINAL_CHUNKS_REGULAR = 8  # Default final chunks for regular questions

# Answer Processing Config
MAX_ANSWER_SENTENCES = 3  # Maximum sentences in answer
MAX_ANSWER_CHARACTERS = 400  # Maximum characters in answer
CHUNK_TEXT_LIMIT = 600  # Character limit for chunk text in evidence

# Scoring Config
GRACE_PERIOD_EXACT_SCORE = 15  # Score for exact grace period match
GRACE_PERIOD_THIRTY_DAYS_SCORE = 12  # Score for thirty days mention
GRACE_PERIOD_PAYMENT_SCORE = 10  # Score for grace + payment
GRACE_PERIOD_PREMIUM_SCORE = 10  # Score for grace + premium
GRACE_PERIOD_RENEWAL_SCORE = 8  # Score for grace + renewal
RENEWAL_SCORE = 8  # Score for renewal mention
CONTINUOUS_COVERAGE_SCORE = 8  # Score for continuous coverage
CONTINUITY_SCORE = 6  # Score for continuity mention
POLICY_RENEWAL_SCORE = 10  # Score for policy renewal
SECTION_RELEVANCE_SCORE = 5  # Score for relevant sections
LENGTH_BONUS_SCORE = 3  # Score for substantial chunks

# New: Document Caching Config
ENABLE_DOCUMENT_CACHING = True
DOCUMENT_CACHE_SIZE = 100  # Number of documents to cache
EMBEDDING_CACHE_SIZE = 200  # Number of embedding sets to cache
CHUNK_CACHE_SIZE = 200  # Number of chunk sets to cache

# New: Hybrid Retrieval Config
ENABLE_HYBRID_RETRIEVAL = True
DENSE_WEIGHT = 0.7  # Weight for dense retrieval (FAISS)
SPARSE_WEIGHT = 0.3  # Weight for sparse retrieval (BM25/TF-IDF)
BM25_K1 = 1.2  # BM25 parameter k1
BM25_B = 0.75  # BM25 parameter b

# New: Model Preloading Config
PRELOAD_MODELS_AT_STARTUP = True
EMBEDDING_MODEL_WARMUP = True
LLM_MODEL_WARMUP = True

# New: Response Time Optimization
MAX_RESPONSE_TIME_MS = 10000  # 10 seconds max response time
ENABLE_RESPONSE_TIME_TRACKING = True
OPTIMIZE_FOR_SPEED = True  # Trade some accuracy for speed if needed
