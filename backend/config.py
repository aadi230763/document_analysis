import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Model Config
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Use HuggingFace model that will be downloaded
LLM_MODEL = BASE_DIR / "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Gemini API Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent")

# Pinecone Config - NEW
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "YOUR_PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "insurance-policies")
PINECONE_DIMENSION = 384  # BGE-small-en dimension
PINECONE_METRIC = "cosine"

# ChromaDB Config (Fallback)
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "insurance_policies_enhanced"

# Processing Config - Optimized for accuracy and performance
CHUNK_SIZE = 800  # Optimized for better context preservation
CHUNK_OVERLAP = 100  # Reduced overlap for efficiency
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
DEFAULT_MAX_TOKENS = 300  # Increased for more detailed responses
DEFAULT_TEMPERATURE = 0.1  # Balanced temperature for comprehensive answers
GEMINI_TIMEOUT = 10  # Balanced timeout for detailed processing
DOCUMENT_DOWNLOAD_TIMEOUT = 25  # Balanced timeout for document downloads

# Chunk Processing Config
DEFAULT_CHUNK_SIZE = 700  # Increased for better context
DEFAULT_MAX_CHUNKS = 18  # Balanced for coverage and speed
DEFAULT_TOP_CHUNKS = 12  # Balanced for retrieval
DEFAULT_FINAL_CHUNKS = 8  # Balanced for grace period questions
DEFAULT_FINAL_CHUNKS_REGULAR = 6  # Balanced for regular questions

# Answer Processing Config
MAX_ANSWER_SENTENCES = 4  # Increased for more detailed answers
MAX_ANSWER_CHARACTERS = 800  # Increased for comprehensive responses
CHUNK_TEXT_LIMIT = 500  # Increased character limit for evidence

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

# Pinecone Optimization Config - NEW
PINECONE_TOP_K = 50  # Retrieve more candidates for better accuracy
PINECONE_FILTER_THRESHOLD = 0.3  # Minimum similarity threshold
PINECONE_NAMESPACE_PREFIX = "policy_"  # Namespace prefix for organization
ENABLE_PINECONE_HYBRID_SEARCH = True  # Enable hybrid search
PINECONE_SPARSE_WEIGHT = 0.3  # Weight for sparse search
PINECONE_DENSE_WEIGHT = 0.7  # Weight for dense search
