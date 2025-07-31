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
