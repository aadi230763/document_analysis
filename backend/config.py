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

# Processing Config - Enhanced for better accuracy
CHUNK_SIZE = 1200  # Increased from 512 for better context
CHUNK_OVERLAP = 128
MIN_CHUNK_LENGTH = 50

# Retrieval Config - Optimized for accuracy and latency
BASE_RESULTS = 15  # Increased for better coverage
EXPANDED_RESULTS = 3
SIMILARITY_THRESHOLD = 0.3

# Performance Config
CACHE_SIZE = 200  # Increased cache size
CACHE_TTL = 3600  # Cache TTL in seconds
MAX_TOKENS_PER_REQUEST = 1000  # Token limit for LLM
RESPONSE_TIMEOUT = 30  # Timeout in seconds

# Accuracy Config
MIN_CONFIDENCE_THRESHOLD = 0.3
MEDICAL_RELEVANCE_THRESHOLD = 0.1
ENHANCED_SCORING = True

# Deployment Config
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENABLE_AUDIT_LOGGING = os.getenv("ENABLE_AUDIT_LOGGING", "True").lower() == "true"

# API Config
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
MAX_REQUESTS_PER_MINUTE = 100
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "False").lower() == "true"

# Medical Terms Config
ENABLE_DYNAMIC_SYNONYMS = True
SYNONYM_CACHE_SIZE = 500
MEDICAL_ENTITY_EXTRACTION = True

# Rule Engine Config
ENABLE_CONFIDENCE_SCORING = True
ENABLE_MEDICAL_RELEVANCE = True
PATTERN_MATCHING_STRICT = False  # Set to True for stricter matching
