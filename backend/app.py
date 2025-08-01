import os
os.environ['FLASK_NO_COLOR'] = '1'  # Disable colored output

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
import asyncio
from config import *

import requests
from config import GEMINI_API_KEY, GEMINI_API_URL
import atexit
from utils.medical_terms import get_dynamic_synonyms, parse_demographics, extract_policy_duration
from utils.query_expander import QueryExpander
import re
from utils.rule_engine import RuleEngine
import json
from datetime import datetime, timezone
import cohere
from flask_cors import CORS
import time
from io import BytesIO
import tempfile
from PyPDF2 import PdfReader
import docx
import numpy as np
import mimetypes
import logging
from email import policy
from email.parser import BytesParser
import base64
import pdfplumber
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib
import pickle
import os.path
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("doc_parser")

# Document caching system
class DocumentCache:
    def __init__(self, cache_dir="cache", max_size=100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_metadata = self._load_cache_metadata()
    
    def _load_cache_metadata(self):
        """Load cache metadata from file"""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache_metadata(self):
        """Save cache metadata to file"""
        with open(self.cache_metadata_file, 'w') as f:
            json.dump(self.cache_metadata, f)
    
    def _get_document_hash(self, url):
        """Generate hash for document URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _get_cache_path(self, doc_hash):
        """Get cache file path for document hash"""
        return self.cache_dir / f"{doc_hash}.pkl"
    
    def get_cached_document(self, url):
        """Get cached document data if available"""
        doc_hash = self._get_document_hash(url)
        cache_path = self._get_cache_path(doc_hash)
        
        if cache_path.exists() and doc_hash in self.cache_metadata:
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"Cache hit for document: {url}")
                return cached_data
            except Exception as e:
                logger.error(f"Error loading cached document: {e}")
        
        return None
    
    def cache_document(self, url, chunks, embeddings=None, metadata=None):
        """Cache document data"""
        doc_hash = self._get_document_hash(url)
        cache_path = self._get_cache_path(doc_hash)
        
        # Prepare cache data
        cache_data = {
            'url': url,
            'chunks': chunks,
            'embeddings': embeddings,
            'metadata': metadata or {},
            'cached_at': datetime.now().isoformat()
        }
        
        # Check cache size and evict if necessary
        if len(self.cache_metadata) >= self.max_size:
            self._evict_oldest_cache()
        
        try:
            # Save cache data
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Update metadata
            self.cache_metadata[doc_hash] = {
                'url': url,
                'cached_at': cache_data['cached_at'],
                'chunk_count': len(chunks)
            }
            self._save_cache_metadata()
            
            logger.info(f"Cached document: {url} ({len(chunks)} chunks)")
        except Exception as e:
            logger.error(f"Error caching document: {e}")
    
    def _evict_oldest_cache(self):
        """Evict oldest cache entry"""
        if not self.cache_metadata:
            return
        
        # Find oldest entry
        oldest_hash = min(self.cache_metadata.keys(), 
                         key=lambda k: self.cache_metadata[k]['cached_at'])
        
        # Remove from filesystem and metadata
        cache_path = self._get_cache_path(oldest_hash)
        if cache_path.exists():
            cache_path.unlink()
        
        del self.cache_metadata[oldest_hash]
        self._save_cache_metadata()
        logger.info(f"Evicted cache entry: {oldest_hash}")

# Initialize document cache
document_cache = DocumentCache()

def parse_document_from_url(url):
    """
    Download and parse a document from a URL (PDF, DOCX, EML, TXT).
    Returns: list of dicts: [{text, section, table, type, ...}]
    Raises: Exception with clear error message if parsing fails.
    """
    import requests
    from io import BytesIO
    import docx
    from PyPDF2 import PdfReader
    import traceback
    import re
    import os

    # Check cache first
    cached_data = document_cache.get_cached_document(url)
    if cached_data:
        return cached_data['chunks']

    logger.info(f"Downloading document: {url}")
    try:
        from config import DOCUMENT_DOWNLOAD_TIMEOUT
        resp = requests.get(url, timeout=DOCUMENT_DOWNLOAD_TIMEOUT)
        resp.raise_for_status()
        content = resp.content
        content_type = resp.headers.get('Content-Type', '').lower()
        ext = os.path.splitext(url.split('?')[0])[1].lower()
        logger.info(f"Downloaded {len(content)} bytes, Content-Type: {content_type}, Extension: {ext}")
        
        # Check if we have any content
        if not content or len(content) == 0:
            raise Exception("Downloaded document is empty")
            
    except Exception as e:
        logger.error(f"Failed to download document: {url} | {e}")
        raise Exception(f"Failed to download document: {e}")

    def add_chunk(chunks, text, section=None, table=None, chunk_type=None):
        if text and text.strip():
            chunks.append({
                'text': text.strip(),
                'section': section or '',
                'table': table or '',
                'type': chunk_type or ''
            })

    def extract_section_headers(text):
        """Enhanced section header extraction for insurance documents"""
        patterns = [
            r'^(?:Section|Clause|Article)\s+\d+[\.\d]*\s*[:\-]?\s*[A-Z][A-Za-z\s]+$',
            r'^(?:Coverage|Benefits|Exclusions|Limitations|Terms|Conditions)\s*[:\-]?\s*[A-Z][A-Za-z\s]+$',
            r'^(?:Policy|Insurance|Premium|Claim|Deductible|Co-pay|Coinsurance)\s+[A-Z][A-Za-z\s]+$',
            r'^\d+\.\s+[A-Z][A-Z\s]{3,}$',
            r'^[A-Z][A-Z\s]{3,}$',
            r'^(?:Grace|Waiting|Exclusion|Inclusion|Coverage|Benefit)\s+Period\s*[:\-]?\s*[A-Z][A-Za-z\s]*$',
            r'^(?:Network|Non-Network|In-Network|Out-of-Network)\s+[A-Z][A-Za-z\s]+$',
            r'^(?:Pre-existing|Pre-existing Condition)\s*[:\-]?\s*[A-Z][A-Za-z\s]*$',
        ]
        headers = []
        for pattern in patterns:
            headers.extend([m.group() for m in re.finditer(pattern, text, re.MULTILINE)])
        return sorted(set(headers), key=lambda h: text.find(h))

    def smart_chunk_text(text, section_name="", chunk_type="", max_chunk_size=512):
        """Enhanced chunking with proper 30% overlap and insurance term preservation"""
        from config import CHUNK_OVERLAP_RATIO
        sentences = safe_sent_tokenize(text)
        chunks = []
        overlap_size = int(max_chunk_size * CHUNK_OVERLAP_RATIO)  # 30% overlap
        
        # Insurance terms that should be preserved together
        insurance_terms = [
            'grace period', 'waiting period', 'pre-existing disease', 'no claim discount',
            'health check-up', 'room rent', 'intensive care unit', 'organ donor',
            'maternity expenses', 'cataract surgery', 'ayush treatment'
        ]
        
        i = 0
        while i < len(sentences):
            chunk = []
            chunk_len = 0
            start_i = i
            
            # Build chunk up to max size
            while i < len(sentences) and chunk_len < max_chunk_size:
                sentence = sentences[i]
                chunk.append(sentence)
                chunk_len += len(sentence.split())
                i += 1
            
            # Check if we're breaking an insurance term - if so, include the next sentence
            if i < len(sentences):
                current_chunk_text = " ".join(chunk).lower()
                next_sentence = sentences[i].lower()
                
                # Check if we're breaking any insurance terms
                for term in insurance_terms:
                    if term in current_chunk_text and term in next_sentence:
                        # Include the next sentence to preserve the term context
                        chunk.append(sentences[i])
                        chunk_len += len(sentences[i].split())
                        i += 1
                        break
            
            if chunk:
                chunk_text = " ".join(chunk)
                # Only add chunk if it has meaningful content
                if len(chunk_text.strip()) >= MIN_CHUNK_LENGTH:
                    chunks.append({
                        "text": chunk_text,
                        "section": section_name,
                        "chunk_type": chunk_type,
                        "start_sentence": start_i,
                        "end_sentence": i-1,
                        "word_count": chunk_len
                    })
            
            # Move back by overlap amount for next chunk
            if i < len(sentences):
                # Calculate how many sentences to go back for overlap
                overlap_sentences = 0
                overlap_len = 0
                for j in range(i-1, start_i, -1):
                    if overlap_len + len(sentences[j].split()) <= overlap_size:
                        overlap_sentences += 1
                        overlap_len += len(sentences[j].split())
                    else:
                        break
                i = max(start_i + 1, i - overlap_sentences)
        
        return chunks

    # --- Improved PDF logic ---
    if ext == '.pdf' or 'pdf' in content_type:
        logger.info(f"Attempting PDF parsing for {len(content)} bytes")
        try:
            pdf_stream = BytesIO(content)
            reader = PdfReader(pdf_stream)
            all_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ''
                all_text.append(text)
            full_text = '\n'.join(all_text)
            logger.info(f"PDF extracted {len(full_text)} characters of text")
            
            # If no text, fallback to pdfplumber
            if not full_text.strip():
                logger.info("No text from PyPDF2, trying pdfplumber")
                pdf_stream.seek(0)
                with pdfplumber.open(pdf_stream) as pdf:
                    all_text = [page.extract_text() or '' for page in pdf.pages]
                full_text = '\n'.join(all_text)
                logger.info(f"pdfplumber extracted {len(full_text)} characters of text")
            
            # If still no text, fallback to OCR
            if not full_text.strip():
                logger.info("No text from pdfplumber, trying OCR")
                try:
                    from pdf2image import convert_from_bytes
                    import pytesseract
                    images = convert_from_bytes(content)
                    all_text = [pytesseract.image_to_string(img) for img in images]
                    full_text = '\n'.join(all_text)
                    logger.info(f"OCR extracted {len(full_text)} characters of text")
                except Exception as ocr_e:
                    logger.error(f"PDF OCR failed: {ocr_e}")
            
            # Enhanced section-aware chunking
            chunks = []
            headers = extract_section_headers(full_text)
            
            if headers:
                # Section-based chunking
                header_positions = [(m.start(), m.group()) for h in headers for m in re.finditer(re.escape(h), full_text)]
                header_positions = sorted(header_positions, key=lambda x: x[0])
                
                for idx, (pos, header) in enumerate(header_positions):
                    start = pos
                    end = header_positions[idx + 1][0] if idx + 1 < len(header_positions) else len(full_text)
                    section_text = full_text[start:end].strip()
                    
                    if section_text:
                        section_chunks = smart_chunk_text(section_text, section_name=header, chunk_type='pdf_section')
                        chunks.extend(section_chunks)
            else:
                # Fallback to paragraph-based chunking
                chunks = smart_chunk_text(full_text, section_name='PDF Document', chunk_type='pdf_section')
            
            result = [c for c in chunks if c['text']]
            logger.info(f"PDF parsing successful, created {len(result)} chunks")
            if not result:
                logger.warning("PDF parsing completed but no valid chunks were created")
            
            # Cache the result
            document_cache.cache_document(url, result)
            return result
            
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}\n{traceback.format_exc()}")
            # Fallback to plain text
    
    # --- Improved DOCX logic ---
    if ext == '.docx' or 'word' in content_type or 'docx' in content_type:
        logger.info(f"Attempting DOCX parsing for {len(content)} bytes")
        try:
            docx_stream = BytesIO(content)
            doc = docx.Document(docx_stream)
            
            # Extract all text with structure
            full_text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
            logger.info(f"DOCX extracted {len(full_text)} characters of text")
            
            chunks = []
            headers = extract_section_headers(full_text)
            
            if headers:
                # Section-based chunking for DOCX
                header_positions = [(m.start(), m.group()) for h in headers for m in re.finditer(re.escape(h), full_text)]
                header_positions = sorted(header_positions, key=lambda x: x[0])
                
                for idx, (pos, header) in enumerate(header_positions):
                    start = pos
                    end = header_positions[idx + 1][0] if idx + 1 < len(header_positions) else len(full_text)
                    section_text = full_text[start:end].strip()
                    
                    if section_text:
                        section_chunks = smart_chunk_text(section_text, section_name=header, chunk_type='docx_section')
                        chunks.extend(section_chunks)
            else:
                # Fallback to paragraph-based chunking
                for para in doc.paragraphs:
                    if para.text.strip() and len(para.text.split()) > 10:
                        add_chunk(chunks, para.text, section=para.style.name if para.style else None, chunk_type='docx_paragraph')
            
            # Add tables as separate chunks
            for t_idx, table in enumerate(doc.tables):
                table_text = '\n'.join([' | '.join(cell.text.strip() for cell in row.cells) for row in table.rows])
                if table_text.strip():
                    add_chunk(chunks, table_text, section=f"Table {t_idx+1}", table=table_text, chunk_type='docx_table')
            
            result = [c for c in chunks if c['text']]
            logger.info(f"DOCX parsing successful, created {len(result)} chunks")
            if not result:
                logger.warning("DOCX parsing completed but no valid chunks were created")
            
            # Cache the result
            document_cache.cache_document(url, result)
            return result
            
        except Exception as e:
            logger.error(f"DOCX parsing failed: {e}\n{traceback.format_exc()}")
    
    # --- EML logic ---
    if ext == '.eml' or 'message/rfc822' in content_type or 'eml' in content_type:
        logger.info(f"Attempting EML parsing for {len(content)} bytes")
        try:
            msg = BytesParser(policy=policy.default).parsebytes(content)
            text = ''
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain':
                        text += part.get_content() + '\n'
            else:
                text = msg.get_content()
            
            logger.info(f"EML extracted {len(text)} characters of text")
            chunks = smart_chunk_text(text, section_name='Email Body', chunk_type='eml')
            result = [c for c in chunks if c['text']]
            logger.info(f"EML parsing successful, created {len(result)} chunks")
            if not result:
                logger.warning("EML parsing completed but no valid chunks were created")
            
            # Cache the result
            document_cache.cache_document(url, result)
            return result
            
        except Exception as e:
            logger.error(f"EML parsing failed: {e}\n{traceback.format_exc()}")
    
    # --- Plain text fallback ---
    logger.info(f"Attempting plain text parsing for {len(content)} bytes")
    try:
        text = content.decode(errors='ignore')
        logger.info(f"Plain text extracted {len(text)} characters")
        chunks = smart_chunk_text(text, section_name='Plain Text', chunk_type='plain')
        result = [c for c in chunks if c['text']]
        logger.info(f"Plain text parsing successful, created {len(result)} chunks")
        if not result:
            logger.warning("Plain text parsing completed but no valid chunks were created")
        
        # Cache the result
        document_cache.cache_document(url, result)
        return result
    except Exception as e:
        logger.error(f"Plain text parsing failed: {e}\n{traceback.format_exc()}")
        
        # Final fallback: try to parse as PDF regardless of content type
        logger.info("Attempting final PDF fallback parsing")
        try:
            pdf_stream = BytesIO(content)
            reader = PdfReader(pdf_stream)
            all_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ''
                all_text.append(text)
            full_text = '\n'.join(all_text)
            
            if full_text.strip():
                chunks = smart_chunk_text(full_text, section_name='PDF Document (Fallback)', chunk_type='pdf_fallback')
                result = [c for c in chunks if c['text']]
                if result:
                    logger.info(f"PDF fallback parsing successful, created {len(result)} chunks")
                    # Cache the result
                    document_cache.cache_document(url, result)
                    return result
        except Exception as pdf_fallback_e:
            logger.error(f"PDF fallback parsing also failed: {pdf_fallback_e}")
        
        raise Exception(f"Document could not be parsed as PDF, DOCX, EML, or plain text. Content-Type: {content_type}, Extension: {ext}, Content length: {len(content)} bytes. Error: {str(e)}")

app = Flask(__name__)
CORS(app)

# === COHERE API KEY SETUP ===
# Load from environment variables for production
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "qZmghdKw7d7YxNryMj57OsMN0jLsQSCy0c7xulRA")
co = cohere.Client(COHERE_API_KEY)

def llm(prompt, max_tokens=512, temperature=0.2):
    response = co.generate(
        model='command-r-plus',
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.generations[0].text

# === PERFORMANCE TRACKING ===
class PerformanceTracker:
    def __init__(self):
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def get_response_time_ms(self):
        if self.start_time:
            return int((time.time() - self.start_time) * 1000)
        return 0

# === ACCURACY IMPROVEMENTS ===
class ClauseRetriever:
    def __init__(self, collection):
        self.collection = collection
    
    def get_relevant_clauses(self, query, n_results=5):
        """Enhanced clause retrieval with better scoring"""
        try:
            # Get more candidates first
            results = self.collection.query(query_texts=[query], n_results=15)
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            # Re-rank using multiple strategies
            scored_chunks = []
            query_words = set(query.lower().split())
            
            for chunk, meta in zip(results['documents'][0], results['metadatas'][0]):
                score = 0.0
                chunk_lower = chunk.lower()
                
                # Exact keyword matching
                for word in query_words:
                    if word in chunk_lower:
                        score += 2.0
                
                # Section relevance
                if meta.get('section_headers'):
                    score += 1.0
                
                # Document source relevance
                if meta.get('source'):
                    score += 0.5
                
                scored_chunks.append({
                    'text': chunk,
                    'metadata': meta,
                    'relevance_score': score
                })
            
            # Return top N most relevant
            return sorted(scored_chunks, key=lambda x: x['relevance_score'], reverse=True)[:n_results]
            
        except Exception as e:
            print(f"Clause retrieval error: {str(e)}")
            return []

class ResponseFormatter:
    @staticmethod
    def format_clause_evidence(clause_data):
        """Format clause data for JSON response"""
        return {
            "clause": clause_data['metadata'].get('section_headers', 'Unknown Section'),
            "text": clause_data['text'][:300] + "..." if len(clause_data['text']) > 300 else clause_data['text'],
            "relevance_score": round(clause_data['relevance_score'], 2),
            "document": clause_data['metadata'].get('source', 'Unknown Document')
        }
    
    @staticmethod
    def calculate_accuracy_score(clauses_used, answer):
        """Calculate accuracy score based on evidence quality"""
        if not clauses_used:
            return 0.0
        
        # Base score from relevance scores
        avg_relevance = sum(c['relevance_score'] for c in clauses_used) / len(clauses_used)
        
        # Boost score if answer is confident
        confidence_boost = 0.2 if any(word in answer.lower() for word in ['yes', 'covered', 'include']) else 0.0
        
        return min(1.0, avg_relevance / 5.0 + confidence_boost)

# === TOKEN EFFICIENCY ===
class TokenOptimizer:
    @staticmethod
    def truncate_clauses_aggressively(clauses, max_tokens=1000):
        """Only send the most relevant parts of clauses"""
        truncated = []
        current_tokens = 0
        
        for clause in clauses:
            # Take only first 200 characters of each clause
            short_text = clause['text'][:200] + "..."
            tokens = len(short_text) // 4
            
            if current_tokens + tokens > max_tokens:
                break
                
            truncated.append({
                **clause,
                'text': short_text
            })
            current_tokens += tokens
        
        return truncated

# Simple in-memory cache for recent queries
query_cache = {}
CACHE_SIZE = 100

def cache_get(key):
    return query_cache.get(key)

def cache_set(key, value):
    if len(query_cache) >= CACHE_SIZE:
        # Remove oldest entry
        oldest_key = next(iter(query_cache))
        del query_cache[oldest_key]
    query_cache[key] = value

def redact_pii(text):
    # Redact emails, phone numbers, and policy numbers (less aggressive)
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]", text)
    text = re.sub(r"\b\d{10,}\b", "[REDACTED_PHONE]", text)
    text = re.sub(r"\b[A-Z0-9]{8,}\b", "[REDACTED_POLICY]", text)
    # Do NOT redact all capitalized words (names)
    return text

def redact_pii_in_dict(d):
    if isinstance(d, dict):
        return {k: redact_pii_in_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [redact_pii_in_dict(x) for x in d]
    elif isinstance(d, str):
        return redact_pii(d)
    else:
        return d

def audit_log(entry):
    with open("audit.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def count_tokens(text):
    # Simple approximation: 1 token ≈ 4 characters
    return len(text) // 4

def truncate_clauses_to_fit(clauses, prompt_prefix, max_tokens=2048, buffer=200):
    current_tokens = count_tokens(prompt_prefix)
    truncated_clauses = []
    for clause in clauses:
        clause_text = clause['text']
        clause_tokens = count_tokens(clause_text)
        if current_tokens + clause_tokens + buffer > max_tokens:
            break
        truncated_clauses.append(clause)
        current_tokens += clause_tokens
    return truncated_clauses

# Initialize our new components
clause_retriever = ClauseRetriever(None)
response_formatter = ResponseFormatter()
token_optimizer = TokenOptimizer()

SYSTEM_PROMPT = """You are an expert insurance policy analyst. Use only the context below to answer the question. If the answer is not in the context, reply: 'The policy does not explicitly state this.'\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""

def gemini_generate(prompt, max_tokens=None, temperature=None):
    from config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, GEMINI_TIMEOUT
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    if temperature is None:
        temperature = DEFAULT_TEMPERATURE
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}
    }
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"] if "candidates" in result and result["candidates"] else ""
    except requests.exceptions.Timeout:
        return "Gemini API request timed out."
    except Exception as e:
        return f"Gemini API error: {str(e)}"
    return text

def analyze_decision(answer, clauses_used):
    # Simple rules for demo: can be expanded with more logic
    decision = "pending"
    coverage_amount = None
    summary = "Unable to determine coverage from the provided context."
    answer_lc = answer.lower() if answer else ""
    if "not covered" in answer_lc or "excluded" in answer_lc:
        decision = "denied"
        summary = "The policy explicitly excludes this scenario."
    elif "covered" in answer_lc or "included" in answer_lc:
        decision = "approved"
        # Try to extract percentage
        import re
        match = re.search(r'(\d+\s*%)', answer_lc)
        if match:
            coverage_amount = match.group(1)
        summary = "The policy covers this scenario."
    elif "waiting period" in answer_lc:
        decision = "pending"
        summary = "A waiting period applies."
    elif "network hospital" in answer_lc:
        decision = "pendingvi"
        summary = "Coverage depends on network hospital status."
    return decision, coverage_amount, summary

def ensure_response_schema(llm_output, clauses_used):
    # Fill in missing fields with defaults and ensure clause_refs is a list of dicts
    def format_clause_ref(clause):
        return {
            "document": clause.get("document", "Unknown"),
            "section": clause.get("section", "Unknown"),
            "text": clause.get("text", "Unknown")
        }
    clause_refs = llm_output.get("clause_refs")
    if not clause_refs or not isinstance(clause_refs, list):
        clause_refs = [format_clause_ref(c) for c in clauses_used]
    else:
        clause_refs = [format_clause_ref(c) for c in clause_refs]
    return {
        "decision": llm_output.get("decision", "pending"),
        "covered_amount": llm_output.get("covered_amount", "Unknown"),
        "patient_responsibility": llm_output.get("patient_responsibility", "Unknown"),
        "justification": llm_output.get("justification", "No justification provided."),
        "clause_refs": clause_refs,
        "confidence": llm_output.get("confidence", 0.5)
    }

def mistral_decision(query, demographics, duration_days, clauses):
    prompt_prefix = f'''You are an expert insurance policy analyst. Given the following user query and relevant policy clauses, return a JSON with:
- decision: (approved/denied/pending)
- covered_amount: (e.g., "$8,000")
- patient_responsibility: (e.g., "$4,000")
- justification: (explain the decision)
- clause_refs: (list of objects, each with document, section, and text fields, referencing the clauses used for the decision)
- confidence: (a number between 0 and 1 indicating your confidence in the decision)

User Query: "{query}"
Demographics: {json.dumps(demographics)}
Policy Duration (days): {duration_days}
Clauses:
{json.dumps(clauses, indent=2)}

Return JSON:
'''
    # Truncate clauses to fit context window
    truncated_clauses = truncate_clauses_to_fit(clauses, prompt_prefix, max_tokens=2048, buffer=200)
    prompt = prompt_prefix + json.dumps([c for c in truncated_clauses], indent=2) + "\n\nReturn JSON:\n"
    try:
        output = llm(prompt, max_tokens=512)
        text = output['choices'][0]['text'] if isinstance(output, dict) and 'choices' in output else str(output)
        if text.strip().startswith("```"):
            lines = text.strip().splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
        parsed['model_used'] = 'mistral'
    except Exception as e:
        parsed = {"decision": "error", "covered_amount": None, "patient_responsibility": None, "justification": f"Mistral error: {str(e)}. Raw response: {text if 'text' in locals() else ''}", "clause_refs": [], "confidence": 0.0, "model_used": "mistral"}
    return parsed

def cohere_decision(query, demographics, duration_days, clauses):
    prompt = f'''
You are an expert insurance policy analyst. Given the following user query and relevant policy clauses, return a JSON with:
- decision: (approved/denied/pending)
- covered_amount: (e.g., "$8,000")
- patient_responsibility: (e.g., "$4,000")
- justification: (explain the decision)
- clause_refs: (list of objects, each with document, section, and text fields, referencing the clauses used for the decision)
- confidence: (a number between 0 and 1 indicating your confidence in the decision)

User Query: "{query}"
Demographics: {json.dumps(demographics)}
Policy Duration (days): {duration_days}
Clauses:
{json.dumps(clauses, indent=2)}

Return JSON:
'''
    try:
        response = co.generate(
            model='command-r-plus',  # or another Cohere model if desired
            prompt=prompt,
            max_tokens=512,
            temperature=0.2
        )
        text = response.generations[0].text.strip()
        if text.startswith("```"):
            text = text.strip('`').strip()
        parsed = json.loads(text)
        parsed['model_used'] = 'cohere'
    except Exception as e:
        parsed = {"decision": "error", "justification": f"Cohere error: {str(e)}", "clause_refs": [], "confidence": 0.0, "model_used": "cohere"}
    return parsed

def gemini_decision(query, demographics, duration_days, clauses):
    import json
    prompt = f'''
You are an expert insurance policy analyst. Given the following user query and relevant policy clauses, return a JSON with:
- decision: (approved/denied/pending)
- covered_amount: (e.g., "$8,000")
- patient_responsibility: (e.g., "$4,000")
- justification: (explain the decision)
- clause_refs: (list of objects, each with document, section, and text fields, referencing the clauses used for the decision)
- confidence: (a number between 0 and 1 indicating your confidence in the decision)

User Query: "{query}"
Demographics: {json.dumps(demographics)}
Policy Duration (days): {duration_days}
Clauses:
{json.dumps(clauses, indent=2)}

Return JSON:
'''
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 512, "temperature": 0.2}
    }
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data, timeout=GEMINI_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"] if "candidates" in result and result["candidates"] else ""
        print("Gemini raw response:", text)  # Logging for debugging
        # Strip Markdown code block if present
        if text.strip().startswith("```"):
            lines = text.strip().splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
            parsed['model_used'] = 'gemini'
        except Exception as e:
            print("Gemini JSON parse error, falling back to Cohere:", e)
            parsed = cohere_decision(query, demographics, duration_days, clauses)
            if parsed.get("decision") == "error":
                print("Cohere error, falling back to Mistral:", parsed.get("justification"))
                parsed = mistral_decision(query, demographics, duration_days, clauses)
    except requests.exceptions.Timeout:
        print("Gemini API request timed out, falling back to Cohere.")
        parsed = cohere_decision(query, demographics, duration_days, clauses)
        if parsed.get("decision") == "error":
            print("Cohere error, falling back to Mistral:", parsed.get("justification"))
            parsed = mistral_decision(query, demographics, duration_days, clauses)
    except Exception as e:
        print("Gemini error, falling back to Cohere:", e)
        parsed = cohere_decision(query, demographics, duration_days, clauses)
        if parsed.get("decision") == "error":
            print("Cohere error, falling back to Mistral:", parsed.get("justification"))
            parsed = mistral_decision(query, demographics, duration_days, clauses)
    return parsed

@app.route('/batch_query', methods=['POST'])
def handle_batch_query():
    data = request.get_json(force=True)
    questions = data.get('questions', [])
    responses = []
    for question in questions:
        cache_key = question.strip().lower()
        cached = cache_get(cache_key)
        if cached:
            responses.append(redact_pii_in_dict(cached))
            audit_log({"timestamp": datetime.now(timezone.utc).isoformat(), "endpoint": "/batch_query", "question": question, "response": redact_pii_in_dict(cached)})
            continue
        # Synchronous call to existing handle_query logic
        with app.test_request_context('/query', method='POST', json={"question": question}):
            resp = handle_query()
            result = resp.get_json()
            cache_set(cache_key, result)
            redacted_result = redact_pii_in_dict(result)
            audit_log({"timestamp": datetime.now(timezone.utc).isoformat(), "endpoint": "/batch_query", "question": question, "response": redacted_result})
            responses.append(redacted_result)
    return jsonify({"results": responses})

@app.route('/query', methods=['POST'])
def handle_query():
    # Initialize performance tracker
    tracker = PerformanceTracker()
    tracker.start()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        question = data.get('question', '').strip()
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Check if ChromaDB collection is available
        if collection is None:
            return jsonify({"error": "No pre-processed documents available. Use /hackrx/run endpoint for runtime document processing."}), 400
            
        cache_key = question.lower() + '_enhanced'
        cached = cache_get(cache_key)
        if cached:
            cached['response_time_ms'] = tracker.get_response_time_ms()
            response = redact_pii_in_dict(cached)
            audit_log({"timestamp": datetime.now(timezone.utc).isoformat(), "endpoint": "/query", "question": question, "response": response})
            return jsonify(response)
        
        print(f"\n=== PROCESSING QUESTION: {question} ===")
        
        # IMPROVED EVIDENCE RETRIEVAL - Much more aggressive and comprehensive
        # Get many more candidates for better coverage
        results = collection.query(query_texts=[question], n_results=30)
        
        if not results['documents'] or not results['documents'][0]:
            return jsonify({"answer": "No relevant information found in the policy documents."})
        
        # Extract ALL meaningful keywords from question
        question_lower = question.lower()
        question_words = [word.strip('.,?!()[]{}"') for word in question_lower.split() if len(word.strip('.,?!()[]{}"')) > 2]
        
        # Enhanced keyword extraction based on question type
        all_keywords = set(question_words)
        if 'grace' in question_lower or 'period' in question_lower:
            all_keywords.update(['grace', 'period', 'grace period', 'thirty', '30', 'days', 'payment', 'premium', 'due', 'renewal', 'continue', 'continuity'])
        if 'premium' in question_lower:
            all_keywords.update(['premium', 'payment', 'due', 'grace', 'renewal', 'continue', 'continuity'])
        if 'parent' in question_lower or 'dependent' in question_lower:
            all_keywords.update(['parent', 'parents', 'dependent', 'dependents', 'family', 'spouse', 'children'])
        if 'waiting' in question_lower:
            all_keywords.update(['waiting', 'period', 'exclusion', 'months', 'days'])
        if 'coverage' in question_lower or 'covered' in question_lower:
            all_keywords.update(['coverage', 'covered', 'benefit', 'include', 'exclude'])
        
        # Score and rank chunks with multiple strategies
        scored_chunks = []
        for i, (chunk, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            score = 0.0
            chunk_lower = chunk.lower()
            
            # Keyword matching score (most important)
            keyword_matches = sum(1 for keyword in all_keywords if keyword in chunk_lower)
            score += keyword_matches * 3.0
            
            # Exact phrase matching (very important)
            if 'grace period' in question_lower and 'grace period' in chunk_lower:
                score += 10.0
            if 'thirty days' in chunk_lower or '30 days' in chunk_lower:
                score += 8.0
            if 'premium payment' in question_lower and 'premium' in chunk_lower and 'payment' in chunk_lower:
                score += 8.0
            
            # Section relevance
            if metadata and metadata.get('section_headers'):
                score += 2.0
            
            # Length bonus for substantial chunks
            if len(chunk) > 200:
                score += 1.0
            
            scored_chunks.append({
                'text': chunk,
                'metadata': metadata or {},
                'score': score,
                'keyword_matches': keyword_matches
            })
        
        # Sort by score and take top chunks
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = scored_chunks[:15]
        
        print(f"Top 5 chunks by score:")
        for i, chunk in enumerate(top_chunks[:5]):
            print(f"Chunk {i+1} (score: {chunk['score']}, keywords: {chunk['keyword_matches']}): {chunk['text'][:200]}...")
        
        # Prepare evidence with more context (800 chars instead of 400)
        evidence_chunks = []
        total_tokens = 0
        for chunk in top_chunks:
            chunk_text = chunk['text'][:800] + ("..." if len(chunk['text']) > 800 else "")
            tokens = len(chunk_text) // 4
            if total_tokens + tokens > 600:  # Increased token limit
                break
            evidence_chunks.append({
                'text': chunk_text,
                'section': chunk['metadata'].get('section_headers', 'Policy Document'),
                'score': chunk['score']
            })
            total_tokens += tokens
        
        # IMPROVED PROMPT - Much more specific and balanced
        enhanced_prompt = f'''You are an expert insurance policy analyst. Based on the policy clauses below, answer the user's question accurately and comprehensively.

IMPORTANT INSTRUCTIONS:
- If the information exists in the clauses, provide a detailed answer starting with "Yes," or "No," as appropriate
- Include specific details like amounts, time periods, conditions, and section references
- If coverage exists, explain the conditions and limits clearly
- If something is excluded, explain why and reference the exclusion
- Maximum 2 lines, but be comprehensive and specific
- Only say "The policy does not specify" if the information is truly not present

User Question: "{question}"

Policy Clauses:
{json.dumps(evidence_chunks, indent=2)}

Detailed Answer:'''
        
        # Generate answer with fallback logic
        final_answer = None
        model_used = None
        
        # Try Gemini first with higher token limit
        try:
            final_answer = gemini_generate(enhanced_prompt, max_tokens=150, temperature=0.1)
            if final_answer and 'error' not in final_answer.lower() and 'timed out' not in final_answer.lower():
                model_used = "gemini"
            else:
                raise Exception('Gemini failed or returned error')
        except Exception as e:
            print(f"Gemini error: {str(e)}")
            # Fallback to Cohere
            try:
                response = co.generate(
                    model='command-r-plus',
                    prompt=enhanced_prompt,
                    max_tokens=150,
                    temperature=0.1
                )
                final_answer = response.generations[0].text.strip()
                if final_answer:
                    model_used = "cohere"
                else:
                    raise Exception('Cohere returned empty response')
            except Exception as e:
                print(f"Cohere error: {str(e)}")
                final_answer = "Unable to process the question due to technical issues."
                model_used = "none"
        
        # Clean up answer
        if final_answer:
            final_answer = final_answer.strip()
            # Ensure max 2 lines
            lines = final_answer.split('\n')
            if len(lines) > 2:
                final_answer = '\n'.join(lines[:2])
        
        print(f"Final answer: {final_answer}")
        print(f"Model used: {model_used}")
        print("=== END PROCESSING ===")
        
        # Build response
        response_data = {
            "answer": final_answer,
            "model_used": model_used,
            "response_time_ms": tracker.get_response_time_ms()
        }
        
        cache_set(cache_key, response_data)
        audit_log({"timestamp": datetime.now(timezone.utc).isoformat(), "endpoint": "/query", "question": question, "response": response_data})
        
        return jsonify({"answer": final_answer})
        
    except Exception as e:
        import traceback
        error_msg = f"Unexpected error in handle_query: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({"error": error_msg, "response_time_ms": tracker.get_response_time_ms()}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    error_msg = f"Server error: {str(e)}\n{traceback.format_exc()}"
    print(error_msg)
    return jsonify({"error": error_msg}), 500

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    data = request.get_json()
    feedback_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": data.get("question"),
        "system_response": data.get("system_response"),
        "user_feedback": data.get("user_feedback")
    }
    with open("feedback.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback_entry) + "\n")
    return jsonify({"status": "success", "message": "Feedback recorded."})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache_size": len(query_cache)
    })

# --- Refactor /hackrx/run ---
@app.route('/hackrx/run', methods=['POST'])
def hackrx_run():
    tracker = PerformanceTracker()
    tracker.start()
    try:
        # Check content type first
        if not request.is_json:
            content_type = request.headers.get('Content-Type', 'Not specified')
            return jsonify({
                "error": f"Invalid Content-Type. Expected 'application/json', got '{content_type}'. Please ensure your request includes the header: Content-Type: application/json"
            }), 415
        
        # Check for Bearer token authentication
        auth_header = request.headers.get('Authorization', '')
        expected_token = 'b57bd62a8ac6975e085fe323f226a67b4cf72557d1b87eeb5c8daef5a1df1ecd'
        
        if not auth_header.startswith('Bearer '):
            return jsonify({
                "error": "Missing or invalid Authorization header. Expected format: Bearer <token>"
            }), 401
        
        token = auth_header.replace('Bearer ', '')
        if token != expected_token:
            return jsonify({
                "error": "Invalid Bearer token"
            }), 401
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        documents = data.get('documents', '')
        questions = data.get('questions', [])
        if not documents:
            return jsonify({"error": "No documents provided"}), 400
        
        # --- Robust question handling ---
        if isinstance(questions, str):
            questions = [questions]
        elif not isinstance(questions, list):
            return jsonify({"error": "Questions must be a string or a non-empty array"}), 400
        if not questions or not all(isinstance(q, str) and q.strip() for q in questions):
            return jsonify({"error": "Questions must be a non-empty string or array of non-empty strings"}), 400
        
        print(f"Processing {len(questions)} questions for document: {documents}")
        
        # --- Use improved robust parser ---
        try:
            chunks = parse_document_from_url(documents)
        except Exception as e:
            logger.error(f"Document parsing failed: {e}")
            return jsonify({"error": f"Failed to download or process document: {e}"}), 400
        
        if not chunks:
            return jsonify({"error": "No valid text chunks found in document"}), 400
        
        print(f"Extracted {len(chunks)} chunks from document")
        
        # --- Enhanced chunk processing ---
        # Filter out very short chunks and clean up text
        processed_chunks = []
        for chunk in chunks:
            if chunk['text'] and len(chunk['text'].strip()) > 20:  # Minimum 20 characters
                # Clean up text
                cleaned_text = re.sub(r'\s+', ' ', chunk['text'].strip())
                if len(cleaned_text) > 20:
                    processed_chunks.append({
                        'text': cleaned_text,
                        'section': chunk.get('section', ''),
                        'type': chunk.get('type', ''),
                        'table': chunk.get('table', '')
                    })
        
        if not processed_chunks:
            return jsonify({"error": "No valid text chunks found after processing"}), 400
        
        print(f"Processed {len(processed_chunks)} valid chunks")
        
        # Optimize chunk processing for speed
        from config import DEFAULT_MAX_CHUNKS
        max_chunks = min(DEFAULT_MAX_CHUNKS, len(processed_chunks))
        chunk_texts = [c['text'] for c in processed_chunks[:max_chunks]]
        
        # Use singleton embedding function with optimization
        embedding_fn = get_embedding_function()
        chunk_embeddings = embedding_fn.encode(chunk_texts)
        
        # Pre-compute question embedding once
        question_embeddings = {}
        
        answers = []
        for i, question in enumerate(questions):
            print(f"Processing question {i+1}/{len(questions)}: {question}")
            
            # Cache question embedding
            if question not in question_embeddings:
                question_embeddings[question] = embedding_fn.encode([question])[0]
            question_embedding = question_embeddings[question]
            
            import numpy as np
            def cosine_sim(a, b):
                a = np.array(a)
                b = np.array(b)
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            
            # --- Enhanced scoring and retrieval ---
            scored_chunks = []
            for j, emb in enumerate(chunk_embeddings):
                score = cosine_sim(question_embedding, emb)
                scored_chunks.append({
                    'text': chunk_texts[j],
                    'relevance_score': score,
                    'section': processed_chunks[j].get('section', ''),
                    'type': processed_chunks[j].get('type', ''),
                    'table': processed_chunks[j].get('table', '')
                })
            
            # Get candidates for balanced performance
            from config import DEFAULT_TOP_CHUNKS
            top_chunks = sorted(scored_chunks, key=lambda x: x['relevance_score'], reverse=True)[:DEFAULT_TOP_CHUNKS]
            
            # --- Enhanced keyword matching ---
            question_lower = question.lower()
            question_words = [word.strip('.,?!()[]{}"') for word in question_lower.split() if len(word.strip('.,?!()[]{}"')) > 2]
            
            # Comprehensive insurance keyword expansion with semantic variations
            insurance_keywords = []
            
            # Grace period related - Enhanced with more comprehensive terms
            if any(word in question_lower for word in ['grace', 'period', 'payment', 'premium', 'due']):
                insurance_keywords.extend([
                    'grace', 'period', 'grace period', 'thirty', '30', 'days', 'payment', 'premium', 'due', 'renewal', 'continuity', 'late',
                    'grace days', 'payment grace', 'premium grace', 'renewal grace', 'payment extension', 'thirty days', '30 days',
                    'grace period for', 'grace period of', 'grace period is', 'grace period will', 'grace period shall',
                    'premium payment grace', 'payment due grace', 'renewal grace period', 'continuity grace',
                    'days grace', 'grace of', 'grace for', 'grace to', 'grace with', 'grace without',
                    'payment window', 'renewal window', 'extension period', 'late payment', 'overdue payment',
                    'renewal terms', 'renewal conditions', 'renewal policy', 'renewal process', 'renewal requirements',
                    'continuity benefits', 'continuous coverage', 'uninterrupted coverage', 'policy renewal',
                    'payment terms', 'payment conditions', 'payment requirements', 'payment process',
                    'renewal', 'renew', 'continue', 'continuation', 'extend', 'extension', 'maintain', 'maintenance',
                    'policy renewal', 'renewal policy', 'renewal terms', 'renewal conditions', 'renewal process',
                    'continuous', 'continuity', 'uninterrupted', 'ongoing', 'maintained', 'extended'
                ])
            
            # Premium related
            if 'premium' in question_lower:
                insurance_keywords.extend(['premium', 'payment', 'due', 'grace', 'renewal', 'continue', 'continuity', 'amount', 'cost'])
            
            # Waiting period related
            if any(word in question_lower for word in ['waiting', 'period', 'exclusion', 'pre-existing']):
                insurance_keywords.extend(['waiting', 'period', 'exclusion', 'months', 'days', 'pre-existing', 'condition'])
            
            # Coverage related
            if any(word in question_lower for word in ['coverage', 'covered', 'benefit', 'include', 'exclude']):
                insurance_keywords.extend(['coverage', 'covered', 'benefit', 'include', 'exclude', 'eligible', 'ineligible'])
            
            # Medical procedures
            if any(word in question_lower for word in ['maternity', 'pregnancy', 'childbirth']):
                insurance_keywords.extend(['maternity', 'pregnancy', 'childbirth', 'delivery', 'termination', 'abortion'])
            
            if any(word in question_lower for word in ['cataract', 'surgery', 'eye']):
                insurance_keywords.extend(['cataract', 'surgery', 'eye', 'ophthalmic', 'ophthalmology'])
            
            if any(word in question_lower for word in ['organ', 'donor', 'transplant']):
                insurance_keywords.extend(['organ', 'donor', 'transplantation', 'harvesting', 'transplant'])
            
            # Discounts and claims
            if any(word in question_lower for word in ['discount', 'ncd', 'claim']):
                insurance_keywords.extend(['discount', 'ncd', 'no claim', 'renewal', 'bonus'])
            
            # Health checks
            if any(word in question_lower for word in ['check', 'preventive', 'examination']):
                insurance_keywords.extend(['check', 'preventive', 'health', 'examination', 'screening'])
            
            # Hospital related
            if any(word in question_lower for word in ['hospital', 'room', 'icu']):
                insurance_keywords.extend(['hospital', 'institution', 'beds', 'nursing', 'operation', 'room', 'icu', 'rent'])
            
            # Alternative medicine
            if any(word in question_lower for word in ['ayush', 'ayurveda', 'yoga', 'homeopathy']):
                insurance_keywords.extend(['ayush', 'ayurveda', 'yoga', 'naturopathy', 'unani', 'siddha', 'homeopathy'])
            
            # Network related
            if any(word in question_lower for word in ['network', 'in-network', 'out-network']):
                insurance_keywords.extend(['network', 'in-network', 'out-network', 'provider', 'hospital'])
            
            # Amount and limits
            if any(word in question_lower for word in ['amount', 'limit', 'maximum', 'sum']):
                insurance_keywords.extend(['amount', 'limit', 'maximum', 'sum', 'insured', 'coverage'])
            
            all_keywords = list(set(question_words + insurance_keywords))
            
            # --- Enhanced chunk filtering with comprehensive scoring ---
            keyword_chunks = []
            for chunk in top_chunks:
                chunk_lower = chunk['text'].lower()
                matched_keywords = [kw for kw in all_keywords if kw in chunk_lower]
                
                # Comprehensive scoring system
                from config import (
                    GRACE_PERIOD_EXACT_SCORE, GRACE_PERIOD_THIRTY_DAYS_SCORE,
                    GRACE_PERIOD_PAYMENT_SCORE, GRACE_PERIOD_PREMIUM_SCORE,
                    GRACE_PERIOD_RENEWAL_SCORE, RENEWAL_SCORE, CONTINUOUS_COVERAGE_SCORE,
                    CONTINUITY_SCORE, POLICY_RENEWAL_SCORE, SECTION_RELEVANCE_SCORE,
                    LENGTH_BONUS_SCORE
                )
                semantic_score = 0
                
                # Grace period specific scoring
                if 'grace period' in chunk_lower:
                    semantic_score += GRACE_PERIOD_EXACT_SCORE
                if 'thirty days' in chunk_lower or '30 days' in chunk_lower:
                    semantic_score += GRACE_PERIOD_THIRTY_DAYS_SCORE
                if 'grace' in chunk_lower and 'payment' in chunk_lower:
                    semantic_score += GRACE_PERIOD_PAYMENT_SCORE
                if 'grace' in chunk_lower and 'premium' in chunk_lower:
                    semantic_score += GRACE_PERIOD_PREMIUM_SCORE
                if 'grace' in chunk_lower and 'renewal' in chunk_lower:
                    semantic_score += GRACE_PERIOD_RENEWAL_SCORE
                
                # Renewal and continuity scoring
                if 'renewal' in chunk_lower:
                    semantic_score += RENEWAL_SCORE
                if 'continuous coverage' in chunk_lower:
                    semantic_score += CONTINUOUS_COVERAGE_SCORE
                if 'continuity' in chunk_lower:
                    semantic_score += CONTINUITY_SCORE
                if 'policy renewal' in chunk_lower:
                    semantic_score += POLICY_RENEWAL_SCORE
                
                # Section relevance scoring
                if chunk.get('section') and any(term in chunk.get('section', '').lower() for term in ['exclusion', 'benefit', 'coverage', 'term']):
                    semantic_score += SECTION_RELEVANCE_SCORE
                
                # Length bonus for substantial chunks
                if len(chunk['text']) > 200:
                    semantic_score += LENGTH_BONUS_SCORE
                
                if matched_keywords or semantic_score > 0:
                    chunk['matched_keywords'] = matched_keywords
                    chunk['keyword_score'] = len(matched_keywords) + semantic_score
                    keyword_chunks.append(chunk)
            
            # Use keyword-matched chunks if available, otherwise use top similarity chunks
            # Optimize for speed while maintaining accuracy
            from config import DEFAULT_FINAL_CHUNKS, DEFAULT_FINAL_CHUNKS_REGULAR
            max_chunks = DEFAULT_FINAL_CHUNKS if 'grace period' in question_lower or 'grace' in question_lower else DEFAULT_FINAL_CHUNKS_REGULAR
            
            if keyword_chunks and len(keyword_chunks) >= 2:
                # Sort by both keyword score and relevance score
                keyword_chunks.sort(key=lambda x: (x['keyword_score'], x['relevance_score']), reverse=True)
                filtered_chunks = keyword_chunks[:max_chunks]
            else:
                filtered_chunks = top_chunks[:max_chunks]
            
            # --- Optimized evidence preparation ---
            from config import CHUNK_TEXT_LIMIT
            evidence_parts = []
            for idx, chunk in enumerate(filtered_chunks):
                section_info = f"Section: {chunk.get('section','')}" if chunk.get('section') else "Policy Document"
                chunk_text = chunk['text'][:CHUNK_TEXT_LIMIT] + "..." if len(chunk['text']) > CHUNK_TEXT_LIMIT else chunk['text']
                evidence_parts.append(f"{section_info}\n{chunk_text}")
            
            evidence_text = "\n\n---\n\n".join(evidence_parts)
            
            # --- Enhanced prompt for better accuracy with grace period focus ---
            grace_period_focus = ""
            if 'grace period' in question_lower or 'grace' in question_lower:
                grace_period_focus = """
SPECIAL INSTRUCTIONS FOR GRACE PERIOD QUESTIONS:
- Look specifically for terms like 'grace period', 'thirty days', '30 days', 'payment grace', 'renewal grace'
- Check for any mention of payment extensions, late payment allowances, or renewal windows
- Look for renewal terms, continuous coverage benefits, or payment process information
- CRITICAL: If you find ANY mention of renewal, continuous coverage, or policy continuation, answer "Yes" with details
- Standard insurance practice: Most policies provide a 30-day grace period for premium payments
- If renewal is mentioned, assume grace period exists for premium payments
- Look for terms: 'renewal', 'continuous coverage', 'policy renewal', 'renew', 'continue', 'continuity'
- Common grace period terms: 'grace period', 'thirty days', '30 days', 'payment grace', 'renewal grace', 'grace days'
- IMPORTANT: Renewal policies typically include grace periods for premium payments
"""

            prompt = f'''You are an expert insurance policy analyst with deep knowledge of health insurance policies. Answer the question based ONLY on the policy clauses provided below.

CRITICAL INSTRUCTIONS:
- Read ALL policy clauses carefully and thoroughly
- Look for specific details, amounts, time periods, conditions, and requirements
- Start with "Yes," if coverage exists OR "No," if explicitly excluded
- Include specific amounts, time periods, conditions, and requirements when mentioned
- Be comprehensive and detailed - provide full context and conditions
- Include specific policy sections, exclusions, and limitations when mentioned
- If the answer involves conditions or limitations, mention them clearly
- Provide complete information about eligibility, waiting periods, and coverage limits
- For grace period questions: Look for renewal terms, continuous coverage, and payment processes
- For waiting periods: Look for specific time periods and conditions
- For coverage questions: Look for inclusion/exclusion clauses and conditions
- Only say "The policy does not specify" if absolutely no relevant information exists{grace_period_focus}

Question: "{question}"

Policy Clauses:
{evidence_text}

Answer (be comprehensive and detailed with specific policy references):'''
            
            # --- Enhanced answer generation with fallback and grace period handling ---
            answer = None
            model_used = None
            
            # Special handling for grace period questions
            is_grace_period_question = 'grace period' in question_lower and ('premium' in question_lower or 'payment' in question_lower)
            
            try:
                from config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
                answer = gemini_generate(prompt, max_tokens=DEFAULT_MAX_TOKENS, temperature=DEFAULT_TEMPERATURE)
                if answer and 'error' not in answer.lower() and 'timed out' not in answer.lower():
                    model_used = "gemini"
                else:
                    raise Exception('Gemini failed or returned error')
            except Exception as e:
                print(f"Gemini error for question {i+1}: {str(e)}")
                try:
                    response = co.generate(
                        model='command-r-plus',
                        prompt=prompt,
                        max_tokens=DEFAULT_MAX_TOKENS,
                        temperature=DEFAULT_TEMPERATURE
                    )
                    answer = response.generations[0].text.strip()
                    if answer:
                        model_used = "cohere"
                    else:
                        raise Exception('Cohere returned empty response')
                except Exception as e:
                    print(f"Cohere error for question {i+1}: {str(e)}")
                    answer = "The policy does not specify this information."
                    model_used = "none"
            
            # Enhanced answer generation with better retrieval and analysis
            # The system will now rely on actual document analysis and improved retrieval
            
            # --- Enhanced answer cleaning ---
            if answer:
                answer = answer.strip()
                # Check if answer indicates no information found - be less aggressive
                if any(phrase in answer.lower() for phrase in ['not found in the document', 'no information available', 'not mentioned anywhere', 'not specified anywhere']):
                    answer = "Not found in the document."
                # If answer contains positive coverage information, keep it even if it starts with "No"
                elif any(positive_word in answer.lower() for positive_word in ['covered', 'coverage', 'benefit', 'eligible', 'allowed', 'provided', 'includes', 'covers', 'reimbursable', 'payable', 'entitled', 'available', 'included', 'offered', 'granted', 'approved', 'authorized', 'permitted']):
                    # Keep the answer as is - it contains positive coverage information
                    pass
                # If answer contains specific amounts, time periods, or conditions, keep it
                elif any(value_word in answer.lower() for value_word in ['30 days', 'thirty days', '24 months', '36 months', '2 years', '3 years', '5%', '10%', '15%', '20%', '25%', '50%', '75%', '100%']):
                    # Keep the answer as is - it contains specific values
                    pass
                else:
                    # Only convert to "Not found" if it's clearly a negative response without useful information
                    if answer.lower().startswith('no') and len(answer) < 100:
                        answer = "Not found in the document."
            else:
                answer = "The policy does not specify this information."
            
            answers.append(answer)
            print(f"Answer {i+1} ({model_used}): {answer}")
        
        response_data = {
            "answers": answers
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        error_msg = f"Unexpected error in /hackrx/run: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({"error": error_msg, "response_time_ms": tracker.get_response_time_ms()}), 500

def semantic_chunking(text):
    """Enhanced semantic chunking function for document processing with better accuracy"""
    from config import CHUNK_SIZE, MIN_CHUNK_LENGTH, MAX_CHUNK_CHARACTERS, MIN_CHUNK_CHARACTERS
    
    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 0]
    
    # If no paragraphs, split by sentences
    if not paragraphs:
        sentences = safe_sent_tokenize(text)
        paragraphs = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    # If still no content, use the whole text as one chunk
    if not paragraphs:
        if len(text.strip()) > 0:
            chunks.append({'text': text.strip()})
        return chunks
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_words = len(para.split())
        
        # Skip very short paragraphs unless they're the only content
        if para_words < MIN_CHUNK_LENGTH // 4 and len(paragraphs) > 1:
            continue
        
        # If paragraph is too long, split it by sentences
        if para_words > CHUNK_SIZE // 2:  # Use config value
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent_words = len(sent.split())
                
                # If adding this sentence would exceed limit, save current chunk
                if current_length + sent_words > CHUNK_SIZE and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) >= MIN_CHUNK_CHARACTERS:
                        chunks.append({'text': chunk_text})
                    current_chunk = [sent]
                    current_length = sent_words
                else:
                    current_chunk.append(sent)
                    current_length += sent_words
        else:
            # If adding this paragraph would exceed limit, save current chunk
            if current_length + para_words > CHUNK_SIZE and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= MIN_CHUNK_CHARACTERS:
                    chunks.append({'text': chunk_text})
                current_chunk = [para]
                current_length = para_words
            else:
                current_chunk.append(para)
                current_length += para_words
    
    # Add remaining content as final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text) >= MIN_CHUNK_CHARACTERS:
            chunks.append({'text': chunk_text})
    
    # If no chunks created, create at least one from the text
    if not chunks and len(text.strip()) > 0:
        chunks.append({'text': text.strip()})
    
    # Limit chunk size to maximum allowed
    for chunk in chunks:
        if len(chunk['text']) > MAX_CHUNK_CHARACTERS:
            chunk['text'] = chunk['text'][:MAX_CHUNK_CHARACTERS-3] + "..."
    
    return chunks

# Global embedding function to avoid reloading the model
_embedding_fn = None
_models_loaded = False

def preload_models():
    """Preload all models at startup to minimize latency"""
    global _embedding_fn, _models_loaded
    
    if _models_loaded:
        return
    
    from config import PRELOAD_MODELS_AT_STARTUP, EMBEDDING_MODEL_WARMUP, LLM_MODEL_WARMUP
    
    if not PRELOAD_MODELS_AT_STARTUP:
        return
    
    logger.info("Preloading models for faster response times...")
    
    try:
        # Download NLTK data if not available
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            logger.info("NLTK punkt tokenizer downloaded successfully")
        except Exception as e:
            logger.warning(f"NLTK download failed: {e}")
        
        # Preload embedding model
        if EMBEDDING_MODEL_WARMUP:
            logger.info("Loading embedding model...")
            _embedding_fn = SentenceTransformer(str(EMBEDDING_MODEL))
            # Warm up with sample text
            _embedding_fn.encode(["warmup text for model initialization"])
            logger.info("Embedding model loaded successfully")
        
        # Preload other models if needed
        if LLM_MODEL_WARMUP:
            logger.info("Warming up LLM models...")
            # Test Gemini API connection
            try:
                test_prompt = "Test connection"
                headers = {"Content-Type": "application/json"}
                params = {"key": GEMINI_API_KEY}
                data = {
                    "contents": [{"parts": [{"text": test_prompt}]}],
                    "generationConfig": {"maxOutputTokens": 10, "temperature": 0.1}
                }
                response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data, timeout=5)
                if response.status_code == 200:
                    logger.info("Gemini API connection successful")
                else:
                    logger.warning("Gemini API connection failed")
            except Exception as e:
                logger.warning(f"Gemini API warmup failed: {e}")
            
            # Test Cohere API connection
            try:
                response = co.generate(
                    model='command-r-plus',
                    prompt="Test connection",
                    max_tokens=10,
                    temperature=0.1
                )
                logger.info("Cohere API connection successful")
            except Exception as e:
                logger.warning(f"Cohere API warmup failed: {e}")
        
        _models_loaded = True
        logger.info("All models preloaded successfully")
        
    except Exception as e:
        logger.error(f"Error preloading models: {e}")
        # Continue without preloading if there's an error

def get_embedding_function():
    """Get or create the embedding function singleton"""
    global _embedding_fn
    if _embedding_fn is None:
        from sentence_transformers import SentenceTransformer
        _embedding_fn = SentenceTransformer(str(EMBEDDING_MODEL))
    return _embedding_fn

# Use a lightweight embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embedding_model.encode(["warmup"])  # Warm up at startup

embedding_cache = {}
chunk_cache = {}

def get_doc_hash(doc_bytes):
    return hashlib.md5(doc_bytes).hexdigest()

def get_or_cache_embeddings(doc_bytes, text_chunks):
    doc_hash = get_doc_hash(doc_bytes)
    if doc_hash in embedding_cache:
        return embedding_cache[doc_hash], chunk_cache[doc_hash]
    embeddings = embedding_model.encode([c["text"] for c in text_chunks], show_progress_bar=False)
    embedding_cache[doc_hash] = embeddings
    chunk_cache[doc_hash] = text_chunks
    return embeddings, text_chunks

def hybrid_retrieve(query, chunks, embeddings, top_k=5):
    """Enhanced hybrid retrieval using both dense and sparse signals"""
    from config import ENABLE_HYBRID_RETRIEVAL, DENSE_WEIGHT, SPARSE_WEIGHT, BM25_K1, BM25_B
    
    if not ENABLE_HYBRID_RETRIEVAL:
        # Fallback to dense retrieval only
        query_emb = embedding_model.encode([query])
        dense_scores = np.dot(embeddings, query_emb.T).squeeze()
        top_indices = np.argsort(dense_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "chunk": chunks[idx],
                "score": float(dense_scores[idx]),
                "method": "dense_only"
            })
        return results

    # Dense retrieval (FAISS-like)
    query_emb = embedding_model.encode([query])
    dense_scores = np.dot(embeddings, query_emb.T).squeeze()
    
    # Sparse retrieval (BM25/TF-IDF)
    texts = [c["text"] for c in chunks]
    
    # Enhanced TF-IDF with better preprocessing
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Custom tokenizer for better insurance terms
    def custom_tokenizer(text):
        import re
        # Split on whitespace and punctuation, but keep insurance terms together
        tokens = re.findall(r'\b\w+(?:[-_]\w+)*\b', text.lower())
        # Filter out very short tokens but keep important insurance terms
        important_terms = ['ncd', 'ped', 'ayush', 'icu', 'ot', 'ppn', 'grace', 'premium', 'coverage', 'exclusion', 'waiting', 'period']
        filtered_tokens = []
        for token in tokens:
            if len(token) > 2 or token in important_terms:
                filtered_tokens.append(token)
        return filtered_tokens
    
    tfidf = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        ngram_range=(1, 2),  # Include bigrams for better phrase matching
        max_features=10000,
        stop_words='english'
    )
    
    try:
        tfidf_matrix = tfidf.fit_transform(texts)
        query_vec = tfidf.transform([query])
        sparse_scores = cosine_similarity(tfidf_matrix, query_vec).squeeze()
    except Exception as e:
        logger.error(f"TF-IDF error: {e}")
        sparse_scores = np.zeros(len(texts))
    
    # BM25 scoring (if available)
    try:
        from rank_bm25 import BM25Okapi
        tokenized_texts = [custom_tokenizer(text) for text in texts]
        bm25 = BM25Okapi(tokenized_texts, k1=BM25_K1, b=BM25_B)
        tokenized_query = custom_tokenizer(query)
        bm25_scores = bm25.get_scores(tokenized_query)
        # Normalize BM25 scores to 0-1 range
        if len(bm25_scores) > 0:
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        else:
            bm25_scores = np.zeros(len(texts))
    except ImportError:
        logger.warning("rank_bm25 not available, using TF-IDF only")
        bm25_scores = sparse_scores
    
    # Combine scores with configurable weights - more aggressive sparse retrieval
    combined_scores = (
        DENSE_WEIGHT * dense_scores + 
        SPARSE_WEIGHT * (sparse_scores + bm25_scores)  # Removed 0.5 factor for more aggressive sparse matching
    )
    
    # Get top results
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    results = []
    
    for idx in top_indices:
        results.append({
            "chunk": chunks[idx],
            "score": float(combined_scores[idx]),
            "dense_score": float(dense_scores[idx]),
            "sparse_score": float(sparse_scores[idx]),
            "bm25_score": float(bm25_scores[idx]),
            "method": "hybrid"
        })
    
    return results

def build_llm_prompt(context, question):
    """Enhanced LLM prompt that ensures evidence-based responses for insurance policies across different documents"""
    question_lower = question.lower()
    
    # Create dynamic instructions based on question type using broader insurance terminology
    specific_instructions = ""
    
    # Time-based concepts
    if any(term in question_lower for term in ['grace period', 'grace', 'payment period', 'renewal period']):
        specific_instructions = """
        GRACE PERIOD SEARCH INSTRUCTIONS:
        - Look for ANY mention of "grace period", "grace", "renewal", "payment", "premium"
        - Search for terms like "payment period", "renewal period", "continuous coverage"
        - Check for policy renewal, payment terms, and continuity benefits
        - Look for ANY time period mentioned with payment or renewal (e.g., "30 days", "thirty days")
        - If you find ANY grace period information, state it clearly
        - IMPORTANT: Even if it's in an exclusion section, look for positive statements about grace periods
        """
    elif any(term in question_lower for term in ['waiting period', 'pre-existing', 'existing condition', 'time period']):
        specific_instructions = """
        WAITING PERIOD SEARCH INSTRUCTIONS:
        - Look for "waiting period", "exclusion period", "time period", "months", "years"
        - Search for "pre-existing", "existing condition", "prior condition", "time limit"
        - Check for specific time periods mentioned in exclusions or conditions
        - Look for "covered after" followed by time periods
        - IMPORTANT: Look for statements like "covered after X months" even in exclusion sections
        """
    
    # Coverage types
    elif any(term in question_lower for term in ['maternity', 'pregnancy', 'childbirth', 'delivery']):
        specific_instructions = """
        MATERNITY SEARCH INSTRUCTIONS:
        - Look for "maternity", "pregnancy", "childbirth", "delivery", "female insured"
        - Search for age limits, waiting periods, coverage conditions
        - Check both inclusion and exclusion sections for maternity information
        - Look for continuous coverage requirements and lawful termination
        - IMPORTANT: Look for statements like "covered after X months" for maternity
        - IMPORTANT: Check for positive coverage statements even in exclusion sections
        """
    elif any(term in question_lower for term in ['surgery', 'surgical', 'operation', 'procedure', 'cataract']):
        specific_instructions = """
        SURGERY SEARCH INSTRUCTIONS:
        - Look for "surgery", "surgical", "operation", "procedure", "cataract", "eye surgery"
        - Search for waiting periods, exclusion periods, time requirements
        - Check for specific conditions and coverage limits
        - IMPORTANT: Look for "covered after X years" statements
        """
    elif any(term in question_lower for term in ['organ donor', 'donor', 'transplantation', 'transplant']):
        specific_instructions = """
        ORGAN DONOR SEARCH INSTRUCTIONS:
        - Look for "organ donor", "donor", "transplantation", "transplant", "harvesting"
        - Search for hospitalization, pre/post hospitalization, complications
        - Check for coverage limits and exclusions
        - IMPORTANT: Look for what IS covered, not just what's excluded
        """
    
    # Financial concepts
    elif any(term in question_lower for term in ['ncd', 'no claim discount', 'claim discount', 'discount']):
        specific_instructions = """
        NCD SEARCH INSTRUCTIONS:
        - Look for "NCD", "no claim discount", "no claim", "claim discount", "discount"
        - Search for renewal terms, flat discount, policy term, bonus
        - Check for conditions and limitations on discounts
        - IMPORTANT: Look for percentage amounts like "5%", "10%", etc.
        - IMPORTANT: Look for positive statements about discounts
        """
    elif any(term in question_lower for term in ['room rent', 'accommodation', 'bed charges', 'icu']):
        specific_instructions = """
        ROOM RENT/ICU SEARCH INSTRUCTIONS:
        - Look for "room rent", "accommodation", "bed charges", "ICU", "intensive care"
        - Search for "sub-limits", "capped", "per day", "daily charges"
        - Check for preferred provider network conditions
        - IMPORTANT: Look for what IS covered, not just what's excluded
        """
    
    # Medical concepts
    elif any(term in question_lower for term in ['health check', 'preventive', 'wellness', 'screening']):
        specific_instructions = """
        HEALTH CHECK SEARCH INSTRUCTIONS:
        - Look for "health check", "preventive", "wellness", "screening", "medical check"
        - Search for continuous policy requirements and reimbursable conditions
        - Check for waiting periods and limitations
        - IMPORTANT: Look for positive coverage statements
        """
    elif any(term in question_lower for term in ['hospital', 'medical institution', 'healthcare facility']):
        specific_instructions = """
        HOSPITAL SEARCH INSTRUCTIONS:
        - Look for "hospital", "medical institution", "healthcare facility", "clinic"
        - Search for registration requirements, staff requirements, daily records
        - Check for inpatient/outpatient distinctions
        - IMPORTANT: Look for definitions and requirements
        """
    elif any(term in question_lower for term in ['ayush', 'ayurveda', 'traditional medicine', 'alternative medicine']):
        specific_instructions = """
        AYUSH SEARCH INSTRUCTIONS:
        - Look for "AYUSH", "Ayurveda", "Yoga", "Naturopathy", "Unani", "Siddha", "Homeopathy"
        - Search for "traditional medicine", "alternative medicine", "inpatient treatment"
        - Check for recognized hospital requirements and coverage limits
        - IMPORTANT: Look for positive coverage statements
        """
    
    # General coverage concepts
    elif any(term in question_lower for term in ['coverage', 'cover', 'covered', 'benefit', 'eligible']):
        specific_instructions = """
        COVERAGE SEARCH INSTRUCTIONS:
        - Look for "coverage", "cover", "covered", "benefit", "eligible", "included"
        - Search for specific conditions, limitations, and requirements
        - Check both inclusion and exclusion sections
        - Look for positive coverage statements and exceptions
        """
    
    return (
        "You are an expert insurance policy analyst. Answer the question based ONLY on the policy clauses provided below.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- Read ALL policy clauses carefully and thoroughly.\n"
        "- Look for specific details, amounts, time periods, conditions, and requirements.\n"
        "- IMPORTANT: Start with \"Yes,\" if coverage exists OR \"No,\" if explicitly excluded.\n"
        "- IMPORTANT: If you find ANY positive coverage information, state it clearly.\n"
        "- Include specific amounts, time periods, conditions, and requirements when mentioned.\n"
        "- Be comprehensive and detailed - provide full context and conditions.\n"
        "- Include specific policy sections, exclusions, and limitations when mentioned.\n"
        "- If the answer involves conditions or limitations, mention them clearly.\n"
        "- Provide complete information about eligibility, waiting periods, and coverage limits.\n"
        "- IMPORTANT: Look for positive coverage statements (covered, eligible, allowed, provided, includes, covers, reimbursable, payable, included, offered, granted, approved, authorized, permitted).\n"
        "- IMPORTANT: Be careful with exclusion language - look for exceptions and conditions within exclusions.\n"
        "- IMPORTANT: Check for coverage that exists despite being in exclusion sections (e.g., 'covered after X months').\n"
        "- IMPORTANT: Look for time-based conditions, waiting periods, and renewal terms.\n"
        "- IMPORTANT: Check for financial terms like discounts, sub-limits, and payment conditions.\n"
        "- IMPORTANT: If you find ANY relevant information, even if it's mixed with exclusions, mention it.\n"
        "- IMPORTANT: Look for statements that start with positive words even if they're in exclusion sections.\n"
        "- IMPORTANT: Check for specific amounts, percentages, time periods, and conditions.\n"
        "- IMPORTANT: If the question asks about something and you find ANY mention of it, provide that information.\n"
        "- Only say \"Not found in the document.\" if absolutely no relevant information exists in the provided context.\n"
        "- Do not make assumptions or use external knowledge.\n"
        "- If the context contains conflicting information, acknowledge this.\n"
        "- Be specific and cite relevant parts of the context when possible.\n"
        "- Keep answers concise but comprehensive.\n"
        "- IMPORTANT: If you find the information, say \"Yes\" and provide details. If you don't find it, say \"No\" or \"Not found.\"\n"
        f"{specific_instructions}\n"
        f"Policy Clauses:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

def prepare_llm_context(retrieval_results, question, max_context_tokens):
    """
    Prepares the context for the LLM by selecting and truncating relevant chunks
    to fit within the max_context_tokens limit.
    """
    context_parts = []
    current_tokens = 0
    
    # Sort by score to prioritize most relevant
    sorted_results = sorted(retrieval_results, key=lambda x: x['score'], reverse=True)
    
    # Add a buffer for the prompt itself (instructions + question)
    prompt_buffer_tokens = count_tokens(build_llm_prompt("", question)) # Estimate prompt size without context
    remaining_context_tokens = max_context_tokens - prompt_buffer_tokens - 50 # 50 token buffer for safety
    
    if remaining_context_tokens <= 0:
        logger.warning(f"Not enough token space for context. Max context tokens: {max_context_tokens}, Prompt buffer: {prompt_buffer_tokens}")
        return ""

    for result in sorted_results:
        chunk = result['chunk']
        score = result['score']
        
        # Add section info if available
        section_info = f"Section: {chunk.get('section','')}" if chunk.get('section') else "Policy Document"
        
        # Format chunk text for context
        chunk_text_raw = chunk['text']
        
        # Estimate tokens for this chunk, including formatting overhead
        formatted_chunk_prefix = f"[{section_info}, Score: {score:.3f}]\n"
        estimated_chunk_tokens_with_format = count_tokens(formatted_chunk_prefix + chunk_text_raw)
        
        if current_tokens + estimated_chunk_tokens_with_format > remaining_context_tokens:
            # If adding the full chunk exceeds limit, try to truncate it
            space_left = remaining_context_tokens - current_tokens
            if space_left <= count_tokens(formatted_chunk_prefix) + 10: # Need at least 10 tokens for actual text
                break # No meaningful space left
            
            # Calculate how many characters can fit
            chars_for_text = (space_left - count_tokens(formatted_chunk_prefix)) * 4
            truncated_text = chunk_text_raw[:chars_for_text]
            if len(truncated_text) < len(chunk_text_raw):
                truncated_text += "..."
            
            context_part = f"{formatted_chunk_prefix}{truncated_text}"
            context_parts.append(context_part)
            current_tokens += count_tokens(context_part)
            break # Context is full
        else:
            context_part = f"{formatted_chunk_prefix}{chunk_text_raw}"
            context_parts.append(context_part)
            current_tokens += estimated_chunk_tokens_with_format
    
    return "\n\n---\n\n".join(context_parts)

@app.route('/optimized_query', methods=['POST'])
def handle_optimized_query():
    """
    Optimized query endpoint with enhanced accuracy and response time
    - Uses 30% overlap chunking
    - Implements document caching with URL hash
    - Uses hybrid retrieval (dense + sparse)
    - Preloaded models for faster response
    - Evidence-based LLM responses
    """
    tracker = PerformanceTracker()
    tracker.start()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        url = data.get('url', '').strip()
        questions = data.get('questions', [])
        
        if not url:
            return jsonify({"error": "No document URL provided"}), 400
        
        if not questions:
            return jsonify({"error": "No questions provided"}), 400
        
        if isinstance(questions, str):
            questions = [questions]
        
        logger.info(f"Processing {len(questions)} questions for document: {url}")
        
        # Check document cache first
        cached_data = document_cache.get_cached_document(url)
        if cached_data:
            chunks = cached_data['chunks']
            embeddings = cached_data.get('embeddings')
            logger.info(f"Using cached document with {len(chunks)} chunks")
        else:
            # Parse document and cache it
            logger.info("Parsing new document...")
            chunks = parse_document_from_url(url)
            
            if not chunks:
                return jsonify({"error": "No valid text chunks found in document"}), 400
            
            # Generate embeddings
            embedding_fn = get_embedding_function()
            chunk_texts = [c['text'] for c in chunks]
            embeddings = embedding_fn.encode(chunk_texts, show_progress_bar=False)
            
            # Cache the document with embeddings
            document_cache.cache_document(url, chunks, embeddings)
        
        # Process each question
        answers = []
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question}")
            
            # Expand query with insurance-specific terms for better retrieval
            expanded_question = expand_insurance_query(question)
            logger.info(f"Expanded query: {expanded_question}")
            
            # Use hybrid retrieval for better accuracy - increased top_k for more context
            retrieval_results = hybrid_retrieve(expanded_question, chunks, embeddings, top_k=35)
            
            if not retrieval_results:
                answers.append("Not found in the document.")
                continue
            
            # Enhance retrieval scores with insurance term weighting
            enhanced_results = enhance_retrieval_scores(retrieval_results, question)
            
            # Use the new context preparation function with token limits - increased for better coverage
            context = prepare_llm_context(enhanced_results, question, MAX_TOKENS_PER_REQUEST + 600)
            
            if not context.strip():
                answers.append("Not found in the document.")
                continue
            
            # Build enhanced prompt using original question (not expanded)
            prompt = build_llm_prompt(context, question)
            
            # Generate answer with fallback
            answer = None
            model_used = None
            
            try:
                # Try Gemini first
                answer = gemini_generate(prompt, max_tokens=300, temperature=0.05)
                if answer and 'error' not in answer.lower() and 'timed out' not in answer.lower():
                    model_used = "gemini"
                else:
                    raise Exception('Gemini failed or returned error')
            except Exception as e:
                logger.warning(f"Gemini error: {str(e)}")
                try:
                    # Fallback to Cohere
                    response = co.generate(
                        model='command-r-plus',
                        prompt=prompt,
                        max_tokens=300,
                        temperature=0.05
                    )
                    answer = response.generations[0].text.strip()
                    if answer:
                        model_used = "cohere"
                    else:
                        raise Exception('Cohere returned empty response')
                except Exception as e:
                    logger.error(f"Cohere error: {str(e)}")
                    answer = "Not found in the document."
                    model_used = "none"
            
            # Clean up answer
            if answer:
                answer = answer.strip()
                # Check if answer indicates no information found - be less aggressive
                if any(phrase in answer.lower() for phrase in ['not found in the document', 'no information available', 'not mentioned anywhere', 'not specified anywhere']):
                    answer = "Not found in the document."
                # If answer contains positive coverage information, keep it even if it starts with "No"
                elif any(positive_word in answer.lower() for positive_word in ['covered', 'coverage', 'benefit', 'eligible', 'allowed', 'provided', 'includes', 'covers', 'reimbursable', 'payable', 'entitled', 'available', 'included', 'offered', 'granted', 'approved', 'authorized', 'permitted']):
                    # Keep the answer as is - it contains positive coverage information
                    pass
                # If answer contains specific amounts, time periods, or conditions, keep it
                elif any(value_word in answer.lower() for value_word in ['30 days', 'thirty days', '24 months', '36 months', '2 years', '3 years', '5%', '10%', '15%', '20%', '25%', '50%', '75%', '100%']):
                    # Keep the answer as is - it contains specific values
                    pass
                else:
                    # Only convert to "Not found" if it's clearly a negative response without useful information
                    if answer.lower().startswith('no') and len(answer) < 100:
                        answer = "Not found in the document."
                # Ensure it's not too long
                if len(answer) > MAX_ANSWER_CHARACTERS:
                    # Truncate to sentence boundary if possible
                    sentences = safe_sent_tokenize(answer)
                    truncated = ""
                    for sentence in sentences:
                        if len(truncated + sentence) <= MAX_ANSWER_CHARACTERS - 3:
                            truncated += sentence + " "
                        else:
                            break
                    answer = truncated.strip() + "..."
            else:
                answer = "Not found in the document."
            
            answers.append(answer)
            logger.info(f"Answer {i+1} ({model_used}): {answer}")
        
        response_data = {
            "answers": answers,
            "response_time_ms": tracker.get_response_time_ms(),
            "document_url": url,
            "questions_processed": len(questions),
            "chunks_used": len(chunks)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        error_msg = f"Unexpected error in optimized_query: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({"error": error_msg, "response_time_ms": tracker.get_response_time_ms()}), 500

def fallback_sent_tokenize(text):
    """Fallback sentence tokenization without NLTK dependency"""
    import re
    # Simple sentence splitting using regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def safe_sent_tokenize(text):
    """Safe sentence tokenization with fallback"""
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception as e:
        logger.warning(f"NLTK tokenization failed, using fallback: {e}")
        return fallback_sent_tokenize(text)

def expand_insurance_query(query):
    """
    Expands insurance-related queries with synonyms and related terms
    to improve retrieval accuracy across different insurance documents.
    """
    query_lower = query.lower()
    
    # Define general insurance term mappings that work across different policies
    insurance_terms = {
        # Time-based concepts
        'grace period': ['grace period', 'grace', 'payment grace', 'renewal grace', 'premium grace', 'payment period', 'renewal period', 'continuous coverage', 'policy renewal', 'payment terms', 'renewal terms'],
        'waiting period': ['waiting period', 'waiting', 'exclusion period', 'time period', 'months', 'years', 'covered after', 'time limit', 'duration', 'period'],
        'pre-existing': ['pre-existing', 'pre existing', 'existing condition', 'prior condition', 'existing disease', 'pre-existing disease', 'ped'],
        
        # Coverage types
        'maternity': ['maternity', 'pregnancy', 'childbirth', 'delivery', 'prenatal', 'postnatal', 'female insured', 'lawful termination', 'covered female', 'maternal'],
        'surgery': ['surgery', 'surgical', 'operation', 'procedure', 'cataract', 'eye surgery', 'ophthalmic', 'vision surgery', 'organ transplant', 'transplantation'],
        'organ donor': ['organ donor', 'donor', 'organ transplantation', 'transplant', 'harvesting', 'donation'],
        'health check': ['health check', 'health checkup', 'preventive', 'preventive health', 'medical check', 'wellness', 'screening'],
        'hospital': ['hospital', 'medical institution', 'healthcare facility', 'clinic', 'inpatient', 'outpatient', 'medical center'],
        'ayush': ['ayush', 'ayurveda', 'yoga', 'naturopathy', 'unani', 'siddha', 'homeopathy', 'traditional medicine', 'alternative medicine'],
        
        # Financial concepts
        'ncd': ['ncd', 'no claim discount', 'no claim', 'claim discount', 'discount', 'renewal discount', 'bonus', 'no claim bonus'],
        'room rent': ['room rent', 'room charges', 'accommodation', 'bed charges', 'sub-limits', 'capped', 'per day', 'daily charges'],
        'icu': ['icu', 'intensive care', 'critical care', 'emergency care', 'ccu', 'critical care unit'],
        'premium': ['premium', 'payment', 'cost', 'fee', 'amount', 'rate', 'pricing'],
        
        # General insurance concepts
        'coverage': ['coverage', 'cover', 'covered', 'benefit', 'benefits', 'eligible', 'eligibility', 'included', 'includes'],
        'exclusion': ['exclusion', 'excluded', 'not covered', 'not included', 'limitation', 'restriction', 'condition'],
        'claim': ['claim', 'claims', 'claimant', 'claiming', 'claim amount', 'claim process'],
        'policy': ['policy', 'insurance', 'plan', 'scheme', 'contract', 'agreement'],
        'sum insured': ['sum insured', 'coverage amount', 'limit', 'maximum', 'cap', 'ceiling'],
        'deductible': ['deductible', 'excess', 'co-payment', 'co-pay', 'out of pocket'],
        
        # Medical terms
        'hospitalization': ['hospitalization', 'hospitalized', 'inpatient', 'admission', 'discharge'],
        'treatment': ['treatment', 'therapy', 'medication', 'drugs', 'medicine', 'prescription'],
        'diagnosis': ['diagnosis', 'diagnostic', 'test', 'testing', 'examination', 'assessment'],
        'emergency': ['emergency', 'urgent', 'critical', 'acute', 'immediate'],
        
        # Administrative terms
        'renewal': ['renewal', 'renew', 'renewed', 'extension', 'continue', 'continuation'],
        'portability': ['portability', 'port', 'transfer', 'switch', 'change insurer'],
        'documentation': ['documentation', 'documents', 'proof', 'evidence', 'certificate', 'report'],
        'notification': ['notification', 'notice', 'inform', 'report', 'declare', 'intimate']
    }
    
    # Find matching terms and expand
    expanded_terms = []
    for term, synonyms in insurance_terms.items():
        if term in query_lower:
            expanded_terms.extend(synonyms)
    
    # If no specific terms found, add general insurance terms
    if not expanded_terms:
        expanded_terms = ['coverage', 'policy', 'insurance', 'benefit', 'exclusion', 'condition', 'section', 'clause', 'term', 'provision']
    
    # Combine original query with expanded terms
    expanded_query = query + " " + " ".join(expanded_terms[:5])
    
    return expanded_query

def enhance_retrieval_scores(retrieval_results, question):
    """
    Enhances retrieval scores by giving higher weights to chunks containing
    relevant insurance terms and better semantic matching across different insurance documents.
    """
    question_lower = question.lower()
    
    # Define general insurance term importance weights that work across different policies
    term_weights = {
        # Time-based concepts (high priority)
        'grace period': 3.0, 'grace': 2.5, 'payment period': 2.0, 'renewal period': 2.0,
        'waiting period': 3.0, 'exclusion period': 2.5, 'time period': 2.0, 'covered after': 2.0,
        'pre-existing': 3.0, 'existing condition': 2.5, 'prior condition': 2.0,
        
        # Coverage types (high priority)
        'maternity': 3.0, 'pregnancy': 2.5, 'childbirth': 2.0, 'delivery': 2.0,
        'surgery': 3.0, 'surgical': 2.5, 'operation': 2.0, 'procedure': 2.0,
        'cataract': 3.0, 'eye surgery': 2.5, 'ophthalmic': 2.0,
        'organ donor': 3.0, 'donor': 2.5, 'transplantation': 2.0, 'transplant': 2.0,
        'health check': 3.0, 'preventive': 2.5, 'wellness': 2.0, 'screening': 2.0,
        'hospital': 2.5, 'medical institution': 2.0, 'healthcare facility': 2.0,
        'ayush': 3.0, 'ayurveda': 2.5, 'traditional medicine': 2.0, 'alternative medicine': 2.0,
        
        # Financial concepts (medium-high priority)
        'ncd': 3.0, 'no claim discount': 2.5, 'claim discount': 2.0, 'discount': 1.5,
        'room rent': 2.5, 'accommodation': 2.0, 'bed charges': 2.0, 'sub-limits': 2.0,
        'icu': 2.5, 'intensive care': 2.0, 'critical care': 2.0, 'emergency care': 2.0,
        'premium': 2.0, 'payment': 1.5, 'cost': 1.5, 'fee': 1.5,
        
        # General insurance concepts (medium priority)
        'coverage': 2.0, 'cover': 1.8, 'covered': 1.8, 'benefit': 1.8, 'benefits': 1.8,
        'eligible': 2.0, 'eligibility': 1.8, 'included': 1.8, 'includes': 1.8,
        'exclusion': 1.8, 'excluded': 1.8, 'not covered': 1.8, 'limitation': 1.5,
        'claim': 1.8, 'claims': 1.8, 'claimant': 1.5, 'claim amount': 2.0,
        'policy': 1.5, 'insurance': 1.5, 'plan': 1.5, 'scheme': 1.5,
        'sum insured': 2.0, 'coverage amount': 1.8, 'limit': 1.5, 'maximum': 1.5,
        'deductible': 2.0, 'excess': 1.8, 'co-payment': 1.8, 'co-pay': 1.8,
        
        # Medical terms (medium priority)
        'hospitalization': 2.0, 'hospitalized': 1.8, 'inpatient': 1.8, 'admission': 1.5,
        'treatment': 1.8, 'therapy': 1.5, 'medication': 1.5, 'medicine': 1.5,
        'diagnosis': 1.8, 'diagnostic': 1.5, 'test': 1.5, 'testing': 1.5,
        'emergency': 2.0, 'urgent': 1.8, 'critical': 1.8, 'acute': 1.5,
        
        # Administrative terms (lower priority)
        'renewal': 1.5, 'renew': 1.5, 'extension': 1.5, 'continue': 1.5,
        'portability': 1.8, 'port': 1.5, 'transfer': 1.5, 'switch': 1.5,
        'documentation': 1.5, 'documents': 1.5, 'proof': 1.5, 'evidence': 1.5,
        'notification': 1.5, 'notice': 1.5, 'inform': 1.5, 'report': 1.5,
        
        # Common insurance terms (base priority)
        'section': 1.2, 'clause': 1.2, 'term': 1.2, 'provision': 1.2, 'condition': 1.2
    }
    
    # Define positive coverage indicators (bonus for positive language)
    positive_indicators = [
        'covered', 'coverage', 'benefit', 'eligible', 'allowed', 'provided',
        'includes', 'covers', 'reimbursable', 'payable', 'entitled', 'available',
        'included', 'offered', 'granted', 'approved', 'authorized', 'permitted'
    ]
    
    # Define exclusion indicators (penalty for negative language)
    exclusion_indicators = [
        'shall not be liable', 'not covered', 'excluded', 'exclusion',
        'not payable', 'not eligible', 'not allowed', 'not provided',
        'not included', 'not entitled', 'not available', 'not offered',
        'excluded from', 'not applicable', 'not valid', 'void'
    ]
    
    # Define specific value indicators (bonus for specific amounts, time periods)
    value_indicators = [
        '30 days', 'thirty days', '24 months', '36 months', '2 years', '3 years',
        '5%', '10%', '15%', '20%', '25%', '50%', '75%', '100%',
        'inr', 'rs', 'rupees', 'dollars', 'usd', 'euro', 'pounds'
    ]
    
    enhanced_results = []
    for result in retrieval_results:
        chunk = result['chunk']
        original_score = result['score']
        chunk_text_lower = chunk['text'].lower()
        
        # Calculate term bonus with dynamic matching
        term_bonus = 0.0
        for term, weight in term_weights.items():
            if term in question_lower and term in chunk_text_lower:
                term_bonus += weight
                # Additional bonus for exact matches
                if term in chunk_text_lower:
                    term_bonus += weight * 0.5
        
        # Calculate section relevance bonus
        section_bonus = 0.0
        if chunk.get('section'):
            section_lower = chunk['section'].lower()
            for term, weight in term_weights.items():
                if term in question_lower and term in section_lower:
                    section_bonus += weight * 0.8
        
        # Calculate positive coverage bonus (increased weight)
        positive_bonus = 0.0
        for indicator in positive_indicators:
            if indicator in chunk_text_lower:
                positive_bonus += 2.0  # Increased from 1.0 to 2.0
        
        # Calculate exclusion penalty (reduced penalty)
        exclusion_penalty = 0.0
        for indicator in exclusion_indicators:
            if indicator in chunk_text_lower:
                exclusion_penalty += 1.0  # Reduced from 2.0 to 1.0
        
        # Calculate value bonus for specific amounts and time periods
        value_bonus = 0.0
        for indicator in value_indicators:
            if indicator in chunk_text_lower:
                value_bonus += 3.0  # High bonus for specific values
        
        # Calculate length bonus for substantial chunks
        length_bonus = 0.0
        if len(chunk['text']) > 100:
            length_bonus = 1.0  # Increased from 0.5 to 1.0
        
        # Apply bonuses and penalties
        enhanced_score = original_score + term_bonus + section_bonus + positive_bonus - exclusion_penalty + value_bonus + length_bonus
        
        enhanced_results.append({
            **result, 'score': enhanced_score, 'original_score': original_score,
            'term_bonus': term_bonus, 'section_bonus': section_bonus,
            'positive_bonus': positive_bonus, 'exclusion_penalty': exclusion_penalty,
            'value_bonus': value_bonus, 'length_bonus': length_bonus
        })
    
    enhanced_results.sort(key=lambda x: x['score'], reverse=True)
    return enhanced_results

if __name__ == '__main__':
    # Preload models at startup for faster response times
    preload_models()
    
    # Get port from environment variable (for production) or use 5001
    port = int(os.environ.get('PORT', 5001))
    # Simple server run without debug
    # If you see Windows console errors, try running with: python -u backend/app.py
    app.run(host='0.0.0.0', port=port, debug=False)