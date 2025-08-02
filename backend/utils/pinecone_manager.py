"""
Pinecone Vector Database Manager for High-Performance Document Retrieval
"""

import pinecone
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from config import (
    PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME,
    PINECONE_DIMENSION, PINECONE_METRIC, PINECONE_TOP_K,
    PINECONE_FILTER_THRESHOLD, PINECONE_NAMESPACE_PREFIX,
    ENABLE_PINECONE_HYBRID_SEARCH, PINECONE_SPARSE_WEIGHT, PINECONE_DENSE_WEIGHT,
    EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)

class PineconeManager:
    def __init__(self):
        self.pinecone = None
        self.index = None
        self.embedding_model = None
        self._initialize_pinecone()
        self._load_embedding_model()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            if not PINECONE_API_KEY or PINECONE_API_KEY == "YOUR_PINECONE_API_KEY":
                logger.warning("Pinecone API key not configured, using fallback")
                return
            
            # New Pinecone SDK initialization
            from pinecone import Pinecone
            self.pinecone = Pinecone(api_key=PINECONE_API_KEY)
            
            # Check if index exists, create if not
            existing_indexes = self.pinecone.list_indexes()
            if PINECONE_INDEX_NAME not in existing_indexes.names():
                logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
                self.pinecone.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=PINECONE_DIMENSION,
                    metric=PINECONE_METRIC
                )
            
            self.index = self.pinecone.Index(PINECONE_INDEX_NAME)
            logger.info(f"Pinecone initialized successfully: {PINECONE_INDEX_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self.index = None
    
    def _load_embedding_model(self):
        """Load the embedding model"""
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Embedding model loaded: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def _generate_namespace(self, document_url: str) -> str:
        """Generate namespace from document URL"""
        url_hash = hashlib.md5(document_url.encode()).hexdigest()[:8]
        return f"{PINECONE_NAMESPACE_PREFIX}{url_hash}"
    
    def _generate_id(self, chunk_index: int, document_url: str) -> str:
        """Generate unique ID for chunk"""
        url_hash = hashlib.md5(document_url.encode()).hexdigest()[:8]
        return f"{url_hash}_{chunk_index}"
    
    def store_document_chunks(self, chunks: List[Dict], document_url: str) -> bool:
        """Store document chunks in Pinecone"""
        if not self.index or not self.embedding_model:
            logger.error("Pinecone or embedding model not available")
            return False
        
        try:
            namespace = self._generate_namespace(document_url)
            
            # Prepare vectors for batch upsert
            vectors = []
            for i, chunk in enumerate(chunks):
                if not chunk.get('text', '').strip():
                    continue
                
                # Generate embedding
                embedding = self.embedding_model.encode([chunk['text']])[0].tolist()
                
                # Prepare metadata
                metadata = {
                    'text': chunk['text'][:1000],  # Limit text length
                    'section': chunk.get('section', ''),
                    'type': chunk.get('type', ''),
                    'document_url': document_url,
                    'chunk_index': i,
                    'length': len(chunk['text'])
                }
                
                # Generate unique ID
                vector_id = self._generate_id(i, document_url)
                
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            if vectors:
                # Batch upsert to Pinecone
                self.index.upsert(vectors=vectors, namespace=namespace)
                logger.info(f"Stored {len(vectors)} chunks in Pinecone namespace: {namespace}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to store chunks in Pinecone: {e}")
            return False
    
    def search_similar_chunks(self, query: str, document_url: str, top_k: int = None) -> List[Dict]:
        """Search for similar chunks using Pinecone"""
        if not self.index or not self.embedding_model:
            logger.error("Pinecone or embedding model not available")
            return []
        
        try:
            if top_k is None:
                top_k = PINECONE_TOP_K
            
            namespace = self._generate_namespace(document_url)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter={"document_url": document_url}
            )
            
            # Process results
            chunks = []
            for match in results.matches:
                if match.score >= PINECONE_FILTER_THRESHOLD:
                    chunks.append({
                        'text': match.metadata.get('text', ''),
                        'section': match.metadata.get('section', ''),
                        'type': match.metadata.get('type', ''),
                        'similarity_score': match.score,
                        'metadata': match.metadata
                    })
            
            logger.info(f"Found {len(chunks)} relevant chunks for query: {query[:50]}...")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to search Pinecone: {e}")
            return []
    
    def hybrid_search(self, query: str, document_url: str, top_k: int = None) -> List[Dict]:
        """Perform hybrid search (dense + sparse) for better accuracy"""
        if not ENABLE_PINECONE_HYBRID_SEARCH:
            return self.search_similar_chunks(query, document_url, top_k)
        
        try:
            # Dense search
            dense_results = self.search_similar_chunks(query, document_url, top_k)
            
            # Sparse search (keyword-based)
            sparse_results = self._sparse_search(query, document_url, top_k)
            
            # Combine and re-rank results
            combined_results = self._combine_search_results(dense_results, sparse_results)
            
            return combined_results[:top_k or PINECONE_TOP_K]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return self.search_similar_chunks(query, document_url, top_k)
    
    def _sparse_search(self, query: str, document_url: str, top_k: int) -> List[Dict]:
        """Simple keyword-based sparse search"""
        # This is a simplified sparse search - in production, you'd use BM25 or similar
        try:
            namespace = self._generate_namespace(document_url)
            
            # Get all vectors from namespace (for keyword matching)
            # Note: This is simplified - in production, use proper sparse retrieval
            results = self.index.query(
                vector=[0] * PINECONE_DIMENSION,  # Dummy vector
                top_k=100,
                namespace=namespace,
                include_metadata=True,
                filter={"document_url": document_url}
            )
            
            # Keyword matching
            query_words = set(query.lower().split())
            keyword_matches = []
            
            for match in results.matches:
                text = match.metadata.get('text', '').lower()
                matched_words = query_words.intersection(set(text.split()))
                
                if matched_words:
                    keyword_matches.append({
                        'text': match.metadata.get('text', ''),
                        'section': match.metadata.get('section', ''),
                        'type': match.metadata.get('type', ''),
                        'keyword_score': len(matched_words) / len(query_words),
                        'metadata': match.metadata
                    })
            
            return sorted(keyword_matches, key=lambda x: x['keyword_score'], reverse=True)[:top_k]
            
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []
    
    def _combine_search_results(self, dense_results: List[Dict], sparse_results: List[Dict]) -> List[Dict]:
        """Combine and re-rank dense and sparse search results"""
        combined = {}
        
        # Add dense results
        for result in dense_results:
            text_key = result['text'][:100]  # Use first 100 chars as key
            combined[text_key] = {
                **result,
                'dense_score': result.get('similarity_score', 0),
                'sparse_score': 0,
                'combined_score': result.get('similarity_score', 0) * PINECONE_DENSE_WEIGHT
            }
        
        # Add sparse results
        for result in sparse_results:
            text_key = result['text'][:100]
            if text_key in combined:
                combined[text_key]['sparse_score'] = result.get('keyword_score', 0)
                combined[text_key]['combined_score'] += result.get('keyword_score', 0) * PINECONE_SPARSE_WEIGHT
            else:
                combined[text_key] = {
                    **result,
                    'dense_score': 0,
                    'sparse_score': result.get('keyword_score', 0),
                    'combined_score': result.get('keyword_score', 0) * PINECONE_SPARSE_WEIGHT
                }
        
        # Sort by combined score
        return sorted(combined.values(), key=lambda x: x['combined_score'], reverse=True)
    
    def delete_document(self, document_url: str) -> bool:
        """Delete all chunks for a document"""
        if not self.index:
            return False
        
        try:
            namespace = self._generate_namespace(document_url)
            self.index.delete(namespace=namespace)
            logger.info(f"Deleted document from Pinecone: {document_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document from Pinecone: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        if not self.index:
            return {"status": "not_initialized"}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "status": "active",
                "total_vector_count": stats.total_vector_count,
                "namespaces": stats.namespaces,
                "dimension": stats.dimension
            }
        except Exception as e:
            logger.error(f"Failed to get Pinecone stats: {e}")
            return {"status": "error", "message": str(e)}

# Global Pinecone manager instance
pinecone_manager = PineconeManager() 