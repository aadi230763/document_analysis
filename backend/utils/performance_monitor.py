"""
Performance monitoring module for tracking response times and accuracy metrics
"""

import time
import json
from datetime import datetime, timezone
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'response_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': {
                'gemini': 0,
                'cohere': 0,
                'fallback': 0
            },
            'errors': [],
            'accuracy_scores': []
        }
        self.start_time = time.time()
    
    def record_response_time(self, endpoint: str, response_time_ms: int):
        """Record response time for an endpoint"""
        self.metrics['response_times'].append({
            'endpoint': endpoint,
            'response_time_ms': response_time_ms,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        # Keep only last 1000 entries
        if len(self.metrics['response_times']) > 1000:
            self.metrics['response_times'] = self.metrics['response_times'][-1000:]
    
    def record_cache_hit(self):
        """Record a cache hit"""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record a cache miss"""
        self.metrics['cache_misses'] += 1
    
    def record_api_call(self, api_name: str):
        """Record an API call"""
        if api_name in self.metrics['api_calls']:
            self.metrics['api_calls'][api_name] += 1
    
    def record_error(self, error_type: str, error_message: str):
        """Record an error"""
        self.metrics['errors'].append({
            'type': error_type,
            'message': error_message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        # Keep only last 100 errors
        if len(self.metrics['errors']) > 100:
            self.metrics['errors'] = self.metrics['errors'][-100:]
    
    def record_accuracy_score(self, score: float, question: str, answer: str):
        """Record accuracy score for a question-answer pair"""
        self.metrics['accuracy_scores'].append({
            'score': score,
            'question': question,
            'answer': answer,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        # Keep only last 500 scores
        if len(self.metrics['accuracy_scores']) > 500:
            self.metrics['accuracy_scores'] = self.metrics['accuracy_scores'][-500:]
    
    def get_average_response_time(self, endpoint: str = None) -> float:
        """Get average response time for an endpoint or overall"""
        if not self.metrics['response_times']:
            return 0.0
        
        if endpoint:
            times = [rt['response_time_ms'] for rt in self.metrics['response_times'] 
                    if rt['endpoint'] == endpoint]
        else:
            times = [rt['response_time_ms'] for rt in self.metrics['response_times']]
        
        return sum(times) / len(times) if times else 0.0
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.metrics['cache_hits'] + self.metrics['cache_misses']
        return self.metrics['cache_hits'] / total if total > 0 else 0.0
    
    def get_average_accuracy(self) -> float:
        """Get average accuracy score"""
        if not self.metrics['accuracy_scores']:
            return 0.0
        
        scores = [score['score'] for score in self.metrics['accuracy_scores']]
        return sum(scores) / len(scores)
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'uptime_seconds': self.get_uptime(),
            'average_response_time_ms': self.get_average_response_time(),
            'cache_hit_rate': self.get_cache_hit_rate(),
            'average_accuracy': self.get_average_accuracy(),
            'total_requests': len(self.metrics['response_times']),
            'api_calls': self.metrics['api_calls'],
            'cache_hits': self.metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'],
            'total_errors': len(self.metrics['errors'])
        }
    
    def save_metrics(self, filepath: str = "performance_metrics.json"):
        """Save metrics to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"Performance metrics saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
    
    def load_metrics(self, filepath: str = "performance_metrics.json"):
        """Load metrics from file"""
        try:
            with open(filepath, 'r') as f:
                self.metrics = json.load(f)
            logger.info(f"Performance metrics loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading performance metrics: {e}")

# Global performance monitor instance
performance_monitor = PerformanceMonitor() 