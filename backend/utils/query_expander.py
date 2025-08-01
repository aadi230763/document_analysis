import requests
from config import GEMINI_API_KEY, GEMINI_API_URL
import json
from .medical_terms import get_expanded_terms, extract_medical_entities, parse_demographics, extract_policy_duration

# Cache for query expansions
EXPANSION_CACHE = {}

class QueryExpander:
    def __init__(self, llm=None):
        self.llm = llm

    def _get_cached_expansion(self, query):
        """Get cached expansion if available"""
        return EXPANSION_CACHE.get(query.lower())

    def _cache_expansion(self, query, expansion):
        """Cache the expansion result"""
        EXPANSION_CACHE[query.lower()] = expansion

    def _create_enhanced_prompt(self, query):
        """Create enhanced prompt with medical terms and context"""
        # Extract medical entities
        medical_entities = extract_medical_entities(query)
        expanded_terms = get_expanded_terms(query)
        
        # Build context from medical terms
        medical_context = ""
        if medical_entities["procedures"]:
            medical_context += f"Procedures mentioned: {', '.join(medical_entities['procedures'])}\n"
        if medical_entities["conditions"]:
            medical_context += f"Conditions mentioned: {', '.join(medical_entities['conditions'])}\n"
        if expanded_terms:
            medical_context += f"Related terms: {', '.join(expanded_terms[:5])}\n"
        
        prompt = f'''
You are an expert insurance query understanding agent. Given the user query below, extract:
- Demographics (age, gender, location)
- Expansions (3-5 alternative phrasings or related terms)
- Inferred intent (e.g., "Check coverage under Section 4.2 (Surgical Benefits)")
- Policy duration in days (if mentioned)

Medical Context:
{medical_context}

Return a JSON like:
{{
  "demographics": {{"age": ..., "gender": ..., "location": ...}},
  "expansions": [...],
  "inferred_intent": "...",
  "duration_days": ...,
  "medical_entities": {medical_entities}
}}

Query: "{query}"
JSON: '''
        return prompt

    def mistral_expand(self, query: str) -> dict:
        if self.llm is None:
            # Enhanced fallback with medical terms
            return self._fallback_expansion(query)
        
        prompt = self._create_enhanced_prompt(query)
        
        try:
            output = self.llm(prompt, max_tokens=256)
            text = output['choices'][0]['text'] if isinstance(output, dict) and 'choices' in output else str(output)
            
            # Clean up response
            if text.strip().startswith("```"):
                lines = text.strip().splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            
            try:
                parsed = json.loads(text)
                # Validate and enhance with medical terms
                parsed = self._validate_and_enhance_expansion(parsed, query)
                return parsed
            except json.JSONDecodeError as e:
                print(f"Mistral JSON parse error: {e}")
                return self._fallback_expansion(query)
                
        except Exception as e:
            print(f"Mistral expansion error: {str(e)}")
            return self._fallback_expansion(query)

    def gemini_expand(self, query: str) -> dict:
        prompt = self._create_enhanced_prompt(query)
        
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 256, "temperature": 0.3}
        }
        
        try:
            from config import GEMINI_TIMEOUT
            response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=data, timeout=GEMINI_TIMEOUT)
            response.raise_for_status()
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"] if "candidates" in result and result["candidates"] else ""
            
            # Clean up response
            if text.strip().startswith("```"):
                lines = text.strip().splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            
            try:
                parsed = json.loads(text)
                # Validate and enhance with medical terms
                parsed = self._validate_and_enhance_expansion(parsed, query)
                return parsed
            except json.JSONDecodeError as e:
                print(f"Gemini JSON parse error: {e}")
                return self._fallback_expansion(query)
                
        except Exception as e:
            print(f"Gemini expansion error: {str(e)}")
            # Fallback to Mistral
            return self.mistral_expand(query)

    def _fallback_expansion(self, query: str) -> dict:
        """Enhanced fallback expansion using medical terms and basic parsing"""
        # Extract basic information
        demographics = parse_demographics(query)
        duration_days = extract_policy_duration(query)
        medical_entities = extract_medical_entities(query)
        expanded_terms = get_expanded_terms(query)
        
        # Create basic expansions
        expansions = [query]
        if expanded_terms:
            # Add variations with medical terms
            for term in expanded_terms[:3]:
                if term not in query.lower():
                    expansions.append(query.replace(query.split()[0], term))
        
        return {
            "demographics": demographics,
            "expansions": expansions,
            "inferred_intent": f"Check coverage for {', '.join(medical_entities.get('procedures', []) or medical_entities.get('conditions', []) or ['medical treatment'])}",
            "duration_days": duration_days,
            "medical_entities": medical_entities
        }

    def _validate_and_enhance_expansion(self, parsed: dict, query: str) -> dict:
        """Validate and enhance the parsed expansion"""
        # Ensure required fields exist
        if "demographics" not in parsed:
            parsed["demographics"] = parse_demographics(query)
        
        if "expansions" not in parsed or not parsed["expansions"]:
            parsed["expansions"] = [query]
        
        if "inferred_intent" not in parsed:
            medical_entities = extract_medical_entities(query)
            parsed["inferred_intent"] = f"Check coverage for {', '.join(medical_entities.get('procedures', []) or medical_entities.get('conditions', []) or ['medical treatment'])}"
        
        if "duration_days" not in parsed:
            parsed["duration_days"] = extract_policy_duration(query)
        
        if "medical_entities" not in parsed:
            parsed["medical_entities"] = extract_medical_entities(query)
        
        # Ensure expansions include original query
        if query not in parsed["expansions"]:
            parsed["expansions"].insert(0, query)
        
        return parsed

    def expand(self, query: str) -> dict:
        """Main expansion method with caching"""
        # Check cache first
        cached = self._get_cached_expansion(query)
        if cached:
            return cached
        
        # Try Gemini first, then fallback to Mistral
        try:
            result = self.gemini_expand(query)
        except Exception as e:
            print(f"Gemini expansion failed, using Mistral: {str(e)}")
            result = self.mistral_expand(query)
        
        # Cache the result
        self._cache_expansion(query, result)
        return result
