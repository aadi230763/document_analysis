import re
import requests
import json
from config import LLM_MODEL, GEMINI_API_KEY, GEMINI_API_URL

# Enhanced medical synonyms with more comprehensive coverage
MEDICAL_SYNONYMS = {
    "knee surgery": ["orthopedic procedure", "joint surgery", "arthroscopy", "knee replacement", "ACL reconstruction"],
    "heart attack": ["myocardial infarction", "cardiac arrest", "coronary event", "heart failure", "cardiac episode"],
    "cancer": ["malignancy", "tumor", "carcinoma", "neoplasm", "oncology treatment"],
    "hospitalization": ["inpatient care", "admission", "hospital stay", "institutional care", "medical confinement"],
    "maternity": ["pregnancy", "childbirth", "delivery", "obstetric care", "prenatal care"],
    "surgery": ["procedure", "operation", "intervention", "surgical treatment", "medical procedure"],
    "knee": ["joint", "articular", "patellar", "knee joint", "lower extremity"],
    "cataract": ["eye surgery", "ophthalmic procedure", "lens replacement", "vision correction"],
    "dental": ["oral care", "dental procedure", "tooth treatment", "dental surgery"],
    "pre-existing": ["existing condition", "prior illness", "pre-existing disease", "chronic condition"],
    "waiting period": ["exclusion period", "waiting time", "coverage delay", "benefit waiting"],
    "network hospital": ["empaneled hospital", "preferred provider", "network facility", "approved hospital"],
    "room rent": ["accommodation charges", "room charges", "hospital accommodation", "bed charges"],
    "ICU": ["intensive care", "critical care", "ICU charges", "intensive care unit"],
    "preventive": ["preventive care", "health checkup", "screening", "preventive health"],
    "AYUSH": ["alternative medicine", "traditional medicine", "ayurveda", "yoga", "naturopathy"],
    "organ donor": ["organ donation", "donor expenses", "transplant donor", "donor care"],
    "grace period": ["payment extension", "premium grace", "payment window", "renewal grace"],
    "no claim discount": ["NCD", "claim free discount", "no claim bonus", "loyalty discount"],
    "sum insured": ["coverage amount", "policy limit", "maximum benefit", "coverage limit"],
    "dependent": ["dependents", "family member", "covered person", "beneficiary", "insured family"],
    "parents": ["parent", "father", "mother", "parents-in-law", "elderly dependent"],
    "spouse": ["husband", "wife", "partner", "married partner", "life partner"],
    "children": ["child", "son", "daughter", "kids", "offspring", "minor dependent"],
    "family": ["family members", "household", "relatives", "kin", "family unit"],
    "definition": ["means", "shall mean", "defined as", "refers to", "includes", "covers"],
    "coverage": ["covered", "benefits", "protection", "insurance", "policy benefits"],
    "exclusion": ["excluded", "not covered", "exceptions", "limitations", "restrictions"],
    "grace period": ["payment grace", "premium grace", "grace days", "payment extension", "renewal grace", "thirty days", "30 days"],
    "premium payment": ["premium", "payment", "due date", "payment due", "premium due", "renewal payment"],
    "policy renewal": ["renewal", "renew", "continue policy", "policy continuation", "renewal date"],
    "continuity benefits": ["continuity", "continuous coverage", "uninterrupted coverage", "benefit continuity"],
    "thirty days": ["30 days", "grace period", "payment grace", "thirty day", "one month"],
    "due date": ["payment due", "premium due", "renewal due", "expiry date", "maturity date"]
}

# Cache for dynamic synonyms
SYNONYM_CACHE = {}

def gemini_generate(prompt, max_tokens=64, temperature=0.3):
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
        return text
    except Exception as e:
        print(f"Gemini API error in medical terms: {str(e)}")
        return ""

def get_dynamic_synonyms(term, use_cache=True):
    """Enhanced dynamic synonym generation with caching"""
    if use_cache and term in SYNONYM_CACHE:
        return SYNONYM_CACHE[term]
    
    # First check static synonyms
    if term.lower() in MEDICAL_SYNONYMS:
        synonyms = MEDICAL_SYNONYMS[term.lower()]
        if use_cache:
            SYNONYM_CACHE[term] = synonyms
        return synonyms
    
    # Generate dynamic synonyms
    prompt = f"""You are a medical insurance expert. List 5-8 synonyms or related terms for: '{term}'.
Focus on insurance policy terminology and medical terms that would appear in health insurance documents.
Return only the terms, separated by commas:"""
    
    try:
        text = gemini_generate(prompt, max_tokens=128, temperature=0.3)
        if text:
            synonyms = [t.strip() for t in text.split(',') if t.strip()]
            # Add original term to synonyms
            synonyms = [term] + synonyms
            if use_cache:
                SYNONYM_CACHE[term] = synonyms
            return synonyms
    except Exception as e:
        print(f"Error generating dynamic synonyms for '{term}': {str(e)}")
    
    # Fallback to original term
    return [term]

def get_expanded_terms(query):
    """Get all possible variations of medical terms in a query"""
    expanded_terms = []
    query_lower = query.lower()
    
    # Check for known medical terms
    for term, synonyms in MEDICAL_SYNONYMS.items():
        if term in query_lower:
            expanded_terms.extend(synonyms)
    
    # Extract potential medical terms and get dynamic synonyms
    medical_keywords = ["surgery", "treatment", "procedure", "care", "therapy", "diagnosis", "test"]
    words = query_lower.split()
    for word in words:
        if any(keyword in word for keyword in medical_keywords):
            dynamic_synonyms = get_dynamic_synonyms(word, use_cache=True)
            expanded_terms.extend(dynamic_synonyms)
    
    return list(set(expanded_terms))  # Remove duplicates

# Demographic parsing: e.g. "46M Pune" or "32F Mumbai"
def parse_demographics(query):
    match = re.search(r"(\d{1,3})([MF])\s*([A-Za-z ]+)?", query)
    if match:
        age = int(match.group(1))
        gender = match.group(2)
        location = match.group(3).strip() if match.group(3) else None
        return {"age": age, "gender": gender, "location": location}
    return {}

# Policy duration extraction: e.g. "3-month", "2 years", "90 days"
def extract_policy_duration(query):
    patterns = [
        (r"(\d+)\s*-?\s*month", lambda m: int(m.group(1)) * 30),
        (r"(\d+)\s*-?\s*year", lambda m: int(m.group(1)) * 365),
        (r"(\d+)\s*-?\s*day", lambda m: int(m.group(1))),
    ]
    for pat, func in patterns:
        match = re.search(pat, query, re.IGNORECASE)
        if match:
            return func(match)
    return None

def extract_medical_entities(query):
    """Extract medical entities from query for better understanding"""
    entities = {
        "procedures": [],
        "conditions": [],
        "body_parts": [],
        "treatments": []
    }
    
    # Simple entity extraction based on keywords
    procedure_keywords = ["surgery", "procedure", "operation", "treatment"]
    condition_keywords = ["disease", "condition", "illness", "disorder"]
    body_part_keywords = ["knee", "heart", "eye", "dental", "organ"]
    
    words = query.lower().split()
    
    for word in words:
        if any(keyword in word for keyword in procedure_keywords):
            entities["procedures"].append(word)
        elif any(keyword in word for keyword in condition_keywords):
            entities["conditions"].append(word)
        elif any(keyword in word for keyword in body_part_keywords):
            entities["body_parts"].append(word)
    
    return entities
