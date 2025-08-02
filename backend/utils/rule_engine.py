import re
from typing import List, Dict, Any
from .medical_terms import get_expanded_terms, extract_medical_entities

class RuleEngine:
    def __init__(self):
        # Enhanced keyword patterns
        self.exclusion_patterns = [
            r"not\s+covered", r"excluded", r"exclusion", r"not\s+payable", 
            r"not\s+included", r"not\s+eligible", r"not\s+applicable",
            r"coverage\s+not\s+provided", r"benefit\s+not\s+available"
        ]
        
        self.coverage_patterns = [
            r"covered", r"included", r"payable", r"reimbursed", r"allowed",
            r"eligible", r"applicable", r"provided", r"available", r"covered\s+under"
        ]
        
        self.waiting_period_patterns = [
            r"(\d+)\s*month[s]?\s*waiting\s+period",
            r"(\d+)\s*day[s]?\s*waiting\s+period",
            r"waiting\s+period\s+of\s+(\d+)\s*month[s]?",
            r"waiting\s+period\s+of\s+(\d+)\s*day[s]?"
        ]

    def extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text"""
        return [float(x) for x in re.findall(r"\d+\.?\d*", text)]

    def extract_percentages(self, text: str) -> List[str]:
        """Extract percentage values"""
        return re.findall(r"\d+\s*%", text)

    def extract_dates(self, text: str) -> List[str]:
        """Extract date patterns"""
        date_patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b"
        ]
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        return dates

    def check_exclusion(self, chunk: str) -> Dict[str, Any]:
        """Enhanced exclusion checking with confidence scoring"""
        chunk_lower = chunk.lower()
        exclusion_found = ""
        confidence = 0.0
        
        for pattern in self.exclusion_patterns:
            matches = re.findall(pattern, chunk_lower)
            if matches:
                exclusion_found = matches[0]
                confidence = min(1.0, len(matches) * 0.3)  # Higher confidence for multiple matches
                break
        
        return {
            "exclusion": exclusion_found,
            "confidence": confidence,
            "found": bool(exclusion_found)
        }

    def check_coverage(self, chunk: str) -> Dict[str, Any]:
        """Enhanced coverage checking with confidence scoring"""
        chunk_lower = chunk.lower()
        coverage_found = ""
        confidence = 0.0
        
        for pattern in self.coverage_patterns:
            matches = re.findall(pattern, chunk_lower)
            if matches:
                coverage_found = matches[0]
                confidence = min(1.0, len(matches) * 0.3)
                break
        
        return {
            "coverage": coverage_found,
            "confidence": confidence,
            "found": bool(coverage_found)
        }

    def check_waiting_period(self, chunk: str) -> Dict[str, Any]:
        """Enhanced waiting period extraction"""
        chunk_lower = chunk.lower()
        waiting_period = 0
        confidence = 0.0
        
        for pattern in self.waiting_period_patterns:
            match = re.search(pattern, chunk_lower)
            if match:
                number = int(match.group(1))
                if "month" in pattern:
                    waiting_period = number * 30
                else:
                    waiting_period = number
                confidence = 0.8
                break
        
        return {
            "waiting_period": waiting_period,
            "confidence": confidence,
            "found": waiting_period > 0
        }

    def extract_coverage_amount(self, chunk: str) -> Dict[str, Any]:
        """Extract coverage amounts and limits"""
        amounts = []
        percentages = self.extract_percentages(chunk)
        
        # Extract currency amounts
        currency_patterns = [
            r"Rs\.?\s*(\d+(?:,\d+)*(?:\.\d{2})?)",
            r"₹\s*(\d+(?:,\d+)*(?:\.\d{2})?)",
            r"\$(\d+(?:,\d+)*(?:\.\d{2})?)",
            r"(\d+(?:,\d+)*(?:\.\d{2})?)\s*(?:rupees?|rs)"
        ]
        
        for pattern in currency_patterns:
            matches = re.findall(pattern, chunk, re.IGNORECASE)
            amounts.extend(matches)
        
        return {
            "amounts": amounts,
            "percentages": percentages,
            "found": bool(amounts or percentages)
        }

    def check_medical_relevance(self, chunk: str, query: str) -> float:
        """Check how relevant the chunk is to the medical query"""
        query_entities = extract_medical_entities(query)
        expanded_terms = get_expanded_terms(query)
        
        chunk_lower = chunk.lower()
        relevance_score = 0.0
        
        # Check for medical entities in chunk
        for entity_type, entities in query_entities.items():
            for entity in entities:
                if entity.lower() in chunk_lower:
                    relevance_score += 0.3
        
        # Check for expanded terms
        for term in expanded_terms:
            if term.lower() in chunk_lower:
                relevance_score += 0.2
        
        return min(1.0, relevance_score)

    def evaluate(self, query: str, demographics: Dict[str, Any], duration_days: int, clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced evaluation with better reasoning and confidence scoring"""
        steps = []
        approved = False
        denied = False
        payout = None
        waiting_period = 0
        total_confidence = 0.0
        relevant_clauses = 0
        
        for clause in clauses:
            text = clause["text"]
            medical_relevance = self.check_medical_relevance(text, query)
            
            # Skip clauses with very low medical relevance
            if medical_relevance < 0.1:
                continue
            
            relevant_clauses += 1
            clause_confidence = 0.0
            
            # Exclusion check
            exclusion_result = self.check_exclusion(text)
            if exclusion_result["found"]:
                steps.append({
                    "clause": text,
                    "result": f"Denied due to exclusion: {exclusion_result['exclusion']}",
                    "highlight": exclusion_result["exclusion"],
                    "confidence": exclusion_result["confidence"],
                    "medical_relevance": medical_relevance
                })
                denied = True
                clause_confidence = exclusion_result["confidence"]
                continue
            
            # Coverage check
            coverage_result = self.check_coverage(text)
            if coverage_result["found"]:
                steps.append({
                    "clause": text,
                    "result": f"Clause indicates coverage: {coverage_result['coverage']}",
                    "highlight": coverage_result["coverage"],
                    "confidence": coverage_result["confidence"],
                    "medical_relevance": medical_relevance
                })
                approved = True
                clause_confidence = coverage_result["confidence"]
                
                # Extract coverage amounts
                amount_result = self.extract_coverage_amount(text)
                if amount_result["found"]:
                    if amount_result["amounts"]:
                        payout = amount_result["amounts"][0]
                    elif amount_result["percentages"]:
                        payout = amount_result["percentages"][0]
            
            # Waiting period check
            waiting_result = self.check_waiting_period(text)
            if waiting_result["found"]:
                steps.append({
                    "clause": text,
                    "result": f"Waiting period found: {waiting_result['waiting_period']} days",
                    "highlight": f"waiting period: {waiting_result['waiting_period']} days",
                    "confidence": waiting_result["confidence"],
                    "medical_relevance": medical_relevance
                })
                waiting_period = max(waiting_period, waiting_result["waiting_period"])
                clause_confidence = max(clause_confidence, waiting_result["confidence"])
            
            total_confidence += clause_confidence * medical_relevance
        
        # Calculate overall confidence
        if relevant_clauses > 0:
            overall_confidence = total_confidence / relevant_clauses
        else:
            overall_confidence = 0.0
        
        # Decision logic with enhanced reasoning
        if denied:
            decision = "denied"
            justification = "One or more clauses explicitly exclude this scenario."
        elif approved:
            if waiting_period and (duration_days < waiting_period):
                decision = "pending"
                justification = f"Waiting period of {waiting_period} days not satisfied (current duration: {duration_days} days)."
            else:
                decision = "approved"
                justification = "Relevant clauses indicate coverage."
        else:
            decision = "pending"
            justification = "No explicit coverage or exclusion found."
        
        return {
            "decision": decision,
            "payout": payout,
            "justification": justification,
            "steps": steps,
            "confidence": overall_confidence,
            "relevant_clauses": relevant_clauses,
            "waiting_period": waiting_period
        } 
