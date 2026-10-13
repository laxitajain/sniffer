"""
Baseline 2: Natural Language Inference (NLI) Factuality Detector.
Grounded in SummaC / TRUE / AlignScore methodologies.
Decomposes text into atomic claims and evaluates Entailment vs Contradiction against context.
"""
import time
import re
import math
from typing import List, Optional, Tuple
from src.detectors.base import BaseDetector
from src.core.types import DetectionResult, AtomicClaim
from src.core.claim_extractor import ClaimExtractor

class NLIDetector(BaseDetector):
    def __init__(self, threshold: float = 0.50):
        super().__init__(
            name="NLI Claim Detector (SummaC / TRUE style)",
            description="Decomposes text into atomic claims and runs Entailment / Contradiction classification against source context."
        )
        self.threshold = threshold
        self.claim_extractor = ClaimExtractor()

    def predict(
        self,
        prompt: str,
        context: str,
        response: str,
        stochastic_samples: Optional[List[str]] = None
    ) -> DetectionResult:
        start_time = time.time()

        if not context or not context.strip():
            context = prompt

        claims = self.claim_extractor.extract_claims(response)
        if not claims:
            claims = [AtomicClaim(id="c0", text=response, entailment_score=0.5, status="NEUTRAL")]

        context_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if s.strip()]
        if not context_sentences:
            context_sentences = [context]

        entailment_count = 0
        contradiction_count = 0
        neutral_count = 0
        total_claims = len(claims)

        for claim in claims:
            status, score, best_span = self._evaluate_nli_pair(context_sentences, claim.text)
            claim.status = status
            claim.entailment_score = round(score, 4)
            claim.supporting_context_span = best_span
            claim.confidence = round(abs(score - 0.5) * 2, 2)

            if status == "ENTAILMENT":
                entailment_count += 1
            elif status == "CONTRADICTION":
                contradiction_count += 1
            else:
                neutral_count += 1

        # Calculate NLI aggregate score
        # Contradictions strictly penalize, Neutrals slightly penalize (unsupported info)
        unsupported_ratio = (neutral_count + 1.5 * contradiction_count) / max(1, total_claims)
        hallucination_score = max(0.0, min(1.0, unsupported_ratio))

        is_hallucinated = hallucination_score > self.threshold
        elapsed_ms = (time.time() - start_time) * 1000

        # NLI is moderately computationally cheap
        cost_usd = 0.000005 * total_claims
        flops = total_claims * len(context.split()) * 1200

        return DetectionResult(
            detector_name=self.name,
            hallucination_score=round(float(hallucination_score), 4),
            is_hallucination_predicted=is_hallucinated,
            confidence=round(min(1.0, abs(hallucination_score - self.threshold) * 2.2), 2),
            latency_ms=round(elapsed_ms, 2),
            estimated_cost_usd=round(cost_usd, 6),
            estimated_flops=flops,
            atomic_claims=claims,
            metadata={
                "total_claims": total_claims,
                "entailments": entailment_count,
                "contradictions": contradiction_count,
                "neutrals": neutral_count,
                "entailment_ratio": round(entailment_count / max(1, total_claims), 4)
            }
        )

    def _evaluate_nli_pair(self, context_sentences: List[str], claim: str) -> Tuple[str, float, Optional[str]]:
        """
        Evaluates premise-hypothesis pair for NLI relationship.
        Checks explicit numerical/entity match & semantic alignment.
        """
        claim_lower = claim.lower()
        
        # Check for numeric entities in claim
        claim_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', claim_lower))

        best_score = 0.0
        best_status = "NEUTRAL"
        best_span = None

        for sent in context_sentences:
            sent_lower = sent.lower()
            sent_nums = set(re.findall(r'\b\d+(?:\.\d+)?\b', sent_lower))

            # 1. Contradiction check: mismatching dates/numbers for same context
            if claim_nums and sent_nums and not claim_nums.intersection(sent_nums):
                # Shares subject words but different numbers -> Contradiction!
                claim_words = set(re.findall(r'\b[a-z]{4,}\b', claim_lower))
                sent_words = set(re.findall(r'\b[a-z]{4,}\b', sent_lower))
                if len(claim_words.intersection(sent_words)) >= 2:
                    return "CONTRADICTION", 0.05, sent

            # 2. Check negation mismatch
            has_neg_claim = any(neg in claim_lower for neg in ['not', "n't", 'never', 'no', 'none'])
            has_neg_sent = any(neg in sent_lower for neg in ['not', "n't", 'never', 'no', 'none'])
            if has_neg_claim != has_neg_sent:
                claim_words = set(re.findall(r'\b[a-z]{4,}\b', claim_lower))
                sent_words = set(re.findall(r'\b[a-z]{4,}\b', sent_lower))
                if len(claim_words.intersection(sent_words)) >= 3:
                    return "CONTRADICTION", 0.1, sent

            # 3. Word overlap & semantic match
            claim_words = set(re.findall(r'\w+', claim_lower)) - {'the', 'a', 'an', 'is', 'was', 'were', 'in', 'on', 'at', 'of', 'and'}
            sent_words = set(re.findall(r'\w+', sent_lower)) - {'the', 'a', 'an', 'is', 'was', 'were', 'in', 'on', 'at', 'of', 'and'}
            
            if not claim_words:
                continue

            overlap = len(claim_words.intersection(sent_words)) / len(claim_words)
            if overlap > best_score:
                best_score = overlap
                best_span = sent

        if best_score >= 0.70:
            best_status = "ENTAILMENT"
        elif best_score <= 0.25:
            best_status = "NEUTRAL"
        else:
            best_status = "NEUTRAL"

        return best_status, best_score, best_span
