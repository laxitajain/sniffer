"""
FActScore-style Atomic Claim Extractor.
Decomposes complex LLM responses into fine-grained atomic factual propositions.
"""
import re
import uuid
from typing import List
from src.core.types import AtomicClaim

class ClaimExtractor:
    """
    Decomposes sentences into atomic factual propositions.
    Uses syntactic sentence splitting and rule-based clause decomposition.
    """

    def extract_claims(self, text: str) -> List[AtomicClaim]:
        """
        Extract atomic claims from a text string.
        """
        if not text or not text.strip():
            return []

        # Split into sentences using punctuation rules
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        atomic_claims = []

        for sentence in sentences:
            decomposed = self._decompose_sentence(sentence)
            for claim_text in decomposed:
                atomic_claims.append(
                    AtomicClaim(
                        id=str(uuid.uuid4())[:8],
                        text=claim_text,
                        entailment_score=0.5,
                        status="NEUTRAL",
                        confidence=0.0
                    )
                )

        return atomic_claims

    def _decompose_sentence(self, sentence: str) -> List[str]:
        """
        Splits complex sentences containing conjunctions, relative clauses,
        or numerical lists into simple atomic claims.
        """
        # Clean sentence
        sentence = re.sub(r'^\d+[\.\)]\s*', '', sentence).strip()

        # Handle compound sentences linked by coordinating conjunctions (and, but, which, born in X and studied at Y)
        # Check for clause splits like "X was born in Y, and he died in Z" or "X was founded in 1998 and operates in NY"
        pattern = r'(?<=[a-zA-Z0-9]),\s*(?:and|where|which|who|although)\s+'
        sub_clauses = re.split(pattern, sentence, flags=re.IGNORECASE)

        claims = []
        subject_match = re.match(r'^([A-Z][a-z0-9_\s]+?)\s+(?:was|is|has|had|became|worked|lived|founded|created|discovered)', sentence)
        subject = subject_match.group(1) if subject_match else None

        for clause in sub_clauses:
            clause_clean = clause.strip().rstrip('.,;')
            if not clause_clean:
                continue

            # Split further if clause has 'and also' or 'as well as'
            mini_clauses = re.split(r'\s+(?:and also|as well as)\s+', clause_clean)
            for mini in mini_clauses:
                mini = mini.strip()
                if subject and not re.match(r'^[A-Z]', mini) and not mini.startswith(subject):
                    # Attach subject prefix if missing syntactic head
                    claim_candidate = f"{subject} {mini}"
                else:
                    claim_candidate = mini

                if len(claim_candidate.split()) >= 3:
                    claims.append(claim_candidate)

        if not claims:
            claims.append(sentence)

        return claims

if __name__ == "__main__":
    extractor = ClaimExtractor()
    sample_text = "Albert Einstein was born in Ulm, Germany in 1879, and he won the Nobel Prize in Physics in 1921."
    claims = extractor.extract_claims(sample_text)
    for c in claims:
        print(f"- {c.text}")
