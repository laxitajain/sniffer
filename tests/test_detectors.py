"""
Unit tests for Hallucination Detection framework.
"""
import unittest
from src.core.claim_extractor import ClaimExtractor
from src.detectors.similarity_detector import SimilarityDetector
from src.detectors.nli_detector import NLIDetector
from src.detectors.selfcheckgpt_detector import SelfCheckGPTDetector
from src.detectors.ensemble_detector import EnsembleDetector
from src.dataset.benchmark_loader import BenchmarkDataset
from src.evaluation.evaluator import HallucinationEvaluator

class TestHallucinationDetectors(unittest.TestCase):
    def setUp(self):
        self.claim_extractor = ClaimExtractor()
        self.sim_detector = SimilarityDetector()
        self.nli_detector = NLIDetector()
        self.selfcheck_detector = SelfCheckGPTDetector()
        self.ensemble_detector = EnsembleDetector()
        self.dataset = BenchmarkDataset()
        self.evaluator = HallucinationEvaluator()

    def test_claim_extraction(self):
        text = "Einstein was born in Germany in 1879, and he developed the theory of relativity."
        claims = self.claim_extractor.extract_claims(text)
        self.assertGreaterEqual(len(claims), 2)
        self.assertTrue(any("1879" in c.text for c in claims))

    def test_detectors_prediction(self):
        prompt = "Where was Einstein born?"
        context = "Albert Einstein was born in Ulm, Germany in 1879."
        response = "Albert Einstein was born in Paris, France in 1905."

        for det in [self.sim_detector, self.nli_detector, self.selfcheck_detector, self.ensemble_detector]:
            res = det.predict(prompt, context, response)
            self.assertIsNotNone(res.hallucination_score)
            self.assertTrue(0.0 <= res.hallucination_score <= 1.0)
            self.assertIsNotNone(res.is_hallucination_predicted)

    def test_benchmark_evaluation(self):
        samples = self.dataset.get_samples()
        self.assertGreaterEqual(len(samples), 5)
        
        metrics = self.evaluator.evaluate_detector(self.ensemble_detector, samples)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("auroc", metrics)

if __name__ == "__main__":
    unittest.main()
