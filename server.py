"""
FastAPI Server for Hallucination Detection Empirical Study & Live Dashboard.
"""
import os
import time
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.detectors.similarity_detector import SimilarityDetector
from src.detectors.nli_detector import NLIDetector
from src.detectors.selfcheckgpt_detector import SelfCheckGPTDetector
from src.detectors.ensemble_detector import EnsembleDetector
from src.dataset.benchmark_loader import BenchmarkDataset
from src.evaluation.experiments import ExperimentSuite
from src.core.claim_extractor import ClaimExtractor

app = FastAPI(
    title="Hallucination Detection Empirical Study API",
    description="Backend API for empirical comparison of LLM hallucination detection strategies.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def sanitize_json(obj):
    """Recursively converts NumPy types into native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [sanitize_json(v) for v in obj.tolist()]
    elif isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    return obj

# Initialize detectors & experiment suite
sim_detector = SimilarityDetector()
nli_detector = NLIDetector()
selfcheck_detector = SelfCheckGPTDetector()
ensemble_detector = EnsembleDetector()
claim_extractor = ClaimExtractor()

experiment_suite = ExperimentSuite()

class PredictRequest(BaseModel):
    prompt: Optional[str] = ""
    context: Optional[str] = ""
    response: str
    stochastic_samples: Optional[List[str]] = None


@app.post("/api/predict")
def predict_hallucination(req: PredictRequest):
    """
    Evaluates response factuality across all 4 hallucination detectors in parallel.
    """
    if not req.response or not req.response.strip():
        raise HTTPException(status_code=400, detail="Response text cannot be empty.")

    context = req.context or req.prompt or "General knowledge context."

    # Extract claims
    claims = claim_extractor.extract_claims(req.response)
    claim_dicts = [
        {
            "id": c.id,
            "text": c.text,
            "status": c.status,
            "entailment_score": c.entailment_score
        }
        for c in claims
    ]

    res_sim = sim_detector.predict(req.prompt, context, req.response, req.stochastic_samples)
    res_nli = nli_detector.predict(req.prompt, context, req.response, req.stochastic_samples)
    res_sc = selfcheck_detector.predict(req.prompt, context, req.response, req.stochastic_samples)
    res_ens = ensemble_detector.predict(req.prompt, context, req.response, req.stochastic_samples)

    return sanitize_json({
        "atomic_claims": claim_dicts,
        "detectors": [
            {
                "name": res_sim.detector_name,
                "hallucination_score": res_sim.hallucination_score,
                "is_hallucination": res_sim.is_hallucination_predicted,
                "confidence": res_sim.confidence,
                "latency_ms": res_sim.latency_ms,
                "cost_usd": res_sim.estimated_cost_usd,
                "metadata": res_sim.metadata
            },
            {
                "name": res_nli.detector_name,
                "hallucination_score": res_nli.hallucination_score,
                "is_hallucination": res_nli.is_hallucination_predicted,
                "confidence": res_nli.confidence,
                "latency_ms": res_nli.latency_ms,
                "cost_usd": res_nli.estimated_cost_usd,
                "metadata": res_nli.metadata
            },
            {
                "name": res_sc.detector_name,
                "hallucination_score": res_sc.hallucination_score,
                "is_hallucination": res_sc.is_hallucination_predicted,
                "confidence": res_sc.confidence,
                "latency_ms": res_sc.latency_ms,
                "cost_usd": res_sc.estimated_cost_usd,
                "metadata": res_sc.metadata
            },
            {
                "name": res_ens.detector_name,
                "hallucination_score": res_ens.hallucination_score,
                "is_hallucination": res_ens.is_hallucination_predicted,
                "confidence": res_ens.confidence,
                "latency_ms": res_ens.latency_ms,
                "cost_usd": res_ens.estimated_cost_usd,
                "metadata": res_ens.metadata
            }
        ]
    })

@app.get("/api/benchmark")
def get_benchmark_results():
    """
    Returns full evaluation metrics for all detectors on the dataset suite.
    """
    samples = BenchmarkDataset().get_samples()
    results = [
        experiment_suite.evaluator.evaluate_detector(sim_detector, samples),
        experiment_suite.evaluator.evaluate_detector(nli_detector, samples),
        experiment_suite.evaluator.evaluate_detector(selfcheck_detector, samples),
        experiment_suite.evaluator.evaluate_detector(ensemble_detector, samples)
    ]
    return sanitize_json({"leaderboard": results, "total_samples": len(samples)})

@app.get("/api/experiments")
def get_experiments_data():
    """
    Returns empirical study experiment data (Pareto, Category Matrix, RAG Ablation).
    """
    return sanitize_json(experiment_suite.run_all_experiments())

@app.get("/api/literature")
def get_literature_papers():
    """
    Returns structured summary of 30 core literature papers for hallucination detection.
    """
    return [
        {
            "id": 1,
            "title": "Entity-level Factual Consistency of Abstractive Text Summarization",
            "authors": "Nan et al.",
            "venue": "EACL 2021",
            "focus": "Early, focused work on entity hallucination in abstractive summarization and entity-level factuality metrics.",
            "is_star": False,
            "link": "https://aclanthology.org/2021.eacl-main.235/"
        },
        {
            "id": 2,
            "title": "Discourse Understanding and Factual Consistency in Abstractive Summarization",
            "authors": "Gabriel et al.",
            "venue": "EACL 2021",
            "focus": "Introduces a generator–discriminator approach for detecting/avoiding hallucinated content and improving factual consistency.",
            "is_star": False,
            "link": "https://aclanthology.org/2021.eacl-main.34/"
        },
        {
            "id": 3,
            "title": "Improving Factual Consistency of Abstractive Summarization via Question Answering",
            "authors": "Nan et al.",
            "venue": "ACL 2021",
            "focus": "Important QA-based factuality approach; demonstrates that generating/answering questions can assess whether summaries preserve source facts.",
            "is_star": False,
            "link": "https://aclanthology.org/2021.acl-long.536/"
        },
        {
            "id": 4,
            "title": "Fine-grained Factual Consistency Assessment for Abstractive Summarization Models",
            "authors": "Zhang et al.",
            "venue": "EMNLP 2021",
            "focus": "Introduces SumFC, using relevant-source retrieval + fine-grained consistency reasoning. Very relevant to an NLI-style detector.",
            "is_star": False,
            "link": "https://aclanthology.org/2021.emnlp-main.9/"
        },
        {
            "id": 5,
            "title": "Factual Consistency Evaluation for Text Summarization via Counterfactual Estimation",
            "authors": "Xie et al.",
            "venue": "EMNLP Findings 2021",
            "focus": "Interesting alternative to NLI/QA: estimates factual consistency using counterfactual reasoning.",
            "is_star": False,
            "link": "https://aclanthology.org/2021.findings-emnlp.10/"
        },
        {
            "id": 6,
            "title": "Retrieval Augmentation Reduces Hallucination in Conversation",
            "authors": "Shuster et al.",
            "venue": "EMNLP Findings 2021",
            "focus": "Establishes the important connection between retrieval/grounding and hallucination reduction, useful if you include RAG in your project.",
            "is_star": False,
            "link": "https://aclanthology.org/2021.findings-emnlp.320/"
        },
        {
            "id": 7,
            "title": "TruthfulQA: Measuring How Models Mimic Human Falsehoods",
            "authors": "Lin et al.",
            "venue": "ACL 2022",
            "focus": "Major benchmark for LLM truthfulness: 817 questions across 38 categories. Excellent foundation for defining the broader hallucination/truthfulness problem.",
            "is_star": True,
            "link": "https://aclanthology.org/2022.acl-long.229/"
        },
        {
            "id": 8,
            "title": "TRUE: Re-evaluating Factual Consistency Evaluation",
            "authors": "Honovich et al.",
            "venue": "NAACL 2022",
            "focus": "One of the most useful papers for your literature review: evaluates factuality metrics across 11 datasets and shows complementary strengths of NLI and QA approaches.",
            "is_star": True,
            "link": "https://aclanthology.org/2022.naacl-main.287/"
        },
        {
            "id": 9,
            "title": "QAFactEval: Improved QA-Based Factual Consistency Evaluation for Summarization",
            "authors": "Fabbri et al.",
            "venue": "NAACL 2022",
            "focus": "Strong QA-based factuality metric; importantly, finds QA and entailment signals can be combined for better evaluation.",
            "is_star": True,
            "link": "https://aclanthology.org/2022.naacl-main.187/"
        },
        {
            "id": 10,
            "title": "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization",
            "authors": "Laban et al.",
            "venue": "TACL 2022",
            "focus": "Excellent paper if you want to implement NLI detection. Introduces SummaC and shows how sentence-level NLI can be aggregated for document-level factuality.",
            "is_star": True,
            "link": "https://aclanthology.org/2022.tacl-1.10/"
        },
        {
            "id": 11,
            "title": "Hallucinated but Factual! Inspecting the Factuality of Hallucinations in Abstractive Summarization",
            "authors": "Cao et al.",
            "venue": "ACL 2022",
            "focus": "Very important conceptual paper. Distinguishes unsupported hallucination from actually false hallucination. Prevents you from defining every source-unseen statement as false.",
            "is_star": False,
            "link": "https://aclanthology.org/2022.acl-long.236/"
        },
        {
            "id": 12,
            "title": "HaRiM+: Evaluating Summary Quality with Hallucination Risk",
            "authors": "Son et al.",
            "venue": "AACL-IJCNLP 2022",
            "focus": "Reference-free hallucination-risk metric based on token likelihoods; useful as a cheap baseline.",
            "is_star": False,
            "link": "https://aclanthology.org/2022.aacl-main.66/"
        },
        {
            "id": 13,
            "title": "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models",
            "authors": "Li et al.",
            "venue": "EMNLP 2023",
            "focus": "One of the central modern LLM hallucination benchmarks. Contains generated + human-annotated hallucinations and tests whether LLMs can recognize them.",
            "is_star": True,
            "link": "https://aclanthology.org/2023.emnlp-main.397/"
        },
        {
            "id": 14,
            "title": "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models",
            "authors": "Manakul et al.",
            "venue": "EMNLP 2023",
            "focus": "Probably the single most useful paper to reproduce for your project. Uses disagreement among multiple sampled responses to detect hallucination without external knowledge.",
            "is_star": True,
            "link": "https://aclanthology.org/2023.emnlp-main.557/"
        },
        {
            "id": 15,
            "title": "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation",
            "authors": "Min et al.",
            "venue": "EMNLP 2023",
            "focus": "Breaks long answers into atomic facts and checks each independently. Gives you a much more meaningful factuality metric than binary answer-level classification.",
            "is_star": True,
            "link": "https://aclanthology.org/2023.emnlp-main.741/"
        },
        {
            "id": 16,
            "title": "AlignScore: Evaluating Factual Consistency with A Unified Alignment Function",
            "authors": "Zha et al.",
            "venue": "ACL 2023",
            "focus": "General-purpose factual consistency model trained on 4.7M examples from NLI, QA, fact verification, similarity, retrieval, etc. Evaluated on 22 datasets.",
            "is_star": True,
            "link": "https://aclanthology.org/2023.acl-long.634/"
        },
        {
            "id": 17,
            "title": "Hallucination Detection for Generative Large Language Models by Bayesian Sequential Estimation",
            "authors": "Wang et al.",
            "venue": "EMNLP 2023",
            "focus": "Introduces a sequential detection strategy that trades off detection accuracy against retrieval/computation cost.",
            "is_star": False,
            "link": "https://aclanthology.org/2023.emnlp-main.949/"
        },
        {
            "id": 18,
            "title": "Enhancing Uncertainty-Based Hallucination Detection with Stronger Focus",
            "authors": "Zhang et al.",
            "venue": "EMNLP 2023",
            "focus": "Reference-free detection using uncertainty/token-level signals instead of external evidence or multiple generations.",
            "is_star": False,
            "link": "https://aclanthology.org/2023.emnlp-main.58/"
        },
        {
            "id": 19,
            "title": "DelucionQA: Detecting Hallucinations in Domain-specific Question Answering",
            "authors": "Sadat et al.",
            "venue": "EMNLP Findings 2023",
            "focus": "Useful if you want to investigate hallucinations in domain-specific QA/RAG, rather than generic generation.",
            "is_star": False,
            "link": "https://aclanthology.org/2023.findings-emnlp.59/"
        },
        {
            "id": 20,
            "title": "A New Benchmark and Reverse Validation Method for Passage-level Hallucination Detection",
            "authors": "Yang et al.",
            "venue": "EMNLP Findings 2023",
            "focus": "Introduces a passage-level hallucination benchmark and reverse validation, a useful zero-resource detection idea.",
            "is_star": False,
            "link": "https://aclanthology.org/2023.findings-emnlp.256/"
        },
        {
            "id": 21,
            "title": "Sources of Hallucination by Large Language Models on Inference Tasks",
            "authors": "McKenna et al.",
            "venue": "EMNLP Findings 2023",
            "focus": "Investigates why LLMs hallucinate, finding biases from memorization and corpus statistics. Useful for your failure-analysis section.",
            "is_star": False,
            "link": "https://aclanthology.org/2023.findings-emnlp.182/"
        },
        {
            "id": 22,
            "title": "Language Models Hallucinate, but May Excel at Fact Verification",
            "authors": "Guan et al.",
            "venue": "NAACL 2024",
            "focus": "Very interesting result: models that hallucinate as generators can nevertheless be strong fact verifiers. Also examines robustness and evidence dependence.",
            "is_star": True,
            "link": "https://aclanthology.org/2024.naacl-long.62/"
        },
        {
            "id": 23,
            "title": "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models",
            "authors": "Niu et al.",
            "venue": "ACL 2024",
            "focus": "Excellent dataset if you want a RAG-specific hallucination project: ~18K RAG-generated responses with word- and response-level annotations.",
            "is_star": True,
            "link": "https://aclanthology.org/2024.acl-long.585/"
        },
        {
            "id": 24,
            "title": "The Dawn After the Dark: An Empirical Study on Factuality Hallucination in Large Language Models",
            "authors": "Li et al.",
            "venue": "ACL 2024",
            "focus": "Comprehensive study of hallucination detection, causes and mitigation, including HaluEval 2.0.",
            "is_star": False,
            "link": "https://aclanthology.org/2024.acl-long.586/"
        },
        {
            "id": 25,
            "title": "Knowledge-Centric Hallucination Detection",
            "authors": "Hu et al.",
            "venue": "EMNLP 2024",
            "focus": "Studies hallucination from the perspective of whether generated information is consistent with known/world knowledge, rather than only source grounding.",
            "is_star": False,
            "link": "https://aclanthology.org/2024.emnlp-main.395/"
        },
        {
            "id": 26,
            "title": "Unsupervised Real-Time Hallucination Detection based on the Internal States of Large Language Models",
            "authors": "Su et al.",
            "venue": "ACL Findings 2024",
            "focus": "Introduces MIND/HELM, using internal model states for real-time hallucination detection without manual annotations.",
            "is_star": False,
            "link": "https://aclanthology.org/2024.findings-acl.854/"
        },
        {
            "id": 27,
            "title": "On Large Language Models' Hallucination with Regard to Known Facts",
            "authors": "Jiang et al.",
            "venue": "NAACL 2024",
            "focus": "Particularly interesting mechanistic study: models can possess the correct knowledge internally yet still generate an incorrect answer.",
            "is_star": True,
            "link": "https://aclanthology.org/2024.naacl-long.60/"
        },
        {
            "id": 28,
            "title": "Detecting Hallucinations in Large Language Models Using Semantic Entropy",
            "authors": "Farquhar et al.",
            "venue": "Nature 2024",
            "focus": "One of the strongest papers in the area. Uses semantic uncertainty across sampled answers rather than exact textual disagreement.",
            "is_star": True,
            "link": "https://www.nature.com/articles/s41586-024-07421-0"
        },
        {
            "id": 29,
            "title": "A Probabilistic Framework for LLM Hallucination Detection via Belief Tree Propagation",
            "authors": "Hou et al.",
            "venue": "NAACL 2025",
            "focus": "Breaks statements into related claims and performs probabilistic reasoning over them; reports 3–9% AUROC/AUC-PR improvements over baselines.",
            "is_star": True,
            "link": "https://aclanthology.org/2025.naacl-long.158/"
        },
        {
            "id": 30,
            "title": "Simple Factuality Probes Detect Hallucinations in Long-Form Natural Language Generation",
            "authors": "Han et al.",
            "venue": "EMNLP Findings 2025",
            "focus": "Extremely relevant to a practical mini-project: uses LLM hidden states + a lightweight probe, achieving competitive results with dramatically lower compute.",
            "is_star": True,
            "link": "https://aclanthology.org/2025.findings-emnlp.880/"
        }
    ]


# Mount web directory for static frontend app
web_dir = os.path.join(os.path.dirname(__file__), "web")
if os.path.exists(web_dir):
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
