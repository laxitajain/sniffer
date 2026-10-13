#!/usr/bin/env python3
"""
CLI & Entrypoint Runner for Hallucination Detection Empirical NLP Study.
Usage:
    python3 run.py --serve            # Start web server & live dashboard on http://localhost:8000
    python3 run.py --evaluate         # Run benchmark suite and output accuracy matrix to console
"""
import sys
import argparse
import uvicorn
from src.dataset.benchmark_loader import BenchmarkDataset
from src.evaluation.experiments import ExperimentSuite

def run_evaluation():
    print("=" * 70)
    print(" EMPIRICAL STUDY: HALLUCINATION DETECTION IN LLM-GENERATED TEXT")
    print("=" * 70)
    print("\nRunning benchmark evaluation suite across 4 detectors...\n")

    suite = ExperimentSuite()
    results = suite.run_all_experiments()

    leaderboard = results["leaderboard"]

    print(f"{'DETECTOR STRATEGY':<45} | {'ACC':<6} | {'PREC':<6} | {'REC':<6} | {'F1':<6} | {'AUROC':<6} | {'LATENCY':<8}")
    print("-" * 100)
    for row in leaderboard:
        print(f"{row['detector_name']:<45} | {row['accuracy']*100:5.1f}% | {row['precision']*100:5.1f}% | {row['recall']*100:5.1f}% | {row['f1']*100:5.1f}% | {row['auroc']*100:5.1f}% | {row['avg_latency_ms']:6.1f}ms")

    print("-" * 100)
    print("\n[✓] Evaluation completed successfully.")

def run_server(port=8000):
    print("=" * 70)
    print(f" Starting FactCheck-NLP Dashboard Server on http://localhost:{port}")
    print("=" * 70)
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)

def main():
    parser = argparse.ArgumentParser(description="Hallucination Detection Empirical NLP Study Runner")
    parser.add_argument("--evaluate", action="store_true", help="Run empirical benchmark suite")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI web server")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default 8000)")

    args = parser.parse_args()

    if args.evaluate:
        run_evaluation()
    elif args.serve:
        run_server(args.port)
    else:
        # Default behavior: run evaluation then start server
        run_evaluation()
        print("\nStarting web server...")
        run_server(args.port)

if __name__ == "__main__":
    main()
