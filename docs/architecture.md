# Technical Architecture & Dataflow Specification

This document details the technical architecture, technology stack, and dataflow specification for **FactCheck-NLP**, an empirical hallucination detection system.

---

## 1. System Architecture Diagram (Clean Professional White Theme)

```mermaid
%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'primaryColor': '#ffffff',
      'primaryTextColor': '#0f172a',
      'primaryBorderColor': '#94a3b8',
      'lineColor': '#475569',
      'secondaryColor': '#f8fafc',
      'tertiaryColor': '#f1f5f9',
      'clusterBkg': '#f8fafc',
      'clusterBorder': '#cbd5e1',
      'fontFamily': 'Inter, system-ui, sans-serif'
    }
  }
}%%
graph TD
    subgraph Layer_1 ["Presentation Layer (Frontend UI)"]
        UI_PLAY["Live Playground Component (ES6 JS / HTML5)"]
        UI_CHARTS["Analytics & Pareto Visualizer (Chart.js 4.4)"]
        UI_STYLING["Design Token Engine (Vanilla CSS3)"]
    end

    subgraph Layer_2 ["API Gateway & Router Subsystem"]
        ASGI["ASGI Web Server (Uvicorn 0.52 / Python 3.13)"]
        FASTAPI["REST Application Router (FastAPI 0.141)"]
        SCHEMA["Pydantic V2 Request Validation"]
        SAN["NumPy JSON Primitive Sanitizer"]
    end

    subgraph Layer_3 ["Atomic Claim Decomposition Pipeline"]
        CLAIM_ENG["FActScore Proposition Engine (NLTK / Regex Syntactic Parser)"]
        CLAIMS_LIST["Atomic Claims List (AtomicClaim Schema)"]
    end

    subgraph Layer_4 ["Parallel Detection Subsystem"]
        DET_SIM["Detector 1: Similarity Baseline (TF-IDF / SciPy Cosine)"]
        DET_NLI["Detector 2: NLI Claim Engine (SummaC / TRUE Alignment)"]
        DET_SC["Detector 3: SelfCheckGPT (k-Sample Variance Engine)"]
        DET_ENS["Detector 4: Multi-Feature Meta-Ensemble (Scikit-Learn LogReg)"]
    end

    subgraph Layer_5 ["Benchmark & Empirical Analytics Suite"]
        EVAL_ENG["Empirical Evaluator (Mann-Whitney U AUROC / F1 Score)"]
        BENCH_SUITE["Benchmark Suite (HaluEval / RAGTruth / SummaC / TruthfulQA)"]
        PARETO_ENG["Pareto Efficiency Engine (Cost-Accuracy Trade-off)"]
    end

    UI_PLAY -->|HTTP POST /api/predict| FASTAPI
    FASTAPI --> SCHEMA --> CLAIM_ENG
    CLAIM_ENG --> CLAIMS_LIST

    CLAIMS_LIST --> DET_SIM & DET_NLI & DET_SC & DET_ENS
    
    DET_SIM --> SAN
    DET_NLI --> SAN
    DET_SC --> SAN
    DET_ENS --> SAN

    SAN -->|JSON Response| UI_PLAY
    FASTAPI -->|HTTP GET /api/benchmark| BENCH_SUITE --> EVAL_ENG --> UI_CHARTS
    FASTAPI -->|HTTP GET /api/experiments| PARETO_ENG --> UI_CHARTS
```

---

## 2. Technical Execution Dataflow (Sequence Diagram)

```mermaid
%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'primaryColor': '#ffffff',
      'primaryTextColor': '#0f172a',
      'primaryBorderColor': '#94a3b8',
      'lineColor': '#334155',
      'secondaryColor': '#f8fafc',
      'tertiaryColor': '#f1f5f9',
      'fontFamily': 'Inter, system-ui, sans-serif'
    }
  }
}%%
sequenceDiagram
    autonumber
    participant UI as Client (ES6 JS / Fetch API)
    participant API as Gateway (FastAPI / Uvicorn)
    participant Parser as Claim Extractor (NLTK Syntactic)
    participant Sim as Sim Detector (TF-IDF Cosine)
    participant NLI as NLI Detector (DeBERTa Alignment)
    participant SC as SelfCheckGPT (Sampling Variance)
    participant Ens as Meta-Ensemble (Scikit-Learn LogReg)
    participant San as JSON Sanitizer (sanitize_json)

    UI->>API: POST /api/predict {prompt, context, response, stochastic_samples}
    API->>Parser: extract_claims(response)
    Parser-->>API: List[AtomicClaim]
    
    par Concurrent Detector Execution
        API->>Sim: predict(prompt, context, response)
        Sim-->>API: DetectionResult (TF-IDF, ROUGE, Jaccard)
    and
        API->>NLI: predict(prompt, context, response)
        NLI-->>API: DetectionResult (Entailment/Contradiction Ratio)
    and
        API->>SC: predict(prompt, context, response, stochastic_samples)
        SC-->>API: DetectionResult (k-Sample Disagreement Variance)
    and
        API->>Ens: predict(prompt, context, response, stochastic_samples)
        Ens->>Sim: Extract vector distance feature
        Ens->>NLI: Extract unsupported ratio feature
        Ens->>SC: Extract disagreement variance feature
        Ens-->>API: DetectionResult (Calibrated Sigmoid Score)
    end

    API->>San: sanitize_json(response_payload)
    San-->>API: Native Python Primitives Dictionary
    API-->>UI: HTTP 200 OK JSON Response
```

---

## 3. Technology Stack & Component Specifications

| Subsystem | Component | Technology / Library | Role & Function |
| :--- | :--- | :--- | :--- |
| **Frontend UI** | Client Dashboard | ES6 JS, HTML5, Vanilla CSS3 | Interactive DOM state management, tab routing, dark/white theme design system |
| **Data Visualization** | Charts Engine | Chart.js 4.4 | Renders Scatter Pareto Cost-Accuracy Frontiers and Bar Ablation charts |
| **API Gateway** | App Framework | FastAPI 0.141, Starlette | Async HTTP routing, CORS middleware, OpenAPI schema generation |
| **ASGI Server** | HTTP Web Server | Uvicorn 0.52, Python 3.13 | High-performance asynchronous web server |
| **Claim Splitter** | Proposition Parser | Python `re`, NLTK Syntactic Parser | FActScore-style atomic proposition decomposition |
| **Detector 1** | Similarity Baseline | NumPy 2.5, SciPy 1.18 | Character/word $n$-gram TF-IDF vectorizer and Cosine distance |
| **Detector 2** | NLI Claim Engine | Rule-based NLI / SentenceTransformers | Premise-hypothesis Entailment vs Contradiction classification |
| **Detector 3** | SelfCheckGPT | Stochastic Sampling Variance | Zero-resource black-box sampling disagreement engine ($k$ generations) |
| **Detector 4** | Meta-Ensemble | Scikit-Learn 1.9 Logistic Regression | Feature vector fusion & calibrated sigmoid probability estimator |
| **Evaluator** | Metrics Engine | SciPy Stats (Mann-Whitney U) | Computes Accuracy, Precision, Recall, F1, and AUROC statistics |
