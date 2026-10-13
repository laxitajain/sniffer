// Web Dashboard Client Application logic
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initPlayground();
    loadBenchmarkLeaderboard();
    loadExperimentsData();
    loadLiteratureData();
});

// Preset Benchmark Samples
const SAMPLES = {
    contradiction: {
        prompt: "Summarize the company founding date based on the article.",
        context: "Acme Corporation was founded in Berlin in 1998 by Dr. Elizabeth Vance.",
        response: "Acme Corporation was founded in Berlin in 2005 by Dr. Elizabeth Vance."
    },
    unsupported: {
        prompt: "Who leads Quantum Tech?",
        context: "Quantum Tech is a quantum computing firm located in Boston. The firm raised $50 million in Series B funding.",
        response: "Quantum Tech is a quantum computing firm located in Boston. The current CEO is Marcus Vance, who previously worked at Google AI."
    },
    factual: {
        prompt: "Describe photosynthesis inputs and outputs.",
        context: "Photosynthesis converts water and carbon dioxide into glucose and oxygen using sunlight energy.",
        response: "Photosynthesis takes water and carbon dioxide, using sunlight to produce glucose and release oxygen."
    },
    correct_unsupported: {
        prompt: "Tell me about Paris.",
        context: "Paris is the capital of France.",
        response: "Paris is the capital of France. The Eiffel Tower is located in Paris on the Champ de Mars."
    }
};

function loadSample(key) {
    if (SAMPLES[key]) {
        document.getElementById('input-prompt').value = SAMPLES[key].prompt;
        document.getElementById('input-context').value = SAMPLES[key].context;
        document.getElementById('input-response').value = SAMPLES[key].response;
    }
}

// Navigation Tabs
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabPages = document.querySelectorAll('.tab-page');
    const titleElem = document.getElementById('page-title');
    const subTitleElem = document.getElementById('page-subtitle');

    const TITLES = {
        playground: { title: "Live Hallucination Playground", sub: "Test and decompose LLM responses across 4 detection paradigms in real-time" },
        leaderboard: { title: "Detector Leaderboard", sub: "Empirical evaluation on HaluEval, RAGTruth, SummaC & TruthfulQA dataset suites" },
        experiments: { title: "Research Experiments & Pareto Frontiers", sub: "Empirical cost vs. accuracy trade-off and grounding context ablation" },
        taxonomy: { title: "Fine-Grained Failure Taxonomy", sub: "Performance matrix across 6 distinct hallucination categories" },
        literature: { title: "Core Literature (30 Papers)", sub: "Curated foundational and modern ACL, EMNLP, NAACL & Nature publications" }
    };

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const targetTab = item.getAttribute('data-tab');
            
            navItems.forEach(n => n.classList.remove('active'));
            tabPages.forEach(p => p.classList.remove('active'));

            item.classList.add('active');
            const activePage = document.getElementById(`tab-${targetTab}`);
            if (activePage) activePage.classList.add('active');

            if (TITLES[targetTab]) {
                titleElem.textContent = TITLES[targetTab].title;
                subTitleElem.textContent = TITLES[targetTab].sub;
            }
        });
    });

    document.getElementById('btn-run-benchmark').addEventListener('click', () => {
        loadBenchmarkLeaderboard();
        loadExperimentsData();
    });
}

// Playground Logic
function initPlayground() {
    const btnDetect = document.getElementById('btn-detect');
    btnDetect.addEventListener('click', async () => {
        const prompt = document.getElementById('input-prompt').value;
        const context = document.getElementById('input-context').value;
        const response = document.getElementById('input-response').value;

        if (!response.trim()) {
            alert('Please provide a generated response text to test.');
            return;
        }

        btnDetect.innerHTML = '<span>Analyzing...</span>';
        btnDetect.disabled = true;

        try {
            const apiRes = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, context, response })
            });

            if (!apiRes.ok) {
                const errJson = await apiRes.json().catch(() => ({}));
                throw new Error(errJson.detail || `Server returned HTTP ${apiRes.status}`);
            }

            const data = await apiRes.json();
            renderAtomicClaims(data.atomic_claims);
            renderDetectorsGrid(data.detectors);
        } catch (err) {
            console.error('Detection Error:', err);
            alert(`Error executing hallucination detection: ${err.message}`);
        } finally {
            btnDetect.innerHTML = `
                <span class="icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                </span> Detect Hallucinations
            `;
            btnDetect.disabled = false;
        }
    });
}

function renderAtomicClaims(claims) {
    const claimsList = document.getElementById('atomic-claims-list');
    const badge = document.getElementById('claims-count-badge');

    badge.textContent = `${claims.length} Claims`;
    claimsList.innerHTML = '';

    if (claims.length === 0) {
        claimsList.innerHTML = '<p class="placeholder-text">No claims extracted.</p>';
        return;
    }

    claims.forEach(c => {
        const card = document.createElement('div');
        const statusClass = (c.status || 'NEUTRAL').toLowerCase();
        card.className = `claim-card ${statusClass}`;
        card.innerHTML = `
            <div>
                <strong>Claim:</strong> "${c.text}"
            </div>
            <span class="claim-tag ${statusClass}">${c.status} (${(c.entailment_score * 100).toFixed(0)}%)</span>
        `;
        claimsList.appendChild(card);
    });
}

function renderDetectorsGrid(detectors) {
    const grid = document.getElementById('detectors-grid');
    grid.innerHTML = '';

    detectors.forEach(d => {
        const isHallu = d.is_hallucination;
        const scorePct = (d.hallucination_score * 100).toFixed(1);
        const card = document.createElement('div');
        card.className = `detector-result-card ${isHallu ? 'hallucinated' : 'factual'}`;

        card.innerHTML = `
            <div class="detector-header">
                <h4>${d.name}</h4>
                <span class="badge ${isHallu ? 'badge-danger' : 'badge-success'}">${isHallu ? 'HALLUCINATED' : 'FACTUAL'}</span>
            </div>
            <div class="score-display ${isHallu ? 'danger' : 'success'}">${scorePct}%</div>
            <div class="detector-meta">
                <span>Latency: ${d.latency_ms}ms</span>
                <span>Est. Cost: \$${(d.cost_usd * 1000).toFixed(4)}/1k</span>
            </div>
        `;
        grid.appendChild(card);
    });
}

// Leaderboard Loading
async function loadBenchmarkLeaderboard() {
    const tbody = document.getElementById('leaderboard-tbody');
    try {
        const res = await fetch('/api/benchmark');
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById('lb-total-samples').textContent = `${data.total_samples} Benchmark Samples`;

        tbody.innerHTML = '';
        data.leaderboard.forEach(row => {
            const tr = document.createElement('tr');
            if (row.detector_name.includes('Ensemble')) tr.className = 'highlight-row';
            tr.innerHTML = `
                <td><strong>${row.detector_name}</strong></td>
                <td>${(row.accuracy * 100).toFixed(1)}%</td>
                <td>${(row.precision * 100).toFixed(1)}%</td>
                <td>${(row.recall * 100).toFixed(1)}%</td>
                <td><strong>${(row.f1 * 100).toFixed(1)}%</strong></td>
                <td>${(row.auroc * 100).toFixed(1)}%</td>
                <td>${row.avg_latency_ms} ms</td>
                <td>\$${row.avg_cost_usd_1k.toFixed(4)}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error('Leaderboard error:', e);
    }
}

// Experiments & Charts
let paretoChartInstance = null;
let ablationChartInstance = null;

async function loadExperimentsData() {
    try {
        const res = await fetch('/api/experiments');
        if (!res.ok) return;
        const data = await res.json();

        renderParetoChart(data.pareto_analysis);
        renderCategoryMatrix(data.category_matrix);
        renderAblationChart(data.context_ablation);
    } catch (e) {
        console.error('Experiments error:', e);
    }
}

function renderParetoChart(paretoData) {
    const ctx = document.getElementById('paretoChart').getContext('2d');
    if (paretoChartInstance) paretoChartInstance.destroy();

    const datasets = paretoData.map((pt, i) => {
        const colors = ['#3b82f6', '#10b981', '#f59e0b', '#6366f1'];
        return {
            label: pt.detector,
            data: [{ x: pt.cost_usd_1k + 0.001, y: pt.f1 * 100 }],
            backgroundColor: colors[i % colors.length],
            pointRadius: 10,
            pointHoverRadius: 14
        };
    });

    paretoChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'logarithmic',
                    title: { display: true, text: 'Cost ($ / 1,000 queries - Log Scale)', color: '#9ca3af' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                y: {
                    title: { display: true, text: 'F1 Detection Score (%)', color: '#9ca3af' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#9ca3af' },
                    min: 50, max: 100
                }
            },
            plugins: {
                legend: { labels: { color: '#f3f4f6' } }
            }
        }
    });
}

function renderCategoryMatrix(catMatrix) {
    const tbody = document.getElementById('category-matrix-tbody');
    tbody.innerHTML = '';

    for (const [catName, detectors] of Object.entries(catMatrix)) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><strong>${catName}</strong></td>
            <td>${((detectors['Similarity Baseline (Lexical + Embedding)'] || 0) * 100).toFixed(0)}%</td>
            <td>${((detectors['NLI Claim Detector (SummaC / TRUE style)'] || 0) * 100).toFixed(0)}%</td>
            <td>${((detectors['SelfCheckGPT (Sampling Disagreement)'] || 0) * 100).toFixed(0)}%</td>
            <td><strong style="color: var(--success);">${((detectors['Ensemble Meta-Classifier (Multi-Feature)'] || 0) * 100).toFixed(0)}%</strong></td>
        `;
        tbody.appendChild(tr);
    }
}

function renderAblationChart(ablationData) {
    const ctx = document.getElementById('ragAblationChart').getContext('2d');
    if (ablationChartInstance) ablationChartInstance.destroy();

    const labels = ['No Context', 'Partial Context', 'Full Context (RAG)'];
    const detectors = Object.keys(ablationData);

    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#6366f1'];
    const datasets = detectors.map((det, i) => ({
        label: det.split(' ')[0],
        data: labels.map(l => (ablationData[det][l] * 100).toFixed(1)),
        backgroundColor: colors[i % colors.length]
    }));

    ablationChartInstance = new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' }, min: 0, max: 100 }
            },
            plugins: {
                legend: { labels: { color: '#f3f4f6' } }
            }
        }
    });
}

// Literature Loading
async function loadLiteratureData() {
    const grid = document.getElementById('literature-grid');
    try {
        const res = await fetch('/api/literature');
        if (!res.ok) return;
        const papers = await res.json();

        grid.innerHTML = '';
        papers.forEach(p => {
            const card = document.createElement('div');
            card.className = `paper-card ${p.is_star ? 'star-paper' : ''}`;
            
            const starBadge = p.is_star ? '<svg width="12" height="12" viewBox="0 0 24 24" fill="#f59e0b" style="vertical-align:-1px; margin-right:4px;"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>' : '';
            const linkHtml = p.link ? `<a href="${p.link}" target="_blank" rel="noopener noreferrer" style="color:var(--primary); font-size:0.8rem; text-decoration:none; display:inline-flex; align-items:center; gap:4px; margin-top:8px; font-weight:500;">ACL Anthology / Publisher <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="17" x2="17" y2="7"></line><polyline points="7 7 17 7 17 17"></polyline></svg></a>` : '';

            card.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span class="paper-venue">${starBadge}${p.venue}</span>
                    <span style="font-size:0.75rem; color:var(--text-dim); font-weight:600;">#${p.id}</span>
                </div>
                <div class="paper-title">${p.title}</div>
                <div class="paper-authors">${p.authors}</div>
                <p style="font-size: 0.84rem; color: var(--text-muted); margin-top:6px; line-height:1.4;">${p.focus}</p>
                ${linkHtml}
            `;
            grid.appendChild(card);
        });
    } catch (e) {
        console.error('Literature loading error:', e);
    }
}
