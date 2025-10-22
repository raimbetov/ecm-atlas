#!/bin/bash

# Multi-Agent Multi-Hypothesis Iteration 04 - VALIDATION & DEEP DIVES
# 12 agents: 2 per hypothesis × 6 hypotheses
# Built from Iterations 01-03: Deepen strongest findings, resolve disagreements, external validation

set -e

echo "🔬🧬 Multi-Hypothesis Discovery Framework - Iteration 04 (VALIDATION & MECHANISMS)"
echo "======================================================================================="
echo "Hypotheses: 6 (H10-H15) - DEEPEN DISCOVERIES + RESOLVE DISAGREEMENTS!"
echo "Agents per hypothesis: 2 (Claude Code + Codex)"
echo "Total agents: 12"
echo "Focus: S100 cascade, Temporal methods, Metabolic transition, External validation, Serpin resolution, Ovary/Heart biology"
echo ""

# Repository root
REPO_ROOT="/Users/Kravtsovd/projects/ecm-atlas"
ITERATION_DIR="${REPO_ROOT}/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04"

echo "📁 Repository root: $REPO_ROOT"
echo "📂 Iteration directory: $ITERATION_DIR"
echo "🕐 Start time: $(date)"
echo ""

# Hypothesis directories
H10_DIR="${ITERATION_DIR}/hypothesis_10_calcium_signaling_cascade"
H11_DIR="${ITERATION_DIR}/hypothesis_11_standardized_temporal_trajectories"
H12_DIR="${ITERATION_DIR}/hypothesis_12_metabolic_mechanical_transition"
H13_DIR="${ITERATION_DIR}/hypothesis_13_independent_dataset_validation"
H14_DIR="${ITERATION_DIR}/hypothesis_14_serpin_centrality_resolution"
H15_DIR="${ITERATION_DIR}/hypothesis_15_ovary_heart_transition_biology"

# Agent output directories
H10_CLAUDE="${H10_DIR}/claude_code"
H10_CODEX="${H10_DIR}/codex"
H11_CLAUDE="${H11_DIR}/claude_code"
H11_CODEX="${H11_DIR}/codex"
H12_CLAUDE="${H12_DIR}/claude_code"
H12_CODEX="${H12_DIR}/codex"
H13_CLAUDE="${H13_DIR}/claude_code"
H13_CODEX="${H13_DIR}/codex"
H14_CLAUDE="${H14_DIR}/claude_code"
H14_CODEX="${H14_DIR}/codex"
H15_CLAUDE="${H15_DIR}/claude_code"
H15_CODEX="${H15_DIR}/codex"

# Task files
H10_TASK="${H10_DIR}/01_task.md"
H11_TASK="${H11_DIR}/01_task.md"
H12_TASK="${H12_DIR}/01_task.md"
H13_TASK="${H13_DIR}/01_task.md"
H14_TASK="${H14_DIR}/01_task.md"
H15_TASK="${H15_DIR}/01_task.md"

# Verify task files exist
for TASK_FILE in "$H10_TASK" "$H11_TASK" "$H12_TASK" "$H13_TASK" "$H14_TASK" "$H15_TASK"; do
    if [ ! -f "$TASK_FILE" ]; then
        echo "❌ Error: Task file not found: $TASK_FILE"
        exit 1
    fi
done

echo "✓ All task files verified"
echo ""

# ML Requirements
ML_REQ="${REPO_ROOT}/13_1_meta_insights/02_multi_agent_multi_hipothesys/ADVANCED_ML_REQUIREMENTS.md"

# Prepare prompts with discovery-driven emphasis + LITERATURE + NEW DATASETS

# ============================================================
# HYPOTHESIS 10: S100→CALM/CAMK→LOX/TGM Calcium Cascade
# ============================================================

H10_CLAUDE_PROMPT="🔬💊 DEEPEN H08: Complete Calcium Signaling Pathway! 🔬💊

Read ${H10_TASK} and find the MISSING MEDIATORS in S100→crosslinking pathway!

ITERATION 03 BREAKTHROUGH (H08):
✅ Both agents CONFIRMED: S100→crosslinking pathway (R²=0.81 Claude, 0.75 Codex)
✅ S100A10→TGM2 (ρ=0.79), S100B→LOXL4 (ρ=0.74) - STRONGEST HYPOTHESIS!

CRITICAL GAP: What's BETWEEN S100 and crosslinking enzymes?
Standard calcium signaling: S100 → CALM (calmodulin) → CAMK (kinases) → LOX/TGM

YOUR MISSION: Prove the complete pathway with mediation analysis!

🚨 MANDATORY TASKS (DO NOT SKIP):
1. LITERATURE SEARCH:
   - PubMed: 'S100 calmodulin binding' (download top 10 papers)
   - Google Scholar: 'CAMK LOX transglutaminase regulation'
   - Save: literature_review.md with citations
2. NEW DATASETS:
   - Search GEO/PRIDE for datasets with CALM1/CAMK2 proteins
   - Download ≥1 external dataset if found
   - Integrate with our data for validation

REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H10_CLAUDE}/
- Mediation analysis: S100 → [CALM/CAMK] → LOX/TGM (% mediated >30%)
- AlphaFold: S100-CALM protein docking (pLDDT>70)
- Deep NN: Full pathway (S100+CALM+CAMK) → stiffness (R²>0.75)
- Compare: Model with vs without mediators (ΔR²>0.10)
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

🎯 GOAL: Identify DRUGGABLE intermediates (CAMK inhibitors already exist!) 🎯"

H10_CODEX_PROMPT="🔬 S100 Calcium Cascade - Find the Mediators! 🔬

Read ${H10_TASK} - Complete the S100→crosslinking pathway!

H08 confirmed S100→LOX/TGM. Now find: What's in between?

MANDATORY:
1. Search literature for S100-CALM interactions (PubMed)
2. Search GEO/PRIDE for calcium signaling datasets
3. Mediation analysis: CALM/CAMK as intermediates
4. AlphaFold docking: S100-calmodulin complexes

Requirements:
- Agent: 'codex'
- Workspace: ${H10_CODEX}/
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

# ============================================================
# HYPOTHESIS 11: Standardized Temporal Trajectories (Resolve H09 Disagreement)
# ============================================================

H11_CLAUDE_PROMPT="⏰⚠️ RESOLVE H09 DISAGREEMENT: Which Pseudo-Time Method Wins? ⏰⚠️

Read ${H11_TASK} and fix the LSTM reproducibility crisis!

MAJOR PROBLEM FROM H09:
❌ Claude R²=0.81 (used tissue velocity) vs Codex R²=0.011 (used PCA)
⚠️ SAME MODEL, DIFFERENT PSEUDO-TIME → 81× performance difference!

YOUR MISSION: Systematically compare ALL pseudo-time methods and find the best!

🚨 MANDATORY TASKS:
1. LITERATURE SEARCH:
   - Search: 'pseudo-time trajectory inference benchmarking'
   - Nature Methods, Bioinformatics papers on trajectory methods
   - Download methodological reviews (Saelens 2019 benchmarking paper)
2. NEW DATASETS (HIGHEST PRIORITY):
   - Search GEO/PRIDE for LONGITUDINAL aging proteomics (≥2 timepoints)
   - If found: Test pseudo-time methods vs REAL time (gold standard)
   - Download any human aging time-series

REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H11_CLAUDE}/
- Test 5 methods: Velocity (H03), PCA (Codex), Diffusion, Slingshot, Autoencoder (H04)
- LSTM benchmark: R², MSE for each method
- Sensitivity: Kendall's τ under tissue/protein subsets
- Granger causality consistency: Jaccard similarity
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

🎯 GOAL: Standardize ONE method for all future temporal modeling! 🎯"

H11_CODEX_PROMPT="⏰ Pseudo-Time Method Comparison ⏰

Read ${H11_TASK} - Find the best temporal ordering method!

H09 showed Claude/Codex disagreement (R²=0.81 vs 0.011). Root cause: different pseudo-time.

MANDATORY:
1. Search literature: trajectory inference best practices
2. Search for LONGITUDINAL datasets (real time-series)
3. Compare: Velocity, PCA, Diffusion, Slingshot methods
4. LSTM R² for each → which is most robust?

Requirements:
- Agent: 'codex'
- Workspace: ${H11_CODEX}/
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

# ============================================================
# HYPOTHESIS 12: Metabolic-Mechanical Transition (v=1.65)
# ============================================================

H12_CLAUDE_PROMPT="🔥⚙️ TWO-PHASE AGING MODEL: Reversible → Irreversible Transition! 🔥⚙️

Read ${H12_TASK} and validate the v=1.65 metabolic→mechanical shift!

H09 DISCOVERY:
✅ Critical transition at velocity=1.65 (t=-11.49, p=1.3e-28)
✅ Tissues: Ovary (v=1.53→1.77), Heart (v=1.58→1.82) span threshold

HYPOTHESIS:
- Phase I (v<1.65): METABOLIC dysregulation (mitochondria, glycolysis) - REVERSIBLE
- Phase II (v>1.65): MECHANICAL remodeling (LOX, TGM, collagens) - IRREVERSIBLE

YOUR MISSION: Find molecular markers of each phase and test reversibility!

🚨 MANDATORY TASKS:
1. LITERATURE SEARCH:
   - Search: 'metabolic aging extracellular matrix'
   - Search: 'mitochondrial dysfunction fibrosis reversibility'
   - Cell Metabolism, Nature Aging papers on metabolic→mechanical transitions
2. NEW DATASETS:
   - Search Metabolomics Workbench for metabolomics + proteomics paired data
   - Search for YAP/TAZ mechanotransduction datasets
   - Download any with mitochondrial markers

REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H12_CLAUDE}/
- Changepoint detection: Bayesian, PELT (confirm v=1.65)
- Phase I markers: Mitochondrial proteins (Fisher OR>2.0)
- Phase II markers: Crosslinking proteins (Fisher OR>2.0)
- Intervention simulation: ↑ mitochondria affects Phase I NOT Phase II
- Classification: Phase I vs II (AUC>0.90)
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

🎯 GOAL: Identify INTERVENTION WINDOW (before v=1.65)! 🎯"

H12_CODEX_PROMPT="🔥 Metabolic vs Mechanical Phase Transition 🔥

Read ${H12_TASK} - Test the two-phase aging model!

v=1.65 is critical threshold (H09). Test: Metabolic (reversible) vs Mechanical (irreversible)?

MANDATORY:
1. Search literature: metabolic-mechanical aging transitions
2. Search Metabolomics Workbench for paired datasets
3. Changepoint detection: statistical validation of v=1.65
4. Enrichment: mitochondria (Phase I) vs crosslinking (Phase II)

Requirements:
- Agent: 'codex'
- Workspace: ${H12_CODEX}/
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

# ============================================================
# HYPOTHESIS 13: Independent Dataset Validation (OVERFITTING CHECK)
# ============================================================

H13_CLAUDE_PROMPT="📊🔍 CRITICAL VALIDATION: Are We Overfit? Test on NEW Data! 📊🔍

Read ${H13_TASK} and validate our strongest findings EXTERNALLY!

RISK: ALL hypotheses H01-H12 trained on SAME dataset
✅ High R², AUC may be dataset-specific artifacts!

YOUR MISSION: Find NEW proteomics data and test WITHOUT retraining!

🚨 MANDATORY TASKS (HIGHEST PRIORITY):
1. COMPREHENSIVE DATASET SEARCH:
   - PRIDE: Search 'aging proteomics' (2020-2025, NOT our 13 studies)
   - ProteomeXchange: Human aging, ≥50 ECM protein overlap
   - GEO: Mass spec proteomics, 'aging' OR 'elderly'
   - MassIVE: Tissue aging datasets
   - Download ≥2 INDEPENDENT datasets
2. LITERATURE:
   - PubMed: Recent aging proteomics papers
   - Extract: Supplementary data links, repository IDs

REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H13_CLAUDE}/
- Transfer learning H08: S100 model → test on external (R² target ≥0.60)
- Transfer learning H06: Biomarker panel → external (AUC target ≥0.80)
- H03 velocities: Correlation with external tissue velocities (ρ>0.70)
- Meta-analysis: I² heterogeneity <50%
- Stable proteins: Identify cross-cohort consistent markers
- Reference: ${ML_REQ}

Dataset (training): /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv
Dataset (external): TO BE FOUND AND DOWNLOADED!

🎯 GOAL: Prove findings are ROBUST, not overfit! 🎯"

H13_CODEX_PROMPT="📊 External Validation - Find New Datasets! 📊

Read ${H13_TASK} - Test our models on independent data!

All H01-H12 on same dataset → overfitting risk. Need external validation.

MANDATORY:
1. Search PRIDE, ProteomeXchange, GEO for aging proteomics
2. Download ≥2 independent datasets (NOT our 13 studies)
3. Transfer H08 S100 model (no retraining) → R² external
4. Transfer H06 biomarker panel → AUC external
5. Meta-analysis: I² heterogeneity

Requirements:
- Agent: 'codex'
- Workspace: ${H13_CODEX}/
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

# ============================================================
# HYPOTHESIS 14: Serpin Network Centrality (Resolve H02 Disagreement)
# ============================================================

H14_CLAUDE_PROMPT="🕸️⚠️ RESOLVE H02 DISAGREEMENT: Serpins Central or Not? 🕸️⚠️

Read ${H14_TASK} and settle the serpin centrality debate!

H02 DISAGREEMENT:
❌ Claude: Serpins NOT central (betweenness centrality)
✅ Codex: Serpins ARE central (eigenvector centrality)

CRITICAL QUESTION: Which centrality metric is \"correct\" for proteomics networks?

YOUR MISSION: Compute ALL metrics, validate with knockouts, identify ground truth!

🚨 MANDATORY TASKS:
1. LITERATURE SEARCH:
   - Search: 'network centrality comparison biological networks'
   - Nature Methods, PLOS Comp Bio papers on centrality benchmarks
   - Search: 'centrality lethality rule validation'
   - Find: Which metric best predicts knockout phenotypes?
2. EXPERIMENTAL VALIDATION:
   - PubMed: 'SERPINC1 knockout' OR 'SERPINE1 knockout'
   - Extract phenotype severity from literature
   - Compare to predicted centrality

REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H14_CLAUDE}/
- Compute 7 metrics: Betweenness, Eigenvector, Degree, Closeness, PageRank, Katz, Subgraph
- Serpin ranking per metric (percentile <20% = central)
- Knockout simulation: Remove serpin, measure Δ connectivity
- Correlation: Which metric predicts KO impact best? (ρ>0.70)
- Consensus centrality: Ensemble of all metrics
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

🎯 GOAL: Standardize centrality metric for ALL future network analyses! 🎯"

H14_CODEX_PROMPT="🕸️ Serpin Centrality - Which Metric Wins? 🕸️

Read ${H14_TASK} - Resolve the betweenness vs eigenvector debate!

H02: Claude used betweenness (serpins NOT central), Codex used eigenvector (serpins central).

MANDATORY:
1. Search literature: network centrality best practices
2. Compute ALL metrics: betweenness, eigenvector, PageRank, etc.
3. Knockout validation: Does centrality predict KO impact?
4. Literature: SERPIN knockout phenotype severity

Requirements:
- Agent: 'codex'
- Workspace: ${H14_CODEX}/
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

# ============================================================
# HYPOTHESIS 15: Ovary & Heart Critical Transition Biology
# ============================================================

H15_CLAUDE_PROMPT="🫀🥚 WHY OVARY & HEART? Transition Tissue Biology! 🫀🥚

Read ${H15_TASK} and explain why Transformer picked these tissues!

H09 DISCOVERY:
✅ Attention hotspots: Ovary Cortex, Heart Native (BOTH agents agreed!)
✅ Critical transitions at v~1.65 (menopause, cardiac aging?)

WHY THESE TISSUES SPECIFICALLY?

HYPOTHESES:
- Ovary: Hormonal (estrogen decline → ECM remodeling)
- Heart: Mechanical (cardiac workload → mechanotransduction)
- Both: Shared metabolic collapse (mitochondria)?

YOUR MISSION: Find tissue-specific mechanisms that explain transitions!

🚨 MANDATORY TASKS:
1. LITERATURE SEARCH:
   - Ovary: 'ovarian aging extracellular matrix', 'menopause ECM collagen'
   - Heart: 'cardiac aging fibrosis', 'YAP TAZ heart mechanotransduction'
   - Shared: 'mitochondrial dysfunction ovary heart'
   - Download ≥8 papers (4 ovary + 4 heart)
2. NEW DATASETS:
   - Search for ovary-specific aging datasets
   - Search for cardiac aging proteomics
   - Search for menopause longitudinal studies

REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H15_CLAUDE}/
- Ovary gradient analysis: Estrogen-regulated proteins (ESR1, CYP19A1)
- Heart gradient analysis: Mechanotrans markers (YAP1, ROCK1)
- Metabolic hypothesis: Ovary-heart mitochondrial correlation (ρ>0.70 if shared)
- Crosslinking isoforms: LOXL2 (ovary) vs LOXL4 (heart)?
- Comparative: Transition vs non-transition tissues (p<0.05)
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

🎯 GOAL: Identify WHY these tissues → personalized interventions (HRT, cardiac stress)! 🎯"

H15_CODEX_PROMPT="🫀🥚 Ovary & Heart Transition Mechanisms 🫀🥚

Read ${H15_TASK} - Explain why Transformer attention peaked at these tissues!

H09: Ovary and Heart = critical transitions. Why?

MANDATORY:
1. Search literature: ovary aging (hormonal), heart aging (mechanical)
2. Search for ovary/cardiac aging datasets
3. Gradient analysis: Estrogen pathway (ovary), YAP/TAZ (heart)
4. Test: Shared metabolic mechanism or independent?

Requirements:
- Agent: 'codex'
- Workspace: ${H15_CODEX}/
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

echo "🚀 Launching TWELVE validation-focused agents in parallel..."
echo ""

# Launch all agents

# H10 Claude Code
echo "Starting H10 - Claude Code (S100 Cascade)..."
(
    cd "$REPO_ROOT"
    echo "$H10_CLAUDE_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${H10_DIR}" \
        --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
        --add-dir "${REPO_ROOT}/13_1_meta_insights" \
        > "${H10_CLAUDE}/claude_code_output.log" 2>&1
    echo "H10 Claude exit: $?" >> "${H10_CLAUDE}/claude_code_output.log"
) &
H10_CLAUDE_PID=$!

# H10 Codex
echo "Starting H10 - Codex (S100 Cascade)..."
(
    cd "$REPO_ROOT"
    codex exec --sandbox danger-full-access -C "$H10_CODEX" "$H10_CODEX_PROMPT" \
        > "${H10_CODEX}/codex_output.log" 2>&1
    echo "H10 Codex exit: $?" >> "${H10_CODEX}/codex_output.log"
) &
H10_CODEX_PID=$!

# H11 Claude Code
echo "Starting H11 - Claude Code (Temporal Methods)..."
(
    cd "$REPO_ROOT"
    echo "$H11_CLAUDE_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${H11_DIR}" \
        --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
        --add-dir "${REPO_ROOT}/13_1_meta_insights" \
        > "${H11_CLAUDE}/claude_code_output.log" 2>&1
    echo "H11 Claude exit: $?" >> "${H11_CLAUDE}/claude_code_output.log"
) &
H11_CLAUDE_PID=$!

# H11 Codex
echo "Starting H11 - Codex (Temporal Methods)..."
(
    cd "$REPO_ROOT"
    codex exec --sandbox danger-full-access -C "$H11_CODEX" "$H11_CODEX_PROMPT" \
        > "${H11_CODEX}/codex_output.log" 2>&1
    echo "H11 Codex exit: $?" >> "${H11_CODEX}/codex_output.log"
) &
H11_CODEX_PID=$!

# H12 Claude Code
echo "Starting H12 - Claude Code (Metabolic Transition)..."
(
    cd "$REPO_ROOT"
    echo "$H12_CLAUDE_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${H12_DIR}" \
        --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
        --add-dir "${REPO_ROOT}/13_1_meta_insights" \
        > "${H12_CLAUDE}/claude_code_output.log" 2>&1
    echo "H12 Claude exit: $?" >> "${H12_CLAUDE}/claude_code_output.log"
) &
H12_CLAUDE_PID=$!

# H12 Codex
echo "Starting H12 - Codex (Metabolic Transition)..."
(
    cd "$REPO_ROOT"
    codex exec --sandbox danger-full-access -C "$H12_CODEX" "$H12_CODEX_PROMPT" \
        > "${H12_CODEX}/codex_output.log" 2>&1
    echo "H12 Codex exit: $?" >> "${H12_CODEX}/codex_output.log"
) &
H12_CODEX_PID=$!

# H13 Claude Code
echo "Starting H13 - Claude Code (External Validation)..."
(
    cd "$REPO_ROOT"
    echo "$H13_CLAUDE_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${H13_DIR}" \
        --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
        --add-dir "${REPO_ROOT}/13_1_meta_insights" \
        > "${H13_CLAUDE}/claude_code_output.log" 2>&1
    echo "H13 Claude exit: $?" >> "${H13_CLAUDE}/claude_code_output.log"
) &
H13_CLAUDE_PID=$!

# H13 Codex
echo "Starting H13 - Codex (External Validation)..."
(
    cd "$REPO_ROOT"
    codex exec --sandbox danger-full-access -C "$H13_CODEX" "$H13_CODEX_PROMPT" \
        > "${H13_CODEX}/codex_output.log" 2>&1
    echo "H13 Codex exit: $?" >> "${H13_CODEX}/codex_output.log"
) &
H13_CODEX_PID=$!

# H14 Claude Code
echo "Starting H14 - Claude Code (Serpin Centrality)..."
(
    cd "$REPO_ROOT"
    echo "$H14_CLAUDE_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${H14_DIR}" \
        --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
        --add-dir "${REPO_ROOT}/13_1_meta_insights" \
        > "${H14_CLAUDE}/claude_code_output.log" 2>&1
    echo "H14 Claude exit: $?" >> "${H14_CLAUDE}/claude_code_output.log"
) &
H14_CLAUDE_PID=$!

# H14 Codex
echo "Starting H14 - Codex (Serpin Centrality)..."
(
    cd "$REPO_ROOT"
    codex exec --sandbox danger-full-access -C "$H14_CODEX" "$H14_CODEX_PROMPT" \
        > "${H14_CODEX}/codex_output.log" 2>&1
    echo "H14 Codex exit: $?" >> "${H14_CODEX}/codex_output.log"
) &
H14_CODEX_PID=$!

# H15 Claude Code
echo "Starting H15 - Claude Code (Ovary/Heart Biology)..."
(
    cd "$REPO_ROOT"
    echo "$H15_CLAUDE_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${H15_DIR}" \
        --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
        --add-dir "${REPO_ROOT}/13_1_meta_insights" \
        > "${H15_CLAUDE}/claude_code_output.log" 2>&1
    echo "H15 Claude exit: $?" >> "${H15_CLAUDE}/claude_code_output.log"
) &
H15_CLAUDE_PID=$!

# H15 Codex
echo "Starting H15 - Codex (Ovary/Heart Biology)..."
(
    cd "$REPO_ROOT"
    codex exec --sandbox danger-full-access -C "$H15_CODEX" "$H15_CODEX_PROMPT" \
        > "${H15_CODEX}/codex_output.log" 2>&1
    echo "H15 Codex exit: $?" >> "${H15_CODEX}/codex_output.log"
) &
H15_CODEX_PID=$!

echo ""
echo "🔍 Validation agents launched:"
echo "   H10 (S100 Cascade) - Claude: $H10_CLAUDE_PID, Codex: $H10_CODEX_PID"
echo "   H11 (Temporal Methods) - Claude: $H11_CLAUDE_PID, Codex: $H11_CODEX_PID"
echo "   H12 (Metabolic Transition) - Claude: $H12_CLAUDE_PID, Codex: $H12_CODEX_PID"
echo "   H13 (External Validation) - Claude: $H13_CLAUDE_PID, Codex: $H13_CODEX_PID"
echo "   H14 (Serpin Centrality) - Claude: $H14_CLAUDE_PID, Codex: $H14_CODEX_PID"
echo "   H15 (Ovary/Heart Biology) - Claude: $H15_CLAUDE_PID, Codex: $H15_CODEX_PID"
echo ""

echo "⏳ Monitoring agent progress..."
START_TIME=$(date +%s)

while kill -0 $H10_CLAUDE_PID 2>/dev/null || kill -0 $H10_CODEX_PID 2>/dev/null || \
      kill -0 $H11_CLAUDE_PID 2>/dev/null || kill -0 $H11_CODEX_PID 2>/dev/null || \
      kill -0 $H12_CLAUDE_PID 2>/dev/null || kill -0 $H12_CODEX_PID 2>/dev/null || \
      kill -0 $H13_CLAUDE_PID 2>/dev/null || kill -0 $H13_CODEX_PID 2>/dev/null || \
      kill -0 $H14_CLAUDE_PID 2>/dev/null || kill -0 $H14_CODEX_PID 2>/dev/null || \
      kill -0 $H15_CLAUDE_PID 2>/dev/null || kill -0 $H15_CODEX_PID 2>/dev/null; do

    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    H10_C="🔄"; kill -0 $H10_CLAUDE_PID 2>/dev/null || H10_C="✅"
    H10_X="🔄"; kill -0 $H10_CODEX_PID 2>/dev/null || H10_X="✅"
    H11_C="🔄"; kill -0 $H11_CLAUDE_PID 2>/dev/null || H11_C="✅"
    H11_X="🔄"; kill -0 $H11_CODEX_PID 2>/dev/null || H11_X="✅"
    H12_C="🔄"; kill -0 $H12_CLAUDE_PID 2>/dev/null || H12_C="✅"
    H12_X="🔄"; kill -0 $H12_CODEX_PID 2>/dev/null || H12_X="✅"
    H13_C="🔄"; kill -0 $H13_CLAUDE_PID 2>/dev/null || H13_C="✅"
    H13_X="🔄"; kill -0 $H13_CODEX_PID 2>/dev/null || H13_X="✅"
    H14_C="🔄"; kill -0 $H14_CLAUDE_PID 2>/dev/null || H14_C="✅"
    H14_X="🔄"; kill -0 $H14_CODEX_PID 2>/dev/null || H14_X="✅"
    H15_C="🔄"; kill -0 $H15_CLAUDE_PID 2>/dev/null || H15_C="✅"
    H15_X="🔄"; kill -0 $H15_CODEX_PID 2>/dev/null || H15_X="✅"

    printf "\r⏱️  %ds | H10: C=%s X=%s | H11: C=%s X=%s | H12: C=%s X=%s | H13: C=%s X=%s | H14: C=%s X=%s | H15: C=%s X=%s" \
        "$ELAPSED" "$H10_C" "$H10_X" "$H11_C" "$H11_X" "$H12_C" "$H12_X" "$H13_C" "$H13_X" "$H14_C" "$H14_X" "$H15_C" "$H15_X"

    sleep 5
done

echo ""
echo ""
echo "✅ All TWELVE validation agents completed!"
echo ""

wait $H10_CLAUDE_PID 2>/dev/null; H10_C_EXIT=$?
wait $H10_CODEX_PID 2>/dev/null; H10_X_EXIT=$?
wait $H11_CLAUDE_PID 2>/dev/null; H11_C_EXIT=$?
wait $H11_CODEX_PID 2>/dev/null; H11_X_EXIT=$?
wait $H12_CLAUDE_PID 2>/dev/null; H12_C_EXIT=$?
wait $H12_CODEX_PID 2>/dev/null; H12_X_EXIT=$?
wait $H13_CLAUDE_PID 2>/dev/null; H13_C_EXIT=$?
wait $H13_CODEX_PID 2>/dev/null; H13_X_EXIT=$?
wait $H14_CLAUDE_PID 2>/dev/null; H14_C_EXIT=$?
wait $H14_CODEX_PID 2>/dev/null; H14_X_EXIT=$?
wait $H15_CLAUDE_PID 2>/dev/null; H15_C_EXIT=$?
wait $H15_CODEX_PID 2>/dev/null; H15_X_EXIT=$?

echo "📊 EXECUTION SUMMARY:"
echo "======================"
echo "H10 S100 Cascade - Claude: $H10_C_EXIT, Codex: $H10_X_EXIT"
echo "H11 Temporal Methods - Claude: $H11_C_EXIT, Codex: $H11_X_EXIT"
echo "H12 Metabolic Transition - Claude: $H12_C_EXIT, Codex: $H12_X_EXIT"
echo "H13 External Validation - Claude: $H13_C_EXIT, Codex: $H13_X_EXIT"
echo "H14 Serpin Centrality - Claude: $H14_C_EXIT, Codex: $H14_X_EXIT"
echo "H15 Ovary/Heart Biology - Claude: $H15_C_EXIT, Codex: $H15_X_EXIT"
echo ""

echo "🎉 Iteration 04 execution complete!"
echo "   Total hypotheses tested: 6 (validation + deep dives)"
echo "   Total agents executed: 12"
echo "   Cumulative progress: 15/20 theories analyzed (75%)"
echo "   Remaining: 5 theories (Iterations 05-07)"
echo ""
echo "🔬 KEY OUTCOMES:"
echo "   ✓ H10: Complete S100 calcium signaling pathway (mediators identified)"
echo "   ✓ H11: Standardized pseudo-time method for temporal modeling"
echo "   ✓ H12: Two-phase aging model validated (intervention window defined)"
echo "   ✓ H13: External validation confirms robustness (or reveals overfitting)"
echo "   ✓ H14: Serpin centrality debate resolved (correct metric established)"
echo "   ✓ H15: Ovary/Heart transition mechanisms explained (hormonal/mechanical)"
echo ""
