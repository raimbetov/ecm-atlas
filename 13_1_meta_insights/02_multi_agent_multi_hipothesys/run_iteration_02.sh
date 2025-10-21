#!/bin/bash

# Multi-Agent Multi-Hypothesis Iteration 02 - ADVANCED ML
# 6 agents: 2 per hypothesis × 3 ML-focused hypotheses
# Features: Deep Learning, GNNs, Ensemble Methods

set -e

echo "🚀🧠 Multi-Hypothesis Discovery Framework - Iteration 02 (ADVANCED ML)"
echo "===================================================================="
echo "Hypotheses: 3 (H04, H05, H06) - ALL WITH ADVANCED ML!"
echo "Agents per hypothesis: 2 (Claude Code + Codex)"
echo "Total agents: 6"
echo "ML Focus: Autoencoders, GNNs, Ensemble Learning, SHAP"
echo ""

# Repository root
REPO_ROOT="/Users/Kravtsovd/projects/ecm-atlas"
ITERATION_DIR="${REPO_ROOT}/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_02"

echo "📁 Repository root: $REPO_ROOT"
echo "📂 Iteration directory: $ITERATION_DIR"
echo "🕐 Start time: $(date)"
echo ""

# Hypothesis directories
H04_DIR="${ITERATION_DIR}/hypothesis_04_deep_protein_embeddings"
H05_DIR="${ITERATION_DIR}/hypothesis_05_gnn_aging_networks"
H06_DIR="${ITERATION_DIR}/hypothesis_06_ml_ensemble_biomarkers"

# Agent output directories
H04_CLAUDE="${H04_DIR}/claude_code"
H04_CODEX="${H04_DIR}/codex"
H05_CLAUDE="${H05_DIR}/claude_code"
H05_CODEX="${H05_DIR}/codex"
H06_CLAUDE="${H06_DIR}/claude_code"
H06_CODEX="${H06_DIR}/codex"

# Task files
H04_TASK="${H04_DIR}/01_task.md"
H05_TASK="${H05_DIR}/01_task.md"
H06_TASK="${H06_DIR}/01_task.md"

# Verify task files exist
for TASK_FILE in "$H04_TASK" "$H05_TASK" "$H06_TASK"; do
    if [ ! -f "$TASK_FILE" ]; then
        echo "❌ Error: Task file not found: $TASK_FILE"
        exit 1
    fi
done

echo "✓ All task files verified"
echo ""

# ML Requirements reminder
ML_REQ="${REPO_ROOT}/13_1_meta_insights/02_multi_agent_multi_hipothesys/ADVANCED_ML_REQUIREMENTS.md"

# Prepare prompts with ML emphasis

# ============================================================
# HYPOTHESIS 04: Deep Protein Embeddings
# ============================================================

H04_CLAUDE_PROMPT="🧠 ADVANCED ML TASK: Deep Protein Embeddings 🧠

Read the task file at ${H04_TASK} and use DEEP LEARNING to discover hidden aging patterns!

CRITICAL ML REQUIREMENTS:
- Your agent name is 'claude_code'
- Your working directory is ${REPO_ROOT}
- Create ALL files in ${H04_CLAUDE}/
- **MANDATORY:** Use ≥3 advanced ML techniques (Autoencoder, ESM-2, VAE, etc.)
- **DON'T BE SHY:** Download pre-trained models, train deep networks, go deep!
- Reference: ${ML_REQ}

ML CHECKLIST (YOU MUST DO):
✅ Train autoencoder (≥3 layers)
✅ Download ESM-2 from HuggingFace (facebook/esm2_t33_650M_UR50D)
✅ Use UMAP/t-SNE for visualization
✅ Extract latent factors and interpret biologically
✅ Save trained models (.pth files)

Required artifacts:
1. 01_plan_claude_code.md - Analysis plan
2. analysis_ml_claude_code.py - ML pipeline code
3. autoencoder_weights_claude_code.pth - Trained model
4. latent_factors_claude_code.csv - Embeddings
5. training_loss_curve_claude_code.png
6. visualizations_claude_code/ - Heatmaps, UMAP plots
7. 90_results_claude_code.md - Results in Knowledge Framework format

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

🚀 BE CREATIVE! TRY COOL ALGORITHMS! DON'T LIMIT YOURSELF! 🚀"

H04_CODEX_PROMPT="🧠 ADVANCED ML TASK: Deep Protein Embeddings 🧠

Read ${H04_TASK} and apply cutting-edge deep learning!

YOUR MISSION: Discover non-linear aging patterns using autoencoders and protein language models.

ML REQUIREMENTS:
- Agent: 'codex'
- Workspace: ${H04_CODEX}/
- Prefix: 'codex_'
- ≥3 ML techniques required
- Reference: ${ML_REQ}

MUST USE:
- PyTorch autoencoder
- ESM-2 protein embeddings
- Advanced clustering (HDBSCAN, UMAP)
- Model interpretability

Artifacts: Plan, Python code, trained models, CSVs, visualizations, results

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

# ============================================================
# HYPOTHESIS 05: GNN Aging Networks
# ============================================================

H05_CLAUDE_PROMPT="🕸️ ADVANCED ML TASK: Graph Neural Networks 🕸️

Read ${H05_TASK} and build GRAPH NEURAL NETWORKS to find master regulators!

CRITICAL REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H05_CLAUDE}/
- **MANDATORY:** Train GCN or GAT on protein networks
- Use PyTorch Geometric (torch_geometric)
- Attention mechanisms for master regulator discovery
- Reference: ${ML_REQ}

ML CHECKLIST:
✅ Build protein correlation network
✅ Train GCN/GAT (≥2 GNN layers)
✅ Extract attention weights
✅ Identify master regulators via GNN importance
✅ Compare with traditional network metrics
✅ Save GNN model and embeddings

Artifacts: Plan, GNN code (.py), trained model (.pth), embeddings CSV, network visualizations, results

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

🔥 USE ATTENTION! FIND HUBS! BE BOLD! 🔥"

H05_CODEX_PROMPT="🕸️ GNN TASK: Master Regulator Discovery 🕸️

Read ${H05_TASK} - Train Graph Neural Networks on protein interaction networks!

Requirements:
- Agent: 'codex'
- Workspace: ${H05_CODEX}/
- GCN or GAT required
- Attention-based importance scoring
- Reference: ${ML_REQ}

Build protein network → Train GNN → Extract embeddings → Find master regulators

Artifacts: All in ${H05_CODEX}/

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

# ============================================================
# HYPOTHESIS 06: ML Ensemble Biomarkers
# ============================================================

H06_CLAUDE_PROMPT="🎯 ADVANCED ML TASK: Ensemble Biomarker Discovery 🎯

Read ${H06_TASK} and build ENSEMBLE ML to find optimal aging biomarkers!

CRITICAL REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H06_CLAUDE}/
- **MANDATORY:** Train RF + XGBoost + Neural Network
- Ensemble stacking
- SHAP interpretability
- Reference: ${ML_REQ}

ML CHECKLIST:
✅ Train Random Forest (feature importance)
✅ Train XGBoost (gradient boosting)
✅ Train Neural Network (MLP, ≥3 layers)
✅ Build stacking ensemble
✅ Compute SHAP values across models
✅ Select 5-10 protein biomarker panel
✅ Validate reduced panel performance

Artifacts: Plan, code, RF model (.pkl), XGBoost (.pkl), NN (.pth), biomarker panel CSV, SHAP plots, results

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

💡 ENSEMBLE > SINGLE MODEL! SHAP FOR INSIGHTS! 💡"

H06_CODEX_PROMPT="🎯 ENSEMBLE ML TASK: Biomarker Panel 🎯

Read ${H06_TASK} - Combine RF + XGBoost + NN for biomarker discovery!

Requirements:
- Agent: 'codex'
- Workspace: ${H06_CODEX}/
- All 3 models required
- SHAP for interpretability
- Reference: ${ML_REQ}

Train ensemble → SHAP → Select biomarkers → Validate

Artifacts: All in ${H06_CODEX}/

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

echo "🚀 Launching SIX ML-powered agents in parallel..."
echo ""

# Launch all agents

# H04 Claude Code
echo "Starting H04 - Claude Code (Deep Learning)..."
(
    cd "$REPO_ROOT"
    echo "$H04_CLAUDE_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${H04_DIR}" \
        --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
        --add-dir "${REPO_ROOT}/13_1_meta_insights" \
        > "${H04_CLAUDE}/claude_code_output.log" 2>&1
    echo "H04 Claude exit: $?" >> "${H04_CLAUDE}/claude_code_output.log"
) &
H04_CLAUDE_PID=$!

# H04 Codex
echo "Starting H04 - Codex (Deep Learning)..."
(
    cd "$REPO_ROOT"
    codex exec --sandbox danger-full-access -C "$H04_CODEX" "$H04_CODEX_PROMPT" \
        > "${H04_CODEX}/codex_output.log" 2>&1
    echo "H04 Codex exit: $?" >> "${H04_CODEX}/codex_output.log"
) &
H04_CODEX_PID=$!

# H05 Claude Code
echo "Starting H05 - Claude Code (GNNs)..."
(
    cd "$REPO_ROOT"
    echo "$H05_CLAUDE_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${H05_DIR}" \
        --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
        --add-dir "${REPO_ROOT}/13_1_meta_insights" \
        > "${H05_CLAUDE}/claude_code_output.log" 2>&1
    echo "H05 Claude exit: $?" >> "${H05_CLAUDE}/claude_code_output.log"
) &
H05_CLAUDE_PID=$!

# H05 Codex
echo "Starting H05 - Codex (GNNs)..."
(
    cd "$REPO_ROOT"
    codex exec --sandbox danger-full-access -C "$H05_CODEX" "$H05_CODEX_PROMPT" \
        > "${H05_CODEX}/codex_output.log" 2>&1
    echo "H05 Codex exit: $?" >> "${H05_CODEX}/codex_output.log"
) &
H05_CODEX_PID=$!

# H06 Claude Code
echo "Starting H06 - Claude Code (Ensemble)..."
(
    cd "$REPO_ROOT"
    echo "$H06_CLAUDE_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${H06_DIR}" \
        --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
        --add-dir "${REPO_ROOT}/13_1_meta_insights" \
        > "${H06_CLAUDE}/claude_code_output.log" 2>&1
    echo "H06 Claude exit: $?" >> "${H06_CLAUDE}/claude_code_output.log"
) &
H06_CLAUDE_PID=$!

# H06 Codex
echo "Starting H06 - Codex (Ensemble)..."
(
    cd "$REPO_ROOT"
    codex exec --sandbox danger-full-access -C "$H06_CODEX" "$H06_CODEX_PROMPT" \
        > "${H06_CODEX}/codex_output.log" 2>&1
    echo "H06 Codex exit: $?" >> "${H06_CODEX}/codex_output.log"
) &
H06_CODEX_PID=$!

echo ""
echo "🔍 ML Agents launched:"
echo "   H04 (Deep Learning) - Claude: $H04_CLAUDE_PID, Codex: $H04_CODEX_PID"
echo "   H05 (GNNs) - Claude: $H05_CLAUDE_PID, Codex: $H05_CODEX_PID"
echo "   H06 (Ensemble) - Claude: $H06_CLAUDE_PID, Codex: $H06_CODEX_PID"
echo ""

echo "⏳ Monitoring ML agent progress..."
START_TIME=$(date +%s)

while kill -0 $H04_CLAUDE_PID 2>/dev/null || kill -0 $H04_CODEX_PID 2>/dev/null || \
      kill -0 $H05_CLAUDE_PID 2>/dev/null || kill -0 $H05_CODEX_PID 2>/dev/null || \
      kill -0 $H06_CLAUDE_PID 2>/dev/null || kill -0 $H06_CODEX_PID 2>/dev/null; do

    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    H04_C="🔄"; kill -0 $H04_CLAUDE_PID 2>/dev/null || H04_C="✅"
    H04_X="🔄"; kill -0 $H04_CODEX_PID 2>/dev/null || H04_X="✅"
    H05_C="🔄"; kill -0 $H05_CLAUDE_PID 2>/dev/null || H05_C="✅"
    H05_X="🔄"; kill -0 $H05_CODEX_PID 2>/dev/null || H05_X="✅"
    H06_C="🔄"; kill -0 $H06_CLAUDE_PID 2>/dev/null || H06_C="✅"
    H06_X="🔄"; kill -0 $H06_CODEX_PID 2>/dev/null || H06_X="✅"

    printf "\r⏱️  %ds | H04: C=%s X=%s | H05: C=%s X=%s | H06: C=%s X=%s" \
        "$ELAPSED" "$H04_C" "$H04_X" "$H05_C" "$H05_X" "$H06_C" "$H06_X"

    sleep 5
done

echo ""
echo ""
echo "✅ All SIX ML agents completed!"
echo ""

wait $H04_CLAUDE_PID 2>/dev/null; H04_C_EXIT=$?
wait $H04_CODEX_PID 2>/dev/null; H04_X_EXIT=$?
wait $H05_CLAUDE_PID 2>/dev/null; H05_C_EXIT=$?
wait $H05_CODEX_PID 2>/dev/null; H05_X_EXIT=$?
wait $H06_CLAUDE_PID 2>/dev/null; H06_C_EXIT=$?
wait $H06_CODEX_PID 2>/dev/null; H06_X_EXIT=$?

echo "📊 ML EXECUTION SUMMARY:"
echo "======================"
echo "H04 Deep Learning - Claude: $H04_C_EXIT, Codex: $H04_X_EXIT"
echo "H05 GNNs - Claude: $H05_C_EXIT, Codex: $H05_X_EXIT"
echo "H06 Ensemble - Claude: $H06_C_EXIT, Codex: $H06_X_EXIT"
echo ""

echo "🎉 Iteration 02 ML execution complete!"
echo "   Total hypotheses tested: 3 (ML-focused)"
echo "   Total agents executed: 6"
echo "   Cumulative progress: 6/20 theories analyzed"
