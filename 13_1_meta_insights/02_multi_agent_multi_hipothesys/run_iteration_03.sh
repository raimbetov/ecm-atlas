#!/bin/bash

# Multi-Agent Multi-Hypothesis Iteration 03 - COAGULATION & TEMPORAL DYNAMICS
# 6 agents: 2 per hypothesis × 3 emergent pattern hypotheses
# Built from Iterations 01-02 discoveries: Coagulation centrality, S100 family, Temporal gaps

set -e

echo "🔬💉 Multi-Hypothesis Discovery Framework - Iteration 03 (COAGULATION & TEMPORAL)"
echo "=================================================================================="
echo "Hypotheses: 3 (H07, H08, H09) - EMERGENT PATTERNS FROM ITERATIONS 01-02!"
echo "Agents per hypothesis: 2 (Claude Code + Codex)"
echo "Total agents: 6"
echo "Focus: Coagulation hub, S100 calcium signaling, Temporal RNN trajectories"
echo ""

# Repository root
REPO_ROOT="/Users/Kravtsovd/projects/ecm-atlas"
ITERATION_DIR="${REPO_ROOT}/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_03"

echo "📁 Repository root: $REPO_ROOT"
echo "📂 Iteration directory: $ITERATION_DIR"
echo "🕐 Start time: $(date)"
echo ""

# Hypothesis directories
H07_DIR="${ITERATION_DIR}/hypothesis_07_coagulation_central_hub"
H08_DIR="${ITERATION_DIR}/hypothesis_08_s100_calcium_signaling"
H09_DIR="${ITERATION_DIR}/hypothesis_09_temporal_rnn_trajectories"

# Agent output directories
H07_CLAUDE="${H07_DIR}/claude_code"
H07_CODEX="${H07_DIR}/codex"
H08_CLAUDE="${H08_DIR}/claude_code"
H08_CODEX="${H08_DIR}/codex"
H09_CLAUDE="${H09_DIR}/claude_code"
H09_CODEX="${H09_DIR}/codex"

# Task files
H07_TASK="${H07_DIR}/01_task.md"
H08_TASK="${H08_DIR}/01_task.md"
H09_TASK="${H09_DIR}/01_task.md"

# Verify task files exist
for TASK_FILE in "$H07_TASK" "$H08_TASK" "$H09_TASK"; do
    if [ ! -f "$TASK_FILE" ]; then
        echo "❌ Error: Task file not found: $TASK_FILE"
        exit 1
    fi
done

echo "✓ All task files verified"
echo ""

# ML Requirements
ML_REQ="${REPO_ROOT}/13_1_meta_insights/02_multi_agent_multi_hipothesys/ADVANCED_ML_REQUIREMENTS.md"

# Prepare prompts with discovery-driven emphasis

# ============================================================
# HYPOTHESIS 07: Coagulation Cascade Central Hub
# ============================================================

H07_CLAUDE_PROMPT="💉🧬 CONVERGENT DISCOVERY: Coagulation as Central Aging Hub 💉🧬

Read ${H07_TASK} and test if COAGULATION CASCADE is THE central aging mechanism!

CRITICAL DISCOVERY FROM ITERATIONS 01-02:
Coagulation proteins appeared in ALL 6 completed hypotheses despite NO pre-focus:
- H06: F13B, GAS6 in top 8 biomarkers (AUC=1.0)
- H03: F2, SERPINB6A shared across fast-aging tissues
- H02: SERPINC1, SERPINF2 highly dysregulated serpins
- H01: F13B magnitude 7.80 SD (rank #2 antagonism)

YOUR MISSION: Prove or disprove that coagulation dysregulation DRIVES aging (not just correlates)

REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H07_CLAUDE}/
- Deep NN: Coagulation proteins → Aging velocity (R²>0.85 target)
- LSTM: Temporal precedence (coagulation → ECM remodeling)
- Transfer learning: Pre-trained thrombosis models
- Network centrality: Coagulation module vs Serpin module vs Collagen module
- SHAP: Which coagulation proteins matter most?
- Reference: ${ML_REQ}

COAGULATION PROTEINS: F2, F13B, GAS6, SERPINC1, PLAU, PLAUR, FGA/B/G, VWF, PROC, PROS1, THBD, SERPINE1, SERPINF2

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

🎯 GOAL: If R²>0.85 with coagulation-only → CONFIRMED as central hub! 🎯"

H07_CODEX_PROMPT="💉 Coagulation Hub Discovery 💉

Read ${H07_TASK} - Test the convergent hypothesis!

Coagulation proteins recurred across ALL prior hypotheses. Your job: Quantify centrality.

Requirements:
- Agent: 'codex'
- Workspace: ${H07_CODEX}/
- Deep NN + LSTM + Network analysis
- Prove temporal precedence
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

# ============================================================
# HYPOTHESIS 08: S100 Calcium Signaling
# ============================================================

H08_CLAUDE_PROMPT="🧬🔬 PARADOX RESOLUTION: S100 Calcium Signaling (NOT Inflammation!) 🔬🧬

Read ${H08_TASK} and solve the S100 paradox!

PARADOX FROM ITERATIONS 01-02:
S100 proteins selected by 3 independent ML methods (H04 autoencoder, H06 SHAP, H03 TSI)
BUT inflammation hypothesis REJECTED (p=0.41-0.63)

RESOLUTION HYPOTHESIS: S100 acts via calcium-dependent mechanotransduction and ECM crosslinking (LOX, TGM2), NOT inflammation!

YOUR MISSION: Prove S100 → Calcium → Crosslinking → Tissue stiffness

REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H08_CLAUDE}/
- Transfer learning: AlphaFold S100 structures (or ESM-2 proxy)
- Deep NN: S100 expression → Tissue stiffness (R²>0.70)
- Attention network: S100 → LOX/TGM2 relationships
- Correlation test: S100-crosslinking > S100-inflammation
- Reference: ${ML_REQ}

S100 PROTEINS: S100A8, S100A9, S100B, S100A1, S100A4, S100A6, S100P
CROSSLINKING: LOX, LOXL1-4, TGM2, TGM1, TGM3
INFLAMMATION: IL6, IL1B, TNF, CXCL8, CCL2

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

🔬 GOAL: S100-LOX correlation > S100-IL6 correlation → Mechanism confirmed! 🔬"

H08_CODEX_PROMPT="🔬 S100 Calcium Signaling Analysis 🔬

Read ${H08_TASK} - Resolve the S100 paradox!

Test: S100 → Calcium signaling → ECM crosslinking (NOT inflammation)

Requirements:
- Agent: 'codex'
- Workspace: ${H08_CODEX}/
- Transfer learning + Attention + Network analysis
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

# ============================================================
# HYPOTHESIS 09: Temporal RNN Trajectories
# ============================================================

H09_CLAUDE_PROMPT="⏰🔮 TEMPORAL DYNAMICS: Predict the Future with RNNs ⏰🔮

Read ${H09_TASK} and model WHEN proteins change during aging!

CRITICAL GAP FROM ITERATIONS 01-02:
ALL prior hypotheses used cross-sectional analysis (snapshots)
NONE modeled temporal dynamics or predicted future states

YOUR MISSION: Train LSTM to predict future protein trajectories and find critical transition points

REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H09_CLAUDE}/
- LSTM encoder-decoder: Predict future protein states (accuracy >85%)
- Transformer attention: Identify critical transitions (\"point of no return\")
- Early vs Late-change proteins: Q1 predicts Q4 with R²>0.70
- Granger causality: Early → Late causal testing
- Time-series CV: Leave-future-out validation
- Reference: ${ML_REQ}

PSEUDO-TIME OPTIONS:
1. Age metadata (if available in Study_ID)
2. Tissue velocity ranking (H03): slow → fast as time proxy
3. Latent space traversal (H04 autoencoder)

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

🔮 GOAL: Predict 6-month-ahead protein states with MSE<0.3 → Future is predictable! 🔮"

H09_CODEX_PROMPT="⏰ Temporal Trajectory Prediction ⏰

Read ${H09_TASK} - Model aging dynamics over time!

Train LSTM + Transformer to predict future ECM states and identify early-change proteins.

Requirements:
- Agent: 'codex'
- Workspace: ${H09_CODEX}/
- LSTM + Transformer + Granger causality
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

echo "🚀 Launching SIX discovery-driven agents in parallel..."
echo ""

# Launch all agents

# H07 Claude Code
echo "Starting H07 - Claude Code (Coagulation Hub)..."
(
    cd "$REPO_ROOT"
    echo "$H07_CLAUDE_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${H07_DIR}" \
        --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
        --add-dir "${REPO_ROOT}/13_1_meta_insights" \
        > "${H07_CLAUDE}/claude_code_output.log" 2>&1
    echo "H07 Claude exit: $?" >> "${H07_CLAUDE}/claude_code_output.log"
) &
H07_CLAUDE_PID=$!

# H07 Codex
echo "Starting H07 - Codex (Coagulation Hub)..."
(
    cd "$REPO_ROOT"
    codex exec --sandbox danger-full-access -C "$H07_CODEX" "$H07_CODEX_PROMPT" \
        > "${H07_CODEX}/codex_output.log" 2>&1
    echo "H07 Codex exit: $?" >> "${H07_CODEX}/codex_output.log"
) &
H07_CODEX_PID=$!

# H08 Claude Code
echo "Starting H08 - Claude Code (S100 Calcium)..."
(
    cd "$REPO_ROOT"
    echo "$H08_CLAUDE_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${H08_DIR}" \
        --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
        --add-dir "${REPO_ROOT}/13_1_meta_insights" \
        > "${H08_CLAUDE}/claude_code_output.log" 2>&1
    echo "H08 Claude exit: $?" >> "${H08_CLAUDE}/claude_code_output.log"
) &
H08_CLAUDE_PID=$!

# H08 Codex
echo "Starting H08 - Codex (S100 Calcium)..."
(
    cd "$REPO_ROOT"
    codex exec --sandbox danger-full-access -C "$H08_CODEX" "$H08_CODEX_PROMPT" \
        > "${H08_CODEX}/codex_output.log" 2>&1
    echo "H08 Codex exit: $?" >> "${H08_CODEX}/codex_output.log"
) &
H08_CODEX_PID=$!

# H09 Claude Code
echo "Starting H09 - Claude Code (Temporal RNN)..."
(
    cd "$REPO_ROOT"
    echo "$H09_CLAUDE_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${H09_DIR}" \
        --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
        --add-dir "${REPO_ROOT}/13_1_meta_insights" \
        > "${H09_CLAUDE}/claude_code_output.log" 2>&1
    echo "H09 Claude exit: $?" >> "${H09_CLAUDE}/claude_code_output.log"
) &
H09_CLAUDE_PID=$!

# H09 Codex
echo "Starting H09 - Codex (Temporal RNN)..."
(
    cd "$REPO_ROOT"
    codex exec --sandbox danger-full-access -C "$H09_CODEX" "$H09_CODEX_PROMPT" \
        > "${H09_CODEX}/codex_output.log" 2>&1
    echo "H09 Codex exit: $?" >> "${H09_CODEX}/codex_output.log"
) &
H09_CODEX_PID=$!

echo ""
echo "🔍 Discovery agents launched:"
echo "   H07 (Coagulation) - Claude: $H07_CLAUDE_PID, Codex: $H07_CODEX_PID"
echo "   H08 (S100 Calcium) - Claude: $H08_CLAUDE_PID, Codex: $H08_CODEX_PID"
echo "   H09 (Temporal RNN) - Claude: $H09_CLAUDE_PID, Codex: $H09_CODEX_PID"
echo ""

echo "⏳ Monitoring agent progress..."
START_TIME=$(date +%s)

while kill -0 $H07_CLAUDE_PID 2>/dev/null || kill -0 $H07_CODEX_PID 2>/dev/null || \
      kill -0 $H08_CLAUDE_PID 2>/dev/null || kill -0 $H08_CODEX_PID 2>/dev/null || \
      kill -0 $H09_CLAUDE_PID 2>/dev/null || kill -0 $H09_CODEX_PID 2>/dev/null; do

    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    H07_C="🔄"; kill -0 $H07_CLAUDE_PID 2>/dev/null || H07_C="✅"
    H07_X="🔄"; kill -0 $H07_CODEX_PID 2>/dev/null || H07_X="✅"
    H08_C="🔄"; kill -0 $H08_CLAUDE_PID 2>/dev/null || H08_C="✅"
    H08_X="🔄"; kill -0 $H08_CODEX_PID 2>/dev/null || H08_X="✅"
    H09_C="🔄"; kill -0 $H09_CLAUDE_PID 2>/dev/null || H09_C="✅"
    H09_X="🔄"; kill -0 $H09_CODEX_PID 2>/dev/null || H09_X="✅"

    printf "\r⏱️  %ds | H07: C=%s X=%s | H08: C=%s X=%s | H09: C=%s X=%s" \
        "$ELAPSED" "$H07_C" "$H07_X" "$H08_C" "$H08_X" "$H09_C" "$H09_X"

    sleep 5
done

echo ""
echo ""
echo "✅ All SIX discovery agents completed!"
echo ""

wait $H07_CLAUDE_PID 2>/dev/null; H07_C_EXIT=$?
wait $H07_CODEX_PID 2>/dev/null; H07_X_EXIT=$?
wait $H08_CLAUDE_PID 2>/dev/null; H08_C_EXIT=$?
wait $H08_CODEX_PID 2>/dev/null; H08_X_EXIT=$?
wait $H09_CLAUDE_PID 2>/dev/null; H09_C_EXIT=$?
wait $H09_CODEX_PID 2>/dev/null; H09_X_EXIT=$?

echo "📊 EXECUTION SUMMARY:"
echo "======================"
echo "H07 Coagulation Hub - Claude: $H07_C_EXIT, Codex: $H07_X_EXIT"
echo "H08 S100 Calcium - Claude: $H08_C_EXIT, Codex: $H08_X_EXIT"
echo "H09 Temporal RNN - Claude: $H09_C_EXIT, Codex: $H09_X_EXIT"
echo ""

echo "🎉 Iteration 03 execution complete!"
echo "   Total hypotheses tested: 3 (emergent patterns)"
echo "   Total agents executed: 6"
echo "   Cumulative progress: 9/20 theories analyzed (45%)"
echo "   Remaining: 11 theories (Iterations 04-07)"
