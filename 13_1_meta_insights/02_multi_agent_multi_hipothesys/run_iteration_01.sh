#!/bin/bash

# Multi-Agent Multi-Hypothesis Iteration 01
# 6 agents: 2 per hypothesis × 3 hypotheses
# Claude Code + Codex for each hypothesis

set -e

echo "🚀 Multi-Hypothesis Discovery Framework - Iteration 01"
echo "======================================================"
echo "Hypotheses: 3 (H01, H02, H03)"
echo "Agents per hypothesis: 2 (Claude Code + Codex)"
echo "Total agents: 6"
echo ""

# Repository root
REPO_ROOT="/Users/Kravtsovd/projects/ecm-atlas"
ITERATION_DIR="${REPO_ROOT}/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_01"

echo "📁 Repository root: $REPO_ROOT"
echo "📂 Iteration directory: $ITERATION_DIR"
echo "🕐 Start time: $(date)"
echo ""

# Hypothesis directories
H01_DIR="${ITERATION_DIR}/hypothesis_01_compartment_mechanical_stress"
H02_DIR="${ITERATION_DIR}/hypothesis_02_serpin_cascade_dysregulation"
H03_DIR="${ITERATION_DIR}/hypothesis_03_tissue_aging_clocks"

# Agent output directories
H01_CLAUDE="${H01_DIR}/claude_code"
H01_CODEX="${H01_DIR}/codex"
H02_CLAUDE="${H02_DIR}/claude_code"
H02_CODEX="${H02_DIR}/codex"
H03_CLAUDE="${H03_DIR}/claude_code"
H03_CODEX="${H03_DIR}/codex"

# Task files
H01_TASK="${H01_DIR}/01_task.md"
H02_TASK="${H02_DIR}/01_task.md"
H03_TASK="${H03_DIR}/01_task.md"

# Verify task files exist
if [ ! -f "$H01_TASK" ]; then
    echo "❌ Error: H01 task file not found: $H01_TASK"
    exit 1
fi

if [ ! -f "$H02_TASK" ]; then
    echo "❌ Error: H02 task file not found: $H02_TASK"
    exit 1
fi

if [ ! -f "$H03_TASK" ]; then
    echo "❌ Error: H03 task file not found: $H03_TASK"
    exit 1
fi

echo "✓ All task files verified"
echo ""

# Prepare prompts for each agent

# ============================================================
# HYPOTHESIS 01: Compartment Mechanical Stress
# ============================================================

H01_CLAUDE_PROMPT="Read the task file at ${H01_TASK} and investigate compartment antagonistic mechanical stress adaptation.

CRITICAL REQUIREMENTS:
- Your agent name is 'claude_code'
- Your working directory is ${REPO_ROOT}
- Create ALL files in ${H01_CLAUDE}/
- Use prefix 'claude_code_' for all artifacts
- Follow Knowledge Framework standards from ${REPO_ROOT}/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md
- Create your plan in ${H01_CLAUDE}/01_plan_claude_code.md
- Create final results in ${H01_CLAUDE}/90_results_claude_code.md

Required artifacts:
1. 01_plan_claude_code.md - Analysis plan
2. analysis_claude_code.py - Python analysis script
3. antagonistic_pairs_claude_code.csv - Antagonistic protein-compartment pairs
4. mechanical_stress_correlation_claude_code.csv - Correlation statistics
5. visualizations_claude_code/ - Figures
6. 90_results_claude_code.md - Final results with self-evaluation

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

H01_CODEX_PROMPT="Read the task file at ${H01_TASK} and investigate compartment antagonistic mechanical stress adaptation.

CRITICAL REQUIREMENTS:
- Your agent name is 'codex'
- Your working directory is ${REPO_ROOT}
- Create ALL files in ${H01_CODEX}/
- Use prefix 'codex_' for all artifacts
- Follow Knowledge Framework standards from ${REPO_ROOT}/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md
- Create your plan in ${H01_CODEX}/01_plan_codex.md
- Create final results in ${H01_CODEX}/90_results_codex.md

Required artifacts:
1. 01_plan_codex.md - Analysis plan
2. analysis_codex.py - Python analysis script
3. antagonistic_pairs_codex.csv - Antagonistic protein-compartment pairs
4. mechanical_stress_correlation_codex.csv - Correlation statistics
5. visualizations_codex/ - Figures
6. 90_results_codex.md - Final results with self-evaluation

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

# ============================================================
# HYPOTHESIS 02: Serpin Cascade Dysregulation
# ============================================================

H02_CLAUDE_PROMPT="Read the task file at ${H02_TASK} and investigate serpin cascade dysregulation as central aging mechanism.

CRITICAL REQUIREMENTS:
- Your agent name is 'claude_code'
- Your working directory is ${REPO_ROOT}
- Create ALL files in ${H02_CLAUDE}/
- Use prefix 'claude_code_' for all artifacts
- Follow Knowledge Framework standards from ${REPO_ROOT}/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md
- Create your plan in ${H02_CLAUDE}/01_plan_claude_code.md
- Create final results in ${H02_CLAUDE}/90_results_claude_code.md

Required artifacts:
1. 01_plan_claude_code.md - Analysis plan
2. analysis_claude_code.py - Python analysis script
3. serpin_comprehensive_profile_claude_code.csv - All serpins with metrics
4. network_centrality_claude_code.csv - Centrality scores
5. pathway_dysregulation_claude_code.csv - Pathway statistics
6. visualizations_claude_code/ - Network graphs and figures
7. 90_results_claude_code.md - Final results with self-evaluation

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

H02_CODEX_PROMPT="Read the task file at ${H02_TASK} and investigate serpin cascade dysregulation as central aging mechanism.

CRITICAL REQUIREMENTS:
- Your agent name is 'codex'
- Your working directory is ${REPO_ROOT}
- Create ALL files in ${H02_CODEX}/
- Use prefix 'codex_' for all artifacts
- Follow Knowledge Framework standards from ${REPO_ROOT}/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md
- Create your plan in ${H02_CODEX}/01_plan_codex.md
- Create final results in ${H02_CODEX}/90_results_codex.md

Required artifacts:
1. 01_plan_codex.md - Analysis plan
2. analysis_codex.py - Python analysis script
3. serpin_comprehensive_profile_codex.csv - All serpins with metrics
4. network_centrality_codex.csv - Centrality scores
5. pathway_dysregulation_codex.csv - Pathway statistics
6. visualizations_codex/ - Network graphs and figures
7. 90_results_codex.md - Final results with self-evaluation

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

# ============================================================
# HYPOTHESIS 03: Tissue-Specific Aging Clocks
# ============================================================

H03_CLAUDE_PROMPT="Read the task file at ${H03_TASK} and investigate tissue-specific aging velocity clocks.

CRITICAL REQUIREMENTS:
- Your agent name is 'claude_code'
- Your working directory is ${REPO_ROOT}
- Create ALL files in ${H03_CLAUDE}/
- Use prefix 'claude_code_' for all artifacts
- Follow Knowledge Framework standards from ${REPO_ROOT}/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md
- Create your plan in ${H03_CLAUDE}/01_plan_claude_code.md
- Create final results in ${H03_CLAUDE}/90_results_claude_code.md

Required artifacts:
1. 01_plan_claude_code.md - Analysis plan
2. analysis_claude_code.py - Python analysis script
3. tissue_aging_velocity_claude_code.csv - Velocity for each tissue
4. tissue_specific_markers_claude_code.csv - TSI and markers
5. fast_aging_mechanisms_claude_code.csv - Shared proteins and pathways
6. visualizations_claude_code/ - Velocity charts and heatmaps
7. 90_results_claude_code.md - Final results with self-evaluation

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

H03_CODEX_PROMPT="Read the task file at ${H03_TASK} and investigate tissue-specific aging velocity clocks.

CRITICAL REQUIREMENTS:
- Your agent name is 'codex'
- Your working directory is ${REPO_ROOT}
- Create ALL files in ${H03_CODEX}/
- Use prefix 'codex_' for all artifacts
- Follow Knowledge Framework standards from ${REPO_ROOT}/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md
- Create your plan in ${H03_CODEX}/01_plan_codex.md
- Create final results in ${H03_CODEX}/90_results_codex.md

Required artifacts:
1. 01_plan_codex.md - Analysis plan
2. analysis_codex.py - Python analysis script
3. tissue_aging_velocity_codex.csv - Velocity for each tissue
4. tissue_specific_markers_codex.csv - TSI and markers
5. fast_aging_mechanisms_codex.csv - Shared proteins and pathways
6. visualizations_codex/ - Velocity charts and heatmaps
7. 90_results_codex.md - Final results with self-evaluation

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv"

echo "🚀 Launching SIX agents in parallel (3 hypotheses × 2 agents each)..."
echo ""

# Launch all agents in parallel

# H01 Claude Code
echo "Starting H01 - Claude Code Agent..."
(
    cd "$REPO_ROOT"
    echo "$H01_CLAUDE_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${H01_DIR}" \
        --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
        --add-dir "${REPO_ROOT}/13_1_meta_insights" \
        > "${H01_CLAUDE}/claude_code_output.log" 2>&1
    echo "H01 Claude Code exit code: $?" >> "${H01_CLAUDE}/claude_code_output.log"
) &
H01_CLAUDE_PID=$!

# H01 Codex
echo "Starting H01 - Codex Agent..."
(
    cd "$REPO_ROOT"
    codex exec --sandbox danger-full-access -C "$H01_CODEX" "$H01_CODEX_PROMPT" \
        > "${H01_CODEX}/codex_output.log" 2>&1
    echo "H01 Codex exit code: $?" >> "${H01_CODEX}/codex_output.log"
) &
H01_CODEX_PID=$!

# H02 Claude Code
echo "Starting H02 - Claude Code Agent..."
(
    cd "$REPO_ROOT"
    echo "$H02_CLAUDE_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${H02_DIR}" \
        --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
        --add-dir "${REPO_ROOT}/13_1_meta_insights" \
        > "${H02_CLAUDE}/claude_code_output.log" 2>&1
    echo "H02 Claude Code exit code: $?" >> "${H02_CLAUDE}/claude_code_output.log"
) &
H02_CLAUDE_PID=$!

# H02 Codex
echo "Starting H02 - Codex Agent..."
(
    cd "$REPO_ROOT"
    codex exec --sandbox danger-full-access -C "$H02_CODEX" "$H02_CODEX_PROMPT" \
        > "${H02_CODEX}/codex_output.log" 2>&1
    echo "H02 Codex exit code: $?" >> "${H02_CODEX}/codex_output.log"
) &
H02_CODEX_PID=$!

# H03 Claude Code
echo "Starting H03 - Claude Code Agent..."
(
    cd "$REPO_ROOT"
    echo "$H03_CLAUDE_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${H03_DIR}" \
        --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
        --add-dir "${REPO_ROOT}/13_1_meta_insights" \
        > "${H03_CLAUDE}/claude_code_output.log" 2>&1
    echo "H03 Claude Code exit code: $?" >> "${H03_CLAUDE}/claude_code_output.log"
) &
H03_CLAUDE_PID=$!

# H03 Codex
echo "Starting H03 - Codex Agent..."
(
    cd "$REPO_ROOT"
    codex exec --sandbox danger-full-access -C "$H03_CODEX" "$H03_CODEX_PROMPT" \
        > "${H03_CODEX}/codex_output.log" 2>&1
    echo "H03 Codex exit code: $?" >> "${H03_CODEX}/codex_output.log"
) &
H03_CODEX_PID=$!

echo ""
echo "🔍 Agents launched:"
echo "   H01 Claude Code PID: $H01_CLAUDE_PID"
echo "   H01 Codex PID: $H01_CODEX_PID"
echo "   H02 Claude Code PID: $H02_CLAUDE_PID"
echo "   H02 Codex PID: $H02_CODEX_PID"
echo "   H03 Claude Code PID: $H03_CLAUDE_PID"
echo "   H03 Codex PID: $H03_CODEX_PID"
echo ""

echo "⏳ Monitoring agent progress..."
echo "   Monitor logs:"
echo "   tail -f ${H01_CLAUDE}/claude_code_output.log"
echo "   tail -f ${H01_CODEX}/codex_output.log"
echo "   tail -f ${H02_CLAUDE}/claude_code_output.log"
echo "   tail -f ${H02_CODEX}/codex_output.log"
echo "   tail -f ${H03_CLAUDE}/claude_code_output.log"
echo "   tail -f ${H03_CODEX}/codex_output.log"
echo ""

# Monitor progress
START_TIME=$(date +%s)

while kill -0 $H01_CLAUDE_PID 2>/dev/null || \
      kill -0 $H01_CODEX_PID 2>/dev/null || \
      kill -0 $H02_CLAUDE_PID 2>/dev/null || \
      kill -0 $H02_CODEX_PID 2>/dev/null || \
      kill -0 $H03_CLAUDE_PID 2>/dev/null || \
      kill -0 $H03_CODEX_PID 2>/dev/null; do

    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    # Check status of each agent
    H01_C_STATUS="🔄"; kill -0 $H01_CLAUDE_PID 2>/dev/null || H01_C_STATUS="✅"
    H01_X_STATUS="🔄"; kill -0 $H01_CODEX_PID 2>/dev/null || H01_X_STATUS="✅"
    H02_C_STATUS="🔄"; kill -0 $H02_CLAUDE_PID 2>/dev/null || H02_C_STATUS="✅"
    H02_X_STATUS="🔄"; kill -0 $H02_CODEX_PID 2>/dev/null || H02_X_STATUS="✅"
    H03_C_STATUS="🔄"; kill -0 $H03_CLAUDE_PID 2>/dev/null || H03_C_STATUS="✅"
    H03_X_STATUS="🔄"; kill -0 $H03_CODEX_PID 2>/dev/null || H03_X_STATUS="✅"

    printf "\r⏱️  %ds | H01: C=%s X=%s | H02: C=%s X=%s | H03: C=%s X=%s" \
        "$ELAPSED" "$H01_C_STATUS" "$H01_X_STATUS" "$H02_C_STATUS" "$H02_X_STATUS" "$H03_C_STATUS" "$H03_X_STATUS"

    sleep 5
done

echo ""
echo ""
echo "✅ All SIX agents completed!"
echo ""

# Wait for processes and get exit codes
wait $H01_CLAUDE_PID 2>/dev/null; H01_CLAUDE_EXIT=$?
wait $H01_CODEX_PID 2>/dev/null; H01_CODEX_EXIT=$?
wait $H02_CLAUDE_PID 2>/dev/null; H02_CLAUDE_EXIT=$?
wait $H02_CODEX_PID 2>/dev/null; H02_CODEX_EXIT=$?
wait $H03_CLAUDE_PID 2>/dev/null; H03_CLAUDE_EXIT=$?
wait $H03_CODEX_PID 2>/dev/null; H03_CODEX_EXIT=$?

echo "📊 EXECUTION SUMMARY:"
echo "===================="
echo "H01 Claude Code exit code: $H01_CLAUDE_EXIT"
echo "H01 Codex exit code: $H01_CODEX_EXIT"
echo "H02 Claude Code exit code: $H02_CLAUDE_EXIT"
echo "H02 Codex exit code: $H02_CODEX_EXIT"
echo "H03 Claude Code exit code: $H03_CLAUDE_EXIT"
echo "H03 Codex exit code: $H03_CODEX_EXIT"
echo ""

echo "📝 OUTPUT FILES:"
echo "==============="
for hypothesis_dir in "$H01_DIR" "$H02_DIR" "$H03_DIR"; do
    hypothesis_name=$(basename "$hypothesis_dir")
    echo ""
    echo "$hypothesis_name:"

    for agent_dir in "$hypothesis_dir/claude_code" "$hypothesis_dir/codex"; do
        agent_name=$(basename "$agent_dir")
        echo "  $agent_name:"

        [ -f "${agent_dir}/01_plan_${agent_name}.md" ] && echo "    ✅ Plan" || echo "    ❌ Plan missing"
        [ -f "${agent_dir}/90_results_${agent_name}.md" ] && echo "    ✅ Results" || echo "    ❌ Results missing"

        CSV_COUNT=$(ls "${agent_dir}"/*.csv 2>/dev/null | wc -l)
        PY_COUNT=$(ls "${agent_dir}"/*.py 2>/dev/null | wc -l)
        echo "    ✅ CSV files: $CSV_COUNT"
        echo "    ✅ Python files: $PY_COUNT"
    done
done

echo ""
echo "📍 Next steps:"
echo "  1. Review all 90_results_*.md files"
echo "  2. Compare Claude Code vs Codex for each hypothesis"
echo "  3. Compile iteration 01 results"
echo "  4. Rank hypotheses by Novelty + Impact scores"
echo "  5. Generate hypotheses for iteration 02"
echo ""
echo "📂 All outputs in: ${ITERATION_DIR}/"
echo ""
echo "🎉 Iteration 01 multi-agent execution complete!"
echo "   Total hypotheses tested: 3"
echo "   Total agents executed: 6"
echo "   Progress: 3/20 theories analyzed"
