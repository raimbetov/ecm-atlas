#!/bin/bash

# Multi-Agent Orchestrator - Meta-Insights Validation
# 2 Claude Code + 1 Codex
# Author: Daniel Kravtsov
# Date: 2025-10-18

set -e

echo "🚀 Multi-Agent Orchestrator - Meta-Insights Validation (2 Claude + 1 Codex)"
echo "============================================================================="

if [ -z "$1" ]; then
    echo "❌ Usage: $0 <task_file.md>"
    echo "   Example: $0 01_task_meta_insights_validation.md"
    exit 1
fi

TASK_FILE="$1"

# Convert to absolute path if relative
if [[ ! "$TASK_FILE" = /* ]]; then
    TASK_FILE="$(pwd)/$TASK_FILE"
fi

# Check if task file exists
if [ ! -f "$TASK_FILE" ]; then
    echo "❌ Error: Task file not found: $TASK_FILE"
    exit 1
fi

# Get directory of task file for output
TASK_DIR=$(dirname "$TASK_FILE")
TASK_NAME=$(basename "$TASK_FILE" .md)

# Create output directories next to task file
CLAUDE_1_OUTPUT_DIR="${TASK_DIR}/claude_1"
CLAUDE_2_OUTPUT_DIR="${TASK_DIR}/claude_2"
CODEX_OUTPUT_DIR="${TASK_DIR}/codex"

echo "📋 Task: $TASK_FILE"
echo "📂 Output directories:"
echo "   - Claude Agent 1: $CLAUDE_1_OUTPUT_DIR"
echo "   - Claude Agent 2: $CLAUDE_2_OUTPUT_DIR"
echo "   - Codex: $CODEX_OUTPUT_DIR"
echo "🕐 Start time: $(date)"
echo ""

# Create output directories
mkdir -p "$CLAUDE_1_OUTPUT_DIR"
mkdir -p "$CLAUDE_2_OUTPUT_DIR"
mkdir -p "$CODEX_OUTPUT_DIR"

# Get the repository root
if git -C "$TASK_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT=$(git -C "$TASK_DIR" rev-parse --show-toplevel)
else
    REPO_ROOT="$TASK_DIR"
fi

echo "📁 Repository root: $REPO_ROOT"
echo ""

# Prepare prompts for agents with absolute paths
CLAUDE_1_PROMPT="Read the task file at ${TASK_FILE} and validate ALL meta-insights on the batch-corrected V2 dataset.

CRITICAL REQUIREMENTS:
- Your agent name is 'claude_1'
- Your working directory is ${REPO_ROOT}
- Create ALL files in ${CLAUDE_1_OUTPUT_DIR}/
- Use prefix 'claude_1_' for all artifacts
- Follow Knowledge Framework standards from ${REPO_ROOT}/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md
- Create your plan in ${CLAUDE_1_OUTPUT_DIR}/01_plan_claude_1.md
- Create final results in ${CLAUDE_1_OUTPUT_DIR}/90_results_claude_1.md

Required artifacts:
1. 01_plan_claude_1.md - Validation plan
2. validation_results_claude_1.csv - V1 vs V2 comparison for all insights
3. new_discoveries_claude_1.csv - Emergent findings (if any)
4. validation_pipeline_claude_1.py - Python validation script
5. v2_validated_proteins_claude_1.csv - NEW batch-corrected data subset
6. 90_results_claude_1.md - Final results with self-evaluation

Self-evaluation criteria:
- Completeness: Validate 7 GOLD + 5 SILVER insights (40 points)
- Accuracy: Correct metrics computation (30 points)
- Insights: Identify new discoveries (20 points)
- Reproducibility: Working Python script (10 points)

Dataset: /Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv

Original insights: /Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/"

CLAUDE_2_PROMPT="Read the task file at ${TASK_FILE} and validate ALL meta-insights on the batch-corrected V2 dataset.

CRITICAL REQUIREMENTS:
- Your agent name is 'claude_2'
- Your working directory is ${REPO_ROOT}
- Create ALL files in ${CLAUDE_2_OUTPUT_DIR}/
- Use prefix 'claude_2_' for all artifacts
- Follow Knowledge Framework standards from ${REPO_ROOT}/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md
- Create your plan in ${CLAUDE_2_OUTPUT_DIR}/01_plan_claude_2.md
- Create final results in ${CLAUDE_2_OUTPUT_DIR}/90_results_claude_2.md

Required artifacts:
1. 01_plan_claude_2.md - Validation plan
2. validation_results_claude_2.csv - V1 vs V2 comparison for all insights
3. new_discoveries_claude_2.csv - Emergent findings (if any)
4. validation_pipeline_claude_2.py - Python validation script
5. v2_validated_proteins_claude_2.csv - NEW batch-corrected data subset
6. 90_results_claude_2.md - Final results with self-evaluation

Self-evaluation criteria:
- Completeness: Validate 7 GOLD + 5 SILVER insights (40 points)
- Accuracy: Correct metrics computation (30 points)
- Insights: Identify new discoveries (20 points)
- Reproducibility: Working Python script (10 points)

Dataset: /Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv

Original insights: /Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/"

CODEX_PROMPT="Read the task file at ${TASK_FILE} and validate ALL meta-insights on the batch-corrected V2 dataset.

CRITICAL REQUIREMENTS:
- Your agent name is 'codex'
- Your working directory is ${REPO_ROOT}
- Create ALL files in ${CODEX_OUTPUT_DIR}/
- Use prefix 'codex_' for all artifacts
- Follow Knowledge Framework standards from ${REPO_ROOT}/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md
- Create your plan in ${CODEX_OUTPUT_DIR}/01_plan_codex.md
- Create final results in ${CODEX_OUTPUT_DIR}/90_results_codex.md

Required artifacts:
1. 01_plan_codex.md - Validation plan
2. validation_results_codex.csv - V1 vs V2 comparison for all insights
3. new_discoveries_codex.csv - Emergent findings (if any)
4. validation_pipeline_codex.py - Python validation script
5. v2_validated_proteins_codex.csv - NEW batch-corrected data subset
6. 90_results_codex.md - Final results with self-evaluation

Self-evaluation criteria:
- Completeness: Validate 7 GOLD + 5 SILVER insights (40 points)
- Accuracy: Correct metrics computation (30 points)
- Insights: Identify new discoveries (20 points)
- Reproducibility: Working Python script (10 points)

Dataset: /Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv

Original insights: /Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/"

echo "🚀 Launching THREE agents in parallel..."
echo ""

# Launch Claude Code Agent 1
echo "Starting Claude Code Agent 1..."
(
    cd "$REPO_ROOT"
    echo "$CLAUDE_1_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${TASK_DIR}" \
        --add-dir "${REPO_ROOT}/13_meta_insights" \
        --add-dir "${REPO_ROOT}/14_exploratory_batch_correction" \
        > "${CLAUDE_1_OUTPUT_DIR}/claude_1_output.log" 2>&1
    echo "Claude 1 exit code: $?" >> "${CLAUDE_1_OUTPUT_DIR}/claude_1_output.log"
) &
CLAUDE_1_PID=$!

# Launch Claude Code Agent 2
echo "Starting Claude Code Agent 2..."
(
    cd "$REPO_ROOT"
    echo "$CLAUDE_2_PROMPT" | claude --print \
        --permission-mode bypassPermissions \
        --add-dir "${TASK_DIR}" \
        --add-dir "${REPO_ROOT}/13_meta_insights" \
        --add-dir "${REPO_ROOT}/14_exploratory_batch_correction" \
        > "${CLAUDE_2_OUTPUT_DIR}/claude_2_output.log" 2>&1
    echo "Claude 2 exit code: $?" >> "${CLAUDE_2_OUTPUT_DIR}/claude_2_output.log"
) &
CLAUDE_2_PID=$!

# Launch Codex agent
echo "Starting Codex agent..."
(
    cd "$REPO_ROOT"
    codex exec --sandbox danger-full-access -C "$CODEX_OUTPUT_DIR" "$CODEX_PROMPT" \
        > "${CODEX_OUTPUT_DIR}/codex_output.log" 2>&1
    echo "Codex exit code: $?" >> "${CODEX_OUTPUT_DIR}/codex_output.log"
) &
CODEX_PID=$!

echo "🔍 Agents launched:"
echo "   - Claude Code Agent 1 PID: $CLAUDE_1_PID"
echo "   - Claude Code Agent 2 PID: $CLAUDE_2_PID"
echo "   - Codex PID: $CODEX_PID"
echo ""

echo "⏳ Monitoring agent progress..."
echo "   Tip: You can monitor logs in real-time with:"
echo "   tail -f ${CLAUDE_1_OUTPUT_DIR}/claude_1_output.log"
echo "   tail -f ${CLAUDE_2_OUTPUT_DIR}/claude_2_output.log"
echo "   tail -f ${CODEX_OUTPUT_DIR}/codex_output.log"
echo ""

# Wait for all agents to complete with progress monitoring
START_TIME=$(date +%s)
LAST_CLAUDE_1_LINES=0
LAST_CLAUDE_2_LINES=0
LAST_CODEX_LINES=0

while kill -0 $CLAUDE_1_PID 2>/dev/null || kill -0 $CLAUDE_2_PID 2>/dev/null || kill -0 $CODEX_PID 2>/dev/null; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    # Check Claude 1 status
    CLAUDE_1_STATUS="🔄 running"
    if ! kill -0 $CLAUDE_1_PID 2>/dev/null; then
        CLAUDE_1_STATUS="✅ completed"
    elif [ -f "${CLAUDE_1_OUTPUT_DIR}/claude_1_output.log" ]; then
        CLAUDE_1_LINES=$(wc -l < "${CLAUDE_1_OUTPUT_DIR}/claude_1_output.log" 2>/dev/null || echo 0)
        if [ "$CLAUDE_1_LINES" -gt "$LAST_CLAUDE_1_LINES" ]; then
            CLAUDE_1_STATUS="📝 writing (${CLAUDE_1_LINES} lines)"
            LAST_CLAUDE_1_LINES=$CLAUDE_1_LINES
        fi
    fi

    # Check Claude 2 status
    CLAUDE_2_STATUS="🔄 running"
    if ! kill -0 $CLAUDE_2_PID 2>/dev/null; then
        CLAUDE_2_STATUS="✅ completed"
    elif [ -f "${CLAUDE_2_OUTPUT_DIR}/claude_2_output.log" ]; then
        CLAUDE_2_LINES=$(wc -l < "${CLAUDE_2_OUTPUT_DIR}/claude_2_output.log" 2>/dev/null || echo 0)
        if [ "$CLAUDE_2_LINES" -gt "$LAST_CLAUDE_2_LINES" ]; then
            CLAUDE_2_STATUS="📝 writing (${CLAUDE_2_LINES} lines)"
            LAST_CLAUDE_2_LINES=$CLAUDE_2_LINES
        fi
    fi

    # Check Codex status
    CODEX_STATUS="🔄 running"
    if ! kill -0 $CODEX_PID 2>/dev/null; then
        CODEX_STATUS="✅ completed"
    elif [ -f "${CODEX_OUTPUT_DIR}/codex_output.log" ]; then
        CODEX_LINES=$(wc -l < "${CODEX_OUTPUT_DIR}/codex_output.log" 2>/dev/null || echo 0)
        if [ "$CODEX_LINES" -gt "$LAST_CODEX_LINES" ]; then
            CODEX_STATUS="📝 writing (${CODEX_LINES} lines)"
            LAST_CODEX_LINES=$CODEX_LINES
        fi
    fi

    # Display progress
    printf "\r⏱️  %ds | Claude 1: %s | Claude 2: %s | Codex: %s" "$ELAPSED" "$CLAUDE_1_STATUS" "$CLAUDE_2_STATUS" "$CODEX_STATUS"

    sleep 5
done

echo ""
echo ""
echo "✅ All THREE agents completed!"
echo ""

# Wait for processes to finish and get exit codes
wait $CLAUDE_1_PID 2>/dev/null
CLAUDE_1_EXIT=$?

wait $CLAUDE_2_PID 2>/dev/null
CLAUDE_2_EXIT=$?

wait $CODEX_PID 2>/dev/null
CODEX_EXIT=$?

echo "📊 EXECUTION SUMMARY:"
echo "===================="
echo "Claude Code Agent 1 exit code: $CLAUDE_1_EXIT"
echo "Claude Code Agent 2 exit code: $CLAUDE_2_EXIT"
echo "Codex exit code: $CODEX_EXIT"
echo ""

# Check outputs
echo "📝 OUTPUT FILES:"
echo "==============="

# Check Claude 1
echo "Claude Agent 1 outputs:"
CLAUDE_1_FILES=0
[ -f "${CLAUDE_1_OUTPUT_DIR}/01_plan_claude_1.md" ] && { echo "  ✅ Plan created"; CLAUDE_1_FILES=$((CLAUDE_1_FILES + 1)); } || echo "  ❌ Plan missing"
[ -f "${CLAUDE_1_OUTPUT_DIR}/90_results_claude_1.md" ] && { echo "  ✅ Results created"; CLAUDE_1_FILES=$((CLAUDE_1_FILES + 1)); } || echo "  ❌ Results missing"
[ -f "${CLAUDE_1_OUTPUT_DIR}/validation_results_claude_1.csv" ] && { echo "  ✅ Validation CSV created"; CLAUDE_1_FILES=$((CLAUDE_1_FILES + 1)); } || echo "  ❌ Validation CSV missing"
if ls "${CLAUDE_1_OUTPUT_DIR}"/*.py >/dev/null 2>&1; then
    PY_COUNT=$(ls "${CLAUDE_1_OUTPUT_DIR}"/*.py | wc -l)
    echo "  ✅ Python files: $PY_COUNT"
    CLAUDE_1_FILES=$((CLAUDE_1_FILES + PY_COUNT))
fi
if ls "${CLAUDE_1_OUTPUT_DIR}"/*.csv >/dev/null 2>&1; then
    CSV_COUNT=$(ls "${CLAUDE_1_OUTPUT_DIR}"/*.csv | wc -l)
    echo "  ✅ CSV files: $CSV_COUNT"
    CLAUDE_1_FILES=$((CLAUDE_1_FILES + CSV_COUNT))
fi

echo ""

# Check Claude 2
echo "Claude Agent 2 outputs:"
CLAUDE_2_FILES=0
[ -f "${CLAUDE_2_OUTPUT_DIR}/01_plan_claude_2.md" ] && { echo "  ✅ Plan created"; CLAUDE_2_FILES=$((CLAUDE_2_FILES + 1)); } || echo "  ❌ Plan missing"
[ -f "${CLAUDE_2_OUTPUT_DIR}/90_results_claude_2.md" ] && { echo "  ✅ Results created"; CLAUDE_2_FILES=$((CLAUDE_2_FILES + 1)); } || echo "  ❌ Results missing"
[ -f "${CLAUDE_2_OUTPUT_DIR}/validation_results_claude_2.csv" ] && { echo "  ✅ Validation CSV created"; CLAUDE_2_FILES=$((CLAUDE_2_FILES + 1)); } || echo "  ❌ Validation CSV missing"
if ls "${CLAUDE_2_OUTPUT_DIR}"/*.py >/dev/null 2>&1; then
    PY_COUNT=$(ls "${CLAUDE_2_OUTPUT_DIR}"/*.py | wc -l)
    echo "  ✅ Python files: $PY_COUNT"
    CLAUDE_2_FILES=$((CLAUDE_2_FILES + PY_COUNT))
fi
if ls "${CLAUDE_2_OUTPUT_DIR}"/*.csv >/dev/null 2>&1; then
    CSV_COUNT=$(ls "${CLAUDE_2_OUTPUT_DIR}"/*.csv | wc -l)
    echo "  ✅ CSV files: $CSV_COUNT"
    CLAUDE_2_FILES=$((CLAUDE_2_FILES + CSV_COUNT))
fi

echo ""

# Check Codex
echo "Codex outputs:"
CODEX_FILES=0
[ -f "${CODEX_OUTPUT_DIR}/01_plan_codex.md" ] && { echo "  ✅ Plan created"; CODEX_FILES=$((CODEX_FILES + 1)); } || echo "  ❌ Plan missing"
[ -f "${CODEX_OUTPUT_DIR}/90_results_codex.md" ] && { echo "  ✅ Results created"; CODEX_FILES=$((CODEX_FILES + 1)); } || echo "  ❌ Results missing"
[ -f "${CODEX_OUTPUT_DIR}/validation_results_codex.csv" ] && { echo "  ✅ Validation CSV created"; CODEX_FILES=$((CODEX_FILES + 1)); } || echo "  ❌ Validation CSV missing"
if ls "${CODEX_OUTPUT_DIR}"/*.py >/dev/null 2>&1; then
    PY_COUNT=$(ls "${CODEX_OUTPUT_DIR}"/*.py | wc -l)
    echo "  ✅ Python files: $PY_COUNT"
    CODEX_FILES=$((CODEX_FILES + PY_COUNT))
fi
if ls "${CODEX_OUTPUT_DIR}"/*.csv >/dev/null 2>&1; then
    CSV_COUNT=$(ls "${CODEX_OUTPUT_DIR}"/*.csv | wc -l)
    echo "  ✅ CSV files: $CSV_COUNT"
    CODEX_FILES=$((CODEX_FILES + CSV_COUNT))
fi

echo ""
echo "🏁 ARTIFACT COMPARISON:"
echo "======================"
echo "Claude Agent 1 artifacts created: $CLAUDE_1_FILES"
echo "Claude Agent 2 artifacts created: $CLAUDE_2_FILES"
echo "Codex artifacts created: $CODEX_FILES"

echo ""
echo "📍 Next steps:"
echo "  1. Review the log files for any errors"
echo "  2. Compare validation_results CSV files across agents"
echo "  3. Review the 90_results files for self-evaluation scores"
echo "  4. Compare which insights were CONFIRMED vs MODIFIED vs REJECTED"
echo "  5. Commit and push results"
echo "  6. Create compilation report comparing all 3 agents"
echo ""
echo "📂 All outputs saved in:"
echo "  ${TASK_DIR}/"
echo ""
echo "🎉 Multi-agent execution complete (2 Claude + 1 Codex)!"
