#!/bin/bash

# Multi-Agent Multi-Hypothesis Iteration 06 - BROWSER AUTOMATION FOR EXTERNAL DATA
# 2 agents: Claude Code + Codex
# Critical focus: Unblock H16 External Validation by downloading supplementary files via Playwright

set -e

echo "🔬🌐 Multi-Hypothesis Discovery Framework - Iteration 06 (BROWSER AUTOMATION)"
echo "======================================================================================"
echo "Hypothesis: 1 (H21) - BROWSER AUTOMATION FOR EXTERNAL DATA ACQUISITION!"
echo "Agents per hypothesis: 2 (Claude Code + Codex)"
echo "Total agents: 2"
echo "Focus: Playwright automation to download supplementary files from eLife, PMC, Nature"
echo ""

# Repository root
REPO_ROOT="/Users/Kravtsovd/projects/ecm-atlas"
ITERATION_DIR="${REPO_ROOT}/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_06"

echo "📁 Repository root: $REPO_ROOT"
echo "📂 Iteration directory: $ITERATION_DIR"
echo "🕐 Start time: $(date)"
echo ""

# Hypothesis directory
H21_DIR="${ITERATION_DIR}/hypothesis_21_browser_automation"

# Agent output directories
H21_CLAUDE="${H21_DIR}/claude_code"
H21_CODEX="${H21_DIR}/codex"

# Task file
H21_TASK="${H21_DIR}/01_task.md"

# Verify task file exists
if [ ! -f "$H21_TASK" ]; then
    echo "❌ Error: Task file not found: $H21_TASK"
    exit 1
fi

echo "✓ Task file verified"
echo ""

# ============================================================
# HYPOTHESIS 21: BROWSER AUTOMATION FOR EXTERNAL DATA
# ============================================================

H21_CLAUDE_PROMPT="🌐🤖 CRITICAL: Unblock H16 with Browser Automation! 🤖🌐

Read ${H21_TASK} - Download ALL 6 external datasets using Playwright!

H16 BLOCKER (Iteration 05):
✅ External validation framework implemented (635 lines)
✅ 6 datasets identified (PXD011967, PXD015982, +4)
✅ Validation pipeline ready
❌ BLOCKED: Cannot download supplementary files programmatically!

ROOT CAUSE:
- PRIDE/MassIVE only have RAW MS files (hundreds of GB)
- Journal APIs (eLife, PMC) don't provide supplementary file access
- Files require JavaScript interaction (browser-only download)

YOUR MISSION: Implement Playwright downloader + download ALL 6 datasets!

🚨 MANDATORY TASKS (IN ORDER):

1. SETUP PLAYWRIGHT:
   pip install playwright pandas requests beautifulsoup4
   playwright install chromium

   # Verify installation
   Test basic Playwright functionality (navigate to example.com)

2. READ H16 RESULTS FOR DATASET LIST:
   File: /Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_16_h13_validation_completion/claude_code/90_results_claude_code.md

   Extract:
   - 6 dataset IDs (PXD011967, PXD015982, etc.)
   - Journal sources (eLife, PMC, Nature, Cell)
   - DOI or PMID for each dataset
   - Expected supplementary file names

3. IMPLEMENT DOWNLOADER CLASS:
   Create: external_data_downloader.py

   class SupplementaryDownloader:
       def __init__(self, headless=True):
           # Launch chromium browser
           # Set viewport, user-agent, accept downloads

       def download_elife(self, doi, output_dir):
           # Navigate to eLife article
           # Wait for supplementary section (JavaScript rendering)
           # Find \"Supplementary file 1\" link
           # Click and save download

       def download_pmc(self, pmid, output_dir):
           # Navigate to Europe PMC
           # Wait for supplementary section
           # Click download button
           # Save file

       def download_nature(self, doi, output_dir):
           # Navigate to Nature article
           # Find supplementary data link
           # Download first supplementary table

       def validate_download(self, file_path):
           # Check file exists, non-empty
           # Parse as Excel/CSV/TSV
           # Return: rows, columns, size, MD5 hash

4. DOWNLOAD ALL 6 DATASETS:
   Create: download_all_datasets.py

   For each dataset:
   - Call appropriate download method (elife/pmc/nature)
   - Retry 3 times if failure (exponential backoff)
   - Validate download (format, size, content)
   - Save metadata.json (timestamp, MD5, validation results)
   - Rate limit: 5 seconds between downloads

   Save to: external_datasets/{PXD_ID}/raw_data.{xlsx|csv|tsv}

5. VALIDATION CHECKS:
   For each downloaded file:
   ✓ File size >1KB (not empty)
   ✓ Valid format (Excel/CSV/TSV parseable)
   ✓ Rows ≥10 (not truncated)
   ✓ Columns ≥5 (has protein abundance data)
   ✓ MD5 hash logged

   Success criteria: ≥5/6 datasets downloaded and validated

6. RE-RUN H16 VALIDATION PIPELINE:
   Load H16 framework: h13_completion_claude_code.py

   Test external validation:
   a) H08 S100 Model:
      - Load model: /iterations/iteration_03/hypothesis_08_s100_calcium_signaling/claude_code/s100_stiffness_model_claude_code.pth
      - Predict on external datasets
      - Target: Mean external R²≥0.60

   b) H06 Biomarker Panel:
      - Test 8-protein panel on external data
      - Target: Mean external AUC≥0.80

   c) H03 Tissue Velocities:
      - Compute velocities on external data
      - Correlation with internal velocities
      - Target: Mean ρ≥0.70

7. META-ANALYSIS:
   Combine internal + external datasets
   - Random-effects model
   - I² heterogeneity testing
   - Identify STABLE proteins (I²<25%)
   - Identify VARIABLE proteins (I²>75%)

TECHNICAL REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H21_CLAUDE}/
- Output: external_datasets/ directory (create if not exists)

Dataset locations:
- Merged data: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv
- H16 results: /iterations/iteration_05/hypothesis_16_h13_validation_completion/claude_code/
- Models: /iterations/iteration_03/hypothesis_08_s100_calcium_signaling/claude_code/

SUCCESS CRITERIA:
✅ 6/6 datasets downloaded (or ≥5/6 if CAPTCHA blocks one)
✅ All validations pass
✅ H16 validation executed successfully
✅ External R²/AUC/ρ reported for H08/H06/H03

⚠️ CRITICAL: This unblocks H16, which validates ALL H01-H15 findings! ⚠️
⚠️ Without external validation, publication is NOT credible! ⚠️

🎯 DELIVERABLES:
1. external_data_downloader.py (Playwright class)
2. download_all_datasets.py (execution script)
3. external_datasets/{PXD_ID}/ (6 directories with files + metadata)
4. download_results.json (summary of all downloads)
5. external_validation_results.csv (H08/H06/H03 performance)
6. 90_results_claude_code.md (final report)

GO!"

H21_CODEX_PROMPT="🌐 Browser Automation for External Data Download 🌐

Read ${H21_TASK} - Unblock H16 by downloading supplementary files!

BLOCKER: H16 needs external proteomics data but can't download from journals.

YOUR MISSION:
1. Install Playwright (pip install playwright, playwright install chromium)
2. Read H16 results for dataset list (6 datasets)
3. Implement SupplementaryDownloader class (eLife, PMC, Nature methods)
4. Download ALL 6 datasets with retry logic
5. Validate downloads (format, size, content)
6. Re-run H16 validation pipeline
7. Report external R²/AUC/ρ

Alternative Approach (Codex):
- Try Selenium instead of Playwright (comparison)
- Test BeautifulSoup + Requests first (may work for some journals)
- Benchmark: Playwright vs Selenium speed/reliability

Requirements:
- Agent: 'codex'
- Workspace: ${H21_CODEX}/
- Output: external_datasets/

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

Target: ≥5/6 downloads successful, H08 R²≥0.60, H06 AUC≥0.80, H03 ρ≥0.70"

# ============================================================
# LAUNCH BOTH AGENTS IN PARALLEL
# ============================================================

echo "🚀 Launching 2 agents in parallel..."
echo ""

# Track PIDs
declare -a PIDS
declare -a AGENT_NAMES

# H21 Claude Code
echo "▶️  H21 Claude Code (Playwright Browser Automation)..."
(cd "$REPO_ROOT"
 echo "$H21_CLAUDE_PROMPT" | claude --print \
     --permission-mode bypassPermissions \
     --add-dir "${H21_CLAUDE}" \
     --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
     --add-dir "${REPO_ROOT}/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_16_h13_validation_completion" \
     > "${H21_CLAUDE}/h21claude_output.log" 2>&1) &
PIDS+=($!)
AGENT_NAMES+=("H21_Claude")

# H21 Codex
echo "▶️  H21 Codex (Selenium Alternative)..."
(cd "$REPO_ROOT"
 codex exec --sandbox danger-full-access -C "$H21_CODEX" "$H21_CODEX_PROMPT" \
     > "${H21_CODEX}/h21codex_output.log" 2>&1) &
PIDS+=($!)
AGENT_NAMES+=("H21_Codex")

echo ""
echo "✓ Both agents launched!"
echo ""

# ============================================================
# MONITOR PROGRESS
# ============================================================

echo "⏳ Monitoring agent execution..."
echo "You can tail individual logs:"
echo "  H21_Claude: tail -f ${H21_CLAUDE}/h21claude_output.log"
echo "  H21_Codex: tail -f ${H21_CODEX}/h21codex_output.log"
echo ""

# Wait for all agents
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    NAME=${AGENT_NAMES[$i]}

    echo "⏳ Waiting for ${NAME} (PID: $PID)..."
    wait $PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ ${NAME} completed successfully"
    else
        echo "❌ ${NAME} failed with exit code $EXIT_CODE"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "===================================================================================="
echo "🏁 Iteration 06 COMPLETE!"
echo "===================================================================================="
echo "🕐 End time: $(date)"
echo ""
echo "📊 Summary:"
echo "  Total agents: 2"
echo "  Successful: $((2 - FAILED))"
echo "  Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ ALL AGENTS COMPLETED SUCCESSFULLY!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Review download results in external_datasets/"
    echo "2. Check external validation results in hypothesis_21_browser_automation/"
    echo "3. Compare Claude Code (Playwright) vs Codex (Selenium) approaches"
    echo "4. Update H16 with successful external validation"
    echo "5. Synthesize ALL Iterations 01-06 for final publication!"
else
    echo "⚠️  Some agents failed. Review logs above."
    echo "   Check output logs for error details."
    echo "   Manual download may be required for blocked datasets."
fi

echo ""
echo "🎯 PRIORITY: If downloads successful, H16 External Validation is COMPLETE!"
echo "🎯 This validates ALL H01-H20 findings! Ready for publication!"
echo ""

exit $FAILED
