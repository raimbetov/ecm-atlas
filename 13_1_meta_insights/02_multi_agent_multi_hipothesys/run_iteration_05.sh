#!/bin/bash

# Multi-Agent Multi-Hypothesis Iteration 05 - VALIDATION & TRANSLATION
# 10 agents: 2 per hypothesis × 5 hypotheses
# Critical focus: External validation (H13 completion), drug targets, multi-modal AI, metabolomics, cross-species

set -e

echo "🔬🧬 Multi-Hypothesis Discovery Framework - Iteration 05 (VALIDATION & CLINICAL TRANSLATION)"
echo "=============================================================================================="
echo "Hypotheses: 5 (H16-H20) - EXTERNAL VALIDATION + DRUG TARGETS + INTEGRATION!"
echo "Agents per hypothesis: 2 (Claude Code + Codex)"
echo "Total agents: 10"
echo "Focus: H13 completion, SERPINE1 drug target, Multi-modal AI, Metabolomics Phase I, Cross-species conservation"
echo ""

# Repository root
REPO_ROOT="/Users/Kravtsovd/projects/ecm-atlas"
ITERATION_DIR="${REPO_ROOT}/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05"

echo "📁 Repository root: $REPO_ROOT"
echo "📂 Iteration directory: $ITERATION_DIR"
echo "🕐 Start time: $(date)"
echo ""

# Hypothesis directories
H16_DIR="${ITERATION_DIR}/hypothesis_16_h13_validation_completion"
H17_DIR="${ITERATION_DIR}/hypothesis_17_serpine1_precision_target"
H18_DIR="${ITERATION_DIR}/hypothesis_18_multimodal_integration"
H19_DIR="${ITERATION_DIR}/hypothesis_19_metabolomics_phase1"
H20_DIR="${ITERATION_DIR}/hypothesis_20_cross_species_conservation"

# Agent output directories
H16_CLAUDE="${H16_DIR}/claude_code"
H16_CODEX="${H16_DIR}/codex"
H17_CLAUDE="${H17_DIR}/claude_code"
H17_CODEX="${H17_DIR}/codex"
H18_CLAUDE="${H18_DIR}/claude_code"
H18_CODEX="${H18_DIR}/codex"
H19_CLAUDE="${H19_DIR}/claude_code"
H19_CODEX="${H19_DIR}/codex"
H20_CLAUDE="${H20_DIR}/claude_code"
H20_CODEX="${H20_DIR}/codex"

# Task files
H16_TASK="${H16_DIR}/01_task.md"
H17_TASK="${H17_DIR}/01_task.md"
H18_TASK="${H18_DIR}/01_task.md"
H19_TASK="${H19_DIR}/01_task.md"
H20_TASK="${H20_DIR}/01_task.md"

# Verify task files exist
for TASK_FILE in "$H16_TASK" "$H17_TASK" "$H18_TASK" "$H19_TASK" "$H20_TASK"; do
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
# HYPOTHESIS 16: H13 EXTERNAL VALIDATION COMPLETION (HIGHEST PRIORITY!)
# ============================================================

H16_CLAUDE_PROMPT="🚨🔬 CRITICAL: Complete H13 External Validation! 🚨🔬

Read ${H16_TASK} - THIS IS THE MOST IMPORTANT HYPOTHESIS FOR ITERATION 05!

H13 INCOMPLETE (Iteration 04):
✅ Claude found 6 independent datasets BUT DID NOT ANALYZE
❌ Codex completely failed (exit 0, no results)
⚠️ RISK: ALL H01-H15 findings based on SAME dataset → OVERFITTING UNDETECTED!

YOUR MISSION: FINISH THE JOB! Download + analyze all 6 datasets Claude found!

🚨 MANDATORY TASKS (DO NOT SKIP ANY):
1. READ H13 CLAUDE RESULTS:
   - File: /Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_04/hypothesis_13_independent_dataset_validation/claude_code/90_results_claude_code.md
   - Extract: 6 dataset IDs/accessions (PRIDE: PXD..., GEO: GSE...)
   - Get download links from that file

2. DOWNLOAD ALL 6 DATASETS:
   - Use wget/curl if direct links
   - Use PRIDE API if PXD accessions
   - Use GEO API if GSE accessions
   - Save: external_datasets/dataset_{1-6}/raw_data.csv

3. PREPROCESS (MATCH OUR PIPELINE):
   - Use universal_zscore_function.py
   - Same normalization as merged_ecm_aging_zscore.csv
   - Gene symbol mapping via UniProt
   - Save: external_datasets/dataset_{1-6}/processed_zscore.csv

4. TRANSFER LEARNING (NO RETRAINING!):
   a) H08 S100 Models:
      - Load: /iterations/iteration_03/hypothesis_08_s100_calcium_signaling/claude_code/s100_stiffness_model_claude_code.pth
      - Load: /iterations/iteration_03/hypothesis_08_s100_calcium_signaling/codex/s100_stiffness_model_codex.pth
      - Predict on external data
      - Target: External R²≥0.60 (vs train 0.75-0.81)

   b) H06 Biomarker Panel:
      - Load: /iterations/iteration_02/hypothesis_06_biomarker_panel/codex/biomarker_classifier.pkl
      - Test 8-protein panel (F13B, SERPINF1, S100A9, FSTL1, GAS6, CTSA, COL1A1, BGN)
      - Target: External AUC≥0.80 (vs train 1.0)

   c) H03 Tissue Velocities:
      - Compute external velocities (same method)
      - Correlation with our velocities
      - Target: ρ>0.70

5. META-ANALYSIS:
   - Combine our data + 6 external datasets
   - Random-effects model (statsmodels)
   - I² heterogeneity for top 20 proteins
   - Target: I²<50% for ≥15/20 proteins

6. STABLE vs VARIABLE PROTEINS:
   - STABLE: I²<30%, same direction in all datasets
   - VARIABLE: I²>70%, inconsistent
   - Clinical implication: STABLE proteins = robust biomarkers

REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H16_CLAUDE}/
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

🎯 SUCCESS = ≥4/6 datasets downloaded, H08 R²≥0.60 external, H06 AUC≥0.80 external 🎯

⚠️ IF YOU DON'T COMPLETE THIS, ALL H01-H15 CONCLUSIONS ARE QUESTIONABLE! ⚠️"

H16_CODEX_PROMPT="🚨 H13 External Validation COMPLETION - CRITICAL! 🚨

Read ${H16_TASK} and complete what Claude started in H13!

H13 Claude found 6 datasets but didn't analyze. Your job: FINISH IT!

MANDATORY:
1. Read H13 Claude results file (path in task)
2. Download all 6 external datasets
3. Test H08 S100 models (transfer learning, no retraining)
4. Test H06 biomarker panel
5. Meta-analysis with I² heterogeneity

Requirements:
- Agent: 'codex'
- Workspace: ${H16_CODEX}/
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

Target: External R²≥0.60 (H08), AUC≥0.80 (H06), I²<50% (meta-analysis)"

# ============================================================
# HYPOTHESIS 17: SERPINE1 Precision Drug Target
# ============================================================

H17_CLAUDE_PROMPT="💊🎯 SERPINE1: From Discovery to Drug Development! 💊🎯

Read ${H17_TASK} - Validate SERPINE1 as a clinical drug target!

H14 DISCOVERY (Iteration 04):
✅ SERPINE1 highest eigenvector centrality (0.891)
✅ Validated by both agents (Claude degree, Codex eigenvector)
✅ Network hub → controls multiple ECM pathways

YOUR MISSION: Prove SERPINE1 is DRUGGABLE and SAFE!

🚨 MANDATORY TASKS:
1. IN-SILICO KNOCKOUT:
   - Load H05 GNN model
   - Perturb: Set SERPINE1 = 0
   - Measure: Cascade effects on LOX, TGM2, COL1A1
   - Target: ≥30% reduction in aging score

2. LITERATURE META-ANALYSIS:
   - PubMed: 'SERPINE1 knockout aging OR PAI-1 knockout lifespan'
   - Download: ALL knockout study abstracts (mouse, human cells)
   - Extract: Effect sizes (mean ± SD for KO vs WT)
   - Meta-analysis: Combined effect, I² heterogeneity
   - Target: Combined effect >0.5, I²<50%

3. DRUG-TARGET NETWORKS:
   - Inhibitors: TM5441, SK-216, TM5275, PAI-039
   - AlphaFold: SERPINE1 structure (Q05682)
   - Docking: Binding affinity prediction (DiffDock or AutoDock)
   - ADMET: pkCSM predictions (hERG, hepatotoxicity, bioavailability)
   - Target: hERG IC50>10µM, oral bioavail>30%

4. CLINICAL TRIALS:
   - Search ClinicalTrials.gov: 'SERPINE1 OR PAI-1'
   - Find: All trials, phases, status, results
   - Focus: PAI-039 (NCT00801112) Phase II completed
   - Extract: Safety data, adverse events

5. ECONOMIC ANALYSIS:
   - Market size: Age 65+ with fibrosis biomarkers
   - Pricing: Compare to pirfenidone (~$100k/year)
   - NPV calculation: Development cost vs revenue
   - Target: NPV positive, ROI>100%

REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H17_CLAUDE}/
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

🎯 GOAL: GO/NO-GO decision for SERPINE1 drug development! 🎯"

H17_CODEX_PROMPT="💊 SERPINE1 Drug Target Validation 💊

Read ${H17_TASK} - Is SERPINE1 druggable and safe?

MANDATORY:
1. In-silico knockout (GNN perturbation)
2. Literature meta-analysis (knockout studies)
3. Drug docking (TM5441, SK-216, AlphaFold)
4. ADMET predictions (toxicity, bioavailability)
5. Clinical trials search (ClinicalTrials.gov)

Requirements:
- Agent: 'codex'
- Workspace: ${H17_CODEX}/
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

Target: Knockout reduces aging ≥30%, drugs safe (hERG>10µM), NPV positive"

# ============================================================
# HYPOTHESIS 18: Multi-Modal Deep Learning Integration
# ============================================================

H18_CLAUDE_PROMPT="🤖🧠 Multi-Modal AI: Integrate ALL Models into ONE! 🤖🧠

Read ${H18_TASK} - Build the ULTIMATE aging predictor!

PREVIOUS MODELS (ALL STRONG INDIVIDUALLY):
✅ H11 LSTM: PCA pseudo-time, R²=0.29
✅ H05 GNN: 103,037 hidden edges
✅ H04 Autoencoder: 648→32D, 89% variance
✅ H08 S100 Pathway: S100→stiffness, R²=0.75-0.81

HYPOTHESIS: Integration beats individual models!

YOUR MISSION: Build unified architecture → R²>0.85!

🚨 MANDATORY ARCHITECTURE:
1. Autoencoder (H04): Compress 648 proteins → 32 latent dimensions
2. GNN (H05): Enrich latent features with graph structure
3. LSTM (H11): Model temporal trajectories on GNN-enriched features
4. S100 Fusion: Add pathway constraints (S100→stiffness mechanistic priors)
5. Multi-task loss: Age prediction + reconstruction + stiffness

TASKS:
1. LOAD PRE-TRAINED MODULES:
   - H04 autoencoder weights
   - H05 GNN edge_index (103,037 edges)
   - H08 S100 pathway genes (S100A9, S100A10, S100B, LOX, TGM2)

2. TRAINING:
   - Multi-task loss: 1.0*age_loss + 0.1*reconstruction_loss
   - Optimizer: Adam, lr=0.001, weight_decay=1e-5
   - Early stopping: validation R²
   - Target: Val R²>0.85, Test R²>0.80, MAE<3 years

3. ABLATION STUDIES:
   - Baseline: Ridge regression
   - Autoencoder only
   - AE + GNN
   - AE + GNN + LSTM
   - Full (AE + GNN + LSTM + S100) ← TARGET
   - Show: Each module contributes ≥10% gain

4. INTERPRETABILITY:
   - Attention weights visualization
   - SHAP values (top 50 proteins)
   - Pathway enrichment of SHAP features
   - Target: S100/LOX/TGM2 in top 20, overlap ≥60% with H06 panel

5. EXTERNAL VALIDATION:
   - Transfer to H16 external datasets
   - Target: Mean external R²≥0.75

REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H18_CLAUDE}/
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

🎯 GOAL: R²>0.85 → NEW STATE-OF-THE-ART AGING PREDICTOR! 🎯"

H18_CODEX_PROMPT="🤖 Multi-Modal AI Integration 🤖

Read ${H18_TASK} - Combine Autoencoder+GNN+LSTM+S100!

MANDATORY:
1. Load H04/H05/H08/H11 models
2. Build unified architecture
3. Multi-task training (age + reconstruction + stiffness)
4. Ablation studies (quantify each module)
5. SHAP interpretability
6. External validation (H16 datasets)

Requirements:
- Agent: 'codex'
- Workspace: ${H18_CODEX}/
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

Target: R²>0.85 (validation), R²>0.80 (test), MAE<3 years"

# ============================================================
# HYPOTHESIS 19: Metabolomics Integration for Phase I
# ============================================================

H19_CLAUDE_PROMPT="⚡🧪 Metabolomics: Find the MISSING Phase I Markers! ⚡🧪

Read ${H19_TASK} - Validate Phase I metabolic hypothesis!

H12 DISCOVERED (Iteration 04):
✅ Metabolic-Mechanical Transition: v=1.65-2.17
✅ Phase II (v>2.17): COL1A1, LOX, TGM2 (CONFIRMED by both agents)
❓ Phase I (v<1.65): Metabolic changes (HYPOTHETICAL, not proven!)

PROBLEM: Proteomics can't see metabolites!
- ATP, NAD+, lactate, pyruvate are NOT proteins
- We may be detecting aging TOO LATE (after metabolic damage)

YOUR MISSION: Prove Phase I exists via metabolomics!

🚨 MANDATORY TASKS:
1. DATABASE SEARCH:
   - Metabolomics Workbench: Search 'aging OR fibrosis' (tissue studies)
   - MetaboLights: Search 'age' (EBI repository)
   - Target: ≥2 datasets with ATP, NAD+, lactate, pyruvate
   - Download: Raw LC-MS/GC-MS data

2. PREPROCESSING:
   - Use universal_zscore_function (SAME as proteomics)
   - Normalize: Within-tissue z-scores
   - Gene symbol mapping: Metabolite names standardization
   - Save: metabolomics_data/ST{id}_zscore.csv

3. METABOLITE-VELOCITY CORRELATION:
   - Load H03 tissue velocities
   - Correlate: ATP vs velocity (expect ρ<-0.50, anticorrelation)
   - Correlate: NAD+ vs velocity (expect ρ<-0.50)
   - Correlate: Lactate/Pyruvate ratio vs velocity (expect ρ>0.50)

4. PHASE I vs PHASE II COMPARISON:
   - Phase I tissues: v<1.65 (Liver, Muscle)
   - Phase II tissues: v>2.17 (Lung, Tubulointerstitial)
   - T-test: ATP Phase I vs Phase II (expect ≥20% lower in Phase I)
   - T-test: NAD+ (expect ≥30% lower)
   - T-test: Lactate/Pyruvate (expect ≥1.5× higher)

5. MULTI-OMICS INTEGRATION:
   - Joint PCA: Proteomics (648) + Metabolomics (~50)
   - Compare variance: Multi-omics vs proteomics-only
   - Target: ≥95% variance (vs 89% proteomics-only)

6. TEMPORAL PREDICTION:
   - Longitudinal data: Does ATP drop BEFORE COL1A1 rises?
   - Lead-lag analysis
   - Target: Metabolites lead proteins by ≥1 timepoint (~3 months)

REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H19_CLAUDE}/
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

🎯 GOAL: Validate Phase I → Enable EARLY intervention (NAD+ boosters)! 🎯"

H19_CODEX_PROMPT="⚡ Metabolomics Phase I Validation ⚡

Read ${H19_TASK} - Prove Phase I is metabolic!

MANDATORY:
1. Search Metabolomics Workbench, MetaboLights
2. Download ≥2 datasets (ATP, NAD, lactate, pyruvate)
3. Preprocess (z-scores, match proteomics pipeline)
4. Correlate with H03 velocities
5. Phase I vs Phase II comparison
6. Multi-omics joint PCA

Requirements:
- Agent: 'codex'
- Workspace: ${H19_CODEX}/
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

Target: ATP ρ<-0.50, NAD ρ<-0.50, Phase I ATP↓≥20%, multi-omics variance≥95%"

# ============================================================
# HYPOTHESIS 20: Cross-Species Conservation
# ============================================================

H20_CLAUDE_PROMPT="🐭🪱 Cross-Species Conservation: Mouse vs Worm! 🐭🪱

Read ${H20_TASK} - Are our findings universal or human-specific?

CRITICAL VALIDATION:
⚠️ ALL H01-H15 based on HUMAN + MOUSE data
⚠️ Risk: Mechanisms may be species-specific
⚠️ FDA requires animal validation → need cross-species confirmation

YOUR MISSION: Test top discoveries in mouse, rat, C. elegans!

🚨 MANDATORY TASKS:
1. DATABASE SEARCH:
   a) Mouse Aging Proteomics:
      - PRIDE: 'mouse aging proteomics ECM'
      - Target: Mouse Aging Cell Atlas (Tabula Muris Senis)
      - Tissues: Heart, liver, muscle (match human)

   b) C. elegans Proteomics:
      - PRIDE, WormBase: 'C. elegans aging proteomics'
      - Whole organism data acceptable

   c) Optional: Rat (Rattus norvegicus)

2. ORTHOLOG MAPPING:
   - Human → Mouse: Ensembl Compara API
   - Expected: ≥85% mapping (648 ECM genes)
   - Human → C. elegans: WormBase
   - Expected: ≥40% mapping (ancient core genes only)
   - Save: orthologs_human_mouse.csv, orthologs_human_worm.csv

3. TEST H08 S100 PATHWAY:
   - Mouse: S100a9 → Tgm2 correlation (expect ρ>0.60 if conserved)
   - Mouse: S100b → Lox correlation
   - C. elegans: tgm-1 → col-19 (S100 absent in worms, test ancestral crosslinking)
   - Save: s100_pathway_conservation.csv

4. TEST H12 TRANSITION:
   - Mouse: Compute tissue velocities (same method as H03)
   - Changepoint detection: Find mouse transition zone
   - Expected: v_mouse ~ 1.4-2.0 (similar to human 1.65-2.17)
   - C. elegans: Expect NO transition (minimal ECM, monotonic aging)

5. TEST H14 SERPINE1 CENTRALITY:
   - Mouse: Build network, compute eigenvector centrality
   - Check Serpine1 centrality (expect >0.80 if conserved hub)
   - C. elegans: Check sri-40 (serpin ortholog)
   - Expected: Lower centrality in worms (different mechanism)

6. EVOLUTIONARY RATE ANALYSIS:
   - dN/dS (non-synonymous / synonymous ratio)
   - Conserved proteins: dN/dS <0.5 (purifying selection)
   - Species-specific: dN/dS >1.0 (rapid evolution)
   - Use Ensembl Compara API

REQUIREMENTS:
- Agent: 'claude_code'
- Workspace: ${H20_CLAUDE}/
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

🎯 GOAL: Identify UNIVERSAL (conserved) vs SPECIES-SPECIFIC mechanisms! 🎯"

H20_CODEX_PROMPT="🐭 Cross-Species Conservation Test 🐭

Read ${H20_TASK} - Mouse vs C. elegans validation!

MANDATORY:
1. Search mouse, C. elegans aging proteomics
2. Ortholog mapping (human → mouse → worm)
3. Test S100 pathway conservation
4. Test metabolic-mechanical transition
5. Test SERPINE1 centrality
6. dN/dS evolutionary rates

Requirements:
- Agent: 'codex'
- Workspace: ${H20_CODEX}/
- Reference: ${ML_REQ}

Dataset: /Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv

Target: Mouse conserves ≥3/4 mechanisms, C. elegans partial (ancient pathways only)"

# ============================================================
# LAUNCH ALL 10 AGENTS IN PARALLEL
# ============================================================

echo "🚀 Launching 10 agents in parallel..."
echo ""

# Track PIDs
declare -a PIDS
declare -a AGENT_NAMES

# H16 Claude
echo "▶️  H16 Claude (External Validation Completion)..."
(cd "$REPO_ROOT"
 echo "$H16_CLAUDE_PROMPT" | claude --print \
     --permission-mode bypassPermissions \
     --add-dir "${H16_CLAUDE}" \
     --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
     --add-dir "${REPO_ROOT}/11_subagent_for_LFQ_ingestion" \
     > "${H16_CLAUDE}/claude_output.log" 2>&1) &
PIDS+=($!)
AGENT_NAMES+=("H16_Claude")

# H16 Codex
echo "▶️  H16 Codex (External Validation Completion)..."
(cd "$REPO_ROOT"
 codex exec --sandbox danger-full-access -C "$H16_CODEX" "$H16_CODEX_PROMPT" \
     > "${H16_CODEX}/codex_output.log" 2>&1) &
PIDS+=($!)
AGENT_NAMES+=("H16_Codex")

# H17 Claude
echo "▶️  H17 Claude (SERPINE1 Drug Target)..."
(cd "$REPO_ROOT"
 echo "$H17_CLAUDE_PROMPT" | claude --print \
     --permission-mode bypassPermissions \
     --add-dir "${H17_CLAUDE}" \
     --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
     > "${H17_CLAUDE}/claude_output.log" 2>&1) &
PIDS+=($!)
AGENT_NAMES+=("H17_Claude")

# H17 Codex
echo "▶️  H17 Codex (SERPINE1 Drug Target)..."
(cd "$REPO_ROOT"
 codex exec --sandbox danger-full-access -C "$H17_CODEX" "$H17_CODEX_PROMPT" \
     > "${H17_CODEX}/codex_output.log" 2>&1) &
PIDS+=($!)
AGENT_NAMES+=("H17_Codex")

# H18 Claude
echo "▶️  H18 Claude (Multi-Modal Integration)..."
(cd "$REPO_ROOT"
 echo "$H18_CLAUDE_PROMPT" | claude --print \
     --permission-mode bypassPermissions \
     --add-dir "${H18_CLAUDE}" \
     --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
     > "${H18_CLAUDE}/claude_output.log" 2>&1) &
PIDS+=($!)
AGENT_NAMES+=("H18_Claude")

# H18 Codex
echo "▶️  H18 Codex (Multi-Modal Integration)..."
(cd "$REPO_ROOT"
 codex exec --sandbox danger-full-access -C "$H18_CODEX" "$H18_CODEX_PROMPT" \
     > "${H18_CODEX}/codex_output.log" 2>&1) &
PIDS+=($!)
AGENT_NAMES+=("H18_Codex")

# H19 Claude
echo "▶️  H19 Claude (Metabolomics Phase I)..."
(cd "$REPO_ROOT"
 echo "$H19_CLAUDE_PROMPT" | claude --print \
     --permission-mode bypassPermissions \
     --add-dir "${H19_CLAUDE}" \
     --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
     --add-dir "${REPO_ROOT}/11_subagent_for_LFQ_ingestion" \
     > "${H19_CLAUDE}/claude_output.log" 2>&1) &
PIDS+=($!)
AGENT_NAMES+=("H19_Claude")

# H19 Codex
echo "▶️  H19 Codex (Metabolomics Phase I)..."
(cd "$REPO_ROOT"
 codex exec --sandbox danger-full-access -C "$H19_CODEX" "$H19_CODEX_PROMPT" \
     > "${H19_CODEX}/codex_output.log" 2>&1) &
PIDS+=($!)
AGENT_NAMES+=("H19_Codex")

# H20 Claude
echo "▶️  H20 Claude (Cross-Species Conservation)..."
(cd "$REPO_ROOT"
 echo "$H20_CLAUDE_PROMPT" | claude --print \
     --permission-mode bypassPermissions \
     --add-dir "${H20_CLAUDE}" \
     --add-dir "${REPO_ROOT}/08_merged_ecm_dataset" \
     > "${H20_CLAUDE}/claude_output.log" 2>&1) &
PIDS+=($!)
AGENT_NAMES+=("H20_Claude")

# H20 Codex
echo "▶️  H20 Codex (Cross-Species Conservation)..."
(cd "$REPO_ROOT"
 codex exec --sandbox danger-full-access -C "$H20_CODEX" "$H20_CODEX_PROMPT" \
     > "${H20_CODEX}/codex_output.log" 2>&1) &
PIDS+=($!)
AGENT_NAMES+=("H20_Codex")

echo ""
echo "✓ All 10 agents launched!"
echo ""

# ============================================================
# MONITOR PROGRESS
# ============================================================

echo "⏳ Monitoring agent execution..."
echo "You can tail individual logs:"
for i in "${!PIDS[@]}"; do
    echo "  ${AGENT_NAMES[$i]}: tail -f ${ITERATION_DIR}/hypothesis_*/${AGENT_NAMES[$i]#*_}*/$(echo ${AGENT_NAMES[$i]} | tr '[:upper:]' '[:lower:]' | sed 's/_claude/claude/' | sed 's/_codex/codex/')_output.log"
done
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
echo "🏁 Iteration 05 COMPLETE!"
echo "===================================================================================="
echo "🕐 End time: $(date)"
echo ""
echo "📊 Summary:"
echo "  Total agents: 10"
echo "  Successful: $((10 - FAILED))"
echo "  Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ ALL AGENTS COMPLETED SUCCESSFULLY!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Review results in each hypothesis folder (90_results_*.md)"
    echo "2. Compare Claude vs Codex findings for each hypothesis"
    echo "3. Synthesize Iteration 05 discoveries"
    echo "4. Update master ranking table with H16-H20"
    echo "5. Prepare final publication!"
else
    echo "⚠️  Some agents failed. Review logs above."
    echo "   Check output logs for error details."
fi

echo ""
echo "🎯 PRIORITY: H16 External Validation results determine validity of ALL prior findings!"
echo ""

exit $FAILED
