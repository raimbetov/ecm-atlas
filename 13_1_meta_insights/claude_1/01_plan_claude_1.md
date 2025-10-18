# Meta-Insights Validation Plan - Agent Claude_1

**Thesis:** Validate 12 meta-insights (7 GOLD + 5 SILVER) from original ECM-Atlas analysis using Codex V2 batch-corrected dataset to determine which discoveries survive harmonization and quantify signal strength improvements.

---

## Agent Identity

**Agent name:** claude_1
**Working directory:** `/Users/Kravtsovd/projects/ecm-atlas`
**Output folder:** `/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/claude_1/`
**File prefix:** `claude_1_*`

**Data source (INPUT):**
`/Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv`

⚠️ **Critical:** Using Codex V2 file (NOT claude_1 file which has duplicate z-scores)

---

## Validation Approach

### Phase 1: Data Loading & Validation (15 min)
1. Load Codex V2 dataset from specified path
2. Run mandatory sanity checks:
   - Row count = 9300
   - Column count = 28
   - PCOLCE studies = 7
   - Schema verification (Canonical_Gene_Symbol, Study_ID, Tissue_Compartment, Zscore_Delta)
3. Generate data quality report

### Phase 2: GOLD-Tier Validation (90 min)
Sequential validation of 7 critical insights:

**G1. Universal Markers (15 min)**
- Compute universality scores for all proteins
- Compare to original 12.2% threshold
- Expected: Stronger effect sizes, more universal proteins

**G2. PCOLCE Quality Paradigm (10 min)**
- Extract PCOLCE data, compute mean Δz
- Count studies, calculate consistency
- Expected: Δz ≈ -1.41, 7 studies, ~92% consistency

**G3. Batch Effects Dominate Biology (15 min)**
- Run PCA on V2 dataset
- Extract PC1 loadings for Age_Group vs Study_ID
- Expected: Age_Group loading > 0.4 (vs original -0.051)

**G4. Weak Signals Compound (15 min)**
- Filter proteins: 0.3 < |Δz| < 0.8, consistency ≥ 65%
- Aggregate by Matrisome_Category
- Expected: More weak-signal proteins emerge

**G5. Entropy Transitions (15 min)**
- Calculate Shannon entropy, CV, predictability per protein
- Classify ordered vs chaotic transitions
- Expected: Clearer entropy clusters

**G6. Compartment Antagonistic Remodeling (10 min)**
- Identify within-tissue opposite directions across compartments
- Calculate divergence SD
- Expected: Same antagonistic pairs, stronger divergence

**G7. Species Divergence (10 min)**
- Extract human vs mouse proteins
- Calculate Δz correlation
- Expected: R ≈ -0.71, low concordance persists

### Phase 3: SILVER-Tier Validation (45 min)
Validate 5 therapeutic-focused insights:

**S1. Fibrinogen Cascade (10 min)**
- Verify FGA, FGB, SERPINC1 upregulation

**S2. Temporal Windows (10 min)**
- Recompute age-stratified trajectories

**S3. TIMP3 Lock-in (5 min)**
- Verify extreme accumulation (Δz > 3.0)

**S4. Tissue-Specific Signatures (10 min)**
- Recompute TSI scores

**S5. Biomarker Panel (10 min)**
- Check 7-protein panel validity

### Phase 4: Analysis & Reporting (30 min)
1. Create validation_results_claude_1.csv
2. Identify new discoveries (new_discoveries_claude_1.csv)
3. Generate validated proteins subset (v2_validated_proteins_claude_1.csv)
4. Write final report (90_results_claude_1.md)
5. Self-evaluate against success criteria

---

## Expected Deliverables

1. ✅ `01_plan_claude_1.md` (this document)
2. ⏳ `validation_pipeline_claude_1.py` - Reproducible analysis script
3. ⏳ `validation_results_claude_1.csv` - 12 rows (7 GOLD + 5 SILVER)
4. ⏳ `new_discoveries_claude_1.csv` - Emergent findings
5. ⏳ `v2_validated_proteins_claude_1.csv` - Subset of validated proteins
6. ⏳ `90_results_claude_1.md` - Final report with self-evaluation

---

## Success Criteria Target

**Target score:** ≥90/100 (EXCELLENT)

- Completeness (40): All 12 insights + 6 artifacts
- Accuracy (30): Correct metrics, passed sanity checks
- Insights (20): New discoveries identified
- Reproducibility (10): Working Python script

---

## Progress Tracker

- [x] 2025-10-18 14:00 - Plan created
- [ ] Data loading & sanity checks
- [ ] G1: Universal markers
- [ ] G2: PCOLCE paradigm
- [ ] G3: Batch effects
- [ ] G4: Weak signals
- [ ] G5: Entropy
- [ ] G6: Compartment antagonism
- [ ] G7: Species divergence
- [ ] S1: Fibrinogen
- [ ] S2: Temporal windows
- [ ] S3: TIMP3
- [ ] S4: Tissue-specific
- [ ] S5: Biomarker panel
- [ ] Final deliverables

---

**Estimated total time:** 3 hours
**Start time:** 2025-10-18 14:00
**Expected completion:** 2025-10-18 17:00

**Framework:** ECM-Atlas Knowledge Framework
**Contact:** daniel@improvado.io
