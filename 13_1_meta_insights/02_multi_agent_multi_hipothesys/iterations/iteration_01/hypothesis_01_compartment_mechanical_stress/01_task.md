# Hypothesis 01: Compartment Antagonistic Mechanical Stress Adaptation

## Scientific Question

Does compartment-specific ECM remodeling during aging reflect mechanical stress adaptation, where high-load compartments (e.g., Soleus muscle, nucleus pulposus disc) upregulate structural proteins while low-load compartments (e.g., TA muscle, annulus fibrosus) downregulate the same proteins?

## Background Context

**Source Insight:** G6 Compartment Antagonistic Remodeling
- Codex identified 264 antagonistic events (+450% increase post-batch correction)
- Skeletal muscle divergence: Soleus vs TA compartments
- Top proteins showing antagonism: Col11a2 (4.48 SD), Col2a1, Fbn2, Cilp2, Postn
- Intervertebral disc compartments (NP, IAF, OAF) also exhibit LMAN1, CPN2, F9, MFAP4 antagonism

**Hypothesis:** Mechanical stress drives opposite aging trajectories. High-load tissues experience ECM reinforcement (↑ collagens, fibrillins) while low-load tissues experience ECM degradation (↓ same proteins), creating compartment antagonism.

## Data Source

**Primary Dataset:**
```
/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv
```

**Critical Columns:**
- `Tissue_Compartment`: Contains granular compartment data (Soleus, TA, NP, IAF, OAF, etc.)
- `Gene_Symbol` / `Canonical_Gene_Symbol`: Protein identifiers
- `Zscore_Delta`: Aging trajectory (positive = upregulation, negative = downregulation)
- `Tissue`: Tissue category
- `Study_ID`: Study identifier
- `Matrisome_Category`: Core vs Associated ECM proteins

**Validation Data:**
- Prior G6 findings: `/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/compare previos insights wiht new dataset/codex/90_results_codex.md`

## Success Criteria

### Criterion 1: Compartment Antagonism Quantification (40 pts)

**Required:**
1. Identify all protein-compartment pairs showing antagonistic aging (opposite signs of Δz between compartments within same tissue)
2. Calculate antagonism magnitude: `|Δz_compartment_A - Δz_compartment_B|`
3. Classify compartments by mechanical stress profile (high-load vs low-load) based on biomechanics literature or anatomical function
4. Statistical test: Do high-load compartments show HIGHER Δz than low-load for structural proteins? (Mann-Whitney U, p<0.05)

**Deliverables:**
- CSV: `antagonistic_pairs_[agent].csv` with columns: Gene_Symbol, Compartment_A, Compartment_B, Δz_A, Δz_B, Antagonism_Magnitude, Tissue
- Top 20 antagonistic proteins ranked by magnitude

### Criterion 2: Mechanical Stress Correlation (30 pts)

**Required:**
1. Define mechanical stress proxy variable for each compartment (e.g., 1=high load, 0=low load)
2. Test correlation: Does Δz correlate with mechanical stress for:
   - Structural proteins (collagens, fibrillins): Expect positive correlation
   - Regulatory proteins (serpins, proteases): Expect negative or no correlation
3. Statistical validation: Spearman ρ and p-value

**Deliverables:**
- Scatter plots: Δz vs mechanical stress, separated by protein type
- Statistics table: ρ, p-value, n for structural vs regulatory

### Criterion 3: Mechanistic Interpretation (20 pts)

**Required:**
1. Explain WHY mechanical stress would drive opposite aging patterns
2. Identify protein families showing strongest load-dependence
3. Propose mechanism: ECM mechanotransduction → differential aging?
4. Compare with prior G6 findings: Are new insights consistent?

**Deliverables:**
- Mechanistic hypothesis diagram (Mermaid)
- Evidence document linking biomechanics to protein changes

### Criterion 4: Therapeutic Implications (10 pts)

**Required:**
1. If hypothesis confirmed: Which compartments should be targeted for intervention?
2. Load modulation strategies: Would increasing/decreasing mechanical stress reverse aging?
3. Identify druggable proteins showing load-dependent aging

**Deliverables:**
- Therapeutic targets table: Protein, Compartment, Load-dependence, Druggability

## Required Artifacts

All files MUST be created in your agent-specific subfolder (`claude_code/` or `codex/`).

### Mandatory Files:

1. **01_plan_[agent].md** - Your analysis strategy and approach
2. **analysis_[agent].py** - Executable Python script with comments
3. **antagonistic_pairs_[agent].csv** - All antagonistic protein-compartment pairs
4. **mechanical_stress_correlation_[agent].csv** - Correlation statistics
5. **visualizations_[agent]/** - Folder with figures:
   - Antagonism heatmap (compartments × proteins)
   - Scatter plots (Δz vs load for protein types)
   - Top 20 antagonistic proteins bar chart
6. **90_results_[agent].md** - Final report in Knowledge Framework format (see standards below)

### Optional Supporting Files:

- Pathway enrichment results
- Network analysis graphs
- Supplementary statistics

## Self-Evaluation Rubric

Score yourself out of 100 points total:

### Completeness (40 pts)
- [ ] All 4 success criteria addressed with quantitative results
- [ ] Antagonistic pairs identified and quantified
- [ ] Mechanical stress correlation tested statistically
- [ ] Mechanistic interpretation provided
- [ ] Therapeutic implications documented

### Accuracy (30 pts)
- [ ] Statistical tests applied correctly (Mann-Whitney, Spearman)
- [ ] P-values and effect sizes reported
- [ ] Compartment classifications justified (high-load vs low-load)
- [ ] Protein categorization correct (structural vs regulatory)
- [ ] Results reproducible from provided code

### Novelty (20 pts)
- [ ] Goes beyond G6 original finding (adds mechanical stress dimension)
- [ ] Identifies new antagonistic proteins not in prior analysis
- [ ] Proposes testable mechanism (not just pattern description)
- [ ] Suggests novel therapeutic approach (load modulation)

### Reproducibility (10 pts)
- [ ] Python code runs without errors
- [ ] All file paths correct
- [ ] Results CSV generated successfully
- [ ] Visualizations saved as PNG/PDF
- [ ] README or comments explain how to reproduce

**Grading:**
- 90-100: Excellent - hypothesis strongly supported, novel insights, publication-ready
- 70-89: Good - hypothesis supported, some novel findings, needs refinement
- 50-69: Adequate - partial support, incremental findings
- <50: Insufficient - hypothesis not supported or methods flawed

## Documentation Standards

All `.md` files MUST follow Knowledge Framework:

**Required Structure:**
1. **Thesis:** 1 sentence summarizing main finding
2. **Overview:** 1 paragraph expanding thesis and previewing sections
3. **Mermaid Diagrams:**
   - TD (top-down) for structure (compartment hierarchy, protein categories)
   - LR (left-right) for process (mechanical stress → ECM remodeling)
4. **MECE Sections:** 1.0, 2.0, 3.0 (numbered, mutually exclusive, collectively exhaustive)
5. **Numbered Paragraphs:** ¶1, ¶2, ¶3 within each section

**Reference:**
```
/Users/Kravtsovd/projects/ecm-atlas/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md
```

## Agent Identity vs Data Source

**CRITICAL SEPARATION:**

**YOU ARE:** `[agent name]` (e.g., `claude_code` or `codex`)
- Create files with YOUR agent prefix: `01_plan_claude_code.md` or `01_plan_codex.md`
- Work in YOUR folder: `claude_code/` or `codex/`

**YOU READ:** The SAME dataset regardless of agent identity
- Dataset: `/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`
- DO NOT create or load agent-specific datasets
- Both agents analyze IDENTICAL data

## Sanity Checks

Before proceeding, verify:

```python
import pandas as pd

# Load dataset
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# Sanity checks
assert df.shape[0] >= 9000, f"Expected ≥9000 rows, got {df.shape[0]}"
assert 'Tissue_Compartment' in df.columns, "Missing Tissue_Compartment column!"
assert 'Zscore_Delta' in df.columns, "Missing Zscore_Delta column!"

# Check compartment granularity
compartments = df['Tissue_Compartment'].unique()
assert any('Soleus' in str(c) or 'TA' in str(c) for c in compartments), "Missing muscle compartments!"
assert any('NP' in str(c) or 'IAF' in str(c) or 'OAF' in str(c) for c in compartments), "Missing disc compartments!"

print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"✓ Compartments found: {len(compartments)}")
```

## Expected Results (for comparison)

**From prior G6 analysis (codex):**
- 264 antagonistic events total
- Top antagonistic magnitude: ≥4.48 SD (Col11a2)
- Skeletal muscle: Soleus vs TA show strongest divergence
- Intervertebral disc: NP vs IAF/OAF antagonism

**Your analysis should:**
- Replicate these findings AND
- Add mechanical stress classification AND
- Quantify load-dependence correlation

## Timeline

**Recommended workflow:**
1. Load and validate dataset (30 min)
2. Identify antagonistic pairs (1-2 hours)
3. Classify compartments by mechanical load (1 hour)
4. Statistical testing (1-2 hours)
5. Visualization (1 hour)
6. Interpretation and write-up (2-3 hours)

**Total:** 6-10 hours depending on agent speed

## Contact

Questions or clarifications: daniel@improvado.io

---

**Task Created:** 2025-10-20
**Hypothesis ID:** H01
**Iteration:** 01
**Predicted Scores:** Novelty 8/10, Impact 7/10
