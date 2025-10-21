# Hypothesis 03: Tissue-Specific Aging Velocity Clocks

## Scientific Question

Do different tissues age at different RATES (velocities) rather than merely different patterns, and can tissue-specific ECM markers define organ-level aging clocks that reveal which organs age fastest and share common acceleration mechanisms?

## Background Context

**Source Insight:** S4 Tissue-Specific Markers (50 proteins, TSI > threshold)

**Key Evidence:**
- S100a5: TSI=33.33 (hippocampus ultra-specific)
- Col6a4: TSI=27.46 (lung-specific)
- PLOD1: TSI=24.49 (dermis-specific)
- 50 total proteins with high tissue-exclusivity

**Current Understanding:**
- Prior work identifies WHICH tissues have unique markers
- Unexplored: HOW FAST do different tissues age?

**Hypothesis:** Tissue-specific markers define aging "clocks" ticking at different speeds. Hypothesis: Vascular tissue ages FASTEST (high metabolic stress), bone ages SLOWEST (low metabolic rate), and fast-aging tissues share inflammation/oxidative stress signatures.

**Clinical Relevance:** If true, prioritize therapeutic interventions for fast-aging tissues (e.g., target vascular aging before bone aging).

## Data Source

**Primary Dataset:**
```
/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv
```

**Critical Columns:**
- `Tissue` or `Tissue_Compartment`: Tissue identity
- `Gene_Symbol`: Protein identifier
- `Zscore_Delta`: Aging magnitude and direction
- `Age` (if available): Age metadata for velocity calculation
- `Study_ID`: Multi-study validation

**Validation Data:**
- S4 findings: `/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/compare previos insights wiht new dataset/codex/90_results_codex.md`

## Success Criteria

### Criterion 1: Tissue Aging Velocity Calculation (40 pts)

**Required:**
1. For each tissue, calculate **aging velocity:**
   - If age data available: `Velocity = Δz / Δyears` (z-score change per year)
   - If no age data: Use **tissue-specific mean |Δz|** as aging magnitude proxy
2. Tissue-specific metric calculation:
   - Use ONLY tissue-specific markers (high TSI proteins, TSI > 3.0)
   - Calculate mean Δz across tissue-specific markers
   - Calculate % proteins upregulated vs downregulated (directional bias)
3. Rank tissues by aging velocity:
   - Fastest → Slowest aging
   - Statistical confidence: Bootstrap CI for each tissue velocity
4. Validate with literature: Do high metabolic tissues age faster?

**Deliverables:**
- CSV: `tissue_aging_velocity_[agent].csv` with columns: Tissue, Mean_Δz, Velocity, N_Markers, Upregulated_%, Downregulated_%, Bootstrap_CI
- Ranking table: Tissues ordered by velocity
- Expected ranking hypothesis: Vascular > Skin > Lung > Muscle > Bone

### Criterion 2: Tissue-Specific Marker Identification (30 pts)

**Required:**
1. Calculate **Tissue Specificity Index (TSI)** for ALL proteins:
   - TSI = (Max_tissue_expression - Mean_other_tissues) / SD_other_tissues
   - Higher TSI = more tissue-specific
2. For each tissue, identify top 10 tissue-specific markers (highest TSI)
3. Classify markers by function:
   - Structural (collagens, laminins, elastins)
   - Regulatory (proteases, inhibitors, growth factors)
   - Signaling (integrins, matricellular proteins)
4. Compare with prior S4 findings:
   - Validate S100a5 (hippocampus), Col6a4 (lung), PLOD1 (dermis)
   - Identify NEW tissue-specific markers not in S4

**Deliverables:**
- CSV: `tissue_specific_markers_[agent].csv` with columns: Gene_Symbol, Tissue, TSI, Δz, Function_Category
- Top 10 markers per tissue table
- Comparison with S4: Agreement % on top markers

### Criterion 3: Fast-Aging Tissue Mechanisms (20 pts)

**Required:**
1. Define "fast-aging" tissues: Top 33% by velocity
2. Identify common mechanisms across fast-aging tissues:
   - Shared proteins (appear in ≥2 fast tissues)
   - Pathway enrichment (coagulation, inflammation, oxidative stress?)
   - Directional patterns (more upregulation vs downregulation?)
3. Test hypothesis: **Do fast-aging tissues share inflammatory signatures?**
   - Extract inflammatory proteins (interleukins, TNF, NF-κB targets)
   - Compare mean Δz_inflammation: Fast tissues vs slow tissues
   - Statistical test: Mann-Whitney p<0.05
4. Mechanistic interpretation:
   - Why do some tissues age faster?
   - Link to metabolic rate, oxidative stress, mechanical stress?

**Deliverables:**
- Fast-aging shared proteins list
- Inflammation signature comparison (box plot)
- Mechanistic hypothesis diagram (Mermaid)
- Literature support for metabolic rate hypothesis

### Criterion 4: Therapeutic Prioritization (10 pts)

**Required:**
1. **Clinical urgency:** Which tissues age fastest and cause most morbidity?
   - Vascular aging → cardiovascular disease
   - Skin aging → wound healing impairment
   - Lung aging → respiratory decline
2. **Intervention strategies:** For fast-aging tissues:
   - Target shared inflammatory pathways?
   - Tissue-specific approaches (e.g., collagen boosting in skin)?
3. **Biomarker potential:** Can tissue-specific markers serve as aging clocks?
   - Measure S100a5 → hippocampal aging status
   - Measure Col6a4 → lung aging status
4. Prioritize top 3 tissues for therapeutic intervention:
   - Rank by: Aging velocity × clinical impact × druggability

**Deliverables:**
- Therapeutic priority ranking table
- Intervention strategy for each priority tissue
- Biomarker candidates table

## Required Artifacts

All files MUST be created in your agent-specific subfolder (`claude_code/` or `codex/`).

### Mandatory Files:

1. **01_plan_[agent].md** - Analysis strategy
2. **analysis_[agent].py** - Executable Python script
3. **tissue_aging_velocity_[agent].csv** - Velocity for each tissue
4. **tissue_specific_markers_[agent].csv** - TSI and markers
5. **fast_aging_mechanisms_[agent].csv** - Shared proteins and pathways
6. **visualizations_[agent]/** - Folder with:
   - Tissue aging velocity bar chart (ranked fastest → slowest)
   - Heatmap: Tissues × top specific markers
   - Inflammation signature comparison (fast vs slow tissues)
   - Mechanistic diagram (Mermaid)
7. **90_results_[agent].md** - Final report in Knowledge Framework format

### Optional Supporting Files:

- Pathway enrichment statistics
- Literature evidence tables
- Multi-tissue correlation network

## Self-Evaluation Rubric

Score yourself out of 100 points total:

### Completeness (40 pts)
- [ ] Aging velocity calculated for all tissues
- [ ] Tissue-specific markers identified (TSI method)
- [ ] Fast-aging tissues defined and mechanisms explored
- [ ] Therapeutic priorities ranked
- [ ] Biomarker candidates proposed

### Accuracy (30 pts)
- [ ] TSI formula applied correctly
- [ ] Bootstrap confidence intervals for velocities
- [ ] Statistical tests valid (Mann-Whitney for inflammation)
- [ ] Tissue rankings biologically plausible
- [ ] Comparison with S4 findings accurate

### Novelty (20 pts)
- [ ] Moves from "tissue patterns" to "tissue velocity" (NEW)
- [ ] Identifies aging speed differences (not just protein differences)
- [ ] Proposes tissue-specific aging clocks (biomarker innovation)
- [ ] Links velocity to mechanisms (metabolic rate, inflammation)
- [ ] Prioritizes interventions by urgency (clinical translation)

### Reproducibility (10 pts)
- [ ] Python code runs without errors
- [ ] Velocity calculations documented
- [ ] Visualizations clear and publication-ready
- [ ] Results reproducible from code

**Grading:**
- 90-100: Hypothesis confirmed, tissue velocities differ significantly, mechanisms identified
- 70-89: Hypothesis supported, some velocity differences, partial mechanism clarity
- 50-69: Mixed evidence, velocity differences modest
- <50: Hypothesis not supported, tissues age uniformly

## Documentation Standards

All `.md` files MUST follow Knowledge Framework:

**Required Structure:**
1. **Thesis:** 1 sentence on tissue velocity differences
2. **Overview:** 1 paragraph expanding thesis and previewing sections
3. **Mermaid Diagrams:**
   - TD: Tissue hierarchy by velocity (fastest → slowest)
   - LR: Metabolic stress → inflammation → accelerated aging
4. **MECE Sections:** 1.0 Velocity Calculation, 2.0 Tissue Markers, 3.0 Fast-Aging Mechanisms, 4.0 Therapeutics
5. **Numbered Paragraphs:** ¶1, ¶2, ¶3

**Reference:**
```
/Users/Kravtsovd/projects/ecm-atlas/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md
```

## Agent Identity vs Data Source

**YOU ARE:** `[agent name]` (e.g., `claude_code` or `codex`)

**YOU READ:** Same dataset:
```
/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv
```

## Sanity Checks

Before proceeding:

```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# Identify tissues
tissues = df['Tissue'].unique() if 'Tissue' in df.columns else df['Tissue_Compartment'].unique()

# Sanity checks
assert len(tissues) >= 5, f"Expected ≥5 tissues, found {len(tissues)}"
assert 'Zscore_Delta' in df.columns, "Missing Zscore_Delta column!"

# Check for tissue-specific markers (example: S100a5)
s100a5_data = df[df['Gene_Symbol'].str.contains('S100a5|S100A5', na=False, case=False)]
if not s100a5_data.empty:
    print(f"✓ S100a5 found in {s100a5_data['Tissue'].nunique()} tissues")

print(f"✓ Dataset loaded: {df.shape[0]} rows")
print(f"✓ Tissues identified: {len(tissues)}")
print(f"✓ Example tissues: {list(tissues)[:5]}")
```

## Expected Results (for validation)

**From prior S4 analysis:**
- 50 tissue-specific markers (TSI > threshold)
- S100a5 = hippocampus-specific (TSI=33.33)
- Col6a4 = lung-specific (TSI=27.46)
- PLOD1 = dermis-specific (TSI=24.49)

**Your analysis should:**
- Validate these markers AND
- Calculate aging VELOCITY for each tissue AND
- Rank tissues by speed AND
- Identify fast-aging mechanisms

**Predicted velocity ranking (hypothesis):**
1. Vascular (high metabolic stress)
2. Skin (environmental exposure)
3. Lung (oxidative stress)
4. Muscle (moderate metabolic rate)
5. Bone (low metabolic rate)

## Timeline

**Recommended workflow:**
1. Load data and identify tissues (1 hour)
2. Calculate TSI for all proteins (2 hours)
3. Compute tissue aging velocities (2 hours)
4. Identify fast-aging mechanisms (2 hours)
5. Statistical testing and validation (1-2 hours)
6. Visualization and write-up (2-3 hours)

**Total:** 10-12 hours

## Contact

Questions: daniel@improvado.io

---

**Task Created:** 2025-10-20
**Hypothesis ID:** H03
**Iteration:** 01
**Predicted Scores:** Novelty 8/10, Impact 8/10
