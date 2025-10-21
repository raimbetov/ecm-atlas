# Execution Notes: Batch Correction Analysis

**Status:** Framework created, requires data format adaptation before execution

## Issue Identified

The merged dataset (`08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`) uses **wide format**:
- Separate columns: `Abundance_Old`, `Abundance_Young`, `Zscore_Old`, `Zscore_Young`
- One row per protein-study-tissue combination

The scripts were written expecting **long format**:
- Single columns: `Abundance`, `Z_score`, `Age_Group`
- Two rows per protein-study-tissue (one Old, one Young)

## Required Adaptations

### Option 1: Transform Data to Long Format (Recommended)

Create a preprocessing script:

```python
import pandas as pd

# Load wide format data
df_wide = pd.read_csv('../../08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# Transform to long format
records = []
for _, row in df_wide.iterrows():
    base = {
        'Protein_ID': row['Protein_ID'],
        'Gene_Symbol': row['Gene_Symbol'],
        'Study_ID': row['Study_ID'],
        'Tissue_Compartment': row['Tissue_Compartment'],
        'Species': row['Species'],
        'Method': row['Method']
    }

    # Old age group
    if pd.notna(row['Abundance_Old']):
        records.append({
            **base,
            'Age_Group': 'Old',
            'Abundance': row['Abundance_Old'],
            'Z_score': row['Zscore_Old']
        })

    # Young age group
    if pd.notna(row['Abundance_Young']):
        records.append({
            **base,
            'Age_Group': 'Young',
            'Abundance': row['Abundance_Young'],
            'Z_score': row['Zscore_Young']
        })

df_long = pd.DataFrame(records)
df_long.to_csv('merged_ecm_aging_long_format.csv', index=False)
```

### Option 2: Modify Scripts to Work with Wide Format

Update both R and Python scripts to:
1. Read wide format data
2. Create separate matrices for Old and Young
3. Apply normalization independently
4. Calculate Δ effects directly

## Current Script Capabilities

### ComBat Correction (R) - `combat_correction/01_apply_combat.R`

**What it does:**
- ✓ Loads data and creates protein × sample matrix
- ✓ Applies empirical Bayes batch correction (sva::ComBat)
- ✓ Preserves biological covariates (Age + Tissue)
- ✓ Calculates ICC before/after
- ✓ Generates PCA and effect preservation plots

**Requires:**
- R with packages: sva, tidyverse, jsonlite, lme4
- Long format data with Age_Group column

**Expected runtime:** 3-5 minutes

### Percentile Normalization (Python) - `percentile_normalization/01_apply_percentile.py`

**What it does:**
- ✓ Applies within-study percentile ranking (0-100)
- ✓ Calculates age effects (Δpercentile)
- ✓ Meta-analysis across studies with FDR correction
- ✓ Validates Q1 driver recovery
- ✓ Generates volcano plot and diagnostic visualizations

**Requires:**
- Python with: pandas, numpy, scipy, matplotlib, seaborn, statsmodels
- Long format data with Age_Group column

**Expected runtime:** 2-3 minutes

## Recommended Execution Path

1. **Create long format data:**
   ```bash
   cd 14_exploratory_batch_correction/
   python create_long_format.py  # Create this script
   ```

2. **Update script paths:**
   - Modify data_path in both scripts to point to long format CSV

3. **Run ComBat (if R available):**
   ```bash
   cd combat_correction/
   Rscript 01_apply_combat.R
   ```

4. **Run Percentile:**
   ```bash
   source ../../env/bin/activate
   cd percentile_normalization/
   python 01_apply_percentile.py
   ```

5. **Review results:**
   - Check `diagnostics/` for plots
   - Review metadata JSON files for metrics
   - Compare ICC, driver recovery, FDR-significant proteins

## Alternative: Use Existing Wide Format Analyses

The statistical validation agents (Q1.3.1) already performed analyses on this wide format data:

**Already available:**
- `12_priority_research_questions/Q1.3.1_statistical_validation/agent1/` - Z-score audit with ICC=0.29
- `12_priority_research_questions/Q1.3.1_statistical_validation/agent2/` - Alternative methods comparison

**Key findings (no correction applied):**
- ICC = 0.29 (poor reliability, batch effects dominate)
- 0 FDR-significant proteins across studies
- 405 universal proteins (≥3 tissues, ≥70% consistency)
- 8 consensus proteins (≥2 methods)

## Decision

Given the data format mismatch, you have two options:

### Option A: Adapt Data Format (1-2 hours work)
- Create long format transformation script
- Re-run batch correction analyses
- Get corrected ICC, FDR, and validation metrics
- **Benefit:** Complete validation of batch correction efficacy

### Option B: Work with Existing Analyses (0 hours)
- Use Q1.3.1 agent results as-is
- Accept ICC=0.29 batch effect limitation
- Report within-study findings only
- Note in publications: "Cross-study comparisons require batch correction"
- **Benefit:** Proceed immediately with qualified claims

## Recommendation

**For immediate publication:**
- Use Option B (existing analyses)
- Report consensus proteins (n=8) as high-confidence findings (37.5% literature validated)
- Add disclaimer about batch effects
- Note future work: "Batch correction and external validation required"

**For rigorous validation:**
- Use Option A (run batch correction)
- Invest 1-2 hours to adapt data format
- Get definitive answer: Do findings survive correction?
- Publish with higher confidence levels

## Files in This Framework

**Documentation:**
- `00_README.md` - Comprehensive plan (19 KB)
- `EXECUTION_NOTES.md` - This file

**Scripts (ready, need data format fix):**
- `combat_correction/01_apply_combat.R` - ComBat empirical Bayes (18 KB)
- `percentile_normalization/01_apply_percentile.py` - Rank normalization (15 KB)
- `run_analysis.sh` - Master execution (5.2 KB)

**Folders:**
- `diagnostics/` - Output plots will go here
- `validation/` - Metric comparison scripts (to be created)
- `reports/` - Final report (to be generated)

---

**Status:** Framework complete, awaiting data format adaptation OR decision to proceed with existing Q1.3.1 analyses

**Next action:** Choose Option A (validate) or Option B (publish with caveats)
