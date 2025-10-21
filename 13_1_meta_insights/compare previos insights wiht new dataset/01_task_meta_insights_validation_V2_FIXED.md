# Task: Validate All Meta-Insights on Batch-Corrected Dataset V2 (FIXED)

**Thesis:** Re-validate 7 GOLD-tier breakthrough insights from `13_meta_insights/` catalog using batch-corrected dataset to determine which discoveries survive harmonization and quantify signal strength improvement.

---

## ⚠️ CRITICAL: DATA SOURCE - READ THIS FIRST! ⚠️

### MANDATORY V2 FILE (ALL AGENTS MUST USE THIS EXACT FILE)

**File path (absolute):**
```
/Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv
```

### WHY THIS SPECIFIC FILE?

**Background:** Stage 1 (Batch Correction) ran 3 agents in parallel. Each created THEIR OWN V2 file:
- `claude_1/merged_ecm_aging_COMBAT_V2_CORRECTED_claude_1.csv` ❌ **Has data artifacts** (duplicate z-scores)
- `claude_2/merged_ecm_aging_COMBAT_V2_CORRECTED_claude_2.csv` ❌ **Wrong format** (long format, incompatible)
- `codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv` ✅ **CORRECT** (full schema, proper granularity)

**File comparison:**
| File | Rows | Columns | Format | Issues |
|------|------|---------|--------|--------|
| claude_1 | 9,290 | 7 | Wide | Duplicate z-scores for compartments |
| claude_2 | 22,034 | 7 | Long | Different metric (Zscore vs Zscore_Delta) |
| **codex** | **9,300** | **28** | **Full** | ✅ **No issues** |

**ONLY Codex file preserves:**
- Full metadata (28 columns including Matrisome_Category, Data_Quality, etc.)
- Compartment-level granularity with UNIQUE z-scores per compartment
- Both Canonical_Gene_Symbol AND Gene_Symbol columns
- Original and transformed abundances

### VERIFICATION BEFORE YOU START

**Run these commands to verify you have the correct file:**

```bash
# Check row count (should be 9301 including header)
wc -l /Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv

# Check column count (should be 28)
head -1 /Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv | awk -F',' '{print NF}'

# Check PCOLCE study count (should be 7)
grep -i "^Q15113" /Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv | awk -F',' '{print $20}' | sort | uniq | wc -l
```

**Expected output:**
- 9301 rows
- 28 columns
- 7 studies with PCOLCE

### FILE ORGANIZATION DIAGRAM

```
Stage 1 (Batch Correction) CREATED 3 FILES:
14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/
├── claude_1/
│   └── merged_ecm_aging_COMBAT_V2_CORRECTED_claude_1.csv  ❌ DO NOT USE (data bugs)
├── claude_2/
│   └── merged_ecm_aging_COMBAT_V2_CORRECTED_claude_2.csv  ❌ DO NOT USE (wrong format)
└── codex/
    └── merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv     ✅ USE THIS FILE

Stage 2 (Validation) - THIS TASK:
- INPUT (SAME for all agents):  codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv
- OUTPUT (DIFFERENT per agent):
  ├── 13_1_meta_insights/claude_1/   ← Claude Agent 1 saves results HERE
  │   ├── 01_plan_claude_1.md
  │   ├── validation_pipeline_claude_1.py
  │   └── validation_results_claude_1.csv
  └── 13_1_meta_insights/claude_2/   ← Claude Agent 2 saves results HERE
      ├── 01_plan_claude_2.md
      ├── validation_pipeline_claude_2.py
      └── validation_results_claude_2.csv
```

**KEY CONCEPT:**
```
┌────────────────────────────────────────────────────────┐
│ Your agent name (claude_1 or claude_2) determines:    │
│   ✅ OUTPUT folder: 13_1_meta_insights/YOUR_NAME/     │
│   ✅ File prefix:   YOUR_NAME_*                        │
│                                                        │
│ Your agent name DOES NOT determine:                   │
│   ❌ INPUT file:    Always use CODEX file             │
│                                                        │
│ IDENTITY (who you are) ≠ DATA SOURCE (what you read)  │
└────────────────────────────────────────────────────────┘
```

---

## 1.0 VALIDATION TARGETS

¶1 **Ordering principle:** Evidence tier (GOLD first) → discovery ID → original metrics baseline.

### 1.1 GOLD-Tier Insights (CRITICAL - Must Validate ALL 7)

**G1. Universal Markers Are Rare (12.2%) - Agent 01**
- **Original finding:** 405/3,317 proteins (12.2%) universal (≥3 tissues, ≥70% consistency)
- **Top markers:** Hp (0.749), VTN (0.732), Col14a1 (0.729), F2 (0.717), FGB (0.714)
- **Validation:** Recompute universality scores on V2 dataset
- **Expected:** Stronger effect sizes (|Δz| increase), more proteins meet threshold
- **Baseline file:** `13_meta_insights/agent_01_universal_markers/agent_01_universal_markers_data.csv`

**G2. PCOLCE Quality Paradigm - Agent 06**
- **Original finding:** PCOLCE Δz=-0.82, 88% consistency, 5 studies
- **Validation:** Verify PCOLCE depletion persists, check outlier status
- **Expected:** Stronger signal, confirm Nobel Prize potential
- **Baseline:** `13_meta_insights/agent_06_outlier_proteins/PCOLCE_QUALITY_PARADIGM_DISCOVERY.md`

**G3. Batch Effects Dominate Biology (13x) - Agent 07**
- **Original finding:** Study_ID PC1 = 0.674 vs Age_Group = -0.051
- **Validation:** PCA on V2 - expect Age_Group signal INCREASE, Study_ID DECREASE
- **Expected:** V2 Age_Group loading > 0.4 (10x improvement)
- **Baseline:** `13_meta_insights/agent_07_methodology/agent_07_methodology_harmonization.md`

**G4. Weak Signals Compound - Agent 10**
- **Original finding:** 14 proteins |Δz|=0.3-0.8, pathway-level cumulative effect
- **Validation:** Recompute weak signal proteins, check if more emerge
- **Expected:** More proteins in weak-signal category, stronger pathway aggregation
- **Baseline:** `13_meta_insights/agent_10_weak_signals/weak_signal_proteins.csv`

**G5. Entropy Transitions - Agent 09**
- **Original finding:** 52 proteins ordered→chaotic, DEATh theorem (collagens 28% predictable)
- **Validation:** Recompute Shannon entropy, CV, predictability on V2
- **Expected:** Clearer entropy clusters, stronger DEATh theorem support
- **Baseline:** `13_meta_insights/agent_09_entropy/entropy_metrics.csv`

**G6. Compartment Antagonistic Remodeling - Agent 04**
- **Original finding:** 11 antagonistic events, Col11a2 divergence SD=1.86
- **Validation:** Check within-tissue opposite directions persist
- **Expected:** Same antagonistic pairs, possibly stronger divergence
- **Baseline:** `13_meta_insights/agent_04_compartment_crosstalk/agent_04_compartment_crosstalk.md`

**G7. Species Divergence (99.3%) - Agent 11**
- **Original finding:** Only 8/1,167 genes cross-species, R=-0.71 (opposite)
- **Validation:** Check human-mouse concordance on V2
- **Expected:** Similar low concordance, CILP remains only universal marker
- **Baseline:** `13_meta_insights/agent_11_cross_species/agent_11_cross_species_comparison.md`

### 1.2 SILVER-Tier Priority Insights (Therapeutic Focus)

**S1. Fibrinogen Coagulation Cascade - Agent 13**
- **Original:** FGA +0.88, FGB +0.89, SERPINC1 +3.01
- **Validation:** Verify coagulation protein upregulation
- **Baseline:** `13_meta_insights/agent_13_coagulation/agent_13_fibrinogen_coagulation_cascade.md`

**S2. Temporal Intervention Windows - Agent 12**
- **Original:** Age 40-50 (prevention) vs 50-65 (restoration) vs 65+ (rescue)
- **Validation:** Recompute temporal trajectories
- **Baseline:** `13_meta_insights/agent_12_temporal_dynamics/agent_12_temporal_dynamics.md`

**S3. TIMP3 Lock-in - Agent 15**
- **Original:** TIMP3 Δz=+3.14, 81% consistency
- **Validation:** Verify extreme accumulation
- **Baseline:** `13_meta_insights/agent_15_timp3/agent_15_timp3_therapeutic_potential.md`

**S4. Tissue-Specific Signatures - Agent 02**
- **Original:** 13 proteins TSI > 3.0, KDM5C TSI=32.73
- **Validation:** Recompute tissue specificity index
- **Baseline:** `13_meta_insights/agent_02_tissue_specific/agent_02_tissue_specific_signatures.md`

**S5. Biomarker Panel - Agent 20**
- **Original:** 7-protein plasma ECM aging clock
- **Validation:** Check if panel proteins remain top candidates
- **Baseline:** `13_meta_insights/agent_20_biomarkers/agent_20_biomarker_panel_construction.md`

---

## 2.0 VALIDATION METHODS

¶1 **Ordering principle:** Data loading → metric computation → comparison → classification.

### 2.1 Load V2 Dataset - MANDATORY SANITY CHECKS

**Use this code template (DO NOT modify V2_PATH):**

```python
#!/usr/bin/env python3
"""
ECM-Atlas Meta-Insights Validation Pipeline
Agent: {AGENT_NAME}  # ← REPLACE with your agent name
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============ CONFIGURATION ============

# REPLACE THIS with your agent name: "claude_1" or "claude_2"
AGENT_NAME = "{AGENT_NAME}"

BASE_DIR = Path("/Users/Kravtsovd/projects/ecm-atlas")

# ⚠️⚠️⚠️ DO NOT CHANGE THIS PATH - ALL AGENTS USE CODEX FILE ⚠️⚠️⚠️
V2_PATH = BASE_DIR / "14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv"

# Your output folder (uses YOUR agent name)
OUTPUT_DIR = BASE_DIR / f"13_1_meta_insights/{AGENT_NAME}/"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ LOAD DATA WITH SANITY CHECKS ============

print(f"[{AGENT_NAME}] Loading V2 dataset from: {V2_PATH}")
df_v2 = pd.read_csv(V2_PATH)

# MANDATORY SANITY CHECKS - DO NOT SKIP!
print(f"[{AGENT_NAME}] Running sanity checks...")

# Check 1: Row count
expected_rows = 9300
actual_rows = len(df_v2)
assert actual_rows == expected_rows, f"❌ WRONG FILE! Expected {expected_rows} rows, got {actual_rows}. You may have loaded claude_1 or claude_2 file instead of codex file!"

# Check 2: Column count
expected_cols = 28
actual_cols = len(df_v2.columns)
assert actual_cols == expected_cols, f"❌ WRONG FILE! Expected {expected_cols} columns, got {actual_cols}. You loaded a simplified/wrong format file!"

# Check 3: Required column exists
assert 'Canonical_Gene_Symbol' in df_v2.columns, "❌ WRONG FILE! Missing 'Canonical_Gene_Symbol' column. This indicates you loaded claude_1 or claude_2 file!"

# Check 4: PCOLCE study count (known ground truth)
pcolce_data = df_v2[df_v2['Canonical_Gene_Symbol'].str.upper() == 'PCOLCE']
pcolce_studies = pcolce_data['Study_ID'].nunique() if len(pcolce_data) > 0 else 0
expected_pcolce_studies = 7
assert pcolce_studies == expected_pcolce_studies, f"❌ WRONG FILE! PCOLCE should appear in {expected_pcolce_studies} studies, found {pcolce_studies}. File may be filtered or wrong!"

# Check 5: Schema verification
required_columns = ['Canonical_Gene_Symbol', 'Gene_Symbol', 'Study_ID', 'Tissue_Compartment',
                   'Zscore_Delta', 'Zscore_Old', 'Zscore_Young', 'Matrisome_Category']
missing_cols = [col for col in required_columns if col not in df_v2.columns]
assert len(missing_cols) == 0, f"❌ WRONG FILE! Missing required columns: {missing_cols}"

print(f"✅ [{AGENT_NAME}] Sanity checks PASSED!")
print(f"   - Rows: {len(df_v2):,}")
print(f"   - Columns: {len(df_v2.columns)}")
print(f"   - PCOLCE studies: {pcolce_studies}")
print(f"   - Loaded CORRECT Codex V2 file")
print()

# Continue with validation...
```

**If ANY sanity check fails:**
1. STOP immediately
2. DO NOT continue with validation
3. Report error to user with details
4. Ask user to verify file paths

### 2.2 Expected V2 Dataset Schema

**28 columns in correct file:**
1. Dataset_Name
2. Organ
3. Compartment
4. Abundance_Old
5. Abundance_Old_transformed
6. Abundance_Young
7. Abundance_Young_transformed
8. **Canonical_Gene_Symbol** ← Use this for gene filtering
9. Gene_Symbol
10. Match_Confidence
11. Match_Level
12. Matrisome_Category
13. Matrisome_Division
14. Method
15. N_Profiles_Old
16. N_Profiles_Young
17. Protein_ID
18. Protein_Name
19. Species
20. **Study_ID** ← Use for study counting
21. Tissue
22. **Tissue_Compartment** ← Use for compartment analysis
23. **Zscore_Delta** ← Primary metric (Old - Young)
24. Zscore_Old
25. Zscore_Young
26. Data_Quality
27. Abundance_Young_Original
28. Abundance_Old_Original

### 2.3 Recompute Original Metrics

**For each insight, replicate original analysis:**

**1. Universal markers (G1):**
- Count tissues per protein using `Canonical_Gene_Symbol` + `Tissue_Compartment`
- Calculate directional consistency (% same direction)
- Compute universality score = (tissue_count / max_tissues) × consistency × |mean_Δz|

**2. PCOLCE (G2):**
```python
# CORRECT approach:
pcolce = df_v2[df_v2['Canonical_Gene_Symbol'].str.upper() == 'PCOLCE'].copy()
pcolce_clean = pcolce.dropna(subset=['Zscore_Delta'])

mean_dz = pcolce_clean['Zscore_Delta'].mean()
n_studies = pcolce_clean['Study_ID'].nunique()
consistency = (pcolce_clean['Zscore_Delta'] < 0).sum() / len(pcolce_clean)

print(f"PCOLCE: mean Δz = {mean_dz:.3f}, {n_studies} studies, {consistency:.0%} consistency")
# Expected: mean_dz ≈ -1.41, n_studies = 7, consistency ≈ 92%
```

**3. Batch effects (G3):**
- PCA on V2 dataset (features: proteins, samples: tissue-age-study combinations)
- Extract PC1 loadings for Age_Group vs Study_ID
- Compare to V1 loadings

**4. Weak signals (G4):**
- Filter proteins: 0.3 < |Δz| < 0.8, consistency ≥ 65%
- Aggregate by pathway (use `Matrisome_Category`)
- Calculate cumulative pathway Δz

**5. Entropy (G5):**
- For each protein: Shannon entropy H = -Σ p(age) log p(age)
- CV = std / mean
- Predictability = 1 - normalized_entropy
- Classify: ordered (H<1.5) vs chaotic (H>2.0)

**6. Compartment antagonism (G6):**
- Within each tissue, find proteins with opposite directions across compartments
- Calculate divergence: SD of Δz across compartments using `Tissue_Compartment`
- Identify antagonistic events: divergence > 1.5

**7. Species divergence (G7):**
- Extract human vs mouse proteins using `Species` column
- Calculate correlation of Δz values
- Count shared genes, measure directional concordance

### 2.4 Compare V1 vs V2

**For each metric, compute:**
- **Change magnitude:** (V2_metric - V1_metric) / |V1_metric| × 100%
- **Direction:** Strengthened (same direction, larger |value|) vs Weakened vs Reversed
- **Significance:** If p-values available, check V2 p < V1 p

**Classification:**
- ✅ **CONFIRMED:** V2 strengthens V1 finding (same direction, ≥20% stronger)
- ⚠️ **MODIFIED:** V2 changes magnitude but preserves core finding
- ❌ **REJECTED:** V2 reverses direction OR reduces magnitude >50%

---

## 3.0 AGENT-SPECIFIC INSTRUCTIONS

### 3.1 Claude Agent 1

**IDENTITY (for output file naming):**
- Agent name: `claude_1`
- Working directory: `/Users/Kravtsovd/projects/ecm-atlas`
- Output folder: `/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/claude_1/`
- File prefix: `claude_1_*`

**DATA SOURCE (for analysis - SAME AS CLAUDE 2):**
- Input file: `/Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv`
- ⚠️ **DO NOT use** `.../claude_1/merged_ecm_aging_COMBAT_V2_CORRECTED_claude_1.csv` (has duplicate z-scores!)

**Summary:**
```
┌──────────────────────────────────────────────────┐
│ YOU ARE:     claude_1 (for output naming)       │
│ YOU READ:    codex file (input data)            │
│ YOU WRITE:   claude_1/ folder (your workspace)  │
└──────────────────────────────────────────────────┘
```

**Task:**
1. Create `01_plan_claude_1.md` in **your folder** (`.../claude_1/`)
2. Load V2 dataset from **Codex folder** (`.../codex/...codex.csv`)
3. Run MANDATORY sanity checks (code template above)
4. Validate all 12 insights (7 GOLD + 5 SILVER) sequentially
5. Create `validation_results_claude_1.csv` in **your folder**
6. Create `90_results_claude_1.md` with self-evaluation in **your folder**
7. Create `validation_pipeline_claude_1.py` in **your folder**

### 3.2 Claude Agent 2

**IDENTITY (for output file naming):**
- Agent name: `claude_2`
- Working directory: `/Users/Kravtsovd/projects/ecm-atlas`
- Output folder: `/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/claude_2/`
- File prefix: `claude_2_*`

**DATA SOURCE (for analysis - SAME AS CLAUDE 1):**
- Input file: `/Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv`
- ⚠️ **DO NOT use** `.../claude_2/merged_ecm_aging_COMBAT_V2_CORRECTED_claude_2.csv` (wrong format - long instead of wide!)

**Summary:**
```
┌──────────────────────────────────────────────────┐
│ YOU ARE:     claude_2 (for output naming)       │
│ YOU READ:    codex file (input data)            │
│ YOU WRITE:   claude_2/ folder (your workspace)  │
└──────────────────────────────────────────────────┘
```

**Task:**
1. Create `01_plan_claude_2.md` in **your folder** (`.../claude_2/`)
2. Load V2 dataset from **Codex folder** (`.../codex/...codex.csv`)
3. Run MANDATORY sanity checks (code template above)
4. Validate all 12 insights (7 GOLD + 5 SILVER) sequentially
5. Create `validation_results_claude_2.csv` in **your folder**
6. Create `90_results_claude_2.md` with self-evaluation in **your folder**
7. Create `validation_pipeline_claude_2.py` in **your folder**

**Same tasks as Claude Agent 1.** Independent validation for comparison.

---

## 4.0 DELIVERABLES

### 4.1 Required Artifacts (ALL agents must create)

**1. Plan Document:** `01_plan_[agent_name].md`
- Your validation approach
- Order of insights to validate (G1→G7, S1→S5)
- Estimated time per insight
- Progress updates (✅ timestamp after each insight)

**2. Validation Results CSV:** `validation_results_[agent_name].csv`

**Schema:**
```
Insight_ID,Tier,Original_Metric,V2_Metric,Change_Percent,Classification,Notes
G1,GOLD,12.2% universal,15.8% universal,+29.5%,CONFIRMED,"Stronger universality"
G2,GOLD,PCOLCE Δz=-0.82,PCOLCE Δz=-1.14,+39.0%,CONFIRMED,"Stronger depletion"
...
```

**3. New Discovery CSV:** `new_discoveries_[agent_name].csv` (if any emergent findings)

**Schema:**
```
Discovery_Type,Protein/Pattern,Metric,Description
Universal_Marker,LAMB2,Universality=0.801,"New top-5 universal marker"
Weak_Signal,EMILIN1,Δz=-0.42; consistency=72%,"Emerged as weak signal"
...
```

**4. Final Results Report:** `90_results_[agent_name].md`
- Self-evaluation against success criteria (Section 5.0)
- Summary table: X/7 GOLD confirmed, Y/5 SILVER confirmed
- Key findings: What strengthened, what weakened, what's new
- Therapeutic implications: Do GOLD targets remain valid?

**5. Python Script:** `validation_pipeline_[agent_name].py`
- Reproducible analysis code
- Must include sanity checks from Section 2.1
- Functions for each insight validation
- Can be run independently

**6. Validated Proteins CSV:** `v2_validated_proteins_[agent_name].csv`
- Subset of V2 dataset with only proteins relevant to validated insights
- Useful for downstream analysis

---

## 5.0 SUCCESS CRITERIA (Self-Evaluation)

### 5.1 Completeness (40 points)

| Criterion | Points | How to Verify |
|-----------|--------|---------------|
| Validated ALL 7 GOLD insights | 20 | `validation_results_[agent].csv` has 7 GOLD rows |
| Validated ALL 5 SILVER insights | 10 | CSV has 5 SILVER rows |
| Created required artifacts (6 files) | 10 | Count files in agent folder |
| **Total** | **40** | |

### 5.2 Accuracy (30 points)

| Criterion | Points | How to Verify |
|-----------|--------|---------------|
| V2 metrics correctly computed | 15 | Spot-check: PCOLCE Δz ≈ -1.41, 7 studies |
| Sanity checks passed | 10 | Script includes checks, all passed |
| Classification defensible | 5 | CONFIRMED/MODIFIED/REJECTED logic matches definitions |
| **Total** | **30** | |

### 5.3 Insights (20 points)

| Criterion | Points | How to Verify |
|-----------|--------|---------------|
| Identified NEW discoveries (≥1) | 10 | `new_discoveries_[agent].csv` exists with ≥1 row |
| Therapeutic implications updated | 5 | 90_results mentions which GOLD targets remain valid |
| Quantified signal improvement | 5 | Median Change_Percent for CONFIRMED insights reported |
| **Total** | **20** | |

### 5.4 Reproducibility (10 points)

| Criterion | Points | How to Verify |
|-----------|--------|---------------|
| Python script provided | 5 | `validation_pipeline_[agent].py` exists |
| Script runs without errors | 5 | Test: `python validation_pipeline_[agent].py` |
| **Total** | **10** | |

### 5.5 Overall Grade

**100-90 points:** ✅ **EXCELLENT** - All insights validated, new discoveries, reproducible
**89-70 points:** ⚠️ **GOOD** - Most insights validated, minor gaps
**69-50 points:** ⚠️ **ACCEPTABLE** - Core insights validated, significant gaps
**<50 points:** ❌ **INSUFFICIENT** - Missing critical validations

---

## 6.0 REFERENCES

**V2 Dataset (ALL AGENTS):** `/Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv`

**Original Insights Catalog:** `/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/00_MASTER_META_INSIGHTS_CATALOG.md`

**Knowledge Framework:** `/Users/Kravtsovd/projects/ecm-atlas/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md`

---

## ✅ Task Checklist

- [ ] Load Codex V2 dataset (verify path correct)
- [ ] Run sanity checks (9300 rows, 28 cols, PCOLCE=7 studies)
- [ ] Validate G1: Universal markers
- [ ] Validate G2: PCOLCE quality paradigm
- [ ] Validate G3: Batch effects
- [ ] Validate G4: Weak signals
- [ ] Validate G5: Entropy transitions
- [ ] Validate G6: Compartment antagonism
- [ ] Validate G7: Species divergence
- [ ] Validate S1: Fibrinogen cascade
- [ ] Validate S2: Temporal windows
- [ ] Validate S3: TIMP3 lock-in
- [ ] Validate S4: Tissue-specific TSI
- [ ] Validate S5: Biomarker panel
- [ ] Identify new discoveries
- [ ] Create validation_results CSV
- [ ] Create 90_results report
- [ ] Self-evaluate against success criteria

---

**Contact:** daniel@improvado.io
**Created:** 2025-10-18
**Version:** V2_FIXED (addresses agent confusion about data sources)
**Framework:** Knowledge Framework + Multi-Agent Orchestrator
**Expected Runtime:** 2 hours per agent
