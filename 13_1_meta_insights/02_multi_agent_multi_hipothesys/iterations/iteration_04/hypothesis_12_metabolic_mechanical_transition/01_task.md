# H12 – Metabolic-Mechanical Transition Point (v=1.65): Two-Phase Aging Model

## Scientific Question
Does ECM aging follow a two-phase model with a discrete transition from reversible metabolic dysregulation (Phase I, v<1.65) to irreversible mechanical remodeling (Phase II, v>1.65), and can we identify molecular markers of this transition?

## Background & Rationale

**Discovery from Iteration 03 (H09):**
- **Critical transition identified:** Velocity = 1.65 separates slow-aging from fast-aging tissues
- **Transition tissues:** Ovary Cortex (v=1.53→1.77) and Heart (v=1.58→1.82) span threshold
- **Gradient shift:** Paired t-test around transitions: t=-11.49, p=1.3×10⁻²⁸ (acceleration confirmed)

**Two-Phase Hypothesis:**
| Phase | Velocity Range | Dominant Process | Reversibility | Biomarkers |
|-------|----------------|------------------|---------------|------------|
| **Phase I** | v < 1.65 | Metabolic dysregulation, oxidative stress | REVERSIBLE (nutrition, exercise) | Mitochondrial, glycolysis |
| **Phase II** | v > 1.65 | Mechanical ECM remodeling, fibrosis | IRREVERSIBLE (crosslinking fixed) | LOX, TGM, collagens |

**Clinical Implication:**
If Phase I is reversible but Phase II is not → interventions MUST occur before v=1.65 threshold!

**Critical Gap:**
- H09 identified the transition but did NOT characterize molecular changes
- Need metabolic markers (mitochondria, ATP, glucose) vs mechanical markers (LOX, TGM, YAP/TAZ)
- Need to TEST if Phase I interventions work, Phase II interventions fail

## Objectives

### Primary Objective
Validate the two-phase aging model by identifying distinct molecular signatures for Phase I (metabolic) vs Phase II (mechanical) and testing if metabolic intervention markers predict transition timing.

### Secondary Objectives
1. **Literature validation:** Find evidence for metabolic→mechanical shift in aging/fibrosis
2. **New datasets:** Identify metabolomics + proteomics paired datasets or mechanotransduction markers (YAP/TAZ)
3. **Changepoint detection:** Statistically confirm v=1.65 as optimal breakpoint
4. **Intervention simulation:** Model effect of metabolic rescue (mitochondrial enhancement) on Phase I vs Phase II tissues

## Hypotheses to Test

### H12.1: Metabolic Phase I Signature
Tissues with v<1.65 are enriched for mitochondrial/glycolysis proteins (OR>2.0, p<0.05).

### H12.2: Mechanical Phase II Signature
Tissues with v>1.65 are enriched for crosslinking/collagen proteins (OR>2.0, p<0.05).

### H12.3: Transition Predictability
Baseline metabolomics (Phase I markers) predict time to Phase II transition with R²>0.70.

### H12.4: Reversibility Asymmetry
Simulated metabolic intervention (↑ mitochondrial proteins) shifts Phase I tissues but NOT Phase II.

## Required Analyses

### 1. LITERATURE SEARCH (MANDATORY)

**Search queries:**
```
1. "metabolic aging extracellular matrix"
2. "mechanotransduction aging transition"
3. "mitochondrial dysfunction fibrosis"
4. "YAP TAZ mechanical aging"
5. "reversible aging interventions metabolic"
6. "fibrosis point of no return crosslinking"
7. "Warburg effect aging collagen"
```

**Tasks:**
- Search PubMed, Cell Metabolism, Aging Cell, Nature Aging
- Download papers on metabolic→mechanical transitions (aging, fibrosis, wound healing)
- Extract: Known markers, transition mechanisms, intervention windows
- Save: `literature_metabolic_mechanical.md`

### 2. NEW DATASET SEARCH (MANDATORY)

**Highest priority: Metabolomics + Proteomics paired data**

**Search targets:**
```
- Metabolomics Workbench: "aging" + "tissue"
- MetaboLights: Homo sapiens aging studies
- GEO: "metabolomics proteomics" + "aging"
- PRIDE: Datasets with mitochondrial markers (ATP5A1, COX4I1, NDUFA, etc.)
```

**Criteria:**
- Paired metabolomics + proteomics (same samples)
- OR: Proteomics with mitochondrial/glycolysis proteins
- OR: Mechanotransduction markers: YAP1, TAZ (WWTR1), ROCK1/2, TEAD1-4

**Alternative if metabolomics not found:**
- Proxy metabolic markers from proteomics:
  - **Mitochondria:** ATP5A1, COX4I1, NDUFA9, UQCRC1, SDHB
  - **Glycolysis:** HK1, PFKM, ALDOA, GAPDH, PKM
  - **Mechanotrans:** YAP1, WWTR1, ROCK1, LATS1, TEAD1

**Download:**
- At least 1 metabolomics dataset (if available)
- Integrate with merged_ecm_aging_zscore.csv

### 3. TISSUE CLASSIFICATION (PHASE I vs PHASE II)

**Velocity threshold:** v = 1.65 (from H09)

**Phase I tissues (v<1.65):**
```
Skeletal_muscle_Gastrocnemius: 1.02
Brain_Hippocampus: 1.18
Liver: 1.34
Heart_Native: 1.58
...
```

**Phase II tissues (v>1.65):**
```
Tubulointerstitial: 3.45
Lung: 4.29
Intervertebral_disc_IAF: 3.12
...
```

**Binary labels:**
```python
labels = [1 if velocity < 1.65 else 2 for velocity in tissue_velocities]
```

### 4. CHANGEPOINT DETECTION (STATISTICAL VALIDATION)

**Methods:**
1. **Bayesian changepoint** (R package: `bcp`)
   ```R
   library(bcp)
   bcp_result <- bcp(sorted_velocities)
   optimal_changepoint <- which.max(bcp_result$posterior.prob)
   ```

2. **PELT algorithm** (R package: `changepoint`)
   ```R
   library(changepoint)
   cpt <- cpt.mean(sorted_velocities, method="PELT")
   breakpoint <- cpts(cpt)
   ```

3. **Binary segmentation**
   ```python
   from ruptures import Binseg
   model = Binseg(model="l2").fit(velocities)
   breakpoints = model.predict(n_bkps=1)
   ```

**Compare:**
- Does statistical method confirm v=1.65?
- Or suggest alternative threshold (e.g., v=2.0)?

**Sensitivity:**
- Test changepoint on protein gradients (not just tissue velocity)
- Most proteins should have changepoint ~1.65 if real

### 5. PHASE-SPECIFIC PROTEIN ENRICHMENT

#### Phase I Markers (Metabolic)

**Candidate proteins:**
- **Mitochondrial Complex I:** NDUFA1, NDUFA9, NDUFB8, NDUFS1
- **Complex II:** SDHB, SDHA
- **Complex III:** UQCRC1, UQCRC2
- **Complex IV:** COX4I1, COX5A
- **Complex V (ATP synthase):** ATP5A1, ATP5B
- **Glycolysis:** HK1, HK2, PFKM, ALDOA, GAPDH, PKM
- **TCA cycle:** IDH2, MDH2, ACO2

**Enrichment test:**
```python
from scipy.stats import fisher_exact

# Contingency table
#                 Phase I | Phase II
# Metabolic upregulated    a    |    b
# Other proteins           c    |    d

OR, p = fisher_exact([[a, b], [c, d]])
```

**Success:** OR>2.0, p<0.05

#### Phase II Markers (Mechanical)

**Candidate proteins:**
- **Crosslinking:** LOX, LOXL1-4, TGM1-3 (from H08)
- **Collagens:** COL1A1, COL1A2, COL3A1, COL4A1, COL6A1
- **Mechanotrans:** YAP1, WWTR1 (TAZ), ROCK1, ROCK2, TEAD1-4
- **ECM glycoproteins:** FN1, TNC, THBS1, POSTN, SPARC

**Enrichment test:** Same Fisher exact, expect OR>2.0 for Phase II

### 6. CLASSIFICATION MODEL (PHASE I vs II)

**Features:**
- Model A: All proteins (baseline)
- Model B: Metabolic markers only (Phase I signature)
- Model C: Mechanical markers only (Phase II signature)
- Model D: Combined metabolic + mechanical

**Architecture:**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced'
)
```

**Metrics:**
- AUC, Accuracy, Precision, Recall
- Feature importance (SHAP)

**Success criteria:**
- Model D AUC > 0.90 (excellent separation)
- Model B identifies Phase I, Model C identifies Phase II independently

### 7. TRANSITION TIMING PREDICTION

**Goal:** Predict WHEN a tissue will transition from Phase I → Phase II based on baseline metabolic state.

**Approach:**
```python
# For Phase I tissues only
X = metabolic_markers_baseline
y = distance_to_threshold  # y = 1.65 - current_velocity

model = Ridge(alpha=1.0)
model.fit(X, y)

# Predict: How far from transition?
predicted_distance = model.predict(X_test)
```

**Metrics:**
- R², MAE for distance prediction
- Classify "near transition" (v=1.4-1.65) vs "far" (v<1.4)

**Success:** R²>0.70, AUC>0.85 for near/far classification

### 8. INTERVENTION SIMULATION (IN SILICO)

**Scenario:** Enhance mitochondrial function (↑ ATP5A1, COX4I1 by +1 SD)

**Test:**
```python
# Baseline
phase1_velocity_baseline = mean(velocities[v < 1.65])
phase2_velocity_baseline = mean(velocities[v > 1.65])

# Intervention (increase mitochondrial markers)
mitochondrial_proteins = ['ATP5A1', 'COX4I1', 'NDUFA9', ...]
data_augmented = data.copy()
data_augmented[mitochondrial_proteins] += 1.0  # +1 SD

# Recompute velocity proxy
phase1_velocity_intervention = ...
phase2_velocity_intervention = ...

# Effect size
delta_phase1 = phase1_velocity_intervention - phase1_velocity_baseline
delta_phase2 = phase2_velocity_intervention - phase2_velocity_baseline
```

**Hypothesis:**
- Phase I tissues show velocity DECREASE (slower aging) → reversible
- Phase II tissues show NO change or INCREASE → irreversible (crosslinking dominates)

**Statistical test:**
```python
from scipy.stats import ttest_rel
t_stat, p_val = ttest_rel(velocities_baseline, velocities_intervention)
```

**Success:**
- Phase I: p<0.05, effect size >0.3 SD
- Phase II: p>0.10 (no effect) OR paradoxical increase

### 9. EXTERNAL VALIDATION (IF METABOLOMICS FOUND)

**Paired metabolomics + proteomics:**
- Metabolites: Glucose, ATP, NAD+, lactate, pyruvate
- Proteins: ECM remodeling markers

**Test:**
- Do metabolites correlate with Phase I proteins?
- Do Phase I metabolites predict Phase II protein levels?

**Causal mediation:**
```
Metabolite (ATP) → Mitochondrial protein (ATP5A1) → Velocity
```

**Success:** Indirect effect >30%

## Deliverables

### Code & Models
- `analysis_metabolic_mechanical_{agent}.py` — main script
- `changepoint_detection_{agent}.py` — statistical breakpoint analysis
- `phase_classifier_{agent}.pkl` — trained RF model
- `intervention_simulation_{agent}.py` — in silico mitochondrial rescue
- `literature_search_{agent}.py` — automated queries

### Data Tables
- `phase_assignments_{agent}.csv` — Tissue, velocity, phase (I or II)
- `changepoint_results_{agent}.csv` — Optimal breakpoint by method
- `enrichment_analysis_{agent}.csv` — Fisher OR for metabolic/mechanical markers
- `classification_performance_{agent}.csv` — AUC, accuracy for models A/B/C/D
- `transition_prediction_{agent}.csv` — Predicted distance to threshold
- `intervention_effects_{agent}.csv` — Velocity change Phase I vs II
- `metabolomics_validation_{agent}.csv` — Metabolite-protein correlations (if data)
- `literature_findings_{agent}.csv` — Evidence for two-phase model

### Visualizations
- `visualizations_{agent}/velocity_distribution_{agent}.png` — Histogram with v=1.65 line
- `visualizations_{agent}/changepoint_plot_{agent}.png` — Bayesian posterior probability
- `visualizations_{agent}/enrichment_heatmap_{agent}.png` — Metabolic vs mechanical markers
- `visualizations_{agent}/phase_classifier_roc_{agent}.png` — AUC curves for models
- `visualizations_{agent}/intervention_effects_{agent}.png` — Δ velocity Phase I vs II (violin plot)
- `visualizations_{agent}/protein_trajectories_{agent}.png` — Metabolic (blue) vs mechanical (red) across velocity axis

### Report
- `90_results_{agent}.md` — comprehensive findings with:
  - Literature synthesis (metabolic→mechanical transitions in aging)
  - Changepoint validation (is v=1.65 statistically optimal?)
  - Phase-specific signatures (which proteins define each phase?)
  - Transition predictability (can we forecast Phase II onset?)
  - Intervention simulation (is Phase I reversible?)
  - External validation (metabolomics evidence if found)
  - **CLINICAL RECOMMENDATION:** Intervention window (before v=1.65)

## Success Criteria

| Metric | Target | Source |
|--------|--------|--------|
| Changepoint confirmation | v=1.6-1.7 | Bayesian/PELT analysis |
| Phase I enrichment OR | >2.0 | Metabolic markers Fisher test |
| Phase II enrichment OR | >2.0 | Mechanical markers Fisher test |
| Phase classifier AUC | >0.90 | RF model |
| Transition prediction R² | >0.70 | Baseline → distance model |
| Phase I intervention effect | p<0.05 | Mitochondrial rescue simulation |
| Phase II intervention effect | p>0.10 | No effect (irreversible) |
| Literature papers | ≥5 relevant | Metabolic-mechanical reviews |
| Metabolomics datasets | ≥1 found | Workbench/MetaboLights |

## Expected Outcomes

### If Two-Phase Model Confirmed:
- **Phase I (v<1.65):** Metabolic dysregulation dominates → reversible with nutrition, exercise, mitochondrial drugs (NAD+ boosters)
- **Phase II (v>1.65):** Mechanical remodeling dominates → requires LOX inhibitors, crosslinking reversal (not just metabolic fix)
- **Critical window:** Interventions MUST start before v=1.65 for maximal benefit

### If Transition is Gradual (No Changepoint):
- Aging is continuous, no discrete threshold → interventions effective at all stages but diminishing returns

### If Multiple Breakpoints Found:
- Three-phase model (e.g., v=1.2 metabolic onset, v=1.65 mechanical onset, v=3.0 fibrotic endpoint)

## Clinical Translation

- **Biomarker panel:** ATP5A1, COX4I1 (Phase I), LOX, TGM2 (Phase II) → stage patients
- **Intervention stratification:**
  - **Phase I patients:** NAD+ precursors (NMN, NR), CoQ10, metformin, exercise
  - **Phase II patients:** LOX inhibitors (BAPN), senolytic drugs, anti-fibrotics
- **Timing:** Screen at age 40 (likely Phase I) → intervene before transition
- **Monitoring:** Track velocity proxy (tissue stiffness, ECM markers) → detect Phase I→II shift early

## Dataset

**Primary:**
`/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`

**Tissue velocities (from H03):**
```
Skeletal_muscle: 1.02
Liver: 1.34
Heart: 1.58
Ovary_Cortex: 1.65  ← TRANSITION
Lung: 4.29
```

**External metabolomics (if found):**
- Save to: `external_metabolomics/` within workspace

## References

1. H09 Results: `/iterations/iteration_03/hypothesis_09_temporal_rnn_trajectories/{claude_code,codex}/90_results_{agent}.md`
2. H03 Velocity Results: `/iterations/iteration_01/hypothesis_03_tissue_specific_inflammation/{claude_code,codex}/`
3. ADVANCED_ML_REQUIREMENTS.md
4. López-Otín et al. (2023). "Hallmarks of aging: An expanding universe." Cell.
5. Wiley & Campisi (2021). "The metabolic roots of senescence: mechanisms and opportunities for intervention." Nature Metabolism.
6. Bateman et al. (2023). "Senescence and fibrosis: A convergence of pathways." Aging Cell.
