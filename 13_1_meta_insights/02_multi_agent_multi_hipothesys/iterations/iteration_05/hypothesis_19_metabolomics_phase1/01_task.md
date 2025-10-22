# H19 – Metabolomics Integration: Phase I Markers & Transition Prediction

## Scientific Question
Can integrating metabolomics data (ATP, NAD+, lactate, pyruvate) with ECM proteomics reveal Phase I metabolic markers missing from protein-only analysis, enabling earlier detection of the metabolic→mechanical transition (v=1.65-2.17) before irreversible fibrosis occurs?

## Background & Rationale

**H12 Discovered Metabolic-Mechanical Transition:**
- Phase I (v<1.65): Metabolic changes dominant (hypothetical, not proven)
- Transition zone (v=1.65-2.17): Mixed metabolic+mechanical
- Phase II (v>2.17): Mechanical changes (collagen, crosslinking) — CONFIRMED

**Critical Gap: Phase I Not Validated**
- Proteomics captures Phase II well (COL1A1, LOX, TGM2)
- But metabolic intermediates (ATP, NAD+, lactate) are **NOT proteins** → invisible to proteomics
- **Risk:** We may be detecting aging TOO LATE (after metabolic damage done)

**Why Metabolomics is Critical:**
1. **Earlier Detection:** Metabolic changes precede protein expression by days-weeks
2. **Reversibility:** ATP/NAD+ depletion can be reversed (NMN, nicotinamide); collagen crosslinking cannot
3. **Drug Targets:** NAD+ boosters already in clinical trials (Elysium Health, ChromaDex)
4. **Phase I Validation:** If ATP↓ NAD+↓ lactate↑ in low-velocity tissues → Phase I CONFIRMED

**Metabolomics Databases:**
- Metabolomics Workbench: 1,600+ studies, aging metabolomics available
- MetaboLights: EMBL-EBI repository
- HMDB (Human Metabolome Database): Reference values

**Expected Findings:**
- Phase I tissues (v<1.65): ATP↓30%, NAD+↓40%, lactate/pyruvate ratio↑2×
- Transition (v=1.65-2.17): Mixed metabolic+protein changes
- Phase II (v>2.17): Metabolic changes plateau, protein changes accelerate

**Clinical Impact:**
If Phase I markers validated → intervention window BEFORE fibrosis (NAD+ boosters can prevent Phase II entry).

## Objectives

### Primary Objective
Download and integrate aging metabolomics datasets (≥2 independent studies), correlate ATP/NAD+/lactate/pyruvate with ECM proteomics velocity (H03), and validate Phase I metabolic signature in low-velocity tissues.

### Secondary Objectives
1. Search Metabolomics Workbench, MetaboLights, HMDB for tissue aging studies
2. Download raw metabolomics data (LC-MS, GC-MS, NMR)
3. Normalize metabolomics (Z-scores, same pipeline as proteomics)
4. Correlate metabolites with tissue velocity (from H03)
5. Test Phase I hypothesis: ATP/NAD+ anticorrelate with velocity (r<-0.5)
6. Multi-omics integration: joint PCA of proteomics + metabolomics
7. Predict Phase II transition timing from metabolic markers

## Hypotheses to Test

### H19.1: Phase I Metabolic Signature
Tissues with v<1.65 show ATP↓≥20%, NAD+↓≥30%, lactate/pyruvate ratio↑≥1.5× compared to young controls, confirming metabolic phase precedes mechanical phase.

### H19.2: Metabolite-Velocity Anticorrelation
ATP and NAD+ negatively correlate with tissue velocity (Spearman ρ<-0.60), while lactate/pyruvate ratio positively correlates (ρ>0.50).

### H19.3: Early Warning Biomarkers
Metabolic markers (ATP, NAD+) change ≥3 months before proteomic markers (COL1A1, LOX) in longitudinal datasets, enabling early intervention.

### H19.4: Multi-Omics Synergy
Joint PCA of proteomics + metabolomics explains >95% variance (vs 89% proteomics-only), revealing hidden aging axes invisible to single-omic approaches.

## Required Analyses

### 1. DATABASE SEARCH & DATASET SELECTION

**Search Metabolomics Workbench:**
```python
import requests

# Metabolomics Workbench REST API
base_url = "https://www.metabolomicsworkbench.org/rest"

# Search for aging studies
search_params = {
    'context': 'study',
    'input': 'study_type',
    'search': 'aging OR senescence OR fibrosis',
    'output': 'json'
}

response = requests.get(f"{base_url}/study/study_id/all/summary", params=search_params)
studies = response.json()

# Filter for tissue studies (not cell culture)
tissue_studies = [s for s in studies if 'tissue' in s['study_type'].lower()]
print(f"Found {len(tissue_studies)} tissue aging metabolomics studies")

# Target studies with ATP, NAD, lactate, pyruvate
for study in tissue_studies[:20]:
    print(f"ST{study['study_id']}: {study['study_title']}")
    # Check if metabolites of interest measured
    metabolites = requests.get(f"{base_url}/study/study_id/{study['study_id']}/metabolites").json()
    has_atp = any('ATP' in m['metabolite_name'] for m in metabolites)
    has_nad = any('NAD' in m['metabolite_name'] for m in metabolites)
    has_lactate = any('lactate' in m['metabolite_name'].lower() for m in metabolites)

    if has_atp and has_nad:
        print(f"  → SELECTED (has ATP, NAD)")
        selected_studies.append(study['study_id'])
```

**Search MetaboLights:**
```python
# MetaboLights API
metabo_url = "https://www.ebi.ac.uk/metabolights/ws/studies"

response = requests.get(metabo_url)
metabo_studies = response.json()['content']

# Search for aging keywords
aging_studies = [s for s in metabo_studies if 'aging' in s['title'].lower() or 'age' in s['title'].lower()]
print(f"Found {len(aging_studies)} aging studies in MetaboLights")

for study in aging_studies[:10]:
    print(f"{study['accession']}: {study['title']}")
```

**Success Criteria:**
- ≥2 independent metabolomics datasets downloaded
- Datasets include ATP, NAD+, lactate, pyruvate measurements
- Tissue-level data (not cell culture)
- Age metadata available

### 2. DATA DOWNLOAD & PREPROCESSING

**Download from Metabolomics Workbench:**
```python
# Example: Download ST001234
study_id = "ST001234"

# Get data files
data_url = f"{base_url}/study/study_id/{study_id}/data"
data = requests.get(data_url).json()

# Save raw data
df_metabolomics = pd.DataFrame(data)
df_metabolomics.to_csv(f"metabolomics_data/ST{study_id}_raw.csv", index=False)

print(f"Downloaded: {len(df_metabolomics)} samples, {len(df_metabolomics.columns)} metabolites")
```

**Normalization (Match Proteomics Pipeline):**
```python
from universal_zscore_function import calculate_zscore

# Load raw metabolomics
df_raw = pd.read_csv("metabolomics_data/ST001234_raw.csv")

# Pivot to long format (if needed)
# Columns: Sample_ID, Metabolite_Name, Abundance, Tissue, Age

df_long = pd.melt(
    df_raw,
    id_vars=['Sample_ID', 'Tissue', 'Age'],
    var_name='Metabolite_Name',
    value_name='Abundance'
)

# Calculate Z-scores (same as proteomics)
df_metabolomics_zscore = calculate_zscore(
    df_long,
    group_by=['Tissue'],  # Within-tissue normalization
    value_col='Abundance'
)

# Save preprocessed
df_metabolomics_zscore.to_csv(f"metabolomics_data/ST001234_zscore.csv", index=False)
```

**Success Criteria:**
- Z-scores calculated for all metabolites
- Same normalization as `/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`
- Metadata (Tissue, Age) aligned with proteomics

### 3. METABOLITE-VELOCITY CORRELATION

**Load H03 Tissue Velocities:**
```python
# From H03 analysis
h03_velocities = {
    'Lung': 4.29,
    'Tubulointerstitial': 3.45,
    'Liver': 1.34,
    'Skeletal_muscle': 1.02,
    'Heart': 2.89,
    'Aorta': 3.12,
    # ... all 17 tissues
}
```

**Correlate Metabolites with Velocity:**
```python
from scipy.stats import spearmanr

# Filter for key metabolites
key_metabolites = ['ATP', 'NAD+', 'NADH', 'Lactate', 'Pyruvate', 'Glucose', 'Citrate']

metabolite_velocity_corr = []

for metabolite in key_metabolites:
    # Get mean Z-score per tissue
    metabolite_data = df_metabolomics_zscore[df_metabolomics_zscore['Metabolite_Name'] == metabolite]
    tissue_means = metabolite_data.groupby('Tissue')['Z_score'].mean()

    # Match tissues with H03 velocities
    matched_tissues = []
    metabolite_values = []
    velocity_values = []

    for tissue, velocity in h03_velocities.items():
        if tissue in tissue_means.index:
            matched_tissues.append(tissue)
            metabolite_values.append(tissue_means[tissue])
            velocity_values.append(velocity)

    # Correlation
    if len(matched_tissues) >= 5:  # Need at least 5 tissues
        rho, p_value = spearmanr(metabolite_values, velocity_values)

        metabolite_velocity_corr.append({
            'Metabolite': metabolite,
            'n_tissues': len(matched_tissues),
            'Spearman_rho': rho,
            'p_value': p_value
        })

        print(f"{metabolite}: ρ={rho:.3f}, p={p_value:.4f} (n={len(matched_tissues)})")

df_corr = pd.DataFrame(metabolite_velocity_corr)
df_corr.to_csv(f'metabolite_velocity_correlation_{agent}.csv', index=False)
```

**Success Criteria:**
- ATP: ρ<-0.50, p<0.05 (anticorrelation with velocity)
- NAD+: ρ<-0.50, p<0.05
- Lactate/Pyruvate ratio: ρ>0.50, p<0.05 (positive correlation)

### 4. PHASE I SIGNATURE VALIDATION

**Identify Phase I Tissues (v<1.65):**
```python
phase1_tissues = [tissue for tissue, v in h03_velocities.items() if v < 1.65]
phase2_tissues = [tissue for tissue, v in h03_velocities.items() if v > 2.17]
transition_tissues = [tissue for tissue, v in h03_velocities.items() if 1.65 <= v <= 2.17]

print(f"Phase I: {phase1_tissues}")
print(f"Transition: {transition_tissues}")
print(f"Phase II: {phase2_tissues}")
```

**Compare Metabolic Profiles:**
```python
from scipy.stats import ttest_ind

phase1_vs_phase2 = []

for metabolite in key_metabolites:
    metabolite_data = df_metabolomics_zscore[df_metabolomics_zscore['Metabolite_Name'] == metabolite]

    phase1_values = metabolite_data[metabolite_data['Tissue'].isin(phase1_tissues)]['Z_score']
    phase2_values = metabolite_data[metabolite_data['Tissue'].isin(phase2_tissues)]['Z_score']

    # T-test
    t_stat, p_value = ttest_ind(phase1_values, phase2_values)
    mean_delta = phase1_values.mean() - phase2_values.mean()
    percent_change = (mean_delta / phase2_values.mean()) * 100 if phase2_values.mean() != 0 else 0

    phase1_vs_phase2.append({
        'Metabolite': metabolite,
        'Phase1_mean': phase1_values.mean(),
        'Phase2_mean': phase2_values.mean(),
        'Delta': mean_delta,
        'Percent_Change': percent_change,
        't_stat': t_stat,
        'p_value': p_value
    })

    print(f"{metabolite}: Phase I vs Phase II Δ={percent_change:.1f}%, p={p_value:.4f}")

df_phase_comparison = pd.DataFrame(phase1_vs_phase2)
df_phase_comparison.to_csv(f'phase1_vs_phase2_metabolites_{agent}.csv', index=False)
```

**Success Criteria:**
- ATP: Phase I ≥20% lower than Phase II, p<0.05
- NAD+: Phase I ≥30% lower, p<0.05
- Lactate/Pyruvate: Phase I ≥1.5× higher, p<0.05

### 5. MULTI-OMICS INTEGRATION (JOINT PCA)

**Merge Proteomics + Metabolomics:**
```python
# Load proteomics
df_proteomics = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# Pivot proteomics to wide (samples × proteins)
proteomics_wide = df_proteomics.pivot_table(
    index=['Sample_ID', 'Tissue', 'Age'],
    columns='Gene_Symbol',
    values='Z_score'
).fillna(0)

# Pivot metabolomics to wide (samples × metabolites)
metabolomics_wide = df_metabolomics_zscore.pivot_table(
    index=['Sample_ID', 'Tissue', 'Age'],
    columns='Metabolite_Name',
    values='Z_score'
).fillna(0)

# Merge on Sample_ID (inner join — only matched samples)
df_multi_omics = proteomics_wide.join(metabolomics_wide, how='inner')

print(f"Multi-omics: {df_multi_omics.shape[0]} samples, {df_multi_omics.shape[1]} features (proteins+metabolites)")
```

**Joint PCA:**
```python
from sklearn.decomposition import PCA

# PCA on proteomics only
pca_proteomics = PCA(n_components=32)
pca_proteomics.fit(proteomics_wide)
variance_proteomics = pca_proteomics.explained_variance_ratio_.sum()

# PCA on multi-omics (proteomics + metabolomics)
pca_multi = PCA(n_components=32)
pca_multi.fit(df_multi_omics)
variance_multi = pca_multi.explained_variance_ratio_.sum()

print(f"Variance explained (proteomics only): {variance_proteomics:.2%}")
print(f"Variance explained (multi-omics): {variance_multi:.2%}")

# Expected: variance_multi > variance_proteomics (e.g., 95% vs 89%)
```

**Identify Metabolic Principal Components:**
```python
# Which PCs capture metabolic variance?
loadings_multi = pca_multi.components_  # (32, n_features)

# Split loadings into proteomics vs metabolomics
n_proteins = proteomics_wide.shape[1]
loadings_proteins = loadings_multi[:, :n_proteins]
loadings_metabolites = loadings_multi[:, n_proteins:]

# PC contribution from metabolites
metabolite_contribution = np.abs(loadings_metabolites).sum(axis=1)
protein_contribution = np.abs(loadings_proteins).sum(axis=1)

for i, (met_contrib, prot_contrib) in enumerate(zip(metabolite_contribution[:10], protein_contribution[:10])):
    ratio = met_contrib / (met_contrib + prot_contrib)
    print(f"PC{i+1}: Metabolite contribution = {ratio:.1%}")

# Expected: PC1 or PC2 dominated by metabolites (>60%)
```

**Success Criteria:**
- Multi-omics variance explained ≥95% (vs 89% proteomics-only)
- ≥1 PC dominated by metabolites (>60% contribution)

### 6. TEMPORAL PREDICTION: METABOLITES LEAD PROTEINS?

**Longitudinal Analysis (if available):**
```python
# If dataset has longitudinal timepoints (e.g., t=0, 3mo, 6mo, 12mo)

# Example: Check if ATP drops BEFORE COL1A1 rises

longitudinal_data = df_multi_omics[df_multi_omics['Sample_ID'].str.contains('_t')]  # Timepoint suffix

# Group by subject
subjects = longitudinal_data.index.get_level_values('Sample_ID').str.split('_').str[0].unique()

temporal_leads = []

for subject in subjects:
    subject_data = longitudinal_data[longitudinal_data.index.get_level_values('Sample_ID').str.startswith(subject)]

    # Sort by timepoint
    subject_data = subject_data.sort_index()

    # Check if ATP drops before COL1A1 rises
    atp_trajectory = subject_data['ATP'].values
    col1a1_trajectory = subject_data['COL1A1'].values

    # Find first time ATP drops <-1 Z-score
    atp_drop_time = np.argmax(atp_trajectory < -1) if any(atp_trajectory < -1) else None

    # Find first time COL1A1 rises >1 Z-score
    col1a1_rise_time = np.argmax(col1a1_trajectory > 1) if any(col1a1_trajectory > 1) else None

    if atp_drop_time is not None and col1a1_rise_time is not None:
        temporal_lead = atp_drop_time - col1a1_rise_time  # Negative = ATP leads
        temporal_leads.append(temporal_lead)

mean_lead = np.mean(temporal_leads) if temporal_leads else None
print(f"ATP drops {-mean_lead:.1f} timepoints BEFORE COL1A1 rises")
```

**Success Criteria:**
- ATP/NAD+ changes precede protein changes by ≥1 timepoint (e.g., 3 months)

### 7. INTERVENTION WINDOW PREDICTION

**Predict Phase II Entry from Metabolic Markers:**
```python
from sklearn.linear_model import LogisticRegression

# Binary classification: Phase I vs Phase II
X_metabolites = df_multi_omics[['ATP', 'NAD+', 'NADH', 'Lactate', 'Pyruvate']]
y_phase = [1 if v > 2.17 else 0 for v in df_multi_omics['velocity']]  # 1=Phase II, 0=Phase I

# Train classifier
clf = LogisticRegression()
clf.fit(X_metabolites, y_phase)

# Predict probability of Phase II entry
proba_phase2 = clf.predict_proba(X_metabolites)[:, 1]

# For subjects in transition zone (v=1.65-2.17), predict if they will enter Phase II
transition_samples = df_multi_omics[(df_multi_omics['velocity'] >= 1.65) & (df_multi_omics['velocity'] <= 2.17)]
X_transition = transition_samples[['ATP', 'NAD+', 'NADH', 'Lactate', 'Pyruvate']]
proba_transition = clf.predict_proba(X_transition)[:, 1]

print(f"Transition samples: {len(transition_samples)}")
print(f"Mean probability Phase II entry: {proba_transition.mean():.2%}")

# High-risk samples (>70% probability) → prioritize for NAD+ intervention
high_risk = transition_samples[proba_transition > 0.70]
print(f"High-risk samples (>70% Phase II risk): {len(high_risk)}")
```

**Success Criteria:**
- Classifier AUC>0.80 (metabolites predict Phase II entry)
- Identify ≥10% of transition samples as high-risk

## Deliverables

### Code & Models
- `metabolomics_search_{agent}.py` — Database search (Metabolomics Workbench, MetaboLights)
- `metabolomics_download_{agent}.py` — Data download and preprocessing
- `metabolite_velocity_correlation_{agent}.py` — Correlation with H03 velocities
- `phase1_validation_{agent}.py` — Phase I vs Phase II comparison
- `multi_omics_integration_{agent}.py` — Joint PCA, variance decomposition
- `temporal_prediction_{agent}.py` — Longitudinal lead-lag analysis
- `intervention_window_{agent}.py` — Risk prediction for Phase II entry

### Data Tables
- `metabolomics_datasets_{agent}.csv` — Downloaded studies (accession, n_samples, metabolites)
- `metabolite_velocity_correlation_{agent}.csv` — Spearman ρ for ATP, NAD, lactate, pyruvate
- `phase1_vs_phase2_metabolites_{agent}.csv` — Mean, delta, % change, p-value
- `multi_omics_pca_variance_{agent}.csv` — Variance explained (proteomics vs multi-omics)
- `pc_loadings_{agent}.csv` — Top metabolites/proteins per PC
- `temporal_leads_{agent}.csv` — Lead-lag analysis (metabolites precede proteins?)
- `phase2_risk_prediction_{agent}.csv` — High-risk samples in transition zone

### Visualizations
- `visualizations_{agent}/metabolite_velocity_scatter_{agent}.png` — ATP/NAD vs velocity (scatterplot)
- `visualizations_{agent}/phase1_vs_phase2_boxplot_{agent}.png` — Metabolite distributions by phase
- `visualizations_{agent}/multi_omics_pca_biplot_{agent}.png` — PC1 vs PC2 (proteins=blue, metabolites=red)
- `visualizations_{agent}/variance_comparison_{agent}.png` — Bar chart (proteomics vs multi-omics variance)
- `visualizations_{agent}/temporal_trajectory_{agent}.png` — ATP vs COL1A1 over time (lead-lag)
- `visualizations_{agent}/intervention_window_{agent}.png` — Risk score distribution (transition samples)

### Report
- `90_results_{agent}.md` — CRITICAL findings:
  - **Dataset Success:** How many metabolomics datasets downloaded? ATP/NAD coverage?
  - **Phase I Validation:** Does ATP↓NAD+↓ in low-velocity tissues? Effect sizes?
  - **Metabolite-Velocity Correlation:** ρ for ATP, NAD, lactate? p-values?
  - **Multi-Omics Synergy:** Variance gain from adding metabolomics?
  - **Temporal Leads:** Do metabolites change before proteins? How many months?
  - **Intervention Window:** Can we predict Phase II entry? High-risk samples identified?
  - **FINAL VERDICT:** Is Phase I metabolic signature CONFIRMED? Clinical readiness?
  - **Recommendations:** NAD+ intervention trials? Target high-risk patients in transition zone?

## Success Criteria

| Metric | Target | Source |
|--------|--------|--------|
| Metabolomics datasets downloaded | ≥2 independent | Metabolomics Workbench, MetaboLights |
| ATP-velocity correlation | ρ<-0.50, p<0.05 | Spearman correlation |
| NAD-velocity correlation | ρ<-0.50, p<0.05 | Spearman correlation |
| Lactate/Pyruvate-velocity correlation | ρ>0.50, p<0.05 | Spearman correlation |
| Phase I ATP depletion | ≥20% vs Phase II, p<0.05 | T-test |
| Phase I NAD depletion | ≥30% vs Phase II, p<0.05 | T-test |
| Multi-omics variance explained | ≥95% (vs 89% proteomics) | Joint PCA |
| Temporal lead (metabolites before proteins) | ≥1 timepoint (~3mo) | Longitudinal analysis |
| Phase II entry prediction AUC | ≥0.80 | Logistic regression |
| Overall VALIDATION | ≥7/9 criteria met | Comprehensive |

## Expected Outcomes

### Scenario 1: PHASE I CONFIRMED (Strong Validation)
- 3 metabolomics datasets downloaded (ATP, NAD, lactate, pyruvate all measured)
- ATP: ρ=-0.68, p=0.002 (strong anticorrelation with velocity)
- NAD: ρ=-0.72, p<0.001
- Phase I: ATP↓35%, NAD↓42% vs Phase II
- Multi-omics variance: 96% (vs 89% proteomics-only)
- Temporal lead: ATP drops 4.2 months BEFORE COL1A1 rises
- **Action:** Phase I VALIDATED, launch NAD+ booster trials (NMN, NR), target v<1.65 tissues

### Scenario 2: PARTIAL VALIDATION (Some Markers Work)
- NAD correlates (ρ=-0.55) but ATP doesn't (ρ=-0.22)
- Only 1-2 metabolites confirm Phase I
- Multi-omics variance gain modest (91% vs 89%)
- **Action:** Focus on validated markers only (NAD), refine Phase I definition

### Scenario 3: NO VALIDATION (Phase I Not Metabolic)
- ATP, NAD show no correlation with velocity (ρ~0)
- No difference between Phase I and Phase II metabolic profiles
- Multi-omics variance unchanged
- **Action:** Reject Phase I metabolic hypothesis, H12 transition is protein-driven only

### Scenario 4: DATA UNAVAILABLE (Technical Failure)
- No aging metabolomics datasets found with ATP/NAD measurements
- Datasets exist but tissue-level data not available (only serum/plasma)
- **Action:** Generate new metabolomics data (expensive, 6-12 months timeline) or use proxies

## Dataset

**Primary (Proteomics):**
`/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`

**External (Metabolomics — to download):**
- Metabolomics Workbench: https://www.metabolomicsworkbench.org/
- MetaboLights: https://www.ebi.ac.uk/metabolights/
- HMDB: https://hmdb.ca/

**Reference Data:**
- H03 Tissue Velocities: `/iterations/iteration_01/hypothesis_03_tissue_specific_inflammation/`
- H12 Phase Transition: `/iterations/iteration_04/hypothesis_12_metabolic_mechanical_transition/`

## References

1. **H03 Tissue Velocities**: `/iterations/iteration_01/hypothesis_03_tissue_specific_inflammation/`
2. **H12 Metabolic-Mechanical Transition**: `/iterations/iteration_04/hypothesis_12_metabolic_mechanical_transition/`
3. **NAD+ and Aging**: Rajman et al. 2018, "Therapeutic Potential of NAD-Boosting Molecules" (PMID: 29514064)
4. **Metabolomics Workbench**: Sud et al. 2016, Nucleic Acids Research
5. **MetaboLights**: Haug et al. 2020, Nucleic Acids Research
6. **Phase I Metabolism**: Wallace & Fan 2010, "Energetics, epigenetics, mitochondrial genetics" (PMID: 20166139)
