# H17 – SERPINE1 Precision Target Validation & Drug Readiness

## Scientific Question
Is SERPINE1 (PAI-1) a druggable precision target for ECM aging intervention, with in-silico knockout confirming mechanistic causality and existing inhibitors (TM5441, SK-216) showing favorable safety profiles?

## Background & Rationale

**SERPINE1 Appears Across Multiple Hypotheses:**
- **H02:** Highest eigenvector centrality (0.891) → Central hub protein
- **H06:** Top 8 biomarker panel (F13B, SERPINF1, **potential SERPINE1**)
- **H12:** Phase II enrichment (metabolic-mechanical transition)
- **Literature:** PAI-1 inhibitors in clinical trials for fibrosis, aging

**Why SERPINE1 is Top Drug Target:**
1. **High Centrality:** Controls multiple ECM pathways (rank #1 in H14)
2. **Druggability:** Small-molecule inhibitors available (TM5441, SK-216, TM5275)
3. **Clinical Trials:** NCT00801112 (PAI-039 for cardiovascular), ongoing longevity trials
4. **Mechanism:** Blocks plasminogen activation → reduces matrix remodeling → stiffness

**Critical Gap from H01-H15:**
- Correlation evidence only (high centrality, biomarker status)
- **NO causal validation** (does knocking out SERPINE1 reverse aging?)
- **NO toxicity/safety assessment** for aging context

**This Hypothesis Bridges Discovery → Translation:**
If SERPINE1 knockout shows reversal AND inhibitors are safe → immediate clinical trial candidate.

## Objectives

### Primary Objective
Perform in-silico SERPINE1 knockout using GNN perturbation models, measuring cascade effects on downstream ECM proteins to establish causality beyond correlation.

### Secondary Objectives
1. Literature meta-analysis of ALL SERPINE1 knockout studies (mouse, human cells)
2. Drug-target interaction networks (existing inhibitors: TM5441, SK-216, TM5275, PAI-039)
3. ADMET prediction (toxicity, off-target effects, bioavailability)
4. Clinical trial landscape analysis
5. Economic/regulatory pathway assessment (FDA approval timeline, cost estimates)

## Hypotheses to Test

### H17.1: Causal Knockout Effect
In-silico SERPINE1 knockout reduces downstream stiffness markers (LOX, TGM2, COL1A1) by ≥30%, confirming mechanistic role beyond correlation.

### H17.2: Literature Validation
Meta-analysis of knockout studies shows consistent anti-aging phenotype (lifespan extension, reduced fibrosis) with I²<50% heterogeneity.

### H17.3: Drug Safety Profile
Existing PAI-1 inhibitors (TM5441, SK-216) show favorable ADMET: no cardiotoxicity (hERG IC50>10µM), no hepatotoxicity (ALT elevation), oral bioavailability >30%.

### H17.4: Clinical Readiness
FDA approval pathway: orphan drug designation possible (rare aging syndrome), Phase I safety data available, estimated timeline 5-7 years to approval.

## Required Analyses

### 1. IN-SILICO KNOCKOUT PERTURBATION

**Load Pre-trained GNN from H05:**
```python
import torch
from torch_geometric.nn import GCNConv, SAGEConv

# Load H05 GNN model (103,037 edges discovered)
gnn_model = torch.load('/iterations/iteration_02/hypothesis_05_hidden_relationships/claude_code/gnn_ecm_aging_claude_code.pth')
gnn_model.eval()

# Baseline: Full network prediction
X_full = protein_features  # All 648 ECM genes
edge_index = edge_list  # 103,037 edges from H05

with torch.no_grad():
    y_baseline = gnn_model(X_full, edge_index)  # Predicted stiffness/aging

# Knockout: Set SERPINE1 to zero
X_knockout = X_full.clone()
serpine1_idx = gene_list.index('SERPINE1')
X_knockout[serpine1_idx, :] = 0  # Simulate knockout

with torch.no_grad():
    y_knockout = gnn_model(X_knockout, edge_index)

# Effect size
delta_stiffness = (y_baseline - y_knockout).mean()
percent_reduction = (delta_stiffness / y_baseline.mean()) * 100

print(f"SERPINE1 Knockout Effect: {percent_reduction:.1f}% reduction in aging score")
```

**Cascade Analysis (which proteins affected?):**
```python
# Per-protein changes
protein_deltas = []

for i, gene in enumerate(gene_list):
    baseline_val = y_baseline[i].item()
    knockout_val = y_knockout[i].item()
    delta = baseline_val - knockout_val

    protein_deltas.append({
        'Gene': gene,
        'Baseline': baseline_val,
        'Knockout': knockout_val,
        'Delta': delta,
        'Percent_Change': (delta / baseline_val) * 100 if baseline_val != 0 else 0
    })

df_cascade = pd.DataFrame(protein_deltas).sort_values('Delta', ascending=False)

# Top affected proteins
top_affected = df_cascade.head(20)
print("Top 20 proteins affected by SERPINE1 knockout:")
print(top_affected[['Gene', 'Percent_Change']])

# Expected: LOX, TGM2, COL1A1 should be in top 20 (mechanism confirmation)
```

**Success Criteria:**
- ≥30% reduction in aging score → STRONG causal effect
- LOX, TGM2, COL1A1 in top 20 affected → mechanism confirmed
- No catastrophic cascade (e.g., essential survival genes unaffected)

### 2. LITERATURE META-ANALYSIS

**Search Strategy:**
```python
from Bio import Entrez
Entrez.email = "your_email@domain.com"

# PubMed search
search_queries = [
    "SERPINE1 knockout aging",
    "PAI-1 knockout lifespan",
    "SERPINE1 deficiency fibrosis",
    "PAI-1 inhibitor aging trial",
    "SERPINE1 senescence"
]

papers = []
for query in search_queries:
    handle = Entrez.esearch(db="pubmed", term=query, retmax=100)
    record = Entrez.read(handle)
    papers.extend(record['IdList'])

papers_unique = list(set(papers))
print(f"Found {len(papers_unique)} unique papers")

# Download abstracts
for pmid in papers_unique[:50]:  # Top 50
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
    abstract = handle.read()
    # Save to file
    with open(f"literature/serpine1_{pmid}.txt", 'w') as f:
        f.write(abstract)
```

**Extract Effect Sizes:**
```python
# For each knockout study, extract:
# - Species (mouse, human cells, etc.)
# - Phenotype (lifespan, fibrosis score, collagen content)
# - Effect size (mean ± SD for KO vs WT)

knockout_studies = [
    {'Study': 'Vaughan 2000', 'Species': 'Mouse', 'Phenotype': 'Lifespan', 'WT': 24.2, 'KO': 26.8, 'SD': 2.1, 'Unit': 'months'},
    {'Study': 'Eren 2014', 'Species': 'Mouse', 'Phenotype': 'Fibrosis_score', 'WT': 3.5, 'KO': 1.2, 'SD': 0.8, 'Unit': 'arbitrary'},
    # ... add more from literature search
]

# Meta-analysis
from statsmodels.stats.meta_analysis import combine_effects

# Convert to effect sizes (Cohen's d)
for study in knockout_studies:
    study['Effect_Size'] = (study['KO'] - study['WT']) / study['SD']

# Combine
effect_sizes = [s['Effect_Size'] for s in knockout_studies]
se_list = [s['SD'] / np.sqrt(30) for s in knockout_studies]  # Assume n=30

combined_effect, combined_se, I2, Q, p_het = combine_effects(effect_sizes, se_list)

print(f"Meta-Analysis: Combined Effect = {combined_effect:.2f} ± {combined_se:.2f}")
print(f"Heterogeneity I² = {I2:.1f}%")
print(f"p_heterogeneity = {p_het:.4f}")
```

**Success Criteria:**
- ≥5 independent studies found
- Combined effect >0.5 (medium-large benefit)
- I²<50% (consistent across studies)

### 3. DRUG-TARGET INTERACTION NETWORKS

**Existing SERPINE1 Inhibitors:**
```python
inhibitors = {
    'TM5441': {
        'SMILES': 'CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC(=O)C4=CC=C(C=C4)C(F)(F)F',
        'IC50_PAI1': 6.4,  # nM
        'Status': 'Preclinical',
        'Developer': 'University of Michigan',
        'Mechanism': 'Allosteric inhibitor'
    },
    'SK-216': {
        'SMILES': 'C1CN(CCN1C(=O)C2=CC=C(C=C2)NC(=O)C3=CC=C(C=C3)OC)C',
        'IC50_PAI1': 5.1,  # nM
        'Status': 'Preclinical',
        'Developer': 'Seoul National',
        'Mechanism': 'Active site blocker'
    },
    'PAI-039': {
        'IC50_PAI1': 9.0,  # µM (weaker)
        'Status': 'Phase II (NCT00801112)',
        'Developer': 'Bristol-Myers Squibb',
        'Indication': 'Cardiovascular fibrosis'
    }
}
```

**Binding Affinity Prediction (AlphaFold + DiffDock):**
```python
from openmm import *
from pdbfixer import PDBFixer

# Get SERPINE1 structure (AlphaFold or PDB)
# AlphaFold ID: Q05682 (SERPINE1 human)

# Use DiffDock for molecular docking
import subprocess

for drug, data in inhibitors.items():
    if 'SMILES' in data:
        # Run DiffDock
        cmd = f"diffdock --protein serpine1_alphafold.pdb --ligand {data['SMILES']} --out {drug}_docking.sdf"
        subprocess.run(cmd, shell=True)

        # Parse binding affinity
        # (DiffDock outputs confidence score)
```

**Off-Target Analysis:**
```python
# Check similarity to other SERPIN family members
serpins = ['SERPINF1', 'SERPINC1', 'SERPINB2', 'SERPINA1']

from rdkit import Chem
from rdkit.Chem import AllChem

tm5441_mol = Chem.MolFromSmiles(inhibitors['TM5441']['SMILES'])
tm5441_fp = AllChem.GetMorganFingerprint(tm5441_mol, 2)

# Check if TM5441 binds other serpins (off-target risk)
# (Would need structures for all serpins)
```

**Success Criteria:**
- ≥2 inhibitors with IC50<100nM (high potency)
- ≥1 inhibitor in clinical trials (de-risked)
- Off-target binding to <3 proteins (selectivity)

### 4. ADMET PREDICTION

**Toxicity Prediction (pkCSM, admetSAR):**
```python
# Use pre-trained ADMET models
# pkCSM: https://biosig.lab.uq.edu.au/pkcsm/

admet_predictions = {}

for drug, data in inhibitors.items():
    if 'SMILES' in data:
        # Predict toxicity endpoints
        # (In practice, use API or local model)

        admet_predictions[drug] = {
            'hERG_IC50': 15.2,  # µM (>10 = safe, cardiotoxicity risk if <10)
            'Hepatotoxicity': 'No',  # Binary prediction
            'Oral_Bioavailability': 0.45,  # 45% (>30% = acceptable)
            'CYP_Inhibition': ['2D6'],  # Which cytochromes inhibited
            'AMES_Mutagenicity': 'No',
            'LD50_rat_oral': 850  # mg/kg (>300 = safe)
        }

df_admet = pd.DataFrame(admet_predictions).T
print("ADMET Predictions:")
print(df_admet)
```

**Blood-Brain Barrier (BBB) Permeability:**
```python
# For aging, peripheral targets preferred (no BBB needed)
# But check to avoid CNS side effects

for drug in inhibitors:
    bbb_permeability = predict_bbb(inhibitors[drug]['SMILES'])
    print(f"{drug}: BBB permeability = {bbb_permeability:.2f} (want <0.1 for peripheral target)")
```

**Success Criteria:**
- hERG IC50 >10µM (no cardiotoxicity)
- No hepatotoxicity predicted
- Oral bioavailability >30%
- BBB permeability <0.1 (peripheral action)

### 5. CLINICAL TRIAL LANDSCAPE

**Search ClinicalTrials.gov:**
```python
import requests

# API search
url = "https://clinicaltrials.gov/api/query/full_studies"
params = {
    'expr': 'SERPINE1 OR PAI-1',
    'fmt': 'json',
    'max_rnk': 100
}

response = requests.get(url, params=params)
trials = response.json()

# Parse trials
trial_data = []
for trial in trials['FullStudiesResponse']['FullStudies']:
    study = trial['Study']
    trial_data.append({
        'NCT': study['ProtocolSection']['IdentificationModule']['NCTId'],
        'Title': study['ProtocolSection']['IdentificationModule']['OfficialTitle'],
        'Status': study['ProtocolSection']['StatusModule']['OverallStatus'],
        'Phase': study['ProtocolSection']['DesignModule'].get('PhaseList', ['N/A'])[0],
        'Condition': study['ProtocolSection']['ConditionsModule']['ConditionList'][0]
    })

df_trials = pd.DataFrame(trial_data)
print(f"Found {len(df_trials)} trials mentioning SERPINE1/PAI-1")
print(df_trials[['NCT', 'Phase', 'Status', 'Condition']])
```

**Regulatory Pathway Analysis:**
```python
# FDA approval timeline estimate

pathway_analysis = {
    'Indication': 'Age-related fibrosis (tissue stiffness)',
    'Orphan_Drug_Status': 'Possible (if <200,000 patients)',
    'Existing_Safety_Data': 'PAI-039 Phase II completed (NCT00801112)',
    'Estimated_Timeline': {
        'Phase_I': '1-2 years (safety, dosing)',
        'Phase_II': '2-3 years (efficacy in aging biomarkers)',
        'Phase_III': '2-3 years (large-scale)',
        'FDA_Review': '1 year',
        'Total': '6-9 years'
    },
    'Cost_Estimate': '$500M - $1B (standard drug development)',
    'Key_Risks': [
        'Bleeding risk (PAI-1 is antifibrinolytic)',
        'Off-target effects on coagulation cascade',
        'Regulatory: aging not FDA-approved indication (need surrogate endpoints)'
    ]
}

print("Clinical Development Pathway:")
print(json.dumps(pathway_analysis, indent=2))
```

**Success Criteria:**
- ≥1 Phase II trial completed (safety de-risked)
- Orphan drug pathway available (faster approval)
- Estimated timeline <10 years

### 6. ECONOMIC ANALYSIS

**Market Size Estimation:**
```python
market_analysis = {
    'Target_Population': {
        'Age_65+_USA': 54_000_000,  # 2025 estimate
        'With_Fibrosis_Biomarkers': 0.15,  # 15% prevalence
        'Addressable': 8_100_000
    },
    'Pricing': {
        'Annual_Cost_per_Patient': 15_000,  # Compare to anti-fibrotic drugs (pirfenidone ~$100k/year)
        'Market_Size': 121_500_000_000  # $121.5B (optimistic)
    },
    'Competition': [
        'Senolytics (dasatinib+quercetin)',
        'mTOR inhibitors (rapamycin)',
        'NAD+ boosters (NMN, NR)',
        'Other anti-fibrotics (pirfenidone, nintedanib)'
    ],
    'Advantage': 'Precision target (high centrality), existing clinical data (PAI-039), oral bioavailability'
}

print("Market Analysis:")
print(json.dumps(market_analysis, indent=2))
```

**ROI for Pharma Companies:**
```python
# Net Present Value (NPV) calculation
development_cost = 800_000_000  # $800M average
timeline_years = 8
annual_revenue_peak = 2_000_000_000  # $2B/year (conservative)
patent_life = 12  # years remaining

discount_rate = 0.10  # 10% per year

# Revenue stream (years 8-20)
npv = 0
for year in range(timeline_years, timeline_years + patent_life):
    revenue = annual_revenue_peak * (0.8 ** max(0, year - timeline_years - 5))  # Peak at year 5, decay
    npv += revenue / ((1 + discount_rate) ** year)

npv -= development_cost

print(f"NPV: ${npv/1e9:.2f}B")
print(f"ROI: {(npv/development_cost)*100:.1f}%")
```

**Success Criteria:**
- Market size >$1B (attracts Big Pharma)
- NPV positive (profitable investment)
- Clear competitive advantage over existing drugs

## Deliverables

### Code & Models
- `serpine1_knockout_simulation_{agent}.py` — In-silico GNN perturbation
- `literature_meta_analysis_{agent}.py` — PubMed search, effect size extraction
- `drug_target_networks_{agent}.py` — Docking, ADMET, off-target analysis
- `clinical_trials_analysis_{agent}.py` — ClinicalTrials.gov scraping, pathway analysis
- `economic_model_{agent}.py` — Market sizing, NPV, ROI

### Data Tables
- `knockout_cascade_{agent}.csv` — Per-protein effects of SERPINE1 knockout (top 100 affected)
- `literature_studies_{agent}.csv` — All knockout studies (PMID, species, effect size, I²)
- `drug_properties_{agent}.csv` — Inhibitors (IC50, ADMET, clinical status)
- `clinical_trials_{agent}.csv` — All PAI-1 trials (NCT, phase, status, results)
- `regulatory_pathway_{agent}.csv` — FDA timeline, costs, risks
- `market_analysis_{agent}.csv` — Population, pricing, competition, NPV

### Visualizations
- `visualizations_{agent}/knockout_waterfall_{agent}.png` — Top 50 proteins affected by SERPINE1 KO
- `visualizations_{agent}/literature_forest_plot_{agent}.png` — Meta-analysis of knockout studies
- `visualizations_{agent}/drug_comparison_radar_{agent}.png` — ADMET properties (TM5441 vs SK-216 vs PAI-039)
- `visualizations_{agent}/clinical_timeline_{agent}.png` — Trial phases over time (Gantt chart)
- `visualizations_{agent}/market_sizing_{agent}.png` — TAM/SAM/SOM funnel
- `visualizations_{agent}/network_perturbation_{agent}.png` — Before/after SERPINE1 knockout (edge weights)

### Report
- `90_results_{agent}.md` — CRITICAL findings:
  - **Causal Validation:** Does knockout reduce aging by ≥30%? Which proteins affected?
  - **Literature Consensus:** Do knockout studies agree (I²<50%)? Lifespan extension?
  - **Drug Readiness:** Are TM5441/SK-216 safe (ADMET pass)? Clinical trials status?
  - **Regulatory Path:** Timeline to FDA approval? Orphan drug possible?
  - **Economic Case:** Market size, NPV, ROI for pharma investment?
  - **FINAL VERDICT:** Is SERPINE1 a GO/NO-GO for clinical development?
  - **Recommendations:** Phase I trial design, dosing, endpoints, patient selection

## Success Criteria

| Metric | Target | Source |
|--------|--------|--------|
| Knockout aging reduction | ≥30% | In-silico GNN perturbation |
| Downstream cascade (LOX/TGM2 affected) | ≥2/3 in top 20 | Perturbation analysis |
| Literature studies found | ≥5 independent | PubMed meta-analysis |
| Meta-analysis I² | <50% | Random-effects model |
| Inhibitors identified | ≥2 with IC50<100nM | Drug databases |
| Clinical trials found | ≥1 Phase II+ | ClinicalTrials.gov |
| ADMET safety (hERG, hepatotox) | PASS (hERG>10µM, no hepatotox) | pkCSM predictions |
| Market size | >$1B | Economic model |
| Overall GO/NO-GO | GO if ≥6/8 criteria met | Comprehensive assessment |

## Expected Outcomes

### Scenario 1: STRONG GO (Clinical Development Recommended)
- Knockout reduces aging by 35-50% (strong causal effect)
- Literature: 8+ studies, combined effect >0.7, I²=25%
- TM5441: IC50=6.4nM, hERG IC50=18µM, no hepatotoxicity, oral bioavail=52%
- PAI-039 Phase II completed with safety data
- Market: $2B+ addressable, NPV=$4B, ROI=400%
- **Action:** Initiate Phase I trial, partner with Big Pharma, file IND

### Scenario 2: MODERATE GO (Further Preclinical Work)
- Knockout reduces aging by 20-30% (moderate effect)
- Literature heterogeneous (I²=60%), some studies negative
- Drugs show toxicity concerns (hERG IC50=8µM, borderline)
- No completed Phase II trials
- **Action:** Additional animal studies, optimize drug structure, seek orphan designation

### Scenario 3: NO-GO (Deprioritize)
- Knockout <15% effect (correlation ≠ causation)
- Literature inconsistent (I²>75%), publication bias detected
- Drugs fail ADMET (hepatotoxic, cardiotoxic)
- Bleeding risks outweigh benefits
- **Action:** Focus on other targets (e.g., LOX, TGM2 direct inhibitors)

### Scenario 4: ALTERNATIVE PATH (Biomarker, Not Drug)
- Knockout effective BUT no safe inhibitors
- SERPINE1 useful as diagnostic/prognostic biomarker
- **Action:** Develop ELISA/immunoassay for clinical labs, not therapeutics

## Dataset

**Primary (for baseline correlations):**
`/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`

**Models to Load:**
- GNN from H05: `/iterations/iteration_02/hypothesis_05_hidden_relationships/{claude_code,codex}/gnn_ecm_aging_{agent}.pth`
- Centrality data from H14: `/iterations/iteration_04/hypothesis_14_serpin_centrality_resolution/{claude_code,codex}/centrality_comparison_{agent}.csv`

**External Resources:**
- PubMed API: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- ClinicalTrials.gov API: https://clinicaltrials.gov/api/
- ChEMBL (drug database): https://www.ebi.ac.uk/chembl/
- pkCSM (ADMET): https://biosig.lab.uq.edu.au/pkcsm/
- AlphaFold (SERPINE1 structure): Q05682

## References

1. **H02 Serpin Centrality**: `/iterations/iteration_01/hypothesis_02_serpin_centrality/`
2. **H05 Hidden Relationships**: `/iterations/iteration_02/hypothesis_05_hidden_relationships/`
3. **H14 Eigenvector Validation**: `/iterations/iteration_04/hypothesis_14_serpin_centrality_resolution/`
4. **Vaughan et al. 2000**: PAI-1 deficiency protects against age-related cardiac fibrosis. PMID: 10636155
5. **Eren et al. 2014**: PAI-1–regulated extracellular proteolysis governs senescence and survival. PMID: 25237099
6. **TM5441 Discovery**: Baker et al. 2013. PMID: 23897865
7. **Clinical Trial NCT00801112**: PAI-039 for cardiovascular fibrosis
