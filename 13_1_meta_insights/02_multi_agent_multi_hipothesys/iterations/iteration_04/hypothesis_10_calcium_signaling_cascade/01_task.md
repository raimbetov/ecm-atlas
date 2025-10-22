# H10 – Calcium Signaling Cascade: S100→CALM/CAMK→LOX/TGM

## Scientific Question
Does S100 calcium signaling regulate ECM crosslinking through calmodulin (CALM) and calcium/calmodulin-dependent kinase (CAMK) mediators, creating a complete mechanistic pathway from calcium sensing to tissue stiffness?

## Background & Rationale

**Discovery from Iteration 03 (H08):**
- **STRONGEST HYPOTHESIS:** Both agents confirmed S100→crosslinking pathway (R²=0.81 Claude, 0.75 Codex)
- **Key correlations:**
  - S100A10→TGM2 (ρ=0.79, p=0.036) — Claude
  - S100B→LOXL4 (ρ=0.74, p=6.7e-4) — Codex
  - S100A10→LOX, S100B→LOXL3 also confirmed
- **Mechanism:** S100 proteins act via calcium-dependent crosslinking (NOT inflammation)

**Critical Gap:**
H08 showed correlation but did NOT identify intermediate signaling proteins between S100 and crosslinking enzymes. Standard calcium signaling pathway includes:
1. **S100 proteins** (calcium sensors)
2. **Calmodulin (CALM1/2/3)** (calcium effectors)
3. **CAMK family** (calcium-dependent kinases)
4. **LOX/TGM enzymes** (crosslinking effectors)

**Research Question:**
Is the pathway: **S100 → CALM → CAMK → LOX/TGM** complete with detectable mediation effects?

## Objectives

### Primary Objective
Identify and validate intermediate mediators (CALM, CAMK) in the S100→crosslinking pathway using mediation analysis and structural modeling.

### Secondary Objectives
1. **Literature validation:** Find published evidence for S100-CALM interactions and CAMK regulation of LOX/TGM
2. **New datasets:** Identify external proteomics data with calcium signaling proteins for independent validation
3. **Structural modeling:** Use AlphaFold to model S100-calmodulin protein-protein interactions
4. **Predictive model:** Predict tissue stiffness from complete calcium signaling signature (R²>0.75 target)

## Hypotheses to Test

### H10.1: Mediation Hypothesis
CALM and CAMK proteins mediate the S100→LOX/TGM relationship (indirect effect >50% of total effect).

### H10.2: Structural Hypothesis
AlphaFold predicts high-confidence S100-CALM binding (pLDDT>70, interface PAE<10Å).

### H10.3: Pathway Completeness
Full pathway (S100+CALM+CAMK) predicts stiffness better than S100-only (ΔR²>0.10).

## Required Analyses

### 1. LITERATURE SEARCH (MANDATORY)

**Search queries:**
```
1. "S100 calcium signaling aging" (2020-2025)
2. "calmodulin CAMK extracellular matrix"
3. "LOX transglutaminase calcium regulation"
4. "S100 calmodulin binding"
5. "CAMK2 fibrosis collagen"
```

**Tasks:**
- Search PubMed, Google Scholar, bioRxiv
- Download top 10 papers by citation count
- Extract: Known S100-CALM interactions, CAMK substrates, calcium-dependent LOX/TGM regulation
- Save: `literature_review.md` with citations and key findings

### 2. NEW DATASET SEARCH (MANDATORY)

**Search targets:**
```
- GEO (Gene Expression Omnibus): calcium signaling + aging
- PRIDE (proteomics): calmodulin OR CAMK
- ProteomeXchange: ECM + calcium
- Human Protein Atlas: tissue expression CALM1/CAMK2A
```

**Criteria:**
- Must include: CALM1/CALM2/CALM3 or CAMK2A/CAMK2B/CAMK2D
- Preferred: Human aging studies, ECM-related tissues
- Download: Raw data files (at least 1 dataset if available)

**If found:**
- Integrate with merged_ecm_aging_zscore.csv
- Run independent validation of S100→CALM→CAMK→LOX pathway

### 3. MEDIATION ANALYSIS

**Pathway models:**
```python
# Total effect
S100 → LOX/TGM (total β)

# Direct effect (with mediator)
S100 → LOX/TGM (direct β) + S100 → CALM → LOX/TGM (indirect β)

# Sequential mediation
S100 → CALM → CAMK → LOX/TGM
```

**Methods:**
- Baron & Kenny method or Sobel test
- Bootstrap confidence intervals (n=10,000)
- Report: % mediated = (indirect / total) × 100

**Proteins to test:**
- **S100 family:** S100A10, S100B, S100A8, S100A9
- **Mediators:** CALM1, CALM2, CALM3, CAMK2A, CAMK2B, CAMK2D, CAMK1
- **Targets:** LOX, LOXL2, LOXL3, LOXL4, TGM2, TGM1

**Success criteria:**
- Indirect effect significant (p<0.05)
- % mediated >30% (meaningful mediation)

### 4. CORRELATION NETWORK EXPANSION

**Update H08 networks with calcium mediators:**
- S100↔CALM correlations (Spearman)
- CALM↔CAMK correlations
- CAMK↔LOX/TGM correlations
- Full pathway: S100→CALM→CAMK→LOX (4-step correlation chain)

**Visualizations:**
- Heatmap: S100×CALM×CAMK×LOX matrix
- Network graph: Nodes=proteins, edges=|ρ|>0.5, color by pathway position

### 5. ALPHAFOLD STRUCTURAL MODELING

**Protein pairs:**
1. S100A10–CALM1 (top correlation from H08)
2. S100B–CALM1
3. S100A9–CALM2

**Methods:**
- Fetch AlphaFold v4 structures from PDB or RCSB
- Protein-protein docking: HDOCK or ClusPro
- Interface analysis: Binding affinity prediction, contact residues

**AlphaFold-Multimer (if available):**
- Model S100A10:CALM1 heterodimer directly
- Extract: pLDDT (confidence), PAE (interface accuracy)

**Output:**
- PDB files: S100-CALM complexes
- Figures: Binding interface with Ca²⁺ binding sites highlighted
- Table: Binding scores, interface residues

### 6. PREDICTIVE MODEL: FULL CALCIUM SIGNALING SIGNATURE

**Features:**
- Model A: S100 proteins only (20 genes) — baseline from H08
- Model B: S100 + CALM (23 genes)
- Model C: S100 + CALM + CAMK (30 genes)

**Architecture:**
- Deep MLP: [input] → [128, 64, 32] → [1] (stiffness proxy)
- Dropout 0.3, Adam optimizer, 200 epochs
- Train/test split: 80/20

**Stiffness proxy (from H08):**
```
Stiffness = 0.5×LOX + 0.3×TGM2 + 0.2×(COL1A1/COL3A1)
```

**Performance metrics:**
- R², MAE, RMSE on test set
- Compare: Model B vs A (does CALM improve?), Model C vs B (does CAMK improve?)

**Success criteria:**
- Model C R² > 0.75 (match H08 performance)
- Model C R² > Model A R² + 0.10 (pathway completeness adds value)

### 7. EXTERNAL VALIDATION (IF NEW DATA FOUND)

**Transfer learning:**
- Train model on merged_ecm_aging_zscore.csv
- Test on external dataset (no retraining)
- Report: R² on external data, correlation external vs predicted

## Deliverables

### Code & Models
- `analysis_calcium_cascade_{agent}.py` — main analysis script
- `calcium_signaling_model_{agent}.pth` — trained PyTorch model
- `literature_search_{agent}.py` — automated PubMed/GEO queries
- `alphafold_docking_{agent}.py` — structural modeling pipeline

### Data Tables
- `mediation_results_{agent}.csv` — indirect/direct effects for all S100-CALM-CAMK-LOX combinations
- `correlation_network_calcium_{agent}.csv` — full S100↔CALM↔CAMK↔LOX matrix
- `model_comparison_{agent}.csv` — R² for models A/B/C
- `literature_findings_{agent}.csv` — extracted evidence from papers
- `new_datasets_{agent}.csv` — metadata for downloaded external data (if any)
- `structural_binding_{agent}.csv` — AlphaFold binding scores

### Visualizations
- `visualizations_{agent}/mediation_diagram_{agent}.png` — path diagram with coefficients
- `visualizations_{agent}/calcium_network_{agent}.png` — 4-layer network (S100→CALM→CAMK→LOX)
- `visualizations_{agent}/alphafold_complex_{agent}.png` — S100-CALM structure
- `visualizations_{agent}/model_performance_{agent}.png` — R² comparison bar chart

### Report
- `90_results_{agent}.md` — comprehensive findings with:
  - Literature synthesis (key papers, citations)
  - Mediation analysis results (which pathways confirmed?)
  - Structural modeling insights (binding affinity, residues)
  - Model performance (does full pathway improve predictions?)
  - External validation (if datasets found)
  - **CRITICAL:** Is CALM/CAMK the missing link? Or is S100→LOX direct?

## Success Criteria

| Metric | Target | Source |
|--------|--------|--------|
| Mediation % | >30% | Indirect effect / total effect |
| CALM-LOX correlation | \|ρ\|>0.5 | Spearman test |
| AlphaFold confidence | pLDDT>70 | Structure quality |
| Model C R² | >0.75 | Stiffness prediction |
| ΔR² (C vs A) | >0.10 | Pathway completeness value |
| Literature papers | ≥5 relevant | PubMed search |
| External datasets | ≥1 downloaded | GEO/PRIDE |

## Expected Outcomes

### If Mediation Confirmed:
**S100 → CALM → CAMK → LOX/TGM** is the complete mechanistic pathway. CALM/CAMK are druggable intermediates.

### If Mediation Weak:
S100 may bind LOX/TGM directly (skip CALM/CAMK). Test alternative pathways (S100→Annexins, S100→Integrin signaling).

### Clinical Translation:
- **Target:** CAMK2 inhibitors (e.g., KN-93) to block crosslinking
- **Biomarker:** CALM1 plasma levels (easier to measure than intracellular CAMK)
- **Combination therapy:** S100 inhibitors + CAMK inhibitors for synergistic effect

## Dataset

**Primary:**
`/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`

**Proteins required:**
- S100 family (20 genes from H08)
- CALM1, CALM2, CALM3
- CAMK1, CAMK2A, CAMK2B, CAMK2D, CAMK4
- LOX, LOXL1-4, TGM1-3 (from H08)

**External data (if found):**
- Save to: `external_datasets/` within workspace

## References

1. H08 Results: `/iterations/iteration_03/hypothesis_08_s100_calcium_signaling/{claude_code,codex}/90_results_{agent}.md`
2. ADVANCED_ML_REQUIREMENTS.md
3. Literature (to be found): S100-calmodulin binding studies, CAMK substrate specificity
4. AlphaFold DB: https://alphafold.ebi.ac.uk/
5. PubMed API: https://www.ncbi.nlm.nih.gov/home/develop/api/
6. GEO API: https://www.ncbi.nlm.nih.gov/geo/info/geo_paccess.html
