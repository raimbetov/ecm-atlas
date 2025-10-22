# SERPINE1 Precision Target Validation — Codex Results

## 1. GNN Knockout Perturbation (In-silico)
- Model: Pre-trained H05 multi-head GAT (`gnn_weights_codex.pth`) evaluated on full ECM network.
- Outcome: SERPINE1 knockout increased the mean aging score by 0.02% (`global_percent_reduction = -0.0203%`).
- Top cascade shifts: `Sulf2` (+0.57%), `SERPINB6` (+0.07%), `NID1` (+0.05%). Core stiffness markers showed negligible change (`LOX` +0.004%, `TGM2` −0.002%, `COL1A1` −0.020%).
- Interpretation: Network dynamics suggest SERPINE1 removal **fails to produce the ≥30% rejuvenation target** and slightly worsens composite aging signal.
- Artifacts: `data/knockout_cascade_codex.csv`, `data/knockout_summary_codex.json`, `visualizations_codex/knockout_waterfall_codex.png`, `visualizations_codex/network_perturbation_codex.png`.

## 2. Literature Meta-analysis of SERPINE1 Knockout/Inhibition
- Dataset: 8 curated knockout/inhibitor studies spanning cardiac fibrosis, senescence, metabolic models (`data/literature_studies_codex.csv`).
- Effect metric: Cohen's d (positive = higher pro-aging outcome). Random-effects model ⇒ **d = −1.10 (95% CI −3.03 to 0.82, p = 0.26)**.
- Heterogeneity: Q = 223.3, I² = 96.9% → extremely heterogeneous evidence.
- Direction: 7/8 studies report beneficial phenotypes (reduced fibrosis, lifespan gains), but variance and opposing cardioprotective reports (e.g., Xu et al., Blood 2010) prevent consensus.
- Visualization: `visualizations_codex/literature_forest_plot_codex.png`.
- Takeaway: Literature leans pro-knockout but is inconsistent and not statistically conclusive.

## 3. Docking with AlphaFold SERPINE1 (P05121)
- Receptor: AlphaFold v6 model (downloaded via API, rigid PDBQT prepared with Open Babel `-xr`).
- Ligands: TM5441 (CID 44250349), SK-216 (CID 23624303). `vina` exhaustiveness 24, 50 Å search box.
- Binding scores: TM5441 −7.89 kcal/mol, SK-216 −7.52 kcal/mol (`data/docking/docking_results_codex.csv`).
- Both pose below the −7 kcal/mol heuristic but lack pocket selectivity; large grid indicates diffuse binding, no enrichment of ECM targets.
- Artifact: `visualizations_codex/docking_scores_codex.png`.

## 4. ADMET Predictions (ADMETlab 2.0 API)
- Workflow: Programmatic CSRF session (`admet_prediction_codex.py`), outputs in `data/drug_admet_codex.csv`.
- Key findings:
  - **Absorption**: HIA, F20%, F30% all `Very Low` for TM5441 and SK-216 (poor oral bioavailability predicted).
  - **Distribution**: PPB >98% (high plasma protein binding); negligible free fraction (Fu ≤1.1%).
  - **Metabolism**: CYP2C9 inhibition flagged (TM5441 – low risk; SK-216 data sparse), potential drug-drug interactions.
  - **Cardiotoxicity**: hERG Blocker = `Negative` (TM5441) / `Very Low` (SK-216) – meets hERG>10 µM safety goal.
  - **Hepatic risk**: DILI = `High` for both; TM5441 literature (Lang et al. 2019) confirms reactive metabolite liabilities.
  - **Acute toxicity**: Rat oral toxicity `Slight` (TM5441) vs `High` (SK-216) risk categories.
- Conclusion: ADMET profile dominated by poor exposure and liver liability despite acceptable cardiac safety.

## 5. ClinicalTrials.gov Landscape
- Query (`clinical_trials_codex.py`): “SERPINE1 OR PAI-1 OR tiplaxtinin” → 250 studies.
- Breakdown: Phase ≥II = 85 trials (57 completed); majority observational or targeting metabolic sequelae, **no registered interventional trials using TM5441, SK-216, or direct PAI-1 inhibitors**.
- Active interventional studies mentioning PAI-1 focus on biomarker monitoring (e.g., varicose vein remodeling, obesity diets) rather than therapeutic inhibition.
- Dataset: `data/clinical_trials_codex.csv`, summary in `data/clinical_trials_summary_codex.json`.

## Integrated Assessment vs. Success Criteria
| Metric | Target | Observation |
|--------|--------|-------------|
| Knockout aging reduction | ≥30% drop | **Failed** (−0.02%, slight worsening). |
| LOX/TGM2/COL1A1 cascade | ≥2 of 3 in top 20 | **Failed** (changes <0.02%). |
| Literature consensus | ≥5 studies, I² <50% | 8 studies but **I² = 96.9%**, conflicting cardiometabolic outcomes. |
| Docking affinity | Potent (<−8 kcal/mol) | TM5441 borderline; SK-216 weaker. |
| ADMET safety (hERG, hepatotoxicity) | hERG>10 µM, no hepatotox | hERG acceptable; **DILI risk high**; poor oral exposure. |
| Clinical readiness | ≥1 Phase II+ on target | None for direct PAI-1 inhibitors. |
| Economic/Regulatory | (not met) | Not pursued due to above failures. |

## Final Verdict — NO GO
- The GNN perturbation contradicts the hypothesised rejuvenation, indicating SERPINE1 expression may be compensatory in ECM aging rather than causal.
- Literature meta-analysis shows heavy heterogeneity and cardioprotective counter-evidence; mechanistic certainty is lacking.
- TM5441/SK-216 demonstrate only modest docking affinity and predicted oral bioavailability shortfalls coupled with hepatic risk signals.
- No clinical trial momentum exists for SERPINE1 inhibitors in aging contexts.

**Recommendation:** De-prioritize SERPINE1 as a precision anti-aging target. Redirect efforts toward downstream ECM effectors (e.g., LOX, TGM2) or combination strategies that modulate PAI-1 pathways indirectly while mitigating fibrosis and hepatotoxic liabilities.

## Key Files
- Knockout: `serpine1_knockout_codex.py`, `data/knockout_cascade_codex.csv`
- Meta-analysis: `literature_meta_codex.py`, `visualizations_codex/literature_forest_plot_codex.png`
- Docking: `drug_docking_codex.py`, `data/docking/docking_results_codex.csv`
- ADMET: `admet_prediction_codex.py`, `data/drug_admet_codex.csv`
- Trials: `clinical_trials_codex.py`, `data/clinical_trials_codex.csv`
