# H15 — Ovary & Heart Transition Biology (Agent: codex)

## 1. Approach & Tooling
- **Dataset:** `/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv` (3715 rows, 17 tissues).
- **Pseudo-time anchor:** Imported ordering/scores from Iteration 03 Transformer analysis (`../../../iteration_03/hypothesis_09_temporal_rnn_trajectories/codex/analysis_metadata_codex.json`).
- **Analytical stack:**
  - Central-difference tissue gradients per protein (`outputs_codex/gradient_matrix_codex.csv`).
  - PyTorch autoencoder (8 latent dims) + reconstruction error profiling for protein trajectories (`outputs_codex/autoencoder_gene_latents_codex.csv`).
  - Spectral clustering of gradient signatures (`outputs_codex/spectral_clusters_codex.csv`).
  - NetworkX correlation graph (|r|≥0.6) across estrogen, YAP/TAZ, and crosslinker panels (`outputs_codex/pathway_network_metrics_codex.csv`).
  - Pathway-specific gradient tables (`outputs_codex/estrogen_gradient_profile_codex.csv`, `outputs_codex/yap_taz_gradient_profile_codex.csv`, `outputs_codex/metabolic_overlap_codex.csv`).
  - Literature synthesis (`literature_ovary_heart.md`) and dataset scouting (`dataset_search_codex.md`).

## 2. Literature Pulse (why ovary & heart?)
Key findings from PubMed/E-utility searches:
- **Ovary:** Hormonal cycling drives fibroblast-led ECM remodeling and fibrosis (Cell 2024, PMID:38325365); stromal ECM, LOX/TGM enzymes, and mitochondrial stress co-regulate menopausal transitions (J Assist Reprod Genet 2022, PMID:35352316; Biol Reprod 2022, PMID:34982142; Nanomaterials 2022, PMID:35159690).
- **Heart:** Mechanical overload activates YAP/TAZ and fibrosis circuits (JCI 2024, PMID:39352768; Nat Commun 2024, PMID:38937456; Curr Cardiol Rev 2022, PMID:35379136; Annu Rev Physiol 2020, PMID:32040933). LOXL isoform activity and verteporfin-sensitive YAP/TAZ signaling link stiffness to fibrosis (Circulation Res 2018, PMID:29599116; Bone Reports 2021, PMID:34562507).

## 3. Gradient Diagnostics
### Ovary Cortex (Hormonal transition)
- 12 proteins assign their **maximum gradient** to `Ovary_Cortex` (`outputs_codex/ovary_specific_gradients_codex.csv`). Top hits: THBS4, COL11A1, EFEMP2, LAMC3, LTBP4, SPON1.
- Estrogen-responsive ECM targets show strong slope at ovary: PLOD1 (|∇|=1.30), PLOD3 (0.75), POSTN (0.55), TNC (0.50), COL1A2 (0.33) with suppressed steady-state ΔZ (mean −1.19), consistent with estrogen-withdrawal tightening collagen crosslinking.
- Autoencoder latent space flags ovarian cartilage-like proteins (SMOC2, ASPN, CILP, PCOLCE, FBN2) with extreme loadings—matching literature on follicular basement membrane fragility during menopause.
- Network betweenness peaks for TIMP3, FN1, CYR61 (0.64-0.75), positioning ovary-specific inhibitors/cytokines as control hubs within estrogen-YAP cross-talk loops.

### Heart Native Tissue (Mechanical/YAP)
- 12 proteins peak at `Heart_Native_Tissue` (`outputs_codex/heart_specific_gradients_codex.csv`). Leading signatures: PRG3, ELANE, MPI, TGM3, COL5A3, PAPLN, THSD4.
- YAP/TAZ panel exhibits negative gradients (VCAN −0.46, TNC −0.32, COL6A3 −0.32, COL1A1 −0.26), paired with mildly negative ΔZ (mean −0.23), implying rising mechano-inhibition just before the heart transition point—aligning with verteporfin-sensitive attenuation seen in literature.
- Network centrality ranks COL6A1/2/3 (degree ≥8) and VCAN/TNC (high closeness), supporting a load-bearing proteoglycan scaffold sensing mechanical stress.
- Spectral clusters place heart-transition genes in a shared module with TGM3 and COL5A isoforms, pointing to crosslinking and fibrillar reinforcement as cardiomyocyte stress responses.

## 4. Shared Metabolic Mechanism? (H15.3)
- Available ECM-centric markers offer limited overlap (4 gradient pairs with non-NaN values: COL6A1-3, TGM2). Spearman ρ=0.40 (gradients) but **−0.99 across the sparse ΔZ values** (3 pairs), reflecting divergent directionality (ovary upshifts collagen VI/tissue transglutaminase, heart downshifts).
- Global gradient correlation across 87 genes yields ρ=−0.11, reinforcing **largely independent transitions**.
- Autoencoder errors concentrate on glycoproteins (ANGPTL7, FGG) rather than mitochondrial markers; metabolic convergence appears indirect at best (through ECM glycoproteins that modulate energy homeostasis).

## 5. Interpretation — Why Transformer Attention Peaks Here
1. **Ovary cortex as hormonal switchboard:** Estrogen-responsive crosslinkers (PLOD1/3, POSTN, THBS4) undergo the steepest trajectory turns right at `Ovary_Cortex`, consistent with menopause-triggered stiffening and immune recruitment. Transformer attention likely seized on this rapid curvature because ovary ECM derivatives govern downstream endocrine resilience.
2. **Heart native tissue as mechanical threshold:** Proteoglycan and crosslinking enzymes (COL6A3, VCAN, TGM3) show synchronized inflection, echoing YAP/TAZ literature on mechanotransduction. Attention concentration reflects the heart crossing from adaptable to maladaptive stiffness once these markers invert.
3. **Minimal shared metabolic driver:** Cross-tissue correlation is weak; the ovary spike is dominated by hormonal ECM rewiring, whereas the heart spike is dominated by mechanical mechano-sensors. The model probably treats them as **two independent tipping points** that nevertheless co-define the pseudo-time curvature because both feed into systemic vascular aging.

## 6. Dataset Prospects
- **Ovary:** GSE276193 (single-cell, aging human follicles) offers fibroblast/immune panels to validate collagen-modifying genes in hormone-depleted settings.
- **Heart:** GSE305089 + GSE267468 (aged cardiac fibroblast reprogramming) provide senescence/YAP modulation transcriptomes for cross-validation of COL6/VCAN/TGM modules.
- **Proteomics:** OmicsDI E-PROT-81 adds multi-species cardiac ECM quantification for mechanical benchmarking.

## 7. Next Steps
1. Run hormonal perturbation simulations: overlay estrogen replacement signatures onto PLOD/THBS trajectories to test reversibility.
2. Integrate GSE276193 and GSE305089 via latent space alignment (e.g., scVI) to quantify shared fibroblast states vs tissue-specific modules.
3. Derive intervention timing rules: Map Transformer attention peaks onto clinical biomarkers (AMH, NT-proBNP) for forecasting menopause vs cardiac fibrosis windows.
