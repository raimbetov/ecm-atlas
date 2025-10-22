# External Dataset Leads (Agent: codex)

## Ovary Aging
- **GSE276193 (Homo sapiens)** — Single-cell + spatial atlas on iron overload-induced senescence in ovarian endometriosis with age-comparison; contains follicular fluid immune/fibroblast/ECM trajectories relevant to ovarian aging (esummary via NCBI E-utilities, PDAT 2025-08-27).
- **Biobanking note:** Nanomaterials 2022 study (PMID:35159690) released decellularized ovarian ECM scaffolds; raw proteomics currently local but aligns with structural phenotypes for validation.

## Heart Aging / Mechanical Stress
- **GSE305089 (Mus musculus)** — Transcriptomic profiling of aged cardiac fibroblasts undergoing direct reprogramming; highlights senescence and ECM remodeling under mechanical stress contexts (PDAT 2025-10-11).
- **GSE267468 (Mus musculus)** — Companion RNA-seq/ATAC-seq dataset testing Nr4a3 suppression to overcome aging fibroblast stiffness; captures Hippo/YAP-cofactor shifts and fibrosis programs.
- **E-PROT-81 (Heart proteome across species)** — Identified via OmicsDI search (`omicsdi ws dataset search q="ovary aging"`); includes age-stratified cardiac proteomes with ECM content and mechanical load metadata.

## Gaps & Actions
- No publicly indexed multi-tissue human dataset with paired ovary-heart ECM readouts; consider constructing cross-study meta-cohort by aligning GSE276193 fibroblast signatures with heart fibroblast states from GSE305089.
- For estrogen-specific validation, prioritize datasets with endocrine metadata (GSE8157 request outstanding; manual follow-up needed if access restrictions persist).
