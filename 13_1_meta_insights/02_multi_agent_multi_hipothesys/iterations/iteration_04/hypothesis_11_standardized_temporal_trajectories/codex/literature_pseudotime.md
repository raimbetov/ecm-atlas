# Pseudo-time Trajectory Inference – Key References and Best Practices

## Core Benchmarking & Best Practice Papers

1. **Saelens et al., 2019 – "A comparison of single-cell trajectory inference methods" (Nature Biotechnology)**  
   - Benchmark of 45 trajectory algorithms across >100 synthetic and real datasets.  
   - Key guidance: (i) run multiple methods and assess agreement; (ii) evaluate ordering stability under subsampling; (iii) integrate prior knowledge (landmarks) only after data-driven QC.  
   - Recommends diffusion-map kernels or graph-based approaches for robustness, with Slingshot among the top performers when lineages are tree-like.

2. **Haghverdi et al., 2016 – "Diffusion pseudotime robustly reconstructs lineage branching" (Nature Methods)**  
   - Introduced Diffusion Pseudotime (DPT); highlights the importance of density-aware kernels and redundancy in local neighborhoods to avoid noise-driven shortcuts.  
   - Best practice: choose sigma automatically (e.g., BGH heuristic), perform reachability checks, and validate pseudotime by correlating with known progression markers.

3. **Street et al., 2018 – "Slingshot: cell lineage and pseudotime inference for single-cell transcriptomics" (BMC Genomics)**  
   - Combines clustering, minimum spanning tree (MST), and principal curves to recover branching trajectories.  
   - Recommendations: use cluster-aware smoothing to prevent trajectories running through sparse regions; apply bootstrap resampling to quantify lineage confidence; derive biological interpretation through gene smoothers along pseudotime.

4. **Bergen et al., 2020 – "Generalizing RNA velocity to transient cell states through dynamical modeling" (Nature Biotechnology)**  
   - Extends velocity-based ordering (scVelo) with dynamical models.  
   - Best practice: incorporate spliced/unspliced kinetics to estimate directionality, combine velocity field integration with pseudotime, and verify alignment with diffusion-based embeddings to prevent spurious flow.

5. **Campbell & Yau, 2018 – "A probabilistic model for pseudotime inference" (Bioinformatics)**  
   - Presents probabilistic GP-LVM approach (pseudotime via latent functions).  
   - Highlights need for uncertainty quantification, Bayesian model averaging, and explicit branch probability estimation; encourages reporting posterior intervals for rankings instead of point estimates only.

## Practical Guidelines Extracted

- **Algorithm Selection:** Start with diffusion map or graph-based methods (DPT, Slingshot) for stability; augment with PCA or velocity-based orderings for triangulation.  
- **Quality Control:**  
  - Remove low-coverage entities prior to building neighborhood graphs.  
  - Assess robustness via subsampling, bootstrap, or noise injection (Kendall τ >= 0.8 is a common robustness target).  
- **Reference-based Validation:** Use known age markers, experimental time points, or RNA velocity directions to sanity-check monotonicity.  
- **Branch Awareness:** Slingshot and GPfates support explicit branching; diffusion-based methods require branch detection via graph clustering or community detection before ordering each branch separately.  
- **Uncertainty Reporting:** Adopt credible intervals or consensus ranks (as in Campbell & Yau) to communicate ambiguity—critical when applying pseudo-time to downstream temporal models like LSTMs.  
- **Integration with Longitudinal Data:** Literature stresses validating pseudo-time orderings against true time courses whenever available (e.g., Bergen et al.; Saelens et al. recommend cross-study validation on real temporal cohorts).

These sources collectively define current best practices for trajectory inference and will guide standardized pseudo-time construction for ECM aging analyses.
