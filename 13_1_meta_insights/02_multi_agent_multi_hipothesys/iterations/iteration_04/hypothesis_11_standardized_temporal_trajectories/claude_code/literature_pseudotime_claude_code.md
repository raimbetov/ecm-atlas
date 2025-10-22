# Literature Review: Pseudo-Time Trajectory Inference Methods

**Thesis:** Saelens et al. (2019) benchmarked 45 trajectory inference methods on 110 real and 229 synthetic datasets, finding that Slingshot, TSCAN, and Monocle DDRTree outperform others, but method choice should depend on dataset dimensions and trajectory topology, with recent 2024 advances incorporating Bayesian hierarchical models and variational autoencoders for improved accuracy and uncertainty quantification.

**Overview:** This review synthesizes current best practices for pseudo-time construction in aging proteomics, covering method benchmarking (§1.0), top-performing algorithms (§2.0), longitudinal validation datasets (§3.0), and recommendations for ECM aging analysis (§4.0).

---

## 1.0 Trajectory Inference Benchmarking: State of the Art

¶1 **Ordering:** Landmark study → Evaluation criteria → Key findings → Available resources

¶2 **Saelens et al. (2019) - Nature Biotechnology:**
- **Study scope:** Benchmarked 45 trajectory inference methods on 110 real + 229 synthetic single-cell datasets
- **Evaluation metrics:**
  - Accuracy: Cellular ordering correctness, topology reconstruction
  - Scalability: Performance on varying numbers of cells and features
  - Stability: Robustness under subsampling
  - Usability: Software availability, documentation quality
- **Key conclusion:** No single method dominates all scenarios; choice depends on **trajectory topology** (linear, bifurcating, tree, graph) and **dataset size**

¶3 **Available resources:**
- **Benchmark website:** https://benchmark.dynverse.org
- **dynverse R package collection:** Tools for applying and comparing trajectory inference methods
- **GitHub:** https://github.com/dynverse/dynbenchmark (reproducible pipeline)

¶4 **2024 advances:**
- **Joint trajectory analysis (PNAS 2024):** Bayesian hierarchical models + variational autoencoders for multi-dataset integration
- **Sceptic (2024):** Supervised pseudo-time via support vector machines, extends to scATAC-seq and imaging data
- **Ensemble approaches (BMC Bioinformatics 2023):** Combining multiple methods for robust pseudo-time estimation

---

## 2.0 Top-Performing Methods (from Saelens et al. 2019)

¶1 **Ordering:** Best performers → Method characteristics → Strengths/weaknesses

### 2.1 Slingshot (Winner: Linear and Bifurcating Trajectories)

**Algorithm:**
- Dimensionality reduction (PCA/diffusion maps/UMAP)
- Cluster identification
- Minimum spanning tree construction
- Smooth curve fitting through clusters

**Strengths:**
- Handles branching trajectories (e.g., disc vs lung endpoints in aging)
- Fast (scales to 100k+ cells)
- Statistically principled (smooth splines)
- Available in Bioconductor

**Weaknesses:**
- Requires pre-clustering (user-defined parameter)
- May miss complex graph structures

**Relevance to H11:** Slingshot can detect if ECM aging follows **multiple trajectories** (e.g., metabolic vs mechanical aging paths), not assumed by velocity/PCA methods.

---

### 2.2 Monocle DDRTree (Winner: Complex Tree Structures)

**Algorithm:**
- Discriminative dimensionality reduction
- Tree reconstruction via minimum spanning tree
- Branching point identification

**Strengths:**
- Discovers bifurcations without prior knowledge
- Well-validated in development biology

**Weaknesses:**
- Slower than Slingshot
- Requires more cells for accurate branching detection

**Relevance to H11:** If tissues follow **divergent aging programs** (e.g., brain vs muscle), DDRTree may reveal branching points.

---

### 2.3 Diffusion Pseudotime (DPT) (Winner: Noisy Data)

**Algorithm:**
- Diffusion map dimensionality reduction (preserves manifold structure)
- Random walk distance from root cell
- Robust to dropouts and noise

**Strengths:**
- Excellent noise tolerance (critical for proteomics with missing values)
- Preserves nonlinear manifold geometry
- No branching assumptions

**Weaknesses:**
- Requires specifying root (starting point)
- Computationally expensive for large datasets

**Relevance to H11:** Proteomic data has **50-80% missing values** (NaNs); diffusion maps are specifically designed for this challenge.

---

### 2.4 TSCAN (Winner: Simple Linear Trajectories)

**Algorithm:**
- Model-based clustering
- Minimum spanning tree ordering

**Strengths:**
- Simple, fast
- Good baseline for linear aging

**Weaknesses:**
- Cannot handle complex topologies

**Relevance to H11:** Useful baseline; if TSCAN performs poorly, aging is likely **nonlinear**.

---

## 3.0 Longitudinal Aging Proteomics Datasets (2023-2024)

¶1 **Ordering:** Recently published cohorts → Dataset characteristics → Accessibility

### 3.1 Nature Metabolism 2025 - Longitudinal Serum Proteome

**Dataset:**
- **Cohort:** 3,796 middle-aged/elderly adults
- **Samples:** 7,565 serum samples across 3 timepoints
- **Follow-up:** 9 years
- **Proteins:** 86 aging-related proteins identified
- **Outcomes:** 32 clinical traits, 14 age-related diseases

**Key findings:**
- Developed PHAS (Proteomic Healthy Aging Score) using 22 proteins
- Predicted cardiometabolic disease incidence

**Access:** Likely in PRIDE/ProteomeXchange (search: "longitudinal serum proteome 9 years")

---

### 3.2 Nature Medicine 2024 - UK Biobank Proteomic Clock

**Dataset:**
- **Cohort:** 45,441 participants
- **Proteins:** 2,897 plasma proteins
- **Age prediction accuracy:** Pearson r=0.94 (204 proteins)
- **Associations:** 18 chronic diseases, multimorbidity, mortality

**Key findings:**
- Cross-sectional with age range 40-70 years (pseudo-longitudinal via age proxy)
- 204 proteins form proteomic age clock

**Access:** UK Biobank data application required (https://www.ukbiobank.ac.uk/)

---

### 3.3 Cell 2025 - Comprehensive 50-Year Lifespan Atlas

**Dataset:**
- **Samples:** 516 samples from 13 human tissues
- **Age range:** 50-year lifespan (likely ages 20-70)
- **Proteins:** Comprehensive proteome profiling
- **Key finding:** Aging inflection at ~age 50; blood vessels age earliest

**Access:** Likely published with manuscript (search: "516 samples 13 tissues 50 year lifespan")

---

### 3.4 BLSA (Baltimore Longitudinal Study of Aging)

**Dataset:**
- **Cohort:** Longest-running longitudinal aging study (since 1958)
- **Proteomics:** TMT-based mass spectrometry, SOMAscan
- **Tissues:** Plasma, skeletal muscle, brain (DLPFC)
- **Samples:** 1,060+ plasma samples analyzed

**Access:**
- Data use application: https://www.blsa.nih.gov/
- Synapse portal: https://adknowledgeportal.synapse.org/Explore/Studies/DetailsPage?Study=syn3606086

**Challenges:**
- Requires pre-analysis plan submission
- Access approval process (2-3 months)

---

## 4.0 Recommendations for H11 Analysis

¶1 **Ordering:** Method selection → Validation strategy → Implementation priority

### 4.1 Methods to Test (Ranked by Priority)

| Rank | Method | Rationale | Implementation Difficulty |
|------|--------|-----------|---------------------------|
| 1 | **Tissue Velocity (H03)** | Current best performer (R²=0.81) | LOW (already implemented) |
| 2 | **Diffusion Pseudotime (DPT)** | Robust to missing data, nonlinear | MEDIUM (R destiny package) |
| 3 | **Slingshot** | Detects branching (metabolic vs mechanical aging) | MEDIUM (Bioconductor) |
| 4 | **PCA-based (Codex)** | Current worst performer (R²=0.011) | LOW (already implemented) |
| 5 | **Autoencoder Latent Traversal (H04)** | Nonlinear dimensionality reduction | MEDIUM (reuse H04 model) |

### 4.2 Validation Strategy

**Primary validation (no external data):**
1. **Internal consistency:** Do all methods identify same critical transitions (Ovary, Heart)?
2. **Granger causality stability:** Jaccard similarity of causal edges across methods
3. **Sensitivity analysis:** Kendall's τ under tissue/protein subsampling

**Secondary validation (if longitudinal data accessible):**
1. **Correlation with real time:** Spearman ρ between pseudo-time and actual age
2. **Prospective prediction:** Train LSTM on cross-sectional, test on longitudinal

**Gold standard (requires dataset access):**
- Apply for BLSA plasma proteomics data (TMT or SOMAscan)
- Test: Do proteins with high pseudo-time gradients also change fastest in real longitudinal follow-up?

### 4.3 Expected Outcomes & Interpretations

**Scenario A: Velocity method wins consistently**
- **Implication:** Tissue aging velocity (H03) is the optimal pseudo-time proxy for ECM aging
- **Action:** Standardize velocity-based ordering for all future temporal analyses (H12-H15)

**Scenario B: Diffusion pseudotime or Slingshot wins**
- **Implication:** Nonlinear manifold structure or branching trajectories exist
- **Action:** Re-run H09 LSTM with winning method; compare to Claude R²=0.81 baseline
- **Update:** Document new standard method in ADVANCED_ML_REQUIREMENTS.md

**Scenario C: All methods fail external validation (if longitudinal data obtained)**
- **Implication:** Cross-sectional pseudo-time is fundamentally unreliable for temporal modeling
- **Action:** Halt temporal analyses; require real longitudinal cohorts for H12+
- **Recommendation:** Focus on cross-sectional biomarker discovery (H01-H08) until longitudinal ECM data available

**Scenario D: Slingshot detects branching**
- **Implication:** ECM aging follows **multiple trajectories** (e.g., brain/kidney metabolic path vs muscle/lung mechanical path)
- **Action:** Develop multi-trajectory LSTM models; abandon single-timeline assumption
- **Update:** Revise aging model from linear to branching in all documentation

### 4.4 Implementation Notes

**For diffusion pseudotime (DPT):**
```R
library(destiny)

# Load ECM data
tissue_matrix <- read.csv("merged_ecm_aging_zscore.csv") %>%
  pivot_wider(names_from = Gene_Symbol, values_from = Zscore_Delta) %>%
  column_to_rownames("Tissue")

# Compute diffusion map
dm <- DiffusionMap(data = tissue_matrix, k = 10)  # k = nearest neighbors

# Extract pseudo-time from DC1 (first diffusion component)
pseudo_time <- rank(dm@eigenvectors[, 1])

# Plot
plot(dm, col_by = "Tissue", pch = 20)
```

**For Slingshot:**
```R
library(slingshot)
library(SingleCellExperiment)

# Create SingleCellExperiment object
sce <- SingleCellExperiment(assays = list(counts = t(tissue_matrix)))

# Run PCA (or use diffusion map)
reducedDims(sce) <- list(PCA = prcomp(t(tissue_matrix))$x[, 1:10])

# Infer lineages
sce <- slingshot(sce, reducedDim = 'PCA')

# Extract pseudo-time
pseudo_time <- slingPseudotime(sce)[, 1]  # First lineage
n_lineages <- ncol(slingPseudotime(sce))  # Check if branching detected
```

**For autoencoder latent traversal:**
```python
# Reuse H04 autoencoder
import torch
from h04_autoencoder import load_model  # Hypothetical

autoencoder = load_model("../hypothesis_04_*/autoencoder_weights.pth")

# Encode tissues
tissue_tensor = torch.tensor(tissue_matrix.values, dtype=torch.float32)
latent_coords = autoencoder.encoder(tissue_tensor).detach().numpy()

# Order by latent factor with highest correlation to velocity (or PC1)
from scipy.stats import spearmanr
correlations = [spearmanr(latent_coords[:, i], velocity_ranking)[0]
                for i in range(latent_coords.shape[1])]
best_factor = np.argmax(np.abs(correlations))
pseudo_time = rank(latent_coords[:, best_factor])
```

---

## 5.0 Critical Questions from Benchmarking Literature

¶1 **Questions ECM-Atlas must answer:**

1. **Is aging topology linear, bifurcating, or tree-like?**
   - **Test:** Compare Slingshot lineage count (1 = linear, 2+ = branching)
   - **Hypothesis:** If >1 lineage detected → brain/kidney vs muscle/lung diverge

2. **Do pseudo-time methods agree on ordering or produce conflicting timelines?**
   - **Test:** Kendall's τ between all method pairs
   - **Target:** τ>0.70 for convergent validity; τ<0.30 indicates method artifacts

3. **Which method is most stable under data perturbations?**
   - **Test:** Subsample 80% tissues 100 times, measure ordering stability
   - **Winner:** Method with highest mean τ across subsamples

4. **Does pseudo-time recapitulate real time (if longitudinal data available)?**
   - **Test:** Spearman correlation between pseudo-time and participant age
   - **Gold standard:** ρ>0.70 required for clinical validity

5. **Root cause of H09 disagreement: Claude (R²=0.81) vs Codex (R²=0.011)?**
   - **Hypothesis A:** Velocity ordering captures true biological time; PCA is noisy
   - **Hypothesis B:** Both are wrong; real aging is nonlinear (diffusion/Slingshot needed)
   - **Hypothesis C:** Short sequence length (17 tissues) makes LSTM unstable; any ordering works if validated externally

---

## 6.0 Key Citations

¶1 **Primary references:**

1. **Saelens, W., Cannoodt, R., Todorov, H. et al.** (2019). A comparison of single-cell trajectory inference methods. *Nature Biotechnology* 37, 547–554. https://doi.org/10.1038/s41587-019-0071-9
   - **Relevance:** Gold standard benchmarking study; 45 methods compared
   - **Key takeaway:** Slingshot, TSCAN, Monocle DDRTree are top performers

2. **Street, K. et al.** (2018). Slingshot: cell lineage and pseudotime inference for single-cell transcriptomics. *BMC Genomics* 19, 477.
   - **Relevance:** Original Slingshot algorithm paper
   - **Implementation:** https://bioconductor.org/packages/slingshot/

3. **Haghverdi, L., Büttner, M., Wolf, F.A. et al.** (2016). Diffusion pseudotime robustly reconstructs lineage branching. *Nature Methods* 13, 845–848.
   - **Relevance:** Diffusion maps for noisy single-cell data
   - **Implementation:** R package `destiny`

4. **Longitudinal serum proteome study** (2025). Nature Metabolism. (3,796 participants, 9-year follow-up)
   - **Relevance:** Gold standard for validating pseudo-time against real time
   - **Dataset:** Likely in PRIDE (search in progress)

5. **UK Biobank proteomic clock** (2024). Nature Medicine. (45,441 participants, 2,897 proteins)
   - **Relevance:** Cross-sectional age prediction (r=0.94) as benchmark
   - **Limitation:** Not true longitudinal (single timepoint)

6. **Comprehensive 50-year lifespan atlas** (2025). Cell. (516 samples, 13 tissues)
   - **Relevance:** Aging inflection at age 50; tissue-specific clocks
   - **Key finding:** Blood vessels age earliest (similar to ECM-Atlas lung finding)

---

## 7.0 Implementation Timeline for H11

¶1 **Phased approach:**

**Phase 1 (Current session):**
- ✅ Literature review complete
- 🔄 Implement 5 pseudo-time methods (in progress)
- 🔄 Benchmark LSTM performance for all methods

**Phase 2 (After LSTM benchmarking):**
- Sensitivity analysis (Kendall's τ under perturbations)
- Critical transitions consistency (attention weights correlation)
- Granger causality stability (Jaccard similarity)

**Phase 3 (External validation - if datasets accessible):**
- Download BLSA or Nature Metabolism 2025 longitudinal data
- Correlate pseudo-time with real age
- Prospective LSTM prediction on longitudinal cohort

**Phase 4 (Synthesis & Recommendation):**
- Identify winning method
- Root cause analysis of H09 disagreement
- Standardize method for Iterations 05-07

---

**Created:** 2025-10-21
**Agent:** claude_code
**Status:** Phase 1 complete; Phase 2 in progress
**Next steps:** Implement diffusion pseudotime and Slingshot trajectory inference
