# Literature Review: Network Centrality Metrics in Biological Networks

**Thesis:** No single centrality metric universally predicts protein essentiality; degree and betweenness correlate with lethality (ρ≈0.64 AUC), eigenvector excels for regulatory networks, and PAI-1/SERPINE1 knockout shows protective aging phenotype validating functional importance.

## Key Findings from Literature

### 1.0 Centrality Metric Performance

¶1 **Ordering:** General findings → Specific metric performance

#### 1.1 Comparative Studies on Essential Protein Detection

**Systematic Survey (BMC Systems Biology 2018):**
- Compared 27 centrality measures on yeast protein-protein interaction networks (PPINs)
- **Degree centrality:** Best predictor (AUC=0.64) for protein essentiality
- **Betweenness centrality:** Good predictor of lethality (centrality-lethality rule validated)
- **Eigenvector centrality:** Achieved 95% accuracy for top 20 prostate cancer genes
- **Finding:** Performance depends on network topology and biological question

#### 1.2 Context-Dependent Performance

**Binding Residue Identification (Frontiers 2021):**
- **Precision leaders:** Closeness, Degree, PageRank
- **Sensitivity leaders:** Betweenness, Eigenvector, Residue Centrality
- **Implication:** Different metrics excel at different validation tasks

**Nitrogen Environment Adaptation (ScienceDirect 2021):**
- **Betweenness centrality** best reflected relevance in information flow
- Path-based metrics superior for detecting phenotypic relevance during adaptation

### 2.0 Centrality-Lethality Rule Validation

¶1 **Ordering:** Rule definition → Multi-species validation → Limitations

#### 2.1 Core Hypothesis

**Jeong et al. (2001) Nature:** Proteins with higher centrality more likely to produce lethal phenotypes on knockout vs. lower centrality nodes. Essential genes identified by single gene knockout, RNA interference, conditional knockout.

#### 2.2 Multi-Species Validation

**Extended Studies (PLOS One 2015, IEEE 2012):**
- Validated across 20 organisms: S. cerevisiae, H. sapiens, M. musculus, D. melanogaster
- **Degree centrality:** Significantly higher for essential vs. nonessential proteins
- **Betweenness centrality:** Validated in protein subcellular localization interaction networks
- **Party/date hub distinction:** Essential party hubs (constant interactions) vs. date hubs (transient)

#### 2.3 Limitations

- Time consuming and inefficient for experimental validation
- Only applicable to a few model species
- Network incompleteness affects centrality calculations

### 3.0 Consensus Recommendations

¶1 **Key finding:** Multiple metrics required for comprehensive centrality assessment

**Multi-Metric Approach:**
1. Use ≥3 complementary metrics (e.g., degree + betweenness + eigenvector)
2. Density of Maximum Neighbourhood Component (DMNC) as complement
3. Subgraph centrality discriminates nodes better than classical measures for subgraph participation
4. Context-specific selection: structural proteins (betweenness), regulatory proteins (eigenvector)

### 4.0 Experimental Validation: Serpin Knockouts

¶1 **Ordering:** SERPINE1 (strong phenotype) → SERPINC1 (moderate phenotype)

#### 4.1 SERPINE1/PAI-1 Knockout - PROTECTIVE ANTI-AGING PHENOTYPE

**Human Studies (Science Advances 2017):**
- **Longevity:** +7 years lifespan (82±10 vs 75±12 years in controls)
- **Telomere length:** +10% longer leukocyte telomere length (LTL)
- **Metabolic health:** -28% fasting insulin, lower diabetes prevalence
- **Cardiovascular:** Better pulse wave velocity, intima-media thickness, LV relaxation

**Mouse Studies:**
- PAI-1 deficiency prolongs lifespan in Klotho-deficient accelerated aging models
- **Mechanism:** PAI-1 induces p53-p21-Rb senescence pathway

**CRITICAL INTERPRETATION:** SERPINE1 knockout has BENEFICIAL phenotype → Suggests PAI-1 is DOWNSTREAM effector of aging, not central regulatory hub (if central regulator, knockout would be lethal/harmful)

#### 4.2 SERPINC1/Antithrombin Mutation/Deficiency

**Clinical Studies (Thrombosis 2017-2023):**
- **Null mutations:** 66.7% 5-year VTE-free survival (severe thrombotic phenotype)
- **Missense mutations:** 92.0% 5-year VTE-free survival (moderate phenotype)
- **Cell validation:** HEK293T transfection shows reduced AT protein expression in mutants

**CRITICAL INTERPRETATION:** SERPINC1 loss-of-function has HARMFUL phenotype (thrombosis) → Suggests functional importance, but not lethal (compatible with life)

### 5.0 Synthesis for ECM Aging Networks

¶1 **Predictions for serpin centrality analysis:**

**Expected Outcome Based on Literature:**
1. **Degree/Betweenness:** Should correlate with knockout impact (ρ≈0.60-0.70)
2. **Eigenvector:** May identify SERPINE1/SERPINC1 as hubs due to regulatory role
3. **SERPINE1 paradox:** High centrality BUT beneficial knockout → Needs resolution
4. **Best validation:** In silico knockout Δ connectivity vs. experimental severity

**Methodological Recommendation:**
- Compute ≥6 metrics (Betweenness, Eigenvector, Degree, Closeness, PageRank, Katz, Subgraph)
- Validate with in silico knockout simulations
- Cross-reference with experimental phenotype severity
- Use ensemble consensus for robust hub identification

---

## References

1. BMC Systems Biology (2018): "Systematic survey of centrality measures for protein-protein interaction networks" - 27 metrics comparison
2. Frontiers (2021): "Centrality Measures in Residue Interaction Networks to Highlight Amino Acids in Protein–Protein Binding"
3. PLOS One (2015): "Rechecking the Centrality-Lethality Rule in Protein Subcellular Localization Interaction Networks"
4. Science Advances (2017): "A null mutation in SERPINE1 protects against biological aging in humans" - **KEY FINDING**
5. Aging Cell (2017): "Serpine1 induces alveolar type II cell senescence through p53-p21-Rb pathway"
6. ScienceDirect (2021): "Path-based centrality measures revealed proteins with phenotypic relevance during nitrogen adaptation"

**Created:** 2025-10-21
**Agent:** claude_code
