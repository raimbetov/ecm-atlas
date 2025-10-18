# Multi-Agent Task: Entropy Analysis After Batch Correction V2

## Thesis
Validate and update entropy-based aging theory (from agent_09) using batch-corrected ECM dataset (merged_ecm_aging_zscore.csv V2, updated Oct 18 2025), proving or refining the DEATh theorem predictions about deterministic matrix stiffening vs dysregulated cellular chaos.

## Task Overview
Re-analyze entropy metrics (Shannon entropy, variance CV, predictability, transitions) on the NEW batch-corrected dataset to verify if previous findings hold after removing technical batch effects. Compare results with original agent_09_entropy analysis to identify: (1) artifacts vs true biology, (2) strengthened vs weakened entropy patterns, (3) new insights from cleaner data. Each agent must create comprehensive markdown reports with generated visualizations, statistical tests, and philosophical synthesis.

## Success Criteria

### 1. Data Processing
- ✅ Load NEW batch-corrected dataset: `/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv` (3.1 MB, updated Oct 18)
- ✅ Calculate all 4 entropy metrics: Shannon entropy, Variance CV, Predictability score, Entropy transitions
- ✅ Include sufficient proteins (≥400 with multi-study data) for statistical power
- ✅ Validate data quality: no NaN inflation, correct z-score distributions

### 2. Entropy Analysis
- ✅ Perform hierarchical clustering on entropy profiles (4-6 clusters)
- ✅ Identify high/low entropy proteins, high/low predictability proteins, high transition proteins
- ✅ Test DEATh theorem predictions:
  - Structural vs regulatory protein entropy differences
  - Collagen predictability scores (should be >0.74 if theory holds)
  - Entropy transition proteins marking regime shifts

### 3. Comparison with Original Analysis
- ✅ Load original entropy results from: `/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/agent_09_entropy/`
- ✅ Compare key metrics before/after batch correction:
  - Protein entropy rankings (correlation coefficient)
  - Cluster assignments (stability analysis)
  - DEATh theorem p-values (stronger/weaker significance?)
- ✅ Identify artifacts removed by batch correction
- ✅ Identify NEW biological insights enabled by cleaner data

### 4. Visualization Requirements
- ✅ Generate publication-quality figures (matplotlib/seaborn, 300 DPI):
  - Entropy distributions (4-panel histogram)
  - Clustering dendrogram + heatmap
  - Entropy-Predictability 2D space (colored by clusters)
  - Before/After comparison plots (entropy rankings, cluster stability)
  - Transition proteins volcano plot or bar chart
- ✅ All plots saved in agent workspace folder

### 5. Documentation (CRITICAL)
- ✅ Create report using **KNOWLEDGE FRAMEWORK** from: `/Users/Kravtsovd/projects/ecm-atlas/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md`
  - Thesis (1 sentence) → Overview (1 paragraph) → Mermaid diagrams (TD for structure, LR for process)
  - MECE sections (numbered 1.0, 2.0, 3.0...)
  - Paragraphs numbered (¶1, ¶2, ¶3)
  - Minimal tokens, maximum clarity
- ✅ Include philosophical synthesis: What does batch-corrected entropy tell us about aging?
- ✅ Statistical validation: p-values, confidence intervals, effect sizes
- ✅ Therapeutic implications: Updated targets based on new entropy patterns

### 6. Artifact Placement (MANDATORY)
- ✅ ALL outputs (CSV, PNG, MD, Python scripts) MUST be in agent's workspace folder
- ✅ NO files in external directories (data_processed/, output/, etc.)
- ✅ Folder structure:
  ```
  [agent_workspace]/
  ├── 01_plan_[agent_name].md          # Your planning document
  ├── 90_results_[agent_name].md       # Final report (Knowledge Framework format)
  ├── entropy_metrics_v2.csv           # Updated entropy calculations
  ├── entropy_distributions_v2.png     # Visualization 1
  ├── entropy_clustering_v2.png        # Visualization 2
  ├── entropy_comparison_v1_v2.png     # Before/after comparison
  ├── analysis_script.py               # Reproducible Python code
  └── execution.log                    # Analysis log
  ```

## Reference Materials

### Required Reading
1. **Previous entropy analysis:** `/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/agent_09_entropy/agent_09_entropy_clustering.md`
   - 532 proteins, 4 entropy classes identified
   - Collagens: predictability=0.764 (deterministic aging)
   - Top transition proteins: FCN2, FGL1, COL10A1
2. **Knowledge Framework:** `/Users/Kravtsovd/projects/ecm-atlas/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md`
3. **Multi-agent orchestrator:** `/Users/Kravtsovd/projects/chrome-extension-tcs/algorithms/product_div/Multi_agent_framework/00_MULTI_AGENT_ORCHESTRATOR.md`

### Data Sources
- **NEW batch-corrected:** `/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`
- **OLD pre-correction:** `/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore_OLD_BACKUP.csv`

## Agent-Specific Instructions

### For Claude Code Agents
- Use pandas, numpy, scipy, matplotlib, seaborn for analysis
- Write modular, well-documented Python code
- Include statistical tests (Mann-Whitney U, Spearman correlation)
- Generate high-quality matplotlib figures with proper labels

### For Codex Agents
- Same technical requirements as Claude Code
- Focus on code efficiency and reproducibility
- Ensure all visualizations are publication-ready
- Document statistical assumptions and limitations

## Evaluation Criteria (Self-Assessment)

Rate yourself on each Success Criterion (1-6):
- ✅ **PASS:** Criterion fully met, evidence provided
- ⚠️ **PARTIAL:** Criterion mostly met, minor gaps
- ❌ **FAIL:** Criterion not met, major issues

**Winner = agent with most ✅ criteria + best philosophical insights + cleanest code**

## Key Questions to Answer

1. **Are entropy patterns ARTIFACTS or BIOLOGY?**
   - Which high-entropy proteins dropped in entropy after batch correction?
   - Which transition proteins remain stable?

2. **Does batch correction STRENGTHEN or WEAKEN DEATh theorem?**
   - Collagen predictability: higher or lower after correction?
   - Core vs Associated entropy difference: more or less significant?

3. **What NEW insights emerge from cleaner data?**
   - New high-transition proteins revealed?
   - Better separation of entropy clusters?

4. **Should we update therapeutic targets?**
   - Which entropy-based biomarkers survive batch correction?
   - New intervention points?

## Timeline
- **Planning:** 5 minutes (01_plan file)
- **Analysis:** 20-30 minutes (entropy calculation, clustering, comparisons)
- **Visualization:** 10 minutes (5 required plots)
- **Documentation:** 15-20 minutes (Knowledge Framework report)
- **TOTAL:** ~60 minutes per agent

## Notes
- 🔴 **CRITICAL:** Use Knowledge Framework for ALL markdown files (thesis → overview → mermaid → MECE sections)
- 🔴 **CRITICAL:** ALL artifacts in your workspace folder ONLY
- 🔴 **NEVER MOCK DATA:** Use real batch-corrected CSV
- 📊 **Be creative:** Suggest additional entropy metrics (conditional entropy, transfer entropy, network entropy)
- 🧬 **Think deeply:** What is entropy really measuring in ECM aging? Thermodynamic vs information entropy?

---

**Ready to launch agents!** 🚀
