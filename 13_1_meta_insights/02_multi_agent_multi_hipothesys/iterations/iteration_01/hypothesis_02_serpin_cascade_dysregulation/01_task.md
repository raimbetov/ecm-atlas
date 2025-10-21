# Hypothesis 02: Serpin Cascade Dysregulation as Central Aging Mechanism

## Scientific Question

Is serpin family dysregulation (upregulation of inhibitory serpins + downregulation of protective serpins) the unifying central mechanism driving multi-pathway ECM aging across coagulation, inflammation, and matrix degradation systems?

## Background Context

**Source Insights - Serpins appear EVERYWHERE:**

1. **Entropy transitions** (highest regime shifters):
   - PZP (pregnancy zone protein α2-macroglobulin-like)
   - SERPINB2 (PAI-2, plasminogen activator inhibitor-2)
   - TNFSF13 (APRIL, related to serpin signaling)

2. **Weak signals** (G4):
   - Serpina3m = top weak signal protein

3. **Universal markers** (G1):
   - Serpinh1, Serpinf1 in cross-tissue expression

4. **Coagulation cascade** (S1):
   - SERPINC1 (antithrombin) upregulated with fibrinogen

**Hypothesis:** Serpins are not peripheral players but CENTRAL HUBS. Their dysregulation creates cascade failures across multiple pathways simultaneously, making them the master regulators of ECM aging. Fix serpins → fix aging.

## Data Source

**Primary Dataset:**
```
/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv
```

**Critical Columns:**
- `Gene_Symbol` / `Canonical_Gene_Symbol`: Identify all SERPIN* genes
- `Zscore_Delta`: Aging trajectory
- `Tissue_Compartment`: Cross-tissue expression
- `Matrisome_Category`: Regulatory vs structural classification
- `Study_ID`: Multi-study validation

**Validation Data:**
- Entropy analysis: `/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/01_entropy_multi_agent_after_batch_corection/00_INTEGRATED_ENTROPY_THEORY_V2.md`
- Prior insights: `/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/compare previos insights wiht new dataset/codex/90_results_codex.md`

## Success Criteria

### Criterion 1: Serpin Family Comprehensive Profiling (40 pts)

**Required:**
1. Identify ALL serpin family members in dataset (SERPIN*, A2M, PZP, etc.)
2. For each serpin:
   - Mean Δz across all studies
   - Tissue breadth (# tissues expressed)
   - Directional consistency (% studies with same sign)
   - Entropy metrics (Shannon, transition score if available)
3. Classify serpins by function:
   - Protease inhibitors (coagulation: SERPINC1, SERPINF2)
   - Anti-inflammatory (SERPINA3)
   - ECM modulators (SERPINH1 = HSP47 collagen chaperone)
4. Statistical test: Are serpins MORE dysregulated than non-serpin proteins? (|Δz|_serpins vs |Δz|_others, Mann-Whitney p<0.05)

**Deliverables:**
- CSV: `serpin_comprehensive_profile_[agent].csv`
- Serpin classification table by function
- Statistical comparison: serpins vs non-serpins

### Criterion 2: Network Centrality Analysis (30 pts)

**Required:**
1. Build protein correlation network:
   - Nodes = proteins
   - Edges = Spearman correlation of Δz across tissues (|ρ| > 0.5, p<0.05)
2. Calculate network centrality metrics for ALL proteins:
   - Degree centrality (# connections)
   - Betweenness centrality (# shortest paths through node)
   - Eigenvector centrality (connected to important nodes)
3. Test hypothesis: **Do serpins have HIGHER centrality than average?**
4. Identify serpin "hubs" (top 10% centrality across metrics)

**Deliverables:**
- Network graph (PNG): Highlight serpins in red, size by centrality
- Centrality statistics table: Serpin mean vs non-serpin mean, effect size
- Top 10 hub proteins (predict: majority are serpins?)

### Criterion 3: Multi-Pathway Involvement (20 pts)

**Required:**
1. Map serpins to biological pathways:
   - Coagulation cascade (SERPINC1, SERPINF2, PZP)
   - Fibrinolysis (SERPINB2 = PAI-2)
   - Complement system (SERPING1 = C1-inhibitor)
   - ECM assembly (SERPINH1 = HSP47)
   - Inflammation (SERPINA3)
2. Test: **Do individual serpins participate in MULTIPLE pathways?**
3. Calculate pathway dysregulation score: Mean |Δz| of all serpins in pathway
4. Rank pathways by serpin dysregulation severity

**Deliverables:**
- Serpin-pathway matrix (serpins × pathways, mark participation)
- Pathway dysregulation ranking table
- Venn diagram: Serpin overlap across pathways

### Criterion 4: Temporal and Therapeutic Implications (10 pts)

**Required:**
1. **Temporal ordering:** Can serpins predict downstream pathway changes?
   - Compare Δz magnitude: Serpins vs pathway target proteins
   - Hypothesis: If serpins are DRIVERS, their |Δz| should be HIGHER than targets
2. **Therapeutic targeting:**
   - Identify druggable serpins (approved drugs exist?)
   - Predict cascade effects: Modulating which serpin would have widest impact?
3. Prioritize top 3 serpin targets by:
   - Network centrality (highest hubs)
   - Multi-pathway involvement
   - Druggability

**Deliverables:**
- Temporal analysis: Serpin Δz vs target Δz comparison
- Top 3 therapeutic targets with rationale
- Drug availability table (if time permits)

## Required Artifacts

All files MUST be created in your agent-specific subfolder (`claude_code/` or `codex/`).

### Mandatory Files:

1. **01_plan_[agent].md** - Analysis strategy
2. **analysis_[agent].py** - Executable Python script
3. **serpin_comprehensive_profile_[agent].csv** - All serpins with metrics
4. **network_centrality_[agent].csv** - Centrality scores for all proteins
5. **pathway_dysregulation_[agent].csv** - Pathway-level statistics
6. **visualizations_[agent]/** - Folder with:
   - Network graph (serpins highlighted)
   - Serpin vs non-serpin comparison (box plot)
   - Pathway dysregulation heatmap
   - Serpin-pathway Venn diagram
7. **90_results_[agent].md** - Final report in Knowledge Framework format

### Optional Supporting Files:

- Protein-protein interaction network data
- Literature evidence for serpin functions
- Drug database integration

## Self-Evaluation Rubric

Score yourself out of 100 points total:

### Completeness (40 pts)
- [ ] All serpins identified and profiled
- [ ] Network centrality calculated for all proteins
- [ ] Multi-pathway involvement mapped
- [ ] Temporal ordering tested
- [ ] Therapeutic targets prioritized

### Accuracy (30 pts)
- [ ] Serpin family members correctly identified (SERPIN*, A2M, PZP)
- [ ] Network analysis methods valid (correlation threshold justified)
- [ ] Centrality metrics computed correctly
- [ ] Statistical tests appropriate (Mann-Whitney, permutation tests)
- [ ] Pathway assignments evidence-based

### Novelty (20 pts)
- [ ] Elevates serpins from supporting to central role
- [ ] Network centrality analysis NEW (not in prior insights)
- [ ] Multi-pathway involvement quantified systematically
- [ ] Identifies serpin cascade as unifying mechanism
- [ ] Proposes specific therapeutic intervention strategy

### Reproducibility (10 pts)
- [ ] Python code executes without errors
- [ ] Network visualization clear and informative
- [ ] All statistics documented
- [ ] Results replicable from code

**Grading:**
- 90-100: Hypothesis strongly confirmed, serpins are central hubs, therapeutic strategy clear
- 70-89: Hypothesis supported, serpins important but not exclusively central
- 50-69: Mixed evidence, serpins involved but centrality unclear
- <50: Hypothesis not supported, serpins peripheral

## Documentation Standards

All `.md` files MUST follow Knowledge Framework:

**Required Structure:**
1. **Thesis:** 1 sentence summarizing serpin centrality finding
2. **Overview:** 1 paragraph expanding thesis and previewing sections
3. **Mermaid Diagrams:**
   - TD: Serpin family structure (by function categories)
   - LR: Serpin dysregulation → cascade failures → aging
4. **MECE Sections:** 1.0 Serpin Profiling, 2.0 Network Centrality, 3.0 Pathway Involvement, 4.0 Therapeutics
5. **Numbered Paragraphs:** ¶1, ¶2, ¶3 within each section

**Reference:**
```
/Users/Kravtsovd/projects/ecm-atlas/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md
```

## Agent Identity vs Data Source

**YOU ARE:** `[agent name]` (e.g., `claude_code` or `codex`)

**YOU READ:** Same dataset for both agents:
```
/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv
```

## Sanity Checks

Before proceeding:

```python
import pandas as pd
import re

# Load dataset
df = pd.read_csv('/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv')

# Identify serpins
serpin_pattern = re.compile(r'SERPIN|A2M|PZP', re.IGNORECASE)
serpins = df[df['Gene_Symbol'].str.contains(serpin_pattern, na=False)]['Gene_Symbol'].unique()

# Sanity checks
assert len(serpins) >= 10, f"Expected ≥10 serpins, found {len(serpins)}"
assert 'PZP' in serpins or 'Pzp' in serpins, "Missing PZP (entropy transition leader)!"
assert any('SERPINC1' in s or 'Serpinc1' in s for s in serpins), "Missing SERPINC1 (coagulation)!"

print(f"✓ Dataset loaded: {df.shape[0]} rows")
print(f"✓ Serpins identified: {len(serpins)}")
print(f"✓ Serpin examples: {list(serpins)[:5]}")
```

## Expected Results (for validation)

**From prior insights:**
- PZP = highest entropy transition score
- SERPINB2 = high entropy transition
- Serpina3m = top weak signal protein
- SERPINC1 upregulated in coagulation cascade

**Your analysis should:**
- Confirm these findings AND
- Show serpins have HIGH network centrality AND
- Demonstrate multi-pathway involvement AND
- Propose serpin-targeting therapeutic strategy

## Timeline

**Recommended workflow:**
1. Load data and identify serpins (1 hour)
2. Comprehensive serpin profiling (2 hours)
3. Network analysis (2-3 hours)
4. Pathway mapping (1-2 hours)
5. Therapeutic prioritization (1 hour)
6. Visualization and write-up (2-3 hours)

**Total:** 9-12 hours

## Contact

Questions: daniel@improvado.io

---

**Task Created:** 2025-10-20
**Hypothesis ID:** H02
**Iteration:** 01
**Predicted Scores:** Novelty 9/10, Impact 9/10
