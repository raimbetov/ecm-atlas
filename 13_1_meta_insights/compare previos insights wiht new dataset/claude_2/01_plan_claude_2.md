# Meta-Insights Validation Plan: Agent claude_2

**Thesis:** Independent re-validation of 12 meta-insights (7 GOLD, 5 SILVER) using batch-corrected CODEX V2 dataset to quantify signal strength improvement and identify emergent discoveries.

---

## 1.0 AGENT IDENTITY & DATA SOURCE

### 1.1 Identity (Output Configuration)
- **Agent name:** claude_2
- **Working directory:** `/Users/Kravtsovd/projects/ecm-atlas`
- **Output folder:** `/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/claude_2/`
- **File prefix:** `claude_2_*`

### 1.2 Data Source (Input Configuration)
- **Input file:** `/Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv`
- **Critical note:** Using CODEX file (NOT claude_2 file) - see task rationale
- **Expected:** 9,300 rows, 28 columns, PCOLCE in 7 studies

---

## 2.0 VALIDATION APPROACH

### 2.1 Validation Sequence (GOLD → SILVER)

**Phase 1: GOLD-tier Insights (Critical - 7 insights)**
1. **G1: Universal Markers** (Est. 15 min)
   - Recompute universality scores (≥3 tissues, ≥70% consistency)
   - Compare V1 (12.2%) vs V2 expected (15-18%)
   - Top markers: Hp, VTN, Col14a1, F2, FGB validation

2. **G2: PCOLCE Quality Paradigm** (Est. 10 min)
   - Verify PCOLCE Δz=-0.82 → expected V2 ~-1.4
   - Confirm 7 studies, >90% consistency
   - Check outlier status preservation

3. **G3: Batch Effects Dominate** (Est. 20 min)
   - PCA on V2 dataset
   - Compare PC1 loadings: Study_ID (0.674 → <0.3) vs Age_Group (-0.051 → >0.4)
   - Quantify batch correction success

4. **G4: Weak Signals Compound** (Est. 15 min)
   - Filter: 0.3 < |Δz| < 0.8, consistency ≥65%
   - V1: 14 proteins → V2: expected 20-30
   - Pathway-level aggregation via Matrisome_Category

5. **G5: Entropy Transitions** (Est. 20 min)
   - Shannon entropy H, CV, predictability per protein
   - DEATh theorem: collagens 28% predictable
   - Ordered (H<1.5) vs chaotic (H>2.0) classification

6. **G6: Compartment Antagonism** (Est. 15 min)
   - Within-tissue opposite directions (Tissue_Compartment)
   - V1: 11 antagonistic events, Col11a2 SD=1.86
   - Calculate divergence: SD(Δz) across compartments

7. **G7: Species Divergence** (Est. 15 min)
   - Human-mouse concordance (Species column)
   - V1: 8/1,167 genes shared, R=-0.71
   - Verify CILP remains only universal cross-species marker

**Phase 2: SILVER-tier Insights (Therapeutic - 5 insights)**
8. **S1: Fibrinogen Cascade** (Est. 10 min)
   - FGA +0.88, FGB +0.89, SERPINC1 +3.01 verification
   - Coagulation protein upregulation pattern

9. **S2: Temporal Windows** (Est. 10 min)
   - Age 40-50 (prevention), 50-65 (restoration), 65+ (rescue)
   - Recompute temporal trajectories

10. **S3: TIMP3 Lock-in** (Est. 10 min)
    - Δz=+3.14, 81% consistency validation
    - Extreme accumulation pattern

11. **S4: Tissue-Specific TSI** (Est. 10 min)
    - 13 proteins TSI > 3.0, KDM5C TSI=32.73
    - Tissue specificity index recalculation

12. **S5: Biomarker Panel** (Est. 10 min)
    - 7-protein plasma ECM aging clock
    - Top candidates remain after batch correction

**Phase 3: Discovery & Documentation (30 min)**
- Identify new emergent patterns
- Classify: CONFIRMED, MODIFIED, REJECTED
- Create all required deliverables

---

## 3.0 VALIDATION METRICS

### 3.1 Primary Metrics Per Insight

| Insight | V1 Baseline | V2 Expected | Success Criterion |
|---------|-------------|-------------|-------------------|
| G1 | 12.2% universal | 15-18% | ≥20% increase |
| G2 | PCOLCE Δz=-0.82 | Δz≈-1.4 | Same direction, stronger |
| G3 | Study PC1=0.674 | PC1<0.3 | 50% reduction |
| G3 | Age PC1=-0.051 | PC1>0.4 | 10x improvement |
| G4 | 14 weak signals | 20-30 proteins | ≥40% increase |
| G5 | 52 entropy transitions | 60-80 proteins | Clearer clusters |
| G6 | 11 antagonistic | 10-15 events | Pattern persists |
| G7 | 8/1,167 shared | 6-10 genes | Low concordance maintained |

### 3.2 Classification Logic

**CONFIRMED (✅):** V2 strengthens V1 finding
- Same direction (positive→positive OR negative→negative)
- Magnitude increase ≥20%
- p-value improvement if applicable

**MODIFIED (⚠️):** V2 changes magnitude but preserves core
- Same direction
- Magnitude change -20% to +20%
- Core biological pattern intact

**REJECTED (❌):** V2 contradicts V1
- Opposite direction
- Magnitude reduction >50%
- Pattern disappears

---

## 4.0 SANITY CHECKS (MANDATORY)

### 4.1 Pre-Validation Checks

```python
# Check 1: Row count
assert len(df_v2) == 9300, "Wrong file - row count mismatch"

# Check 2: Column count
assert len(df_v2.columns) == 28, "Wrong file - column count mismatch"

# Check 3: Schema verification
assert 'Canonical_Gene_Symbol' in df_v2.columns, "Missing key column"

# Check 4: PCOLCE ground truth
pcolce_studies = df_v2[df_v2['Canonical_Gene_Symbol'].str.upper() == 'PCOLCE']['Study_ID'].nunique()
assert pcolce_studies == 7, f"PCOLCE should be in 7 studies, found {pcolce_studies}"

# Check 5: Required columns present
required = ['Canonical_Gene_Symbol', 'Study_ID', 'Tissue_Compartment',
            'Zscore_Delta', 'Matrisome_Category']
assert all(col in df_v2.columns for col in required), "Missing required columns"
```

### 4.2 If ANY Check Fails
1. STOP immediately
2. Do NOT proceed with validation
3. Report error with details
4. Ask user to verify file paths

---

## 5.0 DELIVERABLES CHECKLIST

### 5.1 Required Files (6 files)

- [x] `01_plan_claude_2.md` (THIS FILE) ✅ 2025-10-18 14:30
- [ ] `validation_pipeline_claude_2.py` - Reproducible analysis script
- [ ] `validation_results_claude_2.csv` - 12 rows (7 GOLD + 5 SILVER)
- [ ] `v2_validated_proteins_claude_2.csv` - Subset of V2 data
- [ ] `new_discoveries_claude_2.csv` - Emergent findings (if any)
- [ ] `90_results_claude_2.md` - Final report with self-evaluation

### 5.2 Output Schema Templates

**validation_results_claude_2.csv:**
```
Insight_ID,Tier,Original_Metric,V2_Metric,Change_Percent,Classification,Notes
G1,GOLD,12.2% universal,XX.X% universal,+XX.X%,CONFIRMED/MODIFIED/REJECTED,"Details"
```

**new_discoveries_claude_2.csv:**
```
Discovery_Type,Protein/Pattern,Metric,Description
Universal_Marker,PROTEIN_NAME,Universality=X.XXX,"Description"
```

**v2_validated_proteins_claude_2.csv:**
- Subset of V2 dataset
- Only proteins relevant to validated insights
- All 28 columns preserved

---

## 6.0 PROGRESS TRACKING

### 6.1 Validation Progress (Update as completed)

**GOLD Insights:**
- [ ] G1: Universal Markers - ETA: 14:45
- [ ] G2: PCOLCE Quality - ETA: 14:55
- [ ] G3: Batch Effects - ETA: 15:15
- [ ] G4: Weak Signals - ETA: 15:30
- [ ] G5: Entropy - ETA: 15:50
- [ ] G6: Compartment Antagonism - ETA: 16:05
- [ ] G7: Species Divergence - ETA: 16:20

**SILVER Insights:**
- [ ] S1: Fibrinogen - ETA: 16:30
- [ ] S2: Temporal Windows - ETA: 16:40
- [ ] S3: TIMP3 - ETA: 16:50
- [ ] S4: Tissue TSI - ETA: 17:00
- [ ] S5: Biomarker Panel - ETA: 17:10

**Documentation:**
- [ ] New discoveries identified - ETA: 17:25
- [ ] Results CSV created - ETA: 17:35
- [ ] Final report written - ETA: 17:50

### 6.2 Key Milestones

- [x] Plan created - 2025-10-18 14:30
- [ ] Sanity checks passed - Target: 14:35
- [ ] All GOLD validated - Target: 16:20
- [ ] All SILVER validated - Target: 17:10
- [ ] All deliverables complete - Target: 17:50

---

## 7.0 RISK MITIGATION

### 7.1 Known Risks

**Risk 1: Wrong file loaded**
- Mitigation: MANDATORY sanity checks before any analysis
- Detection: Row count ≠ 9300 OR column count ≠ 28

**Risk 2: PCOLCE metric mismatch**
- Mitigation: Use Canonical_Gene_Symbol (NOT Gene_Symbol)
- Detection: PCOLCE studies ≠ 7

**Risk 3: Missing compartment granularity**
- Mitigation: Use Tissue_Compartment column (codex file specific)
- Detection: Cannot compute antagonistic remodeling

### 7.2 Validation Quality Checks

**Per-insight validation:**
1. Verify metric computation matches original methodology
2. Check for NaN handling (exclude from mean/std)
3. Zero values (0.0) are valid - include in statistics
4. Document any deviations from original approach

---

## 8.0 SUCCESS CRITERIA (Self-Evaluation)

### 8.1 Target Score: 90+ points (EXCELLENT)

**Completeness (40 pts):**
- 7 GOLD validated (20 pts)
- 5 SILVER validated (10 pts)
- 6 files created (10 pts)

**Accuracy (30 pts):**
- Metrics correct (15 pts)
- Sanity checks passed (10 pts)
- Classification defensible (5 pts)

**Insights (20 pts):**
- New discoveries ≥1 (10 pts)
- Therapeutic implications (5 pts)
- Signal improvement quantified (5 pts)

**Reproducibility (10 pts):**
- Python script provided (5 pts)
- Script runs without errors (5 pts)

### 8.2 Minimum Acceptable: 70 points (GOOD)

---

## 9.0 REFERENCES

**V2 Dataset:**
`/Users/Kravtsovd/projects/ecm-atlas/14_exploratory_batch_correction/multi_agents_ver1_for_batch_cerection/step2_batch/codex/merged_ecm_aging_COMBAT_V2_CORRECTED_codex.csv`

**Task File:**
`/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/01_task_meta_insights_validation_V2_FIXED.md`

**Insights Catalog:**
`/Users/Kravtsovd/projects/ecm-atlas/13_meta_insights/00_MASTER_META_INSIGHTS_CATALOG.md`

**Framework:**
`/Users/Kravtsovd/projects/ecm-atlas/03_KNOWLEDGE_FRAMEWORK_DOCUMENTATION_STANDARDS.md`

---

**Created:** 2025-10-18 14:30
**Agent:** claude_2
**Estimated completion:** 2025-10-18 17:50 (3.5 hours)
**Status:** READY TO EXECUTE
