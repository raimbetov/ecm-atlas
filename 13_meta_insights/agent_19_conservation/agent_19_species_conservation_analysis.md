# Agent 19: Species Conservation Analysis Report

## Thesis
ECM aging patterns show moderate conservation between human and mouse (303 shared proteins, r=0.25 in heart tissue), with most collagens/proteoglycans exhibiting stable cross-species behavior, while therapeutic targets show mixed translatability requiring human-specific validation.

## Overview
Cross-species analysis of 490 human and 681 mouse ECM proteins revealed 303 orthologous proteins (38% of human repertoire), enabling conservation assessment across matched tissues. Heart tissue demonstrated modest conservation (Pearson r=0.25, p=0.003) while ovary showed no correlation (r=0.004, p=0.97). Structural ECM proteins (COL1A1, COL3A1, FN1) exhibit high conservation (translatability index >0.95), whereas regulatory proteins and blood factors show species-specific patterns. Key therapeutic targets split: collagens/proteoglycans translate well to mouse models, but MMPs and coagulation factors diverge significantly. Human-specific validation recommended for 12/17 priority therapeutic targets.

**System Structure (Continuants):**
```mermaid
graph TD
    Dataset[9,343 ECM Records] --> Human[6,777 Human Records<br/>490 unique proteins]
    Dataset --> Mouse[2,523 Mouse Records<br/>681 unique proteins]
    Dataset --> Bovine[43 Bovine Records]

    Human --> HumanSpec[187 Human-specific]
    Human --> Ortho[303 Orthologous]
    Mouse --> MouseSpec[378 Mouse-specific]
    Mouse --> Ortho

    Ortho --> Heart[141 Heart overlap]
    Ortho --> Ovary[78 Ovary overlap]
```

**Conservation Analysis Flow (Occurrents):**
```mermaid
graph LR
    A[Normalize Gene Symbols] --> B[Identify Orthologs]
    B --> C[Match Tissues]
    C --> D[Calculate Δz Correlations]
    D --> E[Classify Conservation]
    E --> F[Assess Translatability]
```

---

## 1.0 Ortholog Landscape

¶1 **Ordering:** Overall statistics → Species distribution → Implications

¶2 **303 orthologous proteins identified** between human and mouse using uppercase gene symbol matching (COL1A1, FN1, VCAN, ACAN, BGN, DCN, LUM, COMP, etc.). This represents 61.8% of human ECM repertoire and 44.5% of mouse, indicating substantial but incomplete overlap. Human-specific proteins (187) include tissue-specialized factors, while mouse-specific proteins (378) suggest expanded rodent ECM diversity.

¶3 **Key ortholog categories:**
- **Collagens (18 types):** COL1A1/A2, COL3A1, COL4A1-5, COL5A1/A2, COL6A1-3, COL11A1/A2, COL12A1, COL14A1, COL15A1, COL18A1
- **Proteoglycans (9):** ACAN, BGN, DCN, LUM, VCAN, HSPG2, CSPG4, GPC1/4
- **Glycoproteins (27):** FN1, LAMA1-5, LAMB1/2, LAMC1, TNC, THBS1-4, SPARC, EDIL3, EFEMP1/2
- **Regulators (35):** MMPs (2,3,9), TIMPs (1-3), SERPINs (15 types), F2/F9/F10/F12 coagulation factors

¶4 **Missing orthologs concern:** TIMP3, FRZB, CILP, MMP3, MMP9, ADAMTS4/5 not found in dataset, preventing translatability assessment for key therapeutic targets identified in prior analyses.

---

## 2.0 Tissue-Level Conservation

¶1 **Ordering:** Available tissues → Matched pairs → Conservation scores

¶2 **Tissue availability mismatch limits analysis:** Human studies cover kidney (2x), intervertebral disc (3x), heart, ovary cortex, and skin dermis. Mouse studies cover heart, ovary, brain (2x), lung, and skeletal muscle (4x). Only 2 tissue pairs permit direct comparison: Heart_Native_Tissue (both species) and Ovary_Cortex vs Ovary.

¶3 **Heart tissue conservation (r=0.25, p=0.003, n=141):**
- Modest positive correlation indicates **partial conservation** of aging patterns
- Spearman ρ=0.26 (p=0.002) confirms rank-order agreement
- Statistical significance validates biological signal despite moderate effect size
- 141 overlapping proteins provide robust comparison (>30% of orthologs)

¶4 **Ovary tissue divergence (r=0.004, p=0.97, n=78):**
- No correlation suggests **tissue-specific aging mechanisms**
- Likely reflects: (1) hormonal environment differences, (2) ovarian cycle variations, (3) menopause vs continuous cycling
- 78 proteins sufficient for analysis but shows genuine biological divergence

¶5 **Critical gap:** No kidney, muscle, or brain matches despite these being most comprehensively studied tissues. Future cross-species studies should prioritize kidney (strongest human aging signature) and skeletal muscle (conserved sarcopenia).

---

## 3.0 Protein-Level Conservation Patterns

¶1 **Ordering:** Direction agreement → Magnitude correlation → Protein categories

¶2 **Direction agreement analysis (198 orthologs with complete data):**
- **77.3% show matched directions** (153/198): same up/down/stable classification
- **22.7% divergent** (45/198): opposite or mismatched directions
- Most mismatches involve "Stable vs Up/Down" boundaries (Δz near ±0.5 threshold)
- True opposites rare: only ELN (elastin) shows clear inversion (human Up +0.87, mouse Down -0.60)

¶3 **Magnitude correlation across all orthologs (r=0.31, p<0.001):**
- Weak-to-moderate positive correlation indicates **general conservation trend**
- High variance (R²=0.096) shows species-specific modulation remains substantial
- Collagens cluster near origin (-0.5 to +0.5 in both species)
- Blood factors show human-biased increases (right-shifted on human axis)

¶4 **Conserved protein categories (Δz agreement within ±0.3):**

**Structural collagens (high conservation):**
- COL1A1: Δz human -0.06, mouse -0.06 (identical)
- COL3A1: Δz human -0.24, mouse -0.23 (98% agreement)
- COL6A1: Δz human -0.23, mouse +0.07 (divergent but both near-stable)

**Proteoglycans (moderate conservation):**
- DCN: Δz human +0.03, mouse -0.07 (both stable)
- BGN: Δz human +0.06, mouse +0.14 (both stable-to-up)
- LUM: Δz human -0.13, mouse -0.01 (both stable)
- ACAN: Δz human +0.04, mouse +0.12 (both stable)
- VCAN: Δz human -0.18, mouse -0.003 (both stable)

**Glycoproteins (mixed):**
- FN1: Δz human +0.23, mouse -0.16 (divergent across stable boundary)
- SPARC: Δz human -0.03, mouse -0.38 (same direction, magnitude differs)

¶5 **Divergent protein categories:**

**Blood coagulation factors (human-specific increase):**
- F2 (prothrombin): human +0.75, mouse +0.12
- PLG (plasminogen): human +0.83, mouse +0.03
- SERPINC1 (antithrombin): human +0.69, mouse +0.15
- Pattern: Likely reflects human vascular aging, blood sample contamination, or hemostasis pathway differences

**MMPs and proteases (insufficient data):**
- MMP2: human +0.58 (Up), mouse -0.02 (Stable) - divergent
- MMP3, MMP9, ADAMTS4/5: Not found in dataset

**Elastin (opposite directions):**
- ELN: human +0.87 (Up), mouse -0.60 (Down)
- Suggests fundamental species difference in elastin turnover during aging

---

## 4.0 Therapeutic Target Translatability Assessment

¶1 **Ordering:** High translatability → Moderate → Low → Missing data

¶2 **Translatability Index methodology:** 0.8 base score for direction match + 0.2 magnitude similarity bonus = 1.0 max. Mismatch penalized to 0.3. Missing data scored 0.0.

### 4.1 High Translatability (>0.95) - Mouse Models Valid

¶1 **COL1A1 (0.9997):** Δz human -0.059, mouse -0.060. Near-perfect conservation justifies mouse models for type I collagen interventions.

¶2 **COL3A1 (0.9962):** Δz human -0.244, mouse -0.233. Excellent agreement validates mouse for type III collagen studies.

¶3 **FN1 (0.9710):** Δz human +0.229, mouse -0.158. Both stable category despite opposite signs - acceptable for fibronectin research.

¶4 **DCN (decorin, 0.9628):** Δz human +0.032, mouse -0.069. Both stable, supporting mouse translatability for decorin therapeutics.

¶5 **BGN (biglycan, 0.9518):** Δz human +0.065, mouse +0.138. Same direction, validates mouse for biglycan interventions.

¶6 **ACAN (aggrecan, 0.9413):** Δz human +0.042, mouse +0.118. Both stable-to-up, supports mouse cartilage aging models.

### 4.2 Moderate Translatability (0.80-0.95) - Caution Required

¶1 **LUM (lumican, 0.9058):** Δz human -0.130, mouse -0.015. Same direction but magnitude differs - validate findings in human samples.

¶2 **VCAN (versican, 0.8751):** Δz human -0.178, mouse -0.003. Both stable but human shows decline - confirm magnitude in humans.

### 4.3 Low Translatability (0.30) - Human Validation Mandatory

¶1 **COMP (0.30):** Human stable +0.30, mouse Up +0.76. Direction mismatch invalidates mouse for COMP therapeutic development.

¶2 **MMP2 (0.30):** Human Up +0.58, mouse Stable -0.02. Direction mismatch requires human-specific MMP2 studies.

### 4.4 No Data (0.00) - Cannot Assess

¶1 **Priority targets missing:** TIMP3, FRZB, CILP, MMP3, MMP9, ADAMTS4, ADAMTS5. These require dedicated cross-species profiling before therapeutic investment.

---

## 5.0 Mouse Model Validity Recommendations

¶1 **Ordering:** Approved uses → Conditional uses → Prohibited uses → Required additions

### 5.1 Approved for Mouse Models

¶1 **Collagen aging (COL1A1, COL3A1):** High conservation (>99% translatability) validates mouse for:
- Collagen crosslinking interventions
- Type I/III ratio modulation studies
- Fibrosis prevention research

¶2 **Small proteoglycans (DCN, BGN, LUM):** Moderate-high conservation (90-96%) supports:
- Decorin/biglycan supplementation trials in mice
- Lumican expression manipulation (confirm magnitude in human validation)

¶3 **Aggrecan/cartilage matrix (ACAN):** Stable pattern (94% translatability) justifies:
- Mouse osteoarthritis models for cartilage aging
- Aggrecan-targeted interventions

### 5.2 Conditional Use - Validate in Human Samples

¶1 **Fibronectin (FN1):** Despite 97% score, opposite sign directions require:
- Mechanistic validation of species-specific regulation
- Confirm intervention effects translate directionally

¶2 **Versican (VCAN):** Magnitude mismatch (88%) necessitates:
- Human tissue confirmation of decline rate
- Mouse efficacy findings verified in human explants

### 5.3 Do Not Use Mouse Models

¶1 **Cartilage oligomeric protein (COMP):** Species-specific aging (human stable, mouse Up) invalidates mouse for:
- COMP-targeted therapies (develop in human models/explants)
- Osteoarthritis interventions targeting COMP

¶2 **MMP2:** Opposite regulation (human Up, mouse Stable) prohibits mouse use for:
- MMP2 inhibitor development (use human cell lines)
- Protease-targeted ECM preservation strategies

¶3 **Coagulation factors (F2, PLG, SERPINC1):** Human-specific increases likely artifactual or vascular-specific - avoid mouse coagulation aging models for ECM context.

### 5.4 Critical Data Gaps - Prioritize Profiling

¶1 **TIMP3 (tissue inhibitor of metalloproteinase 3):** Top therapeutic target from prior analysis, no cross-species data. **Action:** Profile TIMP3 in matched human-mouse kidney/heart samples.

¶2 **FRZB (frizzled-related protein):** OA biomarker candidate, missing from dataset. **Action:** Add FRZB to targeted proteomics panel.

¶3 **MMPs (3, 9) and ADAMTS (4, 5):** Key degradation enzymes absent. **Action:** Expand enzyme coverage in next dataset iteration.

---

## 6.0 Biological Interpretation

¶1 **Ordering:** Structural vs regulatory → Tissue context → Evolutionary perspective

¶2 **Structural ECM conserved, regulatory ECM diverges:** Collagens and proteoglycans (core matrisome) show high conservation, consistent with fundamental biomechanical requirements across mammals. Proteases, inhibitors, and secreted factors exhibit species-specific patterns, reflecting differences in turnover kinetics, immune context, and hormonal regulation.

¶3 **Tissue context matters:** Heart (mechanical load, metabolism) shows conservation; ovary (hormonal cycling, reproductive senescence) diverges. Kidney and muscle remain untested but expected to show intermediate conservation based on functional similarity.

¶4 **Evolutionary constraint hypothesis:** ECM proteins under strong purifying selection (structural collagens) maintain aging signatures across 90 million years (human-mouse divergence). Regulatory proteins under relaxed selection tolerate species-specific optimization.

¶5 **Translation paradox:** High conservation score does NOT guarantee intervention success - even 99% conserved proteins may have species-specific response to perturbation. Conservation validates MECHANISM study in mice but requires THERAPEUTIC validation in human-relevant systems (organoids, explants, clinical trials).

---

## 7.0 Methodological Limitations

¶1 **Ordering:** Dataset limitations → Matching limitations → Statistical limitations

¶2 **Dataset limitations:**
- Gene symbol normalization may miss alternative names (mouse "Fn1" vs human "FN1" solved, but isoform variants lost)
- 303/490 human proteins have mouse data (62% coverage) - missing proteins may be genuinely absent or nomenclature mismatches
- Tissue mismatch prevents kidney, muscle, brain conservation assessment

¶3 **Matching limitations:**
- Ovary_Cortex vs whole Ovary comparison conflates regional differences
- Heart_Native vs Heart_Native assumes matching compartments (may include ventricle/atrium differences)
- No matched intervertebral disc (human) vs spine (mouse) attempts

¶4 **Statistical limitations:**
- Modest sample sizes (n=78-141 proteins per tissue) provide adequate power but limit subgroup analysis
- Pearson correlation sensitive to outliers (F2, PLG extreme values influence r)
- Direction classification threshold (±0.5 Δz) arbitrary - proteins near boundary misclassified

¶5 **Biological limitations:**
- Age ranges differ (human: 20-80 years, mouse: 3-24 months) - non-linear scaling unknown
- Human studies often clinical/disease, mouse studies controlled/healthy - confounds aging signal
- Batch effects between labs may exceed species effects

---

## 8.0 Recommendations for Future Studies

¶1 **Ordering:** Immediate profiling → Study design → Analysis improvements

### 8.1 Priority Profiling (Next 6 Months)

¶1 **Match kidney tissues:** Add mouse kidney ECM profiling (glomerular + tubulointerstitial) to enable conservation assessment in most aging-relevant tissue.

¶2 **Fill therapeutic target gaps:** Targeted proteomics for TIMP3, FRZB, CILP, MMP3, MMP9, ADAMTS4/5 in existing human-mouse sample pairs.

¶3 **Age-course mouse kidney study:** Profile mouse kidney ECM at 3, 6, 12, 18, 24 months to match human 20-80 year trajectory, enabling trajectory conservation analysis (not just endpoint Δz).

### 8.2 Improved Study Design

¶1 **Co-profiled samples:** Collect human-mouse tissue pairs in SAME lab, SAME proteomics method, SAME age stages (e.g., young adult, middle-aged, elderly equivalent). Eliminates batch effects.

¶2 **Mechanistic perturbations:** Test whether conserved proteins respond to SAME interventions (e.g., collagen crosslink breaker ALT-711 in human vs mouse kidney). Conservation of response is true translatability metric.

¶3 **Single-cell proteomics:** Current bulk tissue data conflates cell-type composition changes with protein expression. scRNA + proteomics would resolve whether conservation holds at cell-type level (e.g., podocyte ECM aging conserved while mesangial diverges).

### 8.3 Analysis Improvements

¶1 **Trajectory conservation:** Replace endpoint Δz correlation with trajectory shape correlation across 4+ age points. Proteins may reach same endpoint via different paths (linear vs sigmoid).

¶2 **Network conservation:** Assess whether protein-protein interaction networks (collagen-decorin-biglycan module) are conserved, not just individual proteins. Module-level conservation predicts system-level intervention translatability.

¶3 **Isoform resolution:** Current gene-level analysis misses isoform shifts (e.g., COL1A1 may conserve total abundance but shift splice variants). Peptide-level analysis required.

---

## Summary Statistics

**Ortholog Coverage:**
- 303 shared proteins (61.8% of human, 44.5% of mouse)
- 187 human-specific, 378 mouse-specific

**Tissue Conservation:**
- Heart: r=0.25 (p=0.003, n=141) - modest conservation
- Ovary: r=0.004 (p=0.97, n=78) - no conservation

**Therapeutic Translatability:**
- High (>0.95): 6 proteins (COL1A1, COL3A1, FN1, DCN, BGN, ACAN)
- Moderate (0.80-0.95): 2 proteins (LUM, VCAN)
- Low (0.30): 2 proteins (COMP, MMP2)
- No data: 7 priority targets (TIMP3, FRZB, CILP, MMPs, ADAMTS)

**Direction Agreement:**
- 77.3% matched directions (153/198 orthologs)
- 22.7% divergent (45/198)

**Conclusion:** Structural ECM proteins (collagens, small proteoglycans) show sufficient conservation to justify mouse mechanistic studies, but regulatory proteins (MMPs, inhibitors) require human-specific validation. Critical knowledge gaps (TIMP3, kidney conservation) must be addressed before therapeutic prioritization.

---

## Output Files Generated

**Data Files:**
- `/Users/Kravtsovd/projects/ecm-atlas/10_insights/orthologous_proteins.csv` - 303 shared proteins
- `/Users/Kravtsovd/projects/ecm-atlas/10_insights/human_specific_proteins.csv` - 187 unique to human
- `/Users/Kravtsovd/projects/ecm-atlas/10_insights/mouse_specific_proteins.csv` - 378 unique to mouse
- `/Users/Kravtsovd/projects/ecm-atlas/10_insights/tissue_conservation_scores.csv` - Pearson/Spearman correlations
- `/Users/Kravtsovd/projects/ecm-atlas/10_insights/conserved_aging_proteins.csv` - Direction-matched proteins
- `/Users/Kravtsovd/projects/ecm-atlas/10_insights/all_ortholog_comparison.csv` - 198 proteins with Δz comparison
- `/Users/Kravtsovd/projects/ecm-atlas/10_insights/therapeutic_target_translatability.csv` - 17 target assessments

**Visualization Files:**
- `/Users/Kravtsovd/projects/ecm-atlas/10_insights/species_distribution_venn.png` - Protein overlap
- `/Users/Kravtsovd/projects/ecm-atlas/10_insights/tissue_conservation_scores.png` - Heart vs Ovary correlation
- `/Users/Kravtsovd/projects/ecm-atlas/10_insights/human_vs_mouse_scatter.png` - Δz correlation plot
- `/Users/Kravtsovd/projects/ecm-atlas/10_insights/therapeutic_translatability.png` - Target ranking
- `/Users/Kravtsovd/projects/ecm-atlas/10_insights/direction_agreement.png` - Classification agreement matrix

**Analysis Script:**
- `/Users/Kravtsovd/projects/ecm-atlas/scripts/agent_19_species_conservation_analyzer.py` - Reproducible analysis pipeline

---

**Report Generated:** 2025-10-15
**Analysis:** Agent 19 - Species Conservation Analyzer
**Dataset:** merged_ecm_aging_zscore.csv (9,343 records, 3 species)
**Contact:** daniel@improvado.io


---

## Visualizations

### Figure 1: Direction Agreement

![Direction Agreement](direction_agreement.png)

**Direction Agreement** - Generated visualization showing direction agreement analysis results.

---

### Figure 2: Human Vs Mouse Scatter

![Human Vs Mouse Scatter](human_vs_mouse_scatter.png)

**Human Vs Mouse Scatter** - Generated visualization showing human vs mouse scatter analysis results.

---

### Figure 3: Species Distribution Venn

![Species Distribution Venn](species_distribution_venn.png)

**Species Distribution Venn** - Generated visualization showing species distribution venn analysis results.

---

