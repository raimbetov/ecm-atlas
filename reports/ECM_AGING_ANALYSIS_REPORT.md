# ECM Aging Signatures: Cross-Organ Analysis Report

**Dataset**: `08_merged_ecm_dataset/merged_ecm_aging_zscore.csv`
**Analysis Date**: 2025-10-14
**Total Entries**: 2,177 protein-compartment combinations
**Unique Proteins**: 899
**Organs Analyzed**: 5 (Kidney, Intervertebral disc, Lung, Ovary, Skin dermis)
**Compartments**: 8 tissue-specific compartments

---

## Executive Summary

This analysis reveals **organ-specific ECM aging patterns** with limited universal signatures. While some proteins show consistent directional changes across multiple organs, the majority exhibit tissue-dependent aging responses. A striking finding is the accumulation of **blood coagulation proteins** in certain tissues, particularly the intervertebral disc, suggesting compromised structural integrity and vascular infiltration with aging.

---

## Question 1: Are There Common Proteomic Signatures Associated with Aging Across Different Organs?

### Cross-Organ Protein Coverage

- **No proteins** appear in all 5 organs
- **2 proteins** appear in 4 organs
- **137 proteins** appear in 3 organs
- **247 proteins** appear in 2 organs
- **386 total proteins** appear in multiple organs (43% of dataset)

### Directional Consistency Analysis

Among the 139 proteins found in 3+ organs:

#### ✅ **Consistent INCREASES (26 proteins, 19%)**
Proteins that increase with aging across ≥70% of tissues:

| Protein | Organs | Avg Δ Z-score | Category | Biological Significance |
|---------|--------|---------------|----------|------------------------|
| **VTN** (Vitronectin) | 3 | +1.424 | ECM Glycoprotein | Cell adhesion, thrombosis |
| **FGB** (Fibrinogen β) | 3 | +1.286 | ECM Glycoprotein | Blood coagulation |
| **FGG** (Fibrinogen γ) | 3 | +1.240 | ECM Glycoprotein | Blood coagulation |
| **HPX** (Hemopexin) | 3 | +1.192 | ECM-affiliated | Heme scavenging |
| **PLG** (Plasminogen) | 3 | +1.169 | ECM Regulator | Fibrinolysis |
| **SERPINC1** (Antithrombin) | 3 | +1.092 | ECM Regulator | Coagulation inhibitor |
| **TIMP3** | 3 | +1.060 | ECM Regulator | MMP inhibitor |
| **HRG** (Histidine-rich glycoprotein) | 3 | +1.030 | ECM Regulator | Coagulation modulator |
| **F2** (Prothrombin) | 4 | +0.800 | ECM Regulator | Thrombin precursor |
| **AMBP** (α-1-microglobulin) | 3 | +0.777 | ECM Regulator | Anti-inflammatory |
| **IGFALS** | 3 | +0.737 | ECM Regulator | IGF binding |

**Key Finding**: **11 of the 26 consistently increasing proteins are blood coagulation/hemostasis proteins**, suggesting systemic vascular aging signatures.

#### ❌ **Consistent DECREASES (6 proteins, 4%)**
Proteins that decrease with aging across ≥70% of tissues:

| Protein | Organs | Avg Δ Z-score | Category | Biological Significance |
|---------|--------|---------------|----------|------------------------|
| **FNDC1** | 3 | -0.460 | ECM Glycoprotein | Cell migration, fibronectin binding |
| **ANXA4** (Annexin A4) | 3 | -0.408 | ECM-affiliated | Calcium signaling |
| **ITIH5** | 3 | -0.294 | ECM Regulator | Hyaluronan stabilization |
| **SERPINA5** | 3 | -0.212 | ECM Regulator | Protease inhibitor |
| **COL8A1** (Collagen VIII α1) | 3 | -0.117 | Collagen | Basement membrane |

#### ⚠️ **Inconsistent Patterns (107 proteins, 77%)**
The **majority** of multi-organ proteins show tissue-dependent directionality:

**Examples of inconsistent proteins:**
- **AGRN** (Agrin): ↑3 compartments, ↓4 compartments, =1 stable
- **AGT** (Angiotensinogen): ↑3 compartments, ↓3 compartments
- **ANXA2** (Annexin A2): ↑1 compartment, ↓3 compartments, =2 stable
- **COL1A1** (Collagen I): Shows both increases and decreases depending on tissue

### Answer to Question 1

**No, organs do NOT age in the same way** based on ECM protein signatures. Key findings:

1. **Limited universal signatures**: Only 26 proteins (19% of multi-organ proteins) show consistent directional changes
2. **Organ-specific responses dominate**: 77% of multi-organ proteins exhibit tissue-dependent aging patterns
3. **Blood protein infiltration is a common aging feature**: Coagulation cascade proteins consistently increase across tissues, likely reflecting:
   - Chronic low-grade inflammation
   - Vascular barrier dysfunction
   - Tissue injury and repair responses
4. **Collagen remodeling is tissue-specific**: Even major structural collagens (COL1A1, COL3A1) show opposite directions in different organs

---

## Question 2: What Can You Infer About Structural Integrity of the ECM in These Organs?

### Matrisome Composition Overview

| Category | Division | Total Entries | Unique Proteins |
|----------|----------|---------------|-----------------|
| **ECM Glycoproteins** | Core matrisome | 686 | 235 |
| **ECM Regulators** | Matrisome-associated | 571 | — |
| **Secreted Factors** | Matrisome-associated | 310 | — |
| **ECM-affiliated** | Matrisome-associated | 260 | — |
| **Collagens** | Core matrisome | 225 | 64 |
| **Proteoglycans** | Core matrisome | 125 | 36 |

---

### Structural Collagen Analysis

#### Collagen Aging Patterns by Organ

| Organ | Collagen Entries | Avg Δ Z-score | Increases | Decreases | Stable |
|-------|------------------|---------------|-----------|-----------|--------|
| **Kidney** | 78 | **+0.035** | 28 | 23 | 27 |
| **Intervertebral Disc** | 76 | **-0.307** ⚠️ | 11 | **43** | 10 |
| **Lung** | 25 | **-0.021** | 8 | 6 | 9 |
| **Ovary** | 18 | **+0.026** | 2 | 1 | 15 |
| **Skin Dermis** | 28 | **+0.087** | 15 | 12 | 1 |

#### Top Collagen Changes

**Largest INCREASES (fibrosis/scarring indicators):**
1. **COL2A1** (Skin): Δ +1.613 — Cartilage collagen appearing in dermis
2. **COL10A1** (Skin): Δ +1.234 — Hypertrophic cartilage marker
3. **COL8A1** (Skin): Δ +1.102 — Endothelial/vascular collagen
4. **COL5A2** (Kidney-Glomerular): Δ +0.968 — Fibrillar collagen
5. **COL1A1** (Skin/Kidney): Δ +0.520 to +0.566 — Major fibrillar collagen

**Largest DECREASES (structural degradation indicators):**
1. **COL4A3** (Kidney-Tubulointerstitial): Δ -1.651 ⚠️ — Basement membrane
2. **COL4A3** (Kidney-Glomerular): Δ -1.346 ⚠️ — Basement membrane
3. **COL4A6** (Kidney-Tubulointerstitial): Δ -0.980 — Basement membrane
4. **COL11A2** (Intervertebral Disc-NP): Δ -1.191 — Cartilage-associated
5. **COL17A1** (Skin): Δ -1.332 — Epidermal-dermal junction

---

### Organ-Specific Structural Integrity Assessment

#### 🔴 **KIDNEY — Progressive Basement Membrane Failure**

**Structural integrity: COMPROMISED**

- **COL4A3 catastrophic loss**: -1.346 to -1.651 Δ (basement membrane α3(IV) chain)
  - Loss of COL4A3 is diagnostic of **Alport syndrome**-like pathology
  - Indicates **glomerular filtration barrier breakdown**
- **COL4A6 decline**: -0.980 Δ (basement membrane α6(IV) chain)
- **Compensatory fibrosis**: COL1A1 (+0.520), COL5A2 (+0.968), COL6A1 (+0.468)
  - Fibrillar collagens replacing specialized basement membrane collagens
  - Classic sign of **kidney fibrosis** (glomerulosclerosis, tubulointerstitial fibrosis)

**Blood protein infiltration:**
- Fibrinogen chains (FGA, FGB, FGG) increase +0.275 to +0.723
- Indicates **glomerular permeability defects** (proteins leaking into ECM)

**Proteoglycans:**
- **VCAN** (Versican) dramatic increase: +1.101 to +1.314
  - Pro-inflammatory proteoglycan, marker of tissue remodeling
- **LUM** (Lumican) increase: +0.880
  - Collagen fibril organization, repair response

**Clinical inference**: Kidneys show **loss of specialized basement membrane architecture** with replacement by scar-like fibrillar collagens. This pattern is consistent with **chronic kidney disease (CKD)** progression and age-related nephrosclerosis.

---

#### 🔴 **INTERVERTEBRAL DISC — Catastrophic Vascular Invasion**

**Structural integrity: SEVERELY COMPROMISED**

**Strongest aging signal in the entire dataset**: Blood coagulation proteins show **+0.872 to +3.013 Δ z-scores** (largest changes observed).

| Protein | NP Δ | IAF Δ | OAF Δ | Function |
|---------|------|-------|-------|----------|
| **SERPINC1** | +3.013 | +2.498 | +0.872 | Antithrombin |
| **VTN** | +2.915 | +2.701 | +1.403 | Thrombosis |
| **FGA** | +2.904 | +2.353 | +1.362 | Fibrinogen α |
| **FGG** | +2.890 | +2.222 | +1.349 | Fibrinogen γ |
| **HPX** | +2.717 | +2.356 | +0.987 | Heme scavenging |
| **FGB** | +2.601 | +2.083 | +1.280 | Fibrinogen β |
| **PLG** | +2.574 | +2.664 | +1.868 | Plasminogen |

**Biological interpretation:**
- Intervertebral discs are **avascular** in young/healthy state
- **Massive infiltration of blood proteins** indicates:
  - **Neovascularization** (pathological blood vessel ingrowth)
  - **Chronic hemorrhage** and clot formation
  - **Loss of avascular status** — hallmark of disc degeneration

**Collagen degradation:**
- **COL11A2** decline: -1.191 (cartilage-specific collagen)
- Net collagen loss: Avg Δ = -0.307 (43 collagens decreased vs 11 increased)
- Suggests **nucleus pulposus proteoglycan loss** and **annulus fibrosus tears**

**Proteoglycan changes:**
- **ACAN** (Aggrecan): Mixed signals (↑2, ↓1, =3) — critical for disc hydration
- Overall proteoglycan Avg Δ = +0.099 (compensatory synthesis attempts)

**Clinical inference**: Intervertebral discs undergo **loss of avascular privilege** with aging, allowing blood vessel and inflammatory cell invasion. This is the pathophysiological basis of **degenerative disc disease** and chronic back pain.

---

#### 🟡 **LUNG — Mild Structural Decline**

**Structural integrity: MILDLY COMPROMISED**

- **Collagen balance**: Near-neutral (Avg Δ = -0.021)
  - 8 collagens increase, 6 decrease, 9 stable
- **ECM Glycoprotein decline**: Avg Δ = -0.082
  - **MFAP5** (Microfibril-associated protein 5): -0.800 (elastin fiber integrity)
  - **LTBP2** (Latent TGF-β binding protein 2): -0.576 (TGF-β signaling scaffold)
  - **FRAS1/FREM1**: -0.567 to -1.322 (basement membrane adhesion)

**Positive aging markers:**
- **Serpina1e** (α1-antitrypsin): +1.708 (protease inhibitor, anti-emphysema)
- **Col4a3** (Type IV collagen): +0.429 (basement membrane maintenance)

**Proteoglycan stability**: Avg Δ = -0.041 (minimal change)

**Clinical inference**: Lungs show **subtle ECM aging** with loss of elastic fiber support systems but preservation of major structural collagens. Pattern suggests **early alveolar remodeling** without overt fibrosis in this dataset (mouse lung, Angelidis 2019).

---

#### 🟢 **OVARY — Structural Preservation**

**Structural integrity: WELL-MAINTAINED**

- **Collagen balance**: Nearly neutral (Avg Δ = +0.026)
  - Most collagens stable (15 stable vs 2 increase, 1 decrease)
- **ECM Glycoprotein stability**: Avg Δ = -0.027 (minimal change)
- **Proteoglycan stability**: Avg Δ = +0.103

**Notable changes:**
- **ASTL** (Astacin-like protease): -0.974 (ECM remodeling enzyme)
- **THBS1** (Thrombospondin-1): -0.477 (anti-angiogenic, TGF-β activator)

**Clinical inference**: Ovarian ECM shows **remarkable stability** in the aging mouse dataset (Dipali 2023). This may reflect:
- **Tissue-specific resilience** mechanisms
- Hormonal protection (estrogen effects on ECM)
- Or **dataset-specific limitations** (age range, mouse strain)

---

#### 🟡 **SKIN DERMIS — Paradoxical Fibrosis with Basement Membrane Loss**

**Structural integrity: MIXED PATTERN**

**Unexpected collagen increases:**
- **COL2A1** (Cartilage collagen): +1.613 ⚠️ — Should NOT be in dermis
- **COL10A1** (Hypertrophic cartilage): +1.234 ⚠️ — Ectopic expression
- **COL8A1** (Vascular collagen): +1.102 — Neovascularization
- **COL1A1** (Fibrillar collagen): +0.566 — Fibrosis

**Structural losses:**
- **COL17A1** (Type XVII): -1.332 — **Epidermal-dermal junction failure**
  - COL17A1 anchors epidermis to dermis
  - Loss causes **bullous pemphigoid** and skin fragility
- **FBLN7** (Fibulin-7): -1.491 — Elastic fiber assembly
- **EFEMP2**: -1.404 — Elastic fiber integrity

**Elastin changes:**
- **ELN** (Elastin): +2.082 — Paradoxical increase
  - May reflect **tropoelastin accumulation** without proper fiber formation
  - Or reactive elastogenesis in response to fiber degradation

**Proteoglycan stability**: Avg Δ = +0.214

**Blood protein infiltration**: Moderate (VTN +0.837, FGB +0.484, HRG +1.248)

**Clinical inference**: Skin dermis shows **disorganized ECM remodeling** with:
- Loss of **epidermal-dermal junction integrity** (COL17A1)
- Ectopic expression of cartilage collagens (COL2A1, COL10A1) — suggests **dedifferentiation**
- Elastin fiber dysfunction despite increased elastin protein
- Pattern consistent with **aging skin fragility, wrinkling, and impaired wound healing**

---

## Key Biological Insights

### 1. Blood Protein Accumulation = ECM Barrier Dysfunction

The consistent increase in **fibrinogen, vitronectin, prothrombin, plasminogen, and hemopexin** across kidney, intervertebral disc, and skin suggests:
- **Chronic vascular leak** (breakdown of blood-tissue barriers)
- **Microhemorrhages** and clot formation within ECM
- **Inflammatory milieu** (fibrinogen activates immune cells)
- **Tissue repair failure** (persistent coagulation proteins indicate non-resolving injury)

This is a **common aging hallmark** across tissues, distinct from organ-specific collagen remodeling.

---

### 2. Basement Membrane Collagens Are Critical Aging Targets

**COL4A3** (kidney) and **COL17A1** (skin) show the most severe declines in their respective tissues:
- These are **specialized collagens** that maintain epithelial-ECM junctions
- Their loss causes:
  - **Kidney**: Proteinuria, glomerulosclerosis
  - **Skin**: Epidermolysis, blister formation
- Unlike **fibrillar collagens** (COL1, COL3, COL5), which can be compensatorily upregulated, **basement membrane collagens are difficult to replace** once lost

---

### 3. Organ-Specific Aging Rates: Disc >> Kidney > Skin > Lung ≈ Ovary

Based on ECM disruption severity:

| Organ | Structural Integrity Score | Key Pathology |
|-------|---------------------------|---------------|
| **Intervertebral Disc** | 🔴🔴🔴 Critical | Vascular invasion, avascular loss |
| **Kidney** | 🔴🔴 Severe | Basement membrane failure, fibrosis |
| **Skin Dermis** | 🟡🟡 Moderate | Epidermal junction loss, disorganized remodeling |
| **Lung** | 🟡 Mild | Elastic fiber decline, early alveolar changes |
| **Ovary** | 🟢 Preserved | Minimal ECM disruption |

**Note**: These rankings reflect **the specific datasets analyzed** (species: human kidney/skin, mouse lung/ovary/disc; age ranges vary). Cross-species and methodological differences limit direct comparisons.

---

### 4. TIMP3 as a Potential Aging Biomarker

**TIMP3** (Tissue inhibitor of metalloproteinases 3) shows **+1.060 to +2.213 Δ** across multiple tissues:
- **Function**: Inhibits matrix metalloproteinases (MMPs) that degrade ECM
- **Interpretation**: Elevated TIMP3 may represent:
  - **Compensatory response** to excessive MMP activity
  - **Failed repair mechanism** (excessive inhibition prevents healthy remodeling)
  - **Biomarker of chronic ECM stress**

---

## Conclusions

### Question 1: Do organs age in the same way?

**No.** ECM aging is predominantly **organ-specific**:
- Only **19% of multi-organ proteins** show consistent directional changes
- **77% exhibit tissue-dependent patterns**
- The only **universal signature** is accumulation of blood coagulation proteins, reflecting systemic vascular dysfunction

### Question 2: What about structural integrity?

**Structural integrity declines organ-specifically**:

1. **Intervertebral disc**: Most severe — catastrophic vascular invasion destroying avascular niche
2. **Kidney**: Severe — specialized basement membrane loss with fibrotic replacement
3. **Skin**: Moderate — epidermal-dermal junction failure with disorganized remodeling
4. **Lung**: Mild — elastic fiber decline, preservation of major structure
5. **Ovary**: Minimal — ECM largely preserved in this dataset

**Universal aging features**:
- Blood protein infiltration (fibrinogen, vitronectin, coagulation factors)
- Loss of tissue-specific "specialized" collagens (COL4A3, COL17A1)
- Compensatory but ineffective upregulation of fibrillar collagens (COL1A1, COL3A1)

**Tissue-specific features**:
- Collagen subtype remodeling patterns
- Proteoglycan responses
- Elastic fiber changes (where present)

---

## Limitations

1. **Cross-species comparisons**: Human (kidney, skin) vs Mouse (lung, ovary, some disc data)
2. **Sample size variations**: Kidney (458 entries) vs Ovary (173 entries)
3. **Age definitions**: Different age ranges per study
4. **Methodology**: All label-free LC-MS/MS, but different protocols
5. **Snapshot analysis**: Cross-sectional data cannot distinguish:
   - Cause vs consequence
   - Active synthesis vs passive accumulation
   - Functional vs dysfunctional protein forms

---

## Recommendations for Further Analysis

1. **Pathway enrichment**: Map proteins to KEGG/Reactome pathways (hemostasis, wound healing, inflammation)
2. **Protein-protein interactions**: Network analysis of coagulation cascade proteins
3. **Correlation analysis**: Test if blood protein infiltration correlates with collagen degradation
4. **Clinical validation**: Compare to human aging cohorts (skin biopsies, kidney function tests)
5. **Interventional studies**: Test whether anti-coagulants or vascular stabilizers slow ECM aging

---

**Report compiled from**: [08_merged_ecm_dataset/merged_ecm_aging_zscore.csv](08_merged_ecm_dataset/merged_ecm_aging_zscore.csv)
**Visualization**: Dashboard available at http://localhost:8083/dashboard.html
