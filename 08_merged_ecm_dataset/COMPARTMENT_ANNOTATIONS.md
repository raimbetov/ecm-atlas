# Compartment Annotations - ECM Atlas

This document describes the anatomical compartments analyzed in each organ/tissue.

## Dataset Statistics

- **Total ECM proteins**: 1,355 unique proteins
- **Total entries**: 4,541 (ECM-only)
- **Organs**: 8
- **Compartments**: 16

---

## Kidney (Randles_2021)
- **Total ECM proteins**: 229
- **Study**: Randles et al., 2021

### Compartments:

1. **Glomerular** (229 proteins)
   - Glomerular basement membrane (filtration barrier)
   - Key for blood filtration and protein retention
   - Critical in diabetic nephropathy and glomerulosclerosis

2. **Tubulointerstitial** (229 proteins)
   - Tubular and interstitial ECM
   - Peritubular capillaries and tubular basement membranes
   - Key in tubulointerstitial fibrosis

---

## Intervertebral Disc (Tam_2020)
- **Total ECM proteins**: 410
- **Study**: Tam et al., 2020

### Compartments:

1. **NP - Nucleus Pulposus** (300 proteins)
   - Gel-like core of the disc
   - High water content, proteoglycan-rich
   - Key for load distribution and shock absorption

2. **IAF - Inner Annulus Fibrosus** (317 proteins)
   - Inner ring of fibrocartilage
   - Transition zone between NP and OAF
   - Mixed collagen types

3. **OAF - Outer Annulus Fibrosus** (376 proteins)
   - Outer ring of dense fibrous tissue
   - High Type I collagen content
   - Key for mechanical stability

---

## Heart
- **Studies**: Santinha et al., 2024 (Human + Mouse)

### Compartments:

1. **Native_Tissue** (398 proteins)
   - Native cardiac ECM
   - Includes basement membranes and interstitial matrix
   - Studies: Santinha_2024_Human, Santinha_2024_Mouse_NT

2. **Decellularized_Tissue** (155 proteins)
   - Decellularized cardiac scaffold
   - Pure ECM without cellular components
   - Study: Santinha_2024_Mouse_DT

---

## Skeletal Muscle (Schuler_2021)
- **Total ECM proteins**: 1,290 entries (across 4 muscle types)
- **Study**: Schuler et al., 2021

### Compartments by Fiber Type:

1. **EDL - Extensor Digitorum Longus** (278 proteins)
   - Fast-twitch glycolytic fibers (Type IIb)
   - High force, low endurance

2. **Gastrocnemius** (315 proteins)
   - Mixed fiber composition (Type I, IIa, IIb)
   - Combined endurance and power

3. **Soleus** (364 proteins)
   - Slow-twitch oxidative fibers (Type I)
   - High endurance, postural support

4. **TA - Tibialis Anterior** (333 proteins)
   - Fast-twitch fibers (Type IIa/IIb)
   - Dorsiflexion, high power output

---

## Brain (Tsumagari_2023)
- **Total ECM proteins**: 224
- **Study**: Tsumagari et al., 2023

### Compartments:

1. **Cortex** (209 proteins)
   - Cortical ECM
   - Perineuronal nets and basement membranes
   - Key in synaptic plasticity

2. **Hippocampus** (214 proteins)
   - Hippocampal ECM
   - Critical for memory formation
   - Age-related changes in neuroplasticity

---

## Ovary
- **Total ECM proteins**: 271 entries (2 studies)

### Compartments:

1. **Cortex** (98 proteins)
   - Ovarian cortical ECM
   - Study: Ouni_2022
   - Contains follicles at various stages

2. **Ovary** (173 proteins)
   - Whole ovarian ECM
   - Study: Dipali_2023

---

## Lung (Angelidis_2019)
- **Total ECM proteins**: 291
- **Study**: Angelidis et al., 2019

### Compartments:

1. **Lung** (291 proteins)
   - Pulmonary ECM
   - Alveolar and bronchial basement membranes
   - Critical in pulmonary fibrosis and COPD

---

## Skin Dermis (LiDermis_2021)
- **Total ECM proteins**: 262
- **Study**: Li et al., 2021

### Compartments:

1. **Skin dermis** (262 proteins)
   - Dermal ECM
   - Papillary and reticular dermis
   - Key in skin aging and wound healing

---

## ECM Categories Distribution

| Category | Entries | Description |
|----------|---------|-------------|
| **ECM Glycoproteins** | 1,149 | Fibronectin, laminins, tenascins |
| **ECM Regulators** | 1,019 | MMPs, TIMPs, LOX family |
| **Non-ECM** | 826 | ECM-interacting proteins |
| **ECM-affiliated Proteins** | 488 | Annexins, galectins |
| **Secreted Factors** | 446 | Growth factors, cytokines |
| **Collagens** | 378 | 28 collagen types |
| **Proteoglycans** | 235 | Versican, aggrecan, perlecans |

---

## Usage

The ECM-only dataset is now the default for the dashboard:
- **File**: `merged_ecm_aging_zscore_ECM_ONLY.csv`
- **Dashboard**: http://localhost:8083/dashboard.html
- **API**: http://localhost:5004/api/datasets

To switch back to the full dataset (including non-ECM proteins), remove the `.env` file.

---

**Last updated**: 2025-10-18  
**Contact**: daniel@improvado.io
