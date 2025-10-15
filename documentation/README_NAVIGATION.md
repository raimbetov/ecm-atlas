# ECM-Atlas Documentation Navigation Guide

## Document Structure Overview

This documentation follows **fractal knowledge framework** with 3 hierarchical levels:
- **Level 1 (Master):** High-level overview with links to detailed sections
- **Level 2 (Sections):** Deep dives into major topics
- **Level 3 (Details):** Implementation-level specifications

All documents follow **MECE + BFO + DRY principles** from [Knowledge Framework](../../chrome-extension-tcs/How to organize documents_knowladge_framework.md).

---

## Quick Start

**If you want to understand:**
1. **What is ECM-Atlas?** → Start with [00_ECM_ATLAS_MASTER_OVERVIEW.md](./00_ECM_ATLAS_MASTER_OVERVIEW.md)
2. **Why ECM aging matters?** → Read [01_Scientific_Foundation.md](./01_Scientific_Foundation.md)
3. **What did the team discover?** → Read [04_Research_Insights.md](./04_Research_Insights.md)
4. **How to build aging biomarker?** → Read [04a_Biomarker_Framework.md](./04a_Biomarker_Framework.md)

---

## Document Tree

```
documentation/
│
├── 00_ECM_ATLAS_MASTER_OVERVIEW.md        [LEVEL 1 - START HERE]
│   ├─> 1.0 Scientific Foundation
│   ├─> 2.0 Data Assets
│   ├─> 3.0 Technical Implementation
│   ├─> 4.0 Research Insights
│   └─> 5.0 Future Directions
│
├── 01_Scientific_Foundation.md            [LEVEL 2]
│   ├─> ECM biology & functions
│   ├─> Aging mechanisms (glycation, crosslinking)
│   ├─> Hallmarks hierarchy (ECM vs genomic instability)
│   ├─> Knowledge gap in literature
│   └─> Research objectives
│   │
│   └─> 01a_ECM_Aging_Mechanisms.md        [LEVEL 3 - To be created]
│   └─> 01b_Aging_Hallmarks_Hierarchy.md   [LEVEL 3 - To be created]
│
├── 02_Data_Assets_Catalog.md              [LEVEL 2 - To be created]
│   └─> 02a_Study_Summaries.md             [LEVEL 3 - To be created]
│   └─> 02b_Data_Schema_Specification.md   [LEVEL 3 - To be created]
│
├── 03_Technical_Architecture.md            [LEVEL 2 - To be created]
│   └─> 03a_Data_Processing_Pipeline.md    [LEVEL 3 - To be created]
│   └─> 03b_Visualization_Dashboards.md    [LEVEL 3 - To be created]
│
├── Technical_Issues_NaN_Serialization.md   [TECHNICAL DOC - COMPLETE]
│   └─> Bug report: Compare tab JSON serialization (Oct 2025)
│
├── 04_Research_Insights.md                 [LEVEL 2 - COMPLETE]
│   ├─> Biomarker evaluation framework
│   ├─> Key protein discovery hypothesis
│   ├─> Therapeutic intervention strategies
│   ├─> Aging hallmarks hierarchy
│   └─> Commercial potential
│   │
│   ├─> 04a_Biomarker_Framework.md         [LEVEL 3 - COMPLETE]
│   │   ├─> Statistical validation thresholds
│   │   ├─> Multi-modal integration
│   │   ├─> ECM aging log construction
│   │   ├─> Population stratification
│   │   └─> Implementation checklist
│   │
│   └─> 04b_Key_Protein_Candidates.md      [LEVEL 3 - To be created]
│   └─> 04c_Therapeutic_Strategies.md      [LEVEL 3 - To be created]
│
└── 05_Future_Directions.md                 [LEVEL 2 - To be created]
    └─> 05a_Publication_Strategy.md        [LEVEL 3 - To be created]
    └─> 05b_Commercialization_Plan.md      [LEVEL 3 - To be created]
    └─> 05c_Collaboration_Opportunities.md [LEVEL 3 - To be created]
```

---

## Completed Documents (2025-10-12)

✅ **Level 1:**
- [00_ECM_ATLAS_MASTER_OVERVIEW.md](./00_ECM_ATLAS_MASTER_OVERVIEW.md) - Master navigation document

✅ **Level 2:**
- [01_Scientific_Foundation.md](./01_Scientific_Foundation.md) - ECM aging biology and research rationale
- [04_Research_Insights.md](./04_Research_Insights.md) - Synthesis of team discussions (3 calls, 2025-10-12)

✅ **Level 3:**
- [04a_Biomarker_Framework.md](./04a_Biomarker_Framework.md) - Statistical methods for biomarker validation

✅ **Technical Issues:**
- [Technical_Issues_NaN_Serialization.md](./Technical_Issues_NaN_Serialization.md) - Bug report: Compare tab JSON serialization (Oct 2025)

---

## To Be Created (Next Steps)

**High Priority (Complete project structure):**
1. **02_Data_Assets_Catalog.md** - 13 studies inventory, file formats, metadata
2. **03_Technical_Architecture.md** - Code structure, processing pipeline, dashboards
3. **05_Future_Directions.md** - Roadmap, publication plan, commercialization

**Medium Priority (Deep technical details):**
4. **03a_Data_Processing_Pipeline.md** - Parsing algorithms, normalization, validation
5. **03b_Visualization_Dashboards.md** - Randles 2021 dashboard architecture, Plotly implementation

**Lower Priority (Extended analysis):**
6. **01a_ECM_Aging_Mechanisms.md** - Molecular biology deep dive (glycation chemistry, AGE formation)
7. **01b_Aging_Hallmarks_Hierarchy.md** - Theoretical framework, experimental validation
8. **04b_Key_Protein_Candidates.md** - Per-study protein rankings, meta-analysis results
9. **04c_Therapeutic_Strategies.md** - Enzyme engineering protocols, directed evolution
10. **05a_Publication_Strategy.md** - Journal targets, co-author coordination
11. **05b_Commercialization_Plan.md** - Market analysis, IP strategy
12. **05c_Collaboration_Opportunities.md** - Academic partnerships, industry contacts

---

## Document Standards (All Levels)

Every document includes:
1. **Thesis:** One-sentence summary
2. **Overview:** One paragraph previewing sections
3. **Mermaid diagram:** Visual structure (LR for flows, TD for hierarchies)
4. **Numbered sections (1.0, 2.0):** MECE organization
5. **Ordering principle (¶1):** States logic for section order
6. **Author checklist:** Validates framework compliance

---

## Source Materials

**Call transcripts (raw data):**
- [../calls_transcript/20251012_1035_gnr-avbr-ipz.md](../calls_transcript/20251012_1035_gnr-avbr-ipz.md) - Project kickoff (3.0 min)
- [../calls_transcript/20251012_1535_gnr-avbr-ipz.md](../calls_transcript/20251012_1535_gnr-avbr-ipz.md) - Dashboard development (3.0 min)
- [../calls_transcript/20251012_1845_gnr-avbr-ipz.md](../calls_transcript/20251012_1845_gnr-avbr-ipz.md) - Strategic discussion (0.8 min)

**Repository documentation:**
- [../00_REPO_OVERVIEW.md](../00_REPO_OVERVIEW.md) - Repository structure, datasets
- [../01_TASK_DATA_STANDARDIZATION.md](../01_TASK_DATA_STANDARDIZATION.md) - Parsing specifications
- [../CLAUDE.md](../CLAUDE.md) - Project overview for AI agents

---

## Key Insights from Completed Documentation

**From 04_Research_Insights.md:**
1. **Biomarker framework:** Requires longitudinal (r>0.7) + cross-sectional (d>1.0) + population (I²<50%) validation
2. **Key hypothesis:** 2-3 universal ECM proteins change across all 13 studies → therapeutic targets
3. **Therapeutic pathway:** Directed evolution of enzymes to cleave AGE-crosslinked collagen (50-60 year lifespan extension potential)
4. **Hallmarks hierarchy:** ECM dysfunction ranked Tier 1 alongside transposon activation, above genomic instability
5. **Commercial strategy:** Preprint publication + provisional patents + company formation around lead protein

**From 04a_Biomarker_Framework.md:**
1. **Statistical thresholds:** Pearson r>0.7 for longitudinal, Cohen's d>1.0 for cross-sectional, I²<50% for heterogeneity
2. **Multi-modal integration:** Combine ECM proteomics (0.35 weight) + epigenetics (0.25) + functional (0.20) + metabolomics (0.15) + anthropometric (0.05)
3. **Implementation:** UK Biobank plasma proteomics (N=50,000) + elastic net regression → ECM aging clock (MAE target <5 years)

---

## Contact & Contributions

**Project leads:**
- Daniel Kravtsov (daniel@improvado.io) - Technical lead, documentation architect
- Rakhan Aimbetov - Scientific lead, biology expert

**To contribute:**
1. Follow knowledge framework principles (MECE + BFO + DRY)
2. Link to parent documents (maintain hierarchy)
3. Include thesis, overview, Mermaid diagram, author checklist
4. Source all claims (call transcripts, papers, or "hypothesis")

---

**Created:** 2025-10-12
**Framework:** [Knowledge Framework](../../chrome-extension-tcs/How to organize documents_knowladge_framework.md)
**Version:** 1.0
