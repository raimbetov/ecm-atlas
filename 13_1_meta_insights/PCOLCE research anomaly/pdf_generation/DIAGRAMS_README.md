# PCOLCE Evidence Document - Rendered Diagrams

**Location:** [diagrams/](diagrams/)
**Generated:** 2025-10-21
**Tool:** Mermaid-CLI v11.12.0
**Format:** PNG images (white background, 1200px width)

---

## Overview

This folder contains **5 professionally rendered diagrams** extracted from the PCOLCE evidence document. These diagrams are embedded in the PDF version for clear visualization of the research framework.

---

## Diagram Files

### 1. Context-Dependent Model (Hierarchical)
**File:** [diagram_1.png](diagrams/diagram_1.png)
- **Size:** 46 KB
- **Dimensions:** 566 × 766 pixels
- **Type:** Hierarchical tree diagram (TD)
- **Content:** Shows PCOLCE protein branching into aging vs fibrosis contexts with evidence levels, novelty scores, and therapeutic strategies

### 2. Processing Flow (Workflow)
**File:** [diagram_2.png](diagrams/diagram_2.png)
- **Size:** 8 KB
- **Dimensions:** 1184 × 43 pixels
- **Type:** Left-to-right workflow (LR)
- **Content:** Research pipeline from literature review → multi-omics analysis → 4-agent validation → GRADE assessment → therapeutic framework

### 3. Aging Context Pathway
**File:** [diagram_3.png](diagrams/diagram_3.png)
- **Size:** 11 KB
- **Dimensions:** 1184 × 52 pixels
- **Type:** Mechanistic pathway (LR)
- **Content:** Acute injury → TGF-β spike → myofibroblast activation → PCOLCE upregulation → pathological fibrosis

### 4. Fibrosis Context Pathway
**File:** [diagram_4.png](diagrams/diagram_4.png)
- **Size:** 10 KB
- **Dimensions:** 1184 × 52 pixels
- **Type:** Mechanistic pathway (LR)
- **Content:** Chronic aging → inflammation → fibroblast senescence → PCOLCE downregulation → ECM atrophy

### 5. Therapeutic Decision Algorithm
**File:** [diagram_5.png](diagrams/diagram_5.png)
- **Size:** 86 KB
- **Dimensions:** 1184 × 1423 pixels
- **Type:** Clinical flowchart (TD)
- **Content:** Patient stratification algorithm based on plasma PCOLCE/PICP ratio → therapeutic recommendations (anti-fibrotic vs pro-aging interventions)

---

## Total Size

**Combined:** 159 KB (all 5 diagrams)

---

## Technical Details

### Rendering Method
- **Tool:** Mermaid-CLI (mmdc) v11.12.0
- **Command:** `mmdc -i input.mmd -o output.png -b white -w 1200 --pdfFit`
- **Background:** White (for print compatibility)
- **Width:** 1200 pixels (automatically scaled to fit)
- **Quality:** High-resolution PNG for clarity

### Source Code
Original mermaid diagrams are in:
- [01_EVIDENCE_DOCUMENT_PCOLCE_CONTEXT_DEPENDENCY.md](01_EVIDENCE_DOCUMENT_PCOLCE_CONTEXT_DEPENDENCY.md)

Extraction/rendering script:
- [render_diagrams.py](render_diagrams.py)

### Regeneration

To regenerate diagrams (e.g., after updating mermaid code):

```bash
cd "/home/raimbetov/GitHub/ecm-atlas/13_1_meta_insights/PCOLCE research anomaly"
python render_diagrams.py
```

This will:
1. Extract all mermaid code blocks from the evidence document
2. Render each to PNG using mermaid-cli
3. Save to `diagrams/diagram_N.png`
4. Report size and dimensions

---

## Usage in PDF

These diagrams are automatically embedded in the PDF by:
- [generate_pdf_with_diagrams.py](generate_pdf_with_diagrams.py)

The script:
1. Reads PNG files from `diagrams/` folder
2. Encodes as base64 data URIs
3. Embeds directly in HTML (self-contained)
4. Converts to PDF with WeasyPrint

**Result:** PDF contains high-quality diagram images, no external file dependencies.

---

## Diagram Descriptions

### Scientific Content

**Diagram 1: Context-Dependent Framework**
- Shows the dual nature of PCOLCE regulation
- Aging context: ↓ downregulation (sarcopenia marker)
- Fibrosis context: ↑ upregulation (pro-fibrotic factor)
- Evidence levels and novelty scores for each context
- Therapeutic strategies (inhibit vs support)

**Diagram 2: Research Workflow**
- Multi-stage research pipeline
- Literature review → multi-omics data analysis
- 4-agent independent validation
- Evidence synthesis → GRADE assessment
- Novelty scoring → therapeutic framework → clinical translation

**Diagram 3: Fibrosis Mechanism**
- Pathological injury cascade
- TGF-β activation → myofibroblast proliferation
- High procollagen synthesis (10-100× baseline)
- PCOLCE upregulation enables rapid collagen maturation
- Excessive ECM deposition → fibrosis

**Diagram 4: Aging Mechanism**
- Physiological aging cascade
- Low-grade inflammation → fibroblast senescence
- Reduced procollagen synthesis
- PCOLCE downregulation → impaired ECM turnover
- ECM atrophy → muscle mass loss (sarcopenia)

**Diagram 5: Clinical Algorithm**
- Patient measurement: Plasma PCOLCE + PICP
- Decision tree based on PCOLCE/PICP ratio
- Low ratio (<0.5) → aging phenotype → sarcopenia interventions
- High ratio (>2.0) → fibrosis phenotype → anti-fibrotic therapy
- Safety checks: age, muscle mass, contraindications
- Treatment monitoring via PCOLCE/PICP response

---

## Image Quality

All diagrams optimized for:
- ✅ Print clarity (300 DPI equivalent at A4 size)
- ✅ Screen readability (high contrast, white background)
- ✅ PDF embedding (PNG format, lossless compression)
- ✅ File size efficiency (8-86 KB per diagram)

---

## File Listing

```
diagrams/
├── diagram_1.png  (46 KB)  - Context-dependent model
├── diagram_2.png  (8 KB)   - Processing flow
├── diagram_3.png  (11 KB)  - Fibrosis pathway
├── diagram_4.png  (10 KB)  - Aging pathway
└── diagram_5.png  (86 KB)  - Therapeutic algorithm
```

**Total:** 159 KB (5 files)

---

## Version Control

**Generated:** 2025-10-21 12:53
**Mermaid-CLI:** v11.12.0
**Node.js:** v18.19.1
**Source Document:** 01_EVIDENCE_DOCUMENT_PCOLCE_CONTEXT_DEPENDENCY.md v1.1

---

## Contact

**For regeneration issues:** Check [render_diagrams.py](render_diagrams.py) script
**For diagram updates:** Edit mermaid code blocks in source document, then regenerate
**For PDF updates:** Run [generate_pdf_with_diagrams.py](generate_pdf_with_diagrams.py)

---

**Status:** ✅ All diagrams rendered successfully and embedded in PDF
