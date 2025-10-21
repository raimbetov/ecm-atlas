# PCOLCE Evidence Document - PDF Version

**File:** [PCOLCE_Evidence_Document_v1.1.pdf](PCOLCE_Evidence_Document_v1.1.pdf)
**Version:** 1.1 (Corrected) with Rendered Diagrams
**Date Generated:** 2025-10-21 (Updated 12:54)
**File Size:** 304 KB
**Format:** PDF 1.7
**Pages:** 22

---

## Quick Info

✅ **Professional PDF ready for distribution to collaborators**

**Contains:**
- Complete evidence synthesis (22 pages)
- ✅ **5 mermaid diagrams (fully rendered as PNG images)**
- All tables properly formatted
- Professional styling with page numbers
- GRADE quality assessments
- Therapeutic recommendations
- Validation roadmap

---

## What's Inside

### Document Structure

1. **Evidence Grading Framework** (GRADE + Oxford CEBM)
2. **Aging Evidence** (Level 2a, ⊕⊕⊕○ MODERATE)
   - Study characteristics (12 observations, 7 studies)
   - Effect sizes and statistical analysis
   - Network integration evidence
   - Novelty scoring (8.4/10)
3. **Fibrosis Evidence** (Level 1b/4, ⊕⊕⊕○ MODERATE)
   - Literature synthesis
   - Mechanistic evidence
   - Functional validation (knockout mice)
4. **Context-Dependency Model** (Level 4, ⊕⊕○○ LOW)
   - Dual-context biological framework
   - Regulatory logic
   - Unified predictive framework
5. **Novelty and Impact Scoring**
6. **Therapeutic Implications** (Grade B/C recommendations)
7. **Validation Roadmap** (3-tier strategy)
8. **Executive Summary**
9. **Appendices** (GRADE profiles, abbreviations)

### Key Findings

**Main Discovery:**
- PCOLCE downregulation in aging: Δz=-1.41 (muscle Δz=-3.69)
- PCOLCE upregulation in fibrosis: Literature consistent
- Context-dependent regulation: Tissue and physiological state matter

**Evidence Quality:**
- Aging: ⊕⊕⊕○ MODERATE (Level 2a)
- Fibrosis: ⊕⊕⊕○ MODERATE (Level 1b/4)
- Novelty: 8.4/10 (High-impact publication tier)

**Therapeutic Recommendations:**
- Anti-fibrotic: Grade B (PCOLCE inhibition)
- Pro-aging: Grade C (indirect support via exercise/senolytics)
- Biomarker: Grade B (plasma PCOLCE + PCOLCE/PICP ratio)

---

## Diagrams Included

The PDF includes **5 fully rendered diagrams** (PNG images, 159 KB total):

1. **Context-Dependent Model** (hierarchical structure) — 46 KB, 566×766 px
2. **Processing Flow** (LR workflow diagram) — 8 KB, 1184×43 px
3. **Aging Context Pathway** (pathological fibrosis mechanism) — 11 KB, 1184×52 px
4. **Fibrosis Context Pathway** (physiological aging mechanism) — 10 KB, 1184×52 px
5. **Therapeutic Decision Algorithm** (patient stratification flowchart) — 86 KB, 1184×1423 px

✅ **All diagrams are now properly rendered as high-quality PNG images embedded in the PDF.**

---

## How to Regenerate

If you need to regenerate the PDF (e.g., after document updates):

```bash
cd "/home/raimbetov/GitHub/ecm-atlas/13_1_meta_insights/PCOLCE research anomaly"
source ../../env/bin/activate

# Step 1: Render diagrams (if mermaid code changed)
python render_diagrams.py

# Step 2: Generate PDF with rendered diagrams
python generate_pdf_with_diagrams.py
```

**Note:** Mermaid-cli is already installed and configured. Diagrams are automatically rendered to `diagrams/` folder.

---

## PDF Features

### Professional Formatting

- **Typography:** Georgia serif font for body text, Courier for code
- **Page Layout:** A4 size, 2.5cm margins
- **Headers/Footers:** Document title + page numbers
- **Color Scheme:** Professional blue (#2c5aa0) for headings
- **Tables:** Striped rows, clear borders, blue headers
- **Code Blocks:** Gray background with blue left border
- **Diagrams:** Centered with borders and padding

### Print-Ready

- Proper page breaks (avoid orphaned headings)
- Table pagination (avoid splitting mid-table where possible)
- High-quality text rendering
- Professional styling suitable for publication/grants

---

## Use Cases

**Send to:**
- ✅ Collaborators for review
- ✅ Grant reviewers (NIH R01, etc.)
- ✅ Journal editors (Nature Aging, Cell Metabolism)
- ✅ Conference presentations
- ✅ Internal lab meetings
- ✅ Thesis committees

**Appropriate for:**
- Scientific peer review
- Grant applications
- Preliminary publication submission
- Collaboration proposals
- Academic presentations

---

## Corrections Included

This PDF (v1.1) includes all corrections from 2025-10-21:

✅ **Table 2.3 corrected** with verified study IDs:
- Schuler_2021 (4 muscle tissues)
- Tam_2020 (3 disc compartments)
- Santinha_2024_Mouse_NT/DT (2 heart compartments)
- LiDermis_2021, Angelidis_2019, Dipali_2023

✅ **Spurious study IDs removed:**
- ❌ Baranyi_2020, Carlson_2019, Vogel_2021, Tabula_2020 (never existed)

✅ **Correction notice included** at document start

See [corrections_2025-10-21/](corrections_2025-10-21/) for full audit trail.

---

## Technical Details

**Generation Method:**
- Markdown → HTML → PDF pipeline
- Python-based (markdown + weasyprint libraries)
- Automatic mermaid diagram extraction
- Professional CSS styling
- A4 page layout with proper margins

**Dependencies:**
- Python 3.x
- markdown library
- weasyprint library
- pygments (syntax highlighting)
- mermaid-cli (optional, for diagram rendering)

**Source Files:**
- Markdown: [01_EVIDENCE_DOCUMENT_PCOLCE_CONTEXT_DEPENDENCY.md](01_EVIDENCE_DOCUMENT_PCOLCE_CONTEXT_DEPENDENCY.md)
- Generator: [generate_pdf.py](generate_pdf.py)

---

## File Information

| Property | Value |
|----------|-------|
| **Filename** | PCOLCE_Evidence_Document_v1.1.pdf |
| **Format** | PDF 1.7 |
| **Size** | 304 KB (311 KB on disk) |
| **Pages** | 22 pages |
| **Diagrams** | 5 PNG images (159 KB embedded) |
| **Generated** | 2025-10-21 12:54 |
| **Version** | 1.1 (corrected + diagrams) |
| **Status** | ✅ Ready for distribution |

---

## Version History

- **v1.1** (2025-10-21): Corrected version with verified study IDs (PDF generated)
- **v1.0** (2025-10-20): Initial version (contained study ID errors, no PDF)

---

## Contact

**For questions about:**
- Scientific content: daniel@improvado.io
- PDF generation: See [generate_pdf.py](generate_pdf.py) script
- Corrections: See [corrections_2025-10-21/](corrections_2025-10-21/) folder

---

**Generated:** 2025-10-21
**Ready for:** ✅ Collaborator distribution
