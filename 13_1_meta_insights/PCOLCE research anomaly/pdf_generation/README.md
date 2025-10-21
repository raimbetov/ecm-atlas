# PDF Generation Pipeline for PCOLCE Evidence Document

**Purpose:** Tools and resources for generating a professional PDF from the PCOLCE evidence document with fully rendered mermaid diagrams.

**Generated PDF:** [../PCOLCE_Evidence_Document_v1.1.pdf](../PCOLCE_Evidence_Document_v1.1.pdf) (304 KB, 22 pages)

---

## Contents

This folder contains the complete PDF generation pipeline:

### 📜 Scripts

1. **[render_diagrams.py](render_diagrams.py)** — Diagram renderer
   - Extracts mermaid code blocks from markdown
   - Renders to PNG using mermaid-cli
   - Outputs to `diagrams/` folder

2. **[generate_pdf_with_diagrams.py](generate_pdf_with_diagrams.py)** — PDF generator (CURRENT)
   - Converts markdown to HTML
   - Embeds rendered PNG diagrams (base64)
   - Generates PDF with WeasyPrint
   - **Use this for final PDF generation**

3. **[generate_pdf.py](generate_pdf.py)** — PDF generator (LEGACY)
   - Original version without diagram rendering
   - Creates placeholder text for diagrams
   - Kept for reference only

### 🖼️ Diagrams

**[diagrams/](diagrams/)** — Rendered mermaid diagrams (5 PNG files, 159 KB)
- diagram_1.png (46 KB) - Context-dependent model
- diagram_2.png (8 KB) - Processing flow
- diagram_3.png (11 KB) - Fibrosis pathway
- diagram_4.png (10 KB) - Aging pathway
- diagram_5.png (86 KB) - Therapeutic algorithm

### 📖 Documentation

4. **[PDF_README.md](PDF_README.md)** — PDF documentation
   - File specifications
   - Contents overview
   - Usage instructions
   - Regeneration guide

5. **[DIAGRAMS_README.md](DIAGRAMS_README.md)** — Diagram documentation
   - Individual diagram descriptions
   - Technical specifications
   - Rendering details

6. **[README.md](README.md)** — This file
   - Pipeline overview
   - Quick start guide

---

## Quick Start

### Generate PDF (Recommended Method)

```bash
cd "/home/raimbetov/GitHub/ecm-atlas/13_1_meta_insights/PCOLCE research anomaly"
source ../../env/bin/activate

# Option 1: Regenerate everything (if markdown changed)
python pdf_generation/render_diagrams.py
python pdf_generation/generate_pdf_with_diagrams.py

# Option 2: Just regenerate PDF (if only diagrams changed)
python pdf_generation/generate_pdf_with_diagrams.py
```

**Output:** [../PCOLCE_Evidence_Document_v1.1.pdf](../PCOLCE_Evidence_Document_v1.1.pdf)

### Render Only Diagrams

```bash
python pdf_generation/render_diagrams.py
```

**Output:** PNG files in `pdf_generation/diagrams/`

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  01_EVIDENCE_DOCUMENT_PCOLCE_CONTEXT_DEPENDENCY.md         │
│  (Source markdown with mermaid code blocks)                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ├─────────────────┐
                   │                 │
                   ▼                 ▼
┌──────────────────────────┐  ┌─────────────────────────┐
│  render_diagrams.py      │  │ generate_pdf_with_      │
│  ├─ Extract mermaid      │  │  diagrams.py            │
│  ├─ Render to PNG        │  │ ├─ Parse markdown       │
│  └─ Save to diagrams/    │  │ ├─ Embed PNG diagrams   │
└──────────┬───────────────┘  │ ├─ Convert to HTML      │
           │                  │ └─ Generate PDF         │
           │                  └─────────┬───────────────┘
           ▼                            │
┌──────────────────────────┐            │
│  diagrams/               │            │
│  ├─ diagram_1.png (46KB)│            │
│  ├─ diagram_2.png (8KB) │◄───────────┘
│  ├─ diagram_3.png (11KB)│
│  ├─ diagram_4.png (10KB)│
│  └─ diagram_5.png (86KB)│
└──────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  PCOLCE_Evidence_Document_v1.1.pdf  │
│  (304 KB, 22 pages, 5 diagrams)     │
└──────────────────────────────────────┘
```

---

## Dependencies

### Required Python Packages

```bash
# Install via pip
pip install markdown pymdown-extensions weasyprint pygments reportlab
```

### Required System Tools

```bash
# Mermaid-CLI for diagram rendering
npm install -g @mermaid-js/mermaid-cli

# Verify installation
mmdc --version  # Should show 11.12.0 or newer
```

### Installed and Working

✅ **All dependencies are already installed on this system:**
- Python 3.12
- markdown, weasyprint, pygments
- Node.js v18.19.1
- mermaid-cli v11.12.0

---

## File Specifications

### Input
- **Source:** [../01_EVIDENCE_DOCUMENT_PCOLCE_CONTEXT_DEPENDENCY.md](../01_EVIDENCE_DOCUMENT_PCOLCE_CONTEXT_DEPENDENCY.md)
- **Format:** Markdown with mermaid code blocks
- **Version:** v1.1 (corrected study IDs)

### Output
- **PDF:** [../PCOLCE_Evidence_Document_v1.1.pdf](../PCOLCE_Evidence_Document_v1.1.pdf)
- **Format:** PDF 1.7 (A4, 595×842 pts)
- **Size:** 304 KB (310,787 bytes)
- **Pages:** 22
- **Diagrams:** 5 PNG images embedded

---

## Workflow Details

### Step 1: Render Diagrams

**Script:** [render_diagrams.py](render_diagrams.py)

**Process:**
1. Read markdown source
2. Extract all ````mermaid` code blocks using regex
3. For each diagram:
   - Save mermaid code to temp .mmd file
   - Run `mmdc -i temp.mmd -o diagram_N.png -b white -w 1200`
   - Capture PNG output
   - Report size and dimensions
4. Save all diagrams to `diagrams/` folder

**Output:**
- 5 PNG files (diagram_1.png through diagram_5.png)
- Total size: 159 KB
- High-resolution (1200px width, auto height)
- White background (print-friendly)

### Step 2: Generate PDF

**Script:** [generate_pdf_with_diagrams.py](generate_pdf_with_diagrams.py)

**Process:**
1. Read markdown source
2. Find and replace ````mermaid` blocks with `<img>` tags
3. For each diagram:
   - Read PNG file from `diagrams/`
   - Encode as base64 data URI
   - Embed directly in HTML (self-contained)
4. Convert markdown to HTML using python-markdown
5. Apply professional CSS styling:
   - Typography: Georgia serif font
   - Colors: Blue theme (#2c5aa0)
   - Layout: A4 with margins, page numbers
   - Tables: Striped rows, bordered
   - Code blocks: Syntax highlighting
6. Convert HTML to PDF using WeasyPrint
7. Clean up temporary HTML file

**Output:**
- Self-contained PDF (no external dependencies)
- All diagrams embedded as base64 PNG
- Professional formatting

---

## Customization

### Modify Diagram Rendering

Edit `render_diagrams.py`:

```python
# Change diagram width (default: 1200px)
cmd = ['mmdc', '-i', mmd_file, '-o', output_path,
       '-b', 'white', '-w', '1600']  # ← Change here

# Change background color
cmd = ['mmdc', '-i', mmd_file, '-o', output_path,
       '-b', 'transparent']  # ← Or use transparent
```

### Modify PDF Styling

Edit `generate_pdf_with_diagrams.py`, CSS section:

```python
# Change colors
color: #2c5aa0;  # ← Heading color

# Change fonts
font-family: 'Georgia', 'Times New Roman', serif;  # ← Body font

# Change page size
@page {
    size: Letter;  # ← Or use US Letter instead of A4
    margin: 1in;
}
```

---

## Troubleshooting

### Diagrams Not Rendering

**Problem:** render_diagrams.py fails to create PNGs

**Solution:**
```bash
# Check mermaid-cli installation
which mmdc
mmdc --version

# Reinstall if needed
npm install -g @mermaid-js/mermaid-cli
```

### PDF Generation Fails

**Problem:** generate_pdf_with_diagrams.py crashes

**Solution:**
```bash
# Check WeasyPrint installation
python -c "from weasyprint import HTML; print('OK')"

# Reinstall if needed
pip install --upgrade weasyprint
```

### Diagrams Not Embedded

**Problem:** PDF shows diagram placeholders instead of images

**Solution:**
```bash
# Ensure diagrams exist
ls pdf_generation/diagrams/

# Regenerate diagrams
python pdf_generation/render_diagrams.py

# Then regenerate PDF
python pdf_generation/generate_pdf_with_diagrams.py
```

---

## Version History

- **2025-10-21 12:54** — v1.1 with rendered diagrams
  - Added render_diagrams.py
  - Added generate_pdf_with_diagrams.py
  - Rendered all 5 mermaid diagrams to PNG
  - Embedded diagrams in PDF (304 KB)

- **2025-10-21 12:45** — v1.0 initial PDF
  - Created generate_pdf.py
  - Basic PDF without rendered diagrams (158 KB)
  - Diagram placeholders only

---

## Folder Structure

```
pdf_generation/
├── README.md                           ← This file
├── PDF_README.md                       ← PDF documentation
├── DIAGRAMS_README.md                  ← Diagram documentation
│
├── render_diagrams.py                  ← Diagram renderer
├── generate_pdf_with_diagrams.py       ← PDF generator (CURRENT)
├── generate_pdf.py                     ← PDF generator (LEGACY)
│
└── diagrams/                           ← Rendered PNG diagrams
    ├── diagram_1.png (46 KB)
    ├── diagram_2.png (8 KB)
    ├── diagram_3.png (11 KB)
    ├── diagram_4.png (10 KB)
    └── diagram_5.png (86 KB)
```

---

## Related Files

**In parent directory:**
- [../01_EVIDENCE_DOCUMENT_PCOLCE_CONTEXT_DEPENDENCY.md](../01_EVIDENCE_DOCUMENT_PCOLCE_CONTEXT_DEPENDENCY.md) — Source markdown
- [../PCOLCE_Evidence_Document_v1.1.pdf](../PCOLCE_Evidence_Document_v1.1.pdf) — Generated PDF
- [../00_README.md](../00_README.md) — Main analysis overview

---

## Contact

**For PDF generation issues:**
- Check this README first
- Review [PDF_README.md](PDF_README.md) for PDF details
- Review [DIAGRAMS_README.md](DIAGRAMS_README.md) for diagram details
- Check script comments in .py files

**For content updates:**
- Edit source markdown: [../01_EVIDENCE_DOCUMENT_PCOLCE_CONTEXT_DEPENDENCY.md](../01_EVIDENCE_DOCUMENT_PCOLCE_CONTEXT_DEPENDENCY.md)
- Regenerate diagrams: `python render_diagrams.py`
- Regenerate PDF: `python generate_pdf_with_diagrams.py`

---

**Status:** ✅ Pipeline complete and tested
**Last Updated:** 2025-10-21 12:59
**PDF Version:** v1.1 (304 KB, 22 pages, 5 diagrams)
