# H21: Browser Automation for External Data Acquisition

## 🎯 Hypothesis
**Automated browser automation (Playwright/Selenium) can programmatically download supplementary files from journal websites (eLife, PMC, Nature, Cell) to enable H16 External Validation completion.**

## 🚨 Problem Statement

**Current Blocker for H16 (External Validation):**
- ✅ Framework implemented (635 lines code)
- ✅ 6 datasets identified (PXD011967, PXD015982, +4)
- ✅ Validation pipeline ready
- ❌ **BLOCKED:** Cannot download supplementary files programmatically

**Root Cause:**
1. PRIDE/MassIVE only have RAW MS files (hundreds of GB, require full proteomics pipeline)
2. Journal websites (eLife, PMC) have NO API for supplementary files
3. Supplementary Excel/CSV files only accessible via browser (JavaScript required)

**What We Tried (ALL FAILED):**
```python
# ❌ Attempt 1: eLife API
GET https://api.elifesciences.org/articles/49874
Result: 404 Not Found

# ❌ Attempt 2: Europe PMC API
GET https://europepmc.org/webservices/rest/search?query=PMID:31845887
Result: No download URLs in JSON

# ❌ Attempt 3: Direct CDN URL
GET https://cdn.elifesciences.org/articles/49874/elife-49874-supp1-v2.xlsx
Result: 404 Not Found

# ❌ Attempt 4: WebFetch (HTML scraping)
WebFetch('https://elifesciences.org/articles/49874', 'Find download links')
Result: No explicit download links (requires JavaScript interaction)
```

---

## 🎯 H21 Mission

**MANDATORY OBJECTIVES:**

1. **Implement Playwright-based downloader** for 6 journal sources
   - eLife, PMC, Nature, Cell, bioRxiv, Wiley
   - Handle JavaScript-rendered pages
   - Extract dynamic download links

2. **Download ALL 6 external datasets** identified by H13/H16
   - PXD011967 (Ferri 2019, eLife) - Muscle aging
   - PXD015982 (Richter 2021, PMC) - Skin matrisome
   - +4 others (bone marrow, lung, brain, etc.)

3. **Save processed tables** to standard location
   - Format: `external_datasets/{PXD_ID}/raw_data.{xlsx|csv|tsv}`
   - Validate: Check column names, row counts, non-empty
   - Log: Record download date, file size, MD5 hash

4. **Re-run H16 validation pipeline** with real data
   - Transfer learning: Load pre-trained models, test on external data
   - Meta-analysis: Random-effects I² heterogeneity testing
   - Report: External R² (H08), AUC (H06), ρ (H03)

---

## 📋 Technical Implementation

### **Tool Stack**
```bash
pip install playwright pandas requests beautifulsoup4
playwright install chromium  # Headless browser
```

### **Architecture Pattern**

```python
from playwright.sync_api import sync_playwright
import pandas as pd
from pathlib import Path

class SupplementaryDownloader:
    """
    Automated browser-based downloader for journal supplementary files.
    Handles eLife, PMC, Nature, Cell, bioRxiv, Wiley.
    """

    def __init__(self, headless=True):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.context = self.browser.new_context(
            accept_downloads=True,
            viewport={'width': 1920, 'height': 1080}
        )
        self.page = self.context.new_page()

    def download_elife(self, doi: str, output_dir: Path) -> Path:
        """
        Download eLife supplementary files.

        Example DOI: 10.7554/eLife.49874
        Target: Supplementary file 1 (protein abundance Excel)
        """
        url = f"https://elifesciences.org/articles/{doi.split('/')[-1]}"
        self.page.goto(url)

        # Wait for JavaScript to render supplementary files section
        self.page.wait_for_selector('section[id*="supplementary"]', timeout=30000)

        # Find download link for Supplementary file 1
        download_link = self.page.locator('a:has-text("Supplementary file 1")').first

        # Click and wait for download
        with self.page.expect_download() as download_info:
            download_link.click()
        download = download_info.value

        # Save to output directory
        output_path = output_dir / download.suggested_filename
        download.save_as(output_path)

        return output_path

    def download_pmc(self, pmid: str, output_dir: Path) -> Path:
        """Download PMC supplementary files via Europe PMC."""
        url = f"https://europepmc.org/article/MED/{pmid}"
        self.page.goto(url)

        # Wait for supplementary data section
        self.page.wait_for_selector('[data-section="supplementary-material"]', timeout=30000)

        # Click "Download all supplementary files" button
        download_btn = self.page.locator('button:has-text("Download")')

        with self.page.expect_download() as download_info:
            download_btn.click()
        download = download_info.value

        output_path = output_dir / download.suggested_filename
        download.save_as(output_path)

        return output_path

    def download_nature(self, doi: str, output_dir: Path) -> Path:
        """Download Nature supplementary files (requires institution access)."""
        # Implementation for Nature papers
        pass

    def validate_download(self, file_path: Path) -> dict:
        """
        Validate downloaded file.

        Returns:
            {
                'valid': bool,
                'format': str,  # xlsx, csv, tsv
                'rows': int,
                'columns': list,
                'size_mb': float,
                'md5': str
            }
        """
        import hashlib

        # Check file exists
        if not file_path.exists():
            return {'valid': False, 'error': 'File not found'}

        # Detect format and load
        if file_path.suffix == '.xlsx':
            df = pd.read_excel(file_path)
        elif file_path.suffix in ['.csv', '.tsv']:
            sep = '\t' if file_path.suffix == '.tsv' else ','
            df = pd.read_csv(file_path, sep=sep)
        else:
            return {'valid': False, 'error': f'Unsupported format: {file_path.suffix}'}

        # Validate content
        if df.empty:
            return {'valid': False, 'error': 'File is empty'}

        # Calculate MD5
        with open(file_path, 'rb') as f:
            md5 = hashlib.md5(f.read()).hexdigest()

        return {
            'valid': True,
            'format': file_path.suffix,
            'rows': len(df),
            'columns': list(df.columns),
            'size_mb': file_path.stat().st_size / (1024 * 1024),
            'md5': md5
        }

    def close(self):
        """Close browser and cleanup."""
        self.context.close()
        self.browser.close()
        self.playwright.stop()

# USAGE
downloader = SupplementaryDownloader(headless=True)

# Download all 6 external datasets
datasets = [
    {'doi': '10.7554/eLife.49874', 'pxd': 'PXD011967', 'journal': 'eLife'},
    {'pmid': '33543036', 'pxd': 'PXD015982', 'journal': 'PMC'},
    # ... +4 more
]

for ds in datasets:
    output_dir = Path(f"external_datasets/{ds['pxd']}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if 'doi' in ds:
        file_path = downloader.download_elife(ds['doi'], output_dir)
    elif 'pmid' in ds:
        file_path = downloader.download_pmc(ds['pmid'], output_dir)

    # Validate
    validation = downloader.validate_download(file_path)
    print(f"{ds['pxd']}: {validation}")

downloader.close()
```

---

## 🎯 Success Criteria

**MANDATORY (must achieve ALL):**

1. ✅ **6/6 datasets downloaded** - ALL supplementary files saved locally
2. ✅ **Validation passed** - All files non-empty, correct format, valid columns
3. ✅ **H16 pipeline executed** - Transfer learning on real external data
4. ✅ **External validation results** - Report R², AUC, ρ for H08/H06/H03

**BONUS (nice to have):**

5. ⭐ **Retry logic** - Handle network failures, timeouts (3 retries with exponential backoff)
6. ⭐ **CAPTCHA handling** - Detect and alert if CAPTCHA blocks download
7. ⭐ **Rate limiting** - Respect journal servers (1 request/5 seconds)
8. ⭐ **Metadata logging** - JSON log with download timestamps, file hashes, validation status

---

## 📊 Expected Outcomes

**H16 External Validation (UNBLOCKED):**

```
VALIDATION RESULTS (Real External Data)
========================================

H08 S100 Stiffness Model:
  Training R² (internal): 0.75-0.81
  External R² (PXD011967): [TO BE DETERMINED]
  Success threshold: ≥0.60

H06 Biomarker Panel:
  Training AUC (internal): 1.00 (likely overfit)
  External AUC (PXD011967): [TO BE DETERMINED]
  Success threshold: ≥0.80

H03 Tissue Velocities:
  Internal ρ: 0.70-0.98
  External ρ (PXD011967): [TO BE DETERMINED]
  Success threshold: >0.70
```

**Meta-Analysis:**
- Random-effects pooling across 6 external datasets
- I² heterogeneity testing
- STABLE proteins: I²<25%, consistent across datasets
- VARIABLE proteins: I²>75%, context-dependent

---

## 🚨 Potential Blockers

**1. CAPTCHA Protection**
- **Risk:** Some journals use CAPTCHA for download protection
- **Mitigation:** Stealth mode, human-like delays, 2Captcha API integration

**2. Dynamic JavaScript**
- **Risk:** Download buttons generated by client-side JS frameworks
- **Mitigation:** Playwright waits for DOM ready, XPath selectors

**3. Institutional Access Required**
- **Risk:** Nature/Cell may require subscription
- **Mitigation:** Try PMC open-access version first, contact authors if blocked

**4. File Format Variation**
- **Risk:** Supplementary files in PDF/ZIP instead of Excel/CSV
- **Mitigation:** Unzip automatically, extract tables from PDF with tabula-py

---

## 📂 Output Structure

```
external_datasets/
├── PXD011967/
│   ├── raw_data.xlsx  # Downloaded supplementary file
│   ├── metadata.json  # Download timestamp, MD5, validation
│   └── preprocessing_log.txt
├── PXD015982/
│   ├── raw_data.csv
│   ├── metadata.json
│   └── preprocessing_log.txt
└── ...
```

---

## 🔬 Agents

**H21 Claude Code:**
- Implement Playwright downloader
- Download 6 datasets
- Validate files
- Re-run H16 pipeline

**H21 Codex:**
- Implement alternative approach (Selenium + BeautifulSoup)
- Cross-validate downloads
- Compare external validation results
- Benchmark: Playwright vs Selenium speed/reliability

---

## 📝 Deliverables

1. **Code:** `external_data_downloader.py` (Playwright implementation)
2. **Data:** 6 downloaded supplementary files in `external_datasets/`
3. **Validation:** `download_validation_report.json`
4. **H16 Results:** `external_validation_final_results.md`
5. **README:** `BROWSER_AUTOMATION_GUIDE.md` (instructions for future use)

---

**🎯 This hypothesis UNBLOCKS H16 (most critical hypothesis for validating ALL prior findings H01-H15)!**
