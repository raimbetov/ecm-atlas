# H21 – Browser Automation for External Data Acquisition

## Scientific Question
Can automated browser automation (Playwright/Selenium) programmatically download supplementary files from journal websites (eLife, PMC, Nature, Cell) to enable H16 External Validation completion for all 6 identified proteomics datasets?

## Background & Rationale

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

**Why Browser Automation Works:**
- Executes JavaScript to render dynamic download buttons
- Can wait for DOM elements to load
- Simulates human click interactions
- Bypasses API limitations

## Objectives

### Primary Objective
Implement Playwright-based downloader to automatically download ALL 6 external proteomics datasets (supplementary files) from journal websites, enabling H16 external validation with real data.

### Secondary Objectives
1. Validate downloaded files (format, size, column names, row counts)
2. Save to standardized location with metadata (MD5 hash, download timestamp)
3. Re-run H16 validation pipeline with real external data
4. Report external validation results (R², AUC, ρ for H08/H06/H03)
5. Implement retry logic and error handling (network failures, CAPTCHA detection)

## Hypotheses to Test

### H21.1: Download Feasibility
All 6 datasets can be successfully downloaded via Playwright within 10 minutes total execution time.

### H21.2: Data Validity
Downloaded files are valid (non-empty, correct format, expected columns), validation pass rate ≥95%.

### H21.3: H16 Unblocking
External validation pipeline executes successfully with real data, producing reportable R²/AUC/ρ metrics for H08/H06/H03 models.

### H21.4: Cross-Journal Reliability
Downloader works reliably across different journal platforms (eLife, PMC, Nature, Cell) with ≥90% success rate.

## Required Analyses

### 1. SETUP & INSTALLATION

**Install Dependencies:**
```bash
pip install playwright pandas requests beautifulsoup4
playwright install chromium  # Headless browser
```

**Verify Installation:**
```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('https://example.com')
    print(f"✓ Playwright working: {page.title()}")
    browser.close()
```

### 2. CORE DOWNLOADER IMPLEMENTATION

**Architecture:**
```python
from playwright.sync_api import sync_playwright
import pandas as pd
from pathlib import Path
import hashlib
import json
from datetime import datetime

class SupplementaryDownloader:
    """
    Automated browser-based downloader for journal supplementary files.
    Handles eLife, PMC, Nature, Cell, bioRxiv, Wiley.
    """

    def __init__(self, headless=True, timeout=30000):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=headless)
        self.context = self.browser.new_context(
            accept_downloads=True,
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        )
        self.page = self.context.new_page()
        self.timeout = timeout

    def download_elife(self, doi: str, output_dir: Path) -> Path:
        """
        Download eLife supplementary files.

        Example DOI: 10.7554/eLife.49874
        Target: Supplementary file 1 (protein abundance Excel)
        """
        article_id = doi.split('/')[-1]
        url = f"https://elifesciences.org/articles/{article_id}"

        print(f"Navigating to {url}...")
        self.page.goto(url, timeout=self.timeout)

        # Wait for JavaScript to render supplementary files section
        print("Waiting for supplementary files section...")
        self.page.wait_for_selector('section[id*="supplementary"], div[data-test="supplementary"]', timeout=self.timeout)

        # Find download link for Supplementary file 1
        print("Locating download link...")
        # Try multiple selectors
        selectors = [
            'a:has-text("Supplementary file 1")',
            'a:has-text("Supplementary File 1")',
            'a[href*="supp1"]',
            'a[href*="supplementary"]'
        ]

        download_link = None
        for selector in selectors:
            try:
                download_link = self.page.locator(selector).first
                if download_link.count() > 0:
                    break
            except:
                continue

        if not download_link or download_link.count() == 0:
            raise ValueError(f"Could not find supplementary file download link for {doi}")

        # Click and wait for download
        print("Initiating download...")
        with self.page.expect_download(timeout=self.timeout) as download_info:
            download_link.click()
        download = download_info.value

        # Save to output directory
        output_path = output_dir / download.suggested_filename
        print(f"Saving to {output_path}...")
        download.save_as(output_path)

        return output_path

    def download_pmc(self, pmid: str, output_dir: Path) -> Path:
        """Download PMC supplementary files via Europe PMC."""
        url = f"https://europepmc.org/article/MED/{pmid}"

        print(f"Navigating to {url}...")
        self.page.goto(url, timeout=self.timeout)

        # Wait for supplementary data section
        print("Waiting for supplementary section...")
        self.page.wait_for_selector('[data-section="supplementary-material"], section:has-text("Supplementary")', timeout=self.timeout)

        # Click "Download" or "Supplementary file" button
        print("Locating download button...")
        selectors = [
            'button:has-text("Download")',
            'a:has-text("Supplementary")',
            'a[href*="supplementary"]',
            'a[download]'
        ]

        download_btn = None
        for selector in selectors:
            try:
                download_btn = self.page.locator(selector).first
                if download_btn.count() > 0:
                    break
            except:
                continue

        if not download_btn or download_btn.count() == 0:
            raise ValueError(f"Could not find supplementary file for PMID {pmid}")

        print("Initiating download...")
        with self.page.expect_download(timeout=self.timeout) as download_info:
            download_btn.click()
        download = download_info.value

        output_path = output_dir / download.suggested_filename
        print(f"Saving to {output_path}...")
        download.save_as(output_path)

        return output_path

    def download_nature(self, doi: str, output_dir: Path) -> Path:
        """Download Nature supplementary files."""
        url = f"https://doi.org/{doi}"

        print(f"Navigating to {url} (Nature)...")
        self.page.goto(url, timeout=self.timeout)

        # Wait for supplementary section
        self.page.wait_for_selector('section:has-text("Supplementary"), div[data-track-label="Supplementary"]', timeout=self.timeout)

        # Find first supplementary file link
        selectors = [
            'a:has-text("Supplementary Table")',
            'a:has-text("Supplementary Data")',
            'a[href*="supplementary"]'
        ]

        download_link = None
        for selector in selectors:
            try:
                download_link = self.page.locator(selector).first
                if download_link.count() > 0:
                    break
            except:
                continue

        if not download_link or download_link.count() == 0:
            raise ValueError(f"Could not find supplementary file for {doi}")

        with self.page.expect_download(timeout=self.timeout) as download_info:
            download_link.click()
        download = download_info.value

        output_path = output_dir / download.suggested_filename
        download.save_as(output_path)

        return output_path

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
                'md5': str,
                'error': str (if invalid)
            }
        """
        # Check file exists
        if not file_path.exists():
            return {'valid': False, 'error': 'File not found'}

        # Check file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb < 0.001:  # Less than 1KB
            return {'valid': False, 'error': 'File too small (likely empty)'}

        # Detect format and load
        try:
            if file_path.suffix == '.xlsx':
                df = pd.read_excel(file_path, sheet_name=0)
            elif file_path.suffix in ['.csv', '.tsv', '.txt']:
                # Try different separators
                for sep in ['\t', ',', ';']:
                    try:
                        df = pd.read_csv(file_path, sep=sep, nrows=5)
                        if df.shape[1] > 1:  # Valid if multiple columns
                            df = pd.read_csv(file_path, sep=sep)
                            break
                    except:
                        continue
            else:
                return {'valid': False, 'error': f'Unsupported format: {file_path.suffix}'}

            # Validate content
            if df.empty:
                return {'valid': False, 'error': 'File is empty'}

            if df.shape[0] < 10:
                return {'valid': False, 'error': f'Too few rows ({df.shape[0]})'}

            # Calculate MD5
            with open(file_path, 'rb') as f:
                md5 = hashlib.md5(f.read()).hexdigest()

            return {
                'valid': True,
                'format': file_path.suffix,
                'rows': len(df),
                'columns': list(df.columns)[:10],  # First 10 columns
                'total_columns': df.shape[1],
                'size_mb': round(size_mb, 2),
                'md5': md5
            }

        except Exception as e:
            return {'valid': False, 'error': f'Failed to parse: {str(e)}'}

    def close(self):
        """Close browser and cleanup."""
        self.context.close()
        self.browser.close()
        self.playwright.stop()
```

### 3. DOWNLOAD ALL 6 DATASETS

**Dataset Registry:**
```python
# Based on H16 identified datasets
EXTERNAL_DATASETS = [
    {
        'pxd': 'PXD011967',
        'journal': 'eLife',
        'doi': '10.7554/eLife.49874',
        'description': 'Ferri 2019 - Muscle aging proteomics (n=58)',
        'expected_file': 'supp1'  # Supplementary file 1
    },
    {
        'pxd': 'PXD015982',
        'journal': 'PMC',
        'pmid': '33543036',
        'description': 'Richter 2021 - Skin matrisome (n=6)',
        'expected_file': 'supplementary'
    },
    {
        'pxd': 'PXD023456',  # Placeholder - update with real IDs from H16
        'journal': 'Nature',
        'doi': '10.1038/...',
        'description': 'Bone marrow aging',
        'expected_file': 'supplementary_table'
    },
    # Add remaining 3 datasets from H16 results
]
```

**Download Pipeline:**
```python
import time
from datetime import datetime

def download_all_datasets(datasets, output_base_dir, retry_attempts=3):
    """
    Download all external datasets with retry logic.

    Args:
        datasets: List of dataset dictionaries
        output_base_dir: Base directory for external_datasets/
        retry_attempts: Number of retries per dataset

    Returns:
        results: List of download results with metadata
    """
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    downloader = SupplementaryDownloader(headless=True)
    results = []

    for ds in datasets:
        print(f"\n{'='*80}")
        print(f"Processing {ds['pxd']}: {ds['description']}")
        print(f"{'='*80}")

        output_dir = output_base / ds['pxd']
        output_dir.mkdir(parents=True, exist_ok=True)

        success = False
        file_path = None
        validation = None
        error = None

        for attempt in range(retry_attempts):
            try:
                print(f"\nAttempt {attempt + 1}/{retry_attempts}...")

                # Download based on journal
                if 'doi' in ds:
                    if ds['journal'] == 'eLife':
                        file_path = downloader.download_elife(ds['doi'], output_dir)
                    elif ds['journal'] == 'Nature':
                        file_path = downloader.download_nature(ds['doi'], output_dir)
                elif 'pmid' in ds:
                    file_path = downloader.download_pmc(ds['pmid'], output_dir)

                # Validate
                print("Validating download...")
                validation = downloader.validate_download(file_path)

                if validation['valid']:
                    print(f"✓ Download successful: {file_path.name}")
                    print(f"  Rows: {validation['rows']}, Columns: {validation['total_columns']}")
                    print(f"  Size: {validation['size_mb']} MB")
                    print(f"  MD5: {validation['md5']}")

                    # Save metadata
                    metadata = {
                        'pxd': ds['pxd'],
                        'journal': ds['journal'],
                        'doi': ds.get('doi'),
                        'pmid': ds.get('pmid'),
                        'download_timestamp': datetime.now().isoformat(),
                        'file_path': str(file_path),
                        'validation': validation
                    }

                    metadata_path = output_dir / 'metadata.json'
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    success = True
                    break
                else:
                    print(f"✗ Validation failed: {validation.get('error')}")

            except Exception as e:
                error = str(e)
                print(f"✗ Error: {error}")

            # Wait before retry
            if attempt < retry_attempts - 1:
                wait_time = 5 * (attempt + 1)  # Exponential backoff
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

        results.append({
            'pxd': ds['pxd'],
            'success': success,
            'file_path': str(file_path) if file_path else None,
            'validation': validation,
            'error': error if not success else None
        })

        # Rate limiting between datasets
        time.sleep(5)

    downloader.close()

    return results
```

**Execute Downloads:**
```python
# Run download pipeline
results = download_all_datasets(
    EXTERNAL_DATASETS,
    output_base_dir='/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/external_datasets',
    retry_attempts=3
)

# Summary report
successful = sum(1 for r in results if r['success'])
print(f"\n{'='*80}")
print(f"DOWNLOAD SUMMARY: {successful}/{len(results)} successful")
print(f"{'='*80}")

for r in results:
    status = "✓" if r['success'] else "✗"
    print(f"{status} {r['pxd']}: {r.get('error', 'SUCCESS')}")

# Save results
with open('download_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### 4. RE-RUN H16 VALIDATION PIPELINE

**Load H16 Framework:**
```python
# Load external validation framework from H16
import sys
sys.path.append('/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/iterations/iteration_05/hypothesis_16_h13_validation_completion/claude_code')

# Import H16 validation modules
from h13_completion_claude_code import (
    ExternalValidationFramework,
    transfer_learning_validation,
    meta_analysis_heterogeneity
)

# Initialize framework
framework = ExternalValidationFramework(
    internal_data_path='/Users/Kravtsovd/projects/ecm-atlas/08_merged_ecm_dataset/merged_ecm_aging_zscore.csv',
    external_data_dir='/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/external_datasets'
)
```

**Run Validation for H08 (S100 Stiffness Model):**
```python
# Load H08 model
h08_model = torch.load('/iterations/iteration_03/hypothesis_08_s100_calcium_signaling/claude_code/s100_stiffness_model_claude_code.pth')

# Transfer learning on each external dataset
h08_results = []

for ds in EXTERNAL_DATASETS:
    pxd = ds['pxd']
    external_data = framework.load_external_dataset(pxd)

    if external_data is not None:
        # Transfer learning (no retraining, just evaluate)
        r2_external = framework.validate_model(h08_model, external_data, target='stiffness')

        h08_results.append({
            'dataset': pxd,
            'r2_external': r2_external
        })

        print(f"{pxd}: External R² = {r2_external:.3f}")

# Meta-analysis
mean_r2 = np.mean([r['r2_external'] for r in h08_results])
i2_heterogeneity = framework.calculate_i2_statistic(h08_results)

print(f"\nH08 External Validation:")
print(f"  Mean R²: {mean_r2:.3f}")
print(f"  I² heterogeneity: {i2_heterogeneity:.1f}%")
print(f"  Interpretation: {'STABLE' if i2_heterogeneity < 25 else 'VARIABLE'}")
```

**Run Validation for H06 (Biomarker Panel):**
```python
# Load H06 biomarker panel
h06_genes = ['S100A9', 'S100A10', 'LOX', 'TGM2', 'F13B', 'SERPINE1']

# AUC for aging classification
h06_results = []

for ds in EXTERNAL_DATASETS:
    pxd = ds['pxd']
    external_data = framework.load_external_dataset(pxd)

    if external_data is not None:
        # Binary classification: young (<40) vs old (>60)
        auc_external = framework.validate_biomarker_panel(h06_genes, external_data)

        h06_results.append({
            'dataset': pxd,
            'auc_external': auc_external
        })

        print(f"{pxd}: External AUC = {auc_external:.3f}")

# Meta-analysis
mean_auc = np.mean([r['auc_external'] for r in h06_results])
print(f"\nH06 External Validation:")
print(f"  Mean AUC: {mean_auc:.3f}")
print(f"  Success: {'YES' if mean_auc >= 0.80 else 'NO'}")
```

**Run Validation for H03 (Tissue Velocities):**
```python
# Load H03 velocity correlations
h03_results = []

for ds in EXTERNAL_DATASETS:
    pxd = ds['pxd']
    external_data = framework.load_external_dataset(pxd)

    if external_data is not None:
        # Velocity correlation
        rho_external = framework.validate_velocity_correlation(external_data)

        h03_results.append({
            'dataset': pxd,
            'rho_external': rho_external
        })

        print(f"{pxd}: External ρ = {rho_external:.3f}")

# Meta-analysis
mean_rho = np.mean([r['rho_external'] for r in h03_results])
print(f"\nH03 External Validation:")
print(f"  Mean ρ: {mean_rho:.3f}")
print(f"  Success: {'YES' if mean_rho > 0.70 else 'NO'}")
```

### 5. ERROR HANDLING & ROBUSTNESS

**CAPTCHA Detection:**
```python
def detect_captcha(page):
    """Detect if CAPTCHA is present."""
    captcha_indicators = [
        'recaptcha',
        'captcha',
        'g-recaptcha',
        'hcaptcha'
    ]

    for indicator in captcha_indicators:
        if page.locator(f'[class*="{indicator}"], [id*="{indicator}"]').count() > 0:
            return True

    return False

# In downloader methods:
if detect_captcha(self.page):
    raise RuntimeError("CAPTCHA detected - manual intervention required")
```

**Rate Limiting:**
```python
import time

# Add delays between requests
time.sleep(5)  # 5 seconds between downloads (respect journal servers)
```

**Stealth Mode:**
```python
# Add stealth headers
self.context = self.browser.new_context(
    accept_downloads=True,
    viewport={'width': 1920, 'height': 1080},
    user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    extra_http_headers={
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive'
    }
)
```

## Deliverables

### Code & Scripts
- `external_data_downloader.py` — Core Playwright downloader class
- `download_all_datasets.py` — Pipeline to download all 6 datasets
- `validate_downloads.py` — Validation and metadata logging
- `run_h16_validation.py` — Re-run H16 with real external data

### Data Files
- `external_datasets/{PXD_ID}/raw_data.{xlsx|csv|tsv}` — Downloaded supplementary files (6 total)
- `external_datasets/{PXD_ID}/metadata.json` — Download timestamp, MD5, validation results
- `download_results.json` — Summary of all downloads (success/failure)

### Validation Results
- `external_validation_h08_results.csv` — External R² for S100 stiffness model across 6 datasets
- `external_validation_h06_results.csv` — External AUC for biomarker panel across 6 datasets
- `external_validation_h03_results.csv` — External ρ for tissue velocities across 6 datasets
- `meta_analysis_summary.csv` — I² heterogeneity, mean metrics, stability assessment

### Visualizations
- `visualizations_{agent}/download_success_rate_{agent}.png` — Bar chart of successful downloads
- `visualizations_{agent}/external_validation_r2_{agent}.png` — Violin plot of R² across datasets (H08)
- `visualizations_{agent}/external_validation_auc_{agent}.png` — ROC curves for external datasets (H06)
- `visualizations_{agent}/heterogeneity_i2_{agent}.png` — Forest plot of I² statistics

### Report
- `90_results_{agent}.md` — CRITICAL findings:
  - **Download Success:** 6/6 datasets downloaded? Any failures?
  - **Validation Quality:** All files valid? Row/column counts?
  - **H16 Unblocked:** External validation executed successfully?
  - **H08 External R²:** Mean R² across datasets? I² heterogeneity?
  - **H06 External AUC:** Mean AUC across datasets? Biomarker panel robust?
  - **H03 External ρ:** Mean correlation? Velocity model generalizes?
  - **FINAL VERDICT:** Are H01-H20 findings validated externally? Publication-ready?
  - **Recommendations:** Manual downloads needed for any failed datasets?

## Success Criteria

| Metric | Target | Source |
|--------|--------|--------|
| Downloads successful | 6/6 (100%) | Playwright automation |
| Validation pass rate | ≥95% | File format, size, content checks |
| H08 external R² (mean) | ≥0.60 | Transfer learning on external data |
| H06 external AUC (mean) | ≥0.80 | Biomarker panel generalization |
| H03 external ρ (mean) | ≥0.70 | Velocity correlation robustness |
| I² heterogeneity (H08) | <50% | Meta-analysis stability |
| Execution time | <10 min | Total download + validation |
| Overall SUCCESS | ≥6/7 criteria met | Comprehensive assessment |

## Expected Outcomes

### Scenario 1: COMPLETE SUCCESS (H16 Unblocked)
- 6/6 datasets downloaded successfully
- All validations pass (100% pass rate)
- H08: Mean R²=0.68, I²=22% (STABLE)
- H06: Mean AUC=0.83 (robust biomarker panel)
- H03: Mean ρ=0.74 (velocity model generalizes)
- **Action:** Publish external validation results, integrate into H01-H20 synthesis, prepare manuscript

### Scenario 2: PARTIAL SUCCESS (Most Data Downloaded)
- 4-5/6 datasets downloaded (CAPTCHA blocks 1-2)
- Validations pass for downloaded files
- H08/H06/H03 show moderate external performance (R²=0.55, AUC=0.75, ρ=0.65)
- **Action:** Manual download for failed datasets, adjust expectations for external generalization

### Scenario 3: DOWNLOAD FAILURE (JavaScript/CAPTCHA Blocks)
- <4/6 datasets downloaded (automation blocked)
- Need manual intervention or alternative approach
- **Action:** Contact journal authors directly, request files via email, try Selenium alternative

### Scenario 4: EXTERNAL VALIDATION FAILURE (Data Downloaded but Models Don't Generalize)
- 6/6 downloads successful
- BUT: H08 R²=0.25 (poor), H06 AUC=0.60 (random), H03 ρ=0.30 (weak)
- **Action:** Critical reassessment of H01-H20 findings, investigate overfitting, require larger validation cohorts

## Dataset

**Target Datasets (from H16):**
1. PXD011967 (eLife, Ferri 2019) - Muscle aging
2. PXD015982 (PMC, Richter 2021) - Skin matrisome
3-6. [Update with remaining 4 datasets from H16 results]

**Output Location:**
`/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/external_datasets/`

**H16 Validation Framework:**
`/iterations/iteration_05/hypothesis_16_h13_validation_completion/claude_code/h13_completion_claude_code.py`

**Models to Validate:**
- H08: `/iterations/iteration_03/hypothesis_08_s100_calcium_signaling/claude_code/s100_stiffness_model_claude_code.pth`
- H06: `/iterations/iteration_02/hypothesis_06_biomarker_panel/claude_code/biomarker_genes_claude_code.json`
- H03: `/iterations/iteration_01/hypothesis_03_tissue_velocities/claude_code/velocity_correlations_claude_code.csv`

## References

1. **H16 External Validation**: `/iterations/iteration_05/hypothesis_16_h13_validation_completion/`
2. **Playwright Documentation**: https://playwright.dev/python/
3. **eLife API**: https://api.elifesciences.org/documentation
4. **Europe PMC API**: https://europepmc.org/RestfulWebService
5. **Web Scraping Ethics**: Respect robots.txt, rate limiting, terms of service
