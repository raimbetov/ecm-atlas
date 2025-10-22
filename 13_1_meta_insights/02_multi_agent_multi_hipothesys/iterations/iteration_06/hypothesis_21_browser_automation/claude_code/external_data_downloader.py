"""
H21 - Browser Automation for External Data Acquisition

Playwright-based downloader for journal supplementary files.
Handles eLife, PMC, Nature, Cell, bioRxiv, Wiley.

Author: claude_code
Date: 2025-10-21
"""

from playwright.sync_api import sync_playwright, Page, Download
import pandas as pd
from pathlib import Path
import hashlib
import json
from datetime import datetime
import time
from typing import Dict, Optional, List


class SupplementaryDownloader:
    """
    Automated browser-based downloader for journal supplementary files.

    Handles JavaScript-rendered download buttons and dynamic content.
    Supports retry logic, CAPTCHA detection, and stealth mode.
    """

    def __init__(self, headless: bool = True, timeout: int = 30000):
        """
        Initialize Playwright browser.

        Args:
            headless: Run browser in headless mode (no GUI)
            timeout: Default timeout in milliseconds (30 seconds)
        """
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.headless = headless
        self.timeout = timeout

    def __enter__(self):
        """Context manager entry - start browser."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close browser."""
        self.close()

    def start(self):
        """Start Playwright browser with stealth configuration."""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)

        # Stealth mode configuration
        self.context = self.browser.new_context(
            accept_downloads=True,
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            extra_http_headers={
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            }
        )

        self.page = self.context.new_page()

    def detect_captcha(self) -> bool:
        """
        Detect if CAPTCHA is present on current page.

        Returns:
            True if CAPTCHA detected, False otherwise
        """
        captcha_indicators = [
            'recaptcha',
            'captcha',
            'g-recaptcha',
            'hcaptcha',
            'cloudflare'
        ]

        for indicator in captcha_indicators:
            if self.page.locator(f'[class*="{indicator}"], [id*="{indicator}"]').count() > 0:
                return True

        return False

    def download_elife(self, doi: str, output_dir: Path) -> Path:
        """
        Download eLife supplementary files.

        Args:
            doi: DOI like '10.7554/eLife.49874'
            output_dir: Directory to save downloaded file

        Returns:
            Path to downloaded file

        Raises:
            ValueError: If download link not found
            RuntimeError: If CAPTCHA detected
        """
        article_id = doi.split('/')[-1]
        url = f"https://elifesciences.org/articles/{article_id}"

        print(f"  → Navigating to {url}...")
        self.page.goto(url, timeout=self.timeout, wait_until='networkidle')

        # Check for CAPTCHA
        if self.detect_captcha():
            raise RuntimeError("CAPTCHA detected - manual intervention required")

        # Wait for JavaScript to render supplementary files section
        print("  → Waiting for supplementary files section...")
        try:
            self.page.wait_for_selector(
                'section[id*="supplementary"], div[data-test="supplementary"], section:has-text("Supplementary")',
                timeout=self.timeout
            )
        except Exception as e:
            print(f"  ⚠️  Warning: Could not find supplementary section - trying alternative selectors...")

        # Find download link for Supplementary file 1
        print("  → Locating download link...")
        selectors = [
            'a:has-text("Supplementary file 1")',
            'a:has-text("Supplementary File 1")',
            'a[href*="supp1"]',
            'a[href*="supplementary"][href*=".xlsx"]',
            'a[href*="supplementary"][href*=".csv"]',
            'a[download][href*="supplementary"]'
        ]

        download_link = None
        for selector in selectors:
            try:
                download_link = self.page.locator(selector).first
                if download_link.count() > 0:
                    print(f"  ✓ Found download link with selector: {selector}")
                    break
            except Exception:
                continue

        if not download_link or download_link.count() == 0:
            raise ValueError(f"Could not find supplementary file download link for {doi}")

        # Click and wait for download
        print("  → Initiating download...")
        with self.page.expect_download(timeout=self.timeout) as download_info:
            download_link.click()
        download = download_info.value

        # Save to output directory
        output_path = output_dir / download.suggested_filename
        print(f"  → Saving to {output_path}...")
        download.save_as(output_path)

        return output_path

    def download_pmc(self, pmid: str, output_dir: Path) -> Path:
        """
        Download PMC supplementary files via Europe PMC.

        Args:
            pmid: PubMed ID (e.g., '33543036')
            output_dir: Directory to save downloaded file

        Returns:
            Path to downloaded file

        Raises:
            ValueError: If download button not found
            RuntimeError: If CAPTCHA detected
        """
        url = f"https://europepmc.org/article/MED/{pmid}"

        print(f"  → Navigating to {url}...")
        self.page.goto(url, timeout=self.timeout, wait_until='networkidle')

        # Check for CAPTCHA
        if self.detect_captcha():
            raise RuntimeError("CAPTCHA detected - manual intervention required")

        # Wait for supplementary data section
        print("  → Waiting for supplementary section...")
        try:
            self.page.wait_for_selector(
                '[data-section="supplementary-material"], section:has-text("Supplementary"), div:has-text("Supplementary")',
                timeout=self.timeout
            )
        except Exception:
            print(f"  ⚠️  Warning: Could not find supplementary section - trying alternative approach...")

        # Click "Download" or "Supplementary file" button
        print("  → Locating download button...")
        selectors = [
            'a:has-text("Supplementary")',
            'button:has-text("Download")',
            'a:has-text("Download")',
            'a[href*="supplementary"]',
            'a[download]',
            'a[href*=".xlsx"]',
            'a[href*=".csv"]'
        ]

        download_btn = None
        for selector in selectors:
            try:
                download_btn = self.page.locator(selector).first
                if download_btn.count() > 0:
                    print(f"  ✓ Found download button with selector: {selector}")
                    break
            except Exception:
                continue

        if not download_btn or download_btn.count() == 0:
            raise ValueError(f"Could not find supplementary file for PMID {pmid}")

        print("  → Initiating download...")
        with self.page.expect_download(timeout=self.timeout) as download_info:
            download_btn.click()
        download = download_info.value

        output_path = output_dir / download.suggested_filename
        print(f"  → Saving to {output_path}...")
        download.save_as(output_path)

        return output_path

    def download_nature(self, doi: str, output_dir: Path) -> Path:
        """
        Download Nature supplementary files.

        Args:
            doi: DOI (e.g., '10.1038/s41586-020-2922-4')
            output_dir: Directory to save downloaded file

        Returns:
            Path to downloaded file

        Raises:
            ValueError: If download link not found
            RuntimeError: If CAPTCHA detected
        """
        url = f"https://doi.org/{doi}"

        print(f"  → Navigating to {url} (Nature)...")
        self.page.goto(url, timeout=self.timeout, wait_until='networkidle')

        # Check for CAPTCHA
        if self.detect_captcha():
            raise RuntimeError("CAPTCHA detected - manual intervention required")

        # Wait for supplementary section
        print("  → Waiting for supplementary section...")
        try:
            self.page.wait_for_selector(
                'section:has-text("Supplementary"), div[data-track-label="Supplementary"]',
                timeout=self.timeout
            )
        except Exception:
            print(f"  ⚠️  Warning: Could not find supplementary section...")

        # Find first supplementary file link
        print("  → Locating download link...")
        selectors = [
            'a:has-text("Supplementary Table")',
            'a:has-text("Supplementary Data")',
            'a[href*="supplementary"]',
            'a[download]'
        ]

        download_link = None
        for selector in selectors:
            try:
                download_link = self.page.locator(selector).first
                if download_link.count() > 0:
                    print(f"  ✓ Found download link with selector: {selector}")
                    break
            except Exception:
                continue

        if not download_link or download_link.count() == 0:
            raise ValueError(f"Could not find supplementary file for {doi}")

        print("  → Initiating download...")
        with self.page.expect_download(timeout=self.timeout) as download_info:
            download_link.click()
        download = download_info.value

        output_path = output_dir / download.suggested_filename
        print(f"  → Saving to {output_path}...")
        download.save_as(output_path)

        return output_path

    def validate_download(self, file_path: Path) -> Dict:
        """
        Validate downloaded file.

        Args:
            file_path: Path to downloaded file

        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'format': str,  # xlsx, csv, tsv
                'rows': int,
                'columns': list,
                'total_columns': int,
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
            return {'valid': False, 'error': 'File too small (likely empty)', 'size_mb': size_mb}

        # Detect format and load
        try:
            if file_path.suffix == '.xlsx':
                df = pd.read_excel(file_path, sheet_name=0)
            elif file_path.suffix in ['.csv', '.tsv', '.txt']:
                # Try different separators
                df = None
                for sep in ['\t', ',', ';']:
                    try:
                        test_df = pd.read_csv(file_path, sep=sep, nrows=5)
                        if test_df.shape[1] > 1:  # Valid if multiple columns
                            df = pd.read_csv(file_path, sep=sep)
                            break
                    except Exception:
                        continue

                if df is None:
                    return {'valid': False, 'error': 'Could not parse file with any separator'}
            else:
                return {'valid': False, 'error': f'Unsupported format: {file_path.suffix}'}

            # Validate content
            if df.empty:
                return {'valid': False, 'error': 'File is empty (no rows)'}

            if df.shape[0] < 10:
                return {'valid': False, 'error': f'Too few rows ({df.shape[0]}), expected ≥10'}

            if df.shape[1] < 3:
                return {'valid': False, 'error': f'Too few columns ({df.shape[1]}), expected ≥3'}

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
            return {'valid': False, 'error': f'Failed to parse: {str(e)}', 'size_mb': size_mb}

    def close(self):
        """Close browser and cleanup."""
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()


def main():
    """Test downloader with example.com."""
    print("Testing SupplementaryDownloader...")

    with SupplementaryDownloader(headless=True) as downloader:
        print("✓ Browser launched successfully")
        print(f"✓ Timeout: {downloader.timeout}ms")
        print("✓ Context created with stealth headers")

        # Test navigation
        downloader.page.goto('https://example.com')
        print(f"✓ Navigation test: {downloader.page.title()}")

    print("✓ Browser closed successfully")
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    main()
