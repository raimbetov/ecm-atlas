"""
H21 - Direct HTTP Downloader (Alternative to Playwright)

Uses direct HTTP requests to download from known CDN patterns.
Faster and more reliable than browser automation for structured repositories.

Author: claude_code
Date: 2025-10-21
"""

import requests
from pathlib import Path
import hashlib
import json
from datetime import datetime
import time
from typing import Dict, Optional, List
import pandas as pd


class DirectHTTPDownloader:
    """
    Direct HTTP downloader for proteomics data repositories.

    Attempts multiple known URL patterns for each dataset.
    """

    def __init__(self, timeout: int = 60):
        """
        Initialize downloader.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
        })

    def try_elife_cdn_patterns(self, article_id: str) -> List[str]:
        """
        Generate possible eLife CDN URLs for supplementary files.

        Args:
            article_id: eLife article ID (e.g., '49874')

        Returns:
            List of possible URLs to try
        """
        patterns = [
            # eLife CDN patterns
            f"https://cdn.elifesciences.org/articles/elife-{article_id}/elife-{article_id}-supp1-v2.xlsx",
            f"https://cdn.elifesciences.org/articles/elife-{article_id}/elife-{article_id}-supp1-v1.xlsx",
            f"https://cdn.elifesciences.org/articles/elife-{article_id}/elife-{article_id}-fig1-data1-v2.xlsx",
            f"https://cdn.elifesciences.org/articles/elife-{article_id}/elife-{article_id}-fig1-data1-v1.xlsx",
            f"https://cdn.elifesciences.org/articles/{article_id}/elife-{article_id}-supp1-v2.xlsx",
            f"https://cdn.elifesciences.org/articles/{article_id}/elife-{article_id}-fig1-data1.xlsx",
            # Figure source data patterns
            f"https://elifesciences.org/download/aHR0cHM6Ly9jZG4uZWxpZmVzY2llbmNlcy5vcmcvYXJ0aWNsZXMvZWxpZmUtezIwMTl9LWZpZzEtZGF0YTEtdjIueGxzeA==/elife-{article_id}-fig1-data1-v2.xlsx",
        ]
        return patterns

    def try_pmc_patterns(self, pmc_id: str) -> List[str]:
        """
        Generate possible PMC supplementary file URLs.

        Args:
            pmc_id: PMC ID (e.g., 'PMC6803624' or just '6803624')

        Returns:
            List of possible URLs
        """
        if not pmc_id.startswith('PMC'):
            pmc_id = 'PMC' + pmc_id

        patterns = [
            f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/bin/supp_1.xlsx",
            f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/bin/TableS1.xlsx",
            f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/bin/mmc1.xlsx",
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/bin/supp_1.xlsx",
        ]
        return patterns

    def download_file(self, url: str, output_path: Path) -> bool:
        """
        Download file from URL.

        Args:
            url: URL to download from
            output_path: Path to save file

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"    Trying: {url}")
            response = self.session.get(url, timeout=self.timeout, stream=True)

            if response.status_code == 200:
                # Check content type
                content_type = response.headers.get('Content-Type', '')
                if 'html' in content_type.lower() and 'application' not in content_type.lower():
                    print(f"      ✗ Got HTML instead of file (404 page)")
                    return False

                # Save file
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Check file size
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"      ✓ Downloaded: {size_mb:.2f} MB")
                return True
            else:
                print(f"      ✗ HTTP {response.status_code}")
                return False

        except Exception as e:
            print(f"      ✗ Error: {str(e)[:50]}")
            return False

    def download_with_patterns(self, url_patterns: List[str], output_dir: Path, filename: str) -> Optional[Path]:
        """
        Try multiple URL patterns until one succeeds.

        Args:
            url_patterns: List of URLs to try
            output_dir: Directory to save file
            filename: Base filename

        Returns:
            Path to downloaded file if successful, None otherwise
        """
        for url in url_patterns:
            output_path = output_dir / filename
            if self.download_file(url, output_path):
                return output_path

            # Clean up failed download
            if output_path.exists():
                output_path.unlink()

        return None

    def validate_download(self, file_path: Path) -> Dict:
        """Validate downloaded file - same as Playwright version."""
        if not file_path.exists():
            return {'valid': False, 'error': 'File not found'}

        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb < 0.001:
            return {'valid': False, 'error': 'File too small', 'size_mb': size_mb}

        try:
            if file_path.suffix == '.xlsx':
                df = pd.read_excel(file_path, sheet_name=0)
            elif file_path.suffix in ['.csv', '.tsv', '.txt']:
                df = None
                for sep in ['\t', ',', ';']:
                    try:
                        test_df = pd.read_csv(file_path, sep=sep, nrows=5)
                        if test_df.shape[1] > 1:
                            df = pd.read_csv(file_path, sep=sep)
                            break
                    except:
                        continue
                if df is None:
                    return {'valid': False, 'error': 'Could not parse file'}
            else:
                return {'valid': False, 'error': f'Unsupported format: {file_path.suffix}'}

            if df.empty:
                return {'valid': False, 'error': 'File is empty'}

            if df.shape[0] < 10:
                return {'valid': False, 'error': f'Too few rows ({df.shape[0]})'}

            with open(file_path, 'rb') as f:
                md5 = hashlib.md5(f.read()).hexdigest()

            return {
                'valid': True,
                'format': file_path.suffix,
                'rows': len(df),
                'columns': list(df.columns)[:10],
                'total_columns': df.shape[1],
                'size_mb': round(size_mb, 2),
                'md5': md5
            }

        except Exception as e:
            return {'valid': False, 'error': f'Parse error: {str(e)}', 'size_mb': size_mb}


def download_pxd011967(downloader: DirectHTTPDownloader, output_dir: Path) -> Optional[Path]:
    """
    Download PXD011967 (Ferri 2019 - Muscle Aging).

    Try multiple known URL patterns for eLife article 49874.
    """
    print("\n  Trying eLife CDN patterns...")

    patterns = downloader.try_elife_cdn_patterns('49874')

    # Try with different extensions
    additional_patterns = []
    for pattern in patterns:
        if pattern.endswith('.xlsx'):
            additional_patterns.append(pattern.replace('.xlsx', '.csv'))
            additional_patterns.append(pattern.replace('.xlsx', '.zip'))

    all_patterns = patterns + additional_patterns

    return downloader.download_with_patterns(all_patterns, output_dir, 'raw_data.xlsx')


def download_pxd015982(downloader: DirectHTTPDownloader, output_dir: Path) -> Optional[Path]:
    """
    Download PXD015982 (Richter 2021 - Skin Matrisome).

    Try ScienceDirect and PMC patterns.
    """
    print("\n  Trying ScienceDirect/PMC patterns...")

    patterns = [
        # ScienceDirect supplementary data patterns
        "https://ars.els-cdn.com/content/image/1-s2.0-S2590028520300195-mmc1.xlsx",
        "https://ars.els-cdn.com/content/image/1-s2.0-S2590028520300195-mmc2.xlsx",
        "https://ars.els-cdn.com/content/image/1-s2.0-S2590028520300195-mmc3.xlsx",
        # Alternative patterns
        "https://www.sciencedirect.com/science/article/pii/S2590028520300195/mmc1",
    ]

    return downloader.download_with_patterns(patterns, output_dir, 'raw_data.xlsx')


def main():
    """Execute direct HTTP download pipeline."""
    print("\n🌐 H21: DIRECT HTTP DOWNLOAD (Alternative to Playwright)")
    print("="*80)

    output_base = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/external_datasets")
    output_base.mkdir(parents=True, exist_ok=True)

    downloader = DirectHTTPDownloader(timeout=60)
    results = []

    # HIGH PRIORITY 1: PXD011967
    print("\n" + "="*80)
    print("📦 PXD011967: Ferri 2019 - Muscle Aging")
    print("="*80)

    pxd011967_dir = output_base / "PXD011967"
    pxd011967_dir.mkdir(parents=True, exist_ok=True)

    file_path = download_pxd011967(downloader, pxd011967_dir)

    if file_path:
        validation = downloader.validate_download(file_path)
        if validation['valid']:
            print(f"\n  ✅ SUCCESS!")
            print(f"     Rows: {validation['rows']:,}, Columns: {validation['total_columns']}")

            metadata = {
                'pxd': 'PXD011967',
                'success': True,
                'download_timestamp': datetime.now().isoformat(),
                'file_path': str(file_path),
                'validation': validation
            }

            with open(pxd011967_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            results.append(metadata)
        else:
            print(f"\n  ❌ Validation failed: {validation['error']}")
            results.append({'pxd': 'PXD011967', 'success': False, 'error': validation['error']})
    else:
        print(f"\n  ❌ All URL patterns failed")
        results.append({'pxd': 'PXD011967', 'success': False, 'error': 'All URL patterns failed'})

    time.sleep(3)

    # HIGH PRIORITY 2: PXD015982
    print("\n" + "="*80)
    print("📦 PXD015982: Richter 2021 - Skin Matrisome")
    print("="*80)

    pxd015982_dir = output_base / "PXD015982"
    pxd015982_dir.mkdir(parents=True, exist_ok=True)

    file_path = download_pxd015982(downloader, pxd015982_dir)

    if file_path:
        validation = downloader.validate_download(file_path)
        if validation['valid']:
            print(f"\n  ✅ SUCCESS!")
            print(f"     Rows: {validation['rows']:,}, Columns: {validation['total_columns']}")

            metadata = {
                'pxd': 'PXD015982',
                'success': True,
                'download_timestamp': datetime.now().isoformat(),
                'file_path': str(file_path),
                'validation': validation
            }

            with open(pxd015982_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            results.append(metadata)
        else:
            print(f"\n  ❌ Validation failed: {validation['error']}")
            results.append({'pxd': 'PXD015982', 'success': False, 'error': validation['error']})
    else:
        print(f"\n  ❌ All URL patterns failed")
        results.append({'pxd': 'PXD015982', 'success': False, 'error': 'All URL patterns failed'})

    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)

    successful = sum(1 for r in results if r.get('success'))
    print(f"\n✅ Successful: {successful}/2 HIGH priority datasets")

    if successful >= 1:
        print("\n🎯 At least 1 dataset downloaded - can proceed with partial validation")
    elif successful == 2:
        print("\n🎯 BOTH HIGH priority datasets downloaded - ready for H16!")
    else:
        print("\n❌ Manual download required - see MANUAL_DOWNLOAD_GUIDE.md")

    # Save results
    results_path = Path(__file__).parent / 'direct_http_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'method': 'direct_http',
            'successful': successful,
            'results': results
        }, f, indent=2)

    print(f"\n💾 Results saved: {results_path}")


if __name__ == '__main__':
    main()
