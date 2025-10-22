"""
H21 - Download ALL 6 External Datasets

Execute Playwright downloader for all datasets identified in H16.
Includes retry logic, validation, and metadata logging.

Author: claude_code
Date: 2025-10-21
"""

import sys
from pathlib import Path
import json
import time
from datetime import datetime
from typing import List, Dict

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from external_data_downloader import SupplementaryDownloader


# Dataset registry from H16
EXTERNAL_DATASETS = [
    {
        'pxd': 'PXD011967',
        'journal': 'eLife',
        'doi': '10.7554/eLife.49874',
        'description': 'Ferri 2019 - Muscle aging proteomics (n=58)',
        'tissue': 'Skeletal_muscle',
        'priority': 'HIGH',
        'expected_proteins': 4380
    },
    {
        'pxd': 'PXD015982',
        'journal': 'PMC',
        'pmid': '33543036',
        'doi': '10.1016/j.mbplus.2020.100039',
        'description': 'Richter 2021 - Skin matrisome (n=6)',
        'tissue': 'Skin',
        'priority': 'HIGH',
        'expected_proteins': 229,
        'matrisome_focused': True
    },
    {
        'pxd': 'PXD007048',
        'journal': 'PRIDE',
        'description': 'Bone marrow niche',
        'tissue': 'Bone_marrow',
        'priority': 'MEDIUM',
        'note': 'May require PRIDE FTP access - will attempt web interface'
    },
    {
        'pxd': 'MSV000082958',
        'journal': 'MassIVE',
        'description': 'Lung fibrosis model',
        'tissue': 'Lung',
        'priority': 'MEDIUM',
        'note': 'MassIVE repository - may require alternative approach'
    },
    {
        'pxd': 'MSV000096508',
        'journal': 'MassIVE',
        'description': 'Brain cognitive aging (Mouse)',
        'tissue': 'Brain',
        'species': 'Mouse',
        'priority': 'MEDIUM',
        'note': 'Cross-species validation'
    },
    {
        'pxd': 'PXD016440',
        'journal': 'PRIDE',
        'description': 'Skin dermis developmental',
        'tissue': 'Skin',
        'priority': 'MEDIUM',
        'note': 'May require PRIDE FTP access'
    }
]


def download_all_datasets(
    datasets: List[Dict],
    output_base_dir: Path,
    retry_attempts: int = 3,
    headless: bool = True
) -> List[Dict]:
    """
    Download all external datasets with retry logic.

    Args:
        datasets: List of dataset dictionaries
        output_base_dir: Base directory for external_datasets/
        retry_attempts: Number of retries per dataset
        headless: Run browser in headless mode

    Returns:
        results: List of download results with metadata
    """
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    results = []

    print("\n" + "="*80)
    print(f"🚀 STARTING DATASET DOWNLOAD PIPELINE")
    print(f"   Total datasets: {len(datasets)}")
    print(f"   Output directory: {output_base}")
    print(f"   Retry attempts: {retry_attempts}")
    print("="*80)

    # Initialize downloader (reuse browser session)
    downloader = SupplementaryDownloader(headless=headless)
    downloader.start()

    try:
        for idx, ds in enumerate(datasets):
            print(f"\n{'='*80}")
            print(f"📦 DATASET {idx+1}/{len(datasets)}: {ds['pxd']}")
            print(f"   {ds['description']}")
            print(f"   Priority: {ds['priority']}")
            print(f"{'='*80}")

            output_dir = output_base / ds['pxd']
            output_dir.mkdir(parents=True, exist_ok=True)

            success = False
            file_path = None
            validation = None
            error = None

            for attempt in range(retry_attempts):
                try:
                    print(f"\n🔄 Attempt {attempt + 1}/{retry_attempts}...")

                    # Download based on journal
                    if ds['journal'] == 'eLife' and 'doi' in ds:
                        file_path = downloader.download_elife(ds['doi'], output_dir)

                    elif ds['journal'] == 'PMC' and 'pmid' in ds:
                        file_path = downloader.download_pmc(ds['pmid'], output_dir)

                    elif ds['journal'] == 'Nature' and 'doi' in ds:
                        file_path = downloader.download_nature(ds['doi'], output_dir)

                    elif ds['journal'] in ['PRIDE', 'MassIVE']:
                        print(f"  ⚠️  {ds['journal']} repository - requires specialized access")
                        print(f"  💡 Suggestion: {ds.get('note', 'Use FTP or contact authors')}")
                        error = f"Repository type {ds['journal']} not yet automated"
                        break

                    else:
                        error = f"Unknown journal: {ds['journal']}"
                        break

                    # Validate
                    if file_path:
                        print("\n✅ Download complete, validating...")
                        validation = downloader.validate_download(file_path)

                        if validation['valid']:
                            print(f"  ✓ File validated successfully!")
                            print(f"    Format: {validation['format']}")
                            print(f"    Rows: {validation['rows']:,}")
                            print(f"    Columns: {validation['total_columns']}")
                            print(f"    Size: {validation['size_mb']} MB")
                            print(f"    MD5: {validation['md5'][:16]}...")

                            # Save metadata
                            metadata = {
                                'pxd': ds['pxd'],
                                'journal': ds['journal'],
                                'doi': ds.get('doi'),
                                'pmid': ds.get('pmid'),
                                'description': ds['description'],
                                'tissue': ds.get('tissue'),
                                'priority': ds['priority'],
                                'download_timestamp': datetime.now().isoformat(),
                                'file_path': str(file_path),
                                'file_name': file_path.name,
                                'validation': validation,
                                'success': True
                            }

                            metadata_path = output_dir / 'metadata.json'
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f, indent=2)

                            print(f"  ✓ Metadata saved: {metadata_path}")

                            success = True
                            break
                        else:
                            print(f"  ✗ Validation failed: {validation.get('error')}")
                            error = validation.get('error')

                except Exception as e:
                    error = str(e)
                    print(f"  ✗ Error: {error}")

                # Wait before retry
                if attempt < retry_attempts - 1 and not success:
                    wait_time = 5 * (attempt + 1)  # Exponential backoff
                    print(f"  ⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

            # Store result
            results.append({
                'pxd': ds['pxd'],
                'priority': ds['priority'],
                'success': success,
                'file_path': str(file_path) if file_path else None,
                'validation': validation,
                'error': error if not success else None
            })

            # Rate limiting between datasets
            if idx < len(datasets) - 1:
                print(f"\n⏳ Rate limiting: waiting 5s before next dataset...")
                time.sleep(5)

    finally:
        # Close browser
        downloader.close()
        print("\n🔒 Browser closed")

    return results


def print_summary(results: List[Dict]):
    """Print download summary report."""
    print("\n" + "="*80)
    print("📊 DOWNLOAD SUMMARY")
    print("="*80)

    high_priority = [r for r in results if r['priority'] == 'HIGH']
    medium_priority = [r for r in results if r['priority'] == 'MEDIUM']

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\n✅ Successful: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.0f}%)")
    print(f"   HIGH priority: {sum(1 for r in high_priority if r['success'])}/{len(high_priority)}")
    print(f"   MEDIUM priority: {sum(1 for r in medium_priority if r['success'])}/{len(medium_priority)}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(results)}")

    print("\n" + "-"*80)
    print("DETAILED RESULTS:")
    print("-"*80)

    for r in results:
        status = "✅ SUCCESS" if r['success'] else "❌ FAILED"
        print(f"\n{status} - {r['pxd']} ({r['priority']})")

        if r['success'] and r['validation']:
            v = r['validation']
            print(f"  File: {Path(r['file_path']).name}")
            print(f"  Rows: {v['rows']:,}, Columns: {v['total_columns']}, Size: {v['size_mb']} MB")
        else:
            print(f"  Error: {r['error']}")

    print("\n" + "="*80)


def main():
    """Execute download pipeline."""
    # Output directory (external_datasets at project root)
    output_base_dir = Path("/Users/Kravtsovd/projects/ecm-atlas/13_1_meta_insights/02_multi_agent_multi_hipothesys/external_datasets")

    print("\n🌐🤖 H21: BROWSER AUTOMATION FOR EXTERNAL DATA ACQUISITION 🤖🌐")
    print("\nMission: Download 6 external datasets to unblock H16 validation!")

    # Download all datasets
    results = download_all_datasets(
        EXTERNAL_DATASETS,
        output_base_dir=output_base_dir,
        retry_attempts=3,
        headless=True
    )

    # Print summary
    print_summary(results)

    # Save results JSON
    results_path = Path(__file__).parent / 'download_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_datasets': len(EXTERNAL_DATASETS),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'results': results
        }, f, indent=2)

    print(f"\n💾 Results saved to: {results_path}")

    # Success criteria check
    successful_count = sum(1 for r in results if r['success'])
    high_priority_success = sum(1 for r in results if r['priority'] == 'HIGH' and r['success'])

    print("\n" + "="*80)
    print("🎯 SUCCESS CRITERIA CHECK")
    print("="*80)
    print(f"Target: ≥5/6 datasets downloaded (83%)")
    print(f"Actual: {successful_count}/6 datasets ({successful_count/6*100:.0f}%)")
    print(f"HIGH priority: {high_priority_success}/2 (CRITICAL)")

    if successful_count >= 5:
        print("\n✅ SUCCESS: Ready for H16 validation!")
    elif high_priority_success == 2:
        print("\n⚠️  PARTIAL: HIGH priority complete, MEDIUM optional")
    else:
        print("\n❌ BLOCKED: Need manual intervention for failed downloads")

    print("="*80)


if __name__ == '__main__':
    main()
