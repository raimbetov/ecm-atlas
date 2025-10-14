#!/usr/bin/env python3
"""
Comprehensive test of all datasets and all visualizations
"""

from playwright.sync_api import sync_playwright
import time
import json
from datetime import datetime
import requests

def test_all_datasets():
    """Test all datasets and capture screenshots"""

    # Get list of datasets from API
    response = requests.get('http://localhost:5004/api/datasets')
    datasets = response.json()['datasets']

    print(f"Found {len(datasets)} datasets to test")
    print("="*70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "datasets": []
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1920, 'height': 3000})

        page.goto('http://localhost:8083/dashboard.html')
        time.sleep(5)

        # Test each dataset
        for dataset in datasets:
            dataset_name = dataset['name']
            compartments = dataset['compartments']

            print(f"\n📊 Testing dataset: {dataset_name}")
            print(f"   Organ: {dataset['organ']}")
            print(f"   Compartments: {', '.join(compartments)}")
            print(f"   Proteins: {dataset['protein_count']}")

            dataset_result = {
                "name": dataset_name,
                "organ": dataset['organ'],
                "compartments": compartments,
                "status": "testing",
                "charts": {}
            }

            try:
                # Select dataset
                page.select_option('#dataset-select', dataset_name)
                time.sleep(8)  # Wait for all charts to load

                # Check each visualization type
                chart_types = ['heatmap', 'volcano', 'scatter', 'bars']

                for chart_type in chart_types:
                    chart_id = f"{chart_type}-chart"
                    chart_elem = page.locator(f'#{chart_id}')

                    if chart_elem.count() > 0:
                        # Check for error messages
                        error_elem = chart_elem.locator('.error')
                        if error_elem.count() > 0:
                            error_text = error_elem.inner_text()
                            print(f"   ❌ {chart_type}: ERROR - {error_text}")
                            dataset_result['charts'][chart_type] = f"ERROR: {error_text}"
                        else:
                            print(f"   ✅ {chart_type}: OK")
                            dataset_result['charts'][chart_type] = "OK"
                    else:
                        print(f"   ⚠️  {chart_type}: NOT FOUND")
                        dataset_result['charts'][chart_type] = "NOT_FOUND"

                # Check histograms (one per compartment)
                histogram_status = []
                for comp in compartments:
                    hist_id = f"histogram-{comp}"
                    hist_elem = page.locator(f'#{hist_id}')
                    if hist_elem.count() > 0:
                        error_elem = hist_elem.locator('.error')
                        if error_elem.count() > 0:
                            histogram_status.append(f"{comp}:ERROR")
                        else:
                            histogram_status.append(f"{comp}:OK")

                dataset_result['charts']['histograms'] = ', '.join(histogram_status)
                print(f"   ✅ histograms: {', '.join(histogram_status)}")

                # Check comparison (if multi-compartment)
                if len(compartments) > 1:
                    comp_elem = page.locator('#comparison-chart')
                    if comp_elem.count() > 0:
                        error_elem = comp_elem.locator('.error')
                        if error_elem.count() > 0:
                            error_text = error_elem.inner_text()
                            dataset_result['charts']['comparison'] = f"ERROR: {error_text}"
                            print(f"   ❌ comparison: ERROR - {error_text}")
                        else:
                            dataset_result['charts']['comparison'] = "OK"
                            print(f"   ✅ comparison: OK")
                else:
                    dataset_result['charts']['comparison'] = "N/A (single compartment)"
                    print(f"   ⏭️  comparison: N/A (single compartment)")

                # Take screenshot
                screenshot_path = f"test_screenshots/{dataset_name}_full.png"
                page.screenshot(path=screenshot_path, full_page=True)
                dataset_result['screenshot'] = screenshot_path
                print(f"   📸 Screenshot: {screenshot_path}")

                dataset_result['status'] = "success"

            except Exception as e:
                print(f"   ❌ FAILED: {str(e)}")
                dataset_result['status'] = "failed"
                dataset_result['error'] = str(e)

            results['datasets'].append(dataset_result)

        # Test Compare Datasets tab
        print(f"\n📊 Testing Compare Datasets tab")
        try:
            compare_button = page.locator('[data-tab="compare"]')
            compare_button.click()
            time.sleep(5)

            # Check heatmap
            heatmap_elem = page.locator('#heatmap')
            if heatmap_elem.count() > 0:
                error_elem = heatmap_elem.locator('.error')
                if error_elem.count() > 0:
                    print(f"   ❌ Compare heatmap: ERROR")
                else:
                    print(f"   ✅ Compare heatmap: OK")

            # Take screenshot
            screenshot_path = "test_screenshots/compare_tab.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print(f"   📸 Screenshot: {screenshot_path}")

        except Exception as e:
            print(f"   ❌ Compare tab FAILED: {str(e)}")

        browser.close()

    # Save results to JSON
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    total = len(results['datasets'])
    successful = sum(1 for d in results['datasets'] if d['status'] == 'success')
    failed = total - successful

    print(f"Total datasets tested: {total}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")

    # Count chart successes
    all_charts = {}
    for dataset in results['datasets']:
        for chart, status in dataset['charts'].items():
            if chart not in all_charts:
                all_charts[chart] = {'ok': 0, 'error': 0}
            if 'OK' in str(status):
                all_charts[chart]['ok'] += 1
            elif 'ERROR' in str(status):
                all_charts[chart]['error'] += 1

    print(f"\nChart Statistics:")
    for chart, counts in sorted(all_charts.items()):
        total_charts = counts['ok'] + counts['error']
        if total_charts > 0:
            success_rate = (counts['ok'] / total_charts) * 100
            print(f"  {chart}: {counts['ok']}/{total_charts} OK ({success_rate:.0f}%)")

    print(f"\nDetailed results saved to: test_results.json")
    print(f"Screenshots saved to: test_screenshots/")
    print("="*70)

    return results

if __name__ == '__main__':
    print("="*70)
    print("COMPREHENSIVE DASHBOARD TEST")
    print("="*70)

    import os
    os.makedirs('test_screenshots', exist_ok=True)

    try:
        results = test_all_datasets()
        print("\n✅ All tests completed!")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
