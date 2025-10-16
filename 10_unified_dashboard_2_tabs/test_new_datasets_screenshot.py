#!/usr/bin/env python3
"""
Test screenshot for new datasets in ECM Atlas dashboard
"""

from playwright.sync_api import sync_playwright
import time

def capture_dashboard_screenshot():
    """Capture screenshot of dashboard with new datasets"""

    with sync_playwright() as p:
        print("Starting browser...")
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1920, 'height': 3000})

        print("Loading dashboard...")
        page.goto('http://localhost:8083/dashboard.html', wait_until='networkidle')

        # Wait for dropdown to load
        print("Waiting for dataset dropdown...")
        page.wait_for_selector('#dataset-select', timeout=10000)
        time.sleep(2)

        # Select Tsumagari_2023 dataset
        print("Selecting Tsumagari_2023 dataset...")
        page.select_option('#dataset-select', 'Tsumagari_2023')

        # Wait for compartment dropdown (if exists)
        time.sleep(2)

        # Check if compartment selector exists
        compartment_exists = page.locator('#compartment-select').count() > 0
        if compartment_exists:
            print("Selecting Cortex compartment...")
            page.select_option('#compartment-select', 'Cortex')
            time.sleep(1)

        # Wait for visualizations to render
        print("Waiting for visualizations to render...")
        time.sleep(5)

        # Take screenshot
        screenshot_path = '/Users/Kravtsovd/projects/ecm-atlas/screenshots/new_datasets_tsumagari_2023.png'
        print(f"Capturing screenshot to {screenshot_path}...")
        page.screenshot(path=screenshot_path, full_page=True)

        print("✅ Screenshot captured successfully!")

        browser.close()

if __name__ == '__main__':
    try:
        capture_dashboard_screenshot()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
