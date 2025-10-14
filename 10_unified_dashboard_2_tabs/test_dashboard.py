#!/usr/bin/env python3
"""
Test unified dashboard v2
"""

from playwright.sync_api import sync_playwright
import time
from datetime import datetime

def test_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1920, 'height': 1080})

        print("Opening unified dashboard v2...")
        page.goto('http://localhost:8083/dashboard.html')

        print("Waiting for data to load...")
        time.sleep(8)

        # Check global stats
        try:
            stats = page.locator('#global-stats').inner_text()
            print(f"✅ Global stats loaded:\n{stats}")
        except Exception as e:
            print(f"❌ Error loading stats: {e}")

        # Check tab navigation
        try:
            tabs = page.locator('.tab-button').count()
            print(f"✅ Found {tabs} tabs")
        except Exception as e:
            print(f"❌ Error checking tabs: {e}")

        # Check dataset selector
        try:
            selector = page.locator('#dataset-select')
            if selector.count() > 0:
                print("✅ Dataset selector found")
        except Exception as e:
            print(f"❌ Error checking selector: {e}")

        # Screenshot of initial state
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = f'unified_dashboard_v2_test_{timestamp}.png'
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"📸 Screenshot saved: {screenshot_path}")

        browser.close()
        return screenshot_path

if __name__ == '__main__':
    print("="*70)
    print("Testing Unified Dashboard v2")
    print("="*70)

    try:
        screenshot = test_dashboard()
        print(f"\n✅ Test completed! Screenshot: {screenshot}")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
