#!/usr/bin/env python3
"""
Test script to validate unified dashboard in headless browser
"""

from playwright.sync_api import sync_playwright
import time
from datetime import datetime

def test_dashboard():
    """Test dashboard and capture screenshot"""

    with sync_playwright() as p:
        # Launch browser in headless mode
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1920, 'height': 1080})

        print("Opening dashboard...")
        page.goto('http://localhost:8081/dashboard_ver1.html')

        # Wait for the page to load
        print("Waiting for data to load...")
        time.sleep(5)

        # Check if stats are loaded
        try:
            stats_text = page.locator('#stats-summary').inner_text()
            print(f"Stats loaded: {stats_text}")
        except Exception as e:
            print(f"Error loading stats: {e}")

        # Check if heatmap is rendered
        try:
            heatmap_element = page.locator('#heatmap')
            if heatmap_element.count() > 0:
                print("Heatmap element found!")
            else:
                print("Heatmap element NOT found")
        except Exception as e:
            print(f"Error checking heatmap: {e}")

        # Wait for any dynamic content
        time.sleep(3)

        # Capture full page screenshot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = f'dashboard_test_{timestamp}.png'
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"Screenshot saved: {screenshot_path}")

        # Close browser
        browser.close()

        return screenshot_path

if __name__ == '__main__':
    print("="*70)
    print("Testing Unified ECM Dashboard")
    print("="*70)

    try:
        screenshot = test_dashboard()
        print(f"\n✅ Test completed! Screenshot: {screenshot}")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
