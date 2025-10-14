#!/usr/bin/env python3
"""
Test script to validate Randles 2021 dashboard in headless browser
"""

from playwright.sync_api import sync_playwright
import time
from datetime import datetime

def test_dashboard():
    """Test dashboard and capture screenshot"""

    with sync_playwright() as p:
        # Launch browser in headless mode
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1920, 'height': 3000})

        print("Opening Randles 2021 dashboard...")
        page.goto('http://localhost:8082/dashboard.html')

        # Wait for the page to load
        print("Waiting for data to load...")
        time.sleep(8)

        # Check if stats are loaded
        try:
            stats = page.locator('.stats-container').inner_text()
            print(f"Stats loaded:\n{stats}")
        except Exception as e:
            print(f"Error loading stats: {e}")

        # Check if charts are rendered
        try:
            heatmap = page.locator('#heatmap-chart')
            if heatmap.count() > 0:
                print("✅ Heatmap element found!")

            volcano = page.locator('#volcano-chart')
            if volcano.count() > 0:
                print("✅ Volcano plot element found!")

            scatter = page.locator('#scatter-chart')
            if scatter.count() > 0:
                print("✅ Scatter plot element found!")
        except Exception as e:
            print(f"Error checking charts: {e}")

        # Wait for all charts to render
        time.sleep(5)

        # Capture full page screenshot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot_path = f'randles_dashboard_test_{timestamp}.png'
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"Screenshot saved: {screenshot_path}")

        # Close browser
        browser.close()

        return screenshot_path

if __name__ == '__main__':
    print("="*70)
    print("Testing Randles 2021 Z-Score Dashboard")
    print("="*70)

    try:
        screenshot = test_dashboard()
        print(f"\n✅ Test completed! Screenshot: {screenshot}")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
