#!/usr/bin/env python3
"""
Test the Compare Datasets tab (second tab) in headless browser
Captures screenshots and checks for JavaScript errors
"""

from playwright.sync_api import sync_playwright
import time
import os
from datetime import datetime

def test_compare_tab():
    """Test second tab with headless browser"""

    print("🧪 Testing Compare Datasets Tab...")
    print("="*70)

    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = context.new_page()

        # Track console messages and errors
        console_messages = []
        js_errors = []

        page.on('console', lambda msg: console_messages.append(f"[{msg.type}] {msg.text}"))
        page.on('pageerror', lambda err: js_errors.append(str(err)))

        try:
            # Load dashboard
            print("\n1️⃣ Loading dashboard at http://localhost:8080/dashboard.html")
            page.goto('http://localhost:8080/dashboard.html', wait_until='networkidle')
            time.sleep(2)

            # Take screenshot of initial state (Individual tab)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            screenshot_dir = 'test_screenshots'
            os.makedirs(screenshot_dir, exist_ok=True)

            initial_screenshot = f'{screenshot_dir}/01_individual_tab_{timestamp}.png'
            page.screenshot(path=initial_screenshot, full_page=True)
            print(f"   ✅ Saved: {initial_screenshot}")

            # Check if Compare button exists
            compare_button = page.query_selector('button[data-tab="compare"]')
            if not compare_button:
                print("   ❌ ERROR: Compare button not found!")
                return False

            print("\n2️⃣ Clicking on 'Compare Datasets' tab...")
            compare_button.click()
            time.sleep(3)  # Wait for content to load

            # Take screenshot after clicking
            after_click_screenshot = f'{screenshot_dir}/02_after_click_{timestamp}.png'
            page.screenshot(path=after_click_screenshot, full_page=True)
            print(f"   ✅ Saved: {after_click_screenshot}")

            # Check if compare content is visible
            compare_content = page.query_selector('#tab-compare')
            if not compare_content:
                print("   ❌ ERROR: Compare tab content div not found!")
                return False

            is_visible = compare_content.is_visible()
            print(f"   Compare content visible: {is_visible}")

            # Check for filters panel
            filters_panel = page.query_selector('.filters-panel')
            if filters_panel:
                print(f"   ✅ Filters panel found: {filters_panel.is_visible()}")
            else:
                print("   ⚠️  Filters panel NOT found")

            # Check for heatmap container
            heatmap = page.query_selector('#heatmap')
            if heatmap:
                heatmap_content = heatmap.inner_html()
                print(f"   Heatmap container: {len(heatmap_content)} chars")
                if 'Loading' in heatmap_content:
                    print("   ⚠️  Heatmap still shows 'Loading...'")
                elif 'No data' in heatmap_content or 'No proteins' in heatmap_content:
                    print("   ⚠️  Heatmap shows 'No data' message")
                else:
                    print("   ✅ Heatmap has content")
            else:
                print("   ❌ Heatmap container NOT found")

            # Wait a bit more for async loading
            print("\n3️⃣ Waiting 5 seconds for async data loading...")
            time.sleep(5)

            # Final screenshot
            final_screenshot = f'{screenshot_dir}/03_final_state_{timestamp}.png'
            page.screenshot(path=final_screenshot, full_page=True)
            print(f"   ✅ Saved: {final_screenshot}")

            # Print console messages
            print("\n📜 Console Messages:")
            print("-"*70)
            for msg in console_messages[-20:]:  # Last 20 messages
                print(f"   {msg}")

            # Print JavaScript errors
            if js_errors:
                print("\n❌ JavaScript Errors:")
                print("-"*70)
                for err in js_errors:
                    print(f"   {err}")
                return False
            else:
                print("\n✅ No JavaScript errors detected")

            # Check API calls in network log
            print("\n🌐 Checking API calls...")

            # Try to make API call directly to test endpoint
            import requests
            try:
                response = requests.get('http://localhost:5004/api/compare/filters', timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✅ /api/compare/filters: {response.status_code}")
                    print(f"      Organs: {len(data.get('organs', []))}")
                    print(f"      Compartments: {len(data.get('compartments', []))}")
                    print(f"      Categories: {len(data.get('categories', []))}")
                else:
                    print(f"   ❌ /api/compare/filters: {response.status_code}")
            except Exception as e:
                print(f"   ❌ API call failed: {e}")

            return True

        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            error_screenshot = f'{screenshot_dir}/ERROR_{timestamp}.png'
            page.screenshot(path=error_screenshot, full_page=True)
            print(f"   Error screenshot: {error_screenshot}")
            return False

        finally:
            browser.close()
            print("\n" + "="*70)
            print("Test complete!")

if __name__ == '__main__':
    success = test_compare_tab()
    exit(0 if success else 1)
