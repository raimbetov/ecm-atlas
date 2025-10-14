#!/usr/bin/env python3
"""
Capture screenshot of dashboard header to verify version display
"""

from playwright.sync_api import sync_playwright
import time
from datetime import datetime

def capture_dashboard_screenshot():
    """Capture screenshot of dashboard header with version badge"""

    print("🚀 Starting screenshot capture...")

    with sync_playwright() as p:
        print("📱 Launching browser...")
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1920, 'height': 1080})

        # Navigate to dashboard
        url = 'http://localhost:8083/dashboard.html'
        print(f"🌐 Loading: {url}")

        try:
            page.goto(url, wait_until='networkidle', timeout=10000)

            # Wait for version to load
            print("⏳ Waiting for version badge to load...")
            page.wait_for_selector('#version-badge', timeout=5000)
            time.sleep(2)  # Extra wait for animations

            # Check version number
            version_text = page.locator('#version-number').inner_text()
            print(f"✅ Version detected: v{version_text}")

            # Capture full header
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            screenshot_path = f'screenshots/header_with_version_{timestamp}.png'

            header = page.locator('header.header')
            header.screenshot(path=screenshot_path)
            print(f"📸 Screenshot saved: {screenshot_path}")

            # Also capture full page
            full_page_path = f'screenshots/full_dashboard_{timestamp}.png'
            page.screenshot(path=full_page_path, full_page=True)
            print(f"📸 Full page screenshot: {full_page_path}")

            # Get version tooltip
            badge = page.locator('#version-badge')
            tooltip = badge.get_attribute('title')
            if tooltip:
                print(f"\n📋 Version tooltip:\n{tooltip}")

            print("\n✅ All screenshots captured successfully!")

        except Exception as e:
            print(f"❌ Error: {e}")
            print("\n⚠️  Make sure:")
            print("   1. API server is running (port 5004)")
            print("   2. HTTP server is running (port 8083)")
            print("   3. Dashboard is accessible at http://localhost:8083/dashboard.html")
            raise

        finally:
            browser.close()

if __name__ == '__main__':
    import os

    # Create screenshots directory
    os.makedirs('screenshots', exist_ok=True)

    capture_dashboard_screenshot()
