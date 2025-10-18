#!/usr/bin/env python3
"""
Test dashboard in headless browser and capture screenshots.
"""

import time
import subprocess
from pathlib import Path

# Check if Playwright is installed
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Installing Playwright...")
    subprocess.run(["pip", "install", "playwright"], check=True)
    subprocess.run(["playwright", "install", "chromium"], check=True)
    from playwright.sync_api import sync_playwright

OUTPUT_DIR = Path(__file__).parent / "screenshots"
OUTPUT_DIR.mkdir(exist_ok=True)

DASHBOARD_URL = "http://localhost:8083/dashboard.html"


def test_dashboard():
    """Test dashboard and capture screenshots."""
    with sync_playwright() as p:
        print("🚀 Launching headless browser...")
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})

        try:
            # Navigate to dashboard
            print(f"📍 Navigating to {DASHBOARD_URL}...")
            page.goto(DASHBOARD_URL)

            # Wait for page to load
            print("⏳ Waiting for page to load...")
            page.wait_for_selector(".header", timeout=10000)
            time.sleep(3)  # Wait for all content to render

            # Screenshot 1: About tab (default)
            print("📸 Screenshot 1: About tab...")
            page.screenshot(path=OUTPUT_DIR / "01_about_tab.png", full_page=True)
            print(f"   ✅ Saved to {OUTPUT_DIR / '01_about_tab.png'}")

            # Check if Mermaid diagrams rendered
            mermaid_count = page.locator(".mermaid").count()
            print(f"   ℹ️  Mermaid diagrams found: {mermaid_count}")

            # Click Individual Dataset Analysis tab
            print("📸 Screenshot 2: Individual Dataset tab...")
            page.click('button[data-tab="individual"]')
            time.sleep(2)  # Wait for tab to load
            page.screenshot(path=OUTPUT_DIR / "02_individual_tab.png", full_page=True)
            print(f"   ✅ Saved to {OUTPUT_DIR / '02_individual_tab.png'}")

            # Click Compare Datasets tab
            print("📸 Screenshot 3: Compare Datasets tab...")
            page.click('button[data-tab="compare"]')
            time.sleep(2)
            page.screenshot(path=OUTPUT_DIR / "03_compare_tab.png", full_page=True)
            print(f"   ✅ Saved to {OUTPUT_DIR / '03_compare_tab.png'}")

            # Go back to About tab and scroll down
            print("📸 Screenshot 4: About tab (scrolled)...")
            page.click('button[data-tab="about"]')
            time.sleep(2)
            page.evaluate("window.scrollTo(0, 1000)")
            time.sleep(1)
            page.screenshot(path=OUTPUT_DIR / "04_about_tab_scrolled.png", full_page=False)
            print(f"   ✅ Saved to {OUTPUT_DIR / '04_about_tab_scrolled.png'}")

            print("\n✅ All tests completed successfully!")
            print(f"📁 Screenshots saved to: {OUTPUT_DIR}")

            return True

        except Exception as e:
            print(f"\n❌ Error during testing: {e}")
            # Take error screenshot
            page.screenshot(path=OUTPUT_DIR / "error_screenshot.png")
            print(f"   📸 Error screenshot saved to {OUTPUT_DIR / 'error_screenshot.png'}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            browser.close()


def main():
    """Main function."""
    print("=" * 70)
    print("ECM-Atlas Dashboard Testing")
    print("=" * 70)
    print()

    # Run tests
    success = test_dashboard()

    print()
    print("=" * 70)
    if success:
        print("✅ Testing completed successfully!")
    else:
        print("❌ Testing failed. Check error messages above.")
    print("=" * 70)

    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
