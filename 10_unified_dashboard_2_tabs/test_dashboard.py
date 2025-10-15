#!/usr/bin/env python3
from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    
    page.on("console", lambda msg: print(f"[Console] {msg.text}"))
    
    print("Loading dashboard...")
    page.goto("http://localhost:8083/dashboard.html")
    
    # Wait for page to be ready
    time.sleep(3)
    
    # Get dropdown HTML to check options
    select_html = page.locator("#dataset-select").inner_html()
    
    # Check if Ouni_2022 is in the options
    if "Ouni_2022" in select_html or "Ouni 2022" in select_html:
        print("✅ Ouni 2022 found in dropdown!")
    else:
        print("❌ Ouni 2022 NOT found in dropdown")
        print(f"Dropdown HTML: {select_html[:500]}")
    
    page.screenshot(path="screenshots/dashboard_test.png", full_page=True)
    print("Screenshot saved to screenshots/dashboard_test.png")
    
    browser.close()
