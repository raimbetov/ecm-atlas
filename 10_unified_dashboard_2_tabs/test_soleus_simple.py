#!/usr/bin/env python3
"""
Simple test to capture Soleus visualizations after manual navigation
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os

DASHBOARD_URL = "http://localhost:5004/dashboard.html"
SCREENSHOT_DIR = "screenshots/soleus_final"

def capture_screenshots():
    """Capture screenshots with delays to allow full page load"""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless=new')
    options.add_argument('--window-size=1920,3000')  # Tall window to capture everything
    driver = webdriver.Chrome(options=options)

    try:
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)

        print(f"Loading: {DASHBOARD_URL}")
        driver.get(DASHBOARD_URL)

        # Wait for page to fully load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '.tab-button'))
        )
        time.sleep(5)

        # Capture About tab
        driver.save_screenshot(f"{SCREENSHOT_DIR}/1_about.png")
        print("✓ Captured About tab")

        # Switch to Individual tab
        individual_btn = driver.find_element(By.CSS_SELECTOR, '[data-tab="individual"]')
        individual_btn.click()
        time.sleep(3)
        driver.save_screenshot(f"{SCREENSHOT_DIR}/2_individual_empty.png")
        print("✓ Captured Individual tab (empty)")

        # Execute JavaScript to select Soleus dataset and load it
        print("Loading Soleus dataset via JavaScript...")
        driver.execute_script("""
            const select = document.getElementById('dataset-select');
            for (let option of select.options) {
                if (option.text.includes('Soleus') && option.text.includes('Skeletal muscle')) {
                    select.value = option.value;
                    select.dispatchEvent(new Event('change', { bubbles: true }));
                    break;
                }
            }
        """)

        # Wait for data to load (check for compartment buttons to appear)
        time.sleep(8)
        driver.save_screenshot(f"{SCREENSHOT_DIR}/3_soleus_loaded.png")
        print("✓ Captured Soleus loaded")

        # Click Soleus compartment button if it exists
        try:
            driver.execute_script("""
                const buttons = document.querySelectorAll('.compartment-button');
                for (let btn of buttons) {
                    if (btn.textContent.includes('Soleus')) {
                        btn.click();
                        break;
                    }
                }
            """)
            time.sleep(5)
            driver.save_screenshot(f"{SCREENSHOT_DIR}/4_soleus_compartment.png")
            print("✓ Captured Soleus compartment selected")
        except Exception as e:
            print(f"Could not click compartment button: {e}")

        # Capture full scrollable page
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)
        driver.save_screenshot(f"{SCREENSHOT_DIR}/5_top.png")

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(1)
        driver.save_screenshot(f"{SCREENSHOT_DIR}/6_middle.png")

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        driver.save_screenshot(f"{SCREENSHOT_DIR}/7_bottom.png")

        print(f"\n✓ All screenshots saved to: {SCREENSHOT_DIR}")

        # Print any JavaScript errors
        logs = driver.get_log('browser')
        errors = [log for log in logs if log['level'] in ['SEVERE', 'ERROR']]
        if errors:
            print("\n⚠️  JavaScript errors found:")
            for error in errors:
                print(f"  {error['message']}")

    finally:
        driver.quit()

if __name__ == '__main__':
    capture_screenshots()
