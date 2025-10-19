#!/usr/bin/env python3
"""
Test Schuler_2021_Soleus visualizations with headless browser
Captures screenshots to verify volcano and scatter plots work
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import os

# Configuration
DASHBOARD_URL = "http://localhost:5004/dashboard.html"
SCREENSHOT_DIR = "screenshots/soleus_test"
WAIT_TIMEOUT = 20

def setup_driver():
    """Setup headless Chrome driver"""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1200')
    driver = webdriver.Chrome(options=options)
    return driver

def wait_for_element(driver, by, value, timeout=WAIT_TIMEOUT):
    """Wait for element to be present"""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, value))
        )
        return element
    except TimeoutException:
        print(f"Timeout waiting for element: {value}")
        return None

def test_soleus_dataset(driver):
    """Test Schuler_2021_Soleus dataset visualizations"""

    print(f"Loading dashboard: {DASHBOARD_URL}")
    driver.get(DASHBOARD_URL)

    # Wait for page to load
    time.sleep(3)

    # 1. Capture initial state (About tab)
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    driver.save_screenshot(f"{SCREENSHOT_DIR}/01_about_tab.png")
    print("✓ Captured About tab")

    # 2. Switch to Individual Dataset Analysis tab
    print("\n Switching to Individual Dataset Analysis tab...")
    individual_tab = wait_for_element(driver, By.CSS_SELECTOR, '[data-tab="individual"]')
    if individual_tab:
        individual_tab.click()
        time.sleep(2)
        driver.save_screenshot(f"{SCREENSHOT_DIR}/02_individual_tab.png")
        print("✓ Switched to Individual tab")

    # 3. Select Schuler_2021_Soleus from dropdown
    print("\nSelecting Schuler_2021_Soleus dataset...")
    select_element = wait_for_element(driver, By.ID, 'dataset-select')
    if select_element:
        select = Select(select_element)

        # Find the option with Soleus in it
        for option in select.options:
            if 'Soleus' in option.text and 'Skeletal muscle' in option.text:
                print(f"  Found option: {option.text}")
                select.select_by_visible_text(option.text)
                break

        # Wait for data to load
        time.sleep(5)
        driver.save_screenshot(f"{SCREENSHOT_DIR}/03_soleus_selected.png")
        print("✓ Dataset selected and loaded")

    # 4. Check for error messages
    error_elements = driver.find_elements(By.CSS_SELECTOR, '.error, [style*="color: red"], [style*="background: #ffebee"]')
    if error_elements:
        print("\n⚠️  Found error messages:")
        for i, error in enumerate(error_elements):
            if error.is_displayed():
                error_text = error.text
                if error_text:
                    print(f"  Error {i+1}: {error_text[:100]}")
    else:
        print("\n✓ No error messages found")

    # 5. Capture Soleus compartment
    print("\nCapturing Soleus compartment visualizations...")
    soleus_button = wait_for_element(driver, By.XPATH, "//button[contains(text(), 'Soleus')]")
    if soleus_button:
        soleus_button.click()
        time.sleep(3)
        driver.save_screenshot(f"{SCREENSHOT_DIR}/04_soleus_compartment.png")
        print("✓ Soleus compartment selected")

    # 6. Scroll to volcano plot
    print("\nCapturing Volcano plot...")
    try:
        volcano_section = driver.find_element(By.XPATH, "//h3[contains(text(), 'Volcano Plot')]")
        driver.execute_script("arguments[0].scrollIntoView(true);", volcano_section)
        time.sleep(2)
        driver.save_screenshot(f"{SCREENSHOT_DIR}/05_volcano_plot.png")
        print("✓ Volcano plot captured")
    except Exception as e:
        print(f"Could not find volcano plot: {e}")

    # 7. Scroll to scatter plot
    print("\nCapturing Scatter plot...")
    try:
        scatter_section = driver.find_element(By.XPATH, "//h3[contains(text(), 'Scatter Plot')]")
        driver.execute_script("arguments[0].scrollIntoView(true);", scatter_section)
        time.sleep(2)
        driver.save_screenshot(f"{SCREENSHOT_DIR}/06_scatter_plot.png")
        print("✓ Scatter plot captured")
    except Exception as e:
        print(f"Could not find scatter plot: {e}")

    # 8. Capture full page
    print("\nCapturing full page...")
    driver.save_screenshot(f"{SCREENSHOT_DIR}/07_full_page.png")
    print("✓ Full page captured")

    # 9. Check console logs for errors
    print("\nChecking browser console logs...")
    logs = driver.get_log('browser')
    if logs:
        print("Console logs:")
        for log in logs:
            level = log['level']
            message = log['message']
            if level in ['SEVERE', 'ERROR']:
                print(f"  [{level}] {message}")
    else:
        print("✓ No console errors")

def main():
    """Main test function"""
    driver = None
    try:
        print("="*70)
        print("Testing Schuler_2021_Soleus Visualizations")
        print("="*70)

        driver = setup_driver()
        test_soleus_dataset(driver)

        print("\n" + "="*70)
        print(f"✓ Test completed! Screenshots saved to: {SCREENSHOT_DIR}")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if driver:
            driver.quit()

if __name__ == '__main__':
    main()
