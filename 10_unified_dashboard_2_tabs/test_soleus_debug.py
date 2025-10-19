#!/usr/bin/env python3
"""
Debug test for Soleus visualizations - captures console logs and DOM state
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json

DASHBOARD_URL = "http://localhost:5004/dashboard.html"

def debug_test():
    """Run debug test with detailed logging"""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless=new')
    options.add_argument('--window-size=1920,2400')
    # Enable detailed logging
    options.set_capability('goog:loggingPrefs', {'browser': 'ALL', 'performance': 'ALL'})

    driver = webdriver.Chrome(options=options)

    try:
        print("="*70)
        print("SOLEUS VISUALIZATION DEBUG TEST")
        print("="*70)

        print(f"\n1. Loading: {DASHBOARD_URL}")
        driver.get(DASHBOARD_URL)
        time.sleep(3)

        # Check if JavaScript loaded
        print("\n2. Checking JavaScript globals...")
        js_check = driver.execute_script("""
            return {
                hasECMAtlas: typeof window.ECMAtlas !== 'undefined',
                hasIndividualDataset: typeof window.IndividualDataset !== 'undefined',
                API_BASE: window.API_BASE || 'not set'
            };
        """)
        print(f"   ECMAtlas: {js_check['hasECMAtlas']}")
        print(f"   IndividualDataset: {js_check['hasIndividualDataset']}")
        print(f"   API_BASE: {js_check['API_BASE']}")

        # Wait for datasets to load
        print("\n3. Waiting for datasets to load...")
        time.sleep(5)

        # Check datasets in global state
        datasets_check = driver.execute_script("""
            return {
                datasetsLength: window.ECMAtlas?.globalData?.datasets?.length || 0,
                datasets: window.ECMAtlas?.globalData?.datasets?.map(d => d.name) || []
            };
        """)
        print(f"   Loaded datasets: {datasets_check['datasetsLength']}")
        if datasets_check['datasets']:
            print(f"   Dataset names: {', '.join(datasets_check['datasets'][:5])}...")

        # Switch to Individual tab
        print("\n4. Switching to Individual Dataset Analysis tab...")
        individual_btn = driver.find_element(By.CSS_SELECTOR, '[data-tab="individual"]')
        individual_btn.click()
        time.sleep(2)

        # Check dropdown options
        print("\n5. Checking dropdown options...")
        options_check = driver.execute_script("""
            const select = document.getElementById('dataset-select');
            return {
                optionCount: select.options.length,
                options: Array.from(select.options).map(o => o.text),
                hasSoleus: Array.from(select.options).some(o => o.text.includes('Soleus'))
            };
        """)
        print(f"   Total options: {options_check['optionCount']}")
        print(f"   Has Soleus: {options_check['hasSoleus']}")
        if options_check['hasSoleus']:
            soleus_option = [o for o in options_check['options'] if 'Soleus' in o][0]
            print(f"   Soleus option: {soleus_option}")

        # Select Soleus dataset
        print("\n6. Selecting Schuler_2021_Soleus dataset...")
        result = driver.execute_script("""
            const select = document.getElementById('dataset-select');
            for (let option of select.options) {
                if (option.text.includes('Soleus') && option.text.includes('Skeletal muscle')) {
                    console.log('Found Soleus option:', option.text);
                    select.value = option.value;
                    select.dispatchEvent(new Event('change', { bubbles: true }));
                    return {success: true, value: option.value, text: option.text};
                }
            }
            return {success: false};
        """)
        print(f"   Selection result: {result}")

        # Wait for data to load
        print("\n7. Waiting for data to load (10 seconds)...")
        time.sleep(10)

        # Check if content was rendered
        content_check = driver.execute_script("""
            const content = document.getElementById('individual-content');
            return {
                hasContent: content.innerHTML.length > 100,
                contentLength: content.innerHTML.length,
                hasHeatmap: content.innerHTML.includes('Heatmap'),
                hasVolcano: content.innerHTML.includes('Volcano'),
                hasScatter: content.innerHTML.includes('Scatter')
            };
        """)
        print(f"   Content rendered: {content_check}")

        # Get console logs
        print("\n8. Browser console logs:")
        logs = driver.get_log('browser')
        for log in logs:
            level = log['level']
            message = log['message']
            if 'favicon' not in message:  # Skip favicon errors
                print(f"   [{level}] {message}")

        # Check for API errors
        print("\n9. Checking API responses...")
        api_check = driver.execute_script("""
            // Return any fetch errors stored
            return window.lastAPIError || 'No errors';
        """)
        print(f"   API status: {api_check}")

        # Final screenshot
        print("\n10. Taking final screenshot...")
        driver.save_screenshot("screenshots/soleus_debug_final.png")
        print("   ✓ Saved to screenshots/soleus_debug_final.png")

        print("\n" + "="*70)
        print("DEBUG TEST COMPLETE")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.quit()

if __name__ == '__main__':
    debug_test()
