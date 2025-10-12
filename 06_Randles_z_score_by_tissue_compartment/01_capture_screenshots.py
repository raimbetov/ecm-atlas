#!/usr/bin/env python3
"""
Headless Browser Screenshot Capture for Z-Score Dashboard
Takes screenshots of all visualizations to verify rendering
"""

import asyncio
from playwright.async_api import async_playwright
import os
from datetime import datetime

async def capture_dashboard_screenshots():
    """Capture screenshots of dashboard to verify all charts render correctly"""

    # Create screenshots directory
    screenshot_dir = "01_screenshots"
    os.makedirs(screenshot_dir, exist_ok=True)

    async with async_playwright() as p:
        # Launch browser in headless mode
        print("🚀 Launching headless browser...")
        browser = await p.chromium.launch(
            headless=True,
            args=['--disable-web-security', '--disable-features=IsolateOrigins,site-per-process']
        )

        # Create context with high resolution viewport
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            device_scale_factor=2  # Retina display
        )

        page = await context.new_page()

        try:
            # Navigate to dashboard
            dashboard_url = 'http://localhost:8080/dashboard.html'
            print(f"📊 Loading dashboard: {dashboard_url}")

            response = await page.goto(dashboard_url, wait_until='networkidle', timeout=30000)

            if not response.ok:
                print(f"❌ Failed to load dashboard: HTTP {response.status}")
                return

            print("✅ Dashboard loaded successfully")

            # Wait for Plotly to initialize
            await page.wait_for_function("typeof Plotly !== 'undefined'", timeout=10000)
            print("✅ Plotly library loaded")

            # Wait for all charts to render (longer wait for Plotly)
            print("⏳ Waiting for charts to render (20 seconds)...")
            await asyncio.sleep(20)

            # Scroll to top
            await page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(1)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1. Full page screenshot
            print("\n1️⃣  Capturing full page screenshot...")
            await page.screenshot(
                path=f"{screenshot_dir}/00_full_page_{timestamp}.png",
                full_page=True
            )
            print(f"✅ Saved: {screenshot_dir}/00_full_page_{timestamp}.png")

            # 2. Header section
            print("\n2️⃣  Capturing header section...")
            header = await page.query_selector('.header')
            if header:
                await header.screenshot(path=f"{screenshot_dir}/01_header_{timestamp}.png")
                print(f"✅ Saved: {screenshot_dir}/01_header_{timestamp}.png")

            # 3. Stats cards
            print("\n3️⃣  Capturing stats cards...")
            stats = await page.query_selector('.stats-container')
            if stats:
                await stats.screenshot(path=f"{screenshot_dir}/02_stats_{timestamp}.png")
                print(f"✅ Saved: {screenshot_dir}/02_stats_{timestamp}.png")

            # 4. Individual chart screenshots with tab navigation
            print("\n4️⃣  Capturing individual charts...")

            idx = 3
            # Heatmap - Glomerular (default)
            await page.evaluate("document.getElementById('heatmap-chart').scrollIntoView({behavior: 'smooth', block: 'center'})")
            await asyncio.sleep(2)
            chart = await page.query_selector('#heatmap-chart')
            if chart:
                await chart.screenshot(path=f"{screenshot_dir}/{idx:02d}_heatmap-glom_{timestamp}.png")
                print(f"✅ Saved: {screenshot_dir}/{idx:02d}_heatmap-glom_{timestamp}.png - Glomerular Heatmap")
                idx += 1

            # Heatmap - Tubulointerstitial (click tab)
            await page.evaluate("window.loadHeatmap('tubulointerstitial')")
            await asyncio.sleep(3)
            chart = await page.query_selector('#heatmap-chart')
            if chart:
                await chart.screenshot(path=f"{screenshot_dir}/{idx:02d}_heatmap-tubu_{timestamp}.png")
                print(f"✅ Saved: {screenshot_dir}/{idx:02d}_heatmap-tubu_{timestamp}.png - Tubulointerstitial Heatmap")
                idx += 1

            # Volcano - Glomerular (default)
            await page.evaluate("document.getElementById('volcano-chart').scrollIntoView({behavior: 'smooth', block: 'center'})")
            await asyncio.sleep(2)
            chart = await page.query_selector('#volcano-chart')
            if chart:
                await chart.screenshot(path=f"{screenshot_dir}/{idx:02d}_volcano-glom_{timestamp}.png")
                print(f"✅ Saved: {screenshot_dir}/{idx:02d}_volcano-glom_{timestamp}.png - Glomerular Volcano")
                idx += 1

            # Volcano - Tubulointerstitial
            await page.evaluate("window.loadVolcano('tubulointerstitial')")
            await asyncio.sleep(2)
            chart = await page.query_selector('#volcano-chart')
            if chart:
                await chart.screenshot(path=f"{screenshot_dir}/{idx:02d}_volcano-tubu_{timestamp}.png")
                print(f"✅ Saved: {screenshot_dir}/{idx:02d}_volcano-tubu_{timestamp}.png - Tubulointerstitial Volcano")
                idx += 1

            # Scatter - Glomerular (default)
            await page.evaluate("document.getElementById('scatter-chart').scrollIntoView({behavior: 'smooth', block: 'center'})")
            await asyncio.sleep(2)
            chart = await page.query_selector('#scatter-chart')
            if chart:
                await chart.screenshot(path=f"{screenshot_dir}/{idx:02d}_scatter-glom_{timestamp}.png")
                print(f"✅ Saved: {screenshot_dir}/{idx:02d}_scatter-glom_{timestamp}.png - Glomerular Scatter")
                idx += 1

            # Scatter - Tubulointerstitial
            await page.evaluate("window.loadScatter('tubulointerstitial')")
            await asyncio.sleep(2)
            chart = await page.query_selector('#scatter-chart')
            if chart:
                await chart.screenshot(path=f"{screenshot_dir}/{idx:02d}_scatter-tubu_{timestamp}.png")
                print(f"✅ Saved: {screenshot_dir}/{idx:02d}_scatter-tubu_{timestamp}.png - Tubulointerstitial Scatter")
                idx += 1

            # Bars - Glomerular (default)
            await page.evaluate("document.getElementById('bars-chart').scrollIntoView({behavior: 'smooth', block: 'center'})")
            await asyncio.sleep(2)
            chart = await page.query_selector('#bars-chart')
            if chart:
                await chart.screenshot(path=f"{screenshot_dir}/{idx:02d}_bars-glom_{timestamp}.png")
                print(f"✅ Saved: {screenshot_dir}/{idx:02d}_bars-glom_{timestamp}.png - Glomerular Bars")
                idx += 1

            # Bars - Tubulointerstitial
            await page.evaluate("window.loadBars('tubulointerstitial')")
            await asyncio.sleep(2)
            chart = await page.query_selector('#bars-chart')
            if chart:
                await chart.screenshot(path=f"{screenshot_dir}/{idx:02d}_bars-tubu_{timestamp}.png")
                print(f"✅ Saved: {screenshot_dir}/{idx:02d}_bars-tubu_{timestamp}.png - Tubulointerstitial Bars")
                idx += 1

            # Histograms (both visible)
            await page.evaluate("document.getElementById('histogram-glom').scrollIntoView({behavior: 'smooth', block: 'center'})")
            await asyncio.sleep(2)
            chart = await page.query_selector('#histogram-glom')
            if chart:
                await chart.screenshot(path=f"{screenshot_dir}/{idx:02d}_hist-glom_{timestamp}.png")
                print(f"✅ Saved: {screenshot_dir}/{idx:02d}_hist-glom_{timestamp}.png - Glomerular Histogram")
                idx += 1

            chart = await page.query_selector('#histogram-tubu')
            if chart:
                await chart.screenshot(path=f"{screenshot_dir}/{idx:02d}_hist-tubu_{timestamp}.png")
                print(f"✅ Saved: {screenshot_dir}/{idx:02d}_hist-tubu_{timestamp}.png - Tubulointerstitial Histogram")
                idx += 1

            # Comparison
            await page.evaluate("document.getElementById('comparison-chart').scrollIntoView({behavior: 'smooth', block: 'center'})")
            await asyncio.sleep(2)
            chart = await page.query_selector('#comparison-chart')
            if chart:
                await chart.screenshot(path=f"{screenshot_dir}/{idx:02d}_comparison_{timestamp}.png")
                print(f"✅ Saved: {screenshot_dir}/{idx:02d}_comparison_{timestamp}.png - Compartment Comparison")
                idx += 1

            # 5. Viewport screenshots at different scroll positions
            print("\n5️⃣  Capturing viewport screenshots...")
            scroll_positions = [0, 800, 1600, 2400, 3200]
            for idx, scroll_y in enumerate(scroll_positions):
                await page.evaluate(f"window.scrollTo(0, {scroll_y})")
                await asyncio.sleep(0.5)
                await page.screenshot(
                    path=f"{screenshot_dir}/viewport_{idx:02d}_scroll{scroll_y}_{timestamp}.png"
                )
                print(f"✅ Saved viewport at scroll position {scroll_y}px")

            # 6. Check for JavaScript errors
            print("\n6️⃣  Checking for JavaScript errors...")
            errors = await page.evaluate("""
                () => {
                    const errors = [];
                    // Check if Plotly rendered all charts
                    const chartIds = ['heatmap-chart', 'volcano-chart', 'scatter-chart', 'bars-chart',
                                     'histogram-glom', 'histogram-tubu', 'comparison-chart'];

                    chartIds.forEach(id => {
                        const element = document.getElementById(id);
                        if (!element) {
                            errors.push(`Missing element: ${id}`);
                        } else if (!element.querySelector('.plotly')) {
                            errors.push(`Chart not rendered: ${id}`);
                        }
                    });

                    return errors;
                }
            """)

            if errors:
                print("❌ Errors found:")
                for error in errors:
                    print(f"   - {error}")
            else:
                print("✅ All charts rendered successfully!")

            # 7. Summary report
            print("\n" + "="*60)
            print("📸 SCREENSHOT CAPTURE COMPLETE")
            print("="*60)
            print(f"\n📁 Screenshots saved in: {screenshot_dir}/")
            print(f"🕐 Timestamp: {timestamp}")

            # Count files
            screenshot_count = len([f for f in os.listdir(screenshot_dir) if f.endswith('.png') and timestamp in f])
            print(f"📊 Total screenshots captured: {screenshot_count}")

            if not errors:
                print("\n✅ Dashboard is working correctly! All charts rendered.")
            else:
                print(f"\n⚠️  Found {len(errors)} rendering issues - check screenshots")

        except Exception as e:
            print(f"\n❌ Error capturing screenshots: {e}")
            import traceback
            traceback.print_exc()

        finally:
            await browser.close()
            print("\n🏁 Browser closed")

if __name__ == "__main__":
    print("="*60)
    print("Z-Score Dashboard Screenshot Capture")
    print("="*60)
    print("\n⚠️  IMPORTANT: Make sure HTTP server is running!")
    print("Run in another terminal: python3 00_start_server.py")
    print("\nStarting screenshot capture in 3 seconds...")
    print("="*60 + "\n")

    import time
    time.sleep(3)

    asyncio.run(capture_dashboard_screenshots())
