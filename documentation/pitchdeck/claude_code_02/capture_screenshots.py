#!/usr/bin/env python3
"""
Screenshot capture script for pitch deck validation.
Uses Playwright to capture all 10 slides at 1920x1080 resolution.
"""

import asyncio
import os
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Installing Playwright...")
    import subprocess
    subprocess.run(["pip", "install", "playwright"], check=True)
    subprocess.run(["playwright", "install", "chromium"], check=True)
    from playwright.async_api import async_playwright


async def capture_slides():
    """Capture all 10 slides as PNG screenshots."""

    # Get paths
    script_dir = Path(__file__).parent
    html_file = script_dir / "pitchdeck_improved.html"
    screenshots_dir = script_dir / "screenshots"

    # Create screenshots directory
    screenshots_dir.mkdir(exist_ok=True)

    print(f"HTML file: {html_file}")
    print(f"Screenshots directory: {screenshots_dir}")

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            device_scale_factor=1
        )
        page = await context.new_page()

        # Load HTML file
        await page.goto(f"file://{html_file.absolute()}")

        # Wait for Mermaid to render
        print("Waiting for Mermaid diagrams to render...")
        await asyncio.sleep(3)

        # Capture each slide
        for slide_num in range(1, 11):
            print(f"Capturing slide {slide_num}/10...")

            # Screenshot filename
            screenshot_path = screenshots_dir / f"slide_{slide_num:02d}.png"

            # Take screenshot
            await page.screenshot(
                path=str(screenshot_path),
                full_page=False
            )

            print(f"  ✓ Saved: {screenshot_path.name}")

            # Navigate to next slide (except on last slide)
            if slide_num < 10:
                await page.keyboard.press('ArrowRight')
                await asyncio.sleep(1)  # Wait for transition

        await browser.close()

    print("\n✅ All screenshots captured successfully!")
    print(f"📁 Location: {screenshots_dir}")


if __name__ == "__main__":
    asyncio.run(capture_slides())
