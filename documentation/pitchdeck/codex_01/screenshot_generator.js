const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function generateScreenshots() {
    console.log('🚀 Starting screenshot generation...');

    // Create screenshots directory
    const screenshotsDir = path.join(__dirname, 'screenshots');
    if (!fs.existsSync(screenshotsDir)) {
        fs.mkdirSync(screenshotsDir, { recursive: true });
    }

    // Launch browser
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const page = await browser.newPage();

    // Set viewport to 1920x1080 (standard presentation resolution)
    await page.setViewport({
        width: 1920,
        height: 1080,
        deviceScaleFactor: 2  // High DPI for sharp text
    });

    // Open the improved HTML file
    const htmlPath = `file://${path.join(__dirname, 'pitchdeck_improved.html')}`;
    console.log(`📄 Opening HTML: ${htmlPath}`);

    await page.goto(htmlPath, {
        waitUntil: 'networkidle0',
        timeout: 30000
    });

    // Wait for Mermaid diagrams to render
    await new Promise(resolve => setTimeout(resolve, 3000));

    console.log('📸 Starting screenshot capture...');

    // Capture each slide
    for (let slideNum = 1; slideNum <= 10; slideNum++) {
        console.log(`  Capturing slide ${slideNum}/10...`);

        // Navigate to slide using JavaScript
        await page.evaluate((num) => {
            window.currentSlide = num;
            window.updateSlide();
        }, slideNum);

        // Wait for animations and Mermaid rendering
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Take screenshot
        const screenshotPath = path.join(screenshotsDir, `slide_${String(slideNum).padStart(2, '0')}.png`);
        await page.screenshot({
            path: screenshotPath,
            fullPage: false
        });

        console.log(`  ✅ Saved: ${screenshotPath}`);
    }

    console.log('\n🎉 All screenshots captured successfully!');
    console.log(`📂 Location: ${screenshotsDir}`);

    await browser.close();
}

// Run the script
generateScreenshots().catch(error => {
    console.error('❌ Error generating screenshots:', error);
    process.exit(1);
});
