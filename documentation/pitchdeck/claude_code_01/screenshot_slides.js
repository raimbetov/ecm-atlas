const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

async function captureSlides() {
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });

    const htmlPath = path.join(__dirname, 'pitchdeck_improved.html');
    const htmlUrl = `file://${htmlPath}`;

    console.log(`Loading HTML from: ${htmlUrl}`);
    await page.goto(htmlUrl, { waitUntil: 'networkidle0' });

    // Wait for Mermaid diagrams to render
    console.log('Waiting for Mermaid diagrams to render...');
    await new Promise(resolve => setTimeout(resolve, 3000));

    const screenshotsDir = path.join(__dirname, 'screenshots');
    if (!fs.existsSync(screenshotsDir)) {
        fs.mkdirSync(screenshotsDir, { recursive: true });
    }

    // Capture all 10 slides
    for (let i = 1; i <= 10; i++) {
        console.log(`Capturing slide ${i}...`);

        const screenshotPath = path.join(screenshotsDir, `slide_${String(i).padStart(2, '0')}.png`);
        await page.screenshot({
            path: screenshotPath,
            fullPage: false
        });

        console.log(`Saved: ${screenshotPath}`);

        // Navigate to next slide (except on last slide)
        if (i < 10) {
            await page.keyboard.press('ArrowRight');
            // Wait for slide transition and Mermaid rendering
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
    }

    console.log('All slides captured successfully!');
    await browser.close();
}

captureSlides().catch(console.error);
