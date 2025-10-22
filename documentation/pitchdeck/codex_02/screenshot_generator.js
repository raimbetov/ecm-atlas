const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

async function generateScreenshots() {
    console.log('Launching browser...');
    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });

    const htmlPath = path.join(__dirname, 'pitchdeck_improved.html');
    const htmlUrl = `file://${htmlPath}`;

    console.log(`Loading presentation from: ${htmlUrl}`);
    await page.goto(htmlUrl, { waitUntil: 'networkidle0' });

    // Wait for Mermaid to render
    await new Promise(resolve => setTimeout(resolve, 3000));

    const screenshotsDir = path.join(__dirname, 'screenshots');
    if (!fs.existsSync(screenshotsDir)) {
        fs.mkdirSync(screenshotsDir, { recursive: true });
    }

    // Screenshot all 10 slides (default theme)
    console.log('Generating screenshots for all 10 slides...');
    for (let slideNum = 1; slideNum <= 10; slideNum++) {
        console.log(`  Capturing slide ${slideNum}/10...`);

        // Navigate to slide programmatically
        await page.evaluate((num) => {
            window.currentSlide = num;
            window.updateSlide();
        }, slideNum);

        // Wait for slide transition
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Wait for Mermaid diagrams to render
        await new Promise(resolve => setTimeout(resolve, 2000));

        const filename = path.join(screenshotsDir, `slide_${String(slideNum).padStart(2, '0')}.png`);
        await page.screenshot({ path: filename, fullPage: false });
        console.log(`  ✓ Saved: ${filename}`);
    }

    // Generate theme comparison screenshot
    console.log('\nGenerating theme comparison screenshot...');

    // Go back to slide 1 for comparison
    await page.evaluate(() => {
        window.currentSlide = 1;
        window.updateSlide();
    });
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Create a composite image showing all 3 themes
    const themes = ['dark-blue', 'pure-black', 'high-contrast'];
    const themeScreenshots = [];

    for (const theme of themes) {
        console.log(`  Capturing theme: ${theme}...`);

        await page.evaluate((themeName) => {
            window.applyTheme(themeName);
        }, theme);

        await new Promise(resolve => setTimeout(resolve, 2000)); // Wait for theme change and Mermaid re-render

        const filename = path.join(screenshotsDir, `theme_${theme}.png`);
        await page.screenshot({ path: filename, fullPage: false });
        themeScreenshots.push(filename);
        console.log(`  ✓ Saved: ${filename}`);
    }

    // Font size comparison
    console.log('\nGenerating font size comparison screenshot...');

    await page.evaluate(() => {
        window.applyTheme('dark-blue');
        window.fontSizeMultiplier = 0.8;
        window.updateFontSize();
    });
    await new Promise(resolve => setTimeout(resolve, 1500));
    await page.screenshot({
        path: path.join(screenshotsDir, 'fontsize_small.png'),
        fullPage: false
    });

    await page.evaluate(() => {
        window.fontSizeMultiplier = 1.0;
        window.updateFontSize();
    });
    await new Promise(resolve => setTimeout(resolve, 1500));
    await page.screenshot({
        path: path.join(screenshotsDir, 'fontsize_normal.png'),
        fullPage: false
    });

    await page.evaluate(() => {
        window.fontSizeMultiplier = 1.3;
        window.updateFontSize();
    });
    await new Promise(resolve => setTimeout(resolve, 1500));
    await page.screenshot({
        path: path.join(screenshotsDir, 'fontsize_large.png'),
        fullPage: false
    });

    console.log('✓ Font size comparison screenshots saved');

    // Settings panel screenshot
    console.log('\nCapturing settings panel...');
    await page.evaluate(() => {
        window.fontSizeMultiplier = 1.0;
        window.updateFontSize();
        window.toggleSettings();
    });
    await new Promise(resolve => setTimeout(resolve, 500));
    await page.screenshot({
        path: path.join(screenshotsDir, 'settings_panel.png'),
        fullPage: false
    });
    console.log('✓ Settings panel screenshot saved');

    await browser.close();
    console.log('\n✓ All screenshots generated successfully!');
    console.log(`Total screenshots: ${10 + themes.length + 4} files`);
}

generateScreenshots().catch(console.error);
