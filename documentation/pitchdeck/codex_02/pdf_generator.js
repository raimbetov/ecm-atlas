const puppeteer = require('puppeteer');
const path = require('path');

async function generatePDF() {
    console.log('Launching browser for PDF export...');
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

    // Wait for Mermaid to render all diagrams
    console.log('Waiting for Mermaid diagrams to render...');
    await new Promise(resolve => setTimeout(resolve, 5000));

    // Make all slides visible for PDF export by temporarily changing CSS
    await page.evaluate(() => {
        // Remove position absolute and opacity transitions
        const slides = document.querySelectorAll('.slide');
        slides.forEach(slide => {
            slide.style.position = 'relative';
            slide.style.opacity = '1';
            slide.style.transform = 'none';
            slide.style.pageBreakAfter = 'always';
        });
    });

    const pdfPath = path.join(__dirname, 'presentation.pdf');

    console.log('Generating PDF...');
    await page.pdf({
        path: pdfPath,
        format: 'A4',
        landscape: true,
        printBackground: true,
        margin: {
            top: '0px',
            right: '0px',
            bottom: '0px',
            left: '0px'
        }
    });

    await browser.close();
    console.log(`✓ PDF generated successfully: ${pdfPath}`);
}

generatePDF().catch(console.error);
