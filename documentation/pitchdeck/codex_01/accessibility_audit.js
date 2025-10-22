const puppeteer = require('puppeteer');
const { AxePuppeteer } = require('@axe-core/puppeteer');
const fs = require('fs');
const path = require('path');

async function runAccessibilityAudit() {
    console.log('🔍 Starting accessibility audit with axe-core...');

    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });

    const htmlPath = `file://${path.join(__dirname, 'pitchdeck_improved.html')}`;
    console.log(`📄 Opening HTML: ${htmlPath}`);

    await page.goto(htmlPath, { waitUntil: 'networkidle0', timeout: 30000 });
    await new Promise(resolve => setTimeout(resolve, 3000));

    console.log('🔬 Running axe-core accessibility tests...\n');

    const results = await new AxePuppeteer(page)
        .withTags(['wcag2a', 'wcag2aa', 'wcag2aaa'])
        .analyze();

    // Generate human-readable report
    let report = '='.repeat(80) + '\n';
    report += 'ACCESSIBILITY AUDIT REPORT (axe-core)\n';
    report += '='.repeat(80) + '\n\n';

    report += `URL: ${results.url}\n`;
    report += `Timestamp: ${results.timestamp}\n`;
    report += `Test Engine: ${results.testEngine.name} ${results.testEngine.version}\n\n`;

    // Summary
    report += '─'.repeat(80) + '\n';
    report += 'SUMMARY\n';
    report += '─'.repeat(80) + '\n';
    report += `✅ Passes: ${results.passes.length}\n`;
    report += `⚠️  Violations: ${results.violations.length}\n`;
    report += `ℹ️  Incomplete: ${results.incomplete.length}\n`;
    report += `⊘  Inapplicable: ${results.inapplicable.length}\n\n`;

    // Calculate score
    const totalTests = results.passes.length + results.violations.length;
    const score = totalTests > 0 ? Math.round((results.passes.length / totalTests) * 100) : 0;
    report += `📊 Accessibility Score: ${score}/100\n\n`;

    // Violations (if any)
    if (results.violations.length > 0) {
        report += '─'.repeat(80) + '\n';
        report += 'VIOLATIONS (MUST FIX)\n';
        report += '─'.repeat(80) + '\n';

        results.violations.forEach((violation, index) => {
            report += `\n${index + 1}. ${violation.id} (${violation.impact})\n`;
            report += `   Description: ${violation.description}\n`;
            report += `   Help: ${violation.help}\n`;
            report += `   WCAG: ${violation.tags.filter(t => t.startsWith('wcag')).join(', ')}\n`;
            report += `   Affected elements: ${violation.nodes.length}\n`;

            violation.nodes.slice(0, 3).forEach((node, nodeIndex) => {
                report += `   [${nodeIndex + 1}] ${node.html.substring(0, 80)}...\n`;
                report += `       ${node.failureSummary}\n`;
            });
        });
    } else {
        report += '✅ NO VIOLATIONS FOUND!\n';
    }

    // Passes (sample)
    if (results.passes.length > 0) {
        report += '\n' + '─'.repeat(80) + '\n';
        report += 'PASSES (Sample of first 10)\n';
        report += '─'.repeat(80) + '\n';

        results.passes.slice(0, 10).forEach((pass, index) => {
            report += `${index + 1}. ✅ ${pass.id} - ${pass.help}\n`;
        });

        if (results.passes.length > 10) {
            report += `\n... and ${results.passes.length - 10} more passed checks\n`;
        }
    }

    // Incomplete tests
    if (results.incomplete.length > 0) {
        report += '\n' + '─'.repeat(80) + '\n';
        report += 'INCOMPLETE TESTS (Manual Review Needed)\n';
        report += '─'.repeat(80) + '\n';

        results.incomplete.forEach((incomplete, index) => {
            report += `${index + 1}. ${incomplete.id} - ${incomplete.help}\n`;
            report += `   Description: ${incomplete.description}\n`;
        });
    }

    report += '\n' + '='.repeat(80) + '\n';
    report += 'END OF REPORT\n';
    report += '='.repeat(80) + '\n';

    // Save report
    const reportPath = path.join(__dirname, 'accessibility_audit.txt');
    fs.writeFileSync(reportPath, report);
    console.log(`\n✅ Report saved to: ${reportPath}`);

    // Save JSON for programmatic access
    const jsonPath = path.join(__dirname, 'accessibility_audit.json');
    fs.writeFileSync(jsonPath, JSON.stringify(results, null, 2));
    console.log(`✅ JSON data saved to: ${jsonPath}`);

    // Print summary
    console.log('\n' + '─'.repeat(80));
    console.log('AUDIT SUMMARY');
    console.log('─'.repeat(80));
    console.log(`✅ Passes: ${results.passes.length}`);
    console.log(`⚠️  Violations: ${results.violations.length}`);
    console.log(`📊 Score: ${score}/100`);

    if (results.violations.length === 0) {
        console.log('\n🎉 PERFECT! No accessibility violations found!');
    } else {
        console.log(`\n⚠️  Found ${results.violations.length} violations - see report for details`);
    }

    await browser.close();
}

runAccessibilityAudit().catch(error => {
    console.error('❌ Error running accessibility audit:', error);
    process.exit(1);
});
