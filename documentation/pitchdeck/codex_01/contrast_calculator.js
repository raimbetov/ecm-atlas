const fs = require('fs');
const path = require('path');

/**
 * Calculate relative luminance according to WCAG 2.1
 * https://www.w3.org/TR/WCAG21/#dfn-relative-luminance
 */
function getLuminance(r, g, b) {
    // Normalize RGB values to 0-1
    const [rs, gs, bs] = [r, g, b].map(val => {
        const normalized = val / 255;
        return normalized <= 0.03928
            ? normalized / 12.92
            : Math.pow((normalized + 0.055) / 1.055, 2.4);
    });

    // Calculate luminance using WCAG formula
    return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
}

/**
 * Calculate contrast ratio between two colors
 * https://www.w3.org/TR/WCAG21/#dfn-contrast-ratio
 */
function getContrastRatio(rgb1, rgb2) {
    const l1 = getLuminance(rgb1.r, rgb1.g, rgb1.b);
    const l2 = getLuminance(rgb2.r, rgb2.g, rgb2.b);

    const lighter = Math.max(l1, l2);
    const darker = Math.min(l1, l2);

    return (lighter + 0.05) / (darker + 0.05);
}

/**
 * Convert hex color to RGB
 */
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

/**
 * Check WCAG compliance levels
 */
function checkCompliance(ratio, fontSize, isBold) {
    const isLargeText = (fontSize >= 24 && !isBold) || (fontSize >= 18.66 && isBold);

    return {
        'WCAG AA (Normal)': ratio >= 4.5,
        'WCAG AA (Large)': ratio >= 3.0,
        'WCAG AAA (Normal)': ratio >= 7.0,
        'WCAG AAA (Large)': ratio >= 4.5,
        'Applicable Level': isLargeText ? 'Large Text' : 'Normal Text',
        'Passes AA': isLargeText ? ratio >= 3.0 : ratio >= 4.5,
        'Passes AAA': isLargeText ? ratio >= 4.5 : ratio >= 7.0
    };
}

// Define color palette
const colors = {
    'bg-primary': '#0a0e27',
    'bg-secondary': '#151b3d',
    'bg-tertiary': '#1e2749',
    'text-primary': '#ffffff',
    'text-secondary': '#e8eaf6',
    'accent-gold': '#ffd700',
    'accent-cyan': '#4ecdc4',
    'accent-blue': '#54a0ff',
    'accent-red': '#ff6b6b',
    'accent-green': '#6ab04c',
    'accent-purple': '#9b59b6'
};

// Define critical color pairs used in the presentation
const colorPairs = [
    // Text on backgrounds
    { name: 'Body text on primary bg', fg: 'text-primary', bg: 'bg-primary', fontSize: 29, isBold: false },
    { name: 'Secondary text on primary bg', fg: 'text-secondary', bg: 'bg-primary', fontSize: 29, isBold: false },
    { name: 'Gold accent on tertiary bg', fg: 'accent-gold', bg: 'bg-tertiary', fontSize: 47, isBold: true },
    { name: 'Cyan accent on tertiary bg', fg: 'accent-cyan', bg: 'bg-tertiary', fontSize: 47, isBold: true },

    // Mermaid diagram colors
    { name: 'Mermaid text on nodes', fg: 'text-primary', bg: 'bg-tertiary', fontSize: 28, isBold: true },
    { name: 'Mermaid edge labels (gold)', fg: 'accent-gold', bg: 'bg-primary', fontSize: 24, isBold: true },

    // UI elements
    { name: 'Button text on tertiary bg', fg: 'text-primary', bg: 'bg-tertiary', fontSize: 18, isBold: true },
    { name: 'Gold border/text on primary', fg: 'accent-gold', bg: 'bg-primary', fontSize: 18, isBold: true },

    // Slide titles
    { name: 'Gold gradient (min) on primary', fg: 'accent-gold', bg: 'bg-primary', fontSize: 76, isBold: true },
    { name: 'Cyan gradient (max) on primary', fg: 'accent-cyan', bg: 'bg-primary', fontSize: 76, isBold: true },

    // Stats and cards
    { name: 'Text on stat cards', fg: 'text-secondary', bg: 'bg-tertiary', fontSize: 18, isBold: true },

    // Additional important pairs
    { name: 'White text on secondary bg', fg: 'text-primary', bg: 'bg-secondary', fontSize: 29, isBold: false },
    { name: 'Red accent on tertiary bg', fg: 'accent-red', bg: 'bg-tertiary', fontSize: 28, isBold: true },
    { name: 'Green accent on tertiary bg', fg: 'accent-green', bg: 'bg-tertiary', fontSize: 28, isBold: true },
    { name: 'Blue accent on tertiary bg', fg: 'accent-blue', bg: 'bg-tertiary', fontSize: 28, isBold: true }
];

// Calculate all contrast ratios
console.log('🔬 Calculating contrast ratios...\n');

const results = {
    metadata: {
        title: 'Contrast Analysis - Improved Pitch Deck',
        timestamp: new Date().toISOString(),
        wcagVersion: '2.1',
        colorPalette: colors
    },
    summary: {
        totalPairs: colorPairs.length,
        passingAA: 0,
        passingAAA: 0,
        minRatio: Infinity,
        maxRatio: 0,
        averageRatio: 0
    },
    colorPairs: []
};

let totalRatio = 0;

colorPairs.forEach(pair => {
    const fgColor = hexToRgb(colors[pair.fg]);
    const bgColor = hexToRgb(colors[pair.bg]);

    const ratio = getContrastRatio(fgColor, bgColor);
    const compliance = checkCompliance(ratio, pair.fontSize, pair.isBold);

    totalRatio += ratio;

    if (compliance['Passes AA']) results.summary.passingAA++;
    if (compliance['Passes AAA']) results.summary.passingAAA++;

    results.summary.minRatio = Math.min(results.summary.minRatio, ratio);
    results.summary.maxRatio = Math.max(results.summary.maxRatio, ratio);

    const result = {
        name: pair.name,
        foreground: {
            name: pair.fg,
            hex: colors[pair.fg],
            rgb: fgColor
        },
        background: {
            name: pair.bg,
            hex: colors[pair.bg],
            rgb: bgColor
        },
        contrastRatio: Math.round(ratio * 100) / 100,
        fontSize: pair.fontSize,
        isBold: pair.isBold,
        compliance: compliance
    };

    results.colorPairs.push(result);

    // Print to console
    const aaStatus = compliance['Passes AA'] ? '✅' : '❌';
    const aaaStatus = compliance['Passes AAA'] ? '✅' : '❌';
    console.log(`${pair.name}`);
    console.log(`  Ratio: ${result.contrastRatio.toFixed(2)}:1`);
    console.log(`  ${aaStatus} WCAG AA  ${aaaStatus} WCAG AAA`);
    console.log(`  Font: ${pair.fontSize}px${pair.isBold ? ' bold' : ''} (${compliance['Applicable Level']})`);
    console.log('');
});

results.summary.averageRatio = Math.round((totalRatio / colorPairs.length) * 100) / 100;

console.log('─'.repeat(80));
console.log('SUMMARY');
console.log('─'.repeat(80));
console.log(`Total color pairs analyzed: ${results.summary.totalPairs}`);
console.log(`Minimum contrast ratio: ${results.summary.minRatio.toFixed(2)}:1`);
console.log(`Maximum contrast ratio: ${results.summary.maxRatio.toFixed(2)}:1`);
console.log(`Average contrast ratio: ${results.summary.averageRatio.toFixed(2)}:1`);
console.log(`Passing WCAG AA: ${results.summary.passingAA}/${results.summary.totalPairs}`);
console.log(`Passing WCAG AAA: ${results.summary.passingAAA}/${results.summary.totalPairs}`);
console.log('');

if (results.summary.passingAAA === results.summary.totalPairs) {
    console.log('🎉 PERFECT! All color pairs pass WCAG AAA (7:1 minimum)!');
} else if (results.summary.passingAA === results.summary.totalPairs) {
    console.log('✅ All color pairs pass WCAG AA (4.5:1 minimum)');
} else {
    console.log('⚠️  Some color pairs fail WCAG standards');
}

// Save to JSON
const outputPath = path.join(__dirname, 'contrast_analysis.json');
fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
console.log(`\n✅ Full analysis saved to: ${outputPath}`);
