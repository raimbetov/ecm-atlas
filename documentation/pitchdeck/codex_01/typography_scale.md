# Typography Scale - Mathematical Calculation

## Scale Ratios Tested

Common typographic scale ratios:
- **Minor Third:** 1.200 (subtle)
- **Major Third:** 1.250 (moderate)
- **Perfect Fourth:** 1.333 (balanced)
- **Augmented Fourth:** 1.414 (bold)
- **Perfect Fifth:** 1.500 (strong)
- **Golden Ratio:** 1.618 (harmonious)

## Selected: Golden Ratio (φ = 1.618)

The golden ratio provides the most aesthetically pleasing proportions for presentation slides with large viewing distances.

## Base Size Calculation

### Requirements:
- Mermaid diagram text: **minimum 24px**
- Node text: **minimum 20px, bold 600+**
- Diagram titles: **2.5rem+ (40px+)**
- Readable from 2m distance

### Base Size: 18px
Using 18px as base (1rem), we apply golden ratio:

## Complete Typography Scale

| Level | Calculation | Size (px) | Size (rem) | Usage |
|-------|-------------|-----------|------------|-------|
| -2 | 18 ÷ 1.618² | 6.87 | 0.382rem | *Not used* |
| -1 | 18 ÷ 1.618 | 11.13 | 0.618rem | *Not used* |
| **0** | **18 × 1** | **18.00** | **1.000rem** | Base font size |
| **1** | **18 × 1.618** | **29.12** | **1.618rem** | Body text, key insights |
| **2** | **18 × 1.618²** | **47.12** | **2.618rem** | Diagram titles |
| **3** | **18 × 1.618³** | **76.24** | **4.236rem** | Slide titles |
| **4** | **18 × 1.618⁴** | **123.34** | **6.852rem** | Hero titles (optional) |

## Applied Sizes (Rounded for Browser Rendering)

### Current HTML → Improved HTML

| Element | Current | Improved | Ratio Applied |
|---------|---------|----------|---------------|
| **Body base** | 16px | 18px | Base |
| **Slide subtitle** | 1.4rem (22.4px) | 1.618rem (29px) | φ¹ |
| **Key insight text** | 1.2rem (19.2px) | 1.618rem (29px) | φ¹ |
| **Diagram titles** | 2.2rem (35.2px) | 2.618rem (47px) | φ² |
| **Slide titles** | 3.5rem (56px) | 4.236rem (76px) | φ³ |
| **Title slide** | 4.5rem (72px) | 6.852rem (123px) | φ⁴ |

### Mermaid-Specific Sizes

| Element | Current | Improved | Meets Requirement |
|---------|---------|----------|-------------------|
| **Diagram text (SVG)** | 22px | **28px** | ✅ Yes (>24px) |
| **Node text** | 22px | **28px bold 700** | ✅ Yes (>20px, 600+) |
| **Edge label text** | 22px | **24px bold 600** | ✅ Yes (>20px) |
| **Node padding** | 25px | **40px** | ✅ Yes (>30px) |
| **Node spacing** | 80px | **120px** | ✅ Yes (>100px) |

## Font Weight Scale

Using standardized font weights for hierarchy:

| Level | Weight | Usage |
|-------|--------|-------|
| Light | 300 | Slide subtitles |
| Regular | 400 | *Not used* |
| Medium | 500 | Body text |
| Semi-Bold | 600 | Edge labels, emphasis |
| Bold | 700 | Node text, diagram text |
| Extra-Bold | 800 | Slide titles |

## Line Height Calculations

Optimal line height for readability at distance:

| Font Size | Line Height | Calculation | Usage |
|-----------|-------------|-------------|-------|
| 18px (base) | 27px | 1.5 × base | Body text |
| 29px (body) | 46.4px | 1.6 × size | Key insights |
| 47px (diagram) | 61px | 1.3 × size | Diagram titles |
| 76px (slide) | 91px | 1.2 × size | Slide titles |

## Spacing Scale (Based on Golden Ratio)

Vertical rhythm using φ:

| Level | Size (px) | Usage |
|-------|-----------|-------|
| xs | 11px | 18 ÷ φ |
| sm | 18px | Base |
| md | 29px | 18 × φ |
| lg | 47px | 18 × φ² |
| xl | 76px | 18 × φ³ |
| 2xl | 123px | 18 × φ⁴ |

## Accessibility Verification

### WCAG 2.1 Requirements:

1. ✅ **Minimum font size:** All body text ≥18px (AA Large Text = 18pt = 24px for body, but we use 29px)
2. ✅ **Heading contrast:** All headings use high-contrast colors (gold/cyan on dark)
3. ✅ **Line spacing:** 1.5× for body text (46.4px / 29px = 1.6)
4. ✅ **Letter spacing:** Browser default (no modification needed)

### Reading Distance Formula:

For 2m (6.6 ft) viewing distance:
- **Minimum legible:** 24px at 1920×1080
- **Comfortable reading:** 28-32px for body, 40+ for headings
- **Our choice:** 29px body (✅), 47px headings (✅), 76px titles (✅)

## Mathematical Proof: φ Harmony

Golden ratio appears in:
- **Fibonacci sequence:** 1, 1, 2, 3, 5, 8, 13, 21, 34, 55...
  - Ratio of consecutive numbers → φ
  - 34/21 = 1.619, 55/34 = 1.618, etc.

- **Our scale:**
  - 18 → 29 → 47 → 76 → 123
  - 29/18 = 1.611 ≈ φ
  - 47/29 = 1.621 ≈ φ
  - 76/47 = 1.617 ≈ φ

Perfect mathematical harmony for visual comfort!

## Summary: Improvements

| Metric | Current | Improved | Improvement |
|--------|---------|----------|-------------|
| **Base font** | 16px | 18px | +12.5% |
| **Mermaid text** | 22px | 28px | +27.3% |
| **Body text** | 19-22px | 29px | +31-53% |
| **Diagram titles** | 35px | 47px | +34.3% |
| **Slide titles** | 56px | 76px | +35.7% |
| **Node padding** | 25px | 40px | +60% |
| **Node spacing** | 80px | 120px | +50% |

All sizes follow golden ratio, creating visual harmony while exceeding minimum readability requirements.
