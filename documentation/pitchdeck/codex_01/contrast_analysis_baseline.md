# Baseline Contrast Analysis - Current HTML

## Colors Used in Current HTML

### CSS Variables (from lines 15-27):
- `--bg-primary: #0a0e27` (very dark blue, almost black)
- `--bg-secondary: #151b3d` (dark blue)
- `--bg-tertiary: #1e2749` (medium dark blue)
- `--text-primary: #e8eaf6` (very light purple-gray)
- `--text-secondary: #b0b8d4` (medium light purple-gray)
- `--accent-gold: #ffd700` (bright gold)
- `--accent-cyan: #4ecdc4` (cyan)
- `--accent-blue: #54a0ff` (light blue)
- `--accent-red: #ff6b6b` (coral red)
- `--accent-green: #6ab04c` (green)
- `--accent-purple: #9b59b6` (purple)

## Critical Issues Identified

### Issue #1: Mermaid Diagram Text (Line 145-147)
```css
.mermaid text {
    fill: #f5f7fa !important;  /* Very light gray */
    font-weight: 500 !important;
}
```
- **Problem:** `#f5f7fa` (very light) on Mermaid node backgrounds
- Mermaid uses `primaryColor: '#1e2749'` for nodes (line 939)
- **Need to calculate:** `#f5f7fa` vs `#1e2749` contrast ratio

### Issue #2: Font Size in Mermaid (Line 134)
```css
.mermaid svg {
    font-size: 22px !important;
}
```
- **Current:** 22px
- **Required:** Minimum 24px (from task requirements)
- **Gap:** 2px too small

### Issue #3: Node Spacing (Lines 958-960)
```css
flowchart: {
    nodeSpacing: 80,
    rankSpacing: 80,
    padding: 25,
}
```
- **Current:** 80px spacing, 25px padding
- **Required:** 100px+ spacing, 30px+ padding
- **Gap:** Too cramped

## Manual Contrast Calculations (WCAG 2.1 Formula)

### Formula:
Contrast Ratio = (L1 + 0.05) / (L2 + 0.05)
where L = relative luminance

### Relative Luminance Calculation:
For RGB values divided by 255:
- If RsRGB ≤ 0.03928: R = RsRGB/12.92
- Else: R = ((RsRGB+0.055)/1.055)^2.4
- L = 0.2126 * R + 0.7152 * G + 0.0722 * B

## Key Color Pairs to Test:

### 1. Mermaid Text on Nodes
- **Foreground:** `#f5f7fa` (RGB: 245, 247, 250)
- **Background:** `#1e2749` (RGB: 30, 39, 73)

RGB normalized:
- FG: (245/255, 247/255, 250/255) = (0.961, 0.969, 0.980)
- BG: (30/255, 39/255, 73/255) = (0.118, 0.153, 0.286)

Luminance (using sRGB formula):
- FG: All values > 0.03928, so use power formula
  - R: ((0.961+0.055)/1.055)^2.4 = 0.917
  - G: ((0.969+0.055)/1.055)^2.4 = 0.932
  - B: ((0.980+0.055)/1.055)^2.4 = 0.953
  - L1 = 0.2126*0.917 + 0.7152*0.932 + 0.0722*0.953 = 0.929

- BG:
  - R: ((0.118+0.055)/1.055)^2.4 = 0.015
  - G: ((0.153+0.055)/1.055)^2.4 = 0.024
  - B: ((0.286+0.055)/1.055)^2.4 = 0.078
  - L2 = 0.2126*0.015 + 0.7152*0.024 + 0.0722*0.078 = 0.026

**Contrast Ratio:** (0.929 + 0.05) / (0.026 + 0.05) = 0.979 / 0.076 = **12.88:1**

✅ **PASSES WCAG AAA (7:1 required)**

### 2. Body Text Primary
- **Foreground:** `#e8eaf6` (RGB: 232, 234, 246)
- **Background:** `#0a0e27` (RGB: 10, 14, 39)

Luminance:
- FG: L1 ≈ 0.84
- BG: L2 ≈ 0.004

**Contrast Ratio:** (0.84 + 0.05) / (0.004 + 0.05) = **16.48:1**

✅ **PASSES WCAG AAA**

### 3. Gold on Dark Background
- **Foreground:** `#ffd700` (gold, RGB: 255, 215, 0)
- **Background:** `#1e2749` (RGB: 30, 39, 73)

Luminance:
- FG (gold): L1 ≈ 0.69
- BG: L2 ≈ 0.026

**Contrast Ratio:** (0.69 + 0.05) / (0.026 + 0.05) = **9.74:1**

✅ **PASSES WCAG AAA**

## Conclusion

**Surprising Finding:** The current HTML actually has GOOD contrast ratios mathematically!

The real issues are:
1. ❌ **Font size too small:** 22px vs 24px minimum
2. ❌ **Node spacing too tight:** 80px vs 100px minimum
3. ❌ **Padding too small:** 25px vs 30px minimum
4. ⚠️ **Potential Mermaid rendering issues:** White-on-white may occur in specific node types

## Next Steps

1. Verify actual rendered colors with screenshot analysis
2. Check if Mermaid overrides CSS with its own theme
3. Increase font sizes using golden ratio scale
4. Optimize Mermaid themeVariables for maximum readability
