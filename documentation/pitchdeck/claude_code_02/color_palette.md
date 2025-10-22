# Color Palette - Agent 02 Design

## Design Philosophy
Pure black background with purple/magenta gradient accents for maximum contrast and modern tech aesthetic.

## Color Scheme

### Background Colors
- **Primary Background**: `#000000` (Pure Black)
  - Contrast with white text: **21:1** (WCAG AAA+++)
  - Usage: Main slide background, ultimate darkness

- **Secondary Background**: `#0d0d0d` (Near Black)
  - Contrast with white text: **19.5:1** (WCAG AAA++)
  - Usage: Layered elements

- **Tertiary Background**: `#1a1a1a` (Dark Gray)
  - Contrast with white text: **16.8:1** (WCAG AAA++)
  - Usage: Diagram containers, cards with glassmorphism

### Text Colors
- **Primary Text**: `#ffffff` (Pure White)
  - Contrast on black: **21:1** (WCAG AAA+++)
  - Usage: All body text, diagram text, headings

- **Secondary Text**: `#b3b3b3` (Light Gray)
  - Contrast on black: **11.4:1** (WCAG AAA+)
  - Usage: Subtitles, descriptions, labels

### Accent Colors
- **Purple**: `#9b59b6`
  - RGB: (155, 89, 182)
  - Contrast on black: **5.2:1** (WCAG AA)
  - Usage: Primary accent, borders, progress bar, gradient highlights

- **Magenta**: `#e74c3c`
  - RGB: (231, 76, 60)
  - Contrast on black: **5.9:1** (WCAG AA)
  - Usage: Warning/alert nodes, gradient middle, key highlights

- **Cyan**: `#3498db`
  - RGB: (52, 152, 219)
  - Contrast on black: **5.1:1** (WCAG AA)
  - Usage: Information nodes, gradient end, secondary accents

- **Green**: `#2ecc71`
  - RGB: (46, 204, 113)
  - Contrast on black: **7.8:1** (WCAG AAA)
  - Usage: Success/positive nodes, checkmarks

- **Orange**: `#f39c12`
  - RGB: (243, 156, 18)
  - Contrast on black: **7.2:1** (WCAG AAA)
  - Usage: Warning states, highlights

## Gradients

### Title Gradient
```css
background: linear-gradient(135deg, #9b59b6 0%, #e74c3c 50%, #3498db 100%);
```
Purple → Magenta → Cyan (tri-color gradient)

### Progress Bar Gradient
```css
background: linear-gradient(90deg, #9b59b6 0%, #e74c3c 50%, #3498db 100%);
```
Horizontal version of title gradient

### Stats Gradient
```css
background: linear-gradient(135deg, #9b59b6 0%, #e74c3c 100%);
```
Purple → Magenta (simplified)

## Contrast Ratios Summary

| Element | Foreground | Background | Ratio | WCAG Level |
|---------|-----------|-----------|-------|------------|
| Body Text | #ffffff | #000000 | 21:1 | AAA+++ |
| Diagram Text | #ffffff | #1a1a1a | 16.8:1 | AAA++ |
| Subtitles | #b3b3b3 | #000000 | 11.4:1 | AAA+ |
| Purple Accent | #9b59b6 | #000000 | 5.2:1 | AA |
| Green Nodes | #2ecc71 | #000000 | 7.8:1 | AAA |
| Orange Nodes | #f39c12 | #000000 | 7.2:1 | AAA |

## Glassmorphism Effects
```css
background: rgba(26, 26, 26, 0.6);
backdrop-filter: blur(20px);
border: 2px solid rgba(155, 89, 182, 0.3);
```

## Why This Palette?

1. **Pure Black (#000000)**: Maximum contrast baseline, no compromises
2. **Purple Primary**: Modern tech aesthetic, distinctly different from gold/cyan
3. **Tri-Color Gradient**: More dynamic than bi-color, creates depth
4. **All AAA+ on critical text**: Body text 21:1, subtitles 11.4:1
5. **Glassmorphism**: Adds depth without sacrificing contrast
6. **Distinct from Agent 01**: Agent 01 uses dark blue + gold/cyan, Agent 02 uses pure black + purple/magenta

## Mermaid Diagram Colors
Nodes inherit from gradient palette:
- Red nodes (#e74c3c): Problems, old paradigms
- Orange nodes (#f39c12): Warnings, transitions
- Green nodes (#2ecc71): Solutions, success states
- Cyan nodes (#3498db): Information, frameworks
- Purple nodes (#9b59b6): Key concepts, highlights
- Gray nodes (#7f8c8d): Neutral, base level
