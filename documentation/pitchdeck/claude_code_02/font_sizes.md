# Font Sizes - Agent 02 Design

## Typography Strategy
**Maximum readability through aggressive font sizing and heavy weights.**

## Font Family
**Primary**: Inter (Google Fonts)
- Modern, highly readable tech font
- Optimized for screens
- Excellent weight range (400-900)
- Distinct from system fonts

**Fallback**: `-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`

## Font Sizes

### Title Slide
- **Main Title**: `5.5rem` (88px)
  - Weight: 900 (Black)
  - Line height: 1.1
  - Letter spacing: -0.02em
  - Gradient colored

- **Subtitle**: `2rem` (32px)
  - Weight: 400 (Regular)
  - Line height: 1.5
  - Color: #b3b3b3

- **Author Info**: `1.5rem` (24px)
  - Weight: 600 (Semi-bold)
  - Color: #3498db

### Content Slides
- **Slide Titles**: `4.5rem` (72px)
  - Weight: 900 (Black)
  - Line height: 1.1
  - Letter spacing: -0.02em
  - Gradient colored

- **Slide Subtitles**: `1.6rem` (25.6px)
  - Weight: 400 (Regular)
  - Line height: 1.5
  - Color: #b3b3b3

- **Diagram Titles**: `2.8rem` (44.8px)
  - Weight: 800 (Extra Bold)
  - Line height: 1.3
  - Letter spacing: -0.01em
  - Color: #9b59b6 (purple)

### Mermaid Diagrams (CRITICAL)
- **Base Font Size**: `32px` (SVG fontSize)
  - **Up from original 22px = +45% increase**
  - Weight: 800 (Extra Bold)
  - Color: #ffffff (pure white)
  - Font family: Inter

- **Node Spacing**: `120px` (up from 80px)
- **Rank Spacing**: `120px` (up from 80px)
- **Node Padding**: `40px` (up from 25px)
- **Border Width**: `4px` (up from 3px)

### UI Elements
- **Navigation Buttons**: `1.2rem` (19.2px)
  - Weight: 800 (Extra Bold)
  - Letter spacing: normal

- **Slide Counter**: `1.3rem` (20.8px)
  - Weight: 900 (Black)
  - Color: #9b59b6

- **Keyboard Hint**: `1rem` (16px)
  - Weight: 600 (Semi-bold)
  - Opacity: 0.5

### Stats Cards
- **Stat Number**: `3.8rem` (60.8px)
  - Weight: 900 (Black)
  - Gradient colored

- **Stat Label**: `1.2rem` (19.2px)
  - Weight: 600 (Semi-bold)
  - Color: #b3b3b3

### Key Insights
- **Insight Text**: `1.4rem` (22.4px)
  - Weight: 600 (Semi-bold)
  - Line height: 1.7

- **Insight Strong**: Weight 900 (Black)

## Comparison to Original

| Element | Original | Agent 02 | Change |
|---------|----------|----------|--------|
| Title | 3.5rem | 4.5rem | +29% |
| Title Slide Title | 4.5rem | 5.5rem | +22% |
| Diagram Title | 2.2rem | 2.8rem | +27% |
| **Mermaid Text** | **22px** | **32px** | **+45%** |
| Subtitle | 1.4rem | 1.6rem | +14% |
| Key Insight | 1.2rem | 1.4rem | +17% |
| Nav Buttons | 1.1rem | 1.2rem | +9% |
| Slide Counter | 1.1rem | 1.3rem | +18% |

## Font Weights Used

| Weight | Value | Usage |
|--------|-------|-------|
| Regular | 400 | Subtitles, descriptions |
| Semi-bold | 600 | Body text, labels, insights |
| Bold | 700 | (Not used - skipped for heavier) |
| Extra Bold | 800 | Diagram titles, nav buttons, Mermaid text |
| Black | 900 | Slide titles, stats, counter |

## Why These Sizes?

1. **Mermaid 32px**: Original 22px too small from 2m distance, 32px = +45% larger
2. **Titles 4.5-5.5rem**: Massive impact, immediate hierarchy
3. **Weight 800-900**: Maximum boldness for readability
4. **Inter Font**: Better rendering than system fonts at large sizes
5. **Generous Spacing**: Line height 1.5-1.7 for easy scanning
6. **All sizes 1.2rem+**: Nothing below 19.2px ensures readability

## Responsive Breakpoint (<1024px)
- Title: `3rem` (48px)
- Subtitle: `1.3rem` (20.8px)
- Diagram Title: `2rem` (32px)
- Maintains weight hierarchy
