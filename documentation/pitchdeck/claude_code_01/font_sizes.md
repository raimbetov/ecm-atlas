# Font Sizes Documentation

## Typography Scale

### Slide Titles
- **Title Slide**: `4.8rem` (76.8px) - Main presentation title
- **Regular Slides**: `3.8rem` (60.8px) - Section titles
- **Weight**: 800 (Extra bold)
- **Style**: Gradient fill (gold to cyan)

### Subtitles
- **Title Slide**: `2.0rem` (32px) - Main subtitle
- **Regular Slides**: `1.6rem` (25.6px) - Section subtitles
- **Weight**: 400 (Regular)
- **Color**: White (#ffffff)

### Diagram Titles
- **Size**: `2.8rem` (44.8px)
- **Weight**: 700 (Bold)
- **Color**: Gold (#ffd700)
- **Purpose**: Minto-style explanatory titles above diagrams

### Mermaid Diagram Text
- **Base font size**: `28px` (CRITICAL for readability)
- **Text weight**: 700 (Bold)
- **Text color**: White (#ffffff)
- **Edge labels**: `26px` with bold weight
- **Node padding**: 40px (generous spacing)
- **Node spacing**: 120px (prevents cramping)

### Body Text
- **Key insights**: `1.4rem` (22.4px)
- **Insight weight**: 400 regular, 700 bold for emphasis
- **Line height**: 1.8 (generous spacing)

### Navigation & UI
- **Navigation buttons**: `1.3rem` (20.8px), weight 700
- **Slide counter**: `1.3rem` (20.8px), weight 700
- **Keyboard hint**: `1.1rem` (17.6px), weight 600

### Stats Cards
- **Stat numbers**: `3.5rem` (56px)
- **Stat labels**: `1.3rem` (20.8px), weight 600
- **Author info**: `1.5rem` (24px), weight 600

## Key Improvements from Original

### Original Problems:
1. **Mermaid diagrams**: 22px → **Too small, illegible from distance**
2. **Diagram titles**: 2.2rem (35.2px) → **Not prominent enough**
3. **Node padding**: 25px → **Text felt cramped**
4. **Node spacing**: 80px → **Diagrams felt cluttered**

### New Implementation:
1. **Mermaid diagrams**: **28px** (+27% increase)
2. **Diagram titles**: **2.8rem** (+27% increase)
3. **Node padding**: **40px** (+60% increase)
4. **Node spacing**: **120px** (+50% increase)

## Readability Testing

### Distance Testing:
- **2 meters (6.5 feet)**: All text clearly readable ✅
- **3 meters (10 feet)**: Titles and diagram text still legible ✅
- **Conference room back row**: Main points visible ✅

### Font Weight Strategy:
- **800 (Extra bold)**: Only for main slide titles
- **700 (Bold)**: Diagram text, section titles, navigation
- **600 (Semi-bold)**: Labels, UI elements
- **400 (Regular)**: Body text, subtitles

## Responsive Breakpoints

### Desktop (>1024px):
- All sizes as documented above

### Tablet/Small Desktop (<1024px):
- **Slide titles**: 2.8rem (44.8px)
- **Subtitles**: 1.3rem (20.8px)
- **Diagram titles**: 2rem (32px)
- **Padding reduced**: 2rem diagram containers

## Technical Implementation

### CSS Variables Used:
```css
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
             'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
```

### Mermaid Configuration:
```javascript
mermaid.initialize({
    themeVariables: {
        fontSize: '28px',  // Increased from 22px
        fontFamily: 'system font stack'
    },
    flowchart: {
        nodeSpacing: 120,  // Increased from 80
        rankSpacing: 120,  // Increased from 80
        padding: 40        // Increased from 25
    }
});
```

### Direct SVG Overrides:
```css
.mermaid svg {
    font-size: 28px !important;
    font-weight: 600 !important;
}

.mermaid text {
    fill: #ffffff !important;
    font-weight: 700 !important;
    font-size: 28px !important;
}

.mermaid .edgeLabel {
    font-size: 26px !important;
    font-weight: 700 !important;
}
```

## Result

✅ **All text readable from 2m+ distance**
✅ **Mermaid diagrams 27% larger than original**
✅ **Professional typography hierarchy**
✅ **Conference-room presentation ready**
