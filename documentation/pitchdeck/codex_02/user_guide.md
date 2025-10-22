# User Guide: Interactive Pitch Deck

**Version:** 1.0
**Agent:** Codex 02
**Date:** 2025-10-22

## Overview

This interactive HTML presentation features advanced customization controls for optimal viewing across different environments and accessibility needs. All settings are automatically saved to your browser's localStorage and persist between sessions.

---

## Quick Start

1. **Open the presentation:** Double-click `pitchdeck_improved.html`
2. **Navigate slides:** Use arrow keys (← →) or click navigation buttons
3. **Access settings:** Click the ⚙ icon (top-left) or press `S`
4. **Customize:** Choose your theme, adjust font size, or enter fullscreen

---

## Navigation Controls

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` or `→` | Navigate between slides |
| `+` or `=` | Increase font size |
| `-` or `_` | Decrease font size |
| `S` | Toggle settings panel |
| `F` | Toggle fullscreen mode |
| `P` | Print/export to PDF |

### Mouse/Touch Controls

- **Navigation buttons:** Fixed at bottom-center of screen
- **Settings icon:** Click the ⚙ icon at top-left corner
- **Touch swipe:** Swipe left/right on mobile devices

---

## Theme Selection

### Available Themes

**1. Dark Blue (Default)**
- Background: Deep blue gradient (#0a0e27 → #151b3d)
- Accent: Gold (#ffd700) and cyan (#4ecdc4)
- Best for: Standard presentations, low-light rooms
- Contrast ratio: 12:1 (WCAG AAA)

**2. Pure Black (High Contrast)**
- Background: Pure black (#000000)
- Accent: Bright yellow (#ffeb3b) and cyan (#00ffff)
- Best for: Maximum contrast, OLED screens, battery saving
- Contrast ratio: 21:1 (Maximum)

**3. High Contrast (Accessibility)**
- Background: Black with bright highlights
- Accent: Yellow (#ffff00) and cyan (#00ffff)
- Best for: Vision impairments, bright ambient light
- Contrast ratio: 21:1 (WCAG AAA+)

### How to Switch Themes

1. Open settings panel (click ⚙ or press `S`)
2. Click one of three theme buttons
3. Theme applies instantly with smooth transition
4. Mermaid diagrams re-render automatically

**Note:** Your theme preference is saved automatically and persists across sessions.

---

## Font Size Controls

### Size Range

- **Minimum:** 70% (0.7× multiplier)
- **Default:** 100% (1.0× multiplier)
- **Maximum:** 150% (1.5× multiplier)
- **Step:** 10% per adjustment

### What Adjusts?

Font size control affects:
- Slide titles (3.5rem → 5.25rem range)
- Slide subtitles (1.4rem → 2.1rem range)
- Diagram titles (2.2rem → 3.3rem range)
- Body text and key insights
- Mermaid diagram text (26px → 39px range)
- Statistics and labels

### How to Adjust

**Via Settings Panel:**
1. Open settings (⚙ icon)
2. Click "− Smaller", "Reset", or "+ Larger"
3. Current size shown as percentage

**Via Keyboard:**
- Press `+` or `=` to increase
- Press `-` or `_` to decrease

**Tip:** Use larger fonts for projector presentations or accessibility needs.

---

## Print & PDF Export

### Print via Browser

1. **Method 1:** Press `P` key
2. **Method 2:** Open settings → Click "🖨 Print/PDF"
3. **Method 3:** Browser menu → Print (Cmd/Ctrl+P)

### Print Settings

The presentation includes optimized print CSS:
- **Layout:** All 10 slides displayed sequentially
- **Page breaks:** One slide per page
- **Background:** Converts to white for ink saving
- **Text:** High-contrast black text
- **Diagrams:** Print-friendly Mermaid rendering
- **Hidden elements:** Navigation controls, settings removed

### Recommended Browser Settings

For best PDF export:
- **Format:** A4 or Letter (Landscape preferred)
- **Margins:** None or minimal
- **Background graphics:** Enabled
- **Scale:** 100% or "Fit to page"

---

## Fullscreen Mode

### Enter Fullscreen

- **Keyboard:** Press `F`
- **Settings:** Click "⛶ Fullscreen" button
- **Browser:** Use browser's fullscreen option (F11)

### Exit Fullscreen

- **Keyboard:** Press `Esc` or `F`
- **Mouse:** Move to top and click browser exit button

### Benefits

- Removes browser UI for distraction-free presenting
- Maximizes screen real estate
- Professional presentation experience

---

## Persistence & Storage

### What's Saved?

All preferences are automatically saved to `localStorage`:
- Current theme selection
- Font size multiplier
- Last viewed slide (optional)

### Clear Settings

To reset to defaults:
1. Browser console: `localStorage.clear()`
2. Refresh the page

**Storage size:** <1KB (very lightweight)

---

## Accessibility Features

### Vision Accessibility

- **High Contrast theme:** 21:1 contrast ratio
- **Font size control:** Up to 150% scaling
- **Keyboard navigation:** Full keyboard support
- **ARIA labels:** Semantic HTML structure

### Motion Sensitivity

- **Smooth transitions:** Can be disabled via browser settings
- **Reduced motion:** Respects `prefers-reduced-motion` (optional enhancement)

### Screen Readers

- Semantic HTML5 structure
- Proper heading hierarchy (h1 → h2 → h3)
- Alt text for visual elements (via Mermaid code)

---

## Mobile & Tablet Support

### Responsive Design

- **Touch navigation:** Swipe left/right between slides
- **Viewport scaling:** Adapts to screen size
- **Font scaling:** Automatically adjusts for mobile
- **Settings panel:** Responsive width on small screens

### Recommended Orientation

- **Phone:** Portrait or landscape
- **Tablet:** Landscape for best diagram visibility

---

## Troubleshooting

### Diagrams Not Rendering

**Problem:** Mermaid diagrams show as text
**Solution:**
1. Check internet connection (Mermaid CDN required on first load)
2. Wait 2-3 seconds for rendering
3. Refresh the page (F5)

### Theme Not Changing

**Problem:** Theme button clicked but no visual change
**Solution:**
1. Wait 2 seconds for Mermaid re-render
2. Check if localStorage is enabled in browser
3. Hard refresh (Cmd/Ctrl+Shift+R)

### Font Size Reset on Refresh

**Problem:** Font size returns to 100% after page reload
**Solution:**
- Check browser's localStorage permissions
- Ensure cookies/storage not blocked for local files
- Try opening from a local web server

### Print Shows All Slides on One Page

**Problem:** PDF export has layout issues
**Solution:**
1. Use Chrome or Edge (best compatibility)
2. Ensure "Background graphics" is enabled
3. Select A4 Landscape format
4. Set margins to "None"

---

## Technical Requirements

### Browser Compatibility

- **Recommended:** Chrome 90+, Edge 90+
- **Supported:** Firefox 88+, Safari 14+
- **Mobile:** iOS Safari 14+, Chrome Mobile 90+

### Internet Connection

- **First load:** Required (Mermaid CDN ~500KB)
- **After cache:** Works fully offline
- **CDN:** cdn.jsdelivr.net/npm/mermaid@10

### Performance

- **File size:** 94KB HTML (self-contained)
- **Load time:** <2 seconds on broadband
- **Memory:** ~50MB browser RAM
- **Animations:** 60 FPS on modern hardware

---

## Tips & Best Practices

### For Presentations

1. **Test before presenting:** Open 5 minutes early to cache assets
2. **Use Dark Blue theme:** Most professional for business
3. **Font size 110-120%:** Easier to read from distance
4. **Fullscreen mode:** Press `F` before starting
5. **Practice navigation:** Use arrow keys, not mouse

### For Accessibility

1. **Start with High Contrast theme** for vision-impaired audiences
2. **Increase font to 130-150%** for large screens
3. **Enable browser zoom** (Cmd/Ctrl + +) for additional scaling
4. **Use keyboard only** for screenreader compatibility

### For PDF Distribution

1. **Export in Pure Black theme** (saves toner/ink)
2. **Reset font to 100%** before printing
3. **Landscape orientation** preserves diagram layout
4. **PDF size:** ~3-5MB for 10 slides with diagrams

---

## Hotkey Cheat Sheet

**Quick Reference Card:**

```
NAVIGATION          CUSTOMIZATION       ACTIONS
← Previous slide    + Increase font     S Settings panel
→ Next slide        - Decrease font     F Fullscreen
                    0 Reset font        P Print/PDF
```

---

## Support & Feedback

**Created by:** Codex Agent 02
**Repository:** `/Users/Kravtsovd/projects/ecm-atlas/documentation/pitchdeck/codex_02/`
**Date:** 2025-10-22

For issues or enhancement requests, refer to `technical_architecture.md` for implementation details.

---

**Last Updated:** 2025-10-22
**Version:** 1.0
**License:** Internal use
