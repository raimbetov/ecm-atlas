## Task: Create Beautiful & Readable HTML Pitch Deck for Hackathon

**Context:**
- Source content: `/Users/Kravtsovd/projects/ecm-atlas/documentation/pitchdeck/HACKATHON_PITCH_MULTI_AGENT_FRAMEWORKS.md`
- Current HTML: `/Users/Kravtsovd/projects/ecm-atlas/documentation/pitchdeck/hackathon_presentation.html`
- Problem: Text is too small, white-on-white unreadable, poor contrast
- Goal: Ultra-professional, highly readable presentation for 3-minute pitch

**Success Criteria:**

1. **✅ TEXT READABILITY (CRITICAL)**
   - Minimum font size in Mermaid diagrams: 24px (current 22px is too small)
   - Node text minimum: 20px, bold weight 600+
   - Diagram titles: 2.5rem+, high contrast
   - NO white-on-white or low-contrast text combinations
   - Test: Screenshot must show ALL text clearly readable from 2m distance

2. **✅ COLOR CONTRAST (CRITICAL)**
   - Background-to-text contrast ratio ≥7:1 (WCAG AAA)
   - Node backgrounds: Dark (#1e2749 or darker) with bright text (#ffffff or #ffd700)
   - NO light backgrounds with light text
   - Edge labels: High contrast background (#0a0e27) with bright text
   - Validation: Run contrast checker on screenshot

3. **✅ MERMAID DIAGRAM QUALITY**
   - All 18 Mermaid diagrams render correctly
   - Node spacing: 100px+ (current 80px too cramped)
   - Larger nodes: padding 30px+ (current 25px too small)
   - Clear arrow labels with contrasting backgrounds
   - Each diagram fits viewport without scrolling

4. **✅ PROFESSIONAL DESIGN**
   - Dark tech theme (dark blue/black background)
   - Gold (#ffd700) and cyan (#4ecdc4) accents only for highlights
   - Consistent typography hierarchy
   - Smooth slide transitions (0.6s cubic-bezier)
   - Progress bar and slide counter working

5. **✅ SCREENSHOT VALIDATION**
   - Create headless browser screenshot of each slide (all 10 slides)
   - Save screenshots to agent workspace: `screenshots/slide_{01-10}.png`
   - Screenshot resolution: 1920x1080 (standard presentation)
   - Verify: Text readable, no rendering issues, proper colors
   - Include comparison: before (current) vs after (improved)

6. **✅ SELF-CONTAINED & PORTABLE**
   - Single HTML file, no external dependencies except Mermaid CDN
   - Works offline after initial Mermaid load
   - Cross-browser compatible (Chrome, Firefox, Safari)
   - Mobile-responsive (bonus: swipe navigation)

**Constraints:**
- Preserve all content from markdown source (10 slides, all diagrams)
- Keep Minto-style explanatory diagram titles
- Maintain navigation: arrow keys, buttons, progress bar
- Use only Mermaid for diagrams (no external image dependencies)
- File size <500KB (excluding Mermaid CDN)

**Deliverables (in your agent workspace):**

1. `pitchdeck_improved.html` - The beautiful, readable presentation
2. `screenshots/` folder with:
   - `slide_01.png` to `slide_10.png` (all slides)
   - `comparison.png` (before vs after side-by-side)
3. `90_results_{agent}.md` - Self-evaluation against criteria
4. `color_palette.md` - Document your chosen color scheme with contrast ratios
5. `font_sizes.md` - Document all font sizes used (titles, body, diagrams)

**Testing Requirements:**

Before submitting, test:
1. Open HTML in Chrome headless, screenshot each slide
2. Check contrast ratios: https://webaim.org/resources/contrastchecker/
3. Verify Mermaid rendering: All nodes visible, text readable
4. Responsive test: Resize window to 1024px, 768px
5. Navigation test: Arrow keys, buttons, keyboard shortcuts work

**Reference Materials:**
- Current HTML: `hackathon_presentation.html` (analyze what's wrong)
- Content source: `HACKATHON_PITCH_MULTI_AGENT_FRAMEWORKS.md`
- Multi-agent framework doc: `/Users/Kravtsovd/projects/chrome-extension-tcs/algorithms/product_div/Multi_agent_framework/00_MULTI_AGENT_ORCHESTRATOR.md`

**Examples of What NOT to Do:**
- ❌ Light text (#e8eaf6) on light background (#f5f7fa)
- ❌ Small font (16px) in diagrams with multi-line text
- ❌ Cramped nodes with 10px padding
- ❌ Low contrast gold (#ffd700) on white background
- ❌ Illegible edge labels without background

**Examples of What TO Do:**
- ✅ White (#ffffff) or bright gold (#ffd700) text on dark (#0a0e27) background
- ✅ Large font (24-28px) in diagrams, bold weight
- ✅ Generous node padding (40px+), spacing (120px+)
- ✅ High contrast accents: gold on dark blue, cyan on black
- ✅ Edge labels with solid dark background, bright text

**Evaluation Scoring:**
Each agent will self-score (0-10) on:
- Text readability (weight 3x): Can you read ALL text from 2m away?
- Color contrast (weight 3x): Contrast ratio ≥7:1 everywhere?
- Diagram quality (weight 2x): All 18 Mermaid diagrams perfect?
- Professional design (weight 1x): Looks polished and cohesive?
- Screenshot validation (weight 1x): Screenshots confirm quality?
- Technical correctness (weight 1x): Works in all browsers, responsive?

**WINNER = Highest weighted score. Disagreement = Methodology insight.**

---

**CRITICAL: ALL ARTIFACTS (HTML, screenshots, docs) MUST BE IN YOUR AGENT WORKSPACE FOLDER!**

**CRITICAL: RUN HEADLESS BROWSER TO VALIDATE! Don't guess if it's readable - SCREENSHOT IT!**
