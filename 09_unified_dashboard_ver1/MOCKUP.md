# Dashboard Visual Mockup

## Main Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ECM Atlas - Unified Multi-Tissue Aging Dashboard                          │
│  ───────────────────────────────────────────────────────────────────────    │
│                                                                             │
│  📊 498 Proteins  |  🫀 2 Organs  |  🔬 5 Compartments  |  Δ Avg: +0.13    │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────┬──────────────────────────────────────────────────┬─────────────┐
│            │                                                  │             │
│  FILTERS   │           HEATMAP VISUALIZATION                  │   DETAILS   │
│            │                                                  │             │
│ 🔍 Search  │   Protein    Glom  Tubu   NP   IAF   OAF        │  (Click on  │
│ COL1A1...  │   ────────   ────  ────  ────  ────  ────       │   protein)  │
│            │                                                  │             │
│ 🫀 Organs  │   COL1A1     🔴   🔴    🔵   🟡   🟡          │             │
│ ☑ Kidney   │   (Collagen) +2.3  +1.8  -0.5  +1.2  +0.8       │             │
│ ☑ IVD      │                                                  │             │
│            │   FN1        🔵   🔵    🔴   🔴   🔴          │             │
│ 🧬 Category│   (Fibronec) -1.1  -0.9  +2.1  +1.5  +1.8       │             │
│ ☑ Collagens│                                                  │             │
│ ☑ ECM Glyc │   ACAN       ⬜   ⬜    🟢   🟢   🟢          │             │
│ ☐ Proteog. │   (Aggrecan) N/A   N/A   +0.2  +0.1  +0.3       │             │
│            │                                                  │             │
│ 📈 Trend   │   SERPINA1   🟡   🟡    ⬜   ⬜   ⬜          │             │
│ ☑ Increase │              -0.4  -0.6  N/A   N/A   N/A        │             │
│ ☑ Decrease │                                                  │             │
│ ☑ Stable   │   ...                                            │             │
│            │                                                  │             │
│ [Clear]    │                                                  │             │
│            │                                                  │             │
│            │   Legend:                                        │             │
│            │   🔴 +2 to +3  🟡 +0.5 to +2  🟢 -0.5 to +0.5  │             │
│            │   🔵 -2 to -0.5  ⬛ -3 to -2  ⬜ No data       │             │
│            │                                                  │             │
└────────────┴──────────────────────────────────────────────────┴─────────────┘
```

## Heatmap Example (Full View)

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Multi-Compartment ECM Protein Expression Heatmap                          │
│  ────────────────────────────────────────────────────────────────────────  │
│                                                                            │
│  Sort by: [Magnitude ▼] [Category] [Protein Name]    Export: [CSV] [PNG]  │
│                                                                            │
│  Protein (Gene)      │ Glomerular │ Tubulointerst │   NP   │  IAF  │ OAF  │
│  ────────────────────┼────────────┼───────────────┼────────┼───────┼──────│
│  COL1A1 (Collagen I) │    🔴      │      🔴       │   🔵   │  🟡   │ 🟡   │
│  Collagens           │   +2.31    │     +1.85     │  -0.52 │ +1.21 │ +0.83│
│  ────────────────────┼────────────┼───────────────┼────────┼───────┼──────│
│  COL3A1 (Collagen III│    🔴      │      🔴       │   🟢   │  🟡   │ 🟡   │
│  Collagens           │   +2.15    │     +1.92     │  +0.12 │ +0.95 │ +1.05│
│  ────────────────────┼────────────┼───────────────┼────────┼───────┼──────│
│  FN1 (Fibronectin)   │    🔵      │      🔵       │   🔴   │  🔴   │ 🔴   │
│  ECM Glycoproteins   │   -1.12    │     -0.87     │  +2.13 │ +1.48 │ +1.82│
│  ────────────────────┼────────────┼───────────────┼────────┼───────┼──────│
│  ACAN (Aggrecan)     │    ⬜      │      ⬜       │   🟢   │  🟢   │ 🟢   │
│  Proteoglycans       │    N/A     │      N/A      │  +0.21 │ +0.15 │ +0.28│
│  ────────────────────┼────────────┼───────────────┼────────┼───────┼──────│
│  SERPINA1 (A1AT)     │    🟡      │      🟡       │   ⬜   │  ⬜   │ ⬜   │
│  ECM Regulators      │   -0.38    │     -0.59     │   N/A  │  N/A  │ N/A  │
│  ────────────────────┼────────────┼───────────────┼────────┼───────┼──────│
│  ...                 │    ...     │      ...      │   ...  │  ...  │ ...  │
└────────────────────────────────────────────────────────────────────────────┘

Hover tooltip (на любой ячейке):
┌──────────────────────────────────────┐
│  COL1A1 - Collagen alpha-1(I) chain  │
│  ──────────────────────────────────  │
│  Compartment: Glomerular (Kidney)    │
│  Dataset: Randles_2021               │
│                                      │
│  Zscore Young:  -0.52                │
│  Zscore Old:    +1.79                │
│  Zscore Delta:  +2.31  ⬆️ Increased │
│                                      │
│  Category: Collagens                 │
│  Division: Core matrisome            │
│                                      │
│  [Click for details →]               │
└──────────────────────────────────────┘
```

## Detail Panel (After Click)

```
┌────────────────────────────────────────────────────────┐
│  PROTEIN DETAILS                              [Close ✕]│
│  ────────────────────────────────────────────────────  │
│                                                        │
│  COL1A1 - Collagen alpha-1(I) chain                    │
│  UniProt: P02452                                       │
│  Category: Collagens (Core matrisome)                  │
│                                                        │
│  ────────────────────────────────────────────────────  │
│                                                        │
│  EXPRESSION ACROSS COMPARTMENTS:                       │
│                                                        │
│  Compartment        Young    Old    Delta    Trend    │
│  ────────────────  ──────  ──────  ──────   ───────   │
│  Glomerular        -0.52   +1.79   +2.31    ⬆️ Up     │
│  Tubulointerst.    -0.35   +1.50   +1.85    ⬆️ Up     │
│  NP (Disc)         +0.15   -0.37   -0.52    ⬇️ Down   │
│  IAF (Disc)        -0.21   +1.00   +1.21    ⬆️ Up     │
│  OAF (Disc)        -0.10   +0.73   +0.83    ⬆️ Up     │
│                                                        │
│  ────────────────────────────────────────────────────  │
│                                                        │
│  VISUALIZATION:                                        │
│                                                        │
│   +3 ┤                                                 │
│      │         ●  ● Zscore Old                         │
│   +2 ┤      ●                                          │
│      │   ●                                             │
│   +1 ┤                  ●  ●                           │
│      │                                                 │
│    0 ┼───────────────────────────                      │
│      │      ○           ○                              │
│   -1 ┤   ○  ○  ○  Zscore Young                        │
│      │                                                 │
│   -2 ┤                                                 │
│      └─────────────────────────────                    │
│        Glom Tubu  NP  IAF  OAF                         │
│                                                        │
│  ────────────────────────────────────────────────────  │
│                                                        │
│  [Export Data] [View in External Tool]                │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## Filter Panel (Expanded)

```
┌─────────────────────────────┐
│  FILTERS                    │
│  ─────────────────────────  │
│                             │
│  🔍 SEARCH                  │
│  ┌─────────────────────┐   │
│  │ COL1A1          [×] │   │
│  └─────────────────────┘   │
│    Suggestions:             │
│    • COL1A1 (Collagen)      │
│    • COL1A2 (Collagen)      │
│                             │
│  🫀 ORGANS                  │
│  ☑ Kidney (458)             │
│  ☑ Intervertebral_disc (993)│
│                             │
│  🧬 COMPARTMENTS            │
│  ☑ Glomerular (229)         │
│  ☑ Tubulointerstitial (229) │
│  ☑ NP (300)                 │
│  ☑ IAF (317)                │
│  ☑ OAF (376)                │
│                             │
│  🧬 MATRISOME CATEGORY      │
│  ☑ Collagens (154)          │
│  ☑ ECM Glycoproteins (453)  │
│  ☐ Proteoglycans (85)       │
│  ☑ ECM Regulators (375)     │
│  ☑ Secreted Factors (222)   │
│  ☐ ECM-affiliated (162)     │
│                             │
│  📚 STUDY                   │
│  ☑ Randles_2021 (458)       │
│  ☑ Tam_2020 (993)           │
│                             │
│  📈 AGING TREND             │
│  ☑ Increased (>+0.5)        │
│  ☑ Decreased (<-0.5)        │
│  ☑ Stable (±0.5)            │
│                             │
│  ──────────────────────     │
│                             │
│  Showing: 350 proteins      │
│                             │
│  [Clear All] [Reset]        │
│                             │
└─────────────────────────────┘
```

## Mobile View (Responsive)

```
┌──────────────────────────────┐
│  ECM Atlas Dashboard    [≡]  │
│  ──────────────────────────  │
│  📊 498  🫀 2  🔬 5  Δ+0.13  │
└──────────────────────────────┘
│                              │
│  🔍 [Search proteins...]     │
│                              │
│  [≡ Filters (5 active)]      │
│                              │
├──────────────────────────────┤
│  HEATMAP (Scroll →)          │
│                              │
│  Protein      Glom  Tubu ... │
│  ──────────   ────  ──── ... │
│  COL1A1       🔴   🔴   ... │
│  (Collagen)   +2.3  +1.8 ... │
│                              │
│  FN1          🔵   🔵   ... │
│  (Fibronect)  -1.1  -0.9 ... │
│                              │
│  [Load more...]              │
│                              │
└──────────────────────────────┘
```

## Color Legend (Always Visible)

```
┌────────────────────────────────────────────────────┐
│  Z-Score Delta Color Scale                         │
│  ────────────────────────────────────────────────  │
│                                                    │
│  🔵 ──── 🟡 ──── 🟢 ──── 🟡 ──── 🔴             │
│  -3     -1      0      +1     +3                  │
│                                                    │
│  Decreased     Stable     Increased with aging     │
│                                                    │
│  ⬜ = No data available in this compartment       │
└────────────────────────────────────────────────────┘
```
