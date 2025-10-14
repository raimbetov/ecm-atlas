# Dashboard Version Management Guide

## Current Version: 1.2.0

## How to Update Version

### 1. Edit `version.json`

When making changes to the dashboard, update the version file:

```json
{
  "version": "1.X.Y",
  "date": "YYYY-MM-DD",
  "changelog": [
    {
      "version": "1.X.Y",
      "date": "YYYY-MM-DD",
      "changes": [
        "Added: New feature description",
        "Fixed: Bug fix description",
        "Changed: Modification description"
      ]
    },
    // ... previous versions
  ]
}
```

### 2. Semantic Versioning

Follow semantic versioning: **MAJOR.MINOR.PATCH**

- **MAJOR** (1.x.x): Breaking changes, major redesign
- **MINOR** (x.1.x): New features, enhancements (backward compatible)
- **PATCH** (x.x.1): Bug fixes, small improvements

### 3. Changelog Entry Format

Use prefixes for clarity:
- **Added:** New features
- **Fixed:** Bug fixes
- **Changed:** Changes to existing features
- **Removed:** Removed features
- **Improved:** Performance or UX improvements

### 4. Examples

#### Minor Version (New Feature)
```json
{
  "version": "1.3.0",
  "date": "2025-10-14",
  "changes": [
    "Added: Export heatmap as PNG",
    "Added: Protein annotation details panel"
  ]
}
```

#### Patch Version (Bug Fix)
```json
{
  "version": "1.2.1",
  "date": "2025-10-13",
  "changes": [
    "Fixed: Heatmap tooltip not showing for some proteins",
    "Fixed: Sort by magnitude calculation error"
  ]
}
```

## Current Version History

### v1.2.0 (2025-10-13)
- Fixed: Normalize Gene_Symbol to uppercase (merged 247 duplicates)
- Fixed: Aggregate multiple isoforms per protein
- Fixed: Aging Trend filters now support multiple selections
- Added: Isoform count in tooltips

### v1.1.0 (2025-10-13)
- Added: Multi-dataset comparison heatmap
- Added: Advanced filtering (organs, compartments, categories, trends)
- Added: Protein search functionality

### v1.0.0 (2025-10-12)
- Initial release
- Individual dataset analysis with 6 visualization types
- Support for 5 LFQ proteomics datasets

## Version Display

The version is displayed in the top-right corner of the dashboard header.

**Features:**
- Shows current version number (e.g., "v1.2.0")
- Hover tooltip displays latest changelog
- Badge has semi-transparent background
- Updates automatically on page load

## Testing Version Update

1. Edit `version.json`
2. Restart API server (if running)
3. Refresh dashboard (Ctrl+Shift+R)
4. Check version badge in header
5. Hover over badge to see changelog tooltip

## Notes

- Version is loaded from `/api/version` endpoint
- If version.json is not found, displays "unknown"
- Version shown in browser console on load
- Always commit version.json changes with corresponding code changes
