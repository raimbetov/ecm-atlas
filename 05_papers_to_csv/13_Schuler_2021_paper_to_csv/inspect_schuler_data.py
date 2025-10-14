#!/usr/bin/env python3
"""
Inspect Schuler et al. 2021 supplementary data files
to identify which contains LFQ proteomics data.
"""

import pandas as pd
import os
from pathlib import Path

# Data directory
data_dir = Path("/home/raimbetov/GitHub/ecm-atlas/data_raw/Schuler et al. - 2021/")

# Files to inspect
files = [
    "mmc2.xls",
    "mmc3.xls",
    "mmc4.xls",
    "mmc5.xlsx",
    "mmc6.xlsx",
    "mmc7.xlsx",
    "mmc8.xls",
    "mmc9.xlsx"
]

output_lines = []
output_lines.append("# Schuler et al. 2021 - Data File Inspection Report\n")
output_lines.append(f"Generated: 2025-10-15\n")
output_lines.append(f"Location: {data_dir}\n")
output_lines.append("=" * 80 + "\n\n")

for filename in files:
    filepath = data_dir / filename
    output_lines.append(f"\n## FILE: {filename}\n")
    output_lines.append(f"Size: {filepath.stat().st_size / 1024 / 1024:.2f} MB\n\n")

    try:
        # Read Excel file
        xls = pd.ExcelFile(filepath)
        output_lines.append(f"**Sheet names:** {xls.sheet_names}\n")
        output_lines.append(f"**Number of sheets:** {len(xls.sheet_names)}\n\n")

        # Inspect each sheet
        for sheet_name in xls.sheet_names[:3]:  # Limit to first 3 sheets
            output_lines.append(f"### Sheet: {sheet_name}\n\n")

            try:
                # Read first 10 rows
                df = pd.read_excel(filepath, sheet_name=sheet_name, nrows=10)

                output_lines.append(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns\n\n")
                output_lines.append(f"**Columns ({len(df.columns)} total):**\n")

                # List all columns
                for i, col in enumerate(df.columns[:20]):  # First 20 columns
                    output_lines.append(f"  {i+1}. {col}\n")

                if len(df.columns) > 20:
                    output_lines.append(f"  ... and {len(df.columns) - 20} more columns\n")

                output_lines.append("\n**First 3 rows preview:**\n")
                output_lines.append("```\n")
                output_lines.append(df.head(3).to_string() + "\n")
                output_lines.append("```\n\n")

                # Look for indicators of proteomics data
                indicators = {
                    "Protein ID": any("protein" in str(col).lower() and "id" in str(col).lower() for col in df.columns),
                    "Gene Symbol": any("gene" in str(col).lower() for col in df.columns),
                    "LFQ/Intensity": any("lfq" in str(col).lower() or "intensity" in str(col).lower() for col in df.columns),
                    "Young/Old": any("young" in str(col).lower() or "old" in str(col).lower() or "aged" in str(col).lower() for col in df.columns),
                    "Niche/MuSC": any("niche" in str(col).lower() or "musc" in str(col).lower() or "satellite" in str(col).lower() for col in df.columns),
                    "Age values": any(str(col).lower() in ["age", "age_months", "age_mo"] for col in df.columns),
                }

                output_lines.append("**Proteomics Indicators:**\n")
                for indicator, present in indicators.items():
                    status = "✅" if present else "❌"
                    output_lines.append(f"  {status} {indicator}\n")

                output_lines.append("\n")

            except Exception as e:
                output_lines.append(f"  ⚠️ Error reading sheet: {e}\n\n")

        if len(xls.sheet_names) > 3:
            output_lines.append(f"... and {len(xls.sheet_names) - 3} more sheets\n\n")

    except Exception as e:
        output_lines.append(f"⚠️ **Error reading file:** {e}\n\n")

    output_lines.append("-" * 80 + "\n")

# Write report
report_path = Path("/home/raimbetov/GitHub/ecm-atlas/schuler_2021_data_inspection.md")
with open(report_path, "w") as f:
    f.writelines(output_lines)

print(f"✅ Inspection complete!")
print(f"📄 Report saved to: {report_path}")
print(f"\n🔍 Quick summary:")
print("   Processing 8 files...")
