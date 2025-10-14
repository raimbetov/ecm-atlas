#!/usr/bin/env python3
"""Explore Excel structure to understand the data format"""

import pandas as pd

excel_file = "../data_raw/Tam et al. - 2020/elife-64940-supp1-v3.xlsx"

print("Exploring Excel file structure...")
print("=" * 80)

# Load Excel file
xl = pd.ExcelFile(excel_file)
print(f"Sheet names: {xl.sheet_names}\n")

# Explore Raw data sheet
print("=" * 80)
print("RAW DATA SHEET")
print("=" * 80)

# Try different header rows
for header_row in [0, 1, 2, 3]:
    print(f"\n--- Reading with header={header_row} ---")
    df = pd.read_excel(excel_file, sheet_name="Raw data", header=header_row, nrows=5)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns[:10])}")
    print(f"\nFirst row:")
    print(df.iloc[0] if len(df) > 0 else "Empty")

# Load Sample information
print("\n" + "=" * 80)
print("SAMPLE INFORMATION SHEET")
print("=" * 80)

for header_row in [0, 1, 2]:
    print(f"\n--- Reading with header={header_row} ---")
    df_meta = pd.read_excel(excel_file, sheet_name="Sample information", header=header_row, nrows=10)
    print(f"Shape: {df_meta.shape}")
    print(f"Columns: {list(df_meta.columns)}")
    print(f"\nFirst few rows:")
    print(df_meta.head(3))
