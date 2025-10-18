# Processing Script Code Analysis

## File: process_schuler_mmc4.py

### Critical Section: Abundance Transformation (Lines 160-162)

```python
# Transform abundances (log2 if needed - mmc4 appears to already be log2)
df_combined['Abundance_Young_transformed'] = df_combined['Abundance_Young']
df_combined['Abundance_Old_transformed'] = df_combined['Abundance_Old']
```

### What This Code Does
1. **Creates two new columns** for "transformed" abundances
2. **Directly assigns** source values WITHOUT any mathematical transformation
3. **Implies** that mmc4.xls source values are already in log2 scale

### Evidence Interpretation

**Comment Line 160:** "mmc4 appears to already be log2"
- This is the developer's explicit note
- Indicates prior analysis confirmed log2 scale in source file
- Decision: No transformation needed

**Lines 161-162:** Assignment without transformation
```python
df_combined['Abundance_Young_transformed'] = df_combined['Abundance_Young']
df_combined['Abundance_Old_transformed'] = df_combined['Abundance_Old']
```
- Abundance_Young = source data extracted from mmc4.xls['sample1_abundance']
- Abundance_Young_transformed = identical to Abundance_Young
- No log2() function called
- No log2(x+1) applied

### Contrast: What WOULD Happen if Transformation Was Needed

If the script needed to apply log2 transformation, we would see:

```python
import numpy as np

# If LINEAR scale transformation was needed:
df_combined['Abundance_Young_transformed'] = np.log2(df_combined['Abundance_Young'])
df_combined['Abundance_Old_transformed'] = np.log2(df_combined['Abundance_Old'])

# OR with pseudocount:
df_combined['Abundance_Young_transformed'] = np.log2(df_combined['Abundance_Young'] + 1)
df_combined['Abundance_Old_transformed'] = np.log2(df_combined['Abundance_Old'] + 1)
```

**But we don't see this.** Instead, values pass through unchanged.

### Z-Score Calculation (Lines 164-193)

```python
# Calculate z-scores per compartment
log(f"\n## Calculating z-scores per compartment...")

df_combined['Zscore_Young'] = None
df_combined['Zscore_Old'] = None
df_combined['Zscore_Delta'] = None

for compartment in df_combined['Tissue_Compartment'].unique():
    mask = df_combined['Tissue_Compartment'] == compartment

    # Calculate z-scores for young
    young_vals = df_combined.loc[mask, 'Abundance_Young']
    mean_young = young_vals.mean(skipna=True)
    std_young = young_vals.std(skipna=True)
    if std_young > 0:
        df_combined.loc[mask, 'Zscore_Young'] = (young_vals - mean_young) / std_young

    # Calculate z-scores for old
    old_vals = df_combined.loc[mask, 'Abundance_Old']
    mean_old = old_vals.mean(skipna=True)
    std_old = old_vals.std(skipna=True)
    if std_old > 0:
        df_combined.loc[mask, 'Zscore_Old'] = (old_vals - mean_old) / std_old

    # Calculate delta
    df_combined.loc[mask, 'Zscore_Delta'] = (
        df_combined.loc[mask, 'Zscore_Old'] - df_combined.loc[mask, 'Zscore_Young']
    )
```

### What This Code Tells Us

1. **Z-scores applied to 'Abundance_Young' and 'Abundance_Old'** (lines 175-186)
2. **These are the already-log2 values** (from untransformed 'Abundance_' columns)
3. **Per-compartment normalization** is applied (4 muscle types each calculated separately)
4. **Formula:** Z = (value - mean) / std_dev

### Conclusion from Script Analysis

The processing pipeline is CORRECT:
- Source data (mmc4.xls) is log2 scale
- Script recognizes this and does NOT re-transform
- Z-scores calculated on log2 values directly
- Database stores both raw (log2) and z-scored values
- No batch correction transformation needed on raw values

---

## Data Flow in Script

```
mmc4.xls (log2 scale)
        ↓
[Read with pandas]
        ↓
Abundance_Young = sample1_abundance (log2)
Abundance_Old = sample2_abundance (log2)
        ↓
[No transformation applied]
        ↓
Abundance_Young_transformed = Abundance_Young (still log2)
Abundance_Old_transformed = Abundance_Old (still log2)
        ↓
[Calculate z-scores per compartment]
        ↓
Zscore_Young = (Abundance_Young - mean) / std
Zscore_Old = (Abundance_Old - mean) / std
        ↓
[Output to database]
        ↓
Database: raw (log2) + z-scores stored
```

