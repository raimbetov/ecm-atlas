#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _clean(cell: Optional[str]) -> str:
    if cell is None:
        return ""
    return cell.replace("\xa0", " ").strip().strip('"')


def _split(line: str) -> List[str]:
    return [_clean(part) for part in line.split("\t")]


def _parse_factors(factors: List[str]) -> pd.DataFrame:
    records: List[Dict[str, str]] = []
    for raw in factors:
        entry: Dict[str, str] = {}
        for clause in raw.split("|"):
            if ":" in clause:
                key, value = clause.split(":", 1)
                entry[_clean(key)] = _clean(value)
        records.append(entry)
    return pd.DataFrame(records)


def _parse_block(lines: List[str], start_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    header_line = lines[start_idx + 1]
    factor_line = lines[start_idx + 2]

    headers = _split(header_line)
    factors = _split(factor_line)
    if not headers or headers[0].lower() != 'samples':
        raise RuntimeError('Unexpected header row in mwtab block')
    if not factors or factors[0].lower() != 'factors':
        raise RuntimeError('Unexpected factor row in mwtab block')

    sample_ids = headers[1:]
    factor_df = _parse_factors(factors[1:])
    factor_df.insert(0, 'sample_id', sample_ids)

    data_rows = []
    idx = start_idx + 3
    while idx < len(lines) and 'METABOLITE_DATA_END' not in lines[idx]:
        cells = _split(lines[idx])
        if not cells:
            idx += 1
            continue
        metabolite = cells[0]
        values = cells[1:]
        for sample_id, raw_value in zip(sample_ids, values):
            value = None
            if raw_value not in {'', 'NA', 'NaN'}:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = None
            data_rows.append({'sample_id': sample_id, 'metabolite': metabolite, 'value': value})
        idx += 1
    long_df = pd.DataFrame(data_rows)
    long_df = long_df.merge(factor_df, on='sample_id', how='left')
    return factor_df, long_df


def parse_mwtab(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lines = path.read_text().splitlines()
    factors_list: List[pd.DataFrame] = []
    long_frames: List[pd.DataFrame] = []

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.endswith('METABOLITE_DATA_START'):
            factor_df, long_df = _parse_block(lines, idx)
            factors_list.append(factor_df)
            long_frames.append(long_df)
            # advance to end marker
            while idx < len(lines) and 'METABOLITE_DATA_END' not in lines[idx]:
                idx += 1
        idx += 1

    if not long_frames:
        raise RuntimeError(f'No metabolite data blocks found in {path}')

    factors = pd.concat(factors_list, ignore_index=True).drop_duplicates(subset=['sample_id'])
    long_df = pd.concat(long_frames, ignore_index=True)
    return factors, long_df


def main():
    parser = argparse.ArgumentParser(description='Parse Metabolomics Workbench mwtab files')
    parser.add_argument('mwtab', nargs='+', type=Path)
    parser.add_argument('--outdir', type=Path, default=Path('data_codex'))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for path in args.mwtab:
        factors, long_df = parse_mwtab(path)
        dataset_id = path.stem.split('_')[0]
        factors_path = args.outdir / f'{dataset_id}_sample_factors.csv'
        long_path = args.outdir / f'{dataset_id}_metabolites_long.csv'
        wide_path = args.outdir / f'{dataset_id}_metabolites_wide.csv'

        factors.to_csv(factors_path, index=False)
        long_df.to_csv(long_path, index=False)
        wide_df = long_df.pivot_table(index='metabolite', columns='sample_id', values='value')
        wide_df.to_csv(wide_path)

        manifest.append({
            'dataset_id': dataset_id,
            'mwtab_path': str(path),
            'samples': len(factors),
            'metabolites': long_df['metabolite'].nunique(),
            'sample_factors': str(factors_path),
            'metabolites_long': str(long_path),
            'metabolites_wide': str(wide_path),
        })

    manifest_path = args.outdir / 'mwtab_manifest_codex.json'
    with manifest_path.open('w') as fh:
        json.dump(manifest, fh, indent=2)


if __name__ == '__main__':
    main()
