"""EDX text-file parsing and numerical processing (normalization, column
lookups). No UI code lives here so it can be tested and reused standalone."""

import pandas as pd


def parse_edx_file(filepath: str) -> pd.DataFrame:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    header_idx = next((i for i, line in enumerate(lines) if line.strip().lower().startswith("index")), None)
    if header_idx is None:
        raise ValueError("Header line starting with 'Index' not found.")
    header = lines[header_idx].split()
    rows = [[float(p) for p in l.split()] for l in lines[header_idx + 1:] if l.strip() and len(l.split()) >= 3]
    return pd.DataFrame(rows, columns=header).fillna(0)


def get_elements(df):
    return [c for c in df.columns if not c.lower().startswith(("index", "pos"))]


def get_pos_col(df):
    return next((c for c in df.columns if c.lower().startswith("pos")), df.columns[1])


def normalize_to_100(df, all_elements, active_elements):
    df_norm = df.copy()
    if not active_elements:
        return df_norm
    sum_vals = df_norm[active_elements].sum(axis=1)
    sum_vals = sum_vals.replace(0, 1)
    for el in active_elements:
        df_norm[el] = (df_norm[el] / sum_vals) * 100
    for el in all_elements:
        if el not in active_elements:
            df_norm[el] = 0.0
    return df_norm
