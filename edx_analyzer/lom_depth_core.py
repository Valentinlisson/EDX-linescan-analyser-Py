"""
LOM Depth Analyser  --  CORE (no GUI dependency)
-------------------------------------------------------------------------
Parsing, filtering, statistics and histogram helpers used by the
"LOM Depth Analyser" window of the EDX Line Scan Viewer.

Target file format (LOM / optical microscope measurement export, e.g. Keyence):

    "Title";"Plane measurement results data"
    "Date saved";"2026-08-14";"15:01"
    "Comment";
    "[ Main ]"
    "No.";"Measure";"Result";"Unit"
    "1";"2 Points";11,18;"um"
    "2";"2 Points";0,08;"um"
    ...
    ;"Count";0;"pcs"
    "[ Area ]"
    ...

The parser is deliberately tolerant:
  * encodings tried in order: utf-8-sig, cp1252, latin-1
  * delimiter auto-detected between ';', '\t' and ','
  * decimal comma OR decimal point, optional thousands separators
  * several "No./Measure/Result/Unit" blocks per file are concatenated
  * unrelated blocks ([ Area ], [ XY Measure ], summary rows...) are ignored

This module imports ONLY numpy / pandas so it can be unit-tested headless.
"""

from __future__ import annotations

import csv
import io
import os
import re

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
DELIMITERS = (";", "\t", ",")

H_NO      = {"no", "no.", "n", "n.", "n°", "num", "number", "index", "id"}
H_MEASURE = {"measure", "mesure", "measurement", "type", "label", "name"}
H_RESULT  = {"result", "results", "résultat", "resultat", "value", "valeur",
             "depth", "profondeur", "distance"}
H_UNIT    = {"unit", "units", "unité", "unite"}

COLUMNS = ["No", "Measure", "Result", "Unit", "File", "Source"]


class LomParseError(Exception):
    """Raised when a file contains no usable No./Measure/Result/Unit block."""


# ─────────────────────────────────────────────────────────────────────────────
#  Low level helpers
# ─────────────────────────────────────────────────────────────────────────────
def read_text(path: str) -> str:
    """Read a text file trying several encodings (LOM exports are often latin-1)."""
    last_err = None
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return f.read()
        except UnicodeDecodeError as e:
            last_err = e
    with open(path, "r", encoding="latin-1", errors="replace", newline="") as f:
        return f.read()


def detect_delimiter(text: str) -> str:
    """Pick the most likely field separator (';' wins over ',' because of decimal commas)."""
    head = "\n".join(text.splitlines()[:40])
    counts = {d: head.count(d) for d in DELIMITERS}
    for d in DELIMITERS:                      # priority order: ';' then tab then ','
        if counts[d] > 0:
            return d
    return ";"


def norm_field(value: str) -> str:
    """Normalise a header cell: strip quotes/spaces/BOM and lowercase."""
    if value is None:
        return ""
    return str(value).replace("﻿", "").strip().strip('"').strip().lower()


def parse_number(text) -> float | None:
    """
    Convert a LOM numeric cell to float.
    Handles '11,18', '11.18', '1 234,56', '1,234.56', '5,54 um', '1.2e-3'.
    Returns None when the cell holds no number.
    """
    if text is None:
        return None
    if isinstance(text, (int, float, np.integer, np.floating)):
        return float(text)

    s = str(text).strip().strip('"').strip()
    if not s:
        return None
    s = s.replace("\xa0", "").replace(" ", "")

    try:                                     # fast path: plain float / scientific
        return float(s)
    except ValueError:
        pass

    m = re.match(r"^[+-]?[0-9][0-9.,]*", s)  # drop trailing unit e.g. '5,54um'
    if not m:
        return None
    s = m.group(0)

    has_c, has_d = "," in s, "." in s
    if has_c and has_d:                      # last separator is the decimal one
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thousands = "." if dec == "," else ","
        s = s.replace(thousands, "").replace(dec, ".")
    elif has_c:
        s = s.replace(",", "") if s.count(",") > 1 else s.replace(",", ".")
    elif s.count(".") > 1:
        s = s.replace(".", "")

    try:
        return float(s)
    except ValueError:
        return None


def _is_section_marker(fields) -> bool:
    """True for lines such as '[ Main ]', '[ Area ]', '[ XY Measure ]'."""
    for f in fields:
        v = norm_field(f)
        if v:
            return v.startswith("[")
    return False


def _header_map(fields):
    """
    Return {'no':i, 'measure':j, 'result':k, 'unit':l} when the row looks like a
    'No.;Measure;Result;Unit' header, else None.
    """
    cells = [norm_field(f) for f in fields]
    mapping = {}
    for i, c in enumerate(cells):
        if not c:
            continue
        if "no" not in mapping and c in H_NO:
            mapping["no"] = i
        elif "measure" not in mapping and c in H_MEASURE:
            mapping["measure"] = i
        elif "result" not in mapping and c in H_RESULT:
            mapping["result"] = i
        elif "unit" not in mapping and c in H_UNIT:
            mapping["unit"] = i
    if "result" in mapping and ("no" in mapping or "measure" in mapping):
        return mapping
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  File parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_lom_csv(path: str) -> pd.DataFrame:
    """
    Parse one LOM measurement CSV and return a DataFrame:
        No (int) | Measure (str) | Result (float) | Unit (str) | File | Source
    Raises LomParseError when nothing usable is found.
    """
    text = read_text(path)
    delim = detect_delimiter(text)
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delim, quotechar='"'))

    fname = os.path.basename(path)
    records, skipped = [], 0
    i, n = 0, len(rows)

    while i < n:
        mapping = _header_map(rows[i]) if rows[i] else None
        if mapping is None:
            i += 1
            continue

        i += 1                                     # first data line of the block
        while i < n:
            fields = rows[i]
            if not fields or all(not norm_field(f) for f in fields):
                break                              # blank line ends the block
            if _is_section_marker(fields):
                break                              # '[ ... ]' ends the block
            if _header_map(fields) is not None:
                break                              # another table starts

            def cell(key):
                idx = mapping.get(key)
                if idx is None or idx >= len(fields):
                    return ""
                return str(fields[idx]).strip().strip('"').strip()

            no_val = parse_number(cell("no")) if "no" in mapping else None
            res_val = parse_number(cell("result"))
            if no_val is None or res_val is None or float(no_val) != int(no_val):
                skipped += 1                       # summary rows ('Count', 'Total area'...)
                i += 1
                continue

            records.append({
                "No": int(no_val),
                "Measure": cell("measure") or "Measurement",
                "Result": float(res_val),
                "Unit": cell("unit"),
                "File": fname,
                "Source": path,
            })
            i += 1

    if not records:
        # Fallback: headerless "1;2 Points;11,18;um" style files
        for fields in rows:
            if len(fields) < 3:
                continue
            no_val = parse_number(fields[0])
            res_val = parse_number(fields[2])
            if no_val is None or res_val is None or float(no_val) != int(no_val):
                continue
            records.append({
                "No": int(no_val),
                "Measure": str(fields[1]).strip().strip('"') or "Measurement",
                "Result": float(res_val),
                "Unit": str(fields[3]).strip().strip('"') if len(fields) > 3 else "",
                "File": fname,
                "Source": path,
            })

    if not records:
        raise LomParseError(
            f"No 'No.;Measure;Result;Unit' data block found in '{fname}'."
        )
    return pd.DataFrame.from_records(records, columns=COLUMNS)


def load_files(paths):
    """Parse several files. Returns (list_of_dataframes, list_of_(path, error_msg))."""
    frames, errors = [], []
    for p in paths:
        try:
            frames.append(parse_lom_csv(p))
        except Exception as exc:                      # noqa: BLE001 - reported to the user
            errors.append((p, str(exc)))
    return frames, errors


# ─────────────────────────────────────────────────────────────────────────────
#  Data model
# ─────────────────────────────────────────────────────────────────────────────
class DepthGroup:
    """
    A named experiment = one or several CSV files plotted as a single series.
    Example: 3 CSV files for experiment A, 4 other CSV files for experiment B.
    """

    def __init__(self, name: str, color: str):
        self.name = name
        self.color = color
        self.visible = True
        self.files = []            # list of {"path", "name", "df"}

    # -- files -----------------------------------------------------------
    def add_file(self, path: str) -> int:
        df = parse_lom_csv(path)
        self.files.append({"path": path, "name": os.path.basename(path), "df": df})
        return len(df)

    def add_values(self, values, source_name: str, unit: str = "µm",
                   measure: str = "Image thickness") -> int:
        """
        Add measurements that do not come from a CSV file - typically the
        thicknesses measured on a micrograph by the Image Zone Analyser.
        Returns how many values were added.
        """
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0
        path = f"<{source_name}>"
        i = 2
        while self.has_file(path):                 # keep the pseudo-path unique
            path = f"<{source_name} ({i})>"
            i += 1
        df = pd.DataFrame({
            "No": np.arange(1, vals.size + 1, dtype=int),
            "Measure": measure,
            "Result": vals,
            "Unit": unit,
            "File": source_name,
            "Source": path,
        })[COLUMNS]
        self.files.append({"path": path, "name": source_name, "df": df})
        return int(len(df))

    def remove_file(self, path: str):
        self.files = [f for f in self.files if f["path"] != path]

    def has_file(self, path: str) -> bool:
        return any(f["path"] == path for f in self.files)

    # -- data ------------------------------------------------------------
    @property
    def data(self) -> pd.DataFrame:
        if not self.files:
            return pd.DataFrame(columns=COLUMNS)
        return pd.concat([f["df"] for f in self.files], ignore_index=True)

    def measures(self):
        d = self.data
        return sorted(set(d["Measure"].astype(str))) if len(d) else []

    def units(self):
        d = self.data
        return sorted({u for u in d["Unit"].astype(str) if u}) if len(d) else []

    def select(self, measures=None, vmin=None, vmax=None) -> pd.DataFrame:
        """Return the data with an extra boolean column 'Kept' (measure + min/max filters)."""
        d = self.data.copy()
        if not len(d):
            d["Kept"] = pd.Series(dtype=bool)
            return d
        keep = pd.Series(True, index=d.index)
        if measures is not None:
            keep &= d["Measure"].astype(str).isin(list(measures))
        if vmin is not None:
            keep &= d["Result"] >= vmin
        if vmax is not None:
            keep &= d["Result"] <= vmax
        d["Kept"] = keep
        return d

    def values(self, measures=None, vmin=None, vmax=None):
        """Return (kept_values, n_total_after_measure_filter, n_excluded_by_min_max)."""
        d = self.select(measures, vmin, vmax)
        if not len(d):
            return np.array([]), 0, 0
        in_measure = d if measures is None else d[d["Measure"].astype(str).isin(list(measures))]
        kept = d.loc[d["Kept"], "Result"].to_numpy(dtype=float)
        return kept, int(len(in_measure)), int(len(in_measure) - len(kept))


# ─────────────────────────────────────────────────────────────────────────────
#  Statistics
# ─────────────────────────────────────────────────────────────────────────────
def compute_stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {k: float("nan") for k in
                ("min", "max", "mean", "median", "std", "q1", "q3", "range", "sum")} | {"n": 0}
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    sem = std / np.sqrt(v.size) if v.size else float("nan")
    return {
        "n": int(v.size),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "mean": mean,
        "median": float(np.median(v)),
        "std": std,
        "q1": float(np.percentile(v, 25)),
        "q3": float(np.percentile(v, 75)),
        "range": float(np.max(v) - np.min(v)),
        "sum": float(np.sum(v)),
        # dispersion and how well the mean itself is known
        "sem": float(sem),                       # standard error of the mean
        "ci95": float(1.96 * sem),               # half-width of its 95 % interval
        "cv": float(100.0 * std / mean) if mean else float("nan"),   # in %
        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Histogram / distribution
# ─────────────────────────────────────────────────────────────────────────────
def make_bin_edges(gmin: float, gmax: float, width: float, origin: float) -> np.ndarray:
    """
    Build a regular bin grid of step `width` aligned on `origin`, covering [gmin, gmax].
    All series share the same grid so they stay comparable on one graph.
    """
    if not np.isfinite(gmin) or not np.isfinite(gmax):
        return np.array([0.0, float(width)])
    width = float(width)
    if width <= 0:
        width = 0.05
    start = origin + np.floor((gmin - origin) / width + 1e-9) * width
    n = int(np.floor((gmax - start) / width + 1e-9)) + 1
    n = max(n, 1)
    return start + width * np.arange(n + 1, dtype=float)


def histogram(values, edges, mode: str = "count"):
    """
    Counts per bin. mode = 'count' | 'percent' | 'density'
    (percent = % of the kept values of that series).
    """
    v = np.asarray(values, dtype=float)
    counts, _ = np.histogram(v, bins=edges)
    counts = counts.astype(float)
    if mode == "percent" and counts.sum() > 0:
        counts = 100.0 * counts / counts.sum()
    elif mode == "density":
        w = np.diff(edges)
        total = counts.sum()
        if total > 0:
            counts = counts / (total * w)
    return counts


def bin_centers(edges) -> np.ndarray:
    e = np.asarray(edges, dtype=float)
    return (e[:-1] + e[1:]) / 2.0


def histogram_error(counts, mode: str, total: int, width: float):
    """
    Counting uncertainty of each column, in the units of the drawn histogram.

    A column holding n measurements carries a Poisson uncertainty of sqrt(n);
    that is what the error bars show, converted to the same axis as the bars.
    """
    n = np.asarray(counts, dtype=float)
    err = np.sqrt(np.maximum(n, 0.0))
    if mode == "percent":
        return 100.0 * err / total if total else err
    if mode == "density":
        return err / (total * float(width)) if total and width else err
    return err


def normal_curve(values, edges, mode: str = "count", points: int = 200):
    """
    The normal law of same mean and same standard deviation as the data,
    scaled to the histogram it is drawn over. Returns (x, y) or None.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return None
    mean, std = float(v.mean()), float(v.std(ddof=1))
    if std <= 0:
        return None
    e = np.asarray(edges, dtype=float)
    width = float(np.mean(np.diff(e))) if e.size > 1 else 1.0
    x = np.linspace(e[0], e[-1], int(points))
    pdf = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))
    if mode == "percent":
        return x, pdf * width * 100.0
    if mode == "density":
        return x, pdf
    return x, pdf * width * v.size


def smooth_curve(y, window: int):
    """Rolling mean over `window` bins (used for the distribution line only)."""
    y = np.asarray(y, dtype=float)
    if window is None or window <= 1 or y.size == 0:
        return y
    return pd.Series(y).rolling(window=int(window), center=True, min_periods=1).mean().to_numpy()


# ─────────────────────────────────────────────────────────────────────────────
#  Export helpers (plain DataFrames, written by the GUI)
# ─────────────────────────────────────────────────────────────────────────────
def build_summary_frame(rows) -> pd.DataFrame:
    """rows = list of (group_name, n_files, n_total, n_excluded, stats_dict, unit)."""
    out = []
    for name, n_files, n_total, n_excluded, st, unit in rows:
        out.append({
            "Group": name,
            "Files": n_files,
            "Unit": unit,
            "Values (total)": n_total,
            "Excluded (min/max)": n_excluded,
            "Values (kept)": st["n"],
            "Min": st["min"],
            "Max": st["max"],
            "Mean": st["mean"],
            "Median": st["median"],
            "Std dev": st["std"],
            "Variation coef. (%)": st.get("cv", float("nan")),
            "Std error of mean": st.get("sem", float("nan")),
            "Mean 95% CI low": st["mean"] - st.get("ci95", float("nan")),
            "Mean 95% CI high": st["mean"] + st.get("ci95", float("nan")),
            "Q1": st["q1"],
            "Q3": st["q3"],
            "IQR": st.get("iqr", float("nan")),
            "Range": st["range"],
        })
    return pd.DataFrame(out)


def build_histogram_frame(edges, series, mode: str = "count") -> pd.DataFrame:
    """series = list of (group_name, counts_array) sharing the same bin grid."""
    e = np.asarray(edges, dtype=float)
    df = pd.DataFrame({
        "Bin start": e[:-1],
        "Bin end": e[1:],
        "Bin center": bin_centers(e),
    })
    suffix = {"count": "count", "percent": "%", "density": "density"}.get(mode, "count")
    for name, counts in series:
        df[f"{name} ({suffix})"] = np.asarray(counts, dtype=float)
    return df


def build_values_frame(group: "DepthGroup", measures=None, vmin=None, vmax=None) -> pd.DataFrame:
    d = group.select(measures, vmin, vmax)
    if not len(d):
        return pd.DataFrame(columns=["Group", "File", "No", "Measure", "Result", "Unit", "Kept"])
    d = d.copy()
    d.insert(0, "Group", group.name)
    return d[["Group", "File", "No", "Measure", "Result", "Unit", "Kept"]]
