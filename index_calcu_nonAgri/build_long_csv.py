"""
Build a month-by-month long CSV for non-agricultural parcels.

Input files:
    Optical_Features_out_4.csv ... Optical_Features_out_10.csv
    Texture_Features_out_4.csv ... Texture_Features_out_10.csv

Output file:
    Dynamic_TimeSeries_LongTable.csv

The output columns are:
    parcel_id, feature columns..., mouth

Note: the field name is kept as "mouth" because the requested header uses
"mouth(4-10)". Change MOUTH_COLUMN to "month" if you later want the standard
English spelling.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from project_config import CONFIG


logging.basicConfig(level=logging.INFO, format="%(message)s")

OUT_DIR = CONFIG.date_out_non_agri_dir
FULL_MONTHS = CONFIG.get("processing", "full_months", [4, 5, 6, 7, 8, 9, 10])
OUTPUT_CSV = OUT_DIR / "Dynamic_TimeSeries_LongTable.csv"
MOUTH_COLUMN = "mouth"

# These three texture columns are expected from extract_texture_nonAgri.py.
# If a texture CSV is missing, the script keeps the table structure stable and
# fills these fields with 0.0, matching the old build_csv.py behavior.
TEXTURE_COLUMNS = ["GLCM_Contrast", "GLCM_Correlation", "LBP_Variance"]

# CSV headers must not exceed 10 characters. Most names are already short
# enough; only long names are mapped here. The original CSV files are not
# changed, only the final long table header is shortened.
SHORT_COLUMN_NAMES = {
    "NDVI_median": "NDVI_med",
    "GCVI_median": "GCVI_med",
    "SAVI_median": "SAVI_med",
    "NDWI_median": "NDWI_med",
    "GLCM_Contrast": "GLCM_Con",
    "GLCM_Correlation": "GLCM_Cor",
    "LBP_Variance": "LBP_Var",
}


def normalize_parcel_id(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    """Keep parcel_id as clean text so monthly tables can be merged reliably."""
    if "parcel_id" not in df.columns:
        raise ValueError(f"Missing parcel_id column: {source_path}")

    df = df.copy()
    df["parcel_id"] = (
        df["parcel_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    )
    return df.drop_duplicates(subset=["parcel_id"], keep="first")


def shorten_unknown_column(name: str, used_names: set[str]) -> str:
    """
    Shorten an unexpected column to <= 10 characters.

    This fallback is only for future extra features not listed in
    SHORT_COLUMN_NAMES. Known columns use explicit readable names above.
    """
    cleaned = "".join(ch for ch in name if ch.isalnum() or ch == "_")
    base = cleaned[:10] or "feature"
    candidate = base

    suffix_index = 1
    while candidate in used_names:
        suffix = str(suffix_index)
        candidate = f"{base[:10 - len(suffix)]}{suffix}"
        suffix_index += 1

    return candidate


def build_rename_map(columns: list[str]) -> dict[str, str]:
    """Create a safe header map and stop early if duplicate names appear."""
    rename_map: dict[str, str] = {}
    used_names = {"parcel_id", MOUTH_COLUMN}

    for column in columns:
        # parcel_id and mouth are fixed required fields. They are created or
        # preserved by this script, so they must not be renamed.
        if column in ("parcel_id", MOUTH_COLUMN):
            continue

        short_name = SHORT_COLUMN_NAMES.get(column, column)
        if len(short_name) > 10 or short_name in used_names:
            short_name = shorten_unknown_column(short_name, used_names)

        rename_map[column] = short_name
        used_names.add(short_name)

    long_names = [name for name in used_names if len(name) > 10]
    if long_names:
        raise ValueError(f"Headers longer than 10 characters remain: {long_names}")

    return rename_map


def read_month_table(month: int) -> pd.DataFrame | None:
    """Read one month of optical and texture features, then merge by parcel_id."""
    optical_csv = OUT_DIR / f"Optical_Features_out_{month}.csv"
    texture_csv = OUT_DIR / f"Texture_Features_out_{month}.csv"

    if not optical_csv.exists():
        logging.warning(f"[-] Missing optical CSV for month {month}: {optical_csv}")
        return None

    df_optical = normalize_parcel_id(pd.read_csv(optical_csv), optical_csv)

    if texture_csv.exists():
        df_texture = normalize_parcel_id(pd.read_csv(texture_csv), texture_csv)
        df_month = pd.merge(df_optical, df_texture, on="parcel_id", how="left")
        for column in TEXTURE_COLUMNS:
            if column not in df_month.columns:
                df_month[column] = 0.0
        df_month[TEXTURE_COLUMNS] = df_month[TEXTURE_COLUMNS].fillna(0.0)
    else:
        logging.warning(f"[-] Missing texture CSV for month {month}: {texture_csv}")
        df_month = df_optical.copy()
        for column in TEXTURE_COLUMNS:
            if column not in df_month.columns:
                df_month[column] = 0.0

    df_month[MOUTH_COLUMN] = month
    return df_month


def build_long_csv() -> Path:
    """Combine monthly feature CSV files into one long table."""
    logging.info("=" * 70)
    logging.info("Building non-agricultural long table...")
    logging.info(f"Input/output directory: {OUT_DIR}")

    month_tables: list[pd.DataFrame] = []
    feature_columns: list[str] = []

    for month in FULL_MONTHS:
        df_month = read_month_table(int(month))
        if df_month is None:
            continue

        # Preserve the first-seen feature order: optical columns first, then
        # texture columns. This makes the output stable and easy to inspect.
        for column in df_month.columns:
            if column not in ("parcel_id", MOUTH_COLUMN) and column not in feature_columns:
                feature_columns.append(column)

        month_tables.append(df_month)
        logging.info(f"[+] Month {month}: {len(df_month)} rows")

    if not month_tables:
        raise FileNotFoundError(
            f"No monthly Optical_Features_out_*.csv files were found in {OUT_DIR}"
        )

    ordered_columns = ["parcel_id", *feature_columns, MOUTH_COLUMN]
    rename_map = build_rename_map(ordered_columns)

    aligned_tables = []
    for df_month in month_tables:
        aligned = df_month.reindex(columns=ordered_columns)
        aligned[feature_columns] = aligned[feature_columns].fillna(0.0)
        aligned_tables.append(aligned)

    long_df = pd.concat(aligned_tables, ignore_index=True)
    long_df.rename(columns=rename_map, inplace=True)

    final_columns = ["parcel_id", *[rename_map[col] for col in feature_columns], MOUTH_COLUMN]
    long_df = long_df[final_columns]
    long_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    logging.info("-" * 70)
    logging.info(f"[OK] Output CSV: {OUTPUT_CSV}")
    logging.info(f"[OK] Rows: {len(long_df)}, Columns: {len(long_df.columns)}")
    logging.info("[OK] All output headers are <= 10 characters.")
    logging.info("=" * 70)
    return OUTPUT_CSV


if __name__ == "__main__":
    build_long_csv()
