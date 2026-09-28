"""
data_cleaning.py
Phase 1 & 2: Dataset Inspection and Cleaning
OTT User Retention Analysis
"""

import pandas as pd
import numpy as np
import os


def load_dataset(filepath):
    """Load the CSV dataset and return a DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded from: {filepath}")
    return df


def inspect_dataset(df):
    """Phase 1: Full dataset inspection and data quality summary."""
    print("\n" + "=" * 60)
    print("PHASE 1 — DATASET INSPECTION")
    print("=" * 60)

    print(f"\nDataset Shape : {df.shape[0]} rows x {df.shape[1]} columns")

    print("\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. '{col}'")

    print("\nData Types:")
    print(df.dtypes.to_string())

    print("\nFirst 5 Rows:")
    print(df.head().to_string())

    print("\nLast 5 Rows:")
    print(df.tail().to_string())

    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    print(missing_df.to_string())

    print(f"\nDuplicate Rows : {df.duplicated().sum()}")

    print("\nUnique Values Per Column:")
    for col in df.columns:
        print(f"  {col}: {df[col].nunique()} unique values -> {list(df[col].dropna().unique()[:8])}")

    print("\nDescriptive Statistics (Numerical):")
    print(df.describe().to_string())

    # Verify required columns
    required_columns = [
        "User_ID", "Viewing_History", "Session_Duration",
        "Content_Genre", "Subscription_Plan", "Inactivity_Period", "Retention_Status"
    ]

    print("\nColumn Verification:")
    # Strip whitespace from column names for comparison
    actual_cols = [c.strip() for c in df.columns]
    missing_cols = []
    for req in required_columns:
        if req in actual_cols:
            print(f"  [OK] {req}")
        else:
            print(f"  [MISSING] {req}")
            missing_cols.append(req)

    if missing_cols:
        raise ValueError(
            f"Required columns missing from dataset: {missing_cols}\n"
            "Cannot proceed without these columns."
        )

    print("\nAll required columns are present.")
    print("=" * 60)
    return df


def clean_dataset(df):
    """Phase 2: Full data cleaning pipeline."""
    print("\n" + "=" * 60)
    print("PHASE 2 — DATA CLEANING")
    print("=" * 60)

    original_shape = df.shape
    print(f"\nBefore Cleaning: {original_shape[0]} rows x {original_shape[1]} columns")

    # Step 1: Strip whitespace from column names
    df.columns = df.columns.str.strip()
    print("\n[Step 1] Column names stripped of whitespace.")

    # Step 2: Strip whitespace from string values
    str_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()
    print(f"[Step 2] String whitespace stripped in columns: {list(str_cols)}")

    # Step 3: Remove duplicate rows
    dupes_before = df.duplicated().sum()
    df = df.drop_duplicates()
    dupes_removed = dupes_before - df.duplicated().sum()
    print(f"[Step 3] Duplicate rows removed: {dupes_before} found, {dupes_before} removed.")

    # Step 4: Convert Session_Duration to numeric (in case of string coercion issues)
    df["Session_Duration"] = pd.to_numeric(df["Session_Duration"], errors="coerce")
    print("[Step 4] Session_Duration converted to numeric.")

    # Step 5: Convert Inactivity_Period to numeric
    df["Inactivity_Period"] = pd.to_numeric(df["Inactivity_Period"], errors="coerce")
    print("[Step 5] Inactivity_Period converted to numeric.")

    # Step 6: Handle impossible negative values
    neg_session = (df["Session_Duration"] < 0).sum()
    neg_inactivity = (df["Inactivity_Period"] < 0).sum()
    if neg_session > 0:
        print(f"[Step 6] WARNING: {neg_session} negative Session_Duration values found — replacing with NaN.")
        df.loc[df["Session_Duration"] < 0, "Session_Duration"] = np.nan
    if neg_inactivity > 0:
        print(f"[Step 6] WARNING: {neg_inactivity} negative Inactivity_Period values found — replacing with NaN.")
        df.loc[df["Inactivity_Period"] < 0, "Inactivity_Period"] = np.nan
    if neg_session == 0 and neg_inactivity == 0:
        print("[Step 6] No impossible negative values found.")

    # Step 7: Handle missing numerical values with median
    missing_session = df["Session_Duration"].isnull().sum()
    missing_inactivity = df["Inactivity_Period"].isnull().sum()

    if missing_session > 0:
        median_session = df["Session_Duration"].median()
        df["Session_Duration"] = df["Session_Duration"].fillna(median_session)
        print(f"[Step 7] Session_Duration: {missing_session} missing values filled with median ({median_session:.2f}).")
    else:
        print("[Step 7] Session_Duration: No missing values.")

    if missing_inactivity > 0:
        median_inactivity = df["Inactivity_Period"].median()
        df["Inactivity_Period"] = df["Inactivity_Period"].fillna(median_inactivity)
        print(f"[Step 7] Inactivity_Period: {missing_inactivity} missing values filled with median ({median_inactivity:.2f}).")
    else:
        print("[Step 7] Inactivity_Period: No missing values.")

    # Step 8: Handle missing categorical values with mode
    cat_cols = ["Viewing_History", "Content_Genre", "Subscription_Plan", "Retention_Status"]
    for col in cat_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"[Step 8] {col}: {missing_count} missing values filled with mode ('{mode_val}').")
        else:
            print(f"[Step 8] {col}: No missing values.")

    # Step 9: Validate categorical values (after all imputation)
    print("\n[Step 9] Validating categorical values:")

    valid_viewing = ["Low", "Medium", "High"]
    invalid_viewing = df[~df["Viewing_History"].isin(valid_viewing)]
    print(f"  Viewing_History invalid entries: {len(invalid_viewing)}")

    valid_retention = ["Retained", "Churned"]
    invalid_retention = df[~df["Retention_Status"].isin(valid_retention)]
    print(f"  Retention_Status invalid entries: {len(invalid_retention)}")

    valid_genres = ["Drama", "Sci-Fi", "Action", "Romance", "Sports", "Comedy", "Thriller", "Documentary"]
    invalid_genre = df[~df["Content_Genre"].isin(valid_genres)]
    print(f"  Content_Genre invalid entries: {len(invalid_genre)}")

    valid_plans = ["Basic", "Standard", "Premium"]
    invalid_plan = df[~df["Subscription_Plan"].isin(valid_plans)]
    print(f"  Subscription_Plan invalid entries: {len(invalid_plan)}")

    # Step 10: Preserve User_ID as string
    df["User_ID"] = df["User_ID"].astype(str)
    print("\n[Step 10] User_ID preserved as string type.")

    # After cleaning summary
    final_shape = df.shape
    rows_removed = original_shape[0] - final_shape[0]
    print(f"\nAfter Cleaning: {final_shape[0]} rows x {final_shape[1]} columns")
    print(f"Total rows removed: {rows_removed} (duplicates: {dupes_before})")

    print("\nPost-Cleaning Missing Values:")
    print(df.isnull().sum().to_string())

    print("\nPost-Cleaning Data Types:")
    print(df.dtypes.to_string())

    print("\nData cleaning complete.")
    print("=" * 60)

    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "data", "OTT_User_Data.csv")
    df = load_dataset(filepath)
    df = inspect_dataset(df)
    df_clean = clean_dataset(df)
    print("\nSample cleaned data:")
    print(df_clean.head())
