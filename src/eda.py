"""
eda.py
Phase 3 & 4: Exploratory Data Analysis and Viewing Behaviour Patterns
OTT User Retention Analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os


def run_eda(df, output_dir):
    """Phase 3: Descriptive analysis for all key columns."""
    print("\n" + "=" * 60)
    print("PHASE 3 — EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # --- Session Duration Stats ---
    print("\nSession Duration Statistics:")
    print(f"  Mean   : {df['Session_Duration'].mean():.2f} minutes")
    print(f"  Median : {df['Session_Duration'].median():.2f} minutes")
    print(f"  Min    : {df['Session_Duration'].min():.2f} minutes")
    print(f"  Max    : {df['Session_Duration'].max():.2f} minutes")
    print(f"  Std    : {df['Session_Duration'].std():.2f} minutes")

    # --- Inactivity Period Stats ---
    print("\nInactivity Period Statistics:")
    print(f"  Mean   : {df['Inactivity_Period'].mean():.2f} days")
    print(f"  Median : {df['Inactivity_Period'].median():.2f} days")
    print(f"  Min    : {df['Inactivity_Period'].min():.2f} days")
    print(f"  Max    : {df['Inactivity_Period'].max():.2f} days")

    # --- User Count ---
    print(f"\nTotal Users (unique User_ID) : {df['User_ID'].nunique()}")
    print(f"Total Records                : {len(df)}")

    # --- Retention Status ---
    print("\nRetention Status Counts:")
    retention_counts = df["Retention_Status"].value_counts()
    retention_pct = df["Retention_Status"].value_counts(normalize=True) * 100
    for status in retention_counts.index:
        print(f"  {status}: {retention_counts[status]} ({retention_pct[status]:.2f}%)")

    # --- Viewing History ---
    print("\nViewing History Distribution:")
    vh_counts = df["Viewing_History"].value_counts()
    vh_pct = df["Viewing_History"].value_counts(normalize=True) * 100
    for cat in vh_counts.index:
        print(f"  {cat}: {vh_counts[cat]} ({vh_pct[cat]:.2f}%)")

    # --- Content Genre ---
    print("\nContent Genre Distribution:")
    genre_counts = df["Content_Genre"].value_counts()
    genre_pct = df["Content_Genre"].value_counts(normalize=True) * 100
    for genre in genre_counts.index:
        print(f"  {genre}: {genre_counts[genre]} ({genre_pct[genre]:.2f}%)")

    # --- Subscription Plan ---
    print("\nSubscription Plan Distribution:")
    plan_counts = df["Subscription_Plan"].value_counts()
    plan_pct = df["Subscription_Plan"].value_counts(normalize=True) * 100
    for plan in plan_counts.index:
        print(f"  {plan}: {plan_counts[plan]} ({plan_pct[plan]:.2f}%)")

    print("=" * 60)


def viewing_behaviour_patterns(df):
    """Phase 4: Identify viewing behaviour patterns."""
    print("\n" + "=" * 60)
    print("PHASE 4 — VIEWING BEHAVIOUR PATTERNS")
    print("=" * 60)

    # Session Duration by Retention
    print("\nMean Session Duration by Retention Status:")
    sd_ret = df.groupby("Retention_Status")["Session_Duration"].agg(["mean", "median", "count"])
    sd_ret.columns = ["Mean", "Median", "Count"]
    print(sd_ret.round(2).to_string())

    # Session Duration by Viewing History
    print("\nMean Session Duration by Viewing History:")
    sd_vh = df.groupby("Viewing_History")["Session_Duration"].agg(["mean", "median", "count"])
    sd_vh.columns = ["Mean", "Median", "Count"]
    print(sd_vh.round(2).to_string())

    # Genre preferences
    print("\nTop Genres by Average Session Duration:")
    genre_sd = df.groupby("Content_Genre")["Session_Duration"].mean().sort_values(ascending=False)
    print(genre_sd.round(2).to_string())

    # Subscription Plan patterns
    print("\nSubscription Plan by Retention:")
    plan_ret = pd.crosstab(df["Subscription_Plan"], df["Retention_Status"])
    plan_ret["Retention_Rate_%"] = (
        plan_ret["Retained"] / plan_ret.sum(axis=1) * 100
    ).round(2)
    print(plan_ret.to_string())

    # Inactivity Patterns
    print("\nInactivity Period Statistics by Retention Status:")
    inact = df.groupby("Retention_Status")["Inactivity_Period"].agg(["mean", "median", "min", "max"])
    print(inact.round(2).to_string())

    # Viewing History vs Retention
    print("\nViewing History vs Retention Status:")
    vh_ret = pd.crosstab(df["Viewing_History"], df["Retention_Status"])
    vh_ret["Retention_Rate_%"] = (
        vh_ret["Retained"] / vh_ret.sum(axis=1) * 100
    ).round(2)
    print(vh_ret.to_string())

    print("=" * 60)
    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "data", "OTT_User_Data.csv")
    output_dir = os.path.join(base_dir, "outputs")

    from data_cleaning import load_dataset, inspect_dataset, clean_dataset
    df = load_dataset(filepath)
    df = inspect_dataset(df)
    df = clean_dataset(df)
    run_eda(df, output_dir)
    viewing_behaviour_patterns(df)
