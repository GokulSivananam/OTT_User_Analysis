"""
feature_engineering.py
Phase 5, 6, 7, 8: Customer Segment Analysis, Content Diversity,
Declining Engagement Detection, Subscription Plan Comparison
OTT User Retention Analysis
"""

import pandas as pd
import numpy as np
import os


def customer_segment_analysis(df):
    """Phase 5: Average watch time by customer segment (Subscription_Plan)."""
    print("\n" + "=" * 60)
    print("PHASE 5 — CUSTOMER SEGMENT ANALYSIS")
    print("=" * 60)

    print("\nAverage Watch Time and Inactivity by Subscription Plan:")
    segment = df.groupby("Subscription_Plan").agg(
        User_Count=("User_ID", "count"),
        Avg_Session_Duration=("Session_Duration", "mean"),
        Median_Session_Duration=("Session_Duration", "median"),
        Avg_Inactivity=("Inactivity_Period", "mean"),
    ).round(2)
    print(segment.to_string())

    print("\nRetention Distribution by Subscription Plan:")
    plan_ret = pd.crosstab(df["Subscription_Plan"], df["Retention_Status"])
    plan_ret["Retention_Rate_%"] = (
        plan_ret["Retained"] / plan_ret.sum(axis=1) * 100
    ).round(2)
    plan_ret["Churn_Rate_%"] = (
        plan_ret["Churned"] / plan_ret.sum(axis=1) * 100
    ).round(2)
    print(plan_ret.to_string())

    print("=" * 60)
    return df


def calculate_content_diversity(df):
    """
    Phase 6: Content Diversity Analysis.
    Each User_ID appears multiple times with different Content_Genre rows.
    Content_Diversity = number of unique genres viewed per user.
    """
    print("\n" + "=" * 60)
    print("PHASE 6 — CONTENT DIVERSITY ANALYSIS")
    print("=" * 60)

    # Check if users appear multiple times
    user_record_counts = df["User_ID"].value_counts()
    multi_row_users = (user_record_counts > 1).sum()
    print(f"\nTotal unique User_IDs   : {df['User_ID'].nunique()}")
    print(f"Users with multiple rows: {multi_row_users}")
    print(f"Max rows per user       : {user_record_counts.max()}")

    if multi_row_users > 0:
        print("\nMultiple rows per user detected.")
        print("Calculating Content_Diversity = unique Content_Genre count per user.")

        diversity = (
            df.groupby("User_ID")["Content_Genre"]
            .nunique()
            .reset_index()
            .rename(columns={"Content_Genre": "Content_Diversity"})
        )

        # Also capture other user-level attributes (take mode/first stable attributes)
        user_attrs = df.groupby("User_ID").agg(
            Viewing_History=("Viewing_History", lambda x: x.mode()[0]),
            Avg_Session_Duration=("Session_Duration", "mean"),
            Subscription_Plan=("Subscription_Plan", lambda x: x.mode()[0]),
            Avg_Inactivity=("Inactivity_Period", "mean"),
            Retention_Status=("Retention_Status", lambda x: x.mode()[0]),
        ).reset_index()

        user_df = pd.merge(user_attrs, diversity, on="User_ID")

        print(f"\nContent Diversity Statistics:")
        print(f"  Min Diversity : {user_df['Content_Diversity'].min()}")
        print(f"  Max Diversity : {user_df['Content_Diversity'].max()}")
        print(f"  Mean Diversity: {user_df['Content_Diversity'].mean():.2f}")
        print(f"  Median        : {user_df['Content_Diversity'].median()}")

        print("\nContent Diversity Distribution:")
        print(user_df["Content_Diversity"].value_counts().sort_index().to_string())

        print("\nAvg Session Duration by Content Diversity:")
        cd_sd = user_df.groupby("Content_Diversity")["Avg_Session_Duration"].agg(["mean", "count"])
        cd_sd.columns = ["Avg_Session_Duration", "User_Count"]
        print(cd_sd.round(2).to_string())

        print("\nContent Diversity vs Retention Status:")
        cd_ret = user_df.groupby("Content_Diversity")["Retention_Status"].value_counts().unstack(fill_value=0)
        if "Retained" in cd_ret.columns and "Churned" in cd_ret.columns:
            cd_ret["Retention_Rate_%"] = (
                cd_ret["Retained"] / cd_ret.sum(axis=1) * 100
            ).round(2)
        print(cd_ret.to_string())

        # Merge Content_Diversity back into the original row-level df
        df = pd.merge(df, diversity, on="User_ID", how="left")
        print("\nContent_Diversity column added to dataset.")

    else:
        print("\nWARNING: All users appear only once.")
        print("True user-level content diversity cannot be calculated from a single record per user.")
        print("Setting Content_Diversity = 1 as placeholder (each user watched 1 genre).")
        df["Content_Diversity"] = 1
        user_df = None

    print("=" * 60)
    return df, user_df if multi_row_users > 0 else None


def declining_engagement(df):
    """
    Phase 7: Identify early signs of declining engagement.
    Threshold method: Median-based.
    At Risk = Session_Duration < median AND Inactivity_Period > median
    """
    print("\n" + "=" * 60)
    print("PHASE 7 — EARLY DECLINING ENGAGEMENT")
    print("=" * 60)

    median_session = df["Session_Duration"].median()
    median_inactivity = df["Inactivity_Period"].median()

    print(f"\nThreshold Logic (Median-Based):")
    print(f"  Session Duration Median  : {median_session:.2f} minutes")
    print(f"  Inactivity Period Median : {median_inactivity:.2f} days")
    print(f"\n  'At Risk' = Session_Duration < {median_session:.2f} AND Inactivity_Period > {median_inactivity:.2f}")
    print(f"  'Active'  = all other users")

    at_risk_mask = (
        (df["Session_Duration"] < median_session) &
        (df["Inactivity_Period"] > median_inactivity)
    )
    df["Declining_Engagement"] = np.where(at_risk_mask, "At Risk", "Active")

    at_risk_count = at_risk_mask.sum()
    at_risk_pct = at_risk_count / len(df) * 100

    print(f"\nDeclining Engagement Results:")
    print(f"  Total Records   : {len(df)}")
    print(f"  At Risk         : {at_risk_count} ({at_risk_pct:.2f}%)")
    print(f"  Active          : {len(df) - at_risk_count} ({100 - at_risk_pct:.2f}%)")

    at_risk_df = df[df["Declining_Engagement"] == "At Risk"]
    print(f"\nAt-Risk Users Profile:")
    print(f"  Avg Session Duration : {at_risk_df['Session_Duration'].mean():.2f} minutes")
    print(f"  Avg Inactivity       : {at_risk_df['Inactivity_Period'].mean():.2f} days")

    print("\nRetention Status of At-Risk Users:")
    ar_ret = at_risk_df["Retention_Status"].value_counts()
    ar_ret_pct = at_risk_df["Retention_Status"].value_counts(normalize=True) * 100
    for status in ar_ret.index:
        print(f"  {status}: {ar_ret[status]} ({ar_ret_pct[status]:.2f}%)")

    print("=" * 60)
    return df


def subscription_plan_comparison(df, user_df=None):
    """Phase 8: Comprehensive subscription plan comparison."""
    print("\n" + "=" * 60)
    print("PHASE 8 — SUBSCRIPTION PLAN COMPARISON")
    print("=" * 60)

    print("\nSubscription Plan Metrics (Row-Level):")
    plan_stats = df.groupby("Subscription_Plan").agg(
        Record_Count=("User_ID", "count"),
        Avg_Session_Duration=("Session_Duration", "mean"),
        Avg_Inactivity=("Inactivity_Period", "mean"),
    ).round(2)

    # Retention per plan
    plan_ret = pd.crosstab(df["Subscription_Plan"], df["Retention_Status"])
    plan_ret["Retention_%"] = (
        plan_ret["Retained"] / plan_ret.sum(axis=1) * 100
    ).round(2)

    plan_combined = pd.concat([plan_stats, plan_ret["Retention_%"]], axis=1)
    print(plan_combined.to_string())

    if user_df is not None:
        print("\nContent Diversity by Subscription Plan (User Level):")
        cd_plan = user_df.groupby("Subscription_Plan")["Content_Diversity"].agg(["mean", "median", "max"])
        cd_plan.columns = ["Avg_Diversity", "Median_Diversity", "Max_Diversity"]
        print(cd_plan.round(2).to_string())

    print("\nWhich plan has the highest value per metric:")
    print(f"  Highest Avg Session Duration : {plan_stats['Avg_Session_Duration'].idxmax()} "
          f"({plan_stats['Avg_Session_Duration'].max():.2f} min)")
    print(f"  Lowest Avg Inactivity        : {plan_stats['Avg_Inactivity'].idxmin()} "
          f"({plan_stats['Avg_Inactivity'].min():.2f} days)")
    print(f"  Highest Retention Rate       : {plan_ret['Retention_%'].idxmax()} "
          f"({plan_ret['Retention_%'].max():.2f}%)")

    print("=" * 60)
    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "data", "OTT_User_Data.csv")

    from data_cleaning import load_dataset, inspect_dataset, clean_dataset
    df = load_dataset(filepath)
    df = inspect_dataset(df)
    df = clean_dataset(df)
    df = customer_segment_analysis(df)
    df, user_df = calculate_content_diversity(df)
    df = declining_engagement(df)
    df = subscription_plan_comparison(df, user_df)
