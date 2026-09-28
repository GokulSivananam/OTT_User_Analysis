"""
model.py
Phase 9-13: Visualizations, Random Forest Model, Evaluation,
Feature Importance, Model Saving
OTT User Retention Analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


# ─────────────────────────────────────────────
# PHASE 9 — VISUALIZATIONS
# ─────────────────────────────────────────────

def create_visualizations(df, user_df, output_dir):
    """Create all required visualizations and save to outputs/."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # ── Visualization 1: Avg Session Duration by Subscription Plan ──
    print("\n[Viz 1] Average Session Duration by Subscription Plan")
    plan_avg = df.groupby("Subscription_Plan")["Session_Duration"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Blues_d", len(plan_avg))
    bars = ax.bar(plan_avg.index, plan_avg.values, color=colors, edgecolor="black", width=0.5)
    ax.set_title("Average Session Duration by Subscription Plan", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Subscription Plan", fontsize=12)
    ax.set_ylabel("Average Session Duration (minutes)", fontsize=12)
    ax.set_ylim(0, plan_avg.max() * 1.2)
    for bar, val in zip(bars, plan_avg.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "watch_time_segment.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    # ── Visualization 2: Avg Session Duration by Content Genre ──
    print("[Viz 2] Average Session Duration by Content Genre")
    genre_avg = df.groupby("Content_Genre")["Session_Duration"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("Set2", len(genre_avg))
    bars = ax.bar(genre_avg.index, genre_avg.values, color=colors, edgecolor="black", width=0.6)
    ax.set_title("Average Session Duration by Content Genre", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Content Genre", fontsize=12)
    ax.set_ylabel("Average Session Duration (minutes)", fontsize=12)
    ax.set_ylim(0, genre_avg.max() * 1.2)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    for bar, val in zip(bars, genre_avg.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "genre_engagement.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    # ── Visualization 3: Inactivity Period by Retention Status ──
    print("[Viz 3] Inactivity Period by Retention Status")
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"Retained": "#2ecc71", "Churned": "#e74c3c"}
    sns.boxplot(data=df, x="Retention_Status", y="Inactivity_Period",
                hue="Retention_Status", palette=palette, width=0.5,
                linewidth=1.5, legend=False, ax=ax)
    ax.set_title("Inactivity Period by Retention Status", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Retention Status", fontsize=12)
    ax.set_ylabel("Inactivity Period (days)", fontsize=12)
    # Add mean annotations
    for i, status in enumerate(df["Retention_Status"].unique()):
        mean_val = df[df["Retention_Status"] == status]["Inactivity_Period"].mean()
        ax.text(i, mean_val + 0.3, f"Mean: {mean_val:.1f}", ha="center", fontsize=10,
                color="navy", fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "inactivity_retention.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    # ── Visualization 4: Content Diversity vs Session Duration ──
    print("[Viz 4] Content Diversity vs Session Duration")
    if user_df is not None and "Content_Diversity" in user_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        palette = {"Retained": "#2ecc71", "Churned": "#e74c3c"}
        scatter = sns.scatterplot(
            data=user_df, x="Content_Diversity", y="Avg_Session_Duration",
            hue="Retention_Status", palette=palette, alpha=0.7, s=70, ax=ax
        )
        # Also draw mean line per diversity level
        cd_mean = user_df.groupby("Content_Diversity")["Avg_Session_Duration"].mean()
        ax.plot(cd_mean.index, cd_mean.values, color="navy", linewidth=2,
                linestyle="--", marker="D", markersize=6, label="Mean Duration")
        ax.set_title("Content Diversity vs Average Session Duration", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Content Diversity (Unique Genres per User)", fontsize=12)
        ax.set_ylabel("Average Session Duration (minutes)", fontsize=12)
        ax.legend(title="Retention Status", fontsize=10)
        ax.set_xticks(sorted(user_df["Content_Diversity"].unique()))
        plt.tight_layout()
        out_path = os.path.join(output_dir, "content_diversity.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")
    else:
        # Fallback: Session Duration Distribution by Retention Status
        print("  Content diversity not available — using fallback: Session Duration by Retention Status")
        fig, ax = plt.subplots(figsize=(8, 5))
        palette = {"Retained": "#2ecc71", "Churned": "#e74c3c"}
        for status, color in palette.items():
            subset = df[df["Retention_Status"] == status]["Session_Duration"]
            ax.hist(subset, bins=20, alpha=0.6, label=status, color=color, edgecolor="black")
        ax.set_title("Session Duration Distribution by Retention Status", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Session Duration (minutes)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend(title="Retention Status", fontsize=10)
        plt.tight_layout()
        out_path = os.path.join(output_dir, "content_diversity.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved (fallback): {out_path}")


# ─────────────────────────────────────────────
# PHASE 10 — RANDOM FOREST MODEL
# ─────────────────────────────────────────────

def build_and_evaluate_model(df, output_dir, model_dir):
    """Phase 10–13: Build, evaluate, and save the Random Forest model."""
    print("\n" + "=" * 60)
    print("PHASE 10 — RANDOM FOREST MODEL TRAINING")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Define features
    numerical_features = ["Session_Duration", "Inactivity_Period"]
    categorical_features = ["Viewing_History", "Content_Genre", "Subscription_Plan"]

    if "Content_Diversity" in df.columns:
        numerical_features.append("Content_Diversity")
        print("Content_Diversity included as a feature.")

    target = "Retention_Status"

    # Prepare data
    feature_cols = numerical_features + categorical_features
    X = df[feature_cols].copy()
    y = df[target].copy()

    print(f"\nFeatures used   : {feature_cols}")
    print(f"Target variable : {target}")
    print(f"Dataset size    : {X.shape[0]} rows")

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"\nTarget classes  : {list(le.classes_)}")
    print(f"Encoded as      : {list(range(len(le.classes_)))}")

    # Class distribution
    print("\nClass Distribution:")
    unique, counts = np.unique(y_encoded, return_counts=True)
    for cls, cnt in zip(unique, counts):
        pct = cnt / len(y_encoded) * 100
        print(f"  {le.classes_[cls]}: {cnt} ({pct:.2f}%)")

    # Train/test split with stratification
    min_class_count = min(counts)
    use_stratify = min_class_count >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded if use_stratify else None
    )
    print(f"\nTrain size : {len(X_train)}")
    print(f"Test size  : {len(X_test)}")
    print(f"Stratified : {use_stratify}")

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    # Full pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # Train
    print("\nTraining Random Forest (100 estimators)...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # ── Phase 11: Evaluation ──
    print("\n" + "=" * 60)
    print("PHASE 11 — MODEL EVALUATION")
    print("=" * 60)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\nAccuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    # Class imbalance check
    class_counts = np.unique(y_encoded, return_counts=True)[1]
    imbalance_ratio = max(class_counts) / min(class_counts)
    if imbalance_ratio > 1.5:
        print(f"NOTE: Class imbalance detected (ratio {imbalance_ratio:.2f}:1).")
        print("      'class_weight=balanced' was used to account for this.")
    else:
        print("NOTE: Classes are roughly balanced.")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_, yticklabels=le.classes_,
        linewidths=0.5, linecolor="gray", ax=ax
    )
    ax.set_title("Random Forest — Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix saved: {cm_path}")

    # ── Phase 12: Feature Importance ──
    print("\n" + "=" * 60)
    print("PHASE 12 — FEATURE IMPORTANCE")
    print("=" * 60)

    rf_model = pipeline.named_steps["classifier"]
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = numerical_features + cat_feature_names

    importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print("\nFeature Importances (sorted):")
    print(feat_imp_df.to_string(index=False))

    print(f"\nTop contributing feature: {feat_imp_df.iloc[0]['Feature']} "
          f"({feat_imp_df.iloc[0]['Importance']:.4f})")

    # Feature Importance Plot
    top_n = min(15, len(feat_imp_df))
    top_df = feat_imp_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", top_n)
    bars = ax.barh(top_df["Feature"][::-1], top_df["Importance"][::-1],
                   color=colors[::-1], edgecolor="black")
    ax.set_title("Random Forest — Feature Importance (Top Features)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    for bar, val in zip(bars, top_df["Importance"][::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    fi_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFeature importance chart saved: {fi_path}")

    # ── Phase 13: Save Model ──
    print("\n" + "=" * 60)
    print("PHASE 13 — SAVING MODEL")
    print("=" * 60)

    model_path = os.path.join(model_dir, "random_forest_model.pkl")
    le_path = os.path.join(model_dir, "label_encoder.pkl")

    joblib.dump(pipeline, model_path)
    joblib.dump(le, le_path)

    print(f"Model (pipeline) saved : {model_path}")
    print(f"Label encoder saved   : {le_path}")
    print("=" * 60)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "feature_importances": feat_imp_df,
        "label_encoder": le,
        "pipeline": pipeline
    }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(base_dir, "data", "OTT_User_Data.csv")
    output_dir = os.path.join(base_dir, "outputs")
    model_dir = os.path.join(base_dir, "model")

    from data_cleaning import load_dataset, inspect_dataset, clean_dataset
    from feature_engineering import calculate_content_diversity, declining_engagement

    df = load_dataset(filepath)
    df = inspect_dataset(df)
    df = clean_dataset(df)
    df, user_df = calculate_content_diversity(df)
    df = declining_engagement(df)

    create_visualizations(df, user_df, output_dir)
    results = build_and_evaluate_model(df, output_dir, model_dir)
