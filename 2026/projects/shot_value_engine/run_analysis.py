"""
Run the shot value engine analysis and generate visualisations.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.shot_value_engine.src.preprocessing import canonicalize_layout, layout_moments
from projects.shot_value_engine.src.value_model import fit_value_model, SHOT_FEATURE_COLS

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


STONE_COLS = [
    f"stone_{i}_{axis}"
    for i in range(1, 9)
    for axis in ("x", "y")
]
BASE_COLS = [
    "CompetitionID",
    "SessionID",
    "GameID",
    "EndID",
    "ShotID",
    "TeamID",
    "Task",
]

TASK_LABEL_PATH = Path(__file__).resolve().parent / "data" / "task_labels.csv"


def load_task_labels() -> pd.DataFrame:
    mapping = pd.read_csv(TASK_LABEL_PATH)
    mapping["Task"] = mapping["Task"].astype(int)
    return mapping


def load_powerplay_shots() -> pd.DataFrame:
    ends = pd.read_csv(PROJECT_ROOT / "Ends.csv")[[
        "CompetitionID",
        "SessionID",
        "GameID",
        "EndID",
        "PowerPlay",
        "Result",
    ]]
    shots = pd.read_csv(
        PROJECT_ROOT / "Stones.csv",
        usecols=BASE_COLS + STONE_COLS,
    )
    merged = shots.merge(
        ends,
        on=["CompetitionID", "SessionID", "GameID", "EndID"],
        how="left",
    )
    merged = merged[merged["PowerPlay"].fillna(0) > 0]
    return merged


def compute_layout_features(
    shots: pd.DataFrame,
    task_labels: dict[int, str],
    task_descriptions: dict[int, str],
) -> pd.DataFrame:
    records = []
    canonical_coords = []

    for row in shots.itertuples(index=False):
        coords = []
        for i in range(1, 9):
            x = getattr(row, f"stone_{i}_x")
            y = getattr(row, f"stone_{i}_y")
            coords.append((float(x), float(y)))
        stones = np.array(coords, dtype=float)
        canonical, symmetry = canonicalize_layout(stones)
        stats = layout_moments(canonical)
        record = {
            "CompetitionID": row.CompetitionID,
            "SessionID": row.SessionID,
            "GameID": row.GameID,
            "EndID": row.EndID,
            "ShotID": row.ShotID,
            "TeamID": row.TeamID,
            "Task": row.Task,
            "symmetry": symmetry,
            **stats,
            "Result": row.Result,
        }
        records.append(record)
        canonical_coords.append(canonical)

    features = pd.DataFrame(records)
    features["TaskCode"] = features["Task"].fillna(-1).astype(int)
    features["canonical_layout"] = canonical_coords
    features["TaskLabel"] = features["TaskCode"].map(task_labels).fillna("Task " + features["TaskCode"].astype(str))
    features["TaskDescription"] = features["TaskCode"].map(task_descriptions).fillna("Unspecified shot")
    features["Task"] = features["TaskCode"]
    return features


def plot_task_value(features: pd.DataFrame) -> None:
    summary = (
        features.groupby("TaskLabel")["predicted_result"]
        .mean()
        .reset_index()
        .sort_values("predicted_result", ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=summary,
        x="predicted_result",
        y="TaskLabel",
        color="#1f77b4",
    )
    plt.title("Top Shot Choices by Expected End Score (Power Plays)")
    plt.xlabel("Predicted score contribution")
    plt.ylabel("Shot category")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task_value.png", dpi=200)
    plt.close()


def plot_actual_vs_predicted(features: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 6))
    sns.histplot(
        data=features,
        x="Result",
        y="predicted_result",
        bins=30,
        pthresh=0.05,
        cmap="mako",
        cbar=True,
    )
    plt.plot([-2, 6], [-2, 6], linestyle="--", color="grey")
    plt.xlabel("Observed end score")
    plt.ylabel("Predicted end score")
    plt.title("Calibration Heatmap: Predicted vs Observed")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=200)
    plt.close()


def plot_heatmap(features: pd.DataFrame) -> None:
    high_value = features[features["predicted_result"] >= features["predicted_result"].quantile(0.75)]
    coords = np.vstack([layout for layout in high_value["canonical_layout"] if layout.size > 0])
    if coords.size == 0:
        return

    plt.figure(figsize=(6, 5))
    plt.hexbin(coords[:, 0], coords[:, 1], gridsize=40, cmap="magma")
    plt.colorbar(label="Shot density")
    plt.xlabel("Normalised X")
    plt.ylabel("Normalised Y")
    plt.title("High-value shot landing zones (canonical space)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "landing_zone_density.png", dpi=200)
    plt.close()


def plot_task_gap(features: pd.DataFrame) -> pd.DataFrame:
    summary = (
        features.groupby("TaskLabel")
        .agg(
            predicted_mean=("predicted_result", "mean"),
            actual_mean=("Result", "mean"),
            attempts=("Result", "count"),
        )
        .sort_values("predicted_mean", ascending=False)
    )
    summary["lift_vs_actual"] = summary["predicted_mean"] - summary["actual_mean"]
    top = summary.head(10).reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top,
        x="lift_vs_actual",
        y="TaskLabel",
        color="#2ca02c",
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Predicted - actual mean score")
    plt.ylabel("Shot category")
    plt.title("Opportunity Gap: Expected vs Actual Power-Play Outcomes")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task_gap.png", dpi=200)
    plt.close()

    return summary
def main() -> None:
    shots = load_powerplay_shots()
    task_mapping = load_task_labels()
    label_lookup = task_mapping.set_index("Task")["Label"].to_dict()
    description_lookup = task_mapping.set_index("Task")["Description"].to_dict()

    features = compute_layout_features(shots, label_lookup, description_lookup)
    features["Task"] = features["Task"].astype(str)
    numeric_cols = [col for col in SHOT_FEATURE_COLS if col not in {"Task"}]
    features[numeric_cols] = features[numeric_cols].fillna(0.0)

    artifacts = fit_value_model(features)
    features["predicted_result"] = artifacts.pipeline.predict(features[SHOT_FEATURE_COLS])

    features.to_pickle(OUTPUT_DIR / "shot_value_features.pkl")

    plot_task_value(features)
    plot_actual_vs_predicted(features)
    plot_heatmap(features)
    task_summary = plot_task_gap(features)

    task_summary.to_csv(OUTPUT_DIR / "task_value_summary.csv")

    print("Shot value outputs generated at", OUTPUT_DIR)


if __name__ == "__main__":
    main()

