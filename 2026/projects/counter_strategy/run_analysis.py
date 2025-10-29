"""
Run the counter-strategy simulator analysis with visualisations.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.counter_strategy.src.opponent_clustering import (
    build_feature_table,
    fit_opponent_clusters,
    assign_clusters,
    FEATURE_COLUMNS,
)
from projects.counter_strategy.src.defense_optimizer import DefenseOption, minimax_defense

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAYOUT_LABELS = {
    1: "Center setup",
    2: "Corner setup",
}

def load_powerplay_ends() -> pd.DataFrame:
    ends = pd.read_csv(PROJECT_ROOT / "Ends.csv")
    ends = ends[ends["PowerPlay"].fillna(0) > 0].copy()
    ends["Result"] = ends["Result"].fillna(0).astype(int)
    ends = ends[ends["Result"] < 9]
    return ends


def compute_cluster_distributions(ends: pd.DataFrame, assignment: pd.DataFrame) -> Dict[int, pd.Series]:
    score_values = sorted(ends["Result"].unique())
    cluster_distributions: Dict[int, pd.Series] = {}
    for cluster_id, group in assignment.groupby("cluster"):
        members = group["TeamID"].tolist()
        subset = ends[ends["TeamID"].isin(members)]
        dist = subset["Result"].value_counts().reindex(score_values, fill_value=0)
        cluster_distributions[cluster_id] = dist / dist.sum()
    return cluster_distributions


def compute_layout_distributions(ends: pd.DataFrame, assignment: pd.DataFrame) -> tuple[List[int], Dict[tuple[int, int], pd.Series]]:
    score_values = sorted(ends["Result"].unique())
    distributions: Dict[tuple[int, int], pd.Series] = {}

    for cluster_id, group in assignment.groupby("cluster"):
        members = group["TeamID"].tolist()
        subset = ends[ends["TeamID"].isin(members)]

        for layout_value, layout_group in subset.groupby("PowerPlay"):
            counts = layout_group["Result"].value_counts().reindex(score_values, fill_value=0)
            total = counts.sum()
            if total == 0:
                continue
            distributions[(cluster_id, int(layout_value))] = counts / total

    return score_values, distributions


def build_defense_options(score_values: List[int]) -> List[DefenseOption]:
    guard_impact = [max(val - 1, -1) if val >= 3 else val for val in score_values]
    freeze_impact = [val - 1 if val >= 2 else val for val in score_values]
    runback_impact = [val - 0.5 if val <= 0 else val + 0.3 for val in score_values]

    return [
        DefenseOption(name="Guard Wall", success_prob=0.75, score_impact=guard_impact),
        DefenseOption(name="Freeze Tap", success_prob=0.65, score_impact=freeze_impact),
        DefenseOption(name="Runback Pressure", success_prob=0.55, score_impact=runback_impact),
    ]


def evaluate_defenses(cluster_distributions: Dict[int, pd.Series]) -> pd.DataFrame:
    if not cluster_distributions:
        return pd.DataFrame()

    score_values = cluster_distributions[next(iter(cluster_distributions))].index.tolist()
    options = build_defense_options(score_values)

    records = []
    for cluster_id, dist in cluster_distributions.items():
        opponent_distribution = dist.values
        baseline_ev = np.dot(opponent_distribution, np.array(score_values))
        option_values = {}
        for option in options:
            mitigated = np.dot(opponent_distribution, option.score_impact)
            expected = option.success_prob * mitigated + (1 - option.success_prob) * baseline_ev
            option_values[option.name] = expected

        best_option = min(option_values.items(), key=lambda item: item[1])[0]
        best_value = option_values[best_option]

        records.append(
            {
                "cluster": cluster_id,
                "baseline_expected_points": baseline_ev,
                "recommended_option": best_option,
                "recommended_expected_points": best_value,
                "reduction": baseline_ev - best_value,
                **option_values,
            }
        )
    return pd.DataFrame(records)


SCENARIOS = [
    {"name": "baseline"},
    {"name": "freeze_cooldown", "global": {"Freeze Tap": {"success_prob": 0.5}}},
    {
        "name": "runback_hot",
        "global": {
            "Runback Pressure": {"success_prob": 0.8, "score_shift": -1.0},
            "Freeze Tap": {"success_prob": 0.6, "score_shift": 0.2},
        },
    },
    {
        "name": "guard_wall_plus",
        "global": {
            "Guard Wall": {"success_prob": 0.9, "score_shift": -0.7},
            "Freeze Tap": {"success_prob": 0.55, "score_shift": 0.3},
        },
    },
    {
        "name": "corner_runback_edge",
        "layout": {
            2: {
                "Runback Pressure": {"success_prob": 0.82, "score_shift": -0.8},
                "Freeze Tap": {"success_prob": 0.55, "score_shift": 0.25},
            }
        },
    },
]


def apply_adjustments(options: List[DefenseOption], adjustments: Dict[str, Dict[str, float]]) -> List[DefenseOption]:
    if not adjustments:
        return options

    adjusted = []
    for opt in options:
        overrides = adjustments.get(opt.name, {})
        success_prob = overrides.get("success_prob", opt.success_prob)
        impact = np.array(opt.score_impact, dtype=float)
        if "score_shift" in overrides:
            impact = impact + overrides["score_shift"]
        if "score_multiplier" in overrides:
            impact = impact * overrides["score_multiplier"]
        if "score_impact" in overrides:
            impact = np.array(overrides["score_impact"], dtype=float)
        adjusted.append(
            DefenseOption(name=opt.name, success_prob=success_prob, score_impact=impact.tolist())
        )
    return adjusted


def stress_test_defenses(
    score_values: List[int],
    layout_distributions: Dict[tuple[int, int], pd.Series],
    scenarios: List[Dict[str, Dict[str, Dict[str, float]]]],
) -> pd.DataFrame:
    scenario_records = []

    for scenario in scenarios:
        scenario_name = scenario["name"]
        global_overrides = scenario.get("global", {})
        layout_overrides = scenario.get("layout", {})

        for (cluster_id, layout), dist in layout_distributions.items():
            baseline_ev = np.dot(dist.values, np.array(score_values))
            options = build_defense_options(score_values)
            options = apply_adjustments(options, global_overrides)
            if layout in layout_overrides:
                options = apply_adjustments(options, layout_overrides[layout])

            option_values = {}
            for option in options:
                mitigated = np.dot(dist.values, option.score_impact)
                expected = option.success_prob * mitigated + (1 - option.success_prob) * baseline_ev
                option_values[option.name] = expected

            best_option = min(option_values.items(), key=lambda item: item[1])[0]
            best_value = option_values[best_option]

            scenario_records.append(
                {
                    "scenario": scenario_name,
                    "cluster": cluster_id,
                    "layout": layout,
                    "layout_label": LAYOUT_LABELS.get(layout, f"Layout {layout}"),
                    "baseline_expected_points": baseline_ev,
                    "recommended_option": best_option,
                    "recommended_expected_points": best_value,
                    "reduction": baseline_ev - best_value,
                    **option_values,
                }
            )

    return pd.DataFrame(scenario_records)


def plot_cluster_scatter(assignment: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=assignment,
        x="avg_score_gain",
        y="three_plus_rate",
        size="usage_count",
        hue="cluster",
        palette="viridis",
        sizes=(20, 200),
        alpha=0.7,
    )
    plt.xlabel("Average power-play points")
    plt.ylabel("Rate of 3+ point ends")
    plt.title("Opponent archetypes during power plays")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_scatter.png", dpi=200)
    plt.close()


def plot_defense_bars(summary: pd.DataFrame) -> None:
    value_columns = [
        col
        for col in summary.columns
        if col not in {"cluster", "baseline_expected_points", "recommended_option", "recommended_expected_points", "reduction"}
    ]
    melted = summary.melt(
        id_vars=["cluster", "baseline_expected_points", "recommended_option"],
        value_vars=value_columns,
        var_name="defense_option",
        value_name="expected_points",
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="cluster", y="expected_points", hue="defense_option")
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Expected opponent points")
    plt.xlabel("Cluster")
    plt.title("Defense scenario evaluation by cluster")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "defense_option_comparison.png", dpi=200)
    plt.close()


def plot_defense_delta(summary: pd.DataFrame) -> None:
    ordered = summary.sort_values("reduction", ascending=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=ordered,
        x="reduction",
        y="cluster",
        color="#2ca02c",
    )
    plt.xlabel("Reduction in opponent expected points")
    plt.ylabel("Cluster")
    plt.title("Impact of Recommended Defense")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "defense_reduction.png", dpi=200)
    plt.close()


def plot_stress_test_matrix(scenario_df: pd.DataFrame) -> None:
    if scenario_df.empty:
        return

    scenario_df = scenario_df.copy()
    scenario_df["cluster_layout"] = (
        "Cluster "
        + scenario_df["cluster"].astype(str)
        + " - "
        + scenario_df["layout_label"]
    )

    pivot = scenario_df.pivot(index="cluster_layout", columns="scenario", values="recommended_option")

    option_palette = {"Guard Wall": 0, "Freeze Tap": 1, "Runback Pressure": 2}
    numeric = pivot.replace(option_palette)

    plt.figure(figsize=(max(8, len(SCENARIOS) * 1.2), max(4, len(pivot) * 0.6)))
    ax = sns.heatmap(
        numeric,
        cmap="viridis",
        cbar=False,
        annot=pivot,
        fmt="",
        linewidths=0.5,
    )
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Cluster & layout")
    ax.set_title("Recommended defense across stress tests")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stress_test_matrix.png", dpi=200)
    plt.close()


def main() -> None:
    feature_frame = build_feature_table()
    model = fit_opponent_clusters(feature_frame)
    assignment = assign_clusters(feature_frame, model)

    ends = load_powerplay_ends()
    cluster_distributions = compute_cluster_distributions(ends, assignment)
    score_values, layout_distributions = compute_layout_distributions(ends, assignment)
    defense_summary = evaluate_defenses(cluster_distributions)
    enriched = assignment.merge(defense_summary[["cluster", "recommended_option"]], on="cluster", how="left")

    teams = pd.read_csv(PROJECT_ROOT / "Teams.csv")[["CompetitionID", "TeamID", "Name"]]
    team_lookup = teams.sort_values("CompetitionID").drop_duplicates("TeamID")
    enriched = enriched.merge(team_lookup[["TeamID", "Name"]], on="TeamID", how="left")

    plot_cluster_scatter(enriched)
    if not defense_summary.empty:
        plot_defense_bars(defense_summary)
        plot_defense_delta(defense_summary)

        playbook = (
            enriched.sort_values("usage_count", ascending=False)
            .groupby("cluster")
            .head(5)
            .loc[:, ["cluster", "Name", "usage_count", "avg_score_gain", "three_plus_rate", "recommended_option"]]
            .rename(
                columns={
                    "Name": "Team",
                    "usage_count": "PowerPlayEnds",
                    "avg_score_gain": "AvgPoints",
                    "three_plus_rate": "Rate3Plus",
                }
            )
        )
        playbook.to_csv(OUTPUT_DIR / "cluster_playbook.csv", index=False)

    scenario_df = stress_test_defenses(score_values, layout_distributions, SCENARIOS)
    if not scenario_df.empty:
        scenario_df.to_csv(OUTPUT_DIR / "defense_stress_tests.csv", index=False)
        plot_stress_test_matrix(scenario_df)

    enriched.to_csv(OUTPUT_DIR / "cluster_assignments.csv", index=False)
    defense_summary.to_csv(OUTPUT_DIR / "defense_summary.csv", index=False)

    print("Counter-strategy outputs generated at", OUTPUT_DIR)


if __name__ == "__main__":
    main()

