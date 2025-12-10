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
from projects.counter_strategy.src.empirical_priors import compute_empirical_priors, get_default_priors

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
    # Ensure range covers typical scores e.g. -2 to 5
    full_score_range = list(range(min(score_values), max(score_values) + 1))
    
    cluster_distributions: Dict[int, pd.Series] = {}
    for cluster_id, group in assignment.groupby("cluster"):
        members = group["TeamID"].tolist()
        subset = ends[ends["TeamID"].isin(members)]
        dist = subset["Result"].value_counts().reindex(full_score_range, fill_value=0)
        cluster_distributions[cluster_id] = dist / dist.sum()
    return cluster_distributions, full_score_range


def evaluate_defenses(
    cluster_distributions: Dict[int, pd.Series], 
    score_values: List[int],
    options: List[DefenseOption],
    objective: str = "expected_value"
) -> pd.DataFrame:
    if not cluster_distributions:
        return pd.DataFrame()

    records = []
    for cluster_id, dist in cluster_distributions.items():
        opponent_distribution = dist.values
        
        # Calculate baseline EV
        baseline_ev = np.dot(opponent_distribution, np.array(score_values))
        
        # Run Optimizer
        best_option = minimax_defense(options, opponent_distribution, score_values, objective)
        
        if best_option is None:
            # Should not happen with fixed logic, but safe fallback
            continue
            
        # Calculate values for all options for reporting
        option_values = {}
        for option in options:
            # Re-use logic from optimizer to get the value for this objective
            is_distribution = np.isclose(np.sum(option.score_impact), 1.0)
            if is_distribution:
                final_dist = np.array(option.score_impact)
                if objective == "expected_value":
                    val = np.dot(final_dist, score_values)
                elif objective == "minimize_big_end":
                    big_score_indices = [i for i, s in enumerate(score_values) if s >= 3]
                    val = np.sum(final_dist[big_score_indices])
                elif objective == "maximize_steal":
                    steal_indices = [i for i, s in enumerate(score_values) if s < 0]
                    val = np.sum(final_dist[steal_indices])
            else:
                # Legacy fallback
                val = np.dot(opponent_distribution, option.score_impact) * option.success_prob
            
            option_values[option.name] = val

        records.append(
            {
                "cluster": cluster_id,
                "objective": objective,
                "baseline_expected_points": baseline_ev,
                "recommended_option": best_option.name,
                "best_value": option_values[best_option.name],
                **option_values,
            }
        )
    return pd.DataFrame(records)


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


def plot_score_distributions(options: List[DefenseOption], output_path: Path) -> None:
    """Violin plot of score distributions for each strategy."""
    data = []
    for opt in options:
        # Reconstruct samples from probability distribution
        # We assume range -2 to 6 for visualization
        score_range = list(range(-2, 7))
        # Pad or trim score_impact to match range length if needed
        # But score_impact is likely already aligned to a range.
        # For simplicity, let's just use the indices of the list as offsets from min_score
        # Assuming score_impact corresponds to [min_score, ..., max_score]
        
        # Actually, let's just simulate samples based on the probs
        probs = np.array(opt.score_impact)
        if probs.sum() == 0: continue
        probs = probs / probs.sum()
        
        # Assume standard range start if not provided. 
        # In compute_empirical_priors we used full_score_range.
        # Let's assume it starts at -2 (typical for curling power play)
        start_score = -2
        
        samples = np.random.choice(
            range(start_score, start_score + len(probs)), 
            size=1000, 
            p=probs
        )
        
        for s in samples:
            data.append({"Strategy": opt.name, "Score": s})
            
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="Strategy", y="Score", inner="quartile", palette="muted")
    plt.title("Score Distribution by Defensive Strategy")
    plt.ylabel("Opponent Score (Lower is Better)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', transparent=False)
    plt.close()

from projects.counter_strategy.src.execution_analysis import analyze_execution_sensitivity, plot_execution_risk

def main() -> None:
    print("Loading data and clustering opponents...")
    feature_frame = build_feature_table()
    model = fit_opponent_clusters(feature_frame)
    assignment = assign_clusters(feature_frame, model)

    ends = load_powerplay_ends()
    cluster_distributions, score_values = compute_cluster_distributions(ends, assignment)
    
    print("Computing empirical priors from historical shots...")
    try:
        options, prior_score_range = compute_empirical_priors()
        if not options:
             print("Insufficient data for empirical priors. Using defaults.")
             options = get_default_priors(score_values)
        else:
             aligned_options = []
             for opt in options:
                 prior_series = pd.Series(opt.score_impact, index=prior_score_range)
                 aligned_series = prior_series.reindex(score_values, fill_value=0)
                 if aligned_series.sum() > 0:
                     aligned_series = aligned_series / aligned_series.sum()
                 
                 aligned_options.append(
                     DefenseOption(name=opt.name, success_prob=opt.success_prob, score_impact=aligned_series.values.tolist())
                 )
             options = aligned_options
             
    except Exception as e:
        print(f"Error computing empirical priors: {e}. Using defaults.")
        options = get_default_priors(score_values)

    print(f"Evaluated {len(options)} defensive strategies: {[o.name for o in options]}")
    
    # --- New Visualization ---
    plot_score_distributions(options, OUTPUT_DIR / "score_distributions.png")
    # -------------------------
    
    # --- Execution Analysis ---
    print("Running execution sensitivity analysis...")
    exec_stats, exec_profiles = analyze_execution_sensitivity()
    if not exec_stats.empty:
        exec_stats.to_csv(OUTPUT_DIR / "execution_sensitivity.csv", index=False)
        plot_execution_risk(exec_profiles, OUTPUT_DIR / "execution_risk_scatter.png")
        print("Execution analysis generated.")
    # --------------------------

    # Evaluate for different objectives
    objectives = ["expected_value", "minimize_big_end", "maximize_steal"]
    all_results = []
    
    for obj in objectives:
        print(f"Running optimization for objective: {obj}")
        res = evaluate_defenses(cluster_distributions, score_values, options, objective=obj)
        all_results.append(res)
        
    full_summary = pd.concat(all_results)
    full_summary.to_csv(OUTPUT_DIR / "defense_summary.csv", index=False)

    # Generate Risk Playbook
    risk_playbook = full_summary.pivot(index="cluster", columns="objective", values="recommended_option")
    
    teams = pd.read_csv(PROJECT_ROOT / "Teams.csv")[["CompetitionID", "TeamID", "Name"]]
    team_lookup = teams.sort_values("CompetitionID").drop_duplicates("TeamID")
    
    top_teams = (
        assignment.sort_values("usage_count", ascending=False)
        .groupby("cluster")
        .head(3)
    )
    
    playbook_export = top_teams.merge(risk_playbook, on="cluster", how="left")
    
    available_objectives = [col for col in ["expected_value", "minimize_big_end", "maximize_steal"] if col in playbook_export.columns]
    
    playbook_export = playbook_export[["cluster", "TeamID"] + available_objectives]
    playbook_export = playbook_export.merge(team_lookup[["TeamID", "Name"]], on="TeamID", how="left")
    
    rename_map = {
        "expected_value": "Standard (Best EV)",
        "minimize_big_end": "Conservative (Avoid Big End)",
        "maximize_steal": "Aggressive (Need Steal)"
    }
    playbook_export = playbook_export.rename(columns=rename_map)
    
    playbook_export.to_csv(OUTPUT_DIR / "risk_playbook.csv", index=False)
    
    enriched = assignment.merge(team_lookup[["TeamID", "Name"]], on="TeamID", how="left")
    enriched.to_csv(OUTPUT_DIR / "cluster_assignments.csv", index=False)
    plot_cluster_scatter(enriched)

    print("Counter-strategy outputs generated at", OUTPUT_DIR)


if __name__ == "__main__":
    main()

