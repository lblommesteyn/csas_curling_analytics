"""
Execution sensitivity analysis for defensive shots.
Quantifies the "Forgiveness" of strategies by comparing High Quality vs Low Quality execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_csv
from projects.counter_strategy.src.empirical_priors import TASK_MAP

@dataclass
class ExecutionProfile:
    strategy: str
    high_quality_ev: float
    low_quality_ev: float
    forgiveness: float  # Difference or ratio
    sample_size_high: int
    sample_size_low: int

def analyze_execution_sensitivity(min_samples: int = 1) -> Tuple[pd.DataFrame, List[ExecutionProfile]]:
    """
    Analyze how shot quality (Points) affects the outcome of defensive strategies.
    
    Returns:
        - DataFrame with raw stats
        - List of ExecutionProfile objects
    """
    
    # 1. Load Data
    ends = load_csv("ends", usecols=["CompetitionID", "SessionID", "GameID", "EndID", "PowerPlay", "Result"])
    stones = load_csv("stones", usecols=["CompetitionID", "SessionID", "GameID", "EndID", "ShotID", "Task", "Points"])
    
    # 2. Filter for Power Play Ends
    pp_ends = ends[ends["PowerPlay"].fillna(0) > 0].copy()
    
    # 3. Join with Stones to get Shot 1 (Defensive Shot)
    stones["ShotID"] = pd.to_numeric(stones["ShotID"], errors="coerce")
    first_shots = stones.sort_values("ShotID").groupby(["CompetitionID", "SessionID", "GameID", "EndID"]).head(1).copy()
    
    merged = pp_ends.merge(
        first_shots[["CompetitionID", "SessionID", "GameID", "EndID", "Task", "Points"]],
        on=["CompetitionID", "SessionID", "GameID", "EndID"],
        how="inner"
    )
    
    # 4. Map Tasks to Strategies
    merged["Strategy"] = merged["Task"].map(TASK_MAP)
    merged = merged.dropna(subset=["Strategy", "Points"])
    
    # 5. Define Quality Buckets
    # High Quality: 3 or 4 (Made shot)
    # Low Quality: 0, 1, 2 (Missed or partial miss)
    merged["Quality"] = merged["Points"].apply(lambda x: "High" if x >= 3 else "Low")
    
    # 6. Calculate Stats
    profiles = []
    stats_records = []
    
    for strategy, group in merged.groupby("Strategy"):
        high_group = group[group["Quality"] == "High"]
        low_group = group[group["Quality"] == "Low"]
        
        if len(high_group) < min_samples or len(low_group) < min_samples:
            continue
            
        high_ev = high_group["Result"].mean()
        low_ev = low_group["Result"].mean()
        
        # Forgiveness: How much worse is the result if you miss?
        # Result is "Points Scored by Opponent". Lower is better.
        # So Low EV will likely be higher (worse) than High EV.
        # Forgiveness = Low EV - High EV (The "Cost of a Miss")
        cost_of_miss = low_ev - high_ev
        
        profiles.append(
            ExecutionProfile(
                strategy=strategy,
                high_quality_ev=high_ev,
                low_quality_ev=low_ev,
                forgiveness=cost_of_miss,
                sample_size_high=len(high_group),
                sample_size_low=len(low_group)
            )
        )
        
        stats_records.append({
            "Strategy": strategy,
            "High_EV": high_ev,
            "Low_EV": low_ev,
            "Cost_of_Miss": cost_of_miss,
            "N_High": len(high_group),
            "N_Low": len(low_group)
        })
        
    return pd.DataFrame(stats_records), profiles, merged

def plot_slope_chart(profiles: List[ExecutionProfile], output_path: Path) -> None:
    """
    Generate a Slope Chart (Dumbbell Plot) showing the drop-off from Good to Bad execution.
    """
    if not profiles:
        return

    data = []
    for p in profiles:
        data.append({
            "Strategy": p.strategy,
            "Good Shot": p.high_quality_ev,
            "Bad Shot": p.low_quality_ev,
            "Cost": p.forgiveness
        })
    
    df = pd.DataFrame(data).sort_values("Good Shot")
    
    plt.figure(figsize=(10, 6))
    
    # Draw lines
    for i, row in df.iterrows():
        plt.plot([row["Good Shot"], row["Bad Shot"]], [row["Strategy"], row["Strategy"]], 
                 color="gray", alpha=0.5, zorder=1)
        
    # Draw points
    plt.scatter(df["Good Shot"], df["Strategy"], color="green", s=100, label="Good Shot (3-4)", zorder=2)
    plt.scatter(df["Bad Shot"], df["Strategy"], color="red", s=100, label="Bad Shot (0-2)", zorder=2)
    
    plt.title("The Cost of a Miss: Execution Sensitivity")
    plt.xlabel("Expected Opponent Score (Lower is Better)")
    plt.legend()
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_diverging_bars(profiles: List[ExecutionProfile], output_path: Path) -> None:
    """
    Generate a Diverging Bar Chart showing Risk vs Reward relative to average.
    """
    if not profiles:
        return
        
    data = []
    for p in profiles:
        avg_ev = (p.high_quality_ev + p.low_quality_ev) / 2
        data.append({
            "Strategy": p.strategy,
            "Reward": p.high_quality_ev - avg_ev, # Negative because lower score is better
            "Risk": p.low_quality_ev - avg_ev
        })
    
    df = pd.DataFrame(data).sort_values("Reward")
    
    plt.figure(figsize=(10, 6))
    
    # Since lower score is better, "Reward" is reducing the score (negative change)
    # "Risk" is increasing the score (positive change)
    
    # We want "Reward" (Green) to go Left, "Risk" (Red) to go Right
    # But currently Reward is negative (good) and Risk is positive (bad).
    # So a standard barh will work perfectly if 0 is center.
    
    plt.barh(df["Strategy"], df["Reward"], color="green", alpha=0.7, label="Benefit of Good Shot")
    plt.barh(df["Strategy"], df["Risk"], color="red", alpha=0.7, label="Penalty of Bad Shot")
    
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Risk/Reward Trade-off (Relative to Average)")
    plt.xlabel("Change in Expected Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_degradation_heatmap(merged_df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate a Heatmap showing Expected Score by Strategy and specific Shot Points (0-4).
    """
    if merged_df.empty:
        return
        
    # Calculate Mean Result by Strategy and Points
    pivot = merged_df.groupby(["Strategy", "Points"])["Result"].mean().unstack()
    
    # Sort by best result at Points=4
    if 4 in pivot.columns:
        pivot = pivot.sort_values(4)
        
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", cbar_kws={'label': 'Avg Opponent Score'})
    plt.title("Performance Degradation by Execution Quality")
    plt.xlabel("Shot Execution Rating (0-4)")
    plt.ylabel("Strategy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_forgiveness_quadrant(profiles: List[ExecutionProfile], output_path: Path) -> None:
    """
    Generate a Quadrant Plot: Max Reward vs Forgiveness.
    """
    if not profiles:
        return
        
    data = []
    for p in profiles:
        data.append({
            "Strategy": p.strategy,
            "Max Reward": -p.high_quality_ev, # Negate so higher is better
            "Forgiveness": 1 / (p.forgiveness + 0.01) # Inverse of cost
        })
        
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x="Max Reward", y="Forgiveness", s=200, hue="Strategy", style="Strategy")
    
    # Add quadrant lines
    plt.axvline(df["Max Reward"].mean(), color="gray", linestyle="--")
    plt.axhline(df["Forgiveness"].mean(), color="gray", linestyle="--")
    
    plt.title("Strategic Quadrants: Reward vs. Forgiveness")
    plt.xlabel("Max Reward (Negative Expected Score)")
    plt.ylabel("Forgiveness (Inverse of Miss Cost)")
    
    for i, row in df.iterrows():
        plt.text(row["Max Reward"]+0.01, row["Forgiveness"], row["Strategy"], fontsize=9)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_confidence_bands(merged_df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate a Line Plot with Confidence Bands showing score trend by rating.
    """
    if merged_df.empty:
        return
        
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=merged_df, x="Points", y="Result", hue="Strategy", style="Strategy", markers=True, dashes=False)
    
    plt.title("Execution Consistency Profile")
    plt.xlabel("Shot Execution Rating (0-4)")
    plt.ylabel("Average Opponent Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def generate_all_visualizations(profiles: List[ExecutionProfile], merged_df: pd.DataFrame, output_dir: Path) -> None:
    """Wrapper to generate all 5 visualizations."""
    plot_slope_chart(profiles, output_dir / "viz_slope_chart.png")
    plot_diverging_bars(profiles, output_dir / "viz_diverging_bars.png")
    plot_degradation_heatmap(merged_df, output_dir / "viz_heatmap.png")
    plot_forgiveness_quadrant(profiles, output_dir / "viz_quadrant.png")
    plot_confidence_bands(merged_df, output_dir / "viz_confidence_bands.png")
