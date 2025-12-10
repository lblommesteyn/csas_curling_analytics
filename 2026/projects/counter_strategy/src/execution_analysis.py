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
    print(f"DEBUG: Power Play Ends: {len(pp_ends)}")
    
    # 3. Join with Stones to get Shot 1 (Defensive Shot)
    stones["ShotID"] = pd.to_numeric(stones["ShotID"], errors="coerce")
    first_shots = stones.sort_values("ShotID").groupby(["CompetitionID", "SessionID", "GameID", "EndID"]).head(1).copy()
    print(f"DEBUG: First Shots (Min ShotID): {len(first_shots)}")
    
    print("DEBUG: Ends Keys Dtypes:")
    print(pp_ends[["CompetitionID", "SessionID", "GameID", "EndID"]].dtypes)
    print("DEBUG: Stones Keys Dtypes:")
    print(first_shots[["CompetitionID", "SessionID", "GameID", "EndID"]].dtypes)
    
    print("DEBUG: Ends Keys Head:")
    print(pp_ends[["CompetitionID", "SessionID", "GameID", "EndID"]].head())
    print("DEBUG: Stones Keys Head:")
    print(first_shots[["CompetitionID", "SessionID", "GameID", "EndID"]].head())
    
    merged = pp_ends.merge(
        first_shots[["CompetitionID", "SessionID", "GameID", "EndID", "Task", "Points"]],
        on=["CompetitionID", "SessionID", "GameID", "EndID"],
        how="inner"
    )
    print(f"DEBUG: Merged Data: {len(merged)}")
    
    # 4. Map Tasks to Strategies
    merged["Strategy"] = merged["Task"].map(TASK_MAP)
    merged = merged.dropna(subset=["Strategy", "Points"])
    print(f"DEBUG: Data after mapping and dropna: {len(merged)}")
    if not merged.empty:
        print(f"DEBUG: Points distribution: {merged['Points'].value_counts()}")
    
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
        
    return pd.DataFrame(stats_records), profiles

def plot_execution_risk(profiles: List[ExecutionProfile], output_path: Path) -> None:
    """Generate a scatter plot of Potential vs Risk."""
    if not profiles:
        return
        
    data = []
    for p in profiles:
        data.append({
            "Strategy": p.strategy,
            "Max Potential (High Quality EV)": p.high_quality_ev,
            "Risk of Ruin (Low Quality EV)": p.low_quality_ev,
            "Size": p.sample_size_high + p.sample_size_low
        })
        
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="Max Potential (High Quality EV)",
        y="Risk of Ruin (Low Quality EV)",
        hue="Strategy",
        style="Strategy",
        s=200,
        palette="deep"
    )
    
    # Add diagonal line (where High EV == Low EV, impossible usually)
    # But more useful: Add a reference for "Safe" vs "Risky"
    # Lower X and Lower Y is best.
    
    plt.title("Execution Sensitivity: Reward vs. Risk")
    plt.xlabel("Expected Opponent Score (Good Shot)")
    plt.ylabel("Expected Opponent Score (Bad Shot)")
    
    # Annotate points
    for i, row in df.iterrows():
        plt.text(
            row["Max Potential (High Quality EV)"] + 0.02, 
            row["Risk of Ruin (Low Quality EV)"], 
            row["Strategy"],
            fontsize=9
        )
        
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
