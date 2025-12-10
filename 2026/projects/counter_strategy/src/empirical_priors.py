"""
Empirical priors for defensive shots based on historical data.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_csv
from projects.counter_strategy.src.defense_optimizer import DefenseOption


# Map Task IDs to our high-level strategies
# Expanded based on Stones.csv distribution
TASK_MAP = {
    0: "Draw / Guard",
    1: "Draw / Guard",
    2: "Guard Wall",
    3: "Draw / Guard",
    4: "Guard Wall",
    5: "Freeze Tap",
    6: "Runback Pressure",
    7: "Runback Pressure",
    8: "Runback Pressure",
    9: "Runback Pressure",
    10: "Runback Pressure",
    11: "Runback Pressure",
    12: "Runback Pressure"
}


def compute_empirical_priors(min_samples: int = 10) -> List[DefenseOption]:
    """
    Calculate success rates and score distributions from historical data.
    
    Returns a list of DefenseOption objects populated with real-world data.
    If data is insufficient for a strategy, falls back to hardcoded defaults.
    """
    
    # 1. Load Data
    ends = load_csv("ends", usecols=["CompetitionID", "SessionID", "GameID", "EndID", "PowerPlay", "Result"])
    stones = load_csv("stones", usecols=["CompetitionID", "SessionID", "GameID", "EndID", "ShotID", "Task"])
    
    # 2. Filter for Power Play Ends
    pp_ends = ends[ends["PowerPlay"].fillna(0) > 0].copy()
    
    # 3. Join with Stones to get the first shot of the end
    # We select the stone with the minimum ShotID for each end.
    stones["ShotID"] = pd.to_numeric(stones["ShotID"], errors="coerce")
    first_shots = stones.sort_values("ShotID").groupby(["CompetitionID", "SessionID", "GameID", "EndID"]).head(1).copy()
    
    merged = pp_ends.merge(
        first_shots[["CompetitionID", "SessionID", "GameID", "EndID", "Task"]],
        on=["CompetitionID", "SessionID", "GameID", "EndID"],
        how="inner"
    )
    
    # 4. Map Tasks to Strategies
    merged["Strategy"] = merged["Task"].map(TASK_MAP)
    merged = merged.dropna(subset=["Strategy"])
    
    # 5. Calculate Stats per Strategy
    options = []
    score_range = sorted(pp_ends["Result"].unique().astype(int))
    # Ensure range covers typical scores e.g. -2 to 5
    full_score_range = list(range(min(score_range), max(score_range) + 1))
    
    for strategy, group in merged.groupby("Strategy"):
        n_samples = len(group)
        if n_samples < min_samples:
            continue
            
        # Calculate score distribution (probability of each score)
        counts = group["Result"].value_counts().reindex(full_score_range, fill_value=0)
        probs = counts / n_samples
        
        # Calculate "Success Probability"
        # We define "Success" loosely as "Preventing a big score (>=2)" 
        # This is a simplification for the optimizer's interface, 
        # but the real power comes from the score_impact distribution.
        # For the optimizer, we set success_prob = 1.0 and encode the full distribution 
        # into the score_impact vector. This allows the optimizer to use the 
        # exact empirical distribution rather than a binary success/fail model.
        
        options.append(
            DefenseOption(
                name=strategy,
                success_prob=1.0, # Use full distribution in score_impact
                score_impact=probs.values.tolist() # This is now a probability distribution, not a single outcome
            )
        )
        
    return options, full_score_range

def get_default_priors(score_values: List[int]) -> List[DefenseOption]:
    """Fallback to hardcoded priors if data is missing."""
    # Replicates the logic from original run_analysis.py
    guard_impact = [max(val - 1, -1) if val >= 3 else val for val in score_values]
    freeze_impact = [val - 1 if val >= 2 else val for val in score_values]
    runback_impact = [val - 0.5 if val <= 0 else val + 0.3 for val in score_values]

    return [
        DefenseOption(name="Guard Wall", success_prob=0.75, score_impact=guard_impact),
        DefenseOption(name="Freeze Tap", success_prob=0.65, score_impact=freeze_impact),
        DefenseOption(name="Runback Pressure", success_prob=0.55, score_impact=runback_impact),
    ]
