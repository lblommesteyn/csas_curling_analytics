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


def compute_empirical_priors(min_samples: int = 10) -> Tuple[List[DefenseOption], List[int]]:
    ends = load_csv("ends", usecols=["CompetitionID", "SessionID", "GameID", "EndID", "TeamID", "PowerPlay", "Result"])
    stones = load_csv("stones", usecols=["CompetitionID", "SessionID", "GameID", "EndID", "ShotID", "Task"])

    processed_ends = []
    for key, group in ends.groupby(["CompetitionID", "SessionID", "GameID", "EndID"]):
        if len(group) != 2:
            continue
        
        group["PowerPlay"] = group["PowerPlay"].fillna(0)
        hammer_rows = group[group["PowerPlay"] > 0]
        defensive_rows = group[group["PowerPlay"] == 0]

        if hammer_rows.empty or defensive_rows.empty:
            continue

        hammer_result = hammer_rows["Result"].iloc[0]
        defensive_result = defensive_rows["Result"].iloc[0]
        
        signed_result = 0
        if hammer_result > 0:
            signed_result = hammer_result
        elif defensive_result > 0:
            signed_result = -defensive_result
        
        processed_ends.append(list(key) + [signed_result])

    if not processed_ends:
        return [], []

    end_outcomes = pd.DataFrame(processed_ends, columns=["CompetitionID", "SessionID", "GameID", "EndID", "SignedResult"])

    stones["ShotID"] = pd.to_numeric(stones["ShotID"], errors="coerce")
    first_shots = stones.sort_values("ShotID").groupby(["CompetitionID", "SessionID", "GameID", "EndID"]).head(1).copy()
    
    merged = end_outcomes.merge(
        first_shots[["CompetitionID", "SessionID", "GameID", "EndID", "Task"]],
        on=["CompetitionID", "SessionID", "GameID", "EndID"],
        how="inner"
    )
    
    merged["Strategy"] = merged["Task"].map(TASK_MAP)
    merged = merged.dropna(subset=["Strategy"])
    
    options = []
    all_strategies = set(TASK_MAP.values())
    
    if merged.empty:
        score_range = list(range(-2, 6))
    else:
        score_range = sorted(merged["SignedResult"].unique().astype(int))
    
    full_score_range = list(range(min(score_range), max(score_range) + 1))
    
    strategy_groups = merged.groupby("Strategy")
    
    for strategy in all_strategies:
        if strategy not in strategy_groups.groups or len(strategy_groups.get_group(strategy)) < min_samples:
            default_option = get_default_priors(strategy, full_score_range)
            options.append(default_option)
            continue
            
        group = strategy_groups.get_group(strategy)
        counts = group["SignedResult"].value_counts().reindex(full_score_range, fill_value=0)
        probs = counts / counts.sum()
        
        options.append(
            DefenseOption(
                name=strategy,
                success_prob=1.0,
                score_impact=probs.values.tolist()
            )
        )
        
    return options, full_score_range

def get_default_priors(strategy_name: str, score_values: List[int]) -> DefenseOption:
    base_dist = np.zeros(len(score_values))
    score_map = {score: i for i, score in enumerate(score_values)}

    if strategy_name == "Guard Wall":
        priors = {0: 0.4, 1: 0.3, 2: 0.2, -1: 0.1}
    elif strategy_name == "Freeze Tap":
        priors = {0: 0.5, 1: 0.2, 2: 0.1, -1: 0.2}
    elif strategy_name == "Runback Pressure":
        priors = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.1, -1: 0.2, -2: 0.1}
    else:
        priors = {0: 0.6, 1: 0.25, -1: 0.15}

    for score, prob in priors.items():
        if score in score_map:
            base_dist[score_map[score]] = prob
    
    if base_dist.sum() > 0:
        base_dist = base_dist / base_dist.sum()
    else:
        base_dist[score_map.get(0, 0)] = 1.0

    return DefenseOption(name=strategy_name, success_prob=1.0, score_impact=base_dist.tolist())
