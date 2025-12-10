"""
Defensive minimax optimization for countering power plays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


@dataclass
class DefenseOption:
    name: str
    success_prob: float
    score_impact: List[float]  # Can be impact values OR probability distribution


def minimax_defense(
    options: List[DefenseOption], 
    opponent_distribution: np.ndarray, 
    score_values: List[int],
    objective: str = "expected_value"
) -> DefenseOption:
    """
    Select defense based on objective.
    
    Args:
        options: List of defensive strategies.
        opponent_distribution: Probability of each score outcome (from clustering).
        score_values: The actual score values corresponding to the distribution indices.
        objective: "expected_value", "minimize_big_end", or "maximize_steal".
    """

    best_option = None
    best_value = np.inf if objective != "maximize_steal" else -np.inf

    for option in options:
        # Check if score_impact is a probability distribution (sums to ~1) or impact values
        is_distribution = np.isclose(np.sum(option.score_impact), 1.0)
        
        if is_distribution:
            # If it's a distribution, we convolve or mix it with opponent tendencies
            # For simplicity in this version, we assume the empirical prior *is* the resulting distribution
            # when this defense is played, potentially weighted by the opponent's general skill.
            # A simple model: The defense dictates the shape, the opponent shifts the mean.
            # But since we have direct empirical outcomes for these shots, let's trust the data directly.
            final_dist = np.array(option.score_impact)
        else:
            # Legacy mode: score_impact is a value shift vector
            # This path is kept for backward compatibility or if we fall back to hardcoded priors
            # We need to construct a distribution from the expected values, which is hard.
            # Instead, we calculate the scalar expected value and treat it as a single point.
            # This breaks risk analysis, so we only support EV here.
            expected_score = np.dot(opponent_distribution, option.score_impact) * option.success_prob
            
            # For legacy options, we can only optimize for Expected Value.
            # Even if objective is "maximize_steal", we can't calculate steal prob.
            # So we fallback to minimizing expected score (best generic defense).
            val = expected_score
            
            # Always minimize EV for legacy
            # We compare against best_value. 
            # If objective is maximize_steal, best_value starts at -inf.
            # We need to be careful.
            # Let's just ignore the specific objective for legacy and always pick best EV.
            
            # However, we need to update best_option.
            # If we are mixing legacy and distribution options, this is messy.
            # Assuming all options are of the same type.
            
            # If objective is maximize_steal, we want MAX value.
            # But here 'val' is expected score (lower is better).
            # So we should convert to a utility score?
            # Let's just stick to minimizing expected score for legacy.
            
            # If we are in "maximize_steal" mode, best_value is -inf (seeking max).
            # But 'val' is expected score (minimize).
            # This is incompatible.
            
            # FIX: If legacy, we treat "maximize_steal" and "minimize_big_end" as "expected_value".
            # And we need to reset best_value if it was initialized for maximization.
            
            if best_value == -np.inf and objective == "maximize_steal":
                 best_value = np.inf # Reset to minimization mode for fallback
            
            if val < best_value:
                best_value = val
                best_option = option
            continue

        # Risk-Aware Optimization using Distributions
        if objective == "expected_value":
            # Minimize Expected Score
            val = np.dot(final_dist, score_values)
            if val < best_value:
                best_value = val
                best_option = option
                
        elif objective == "minimize_big_end":
            # Minimize probability of Score >= 3
            # Assuming score_values are sorted
            big_score_indices = [i for i, s in enumerate(score_values) if s >= 3]
            prob_big = np.sum(final_dist[big_score_indices])
            if prob_big < best_value:
                best_value = prob_big
                best_option = option
                
        elif objective == "maximize_steal":
            # Maximize probability of Score < 0
            steal_indices = [i for i, s in enumerate(score_values) if s < 0]
            prob_steal = np.sum(final_dist[steal_indices])
            if prob_steal > best_value:
                best_value = prob_steal
                best_option = option

    return best_option
