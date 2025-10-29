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
    score_impact: List[int]


def minimax_defense(options: List[DefenseOption], opponent_distribution: np.ndarray) -> DefenseOption:
    """Select defense minimizing opponent expected score."""

    best_option = None
    best_value = np.inf

    for option in options:
        expected_score = np.dot(opponent_distribution, option.score_impact) * option.success_prob
        if expected_score < best_value:
            best_value = expected_score
            best_option = option

    return best_option
