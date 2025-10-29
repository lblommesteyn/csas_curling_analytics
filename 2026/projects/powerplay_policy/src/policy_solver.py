"""
Policy solvers for the power play MDP.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


@dataclass
class MDPConfig:
    discount: float = 0.99
    convergence_tol: float = 1e-6
    max_iterations: int = 10_000


def value_iteration(
    states: np.ndarray,
    actions: np.ndarray,
    transition_probs: Dict[Tuple[int, int], np.ndarray],
    rewards: Dict[Tuple[int, int], float],
    config: MDPConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run value iteration for a finite-state MDP.

    Parameters
    ----------
    states, actions:
        Enumerations of state and action ids.
    transition_probs:
        Mapping (state, action) -> probability vector over states.
    rewards:
        Mapping (state, action) -> expected immediate reward.
    config:
        Algorithm controls (discount, tolerance, iterations).

    Returns
    -------
    values, policy
    """

    if config is None:
        config = MDPConfig()

    values = np.zeros(len(states))
    policy = np.zeros(len(states), dtype=int)

    for iteration in range(config.max_iterations):
        delta = 0.0
        for s_idx, state in enumerate(states):
            action_returns = []
            for a_idx, action in enumerate(actions):
                probs = transition_probs.get((state, action))
                if probs is None:
                    continue

                expected_value = rewards.get((state, action), 0.0) + config.discount * np.dot(probs, values)
                action_returns.append((expected_value, a_idx))

            if not action_returns:
                continue

            best_value, best_action = max(action_returns, key=lambda item: item[0])
            delta = max(delta, abs(best_value - values[s_idx]))
            values[s_idx] = best_value
            policy[s_idx] = best_action

        if delta < config.convergence_tol:
            break

    return values, policy
