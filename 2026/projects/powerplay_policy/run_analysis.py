"""
Run the dynamic power-play policy analysis and produce visualisations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_csv
from projects.powerplay_policy.src.feature_engineering import build_team_end_frame
from projects.powerplay_policy.src.policy_solver import value_iteration, MDPConfig

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def discretise_score(diff: int) -> str:
    if diff <= -4:
        return "<= -4"
    if diff >= 4:
        return ">= +4"
    return f"{diff:+d}"


def build_state_table() -> pd.DataFrame:
    states = build_team_end_frame()

    states["power_play_action"] = states["power_play_flag"].astype(int)
    states["score_bucket"] = states["score_diff_before"].apply(discretise_score)
    states["state_id"] = (
        states["EndID"].astype(str)
        + "|"
        + states["score_bucket"]
        + "|"
        + states["hammer"].astype(int).astype(str)
        + "|"
        + states["power_play_available"].astype(int).astype(str)
    )

    states = states.sort_values(["CompetitionID", "SessionID", "GameID", "TeamID", "EndID"])
    states["next_state_id"] = states.groupby(
        ["CompetitionID", "SessionID", "GameID", "TeamID"]
    )["state_id"].shift(-1)
    states["next_state_id"] = states["next_state_id"].fillna("terminal")

    states["reward"] = states["score_diff_after"] - states["score_diff_before"]

    final_scores = load_csv("games")[["CompetitionID", "GameID", "TeamID1", "TeamID2", "ResultStr1", "ResultStr2"]]
    # Expand final results to team perspective.
    team_rows = []
    for _, row in final_scores.iterrows():
        team_rows.append(
            {
                "CompetitionID": row["CompetitionID"],
                "GameID": row["GameID"],
                "TeamID": row["TeamID1"],
                "final_score_diff": row["ResultStr1"] - row["ResultStr2"],
            }
        )
        team_rows.append(
            {
                "CompetitionID": row["CompetitionID"],
                "GameID": row["GameID"],
                "TeamID": row["TeamID2"],
                "final_score_diff": row["ResultStr2"] - row["ResultStr1"],
            }
        )

    finals = pd.DataFrame(team_rows)
    states = states.merge(
        finals,
        on=["CompetitionID", "GameID", "TeamID"],
        how="left",
    )

    # Estimate win probability proxy per state-action by looking at average final diff.
    states["win_proxy"] = (states["final_score_diff"] > 0).astype(int)
    return states


def aggregate_transitions(states: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], float], Dict[int, str]]:
    unique_states = sorted(set(states["state_id"].unique().tolist() + ["terminal"]))
    state_to_idx = {state: idx for idx, state in enumerate(unique_states)}
    actions = np.array([0, 1], dtype=int)  # 0 = hold, 1 = use

    transitions: Dict[Tuple[int, int], np.ndarray] = {}
    rewards: Dict[Tuple[int, int], float] = {}

    for (state, action), group in states.groupby(["state_id", "power_play_action"]):
        state_idx = state_to_idx[state]
        reward = group["reward"].mean()
        rewards[(state_idx, action)] = reward

        next_counts = group["next_state_id"].value_counts()
        probs = np.zeros(len(unique_states))
        for next_state, count in next_counts.items():
            probs[state_to_idx[next_state]] = count / next_counts.sum()
        transitions[(state_idx, action)] = probs

    # Absorbing terminal state.
    terminal_idx = state_to_idx["terminal"]
    transitions[(terminal_idx, 0)] = np.eye(len(unique_states))[terminal_idx]
    transitions[(terminal_idx, 1)] = np.eye(len(unique_states))[terminal_idx]
    rewards[(terminal_idx, 0)] = 0.0
    rewards[(terminal_idx, 1)] = 0.0

    idx_to_state = {idx: state for state, idx in state_to_idx.items()}
    return np.arange(len(unique_states)), actions, transitions, rewards, idx_to_state


def compute_advantage_table(state_indices: np.ndarray, actions: np.ndarray, transitions: Dict[Tuple[int, int], np.ndarray], rewards: Dict[Tuple[int, int], float], values: np.ndarray, idx_to_state: Dict[int, str]) -> pd.DataFrame:
    records = []
    for state_idx in state_indices:
        state_label = idx_to_state[state_idx]
        if state_label == "terminal":
            continue

        q_values = {}
        for action in actions:
            probs = transitions.get((state_idx, action))
            if probs is None:
                continue
            q_values[action] = rewards.get((state_idx, action), 0.0) + 0.97 * np.dot(probs, values)

        if not q_values:
            continue

        best_action = max(q_values.items(), key=lambda kv: kv[1])[0]
        advantage = q_values.get(1, np.nan) - q_values.get(0, np.nan)

        end_id, score_bucket, hammer_flag, availability_flag = state_label.split("|")
        records.append(
            {
                "state_id": state_label,
                "recommended_action": best_action,
                "advantage_use_vs_hold": advantage,
                "EndID": int(end_id),
                "score_bucket": score_bucket,
                "hammer": "Hammer" if hammer_flag == "1" else "No Hammer",
                "power_play_available": "Available" if availability_flag == "1" else "Used",
            }
        )

    return pd.DataFrame(records)


def plot_policy_heatmap(table: pd.DataFrame) -> None:
    pivot = (
        table[table["power_play_available"] == "Available"]
        .pivot_table(index="score_bucket", columns="EndID", values="recommended_action")
        .reindex(index=["<= -4", "-3", "-2", "-1", "+0", "+1", "+2", "+3", ">= +4"], fill_value=np.nan)
    )

    plt.figure(figsize=(10, 5))
    sns.heatmap(pivot, cmap="coolwarm", cbar_kws={"label": "Recommended action (1=use)"}, vmin=0, vmax=1)
    plt.title("MDP Policy: When to Deploy the Power Play")
    plt.xlabel("End Number")
    plt.ylabel("Score Differential (before end)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "policy_heatmap.png", dpi=200)
    plt.close()


def plot_advantage(table: pd.DataFrame) -> None:
    subset = table[
        (table["power_play_available"] == "Available") & table["advantage_use_vs_hold"].notna()
    ]
    plt.figure(figsize=(10, 5))
    sns.pointplot(
        data=subset,
        x="EndID",
        y="advantage_use_vs_hold",
        hue="hammer",
        dodge=True,
        markers=["o", "s"],
        linestyles="-",
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Expected score gain: use minus hold")
    plt.title("Power Play Swing by End & Hammer")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "advantage_by_end.png", dpi=200)
    plt.close()


def plot_policy_gap(states_df: pd.DataFrame, policy_table: pd.DataFrame) -> None:
    available_states = policy_table[policy_table["power_play_available"] == "Available"]

    empirical = (
        states_df[states_df["power_play_available"]]
        .groupby(["EndID", "score_bucket"])["power_play_action"]
        .mean()
        .reset_index()
        .rename(columns={"power_play_action": "empirical_rate"})
    )

    recommended = available_states[["EndID", "score_bucket", "recommended_action"]].rename(
        columns={"recommended_action": "recommended_rate"}
    )

    combined = empirical.merge(recommended, on=["EndID", "score_bucket"], how="inner")
    combined["gap"] = combined["recommended_rate"] - combined["empirical_rate"]

    pivot = combined.pivot_table(index="score_bucket", columns="EndID", values="gap")
    pivot = pivot.reindex(
        ["<= -4", "-3", "-2", "-1", "+0", "+1", "+2", "+3", ">= +4"],
        axis=0,
    )

    plt.figure(figsize=(10, 5))
    sns.heatmap(
        pivot,
        cmap="BrBG",
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Recommended - actual usage"},
    )
    plt.title("Power Play Usage Gap (Positive = underused)")
    plt.xlabel("End Number")
    plt.ylabel("Score Differential (before end)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "usage_gap_heatmap.png", dpi=200)
    plt.close()


def plot_top_swing_states(policy_table: pd.DataFrame) -> None:
    subset = policy_table[
        (policy_table["power_play_available"] == "Available")
        & policy_table["advantage_use_vs_hold"].notna()
    ].copy()
    subset["label"] = (
        "End "
        + subset["EndID"].astype(str)
        + " | score "
        + subset["score_bucket"]
        + " | "
        + subset["hammer"]
    )

    top_states = subset.nlargest(8, "advantage_use_vs_hold")

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=top_states,
        x="advantage_use_vs_hold",
        y="label",
        hue="hammer",
        dodge=False,
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Expected score swing from using power play")
    plt.ylabel("Game situation")
    plt.title("Highest-Impact Power Play Windows")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_swing_states.png", dpi=200)
    plt.close()


def fit_win_probability_model(states: pd.DataFrame) -> tuple[LogisticRegression, int]:
    """Fit logistic regression to map scoreboard states to win probability."""

    total_ends = int(states["EndID"].max())
    features = pd.DataFrame(
        {
            "score_diff_before": states["score_diff_before"],
            "hammer": states["hammer"].astype(int),
            "ends_remaining": total_ends - states["EndID"] + 1,
            "power_play_available": states["power_play_available"].astype(int),
        }
    )
    labels = (states["final_score_diff"] > 0).astype(int)

    model = LogisticRegression(max_iter=500)
    model.fit(features, labels)
    return model, total_ends


def predict_win_probability(
    model: LogisticRegression,
    score_diff: float,
    hammer: int,
    ends_remaining: int,
    power_play_available: int,
) -> float:
    features = pd.DataFrame(
        {
            "score_diff_before": [score_diff],
            "hammer": [hammer],
            "ends_remaining": [ends_remaining],
            "power_play_available": [power_play_available],
        }
    )
    return float(model.predict_proba(features)[0, 1])


def annotate_with_win_probabilities(
    policy_table: pd.DataFrame,
    states_df: pd.DataFrame,
    model: LogisticRegression,
    total_ends: int,
) -> pd.DataFrame:
    state_meta = (
        states_df.groupby("state_id")
        .agg(
            avg_score_before=("score_diff_before", "mean"),
            hammer_flag=("hammer", lambda x: int(x.iloc[0])),
            end_id=("EndID", lambda x: int(x.iloc[0])),
            availability=("power_play_available", lambda x: int(x.iloc[0])),
        )
        .reset_index()
    )

    enriched = policy_table.merge(state_meta, on="state_id", how="left")

    win_hold = []
    win_use = []
    win_gain = []

    for _, row in enriched.iterrows():
        if pd.isna(row["advantage_use_vs_hold"]):
            win_hold.append(np.nan)
            win_use.append(np.nan)
            win_gain.append(np.nan)
            continue

        ends_remaining = total_ends - row["EndID"] + 1
        score_before = row["avg_score_before"]
        score_after = score_before + row["advantage_use_vs_hold"]
        hammer = 1 if row["hammer"] == "Hammer" else 0
        availability_hold = row["availability"]
        availability_use = 0

        p_hold = predict_win_probability(model, score_before, hammer, ends_remaining, availability_hold)
        p_use = predict_win_probability(model, score_after, hammer, ends_remaining, availability_use)

        win_hold.append(p_hold)
        win_use.append(p_use)
        win_gain.append(p_use - p_hold)

    enriched["win_prob_hold"] = win_hold
    enriched["win_prob_use"] = win_use
    enriched["win_prob_gain"] = win_gain
    enriched["win_prob_gain_pct"] = enriched["win_prob_gain"] * 100
    return enriched.drop(columns=["hammer_flag", "end_id", "availability"])


def plot_top_win_states(policy_table: pd.DataFrame) -> None:
    subset = policy_table[
        (policy_table["power_play_available"] == "Available")
        & policy_table["win_prob_gain"].notna()
    ].copy()
    subset["label"] = (
        "End "
        + subset["EndID"].astype(str)
        + " | score "
        + subset["score_bucket"]
        + " | "
        + subset["hammer"]
    )

    top_states = subset.nlargest(8, "win_prob_gain")

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=top_states,
        x="win_prob_gain",
        y="label",
        hue="hammer",
        dodge=False,
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Win probability lift from using power play")
    plt.ylabel("Game situation")
    plt.title("Largest Win-Probability Gains")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_win_prob_states.png", dpi=200)
    plt.close()


def main() -> None:
    states_df = build_state_table()
    state_indices, actions, transitions, rewards, idx_to_state = aggregate_transitions(states_df)

    values, policy = value_iteration(state_indices, actions, transitions, rewards, MDPConfig(discount=0.97))
    policy_table = compute_advantage_table(state_indices, actions, transitions, rewards, values, idx_to_state)
    win_model, total_ends = fit_win_probability_model(states_df)
    policy_table = annotate_with_win_probabilities(policy_table, states_df, win_model, total_ends)

    plot_policy_heatmap(policy_table)
    plot_advantage(policy_table)
    plot_policy_gap(states_df, policy_table)
    plot_top_swing_states(policy_table)
    plot_top_win_states(policy_table)

    policy_table.to_csv(OUTPUT_DIR / "policy_table.csv", index=False)
    states_df.to_csv(OUTPUT_DIR / "state_dataset.csv", index=False)

    print("Generated power play policy outputs at", OUTPUT_DIR)


if __name__ == "__main__":
    main()
