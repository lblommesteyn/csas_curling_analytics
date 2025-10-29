"""
Feature engineering for the dynamic power-play policy project.

Builds per-team, per-end states capturing score, hammer, and power-play availability.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_csv  # noqa: E402


GROUP_COLS = ["CompetitionID", "SessionID", "GameID", "TeamID"]


def _attach_hammer_info(ends: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Annotate hammer possession for every team-end observation."""

    games = games[
        [
            "CompetitionID",
            "SessionID",
            "GameID",
            "TeamID1",
            "TeamID2",
            "LSFE",
        ]
    ]

    merged = ends.merge(games, on=["CompetitionID", "SessionID", "GameID"], how="left")
    merged = merged.sort_values(["CompetitionID", "SessionID", "GameID", "EndID"])

    def process_game(game_df: pd.DataFrame) -> pd.DataFrame:
        hammer_team = game_df["TeamID1"].iloc[0] if game_df["LSFE"].iloc[0] == 1 else game_df["TeamID2"].iloc[0]
        records = []

        for end_id, end_block in game_df.groupby("EndID", sort=True):
            end_block = end_block.copy()
            end_block["hammer"] = end_block["TeamID"] == hammer_team
            records.append(end_block)

            # Update hammer for next end.
            scoring_rows = end_block[end_block["Result"] > 0]
            if not scoring_rows.empty:
                scoring_team = scoring_rows["TeamID"].iloc[0]
                hammer_team = game_df["TeamID1"].iloc[0] if scoring_team != game_df["TeamID1"].iloc[0] else game_df["TeamID2"].iloc[0]

        return pd.concat(records, ignore_index=True)

    result = merged.groupby(["CompetitionID", "SessionID", "GameID"], group_keys=False).apply(process_game)
    return result


def build_team_end_frame() -> pd.DataFrame:
    """
    Produce one row per team per end with engineered features.

    Returns
    -------
    pd.DataFrame
        Contains base fields and engineered state components needed for the policy model.
    """

    ends = load_csv("ends")
    games = load_csv("games")

    ends["Result"] = ends["Result"].fillna(0).astype(int)
    ends = _attach_hammer_info(ends, games)

    ends = ends.sort_values(["CompetitionID", "SessionID", "GameID", "TeamID", "EndID"])

    ends["team_score_after"] = ends.groupby(GROUP_COLS)["Result"].cumsum()
    ends["team_score_before"] = ends["team_score_after"] - ends["Result"]

    opponent_scores = (
        ends[
            [
                "CompetitionID",
                "SessionID",
                "GameID",
                "EndID",
                "TeamID",
                "team_score_before",
                "team_score_after",
            ]
        ]
        .rename(
            columns={
                "TeamID": "OpponentTeamID",
                "team_score_before": "opp_score_before",
                "team_score_after": "opp_score_after",
            }
        )
    )

    ends = ends.merge(
        opponent_scores,
        on=["CompetitionID", "SessionID", "GameID", "EndID"],
        how="left",
    )
    ends = ends[ends["TeamID"] != ends["OpponentTeamID"]]

    ends["score_diff_before"] = ends["team_score_before"] - ends["opp_score_before"]
    ends["score_diff_after"] = ends["team_score_after"] - ends["opp_score_after"]

    ends["power_play_flag"] = ends["PowerPlay"].fillna(0).astype(int) > 0
    ends["power_play_used_to_date"] = (
        ends.groupby(GROUP_COLS)["power_play_flag"].cumsum().shift(fill_value=0)
    )
    ends["power_play_available"] = ends["power_play_used_to_date"] == 0

    ends["total_ends_in_game"] = ends.groupby(GROUP_COLS)["EndID"].transform("max")
    ends["ends_remaining"] = ends["total_ends_in_game"] - ends["EndID"] + 1

    ends["rocks_remaining"] = np.maximum(0, 6 - (ends["EndID"] - 1) * 2)

    return ends[
        [
            "CompetitionID",
            "SessionID",
            "GameID",
            "TeamID",
            "EndID",
            "Result",
            "hammer",
            "rocks_remaining",
            "ends_remaining",
            "score_diff_before",
            "score_diff_after",
            "power_play_flag",
            "power_play_available",
        ]
    ].reset_index(drop=True)
