"""
Initial exploratory analysis for mixed doubles curling power plays.

The script reads the raw CSV files, computes descriptive statistics about how
often power plays are used, and examines the scoring context around them.
Outputs are written to the `outputs/` directory for easy inspection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import load_csv


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_ends_frame() -> pd.DataFrame:
    """Load the Ends.csv file and enrich it with helper columns."""

    ends = load_csv("ends")
    ends = ends.copy()

    ends["Result"] = ends["Result"].fillna(0).astype(int)
    ends["PowerPlayCode"] = ends["PowerPlay"]
    ends["PowerPlayUsed"] = ends["PowerPlayCode"].fillna(0).astype(int) > 0

    sort_cols = ["CompetitionID", "SessionID", "GameID", "TeamID", "EndID"]
    ends = ends.sort_values(sort_cols, kind="mergesort")

    group_cols = ["CompetitionID", "SessionID", "GameID", "TeamID"]
    ends["team_score_after"] = ends.groupby(group_cols, sort=False)["Result"].cumsum()
    ends["team_score_before"] = ends["team_score_after"] - ends["Result"]

    opponent_scores = (
        ends[["CompetitionID", "SessionID", "GameID", "EndID", "TeamID", "team_score_before", "team_score_after"]]
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

    return ends.reset_index(drop=True)


def summarise_usage(ends: pd.DataFrame) -> pd.DataFrame:
    """Aggregate overall power play usage metrics."""

    total_team_ends = len(ends)
    total_games = ends[["CompetitionID", "SessionID", "GameID"]].drop_duplicates().shape[0]
    total_powerplays = ends["PowerPlayUsed"].sum()
    unique_powerplay_ends = (
        ends.loc[ends["PowerPlayUsed"], ["CompetitionID", "SessionID", "GameID", "EndID"]]
        .drop_duplicates()
        .shape[0]
    )

    return pd.DataFrame(
        [
            ("Total games", total_games),
            ("Total team-end observations", total_team_ends),
            ("Power plays used (team perspective)", int(total_powerplays)),
            ("Unique ends with a power play", unique_powerplay_ends),
            ("Power play usage rate (team-end %)", total_powerplays / total_team_ends),
            ("Power play usage rate (end %)", unique_powerplay_ends / total_team_ends * 2),
        ],
        columns=["metric", "value"],
    )


def usage_by_end_number(ends: pd.DataFrame) -> pd.DataFrame:
    """Compute how often power plays are used by end number."""

    grouped = ends.groupby("EndID").agg(
        team_end_count=("TeamID", "size"),
        powerplays=("PowerPlayUsed", "sum"),
    )
    grouped["power_play_rate"] = grouped["powerplays"] / grouped["team_end_count"]
    return grouped.reset_index()


def scoring_outcomes(ends: pd.DataFrame) -> pd.DataFrame:
    """Summarise scoring results when teams use or skip a power play."""

    summary = (
        ends.groupby(["PowerPlayUsed", "Result"])
        .size()
        .reset_index(name="observations")
    )
    summary["share"] = summary["observations"] / summary.groupby("PowerPlayUsed")["observations"].transform("sum")
    return summary.sort_values(["PowerPlayUsed", "Result"])


def score_diff_before_distribution(ends: pd.DataFrame) -> pd.DataFrame:
    """Tabulate score differentials prior to an end, split by power play usage."""

    bins = [-20, -3, -1, 0, 1, 3, 20]
    labels = ["<= -3", "-2/-1", "0", "+1", "+2/+3", ">= +4"]
    ends = ends.copy()
    ends["score_state_bin"] = pd.cut(ends["score_diff_before"], bins=bins, labels=labels, right=True)

    dist = (
        ends.groupby(["PowerPlayUsed", "score_state_bin"])
        .size()
        .reset_index(name="observations")
    )
    dist["share"] = dist["observations"] / dist.groupby("PowerPlayUsed")["observations"].transform("sum")
    return dist.sort_values(["PowerPlayUsed", "score_state_bin"])


def usage_by_competition(ends: pd.DataFrame) -> pd.DataFrame:
    """Compute usage summary for each competition."""

    games = load_csv("games")[["CompetitionID", "GameID"]].drop_duplicates()
    competitions = load_csv("competitions")[["CompetitionID", "CompetitionName"]]

    unique_power_ends = (
        ends.loc[ends["PowerPlayUsed"], ["CompetitionID", "GameID", "EndID"]]
        .drop_duplicates()
        .groupby("CompetitionID")
        .size()
        .rename("unique_powerplay_ends")
        .reset_index()
    )

    ends_comp = (
        ends.groupby("CompetitionID")
        .agg(
            team_end_count=("TeamID", "size"),
            powerplays=("PowerPlayUsed", "sum"),
        )
        .reset_index()
    )
    ends_comp = ends_comp.merge(unique_power_ends, on="CompetitionID", how="left")
    ends_comp["unique_powerplay_ends"] = ends_comp["unique_powerplay_ends"].fillna(0).astype(int)

    ends_comp = ends_comp.merge(competitions, on="CompetitionID", how="left")
    ends_comp = ends_comp.merge(
        games.groupby("CompetitionID").size().rename("games_played").reset_index(),
        on="CompetitionID",
        how="left",
    )

    ends_comp["power_play_rate_team_end"] = ends_comp["powerplays"] / ends_comp["team_end_count"]
    ends_comp["power_play_rate_game"] = ends_comp["powerplays"] / ends_comp["games_played"]

    return ends_comp[
        [
            "CompetitionID",
            "CompetitionName",
            "games_played",
            "team_end_count",
            "powerplays",
            "unique_powerplay_ends",
            "power_play_rate_team_end",
            "power_play_rate_game",
        ]
    ].sort_values("CompetitionID")


def main() -> None:
    ends = prepare_ends_frame()

    summary = summarise_usage(ends)
    summary.to_csv(OUTPUT_DIR / "powerplay_usage_summary.csv", index=False)

    by_end = usage_by_end_number(ends)
    by_end.to_csv(OUTPUT_DIR / "powerplay_usage_by_end.csv", index=False)

    scoring = scoring_outcomes(ends)
    scoring.to_csv(OUTPUT_DIR / "powerplay_scoring_outcomes.csv", index=False)

    score_state = score_diff_before_distribution(ends)
    score_state.to_csv(OUTPUT_DIR / "powerplay_score_state_distribution.csv", index=False)

    by_comp = usage_by_competition(ends)
    by_comp.to_csv(OUTPUT_DIR / "powerplay_usage_by_competition.csv", index=False)

    print("Wrote summary tables to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
