
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_csv

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_MAP = {
    0: "Draw/Guard", 1: "Draw/Guard", 3: "Draw/Guard",
    2: "Guard Wall", 4: "Guard Wall",
    5: "Freeze Tap",
    6: "Takeout", 7: "Takeout",
    8: "Runback", 9: "Runback"
}

DEFAULT_PRIORS = {
    "Guard Wall": {"EV": 1.53, "P(>=3)": 0.202, "P(Steal)": 0.192},
    "Draw/Guard": {"EV": 1.70, "P(>=3)": 0.216, "P(Steal)": 0.208},
    "Freeze Tap": {"EV": 0.20, "P(>=3)": 0.000, "P(Steal)": 0.200},
    "Runback": {"EV": 0.50, "P(>=3)": 0.100, "P(Steal)": 0.300},
    "Takeout": {"EV": 0.50, "P(>=3)": 0.100, "P(Steal)": 0.300}
}

SCENARIOS = [
    {"End": 6, "Score_Diff": -2, "Hammer": "Opp", "Profile": "Aggressive"},
    {"End": 6, "Score_Diff": 0, "Hammer": "Opp", "Profile": "Standard"},
    {"End": 6, "Score_Diff": 2, "Hammer": "Opp", "Profile": "Conservative"},
    {"End": 7, "Score_Diff": -1, "Hammer": "Us", "Profile": "Aggressive"},
    {"End": 7, "Score_Diff": 1, "Hammer": "Us", "Profile": "Standard"},
    {"End": 8, "Score_Diff": -2, "Hammer": "Opp", "Profile": "Aggressive"},
    {"End": 8, "Score_Diff": 0, "Hammer": "Opp", "Profile": "Standard"},
    {"End": 8, "Score_Diff": 2, "Hammer": "Us", "Profile": "Conservative"},
]

def calculate_strategy_metrics(min_samples=10):
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

    strategy_metrics = []
    all_strategies = set(TASK_MAP.values())

    for strategy in all_strategies:
        group = merged[merged["Strategy"] == strategy]
        n_samples = len(group)

        if n_samples < min_samples:
            metrics = DEFAULT_PRIORS[strategy]
            metrics["Strategy"] = strategy
            strategy_metrics.append(metrics)
        else:
            ev = group['SignedResult'].mean()
            p_ge3 = (group['SignedResult'] >= 3).mean()
            p_steal = (group['SignedResult'] < 0).mean()
            strategy_metrics.append({
                "Strategy": strategy,
                "EV": ev,
                "P(>=3)": p_ge3,
                "P(Steal)": p_steal
            })
            
    return pd.DataFrame(strategy_metrics)

def recommend_strategy(profile, metrics_df):
    if profile == "Aggressive":
        best_row = metrics_df.loc[metrics_df["P(Steal)"].idxmax()]
    elif profile == "Conservative":
        best_row = metrics_df.loc[metrics_df["P(>=3)"].idxmin()]
    else:
        best_row = metrics_df.loc[metrics_df["EV"].idxmin()]
    
    return best_row["Strategy"], best_row["EV"], best_row["P(>=3)"], best_row["P(Steal)"]

def main():
    strategy_metrics_df = calculate_strategy_metrics()
    
    playbook_records = []
    for scenario in SCENARIOS:
        profile = scenario["Profile"]
        call, ev, p_ge3, p_steal = recommend_strategy(profile, strategy_metrics_df)
        
        record = scenario.copy()
        record["Call"] = call
        record["EV"] = ev
        record["P(>=3)"] = p_ge3
        record["P(Steal)"] = p_steal
        playbook_records.append(record)
        
    playbook_df = pd.DataFrame(playbook_records)
    
    final_cols = ["End", "Score_Diff", "Hammer", "Profile", "Call", "EV", "P(>=3)", "P(Steal)"]
    playbook_df = playbook_df[final_cols]
    
    playbook_df.to_csv(OUTPUT_DIR / "scenario_playbook_final.csv", index=False, float_format="%.3f")
    print(f"Successfully generated playbook at {OUTPUT_DIR / 'scenario_playbook_final.csv'}")

if __name__ == "__main__":
    main()
