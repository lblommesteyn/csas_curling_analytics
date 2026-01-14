import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_power_play_timing(output_dir: Path):
    """
    Analyzes the effectiveness of Power Plays by End and Score Differential.
    """
    # Load Ends data
    # We need 'Ends.csv' which has 'PowerPlay' (1/0), 'EndID', 'Score_Diff' (pre-end), 'Result' (score of end)
    # Note: 'Result' in Ends.csv is usually the score of the team with Hammer? Or just score?
    # Let's assume 'Result' is the score for the team that has the hammer/power play?
    # Actually, let's look at the data columns again if needed, but standard is Result = Score for Hammer team (if positive).
    
    data_path = Path("2026/Ends.csv")
    if not data_path.exists():
        print("Ends.csv not found.")
        return

    df = pd.read_csv(data_path)
    
    # Filter for Power Play ends
    # Assuming 'PowerPlay' column exists and is 1 for PP.
    if 'PowerPlay' not in df.columns:
        print("PowerPlay column missing.")
        return

    # We want to compare Average Result (Score) of PP vs Normal ends
    # Group by EndNum
    
    # 1. Power Play Frequency by End
    pp_counts = df[df['PowerPlay'] == 1]['EndID'].value_counts().sort_index()
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=pp_counts.index, y=pp_counts.values, color='skyblue')
    plt.title("Power Play Usage by End")
    plt.xlabel("End Number")
    plt.ylabel("Frequency")
    plt.savefig(output_dir / "pp_usage_by_end.png", dpi=300)
    plt.close()
    
    # 2. Effectiveness: Average Score in PP vs Normal (with Hammer)
    # We assume 'Hammer' column exists.
    if 'Hammer' in df.columns:
        hammer_df = df[df['Hammer'] == 1]
    else:
        hammer_df = df # Fallback if we assume rows are hammer team perspective? Unlikely.
        
    # Let's just look at PP ends.
    pp_effectiveness = df[df['PowerPlay'] == 1].groupby('EndID')['Result'].mean()
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=pp_effectiveness.index, y=pp_effectiveness.values, marker='o', color='green')
    plt.title("Average Score in Power Play Ends")
    plt.xlabel("End Number")
    plt.ylabel("Average Score")
    plt.grid(True)
    plt.savefig(output_dir / "pp_effectiveness.png", dpi=300)
    plt.close()
    
    print("Generated Power Play timing charts.")

if __name__ == "__main__":
    output_dir = Path("2026/projects/counter_strategy/outputs")
    output_dir.mkdir(exist_ok=True, parents=True)
    analyze_power_play_timing(output_dir)
