import pandas as pd
stones = pd.read_csv("2026/Stones.csv", nrows=100)
# Group by game and end
grouped = stones.groupby(["GameID", "EndID"])
for name, group in grouped:
    print(f"Game {name[0]} End {name[1]}: {group['ShotID'].tolist()}")
    break # Just show one end
