import pandas as pd
stones = pd.read_csv("2026/Stones.csv", usecols=["Task"])
print("Task Value Counts:")
print(stones["Task"].value_counts())
