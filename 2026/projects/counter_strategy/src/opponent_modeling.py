import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path
import joblib

def train_response_model(df: pd.DataFrame, output_dir: Path):
    """
    Trains a model to predict Shot 2 Task based on Shot 1 outcome.
    """
def train_response_model(df: pd.DataFrame, output_dir: Path):
    """
    Trains a model to predict Shot 2 Task based on Shot 1 outcome.
    """
    # 1. Prepare Data
    # Sort by ShotID
    df = df.sort_values(['GameID', 'EndID', 'ShotID'])
    
    # Calculate ShotNum
    group_keys = ['CompetitionID', 'SessionID', 'GameID', 'EndID']
    df['ShotNum'] = df.groupby(group_keys).cumcount() + 1
    
    if 'Cluster' not in df.columns:
        # Assign dummy clusters if not present
        np.random.seed(42)
        df['Cluster'] = np.random.randint(0, 3, size=len(df))
        
    # Filter for Shot 1 (Defense) and Shot 2 (Offense)
    shot1 = df[df['ShotNum'] == 1].set_index(group_keys)
    shot2 = df[df['ShotNum'] == 2].set_index(group_keys)
    
    # Join
    data = shot1.join(shot2, lsuffix='_1', rsuffix='_2', how='inner')
    
    # Features: Shot 1 coordinates (stone_1_x, stone_1_y) and Cluster
    # Target: Shot 2 Task
    features = ['stone_1_x_1', 'stone_1_y_1', 'Cluster_1'] 
    target = 'Task_2'
    
    # Drop NaNs
    model_data = data[features + [target]].dropna()
    
    if model_data.empty:
        print("No paired shot data found.")
        return

    X = model_data[features]
    y = model_data[target]
    
    # Train Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    clf.fit(X, y)
    
    # Evaluate
    print("Opponent Response Model Accuracy:")
    print(clf.score(X, y))
    
    # Save Model
    joblib.dump(clf, output_dir / "opponent_model.pkl")
    
    # Generate Probabilities for a scenario
    # Scenario: Perfect Freeze (x=0, y=0) against Cluster 0 (Aggressive)
    # Assuming 0,0 is center/button for now.
    scenario = pd.DataFrame({'stone_1_x_1': [0], 'stone_1_y_1': [0], 'Cluster_1': [0]})
    probs = clf.predict_proba(scenario)[0]
    classes = clf.classes_
    
    # Save probabilities to CSV
    prob_df = pd.DataFrame({'Task': classes, 'Probability': probs})
    prob_df.to_csv(output_dir / "response_probabilities.csv", index=False)
    print("Saved response probabilities.")

if __name__ == "__main__":
    data_path = Path("2026/Stones.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        output_dir = Path("2026/projects/counter_strategy/outputs")
        
        train_response_model(df, output_dir)
