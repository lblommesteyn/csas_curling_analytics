import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_spatial_heatmap(df: pd.DataFrame, task_id: int, output_path: Path):
    """
    Generates a spatial heatmap for a specific task.
    """
    # 1. Sort by ShotID to ensure correct order
    df = df.sort_values(['GameID', 'EndID', 'ShotID'])
    
    # 2. Identify Shot Number within each end
    # Group by full key to ensure unique ends
    group_keys = ['CompetitionID', 'SessionID', 'GameID', 'EndID']
    df['ShotNum'] = df.groupby(group_keys).cumcount() + 1
    
    # 3. Filter for Shot 1 (Defensive Shot) and the specific Task
    # We assume Task 5 (Freeze Tap) is played as Shot 1.
    task_data = df[(df['Task'] == task_id) & (df['ShotNum'] == 1)].copy()
    
    if task_data.empty:
        print(f"No data for Task {task_id} at Shot 1")
        return

    # Create the plot
    plt.figure(figsize=(6, 10))
    
    # Draw the House (Curling Rings)
    # Assuming coordinates are in cm? 
    # If stone_1_x is 750, and sheet is ~475 wide. 
    # Maybe 0 is center? 
    # Let's assume standard coordinates where (0,0) is button.
    # If so, 750 is far. 
    # Let's just plot and see.
    
    # Plot KDE Heatmap
    # Use stone_1_x and stone_1_y for Shot 1
    if 'stone_1_x' not in task_data.columns or 'stone_1_y' not in task_data.columns:
        print("Missing coordinate data (stone_1_x, stone_1_y)")
        return
        
    sns.kdeplot(
        data=task_data, 
        x='stone_1_x', 
        y='stone_1_y', 
        fill=True, 
        cmap="Reds", 
        alpha=0.6,
        levels=10,
        thresh=0.1
    )
    
    plt.title(f"Spatial Precision: Task {task_id}")
    plt.xlabel("Lateral Position")
    plt.ylabel("Vertical Position")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {output_path}")

if __name__ == "__main__":
    # Test run
    data_path = Path("2026/Stones.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        output_dir = Path("2026/projects/counter_strategy/outputs")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Plot Freeze Tap (Task 5)
        plot_spatial_heatmap(df, 5, output_dir / "spatial_heatmap_freeze.png")
