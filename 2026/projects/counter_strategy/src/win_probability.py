import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def build_transition_matrix(df: pd.DataFrame):
    """
    Builds a transition matrix for end scores.
    State: (Hammer, Score) -> Next State: (Hammer, Score)
    Simplified: Just P(Score | Hammer)
    """
    # Group by Hammer and Score
    # Assuming 'Hammer' (1/0) and 'Score' (End Score) columns exist.
    # If not, we simulate based on typical Mixed Doubles distributions.
    
    # Typical Mixed Doubles Score Distribution (approximate)
    # Hammer: 0 (Blank), 1 (30%), 2 (30%), 3 (15%), 4+ (5%), Steal -1 (15%), Steal -2 (5%)
    # No Hammer: Steal 1 (25%), Steal 2 (10%), Force 1 (40%), Force 2+ (25%)
    
    # Let's try to extract from data if possible
    if 'Hammer' in df.columns and 'Score' in df.columns:
        transitions = df.groupby('Hammer')['Score'].value_counts(normalize=True).unstack().fillna(0)
        return transitions
    else:
        # Fallback: Hardcoded probabilities based on elite play
        # Rows: Hammer (0=No, 1=Yes)
        # Cols: Score (-2 to 5)
        scores = [-2, -1, 0, 1, 2, 3, 4, 5]
        probs_hammer = [0.05, 0.15, 0.05, 0.30, 0.25, 0.15, 0.04, 0.01] # Sum ~ 1
        probs_no_hammer = [0.05, 0.20, 0.10, 0.40, 0.15, 0.08, 0.02, 0.00] # Stealing is positive for No Hammer team?
        # Wait, Score usually means "Points for Hammer Team". 
        # If Hammer scores, Score > 0. If Steal, Score < 0.
        
        # Let's standardize: Score is from perspective of Team A (who starts with Hammer).
        # Actually, simpler: Score is points scored in the end. 
        # If Hammer team scores X, they lose hammer.
        # If Hammer team gives up steal (score -X), they keep hammer (usually? No, in MD, steal means other team scores).
        # Rules: 
        # - Score > 0: Hammer team scores. Next end: Other team has hammer.
        # - Score = 0: Blank. Next end: Hammer team keeps hammer.
        # - Score < 0: Steal. Next end: Hammer team keeps hammer? No, scoring team gives up hammer.
        # Wait, in curling, the team that scores GIVES UP the hammer. 
        # If you steal (score without hammer), you scored, so you give up hammer? No, you stole, so you didn't have hammer. 
        # The team that got scored ON receives the hammer.
        # So:
        # - Team A has Hammer.
        # - A scores > 0: Team B gets Hammer.
        # - A scores 0 (Blank): Team A keeps Hammer.
        # - B steals (A scores < 0): Team A gets Hammer (because B scored).
        
        return {
            'hammer': dict(zip(scores, probs_hammer)),
            'no_hammer': dict(zip(scores, probs_no_hammer)) # This is tricky without real data
        }

def simulate_game(start_score_diff, start_end, start_hammer, n_sims=1000):
    """
    Monte Carlo simulation of a game.
    start_score_diff: My Score - Opponent Score
    start_end: Current End (1-8)
    start_hammer: 1 if I have hammer, 0 if opponent has hammer
    """
    wins = 0
    
    # Simplified distributions (Prob of scoring X points given Hammer)
    # Key: Points scored by Hammer team. 
    # Value: Probability
    # Note: In MD, blanking is rare.
    dist_hammer = {1: 0.35, 2: 0.30, 3: 0.15, 4: 0.05, 0: 0.05, -1: 0.10} # -1 means steal of 1
    dist_no_hammer = {-1: 0.25, -2: 0.10, 0: 0.05, 1: 0.40, 2: 0.15, 3: 0.05} # Positive means Force
    
    # Actually, let's use a simpler model:
    # P(Score | Hammer)
    # If Hammer:
    #   Score 1: 30%, Score 2: 30%, Score 3: 10%, Score 4: 5%
    #   Blank (0): 5%
    #   Steal 1 (-1): 15%, Steal 2 (-2): 5%
    # If No Hammer (Opponent has Hammer):
    #   Opponent Scores 1: 30% ... (Symmetric)
    
    outcomes = [1, 2, 3, 4, 0, -1, -2]
    probs = [0.30, 0.30, 0.10, 0.05, 0.05, 0.15, 0.05]
    
    for _ in range(n_sims):
        score_diff = start_score_diff
        hammer = start_hammer # 1 = Us, 0 = Them
        
        for end in range(start_end, 9): # Ends 1 to 8
            # Who has hammer?
            if hammer == 1:
                # We have hammer. We score X.
                points = np.random.choice(outcomes, p=probs)
                score_diff += points
                
                if points > 0:
                    hammer = 0 # We scored, lost hammer
                elif points < 0:
                    hammer = 1 # We gave up steal, we keep hammer (Wait, rule: Scoring team gives up hammer. If they stole, they scored. So we get hammer.)
                    # Correct Rule: The team that did NOT score gets the hammer.
                    # If we have hammer and score (points > 0), opponent gets hammer.
                    # If we have hammer and give up steal (points < 0), opponent scored. So WE get hammer.
                    hammer = 1 
                else:
                    hammer = 1 # Blank, keep hammer
            else:
                # Opponent has hammer. They score X (from their perspective).
                # So change in score_diff is -X.
                points = np.random.choice(outcomes, p=probs)
                score_diff -= points
                
                if points > 0:
                    hammer = 1 # They scored, we get hammer
                elif points < 0:
                    hammer = 0 # They gave up steal (we stole), they get hammer
                else:
                    hammer = 0 # Blank, they keep hammer
                    
        if score_diff > 0:
            wins += 1
        elif score_diff == 0:
            wins += 0.5 # Tie
            
    return wins / n_sims

def generate_wpa_matrix(output_path: Path):
    """
    Generates a heatmap of Win Probability for different states.
    Fixed: End 8.
    Varying: Score Diff (-4 to +4), Hammer (Yes/No).
    """
    score_diffs = range(-4, 5)
    ends = range(1, 9)
    
    # Let's do a heatmap for End vs Score Diff (assuming Hammer)
    data = []
    for end in ends:
        row = []
        for diff in score_diffs:
            wp = simulate_game(diff, end, 1, n_sims=500)
            row.append(wp)
        data.append(row)
        
    plt.figure(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt=".2f", xticklabels=score_diffs, yticklabels=ends, cmap="RdYlGn")
    plt.title("Win Probability (With Hammer)")
    plt.xlabel("Score Differential")
    plt.ylabel("Current End")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved WPA matrix to {output_path}")

if __name__ == "__main__":
    output_dir = Path("2026/projects/counter_strategy/outputs")
    output_dir.mkdir(exist_ok=True, parents=True)
    generate_wpa_matrix(output_dir / "wpa_matrix.png")
