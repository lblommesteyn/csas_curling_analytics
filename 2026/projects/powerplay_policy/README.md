# Dynamic Power-Play Policy Project

Goal: Learn an optimal, opponent-aware policy for deploying the power play using a Markov decision process (MDP) / stochastic game formulation.

### Key Ideas
- **State space**: score differential, hammer possession, end number, timeout availability, opponent cluster, recent scoring momentum.
- **Actions**: `use_power_play` vs `hold_power_play`, with optional shot-choice extensions.
- **Transition model**: Estimated from historical ends via conditional outcome probabilities (score changes, hammer retention).
- **Reward**: Win probability or expected score differential at game end (calibrated using logistic regression on final score).
- **Solution methods**: Value iteration / policy iteration for global policy; model-based RL (Bayesian or regularized) to incorporate uncertainty; opponent-specific policies via clustering.

### Project Layout
- `data/` (ignored): intermediate feature tables.
- `src/feature_engineering.py`: transforms raw ends/games into MDP states.
- `src/transition_estimator.py`: fits transition matrices per opponent cluster.
- `src/policy_solver.py`: iterative solvers, including risk-adjusted versions.
- `notebooks/`: exploratory notebooks for validation & visualization.

### Immediate Next Steps
1. Build state representation prototypes; evaluate dimensionality & coverage.
2. Fit baseline transition model (e.g., multinomial logistic regression) to quantify outcome probabilities conditioned on state & action.
3. Run value iteration to obtain first draft policy; compare to empirical usage frequency to validate realism.
4. Translate score swings into win-probability impact using the fitted logistic model for easier coaching communication.
5. Extend to opponent-aware policy by conditioning on cluster-specific transitions and evaluating win probability lift.
