# Shot Value Engine Project

Goal: Estimate the expected scoring value of candidate shots under mixed-doubles layouts using spatial analytics, Bayesian inference, and reinforcement learning.

### Key Ideas
- **State representation**: Encode stone coordinates with rotational/reflection symmetries removed via group-theoretic transformations to reduce the state space.
- **Shot taxonomy**: Standardize shot intents (draw, tap, peel, freeze, etc.) using the Curling Canada Shot Matrix guide (archived 2015-09-11) and parameterize target coordinates, weight, broom placement.
- **Value function**: Estimate expected end-score change via Bayesian regression or fitted Q-iteration that blends historical outcomes with simulated perturbations.
- **Uncertainty modeling**: Capture execution variance per team/player, enabling credible intervals for shot recommendations.
- **What-if simulator**: Combine learned dynamics with Monte Carlo rollouts to evaluate alternative shot sequences.

### Project Layout
- `src/preprocessing.py`: symmetry normalization & shot tagging.
- `src/state_encoder.py`: transforms stone layouts into low-dimensional embeddings using group actions or graph convolutions.
- `src/value_model.py`: Bayesian hierarchical model / RL estimator for shot value.
- `notebooks/`: interactive EDA and validation.

### Immediate Next Steps
1. Build symmetry-reduced coordinate system so equivalent layouts map to the same canonical representation.
2. Train baseline expected score model (e.g., gradient boosted regressor) as a sanity check.
3. Extend to Bayesian or RL formulations with uncertainty estimates.
4. Visualize recommended shot sequences and quantify lift vs empirical outcomes.
5. Validate task code coverage against additional Curlit competitions as more datasets become available.
