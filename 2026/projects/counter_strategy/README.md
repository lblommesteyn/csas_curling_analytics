# Counter-Strategy Simulator Project

Goal: Design defensive responses to opponent power plays using opponent clustering, minimax optimization, and Monte Carlo simulation.

### Key Ideas
- **Opponent archetypes**: Cluster teams by shot tendencies and outcomes during power plays.
- **Scenario generator**: Simulate plausible shot sequences under each archetype using learned stochastic models.
- **Defensive action space**: Enumerate guard / freeze / tapback options and their success probabilities.
- **Game-theoretic evaluation**: Solve a minimax or robust optimization problem to pick defenses that minimize the opponent's expected multi-point score.
- **Coach-facing outputs**: Playbooks highlighting recommended counters conditioned on layout + opponent type.
- **Stress testing**: Explore alternative success rates and layout-specific scenarios to flag when non-standard defenses (runback pressure, guard wall) overtake the baseline freeze response.

### Project Layout
- `src/opponent_clustering.py`: derive opponent clusters using mixture models / HMMs.
- `src/scenario_simulator.py`: roll out opponent shot trees with stochastic weights.
- `src/defense_optimizer.py`: minimax solver producing recommended defensive actions.
- `notebooks/`: qualitative review and playbook creation.

### Immediate Next Steps
1. Aggregate power-play possessions into sequence summaries (first 3 shots, resulting score).
2. Fit clustering model to identify archetypes (e.g., aggressive draw vs hit-heavy teams).
3. Estimate action-success probabilities for key defensive shots.
4. Implement minimax optimization to evaluate defensive choices vs each archetype.
5. Expand scenario stress tests with data-driven success-rate priors (e.g., by sheet conditions or opponent style) to refine counter recommendations.
