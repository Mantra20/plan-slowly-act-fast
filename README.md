# Plan Slowly, Act Fast: When Should a Robot Think?

Research project on cost-aware arbitration between planning and habit for world-model-based robots.

A robot that plans with a learned world model can handle new situations, but each plan costs computation, latency and energy. This project studies when a robot should spend computation to plan, and how it can reuse what it has learned when a situation returns.

The agent has three components:

- **Planner:** imagines trajectories with a world model and picks the best one.
- **Habit:** a small fast policy, learned by imitating the planner and stored per context.
- **Arbiter:** runs the planner only when the expected gain exceeds the cost of computation.

## Status

Early stage. A first simulated testbed with three alternating hidden contexts shows 99.7% success for the arbiter with context memory, against 96.9% for a permanent planner, with 5.8 times less computation. These results are preliminary and the code is not yet released.

## Contents

- `paper/`: research proposal (`proposal.pdf`, LaTeX source `proposal.tex`)
- `docs/ROADMAP.md`: plan of the project
- `src/`: code (to come)
- `experiments/`: experiment configurations and results (to come)

## Citation

If you refer to this work, please cite the repository until a paper is available.
