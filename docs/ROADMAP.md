# Roadmap

Four parts, first in simulation (MuJoCo or ManiSkill), then on an SO-101 arm driven with LeRobot.

## Part 1: Arbitration in simulation
- [ ] Train a compact TD-MPC2-type world model on manipulation tasks with changing dynamics (carried mass, friction, worn actuator)
- [ ] Implement the planner and the small habit policy
- [ ] Implement the arbiter with two signals: world-model surprise and habit competence
- [ ] Compare against a permanent planner and a frozen habit

## Part 2: Context memory
- [ ] Recognise the hidden context online, allowing for a new context
- [ ] Store one habit per context
- [ ] Measure recovery cost when a context returns

## Part 3: Continual learning and neuroscience
- [ ] Offline generative replay against forgetting
- [ ] Faster learning under strong surprise
- [ ] Test known behavioural signatures of arbitration (persistent habits, return to deliberation after a context change, speed-accuracy trade-off)

## Part 4: Real robot and measurement of computation
- [ ] Port to the SO-101 arm with LeRobot
- [ ] Measure milliseconds per action and joules on an embedded board
- [ ] Compare with existing arbiters

## Outputs
- [ ] Paper 1: Architecture and computation
- [ ] Paper 2: Continual learning and computational neuroscience
- [ ] Shared open-source code base compatible with LeRobot
