# Ant Colony Optimization — Fatigue-Constrained Pathfinding

`ants.py` implements an Ant Colony Optimization (ACO) algorithm for the fatigue-weighted shortest path problem. It operates on the same graph model as the rest of the project and can be compared directly against the Dijkstra/Pareto-pruning baseline.

---

## Problem Model

Each edge in the graph has two attributes:

| Attribute | Meaning |
|---|---|
| `distance` | Raw length of the edge |
| `fatigue_cost` | How much this edge increases the traveller's fatigue level |

An ant starts at the source node with **fatigue = 1**. Each time it crosses an edge:

```
travel_time += distance × current_fatigue
fatigue     += edge.fatigue_cost
```

The goal is to find the path from `source` to `target` that minimises total travel time, subject to the constraint that cumulative fatigue never exceeds **F_max**.

---

## Algorithm Overview

### 1. Initialisation

- One pheromone level τ(i, j) is maintained per directed edge, initialised to `initial_pheromone = 1.0`.
- F_max is computed from the network (maximum feasible fatigue level) and tightens dynamically during the run.

### 2. Per-Iteration Loop

For each of the `num_ants` ants:

1. The ant starts at `source` with fatigue = 1, visited = {source}.
2. At each step the ant applies **epsilon-greedy** neighbour selection:
   - With probability `epsilon` → pick a random valid neighbour (exploration).
   - Otherwise → pick the neighbour with the highest pheromone level (exploitation).
3. A neighbour is **valid** if:
   - It has not been visited yet in this path (no cycles).
   - Moving to it would keep fatigue ≤ current F_max.
4. If no valid neighbour exists → ant **FAILS**, turn ends.
5. If the ant reaches `target` → ant **SUCCEEDS**:
   - Pheromones are deposited on every edge of the path (proportional to path quality).
   - If the ant's final fatigue < current F_max → **F_max tightens** to that value.
   - The ant resets to source and tries again under the tighter constraint.
   - This inner loop continues until the ant fails under the new F_max.

After all ants have moved, pheromones evaporate globally:

```
τ(i, j) ← max(min_pheromone, τ(i, j) × (1 − decay_rate))
```

### 3. Pheromone Deposit

When an ant succeeds, each edge it traversed receives a deposit:

```
δ = 1000 / path_cost
path_cost = path_cost_weight × total_time + distance_weight × total_distance
```

Shorter, faster paths deposit more pheromone, reinforcing better routes over time.

### 4. F_max Tightening

The colony maintains a single shared F_max. Every time an ant reaches the goal with a fatigue level strictly below F_max, the colony narrows the search space by lowering F_max to that value. This progressively focuses exploration on lower-fatigue paths and acts as an implicit upper-bound pruning mechanism.

### 5. Convergence

The algorithm stops when:

- **Convergence**: the best cost found by the colony has not improved for `pheromone_history_size` consecutive iterations **and** at least one valid solution has been found; or
- **Max iterations**: `max_iterations` is reached.

> The convergence criterion tracks best-cost stability, not pheromone-level stability. A pheromone-based criterion causes false convergence when F_max tightening stalls exploration (0 successful ants → evaporation drives all pheromones to the floor → artificial stability triggers premature termination).

---

## Configuration

All parameters are in the `ACO_CONFIG` dict at the top of [ants.py](ants.py):

| Parameter | Default | Effect |
|---|---|---|
| `num_ants` | 10000 | Ants deployed per iteration. More ants → better exploration, slower iterations. |
| `epsilon` | 0.15 | Probability of random neighbour choice. Higher → more exploration, slower convergence. |
| `decay_rate` | 0.1 | Fraction of pheromone evaporated each iteration. Higher → faster forgetting of poor paths. |
| `initial_pheromone` | 1.0 | Starting pheromone level on every edge. |
| `min_pheromone` | 0.01 | Floor that prevents total pheromone extinction. |
| `max_iterations` | 10000 | Hard stop if convergence is never reached. |
| `pheromone_history_size` | 20 | Number of consecutive no-improvement iterations required to declare convergence. |
| `path_cost_weight` | 1.0 | Weight of travel time in the path cost used for pheromone deposit. |
| `distance_weight` | 0.1 | Weight of raw distance in the path cost (secondary objective). |

---

## Code Structure

```
ants.py
├── ACO_CONFIG                  Global parameter dictionary
│
├── class Ant                   Single ant agent
│   ├── __init__ / reset        Initialise / restart from source
│   ├── explore_step            One epsilon-greedy move
│   └── complete_exploration    Full source→target traversal
│
├── make_pheromones             Initialise pheromone dict from adjacency list
├── evaporate                   Global pheromone decay (with floor)
├── deposit_pheromones          Reward successful ant's edges
│
├── class Colony                ACO orchestrator
│   ├── __init__                Build adjacency list, initialise state
│   ├── run_iteration           Deploy all ants, update pheromones
│   ├── check_convergence       Best-cost stability criterion
│   └── run                     Main loop with progress output
│
├── load_aco_scenario           Load Network + resolve src/tgt/F_max
└── compare_aco_vs_baseline     Run ACO + Dijkstra side-by-side and print report
```

---

## Usage

### Run the default comparison

```bash
python ants.py
```

This runs ACO against the Dijkstra/Pareto-pruning baseline on `examples/medium-largefatigue.txt` and prints a side-by-side report.

### Use from Python

```python
from ants import Colony, load_aco_scenario, ACO_CONFIG

# Optional: tweak parameters before running
ACO_CONFIG['num_ants'] = 500
ACO_CONFIG['epsilon']  = 0.2

net, src, tgt, fmax = load_aco_scenario('examples/medium-smallfatigue.txt')
colony = Colony(net, src, tgt, fmax)
result = colony.run(max_iterations=200)

print(result['best_path'])   # list of node IDs
print(result['best_cost'])   # weighted travel time
print(result['final_F_max']) # tightest fatigue bound reached
```

### Run on a specific graph

```python
from ants import compare_aco_vs_baseline

compare_aco_vs_baseline(
    'examples/large-smallfatigue.txt',
    source=0,   # optional override
    target=500, # optional override
)
```

### Available graph files

| File | Nodes | Fatigue level |
|---|---|---|
| `examples/small.txt` | ~10 | — |
| `examples/medium-nofatigue.txt` | ~100 | none |
| `examples/medium-smallfatigue.txt` | ~100 | low |
| `examples/medium-largefatigue.txt` | ~100 | high |
| `examples/large-nofatigue.txt` | ~10 000 | none |
| `examples/large-smallfatigue.txt` | ~10 000 | low |
| `examples/large-largefatigue.txt` | ~10 000 | high |

---

## Output Format

```
================================================================================
ANT COLONY OPTIMIZATION — Fatigue-Constrained Pathfinding
================================================================================
  Graph   : examples/medium-smallfatigue.txt
  Start   : 0  |  Target : 99  |  F_max : 991
  Ants    : 10000  |  Epsilon : 0.15  |  Decay : 10%
================================================================================

[Baseline] Running Dijkstra/Pruning...
  Cost : 29934.00  |  Path length : 14 nodes  |  Time : 0.0025s

[ACO] Running Ant Colony Optimization...
  #iter    success/ants     best cost          F_max
  ----------------------------------------------------------
  #1       62/10000         150289.6           37
  #5       21/10000         107991.7           24
  #10      4/10000         88483.2            18
  ...

  → CONVERGED at iteration 29  (best cost unchanged for last 20 iterations)

================================================================================
COMPARISON: ACO vs Dijkstra/Pruning Baseline
================================================================================
  Dijkstra/Pruning : cost=29934.00,  time=0.0025s
  ACO              : cost=88483.20,  time=4.77s
```

The `success/ants` column counts the **total** number of successful trips in that iteration (one ant can succeed multiple times in its turn if F_max keeps tightening).

---

## Relationship to the Rest of the Project

| Component | Role |
|---|---|
| `network.py` — `Network` | Graph loader; provides `net.df`, `net.start`, `net.goal`, `net.compute_F_max()`, `net.pruning()` |
| `main.py` — `Q_pruning` | Exact Dijkstra + Pareto-dominance baseline that ACO is compared against |
| `ants.py` — `Colony` | Heuristic meta-heuristic; trades optimality for scalability on large instances |

ACO is a **heuristic** — it does not guarantee finding the optimal path. It is most useful when the graph is too large for exact methods, or when approximate solutions are acceptable in exchange for faster practical performance on repeated queries.
