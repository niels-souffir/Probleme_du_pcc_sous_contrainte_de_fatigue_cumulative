"""
ants.py — Ant Colony Optimization for fatigue-constrained pathfinding.

Each ant navigates a weighted graph where traversing an edge costs
    distance × current_fatigue
and accumulates fatigue from each edge's fatigue coefficient.

Key behaviour (per iteration):
  - Each ant explores source → target.
  - On SUCCESS: time_max tightens to the ant's total_time, the ant resets
    and tries again under the new budget.  This continues until the ant
    fails (can't beat time_max).
  - On FAILURE: the ant's turn ends.
  - time_max acts as an online branch-and-bound bound: any partial path
    whose accumulated cost already exceeds time_max is pruned immediately.
  - Pheromones are deposited only for successful trips, then evaporated
    globally after all ants have moved.
  - Convergence is detected when the best cost has not improved for the
    last N iterations (and at least one solution has been found).

Initial time_max is computed from the graph structure as an upper bound
on the cost of any simple path:
    time_max = Σ(top n-1 edge distances) × (1 + Σ(top n-1 edge f_costs))
where n is the number of nodes.  It tightens dynamically as ants find
better solutions.
"""

import random
import time
from collections import deque
from typing import Optional

import numpy as np
from network import Network

# ── ACO configuration ─────────────────────────────────────────────────────────
ACO_CONFIG = {
    'num_ants': 10000,
    'epsilon': 0.3,                # exploration rate (random choice probability)
    'decay_rate': 0.05,              # pheromone evaporation per iteration
    'initial_pheromone': 1.0,
    'min_pheromone': 0.2,          # floor to prevent total extinction
    'max_iterations': 10000,
    'pheromone_history_size': 1000, # no-improvement iterations required to converge
}

# ── Ant status constants ──────────────────────────────────────────────────────
EXPLORING = 'EXPLORING'
SUCCESS   = 'SUCCESS'
FAILED    = 'FAILED'


# ══════════════════════════════════════════════════════════════════════════════
# ANT
# ══════════════════════════════════════════════════════════════════════════════

class Ant:
    """A single ant that navigates from source to target under a fatigue cap."""

    def __init__(self, source: int):
        self.source = source
        self._reset_state(source)

    def _reset_state(self, source: int) -> None:
        self.current_node      = source
        self.path              = [source]
        self.visited: set[int] = {source}
        self.fatigue           = 1
        self.total_time        = 0.0
        self.total_distance    = 0
        self.pheromones_secreted: list[tuple[int, int]] = []
        self.status            = EXPLORING

    def reset(self) -> None:
        """Reset ant to source for a new exploration attempt."""
        self._reset_state(self.source)

    # ── Single step ───────────────────────────────────────────────────────────

    def explore_step(
        self,
        adj: dict,
        pheromones: dict,
        target: int,
        time_max: float,
        epsilon: float,
    ) -> None:
        """Choose and move to the next node using epsilon-greedy selection.

        Candidates are pre-filtered to only unvisited neighbours whose single
        edge cost fits within the remaining time budget (change B).
        """
        node = self.current_node

        # Collect unvisited neighbours whose single edge cost fits the remaining budget
        candidates = [
            (nb, dist, f_cost)
            for nb, dist, f_cost in adj.get(node, [])
            if nb not in self.visited
            and self.total_time + dist * self.fatigue <= time_max
        ]

        if not candidates:
            self.status = FAILED
            return

        # Epsilon-greedy selection
        if random.random() < epsilon:
            nb, dist, f_cost = random.choice(candidates)
        else:
            # Proportional sampling weighted by pheromone level
            weights = [
                pheromones.get((node, nb), ACO_CONFIG['initial_pheromone'])
                for nb, _, _ in candidates
            ]
            nb, dist, f_cost = random.choices(candidates, weights=weights)[0]

        # Traverse the chosen edge (guaranteed to stay within budget by pre-filter)
        edge_cost = dist * self.fatigue
        self.total_time     += edge_cost
        self.total_distance += dist
        self.fatigue        += f_cost

        self.current_node = nb
        self.path.append(nb)
        self.visited.add(nb)
        self.pheromones_secreted.append((node, nb))

        if nb == target:
            self.status = SUCCESS

    # ── Full exploration ──────────────────────────────────────────────────────

    def complete_exploration(
        self,
        adj: dict,
        pheromones: dict,
        target: int,
        time_max: float,
        epsilon: float,
    ) -> tuple[list[int], int, float, str]:
        """Loop explore_step until SUCCESS or FAILED.

        Returns: (path, final_fatigue, total_time, status)
        """
        while self.status == EXPLORING:
            self.explore_step(adj, pheromones, target, time_max, epsilon)

        return self.path[:], self.fatigue, self.total_time, self.status



# ══════════════════════════════════════════════════════════════════════════════
# COLONY
# ══════════════════════════════════════════════════════════════════════════════

class Colony:
    """Main ACO system managing a population of ants over multiple iterations."""

    def __init__(
        self,
        net: Network,
        source: int,
        target: int,
        num_ants: int  = ACO_CONFIG['num_ants'],
        epsilon: float = ACO_CONFIG['epsilon'],
        decay_rate: float = ACO_CONFIG['decay_rate'],
    ):
        self.net        = net
        self.source     = source
        self.target     = target
        self.num_ants   = num_ants
        self.epsilon    = epsilon
        self.decay_rate = decay_rate

        # Build adjacency list once from the network DataFrame
        self.adj: dict = {}
        for line in net.df.itertuples(index=False):
            i      = int(line[0])
            j      = int(line[1])
            dist   = int(line[2])
            f_cost = int(line[3])
            self.adj.setdefault(i, []).append((j, dist, f_cost))

        # ── Node set and time_max ──────────────────────────────────────────────
        all_nodes: set[int] = set(self.adj.keys())
        for nbs in self.adj.values():
            for nb, _, _ in nbs:
                all_nodes.add(nb)
        n = len(all_nodes)
        k = max(n - 1, 1)

        all_dists  = sorted((d for nbs in self.adj.values() for _, d, _ in nbs), reverse=True)
        all_fcosts = sorted((f for nbs in self.adj.values() for _, _, f in nbs), reverse=True)
        top_dist   = sum(all_dists[:k])
        top_fcost  = sum(all_fcosts[:k])
        self.time_max: float = float(top_dist * (1 + top_fcost))  # tightens on success

        # ── Numpy graph arrays ─────────────────────────────────────────────────
        # Node index mapping: original IDs → contiguous 0-based indices.
        all_nodes_sorted   = sorted(all_nodes)
        self._node_to_idx  = {nd: i for i, nd in enumerate(all_nodes_sorted)}
        self._idx_to_node  = all_nodes_sorted          # list: array_idx → original id
        self._n_nodes      = len(all_nodes_sorted)
        self._source_idx   = self._node_to_idx[source]
        self._target_idx   = self._node_to_idx[target]

        max_deg = max((len(v) for v in self.adj.values()), default=1)

        # _nb_mat[i, j]   = j-th neighbour's array index for node i  (-1 = padding)
        # _dist_mat[i, j] = distance to that neighbour
        # _fc_mat[i, j]   = fatigue cost of that edge
        self._nb_mat   = np.full((self._n_nodes, max_deg), -1,  dtype=np.int32)
        self._dist_mat = np.zeros((self._n_nodes, max_deg),     dtype=np.float64)
        self._fc_mat   = np.zeros((self._n_nodes, max_deg),     dtype=np.float64)

        for node, neighbors in self.adj.items():
            i = self._node_to_idx[node]
            for slot, (nb, d, f) in enumerate(neighbors):
                j = self._node_to_idx[nb]
                self._nb_mat[i, slot]   = j
                self._dist_mat[i, slot] = d
                self._fc_mat[i, slot]   = f

        # Reverse lookup: (array_idx_from, array_idx_to) → column slot
        # Used for O(1) pheromone deposit by index.
        self._edge_slot: dict[tuple[int, int], int] = {}
        for i in range(self._n_nodes):
            for k_slot in range(max_deg):
                j = int(self._nb_mat[i, k_slot])
                if j >= 0:
                    self._edge_slot[(i, j)] = k_slot

        # Pheromone matrix — same layout as _nb_mat.
        self._pher_mat = np.full(
            (self._n_nodes, max_deg),
            ACO_CONFIG['initial_pheromone'],
            dtype=np.float64,
        )

        # ── Colony state ──────────────────────────────────────────────────────
        self.best_path: list[int]     = []
        self.best_cost: float         = float('inf')
        self.best_cost_history: deque = deque(
            maxlen=ACO_CONFIG['pheromone_history_size']
        )
        self.convergence_history: list[dict] = []
        self.iteration_count = 0

    # ── Vectorised trip ───────────────────────────────────────────────────────

    def _run_trip_vectorized(
        self,
        pos:   np.ndarray,
        fat:   np.ndarray,
        ttime: np.ndarray,
        alive: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Move all alive ants step-by-step until each succeeds or fails.

        Modifies pos, fat, ttime, alive in-place.
        Returns (succeeded, path_mat, step_count):
          succeeded  bool[na]          — True for ants that reached the target
          path_mat   int32[na, n_nodes] — path_mat[i, :step_count[i]] = node indices
          step_count int32[na]          — number of steps taken (including start)

        Notes:
          - Visited tracking uses a (na × n_nodes) boolean matrix.  For medium
            graphs (≤ ~2000 nodes) this is fine; for very large graphs consider
            reducing num_ants or using a step-limited approximation.
          - max_steps is capped at n_nodes so no ant can revisit a node (change B
            pre-filter already prevents this, but the cap is a safety net).
        """
        na        = len(pos)
        n         = self._n_nodes
        max_steps = n                          # simple path ≤ n edges
        max_deg   = self._nb_mat.shape[1]

        # path_mat[:,0] = start node; subsequent steps written as ants move
        path_mat   = np.empty((na, max_steps + 1), dtype=np.int32)
        path_mat[:, 0] = pos
        step_count = np.ones(na, dtype=np.int32)    # starts at 1 (start already stored)

        visited = np.zeros((na, n), dtype=bool)
        visited[np.arange(na), pos] = True

        succeeded = np.zeros(na, dtype=bool)

        for _ in range(max_steps):
            if not alive.any():
                break

            alive_idx = np.where(alive)[0]            # (m,)
            cur_pos   = pos[alive_idx]                 # (m,)
            cur_fat   = fat[alive_idx]                 # (m,)
            cur_time  = ttime[alive_idx]               # (m,)

            # Neighbour properties at each ant's current position
            nbs_m  = self._nb_mat[cur_pos]             # (m, max_deg)
            dsts_m = self._dist_mat[cur_pos]           # (m, max_deg)
            fcs_m  = self._fc_mat[cur_pos]             # (m, max_deg)

            # Edge costs under current fatigue
            edge_costs = dsts_m * cur_fat[:, None]     # (m, max_deg)

            # Valid mask (change B): not padded, not visited, edge fits budget
            safe_nbs = np.where(nbs_m >= 0, nbs_m, 0)
            not_vis  = ~visited[alive_idx[:, None], safe_nbs]
            in_bud   = (cur_time[:, None] + edge_costs) <= self.time_max
            valid    = (nbs_m >= 0) & not_vis & in_bud   # (m, max_deg)

            # Ants with no valid move fail
            has_valid = valid.any(axis=1)              # (m,) bool
            alive[alive_idx[~has_valid]] = False

            move_idx = alive_idx[has_valid]            # ants that will actually move
            if len(move_idx) == 0:
                continue

            valid_m = valid[has_valid]                 # (m2, max_deg)
            m2      = len(move_idx)

            # ── Epsilon-greedy selection (vectorised) ────────────────────────
            is_explore = np.random.rand(m2) < self.epsilon

            # Exploration: uniform over valid neighbours (Gumbel-max trick)
            u = np.where(valid_m, np.random.rand(m2, max_deg), -np.inf)
            explore_sel = np.argmax(u, axis=1)         # (m2,)

            # Exploitation: proportional to pheromone (Gumbel-max trick)
            pher  = self._pher_mat[pos[move_idx]]      # (m2, max_deg)
            log_w = np.where(valid_m, np.log(pher + 1e-10), -np.inf)
            gum   = -np.log(-np.log(np.clip(np.random.rand(m2, max_deg), 1e-10, 1 - 1e-10)))
            exploit_sel = np.argmax(log_w + gum, axis=1)   # (m2,)

            chosen_slot = np.where(is_explore, explore_sel, exploit_sel)  # (m2,)

            # ── Apply chosen move ────────────────────────────────────────────
            rows     = np.arange(m2)
            nbs_sub  = nbs_m[has_valid]                # (m2, max_deg)
            ecs_sub  = edge_costs[has_valid]           # (m2, max_deg)
            fcs_sub  = fcs_m[has_valid]                # (m2, max_deg)

            chosen_nb = nbs_sub[rows, chosen_slot]     # (m2,) neighbour index
            chosen_ec = ecs_sub[rows, chosen_slot]     # (m2,) edge cost
            chosen_fc = fcs_sub[rows, chosen_slot]     # (m2,) fatigue cost

            ttime[move_idx] += chosen_ec
            fat[move_idx]   += chosen_fc
            pos[move_idx]    = chosen_nb
            visited[move_idx, chosen_nb] = True

            sc = step_count[move_idx]
            path_mat[move_idx, sc] = chosen_nb
            step_count[move_idx]  += 1

            # Ants that reached the target succeed
            at_target = (chosen_nb == self._target_idx)
            succeeded[move_idx[at_target]] = True
            alive[move_idx[at_target]]     = False

        return succeeded, path_mat, step_count

    # ── Single iteration ──────────────────────────────────────────────────────

    def run_iteration(self) -> dict:
        """Deploy all ants in vectorised batches; update pheromones."""
        na = self.num_ants
        pos   = np.full(na, self._source_idx, dtype=np.int32)
        fat   = np.ones(na,  dtype=np.float64)
        ttime = np.zeros(na, dtype=np.float64)
        alive = np.ones(na,  dtype=bool)

        successful_trips = 0

        # Each pass through _run_trip_vectorized moves all alive ants until they
        # succeed or fail.  Ants that succeed are reset and made alive again so
        # they can try to beat the tightened time_max.
        while alive.any():
            succeeded, path_mat, step_count = self._run_trip_vectorized(
                pos, fat, ttime, alive
            )

            for ant_i in np.where(succeeded)[0]:
                cost  = float(ttime[ant_i])
                steps = int(step_count[ant_i])
                path_idx = path_mat[ant_i, :steps]    # array indices
                successful_trips += 1

                # Deposit pheromones along this ant's path
                delta = 1000.0 / cost
                for a, b in zip(path_idx, path_idx[1:]):
                    k = self._edge_slot.get((int(a), int(b)))
                    if k is not None:
                        self._pher_mat[int(a), k] += delta

                # Update colony best and tighten time_max
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_path = [self._idx_to_node[int(x)] for x in path_idx]
                if cost < self.time_max:
                    self.time_max = cost

                # Reset this ant to source for another trip under the tighter budget
                pos[ant_i]   = self._source_idx
                fat[ant_i]   = 1.0
                ttime[ant_i] = 0.0
                alive[ant_i] = True

        # ── Evaporation ───────────────────────────────────────────────────────
        floor = ACO_CONFIG['min_pheromone']
        self._pher_mat *= (1.0 - self.decay_rate)
        np.clip(self._pher_mat, floor, None, out=self._pher_mat)

        # ── Elitist deposit (change A) ─────────────────────────────────────
        # Re-reinforce the global best path every iteration so pheromone
        # guidance survives long periods with 0 successes.
        if self.best_path and self.best_cost < float('inf'):
            best_idx = [self._node_to_idx[nd] for nd in self.best_path]
            delta_e  = 1000.0 / self.best_cost
            for a, b in zip(best_idx, best_idx[1:]):
                k = self._edge_slot.get((a, b))
                if k is not None:
                    self._pher_mat[a, k] += delta_e

        self.best_cost_history.append(self.best_cost)
        self.iteration_count += 1

        stats = {
            'iteration':       self.iteration_count,
            'successful_ants': successful_trips,
            'best_cost':       self.best_cost,
            'time_max':        self.time_max,
        }
        self.convergence_history.append(stats)
        return stats

    # ── Convergence check ─────────────────────────────────────────────────────

    def check_convergence(self) -> bool:
        """True when the best cost has not improved for the last N iterations.

        Requires at least one valid solution (best_cost != inf) before
        convergence can be declared — prevents false-positive convergence
        when all ants fail and F_max tightening has stalled exploration.
        """
        history_size = ACO_CONFIG['pheromone_history_size']
        if len(self.best_cost_history) < history_size:
            return False
        if float('inf') in self.best_cost_history:
            return False   # no solution found yet
        return len(set(self.best_cost_history)) == 1  # all equal → no improvement

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, max_iterations: int = ACO_CONFIG['max_iterations']) -> dict:
        """Run the ACO loop until convergence or max_iterations.

        Returns a result dict with best_path, best_cost, iteration_count,
        convergence_history, and final_F_max.
        """
        interval = 1

        print(f"  {'#iter':<8} {'success/ants':<16} {'best cost':<18} {'time_max'}")
        print("  " + "-" * 58)

        while True:
            stats = self.run_iteration()

            if self.iteration_count % interval == 0 or self.iteration_count == 1:
                cost_str = (f"{stats['best_cost']:.1f}"
                            if stats['best_cost'] != float('inf') else "—")
                tmax_str = (f"{stats['time_max']:.1f}"
                            if stats['time_max'] != float('inf') else "—")
                print(f"  #{stats['iteration']:<7} "
                      f"{stats['successful_ants']}/{self.num_ants:<13} "
                      f"{cost_str:<18} {tmax_str}")

            if self.check_convergence():
                print(f"\n  → CONVERGED at iteration {self.iteration_count}"
                      f"  (best cost unchanged for last {ACO_CONFIG['pheromone_history_size']} iterations)")
                break

            if self.iteration_count >= max_iterations:
                print(f"\n  → Max iterations ({max_iterations}) reached")
                break

        return {
            'best_path':           self.best_path,
            'best_cost':           self.best_cost,
            'iteration_count':     self.iteration_count,
            'convergence_history': self.convergence_history,
            'final_time_max':      self.time_max,
        }


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_aco_scenario(
    graph_filename: str,
    source: Optional[int] = None,
    target: Optional[int] = None,
) -> tuple[Network, int, int]:
    """Load a Network and resolve source/target nodes.

    Returns (net, source, target).
    Defaults: source/target from the file header.
    time_max is computed automatically inside Colony.__init__.
    """
    net = Network(filename=graph_filename)

    src = source if source is not None else net.start
    tgt = target if target is not None else net.goal

    # Validate nodes
    valid_nodes: set[int] = set()
    for line in net.df.itertuples(index=False):
        valid_nodes.add(int(line[0]))
        valid_nodes.add(int(line[1]))

    if src not in valid_nodes:
        raise ValueError(f"Source node {src} not found in graph")
    if tgt not in valid_nodes:
        raise ValueError(f"Target node {tgt} not found in graph")

    return net, src, tgt


def compare_aco_vs_baseline(
    graph_filename: str,
    source: Optional[int] = None,
    target: Optional[int] = None,
) -> None:
    """Run ACO and the pruning baseline on the same graph and print a comparison."""
    net, src, tgt = load_aco_scenario(graph_filename, source, target)

    print("=" * 80)
    print("ANT COLONY OPTIMIZATION — Fatigue-Constrained Pathfinding")
    print("=" * 80)
    print(f"  Graph   : {graph_filename}")
    print(f"  Start   : {src}  |  Target : {tgt}")
    print(f"  Ants    : {ACO_CONFIG['num_ants']}  |  "
          f"Epsilon : {ACO_CONFIG['epsilon']}  |  "
          f"Decay : {ACO_CONFIG['decay_rate']*100:.0f}%")
    print("=" * 80)

    # ── Baseline: pruning (Dijkstra + Pareto dominance) ───────────────────────
    print("\n[Baseline] Running Dijkstra/Pruning...")
    t0 = time.perf_counter()
    try:
        baseline_path, baseline_cost = net.pruning(src, tgt)
    except Exception as exc:
        print(f"  Baseline FAILED: {exc}")
        baseline_path, baseline_cost = [], float('inf')
    baseline_time = time.perf_counter() - t0
    print(f"  Cost : {baseline_cost:.2f}  |  Path length : {len(baseline_path)} nodes"
          f"  |  Time : {baseline_time:.4f}s")

    # ── ACO ───────────────────────────────────────────────────────────────────
    print("\n[ACO] Running Ant Colony Optimization...")
    t0 = time.perf_counter()
    colony = Colony(net, src, tgt)
    result = colony.run()
    aco_time = time.perf_counter() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    aco_cost = result['best_cost']

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\n  Best path cost      : {aco_cost:.2f}")
    print(f"  Path length         : {len(result['best_path'])} nodes")
    print(f"  Iterations          : {result['iteration_count']}")
    print(f"  Final time_max      : {result['final_time_max']:.1f}")
    print(f"  Execution time      : {aco_time:.2f}s")

    print("\n" + "=" * 80)
    print("COMPARISON: ACO vs Dijkstra/Pruning Baseline")
    print("=" * 80)
    print(f"\n  Dijkstra/Pruning : cost={baseline_cost:.2f},  time={baseline_time:.4f}s")
    print(f"  ACO              : cost={aco_cost:.2f},  time={aco_time:.2f}s")

    if (baseline_cost not in (float('inf'), 0)
            and aco_cost != float('inf')):
        diff = baseline_cost - aco_cost
        pct  = 100.0 * diff / baseline_cost
        if diff > 0:
            print(f"\n  Improvement       : {pct:.1f}% better path found by ACO")
            print(f"  Trade-off         : ACO is {aco_time/max(baseline_time, 1e-9):.1f}× slower")
        elif diff < 0:
            print(f"\n  Difference        : ACO found a {-pct:.1f}% worse path")
            print(f"  Note              : increase max_iterations or num_ants for better exploration")
        else:
            print("\n  Both approaches found paths of equal cost.")
    elif aco_cost == float('inf'):
        print("\n  ACO did not find any valid path within the constraints.")

    print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    compare_aco_vs_baseline('examples/large-smallfatigue.txt')
