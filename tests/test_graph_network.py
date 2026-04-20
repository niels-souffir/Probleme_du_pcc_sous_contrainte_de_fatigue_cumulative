import math
from typing import Union

import pytest

from graph import Graph, GraphImplicit, Edge
from network import Network


def test_add_edge() -> None:
    g = Graph(2)
    g.add_edge(0, 1, 3.5)
    assert len(g.adj[0]) == 1
    assert g.adj[0][0].to == 1
    assert g.adj[0][0].weight == 3.5


def test_negative_weight_raises() -> None:
    g = Graph(2)
    try:
        g.add_edge(0, 1, -1)
    except ValueError:
        pass
    else:
        assert False, "expected ValueError when adding negative weight"

def test_shortest_path_single_edge() -> None:
    """Test shortest path with a single direct edge."""
    g = Graph(2)
    g.add_edge(0, 1, 7.5)
    
    path, dist = g.shortest_path(0, 1)
    
    assert path == [0, 1]
    assert dist == pytest.approx(7.5)


def test_unreachable_returns_inf() -> None:
    g = Graph(2)
    g.add_edge(0, 1, 3.0)

    path, dist = g.shortest_path(1, 0)

    assert path == []
    assert math.isinf(dist)


# Test: build_simple_graph creates correct graph structure
def test_build_simple_graph() -> None:
    # Using a test file with structure:
    # Header: num_edges source_node goal_node
    # Edges: node1 node2 distance fatigue
    
    net = Network(filename="examples/small.txt")
    g = net.build_simple_graph()
    
    path, dist = g.shortest_path(net.start, net.goal)
    
    # Verifies that the graph was built correctly
    # and shortest path can be computed
    assert len(path) >= 2
    assert path[0] == net.start
    assert path[-1] == net.goal
    assert dist < float("inf")


def test_compute_F_max() -> None:
    """
    Test F_max computation.
    
    F_max = upper bound on tiredness for any shortest path.
    Formula: F_max = max(1, max_fatigue_coefficient * n)
    
    This assumes the worst case where you visit every node once,
    and each edge has the maximum fatigue coefficient.
    """
    net = Network(filename="examples/small.txt")
    
    # Before calling compute_F_max, F_max should not exist
    assert not hasattr(net, 'F_max')
    
    # Call compute_F_max
    net.compute_F_max()
    
    # After calling, F_max should be set
    assert hasattr(net, 'F_max')
    
    # For small.txt: n=4, max_fatigue=2 (from edges)
    # Expected: F_max = max(1, 2 * 4) = 8
    assert net.n == 4
    assert net.F_max == 8
    
    # F_max should always be at least 1
    assert net.F_max >= 1
    
    # F_max should be a reasonable upper bound (not too large)
    # In practice, F_max <= n * max_possible_fatigue
    # For most graphs, this should be less than n^2
    assert net.F_max <= net.n * 100  # Sanity check


def test_encode_mapping_simple() -> None:
    """Test the encoding function (v, F) -> integer for extended graph states."""
    # Load a network and compute F_max
    net = Network(filename="examples/small.txt")
    net.compute_F_max()
    
    # For small.txt: n=4 (nodes 0-3), F_max=8
    # Encoding formula: v * (F_max + 1) + F
    
    # Test node 0 with different fatigue levels
    assert net._encode(0, 0) == 0
    assert net._encode(0, 1) == 1
    assert net._encode(0, 5) == 5
    
    # Test node 1 with different fatigue levels
    # (1, 0) -> 1 * (F_max + 1) + 0
    expected_node1_f0 = 1 * (net.F_max + 1)
    assert net._encode(1, 0) == expected_node1_f0
    
    # (1, 3) -> 1 * (F_max + 1) + 3
    expected_node1_f3 = 1 * (net.F_max + 1) + 3
    assert net._encode(1, 3) == expected_node1_f3
    
    # Test node 2 with F=7 (within valid range)
    # (2, 7) -> 2 * (F_max + 1) + 7
    expected_node2_f7 = 2 * (net.F_max + 1) + 7
    assert net._encode(2, 7) == expected_node2_f7
    
    # Test last node with max fatigue
    # (3, 8) -> 3 * (F_max + 1) + 8 (3 is last node, 8 is F_max)
    expected_node3_f8 = 3 * (net.F_max + 1) + 8
    assert net._encode(3, 8) == expected_node3_f8


def test_build_simple_graph_implicit() -> None:
    """
    Test building implicit graph without tiredness.
    
    build_simple_graph_implicit should:
    - Return a GraphImplicit instance
    - Have working neighbor function that returns edges
    - Be able to find shortest paths
    - Give same results as build_simple_graph (explicit version)
    """
    net = Network(filename="examples/small.txt")
    
    # Build implicit graph
    g_implicit = net.build_simple_graph_implicit()
    
    # Should return GraphImplicit instance
    assert isinstance(g_implicit, GraphImplicit)
    
    # Should have correct number of nodes
    assert g_implicit.n == net.n == 4
    
    # Test neighbor function works
    # From small.txt: node 2 connects to nodes 0 and 1
    neighbors_node2 = list(g_implicit._neighbors(2))
    assert len(neighbors_node2) == 2
    
    # Check neighbors are Edge objects with correct weights
    neighbor_tos = sorted([e.to for e in neighbors_node2])
    assert neighbor_tos == [0, 1]  # Node 2 connects to 0 and 1
    
    # Check weights are correct (distances only, no fatigue)
    for edge in neighbors_node2:
        assert isinstance(edge, Edge)
        assert edge.weight > 0
    
    # Node with no outgoing edges should return empty list
    neighbors_node3 = list(g_implicit._neighbors(3))
    assert len(neighbors_node3) == 0  # Node 3 (saclay) is the goal
    
    # Test that it can find shortest path
    path, dist = g_implicit.shortest_path(net.start, net.goal)
    
    assert len(path) >= 2
    assert path[0] == net.start
    assert path[-1] == net.goal
    assert dist < float("inf")
    
    # Compare with explicit graph version (should give same result)
    g_explicit = net.build_simple_graph()
    path_explicit, dist_explicit = g_explicit.shortest_path(net.start, net.goal)
    
    assert dist == pytest.approx(dist_explicit)
    assert path == path_explicit


def test_build_extended_graph() -> None:
    """
    Test building extended graph with tiredness states.
    
    build_extended_graph should:
    - Create n * (F_max + 1) nodes for all (node, tiredness) combinations
    - Create edges with weights = length * current_tiredness
    - Handle fatigue accumulation: F_new = F + fatigue_coefficient
    - Allow finding paths in extended state space
    """
    net = Network(filename="examples/small.txt")
    
    # Build extended graph (automatically calls compute_F_max)
    g_ext = net.build_extended_graph()
    
    # Verify correct number of nodes
    # For small.txt: n=4, F_max=8, so total nodes = 4 * 9 = 36
    expected_nodes = net.n * (net.F_max + 1)
    assert g_ext.n == expected_nodes == 36
    
    # Verify state encoding works
    start_state = net._encode(net.start, 1)  # Start with F=1
    assert 0 <= start_state < g_ext.n
    
    # Check that edges exist from start state
    assert len(g_ext.adj[start_state]) > 0
    
    # Test a specific edge to verify weight calculation
    # From small.txt: edge (2,0) has length=10, fatigue=2
    # State (2, F=1) should have edge to (0, F=3) with weight = 10 * 1 = 10
    state_2_f1 = net._encode(2, 1)
    state_0_f3 = net._encode(0, 3)
    
    # Find edge from state_2_f1 to state_0_f3
    edges_from_2_f1 = g_ext.adj[state_2_f1]
    edge_to_0_f3 = None
    for edge in edges_from_2_f1:
        if edge.to == state_0_f3:
            edge_to_0_f3 = edge
            break
    
    assert edge_to_0_f3 is not None, "Expected edge from (2,F=1) to (0,F=3)"
    assert edge_to_0_f3.weight == pytest.approx(10.0)  # 10 * 1
    
    # Test another edge with different fatigue
    # State (2, F=2) should have edge to (0, F=4) with weight = 10 * 2 = 20
    state_2_f2 = net._encode(2, 2)
    state_0_f4 = net._encode(0, 4)
    
    edges_from_2_f2 = g_ext.adj[state_2_f2]
    edge_to_0_f4 = None
    for edge in edges_from_2_f2:
        if edge.to == state_0_f4:
            edge_to_0_f4 = edge
            break
    
    assert edge_to_0_f4 is not None, "Expected edge from (2,F=2) to (0,F=4)"
    assert edge_to_0_f4.weight == pytest.approx(20.0)  # 10 * 2
    
    # Test that we can find a path in the extended graph
    # Try to find path from start state (F=1) to any goal state
    # We'll try a few goal states and find the best one
    best_dist = float("inf")
    best_path = []
    
    for F_goal in range(1, min(6, net.F_max + 1)):  # Try first few F values
        goal_state = net._encode(net.goal, F_goal)
        path, dist = g_ext.shortest_path(start_state, goal_state)
        if dist < best_dist:
            best_dist = dist
            best_path = path
    
    # Should find at least one valid path
    assert len(best_path) >= 2
    assert best_path[0] == start_state
    assert best_dist < float("inf")


def test_build_extended_implicit_graph_without_virtual_goal() -> None:
    """
    Test building extended implicit graph WITHOUT virtual goal.
    
    build_extended_implicit_graph(with_virtual_goal=False) should:
    - Return GraphImplicit with n*(F_max+1) nodes
    - Have neighbor function that properly decodes states
    - Update tiredness correctly: F_new = F + fatigue_coefficient
    - Calculate edge weights as: weight = length × current_tiredness
    - Be able to find paths in extended state space
    """
    net = Network(filename="examples/small.txt")
    
    # Build extended implicit graph WITHOUT virtual goal
    g_impl = net.build_extended_implicit_graph(with_virtual_goal=False)
    
    # Should return GraphImplicit instance
    assert isinstance(g_impl, GraphImplicit)
    
    # Should have n * (F_max + 1) nodes (NO extra virtual goal)
    # For small.txt: n=4, F_max=8, so total = 4 * 9 = 36
    expected_nodes = net.n * (net.F_max + 1)
    assert g_impl.n == expected_nodes == 36
    
    # Test neighbor function with a specific state
    # State (2, F=1): node 2 with tiredness 1
    state_2_f1 = net._encode(2, 1)
    neighbors = list(g_impl._neighbors(state_2_f1))
    
    # Node 2 has edges to nodes 0 and 1 in small.txt
    assert len(neighbors) >= 2
    
    # All neighbors should be Edge objects
    for neighbor in neighbors:
        assert isinstance(neighbor, Edge)
        assert neighbor.weight >= 0
    
    # Test specific edge: (2,0) with length=10, fatigue=2
    # From state (2, F=1) → (0, F=3) with weight = 10 * 1 = 10
    state_0_f3 = net._encode(0, 3)
    edge_to_0_f3 = None
    for edge in neighbors:
        if edge.to == state_0_f3:
            edge_to_0_f3 = edge
            break
    
    assert edge_to_0_f3 is not None, "Expected edge from (2,F=1) to (0,F=3)"
    assert edge_to_0_f3.weight == pytest.approx(10.0)  # length * F = 10 * 1
    
    # Test tiredness accumulation with different starting F
    # State (2, F=3): node 2 with tiredness 3
    state_2_f3 = net._encode(2, 3)
    neighbors_f3 = list(g_impl._neighbors(state_2_f3))
    
    # Edge to (0, F=5) should have weight = 10 * 3 = 30
    state_0_f5 = net._encode(0, 5)
    edge_to_0_f5 = None
    for edge in neighbors_f3:
        if edge.to == state_0_f5:
            edge_to_0_f5 = edge
            break
    
    assert edge_to_0_f5 is not None, "Expected edge from (2,F=3) to (0,F=5)"
    assert edge_to_0_f5.weight == pytest.approx(30.0)  # length * F = 10 * 3
    
    # Verify no virtual goal node exists
    # The last valid state should be (n-1, F_max)
    last_state = net._encode(net.n - 1, net.F_max)
    assert last_state < g_impl.n
    
    # There should be no node at index n*(F_max+1) (where virtual goal would be)
    virtual_goal_would_be = net.n * (net.F_max + 1)
    assert virtual_goal_would_be == g_impl.n  # Equal means it's beyond the last valid index
    
    # Test path finding in extended state space
    start_state = net._encode(net.start, 1)  # Start with F=1
    
    # Try finding path to goal with specific tiredness level
    goal_state_f2 = net._encode(net.goal, 2)
    path, dist = g_impl.shortest_path(start_state, goal_state_f2)
    
    # If path exists, verify it's valid
    if len(path) >= 2:
        assert path[0] == start_state
        assert path[-1] == goal_state_f2
        assert dist < float("inf")
    
    # Goal states should NOT have edges to any virtual goal
    goal_state_f1 = net._encode(net.goal, 1)
    goal_neighbors = list(g_impl._neighbors(goal_state_f1))
    
    # All neighbors should be within the valid state range
    for edge in goal_neighbors:
        assert edge.to < g_impl.n


def test_build_extended_implicit_graph_with_virtual_goal() -> None:
    """
    Test building extended implicit graph WITH virtual goal optimization.
    
    build_extended_implicit_graph(with_virtual_goal=True) should:
    - Return GraphImplicit with n*(F_max+1)+1 nodes (extra for virtual goal)
    - Virtual goal is the last node at index n*(F_max+1)
    - All (goal, F) states connect to virtual goal with cost 0
    - Virtual goal has NO outgoing edges
    - Allows finding optimal path to goal in single Dijkstra run
    """
    net = Network(filename="examples/small.txt")
    
    # Build extended implicit graph WITH virtual goal
    g_impl = net.build_extended_implicit_graph(with_virtual_goal=True)
    
    # Should return GraphImplicit instance
    assert isinstance(g_impl, GraphImplicit)
    
    # Should have n * (F_max + 1) + 1 nodes (EXTRA node for virtual goal)
    # For small.txt: n=4, F_max=8, so total = 4 * 9 + 1 = 37
    expected_nodes = net.n * (net.F_max + 1) + 1
    assert g_impl.n == expected_nodes == 37
    
    # Virtual goal should be the last node
    virtual_goal_id = net.n * (net.F_max + 1)
    assert virtual_goal_id == 36  # n=4, F_max=8 → 4*9=36
    
    # Virtual goal should have NO outgoing edges
    vg_neighbors = list(g_impl._neighbors(virtual_goal_id))
    assert len(vg_neighbors) == 0, "Virtual goal should have no outgoing edges"
    
    # Test that ALL (goal, F) states connect to virtual goal with cost 0
    for F in range(1, min(5, net.F_max + 1)):  # Test several F values
        goal_state = net._encode(net.goal, F)
        neighbors = list(g_impl._neighbors(goal_state))
        
        # Find edge to virtual goal
        vg_edges = [e for e in neighbors if e.to == virtual_goal_id]
        
        assert len(vg_edges) == 1, f"Goal state ({net.goal}, F={F}) should have exactly one edge to virtual goal"
        assert vg_edges[0].weight == 0, "Edge to virtual goal should have cost 0"
    
    # Non-goal states should NOT have direct edges to virtual goal
    non_goal_state = net._encode(1, 2)  # Node 1 (not goal) with F=2
    if non_goal_state != net._encode(net.goal, 2):  # Make sure it's not accidentally a goal state
        non_goal_neighbors = list(g_impl._neighbors(non_goal_state))
        vg_edges_non_goal = [e for e in non_goal_neighbors if e.to == virtual_goal_id]
        assert len(vg_edges_non_goal) == 0, "Non-goal states should not connect to virtual goal"
    
    # Test path finding with virtual goal
    # Should be able to find optimal path to virtual goal in ONE Dijkstra run
    start_state = net._encode(net.start, 1)  # Start with F=1
    
    path, dist = g_impl.shortest_path(start_state, virtual_goal_id)
    
    # Should find a valid path
    assert len(path) >= 2, "Should find path from start to virtual goal"
    assert path[0] == start_state, "Path should start at start state"
    assert path[-1] == virtual_goal_id, "Path should end at virtual goal"
    assert dist < float("inf"), "Distance should be finite"
    
    # Decode path to verify it reaches the actual goal node
    # (excluding last node which is the virtual goal)
    decoded_path = [node // (net.F_max + 1) for node in path[:-1]]
    assert decoded_path[-1] == net.goal, "Path should reach the actual goal node before virtual goal"
    
    # Compare with non-virtual-goal version
    # Both should give same optimal distance
    g_impl_no_vg = net.build_extended_implicit_graph(with_virtual_goal=False)
    
    # Try several goal states with non-virtual version
    best_dist_no_vg = float("inf")
    for F_goal in range(1, net.F_max + 1):
        goal_state = net._encode(net.goal, F_goal)
        _, dist_no_vg = g_impl_no_vg.shortest_path(start_state, goal_state)
        if dist_no_vg < best_dist_no_vg:
            best_dist_no_vg = dist_no_vg
    
    # Virtual goal optimization should give same result as brute force
    assert dist == pytest.approx(best_dist_no_vg), "Virtual goal optimization should give same optimal distance"


def test_graphimplicit_neighbors_mixed_types_and_negative_weight() -> None:
    """
    Test GraphImplicit._neighbors() with mixed Edge/tuple types and negative weight validation.
    
    _neighbors() should:
    - Accept neighbor functions that return Edge objects
    - Accept neighbor functions that return (v, weight) tuples
    - Convert tuples to Edge objects
    - Raise ValueError for negative weights in tuples
    - Pass through Edge objects without validation (validation happens elsewhere)
    """
    # Test 1: Mixed types - Edge and tuple
    def mixed_neighbor_fn(u: int) -> list[Union[Edge, tuple[int, float]]]:
        if u == 0:
            return [Edge(to=1, weight=5.0), (2, 3.0)]  # Mix of Edge and tuple
        elif u == 1:
            return [(2, 1.5)]  # Only tuple
        elif u == 2:
            return [Edge(to=3, weight=2.0)]  # Only Edge
        return []
    
    g = GraphImplicit(n=4, neighbor_fn=mixed_neighbor_fn)
    
    # Check neighbors of node 0 - should get 2 edges
    neighbors_0 = list(g._neighbors(0))
    assert len(neighbors_0) == 2
    assert all(isinstance(e, Edge) for e in neighbors_0)
    assert neighbors_0[0].to == 1 and neighbors_0[0].weight == 5.0
    assert neighbors_0[1].to == 2 and neighbors_0[1].weight == 3.0
    
    # Check neighbors of node 1 - tuple converted to Edge
    neighbors_1 = list(g._neighbors(1))
    assert len(neighbors_1) == 1
    assert isinstance(neighbors_1[0], Edge)
    assert neighbors_1[0].to == 2 and neighbors_1[0].weight == 1.5
    
    # Check neighbors of node 2 - Edge passed through
    neighbors_2 = list(g._neighbors(2))
    assert len(neighbors_2) == 1
    assert isinstance(neighbors_2[0], Edge)
    assert neighbors_2[0].to == 3 and neighbors_2[0].weight == 2.0
    
    # Test 2: Negative weight in tuple should raise ValueError
    def negative_tuple_fn(u: int) -> list[tuple[int, float]]:
        if u == 0:
            return [(1, -5.0)]  # Negative weight
        return []
    
    g_neg = GraphImplicit(n=2, neighbor_fn=negative_tuple_fn)
    
    with pytest.raises(ValueError, match="négatifs"):
        # Force evaluation of the generator
        list(g_neg._neighbors(0))
    
    # Test 3: Edge objects are passed through without validation
    # _neighbors() does NOT validate weights in Edge objects
    def edge_with_various_weights_fn(u: int) -> list[Edge]:
        if u == 0:
            return [Edge(to=1, weight=10.0), Edge(to=2, weight=0.0)]
        return []
    
    g_edges = GraphImplicit(n=3, neighbor_fn=edge_with_various_weights_fn)
    
    neighbors_edges = list(g_edges._neighbors(0))
    assert len(neighbors_edges) == 2
    assert neighbors_edges[0].to == 1 and neighbors_edges[0].weight == 10.0
    assert neighbors_edges[1].to == 2 and neighbors_edges[1].weight == 0.0
    
    # Test 4: Tuple weight conversion to float
    def int_weight_fn(u: int) -> list[tuple[int, int]]:
        if u == 0:
            return [(1, 7)]  # Integer weight
        return []
    
    g_int = GraphImplicit(n=2, neighbor_fn=int_weight_fn)
    
    neighbors_int = list(g_int._neighbors(0))
    assert len(neighbors_int) == 1
    assert neighbors_int[0].to == 1
    assert isinstance(neighbors_int[0].weight, float)
    assert neighbors_int[0].weight == 7.0


def test_graphimplicit_shortest_path_on_complex_example() -> None:
    r"""
    Test GraphImplicit.shortest_path() on a complex graph with multiple paths.
    
    Graph structure:
         1 ---2--- 3
        /|        /|
       1 |       3 |
      /  3      /  1
     0   |     /   |
      \  |    /    |
       4 |   5     |
        \|  /      |
         2 ---1--- 4
    
    Edges:
    0 -> 1 (weight 1)
    0 -> 2 (weight 4)
    1 -> 2 (weight 3)
    1 -> 3 (weight 2)
    2 -> 3 (weight 5)
    2 -> 4 (weight 1)
    3 -> 4 (weight 1)
    
    Shortest path from 0 to 4:
    - Option 1: 0->1->2->4 = 1+3+1 = 5
    - Option 2: 0->2->4 = 4+1 = 5
    - Option 3: 0->1->3->4 = 1+2+1 = 4 ✓ (best)
    """
    def complex_neighbor_fn(u: int) -> list[tuple[int, float]]:
        edges = {
            0: [(1, 1.0), (2, 4.0)],
            1: [(2, 3.0), (3, 2.0)],
            2: [(3, 5.0), (4, 1.0)],
            3: [(4, 1.0)],
            4: []
        }
        return edges.get(u, [])
    
    g = GraphImplicit(n=5, neighbor_fn=complex_neighbor_fn)
    
    # Test 1: Shortest path from 0 to 4
    path, dist = g.shortest_path(0, 4)
    
    assert len(path) == 4, f"Expected path length 4, got {len(path)}"
    assert path == [0, 1, 3, 4], f"Expected path [0, 1, 3, 4], got {path}"
    assert dist == 4.0, f"Expected distance 4.0, got {dist}"
    
    # Test 2: Shortest path from 0 to 3
    path_03, dist_03 = g.shortest_path(0, 3)
    
    assert len(path_03) == 3
    assert path_03 == [0, 1, 3]
    assert dist_03 == 3.0  # 0->1 (1) + 1->3 (2) = 3
    
    # Test 3: Shortest path from 1 to 4
    path_14, dist_14 = g.shortest_path(1, 4)
    
    assert len(path_14) == 3
    assert path_14 == [1, 3, 4]
    assert dist_14 == 3.0  # 1->3 (2) + 3->4 (1) = 3
    
    # Test 4: Path from node to itself
    path_self, dist_self = g.shortest_path(2, 2)
    
    assert len(path_self) == 1
    assert path_self == [2]
    assert dist_self == 0.0
    
    # Test 5: Unreachable node (no path from 4 back to 0)
    path_unreachable, dist_unreachable = g.shortest_path(4, 0)
    
    assert path_unreachable == []
    assert dist_unreachable == float("inf")
    
    # Test 6: Using Edge objects instead of tuples
    def edge_neighbor_fn(u: int) -> list[Edge]:
        edges_map = {
            0: [Edge(to=1, weight=1.0), Edge(to=2, weight=4.0)],
            1: [Edge(to=2, weight=3.0), Edge(to=3, weight=2.0)],
            2: [Edge(to=3, weight=5.0), Edge(to=4, weight=1.0)],
            3: [Edge(to=4, weight=1.0)],
            4: []
        }
        return edges_map.get(u, [])
    
    g_edge = GraphImplicit(n=5, neighbor_fn=edge_neighbor_fn)
    
    path_edge, dist_edge = g_edge.shortest_path(0, 4)
    
    # Should get same result with Edge objects
    assert path_edge == [0, 1, 3, 4]
    assert dist_edge == 4.0
    
    # Test 7: Graph with zero-weight edges
    def zero_weight_fn(u: int) -> list[tuple[int, float]]:
        if u == 0:
            return [(1, 0.0), (2, 5.0)]
        elif u == 1:
            return [(2, 1.0)]
        return []
    
    g_zero = GraphImplicit(n=3, neighbor_fn=zero_weight_fn)
    
    path_zero, dist_zero = g_zero.shortest_path(0, 2)
    
    assert path_zero == [0, 1, 2]
    assert dist_zero == 1.0  # 0->1 (0) + 1->2 (1) = 1
    
    # Test 8: Verify early termination when goal is reached
    # (Dijkstra should stop when goal is popped from priority queue)
    call_count = {"count": 0}
    
    def counting_neighbor_fn(u: int) -> list[tuple[int, float]]:
        call_count["count"] += 1
        return complex_neighbor_fn(u)
    
    g_count = GraphImplicit(n=5, neighbor_fn=counting_neighbor_fn)
    
    # Reset counter and find path
    call_count["count"] = 0
    path_term, dist_term = g_count.shortest_path(0, 1)
    
    # Should find direct path 0->1 immediately
    assert path_term == [0, 1]
    assert dist_term == 1.0
    # Should have called neighbor_fn for node 0 and possibly node 1
    # but not explored all nodes
    assert call_count["count"] <= 3, f"Too many neighbor calls: {call_count['count']}"