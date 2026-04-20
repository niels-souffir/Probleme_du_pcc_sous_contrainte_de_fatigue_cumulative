"""
Module de benchmark pour comparer les performances (temps, mémoire, distance)
des différents algorithmes de recherche de chemin sous contrainte de fatigue.

Évalue les variantes de Dijkstra, Pruning et A* sur des réseaux de tailles
variées et exporte les résultats détaillés en console et en fichiers CSV.
"""

import os
import time
import tracemalloc
import pandas as pd
from network import Network

# List of test files
TEST_FILES = [
    "examples/small.txt",
    "examples/medium-nofatigue.txt",
    "examples/medium-smallfatigue.txt",
    "examples/medium-largefatigue.txt",
    "examples/large-nofatigue.txt",
    "examples/large-smallfatigue.txt",
    "examples/large-largefatigue.txt",
]

# Extract friendly filenames
FILE_LABELS = [os.path.basename(f).replace(".txt", "") for f in TEST_FILES]

# Number of repetitions for averaging
NUM_RUNS = 1

def benchmark_dijkstra_simple(net):
    """Dijkstra on simple graph (no fatigue)"""
    tracemalloc.start()
    g = net.build_simple_graph()
    path, dist = g.shortest_path(net.start, net.goal)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return path, dist, peak / 1024 / 1024

def benchmark_dijkstra_extended(net):
    """Dijkstra on extended explicit graph"""
    tracemalloc.start()
    g = net.build_extended_graph()
    enc_goal = g.n - 1  # Virtual goal
    path_enc, dist = g.shortest_path(net._encode(net.start, 1), enc_goal)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if path_enc:
        path = [enc // (net.F_max + 1) for enc in path_enc[:-1]]
    else:
        path = []
    return path, dist, peak / 1024 / 1024

def benchmark_dijkstra_implicit(net):
    """Dijkstra on extended implicit graph"""
    tracemalloc.start()
    g = net.build_extended_implicit_graph()
    enc_goal = g.n - 1
    path_enc, dist = g.shortest_path(net._encode(net.start, 1), enc_goal)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if path_enc:
        path = [enc // (net.F_max + 1) for enc in path_enc[:-1]]
    else:
        path = []
    return path, dist, peak / 1024 / 1024

def benchmark_pruning(net):
    """Pruning standard"""
    tracemalloc.start()
    path, dist = net.pruning(net.start, net.goal)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return path, dist, peak / 1024 / 1024

def benchmark_pruning_implicit(net):
    """Pruning with implicit graph"""
    tracemalloc.start()
    path, dist = net.pruning_implicit(net.start, net.goal)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return path, dist, peak / 1024 / 1024

def benchmark_astar(net):
    """A* standard"""
    tracemalloc.start()
    path, dist = net.astar(net.start, net.goal)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return path, dist, peak / 1024 / 1024

def benchmark_astar_implicit(net):
    """A* with implicit graph"""
    tracemalloc.start()
    path, dist = net.astar_implicit(net.start, net.goal)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return path, dist, peak / 1024 / 1024

# Map of algorithm names to benchmark functions
ALGORITHMS = {
    "Dijkstra Simple": benchmark_dijkstra_simple,
    "Dijkstra Extended": benchmark_dijkstra_extended,
    "Dijkstra Implicit": benchmark_dijkstra_implicit,
    "Pruning": benchmark_pruning,
    "Pruning Implicit": benchmark_pruning_implicit,
    "A*": benchmark_astar,
    "A* Implicit": benchmark_astar_implicit,
}

def run_single_benchmark(file_path, algo_name, algo_func):
    """Run a single benchmark and return (time_ms, peak_memory_mb, distance)"""
    try:
        net = Network(file_path)
        
        times = []
        memory_peak = 0
        distance = 0
        
        for _ in range(NUM_RUNS):
            t_start = time.perf_counter()
            path, dist, peak_mem = algo_func(net)
            t_end = time.perf_counter()
            
            times.append((t_end - t_start) * 1000)
            memory_peak = max(memory_peak, peak_mem)
            distance = dist
        
        avg_time = sum(times) / len(times)
        return avg_time, memory_peak, distance
    
    except Exception as e:
        print(f"  ERROR in {algo_name}: {str(e)}")
        return None, None, None

def run_benchmarks():
    """Run all benchmarks and return results as DataFrames"""
    
    results_time = {}
    results_memory = {}
    results_distance = {}
    
    # Allowed files for Dijkstra Extended
    dijkstra_extended_allowed = {"small", "medium-nofatigue", "medium-smallfatigue"}
    
    # Initialize result dicts
    for algo_name in ALGORITHMS:
        results_time[algo_name] = {}
        results_memory[algo_name] = {}
        results_distance[algo_name] = {}
    
    # Run benchmarks
    for file_idx, file_path in enumerate(TEST_FILES):
        label = FILE_LABELS[file_idx]
        print(f"\n{'='*60}")
        print(f"Testing file: {label}")
        print(f"{'='*60}")
        
        for algo_name, algo_func in ALGORITHMS.items():
            # Skip Dijkstra Extended for files not in the allowed list
            if algo_name == "Dijkstra Extended" and label not in dijkstra_extended_allowed:
                continue
            # Dijkstra Implicit for large-largefatigue
            if algo_name == "Dijkstra Implicit" and label == "large-largefatigue":
                continue
            # For large-largefatigue, only run Pruning
            if label == "large-largefatigue" and algo_name != "Pruning":
                continue
            
            print(f"  {algo_name}...", end=" ", flush=True)
            avg_time, peak_mem, distance = run_single_benchmark(file_path, algo_name, algo_func)
            
            if avg_time is not None:
                results_time[algo_name][label] = round(avg_time, 2)
                results_memory[algo_name][label] = round(peak_mem, 2)
                results_distance[algo_name][label] = round(distance, 1)
                print(f"✓ {avg_time:.2f}ms, {peak_mem:.2f}MB, dist={distance:.1f}")
            else:
                results_time[algo_name][label] = "ERROR"
                results_memory[algo_name][label] = "ERROR"
                results_distance[algo_name][label] = "ERROR"
                print(f"✗ ERROR")
    
    # Convert to DataFrames
    df_time = pd.DataFrame(results_time).T
    df_memory = pd.DataFrame(results_memory).T
    df_distance = pd.DataFrame(results_distance).T
    
    return df_time, df_memory, df_distance

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BENCHMARK: Graph Routing Algorithms")
    print("=" * 80)
    print(f"Files: {len(TEST_FILES)} | Algorithms: {len(ALGORITHMS)} | Runs: {NUM_RUNS}")
    
    df_time, df_memory, df_distance = run_benchmarks()
    
    print("\n" + "=" * 80)
    print("RESULTS: RUNNING TIME (milliseconds)")
    print("=" * 80)
    print(df_time.to_string())
    
    print("\n" + "=" * 80)
    print("RESULTS: PEAK MEMORY (megabytes)")
    print("=" * 80)
    print(df_memory.to_string())
    
    print("\n" + "=" * 80)
    print("RESULTS: SHORTEST DISTANCE")
    print("=" * 80)
    print(df_distance.to_string())
    
    # Export to CSV
    df_time.to_csv("benchmark_time.csv")
    df_memory.to_csv("benchmark_memory.csv")
    df_distance.to_csv("benchmark_distance.csv")
    print("\n✓ CSV files saved: benchmark_time.csv, benchmark_memory.csv, benchmark_distance.csv")
