import os
from network import Network
from ants import Colony, load_aco_scenario


def _compute_memory_threshold() -> int:
    """Estime le nombre maximal de nœuds du graphe étendu pouvant être stockés en RAM sans risquer de saturation.
    
    Cette estimation se base sur une allocation prudente de 60% de la RAM physique totale, en tenant compte de l'empreinte mémoire moyenne d'un 
    objet Python dans une file de priorité ou un dictionnaire.

    Retours
    -------
    int
        Le nombre seuil de nœuds au-delà duquel l'algorithme doit basculer sur une stratégie plus économe ou s'arrêter.
    """
    try:
        # Récupère la RAM physique totale en octets (spécifique aux systèmes POSIX/Linux)
        phys_bytes = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, ValueError):
        # Valeur par défaut pour Windows ou les systèmes non-POSIX (environ 2 millions de nœuds)
        return 2_000_000  
    # Estimation empirique de l'occupation d'un état (nœud + fatigue) en Python.
    # Entre les dictionnaires 'parent', 'distances' et le tas (heapq), un état consomme environ 200 octets de RAM.
    bytes_per_node = 200 
    usable = int(phys_bytes * 0.6)
    return max(usable // bytes_per_node, 500_000)

MEMORY_THRESHOLD = _compute_memory_threshold()

def Q_multi(filename: str, missions: list[tuple[int, int]]):
    """Implémente la Partie_3/Question_1 du projet.
    
    Missions multiples dans un ordre imposé.
    Entre deux missions consécutives, un déplacement intermédiaire est inséré automatiquement si les points d'arrivée/départ ne coïncident pas.

    Paramètres
    ----------
    filename : str
        Chemin du fichier réseau (ex. "examples/small.txt").
    missions : list[tuple[int, int]]
        Liste ordonnée de missions [(vs1, vt1), ...] en IDs entiers.

    Retours
    -------
    tuple[list[int], float, int]
        (chemin_complet, temps_total, fatigue_finale)
    """
    net = Network(filename=filename)
    path, total_time, final_fatigue = net.multi_mission_path(missions)
    return path, total_time, final_fatigue

def Q1(filename):
    """Implémente la Partie_1/Question_1 du projet.
    
    Paramètres
    ----------
    filename : str
        Le chemin vers le fichier texte contenant les données du réseau.

    """
    # Si la fatigue cumulative reste nulle, le problème initiale revient à celui du plus court chemin.
    net = Network(filename=filename)
    g = net.build_simple_graph()
    path, distance = g.shortest_path(net.start, net.goal)
    return path, distance

def Q2(filename):
    """Implémente la Partie_1/Question_2 du projet.
    
    Paramètres
    ----------
    filename : str
        Le chemin vers le fichier texte contenant les données du réseau.

    """
    # Load network from file
    net = Network(filename=filename)
    
    # Compute F_max and build extended graph (with tiredness + virtual goal)
    net.compute_F_max()
    print(f"F_max: {net.F_max}, n: {net.n}")
    g_ext = net.build_extended_graph()
    
    # Start state: (start_node, tiredness=1)
    enc_start = net._encode(net.start, 1)
    print(f"Start: node {net.start}, encoded: {enc_start}")
    
    # Virtual goal has ID = n * (F_max + 1)
    virtual_goal = net.n * (net.F_max + 1)
    
    # Single call to shortest_path to find best path to goal with any tiredness level
    path_enc, best_dist = g_ext.shortest_path(enc_start, virtual_goal)
    
    # Decode: convert extended nodes back to original nodes
    # (exclude the virtual goal node at the end)
    if path_enc and len(path_enc) > 1:
        path = [node // (net.F_max + 1) for node in path_enc[:-1]]  # Remove virtual goal
    else:
        path = []
    return path, best_dist

def Q3(filename):
    """Implémente la Partie_1/Question_3 du projet.
    
    Paramètres
    ----------
    filename : str
        Le chemin vers le fichier texte contenant les données du réseau.

    """
    # Load network from file
    net = Network(filename=filename)
    
    # Build extended implicit graph with virtual goal optimization
    net.compute_F_max()
    g_implicit = net.build_extended_implicit_graph()

    # Start state: (start_node, tiredness=1)
    enc_start = net._encode(net.start, 1)
    
    # Virtual goal has ID = n * (F_max + 1)
    virtual_goal = net.n * (net.F_max + 1)
    
    # Run Dijkstra ONCE to the virtual goal
    path_enc, best_dist = g_implicit.shortest_path(enc_start, virtual_goal)
    
    # Decode: convert extended nodes back to original nodes
    # (exclude the virtual goal node at the end)
    if path_enc and len(path_enc) > 1:
        path = [node // (net.F_max + 1) for node in path_enc[:-1]]  # Remove virtual goal
    else:
        path = []
    
    return path, best_dist

def Q_pruning(filename):
    """Implémente la Partie_2/Question_1 du projet.
    
    Paramètres
    ----------
    filename : str
        Le chemin vers le fichier texte contenant les données du réseau.
    """
    net = Network(filename=filename)
    path, best_dist = net.pruning(net.start, net.goal)
    return path, best_dist

def Q_pruning_implicit(filename):
    """Implémente la Partie_2/Question_1 du projet en utilisant le graph implicite.
    
    Paramètres
    ----------
    filename : str
        Le chemin vers le fichier texte contenant les données du réseau.
    """
    
    net = Network(filename=filename)
    path, best_dist = net.pruning_implicit(net.start, net.goal)
    
    return path, best_dist

def Q_astar(filename):
    """Implémente la Partie_2/Question_2 du projet.
    
    Paramètres
    ----------
    filename : str
        Le chemin vers le fichier texte contenant les données du réseau.
    """
    net = Network(filename=filename)
    path, best_dist = net.astar(net.start, net.goal)
    return path, best_dist

def Q_astar_implicit(filename):
    """Implémente la Partie_2/Question_2 du projet en utilisant le graph implicite.
    
    Paramètres
    ----------
    filename : str
        Le chemin vers le fichier texte contenant les données du réseau.
    """
    net = Network(filename=filename)
    path, best_dist = net.astar_implicit(net.start, net.goal)
    return path, best_dist

def Q_multi(filename: str, missions: list[tuple[int, int]]):
    """Implémente la Partie_3/Question_1 du projet.
    
    Missions multiples dans un ordre imposé.
    Entre deux missions consécutives, un déplacement intermédiaire est inséré automatiquement si les points d'arrivée/départ ne coïncident pas.

    Paramètres
    ----------
    filename : str
        Chemin du fichier réseau (ex. "examples/small.txt").
    missions : list[tuple[int, int]]
        Liste ordonnée de missions [(vs1, vt1), ...] en IDs entiers.

    Retours
    -------
    tuple[list[int], float, int]
        (chemin_complet, temps_total, fatigue_finale)
    """
    net = Network(filename=filename)
    path, total_time, final_fatigue = net.multi_mission_path(missions)
    return path, total_time, final_fatigue

def Q_multi_pruning(filename: str, mission_filename: str):
    """Implémente la Partie_3/Question_1 du projet, avec même interface que Q_multi(), mais en utilisant l'élagage par dominance de Pareto au lieu de l'algorithme de Dijkstra.

    Paramètres
    ----------
    filename : str
        Chemin du fichier réseau (ex. "examples/small.txt").
    missions : list[tuple[int, int]]
        Liste ordonnée de missions [(vs1, vt1), ...] en IDs entiers.
    """
    from format_examples import load_missions
    net = Network(filename=filename)
    missions = load_missions(mission_filename)
    path, total_time, final_fatigue = net.multi_mission_path_pruning(missions)
    return path, total_time, final_fatigue

def Q_multi_opti(filename: str, missions: list[tuple[int, int]]):
    """Implémente la Partie_3/Question_1 du projet en choisisant l'ordre optimal des missions."""
    net = Network(filename=filename)
    best_order, best_path, best_time, best_fatigue = net.optimal_mission_order(missions)
    return best_order, best_path, best_time, best_fatigue

def hybrid_aco_then_pruning(filename: str, stagnation_patience: int = 100) -> tuple[list[int], float]:
    """Solveur hybride adaptatif combinant l'ACO et le Pruning de Pareto.

    Stratégie 
    ---------
    1. Si le problème est simple (tient en RAM, ie n * (F_max + 1) ≤ MEMORY_THRESHOLD), lance le Pruning directement.
    2. Sinon : exécutez l'ACO une itération à la fois. Après chaque amélioration du best_cost T, calculez F_bound(T) à partir de la formule de la preuve. 
    Arrêtez dès que n x (F_bound+1)≤MEMORY_THRESHOLD, ou après stagnation_patience itérations consécutives sans amélioration.
    3. Lancez l'élagage exact avec time_limit = best_cost et f_bound = F_bound. 
    Si le graphe étendu est encore trop volumineux (en cas de stagnation), renvoyez la réponse de l'ACO en l'état.          

    Paramètres
    ----------
    filename : str
        Fichier de données du réseau.
    stagnation_patience : int
        Nombre d'itérations ACO sans amélioration avant d'arrêter (défaut: 100).

    Retours
    -------
    tuple[list[int], float]
        - path : Le meilleur chemin trouvé (garanti optimal si le Pruning a pu tourner).
        - cost : Le temps de trajet total.
    """
    net, src, tgt = load_aco_scenario(filename)
    net.compute_F_max()
    initial_state_space = net.n * (net.F_max + 1)

    if initial_state_space <= MEMORY_THRESHOLD:
        print(f"Easy problem (state space = {initial_state_space:,}). Running pruning directly.")
        return net.pruning(src, tgt)

    print(f"Hard problem (state space = {initial_state_space:,} > {MEMORY_THRESHOLD:,}).")
    print("Running ACO warm-start to tighten F_bound...\n")

    num_ants = min(max(50 * net.n, 200), 5000)
    colony = Colony(net, src, tgt, num_ants=num_ants)

    no_improve = 0
    last_best = float('inf')
    f_bound = net.F_max

    print(f"  {'#iter':<8} {'success/ants':<16} {'best cost':<18} {'F_bound':<12} {'state_space'}")
    print("  " + "-" * 70)

    while True:
        stats = colony.run_iteration()
        T = colony.best_cost

        if T < last_best:
            last_best = T
            no_improve = 0
            f_bound = net.compute_F_bound(T)
            state_space = net.n * (f_bound + 1)
            print(f"  #{colony.iteration_count:<7} "
                  f"{stats['successful_ants']}/{num_ants:<13} "
                  f"{T:<18.0f} {f_bound:<12} {state_space:,}")
            if state_space <= MEMORY_THRESHOLD:
                print("\n  → F_bound small enough. Switching to exact pruning.")
                break
        else:
            no_improve += 1
            if no_improve >= stagnation_patience:
                print(f"\n  → Stagnation ({stagnation_patience} iters without improvement).")
                break

    if colony.best_cost == float('inf'):
        print("ACO found no solution.")
        return [], float('inf')

    f_bound = net.compute_F_bound(colony.best_cost)
    state_space = net.n * (f_bound + 1)

    if state_space <= MEMORY_THRESHOLD:
        print(f"Running exact pruning  (F_bound={f_bound}, state_space={state_space:,})")
        return net.pruning(src, tgt, time_limit=colony.best_cost, f_bound=f_bound)

    print(f"Extended graph still too large (state_space={state_space:,}). Returning ACO answer.")
    return colony.best_path, colony.best_cost

def main(filename, 
        Q1_enabled=False, 
        Q2_enabled=False, 
        Q3_enabled=False, 
        Q_pruning_enabled=False,
        Q_hybrid_enabled=False,
        Q_pruning_implicit_enabled=False, 
        Q_astar_enabled=False, 
        Q_astar_implicit_enabled=False, 
        Q_Multi_enabled=False, 
        Q_Multi_pruning_enabled=False, 
        Q_Multi_opti_enabled=False):
    """Point d'entrée principal pour tester les différents algorithmes du projet.

    Établir la valeur True à la question du projet à tester.
    """
    filepath = 'examples/'+filename+'.txt'

    if Q1_enabled:
        path_Q1, dist_Q1 = Q1(filepath)
        print(f"Q1 - Path: {path_Q1}, Distance: {dist_Q1}")
    if Q2_enabled:
        path_Q2, dist_Q2 = Q2(filepath)
        print(f"Q2 - Path: {path_Q2}, Distance: {dist_Q2}")
    if Q3_enabled:
        path_Q3, dist_Q3 = Q3(filepath)
        print(f"Q3 - Path: {path_Q3}, Distance: {dist_Q3}")
    if Q_pruning_enabled:
        path_pruning, dist_pruning = Q_pruning(filepath)
        print(f"Q_pruning - Path: {path_pruning}, Distance: {dist_pruning}")
    if Q_hybrid_enabled:
        path_hybrid, dist_hybrid = hybrid_aco_then_pruning(filepath)
        print(f"Hybrid ACO + Pruning - Path: {path_hybrid}, Distance: {dist_hybrid}")
    if Q_pruning_implicit_enabled:
        path_pruning_implicit, dist_pruning_implicit = Q_pruning_implicit(filepath)
        print(f"Q_pruning_implicit - Path: {path_pruning_implicit}, Distance: {dist_pruning_implicit}")
    if Q_astar_enabled:
        path_astar, dist_astar = Q_astar(filepath)
        print(f"Q_astar - Path: {path_astar}, Distance: {dist_astar}")
    if Q_astar_implicit_enabled:
        path_astar_implicit, dist_astar_implicit = Q_astar_implicit(filepath)
        print(f"Q_astar_implicit - Path: {path_astar_implicit}, Distance: {dist_astar_implicit}")
    if Q_Multi_enabled:
        missions = [(12, 87), (45, 93), (93, 8), (8, 67)]  # Harder missions with nodes 0-99
        path_multi, total_time_multi, final_fatigue_multi = Q_multi(filepath, missions)
        print(f"Q_multi - Path: {path_multi}, Total Time: {total_time_multi}, Final Fatigue: {final_fatigue_multi}")
    if Q_Multi_pruning_enabled:
        mission_file = 'example_missions/medium-mediummission.txt'
        path_multi_pruning, total_time_multi_pruning, final_fatigue_multi_pruning = Q_multi_pruning(filepath, mission_file)
        print(f"Q_multi_pruning - Path: {path_multi_pruning}, Total Time: {total_time_multi_pruning}, Final Fatigue: {final_fatigue_multi_pruning}")
    if Q_Multi_opti_enabled:
        missions = [(12, 87), (45, 93), (93, 8), (8, 67)]  # Same missions for optimal order
        best_order, best_path, best_time, best_fatigue = Q_multi_opti(filepath, missions)
        print(f"Q_multi_opti - Order: {best_order}, Path: {best_path}, Total Time: {best_time}, Final Fatigue: {best_fatigue}")

if __name__ == "__main__":
    filename = 'medium-smallfatigue'
    main(filename, Q1_enabled=True, Q2_enabled=True, Q3_enabled=True, Q_pruning_enabled=True, Q_hybrid_enabled=True, Q_pruning_implicit_enabled=True, Q_astar_enabled=True, Q_astar_implicit_enabled=True, Q_Multi_enabled=True, Q_Multi_pruning_enabled=True, Q_Multi_opti_enabled=True)
