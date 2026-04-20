"""
Tests unitaires — Structure de données Graph (graph.py)

Vérifie l'implémentation de Dijkstra sur graphes explicites et implicites,
ainsi que la gestion des erreurs de poids.
"""

import pytest
from graph import Graph, GraphImplicit, Edge

# ---------------------------------------------------------------------------
# Test 1 : Dijkstra classique (Graphe Explicite)
# ---------------------------------------------------------------------------

def test_explicit_shortest_path_standard() -> None:
    """
    Vérifie Dijkstra sur un graphe simple stocké en mémoire.
    Chemin : 0 -(10)-> 1 -(5)-> 2  (Total 15)
    Alternative : 0 -(20)-> 2      (Total 20)
    """
    g = Graph(n=3)
    g.add_edge(0, 1, 10.0)
    g.add_edge(1, 2, 5.0)
    g.add_edge(0, 2, 20.0)
    
    path, dist = g.shortest_path(0, 2)
    
    assert dist == 15.0
    assert path == [0, 1, 2]

# ---------------------------------------------------------------------------
# Test 2 : Dijkstra sur Graphe Implicite (Lazy loading)
# ---------------------------------------------------------------------------

def test_implicit_shortest_path_standard() -> None:
    """
    Vérifie que GraphImplicit génère bien les voisins à la volée.
    Structure identique au Test 1.
    """
    def neighbor_fn(u: int):
        if u == 0: return [(1, 10.0), (2, 20.0)]
        if u == 1: return [(2, 5.0)]
        return []

    gi = GraphImplicit(n=3, neighbor_fn=neighbor_fn)
    path, dist = gi.shortest_path(0, 2)
    
    assert dist == 15.0
    assert path == [0, 1, 2]

# ---------------------------------------------------------------------------
# Test 3 : Gestion des erreurs de poids négatifs
# ---------------------------------------------------------------------------

def test_negative_weight_error() -> None:
    """Dijkstra doit lever une ValueError si un poids est négatif."""
    g = Graph(n=2)
    with pytest.raises(ValueError, match="Dijkstra ne supporte pas les poids négatifs"):
        g.add_edge(0, 1, -1.0)

# ---------------------------------------------------------------------------
# Test 4 : Cas limites (Nœuds isolés et auto-boucles)
# ---------------------------------------------------------------------------

def test_unreachable_node() -> None:
    """Un nœud inatteignable doit renvoyer une distance infinie."""
    g = Graph(n=3)
    g.add_edge(0, 1, 5.0)
    # Le nœud 2 n'est relié à rien
    path, dist = g.shortest_path(0, 2)
    
    assert path == []
    assert dist == float('inf')

def test_start_is_goal() -> None:
    """Si départ == arrivée, la distance est nulle."""
    g = Graph(n=1)
    path, dist = g.shortest_path(0, 0)
    
    assert path == [0]
    assert dist == 0.0