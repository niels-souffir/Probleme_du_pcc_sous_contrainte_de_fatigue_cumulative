"""
Tests unitaires — Classe Network (network.py)

Utilise un fichier temporaire pour valider la mécanique de fatigue,
l'encodage des sommets étendus et les algorithmes de recherche.
"""

import pytest
import os
from network import Network

# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def handcrafted_net(tmp_path) -> Network:
    """
    Crée un réseau minimal :
    0 -> 1 : l=10, f=2
    1 -> 2 : l=5,  f=3
    0 -> 2 : l=20, f=1
    """
    content = "3 0 2\n0 1 10 2\n1 2 5 3\n0 2 20 1"
    net_file = tmp_path / "test_net.txt"
    net_file.write_text(content)
    return Network(filename=str(net_file))

# ---------------------------------------------------------------------------
# Test 1 : Initialisation et F_max
# ---------------------------------------------------------------------------

def test_network_initialization(handcrafted_net) -> None:
    """Vérifie le chargement des métadonnées du fichier."""
    assert handcrafted_net.n == 3
    assert handcrafted_net.start == 0
    assert handcrafted_net.goal == 2

def test_f_max_calculation(handcrafted_net) -> None:
    """
    Vérifie le calcul de la fatigue maximale (n-1 pires arêtes).
    n=3, arêtes f=[2, 3, 1]. Top 2 : 3+2=5. F_max = 5+1 = 6.
    """
    handcrafted_net.compute_F_max()
    assert handcrafted_net.F_max == 6

# ---------------------------------------------------------------------------
# Test 2 : Encodage des états (sommet, fatigue)
# ---------------------------------------------------------------------------

def test_encoding_logic(handcrafted_net) -> None:
    """Vérifie que l'encodage est réversible."""
    handcrafted_net.compute_F_max() # F_max = 6
    node, fatigue = 1, 3
    encoded = handcrafted_net._encode(node, fatigue)
    
    # Décodage manuel
    decoded_node = encoded // (handcrafted_net.F_max + 1)
    decoded_fatigue = encoded % (handcrafted_net.F_max + 1)
    
    assert decoded_node == node
    assert decoded_fatigue == fatigue

# ---------------------------------------------------------------------------
# Test 3 : Calcul du temps avec fatigue (Pruning vs A*)
# ---------------------------------------------------------------------------

def test_fatigue_path_calculation(handcrafted_net) -> None:
    """
    Vérifie le calcul dynamique du temps (Distance * Fatigue).
    
    Option A (0->2) : 20 * (F_init=1) = 20.0
    Option B (0->1->2) : (10 * F_init=1) + (5 * F_après_01=3) = 10 + 15 = 25.0
    
    L'optimum doit être 20.0.
    """
    # Test avec Pruning de Pareto
    p_path, p_dist = handcrafted_net.pruning(0, 2)
    # Test avec A*
    a_path, a_dist = handcrafted_net.astar(0, 2)
    
    assert p_dist == pytest.approx(20.0)
    assert a_dist == pytest.approx(20.0)
    assert p_path == [0, 2]

# ---------------------------------------------------------------------------
# Test 4 : Borne de fatigue resserrée (F_bound)
# ---------------------------------------------------------------------------

def test_f_bound_reduction(handcrafted_net) -> None:
    """Vérifie que F_bound est inférieur ou égal à F_max."""
    handcrafted_net.compute_F_max()
    f_max = handcrafted_net.F_max
    f_bound = handcrafted_net.compute_F_bound(T=20.0)
    
    assert f_bound <= f_max
    assert f_bound >= 1
