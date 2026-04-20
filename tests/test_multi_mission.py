"""
Tests unitaires — Partie 3 Q1 : missions multiples avec propagation de fatigue.

small.txt :  4 nœuds, correspondance alphabétique :
    ensae=0, guichet=1, lozere=2, saclay=3
    start = lozere = 2,  goal = saclay = 3

Arêtes :
    lozere(2) → ensae(0)  : l=10, f=2
    lozere(2) → guichet(1): l=20, f=0
    guichet(1)→ ensae(0)  : l=15, f=1
    ensae(0)  → saclay(3) : l=45, f=0
"""

import math
import os
import pytest

from network import Network


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SMALL = "examples/small.txt"


def _net() -> Network:
    return Network(filename=SMALL)


# ---------------------------------------------------------------------------
# Test 1 : k=1 mission → même résultat que Q3
# ---------------------------------------------------------------------------

def test_single_mission_matches_q3() -> None:
    """
    Une seule mission (start → goal) doit donner le même temps que Q3.

    Q3 sur small.txt :
        chemin = [2, 1, 0, 3]  (lozere → guichet → ensae → saclay)
        temps  = 20*1 + 15*1 + 45*2 = 125
        fatigue finale = 2
    """
    net = _net()
    path, time, final_F = net.multi_mission_path([(net.start, net.goal)])

    assert time == pytest.approx(125.0)
    assert path == [2, 1, 0, 3]
    assert final_F == 2


# ---------------------------------------------------------------------------
# Test 2 : k=2, la fatigue de la mission 1 impacte bien la mission 2
# ---------------------------------------------------------------------------

def test_fatigue_propagated_between_missions() -> None:
    """
    Missions : lozere→ensae (mission 1) puis ensae→saclay (mission 2).
    vt1 = vs2 = 0 (ensae), donc pas de trajet inter-mission.

    Mission 1 (lozere→ensae) au départ F=1 :
        - Chemin direct  2→0 : 10*1=10, fatigue finale=3  ← glouton (min temps)
        - Chemin long    2→1→0 : 20+15=35, fatigue finale=2

    Mission 2 (ensae→saclay) au départ F=3 (propagé) :
        - 0→3 : 45*3=135

    Temps total = 10 + 135 = 145.
    Si la fatigue avait été réinitialisée à 1, le temps serait 10 + 45 = 55.
    """
    net = _net()
    # ensae=0 explicitement identifié pour la lisibilité
    ensae = net.dict_matching["ensae"]   # = 0
    path, time, final_F = net.multi_mission_path(
        [(net.start, ensae), (ensae, net.goal)]
    )

    assert time == pytest.approx(145.0), (
        "La fatigue doit être propagée : avec F=3 au départ de mission 2, "
        "le temps mission 2 est 135, total 145."
    )
    assert path == [2, 0, 3]
    assert final_F == 3

    # Vérification explicite que la fatigue n'a PAS été réinitialisée
    assert time != pytest.approx(55.0), "La fatigue ne doit pas être remise à 1 entre missions."


# ---------------------------------------------------------------------------
# Test 3 : trajet inter-mission explicite
# ---------------------------------------------------------------------------

def test_between_mission_travel() -> None:
    """
    Missions : lozere→guichet (mission 1) puis ensae→saclay (mission 2).
    vt1=guichet(1) ≠ vs2=ensae(0), donc un trajet 1→0 est inséré automatiquement.

    Leg 1 (2→1) au départ F=1 : 20*1=20,  F→1
    Trajet intermédiaire (1→0) au départ F=1 : 15*1=15, F→2
    Leg 2 (0→3) au départ F=2 : 45*2=90,  F→2
    Total = 20 + 15 + 90 = 125, chemin = [2, 1, 0, 3]
    """
    net = _net()
    guichet = net.dict_matching["guichet"]  # = 1
    ensae   = net.dict_matching["ensae"]    # = 0

    path, time, final_F = net.multi_mission_path(
        [(net.start, guichet), (ensae, net.goal)]
    )

    assert time == pytest.approx(125.0)
    assert path == [2, 1, 0, 3]
    assert final_F == 2


# ---------------------------------------------------------------------------
# Test 4 : réseau construit à la main (vérification manuelle)
# ---------------------------------------------------------------------------

def test_custom_network_handcrafted(tmp_path) -> None:
    """
    Réseau minimal construit à la main pour validation numérique exacte.

    Nœuds (tri alpha) : A=0, B=1, C=2
    Arêtes :
        A → B : l=1, f=0
        B → C : l=1, f=4
    start=A=0, goal=C=2

    k=2 missions [(A→B), (B→C)] — vt1=vs2=1, pas de trajet inter.

    Leg 1 (A→B) F=1 : 1*1=1, F reste 1  (f_AB=0)
    Leg 2 (B→C) F=1 : 1*1=1, F→5       (f_BC=4)
    Total = 2, chemin = [0, 1, 2], fatigue finale = 5

    k=1 mission (A→C) doit aussi donner 2 (même chemin unique).
    """
    content = "3 A C\nA B 1 0\nB C 1 4\n"
    net_file = tmp_path / "hand.txt"
    net_file.write_text(content)

    # Nécessaire : changer le répertoire courant pour que
    # format_examples.py trouve le fichier (path absolu suffit ici)
    net = Network(filename=str(net_file))

    A = net.dict_matching["A"]  # 0
    B = net.dict_matching["B"]  # 1
    C = net.dict_matching["C"]  # 2

    # k=1
    path1, time1, F1 = net.multi_mission_path([(A, C)])
    assert time1 == pytest.approx(2.0)
    assert path1 == [A, B, C]
    assert F1 == 5

    # k=2 : même résultat car seule l'ordre des trajets change, pas le chemin
    path2, time2, F2 = net.multi_mission_path([(A, B), (B, C)])
    assert time2 == pytest.approx(2.0)
    assert path2 == [A, B, C]
    assert F2 == 5


# ---------------------------------------------------------------------------
# Test 5 : liste de missions vide
# ---------------------------------------------------------------------------

def test_empty_missions() -> None:
    """Aucune mission → chemin vide, temps nul, fatigue initiale inchangée."""
    net = _net()
    path, time, final_F = net.multi_mission_path([])

    assert path == []
    assert time == 0.0
    assert final_F == 1


# ---------------------------------------------------------------------------
# Test 6 : destination inatteignable
# ---------------------------------------------------------------------------

def test_unreachable_mission() -> None:
    """
    saclay(3) n'a aucune arête sortante dans small.txt.
    La mission (3 → 0) est donc infaisable.
    """
    net = _net()
    # Mission 1 réussit (lozere → saclay), mission 2 échoue (saclay → ensae)
    path, time, final_F = net.multi_mission_path(
        [(net.start, net.goal), (net.goal, 0)]
    )

    assert path == []
    assert math.isinf(time)


# ---------------------------------------------------------------------------
# Test 7 : optimal_mission_order — cas trivial k=1
# ---------------------------------------------------------------------------

def test_optimal_mission_order_k1() -> None:
    """k=1 : une seule permutation possible, résultat identique à multi_mission_path."""
    net = _net()
    missions = [(net.start, net.goal)]

    best_order, path_opt, time_opt, F_opt = net.optimal_mission_order(missions)
    _, time_seq, _ = net.multi_mission_path(missions)

    assert best_order == missions
    assert time_opt == pytest.approx(time_seq)


# ---------------------------------------------------------------------------
# Test 8 : optimal_mission_order — trouve le meilleur ordre (réseau tmp)
# ---------------------------------------------------------------------------

def test_optimal_mission_order_finds_better_order(tmp_path) -> None:
    """
    Réseau bidirectionnel simple avec f=0 (fatigue constante = 1).

    Nœuds (tri alpha) : a=0, b=1, end=2, s=3
    Arêtes :
        s  → a  : l=1
        a  → s  : l=1
        a  → b  : l=2
        b  → a  : l=2
        a  → end: l=3
        b  → end: l=1
        end→ a  : l=3
        end→ b  : l=1

    Missions : M1=(s→a), M2=(b→end)

    Ordre [M1, M2] :
        s→a (1) + entre a→b (2) + b→end (1) = 4

    Ordre [M2, M1] :
        b→end (1) + entre end→b→a (2) + a→ ... non, entre end→s :
        end→a (3) + a→s (1) = 4, puis s→a (1) = total 1+4+1=6

    Donc ordre [M1, M2] est optimal avec temps=4.
    """
    content = (
        "4 s end\n"
        "s a 1 0\n"
        "a s 1 0\n"
        "a b 2 0\n"
        "b a 2 0\n"
        "a end 3 0\n"
        "b end 1 0\n"
        "end a 3 0\n"
        "end b 1 0\n"
    )
    net_file = tmp_path / "bidir.txt"
    net_file.write_text(content)
    net = Network(filename=str(net_file))

    s   = net.dict_matching["s"]    # 3
    a   = net.dict_matching["a"]    # 0
    b   = net.dict_matching["b"]    # 1
    end = net.dict_matching["end"]  # 2

    missions = [(s, a), (b, end)]
    best_order, path_opt, time_opt, F_opt = net.optimal_mission_order(missions)

    # L'ordre optimal doit être [(s→a), (b→end)] avec temps=4
    assert time_opt == pytest.approx(4.0)
    assert best_order == [(s, a), (b, end)]

    # L'ordre inverse doit être plus coûteux
    _, time_reversed, _ = net.multi_mission_path([(b, end), (s, a)])
    assert time_reversed > time_opt
