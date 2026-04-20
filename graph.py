import heapq
from __future__ import annotations 
from dataclasses import dataclass
from typing import Iterable, Callable, Union

@dataclass(frozen=True) 
class Edge:
    """Arête orientée et pondérée dans le graphe.

    Attributs
    ---------
    to 
        Numéro (index) du sommet de destination.
    weight 
        Poids ou longueur de l'arête.
    """
    to: int
    weight: float      # Doit être non négatif pour Dijkstra

class Graph:
    """Graphe orienté et pondéré, représenté par listes d'adjacence.

    Gère un réseau de sommets indexés de 0 à n-1. 

    Attributs
    ---------
    n : int
        Le nombre total de sommets dans le graphe.
    adj : list[list[Edge]]
        Liste d'adjacence où chaque entrée i contient la liste des arcs (objets Edge) partant du sommet i.
    """
    def __init__(self, n: int) -> None:
        """Initialise un graphe vide avec n sommets.

        Paramètres
        ----------
        n 
            Ordre du graphe.

        """
        self.n = int(n)
        self.adj: list[list[Edge]] = [[] for _ in range(self.n)]

    def add_edge(self, u: int, v: int, weight: float) -> None:
        """Ajoute une arête orientée au graphe.

        Complexité
        ----------
        O(1)

        Paramètres
        ----------
        u 
            Sommet source.
        v 
            Sommet destination.
        weight 
            Poids (longueur) de l'arête. 

        Raises
        ------
        ValueError
            Si le poids est strictement négatif, car cela brise les invariants de l'algorithme de Dijkstra.
        """
        if weight < 0:
            raise ValueError("Dijkstra ne supporte pas les poids négatifs.")
        self.adj[int(u)].append(Edge(to=int(v), weight=weight))
        
    def shortest_path(self, start: int, goal: int) -> tuple[list[int], float]: # Question I-1 
        """Algorithme de Dijkstra pour la recherche de plus court chemins d’origine fixée dans un graphe avec longueur positive, en utilsant une structure de tas.

        Complexité
        ----------
        Temporelle : O((n+m)log n)
        Spatiale : O(n+m)
        
        Paramètres
        ----------
        start 
            Point Initial, représenté par le numéro du sommet correspondant.
        goal 
            Point Final, représenté par le numéro du sommet correspondant.
        
        Retours
        ------
        tuple[list[int], float]
            1) Il existe un chemin de start vers goal : (plus court chemin de start vers goal, longueur de ce chemin).
            2) Aucun chemin n'existe de start vers goal : (liste vide, infini).

        """
        s, t = int(start), int(goal)
        inf = float("inf")
        
        # Initialisation : distance infinie pour tous les sommets.
        # dist[i] stocke la plus courte distance connue de 's' à 'i'.
        # parent[i] stocke l'indice du sommet précédent par lequel on est passé pour atteindre le sommet i avec la distance la plus courte actuellement connue.
        dist = [inf] * self.n
        parent = [-1] * self.n
        dist[s] = 0.0
        
        # On trie par distance pour toujours explorer le sommet le plus proche (approche gloutonne).
        pq: list[tuple[float, int]] = [(0.0, s)] # (distance, sommet).

        while pq:
            d_u, u = heapq.heappop(pq)
            if d_u > dist[u]:
                continue
            if u == t:
                break
            
            # Exploration des voisins de u.
            for e in self.adj[u]:
                nd = d_u + e.weight # Calcul de la distance potentielle via u.
                if nd < dist[e.to]:
                    dist[e.to] = nd
                    parent[e.to] = u
                    heapq.heappush(pq, (nd, e.to))
        
        # Cas d'échec.
        if dist[t] == inf:
            return ([], inf)
        
        # Reconstruction du chemin en remontant les parents depuis la cible.
        path = [t]
        cur = t
        while cur != s:
            cur = parent[cur]
            path.append(cur)
        
        # Le chemin a été construit de t vers s, on l'inverse pour le résultat final.
        path.reverse()
        return (path, dist[t])

class GraphImplicit(Graph): # Question I-3
    """Variante implicite de Graph où les arêtes ne sont pas stockées dans self.adj mais fournies à la volée par la fonction neighbor_fn.
    """
    def __init__(self, n: int, neighbor_fn: Callable[[int], Iterable[Union[Edge, tuple[int, float]]]],) -> None:
        """Initialise le graphe sans structure de stockage interne pour les arêtes.
        
        Paramètres
        ----------
        n 
            Le nombre total d'états possibles (sommets étendus).
        _neighbor_fn 
            Règle de calcul des voisins : génère les transitions (passage d'un état à un autre) sortantes à la demande.
        
      """
        self.n = n
        self._neighbor_fn = neighbor_fn

    def _neighbors(self, u: int) -> Iterable[Edge]:  # Question I-3
        """Génère les arêtes sortantes de l'état u par évaluation paresseuse.

        Complexité
        ----------
        O(deg(u))

        Paramètres
        ----------
        u 
            Identifiant de l'état actuel (combinaison sommet et fatigue).

        Retours
        -------
        Iterable[Edge]
            Un générateur d'arêtes valides partant de u.
        """
        for item in self._neighbor_fn(u):
            if isinstance(item, Edge):
                yield item
            else:
                
                # Extraction et conversion du sommet de destination et du poids.
                v, w = item
                w = float(w)
                if w < 0:
                    raise ValueError("Dijkstra ne supporte pas les poids négatifs.")
                
                # Production d'un objet Edge à la volée (approche lazy).
                yield Edge(to=int(v), weight=w)

    def shortest_path(self, start: int, goal: int) -> tuple[list[int], float]:  # Question I-3
        """Algorithme de Dijkstra adapté à une structure de graphe implicite.
       
        Cette variante de la recherche de plus court chemin n'explore pas un graphe stocké en mémoire, mais sollicite dynamiquement la méthode _neighbors pour découvrir les sommets adjacents.

        Paramètres
        ----------
        start 
            Identifiant de l'état initial.
        goal
            Identifiant de l'état cible.

        Retours
        -------
        tuple[list[int], float]
            Le chemin optimal sous forme de liste d'états et son coût total.

        """
        # Aproche similaire à la méthode shortest_path de Graph
        s, t = int(start), int(goal)
        inf = float("inf")
        dist = [inf] * self.n
        parent = [-1] * self.n
        dist[s] = 0.0
        pq: list[tuple[float, int]] = [(0.0, s)]
        while pq:
            d_u, u = heapq.heappop(pq)
            if d_u != dist[u]:
                continue
            if u == t:
                break
            for e in self._neighbors(u):
                nd = d_u + e.weight
                if nd < dist[e.to]:
                    dist[e.to] = nd
                    parent[e.to] = u
                    heapq.heappush(pq, (nd, e.to))
        if dist[t] == inf:
            return ([], inf)
        path = [t]
        cur = t
        while cur != s:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return (path, dist[t])

    def add_edge(self, u: int, v: int, weight: float) -> None:
        """Ajoute une arête orientée au graphe.

        Paramètres
        ----------
        u 
            Sommet source.
        v 
            Sommet destination.
        weight 
            Poids (longueur) de l'arête. 

        Raises
        ------
        ValueError
            Si le poids est strictement négatif, car cela brise les invariants de l'algorithme de Dijkstra.
        """
        if weight < 0:
            raise ValueError("Dijkstra ne supporte pas les poids négatifs.")
        self.adj[int(u)].append(Edge(to=int(v), weight=weight))
        
    def shortest_path(self, start: int, goal: int) -> tuple[list[int], float]: # Question I-1 
        """Algorithme de Dijkstra pour la recherche de plus court chemins d’origine fixée dans un graphe avec longueur positive, en utilsant une structure de tas.

        Paramètres
        ----------
        start 
            Point Initial, représenté par le numéro du sommet correspondant.
        goal 
            Point Final, représenté par le numéro du sommet correspondant.
        
        Retours
        ------
        tuple[list[int], float]
            1) Il existe un chemin de start vers goal : (plus court chemin de start vers goal, longueur de ce chemin).
            2) Aucun chemin n'existe de start vers goal : (liste vide, infini).

        """
        s, t = int(start), int(goal)
        inf = float("inf")
        
        # Initialisation : distance infinie pour tous les sommets.
        # dist[i] stocke la plus courte distance connue de 's' à 'i'.
        # parent[i] stocke l'indice du sommet précédent par lequel on est passé pour atteindre le sommet i avec la distance la plus courte actuellement connue.
        dist = [inf] * self.n
        parent = [-1] * self.n
        dist[s] = 0.0
        
        # On trie par distance pour toujours explorer le sommet le plus proche (approche gloutonne).
        pq: list[tuple[float, int]] = [(0.0, s)] # (distance, sommet).

        while pq:
            d_u, u = heapq.heappop(pq)
            if d_u > dist[u]:
                continue
            if u == t:
                break
            
            # Exploration des voisins de u.
            for e in self.adj[u]:
                nd = d_u + e.weight # Calcul de la distance potentielle via u.
                if nd < dist[e.to]:
                    dist[e.to] = nd
                    parent[e.to] = u
                    heapq.heappush(pq, (nd, e.to))
        
        # Cas d'échec.
        if dist[t] == inf:
            return ([], inf)
        
        # Reconstruction du chemin en remontant les parents depuis la cible.
        path = [t]
        cur = t
        while cur != s:
            cur = parent[cur]
            path.append(cur)
        
        # Le chemin a été construit de t vers s, on l'inverse pour le résultat final.
        path.reverse()
        return (path, dist[t])

class GraphImplicit(Graph): # Question I-3
    """Variante implicite de Graph où les arêtes ne sont pas stockées dans self.adj mais fournies à la volée par la fonction neighbor_fn.
    """
    def __init__(self, n: int, neighbor_fn: Callable[[int], Iterable[Union[Edge, tuple[int, float]]]],) -> None:
        """Initialise le graphe sans structure de stockage interne pour les arêtes.
        
        Paramètres
        ----------
        n 
            Le nombre total d'états possibles (sommets étendus).
        _neighbor_fn 
            Règle de calcul des voisins : génère les transitions (passage d'un état à un autre) sortantes à la demande.
        
      """
        self.n = n
        self._neighbor_fn = neighbor_fn

    def _neighbors(self, u: int) -> Iterable[Edge]:  # Question I-3
        """Génère les arêtes sortantes de l'état u par évaluation paresseuse.

        Paramètres
        ----------
        u 
            Identifiant de l'état actuel (combinaison sommet et fatigue).

        Retours
        -------
        Iterable[Edge]
            Un générateur d'arêtes valides partant de u.
        """
        for item in self._neighbor_fn(u):
            if isinstance(item, Edge):
                yield item
            else:
                
                # Extraction et conversion du sommet de destination et du poids.
                v, w = item
                w = float(w)
                if w < 0:
                    raise ValueError("Dijkstra ne supporte pas les poids négatifs.")
                
                # Production d'un objet Edge à la volée (approche lazy).
                yield Edge(to=int(v), weight=w)

    def shortest_path(self, start: int, goal: int) -> tuple[list[int], float]:  # Question I-3
        """Algorithme de Dijkstra adapté à une structure de graphe implicite.
       
        Cette variante de la recherche de plus court chemin n'explore pas un graphe stocké en mémoire, mais sollicite dynamiquement la méthode _neighbors pour découvrir les sommets adjacents.

        Paramètres
        ----------
        start 
            Identifiant de l'état initial.
        goal
            Identifiant de l'état cible.

        Retours
        -------
        tuple[list[int], float]
            Le chemin optimal sous forme de liste d'états et son coût total.

        """
        # Aproche similaire à la méthode shortest_path de Graph
        s, t = int(start), int(goal)
        inf = float("inf")
        dist = [inf] * self.n
        parent = [-1] * self.n
        dist[s] = 0.0
        pq: list[tuple[float, int]] = [(0.0, s)]
        while pq:
            d_u, u = heapq.heappop(pq)
            if d_u != dist[u]:
                continue
            if u == t:
                break
            for e in self._neighbors(u):
                nd = d_u + e.weight
                if nd < dist[e.to]:
                    dist[e.to] = nd
                    parent[e.to] = u
                    heapq.heappush(pq, (nd, e.to))
        if dist[t] == inf:
            return ([], inf)
        path = [t]
        cur = t
        while cur != s:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return (path, dist[t])
