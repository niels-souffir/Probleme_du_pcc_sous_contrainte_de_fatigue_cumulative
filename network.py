import heapq
import math
from __future__ import annotations 
from dataclasses import dataclass 
from itertools import permutations
from typing import Any, Optional 
from graph import Graph, GraphImplicit
from format_examples import main as format_main

@dataclass 
class Network:
    """Classe chargée du chargement et de la structuration des données du réseau.
 
    Fait le lien entre les fichiers de données externes (.txt) et l'objet Graph. 
    Extrait la topologie du réseau (sous forme de DataFrame), les points de départ/arrivée et établit une correspondance entre les noms réels des villes et leurs indices numériques.

    Attributs
    ----------
    filename : str
        Chemin vers le fichier de données source.
    n : int
        Nombre de sommets.
    start : int
        Indice du sommet de départ.
    goal : int
        Indice du sommet de destination.
    dict_matching : dict
        Dictionnaire de correspondance entre les noms des villes et leurs identifiants entiers.
    df : pd.DataFrame
        Tableau contenant les arêtes et leurs caractéristiques (poids, fatigue).

    """
    def __init__(self, filename: str = "examples/small.txt"):
        """Initialise le réseau à partir d'un fichier texte.

        Paramètres
        ----------
        filename 
            Le chemin du fichier à charger (argument par mot clé optionel).

        """
        self.filename = filename
        dict_matching, df, (start, goal) = format_main(self.filename, export_csv=False) # Voire commentaires dans "format_examples.py"
        self.n = len(dict_matching)
        self.start = int(start)
        self.goal = int(goal)
        self.dict_matching = dict_matching
        self.df = df

    def build_simple_graph(self) -> Graph:   # Question I-1 
        """Construit un graphe simple (sans fatigue) contenant la topologie du réseau avec les distances comme poids d'arêtes, à partir du DataFrame formaté.

        Complexité 
        ----------
        Temporelle : O(m)
            Itération unique sur les m lignes du DataFrame.
        Spatiale : O(n + m) pour le stockage de la liste d'adjacence.
            Instanciation de n têtes de listes (une par sommet) et au stockage des m objets représentant les arêtes.

        Retours
        -------
        Graph
            Le graphe simple.
        """
        g = Graph(self.n)
        for line in self.df.itertuples(index=False):
            i = int(line[0])
            j = int(line[1])
            l_ij = int(line[2])
            g.add_edge(i, j, l_ij)
        return g

    def build_simple_graph_implicit(self) -> GraphImplicit: # Question I-3
        """Construit un graphe implicite simple (sans fatigue) à partir du DataFrame.
        
        Prépare la fonction de voisinage nécessaire à l'exploration sans construire de liste d'adjacence au sein d'un objet Graph classique. 
        Elle utilise un dictionnaire de hachage pour indexer les arêtes, permettant un accès aux voisins en temps constant.
        
        Complexité
        ----------
        Temporelle : O(m)
            Parcours linéaire des m lignes du DataFrame. L'insertion dans le dictionnaire (edges_dict) s'effectue en temps constant O(1).
            
        Spatiale : O(n + m)
            Stockage des n sommets (clés) et des m arêtes (valeurs) dans le dictionnaire.

        Retours
        -------
        GraphImplicit
            Une instance de graphe dont les arêtes sont générées par neighbor_fn.
        
        """
        # Organisation des arêtes dans un dictionnaire pour un accès direct O(1).
        edges_dict = {}
        for line in self.df.itertuples(index=False):
            i = int(line[0])
            j = int(line[1])
            l_ij = int(line[2])
            edges_dict.setdefault(i, []).append((j, l_ij))
        
        # Définition de la fonction de transition utilisée par GraphImplicit.
        def neighbor_fn(node: int) -> list[tuple[int, float]]:
            return edges_dict.get(node, [])
    
        return GraphImplicit(self.n, neighbor_fn)

    def compute_F_max(self) -> None: # Question I-2
        """Calcule une borne supérieure de la fatigue pour tout chemin optimal.
        
        Borne nécessaire pour définir la taille de l'espace d'états du graphe étendu. 
        On part du principe qu'un chemin optimal ne repasse jamais deux fois par le même sommet et contient donc au plus (n-1) arêtes.

        Complexité
        ----------
        Temporelle: O(m log m) 
            Tri de la liste des fatigues.
        Spatiale: O(m) 
            Stockage de la liste temporaire des coefficients de fatigue.
        """
        fatigues = [int(line[3]) for line in self.df.itertuples(index=False)]
        if not fatigues:
            self.F_max = 1
            return
        
        # Pour obtenir la pire fatigue possible d'un chemin sans cycle, on additionne les (n-1) coefficients les plus élevés du réseau
        fatigues_sorted = sorted(fatigues, reverse=True)
        top_n_minus_1 = fatigues_sorted[:min(len(fatigues_sorted), self.n - 1)]
        self.F_max = max(1, sum(top_n_minus_1) + 1)

    def compute_F_bound(self, T: float) -> int:
        """Calcule une borne supérieure sur la fatigue de tout chemin de coût ≤ T (voir la preuve dans Proof.pdf).

        Étant donné T = coût d'un chemin réalisable connu, renvoie F_bound tel que tout chemin de coût ≤ T a une fatigue < 1 + (n - k_max) * f_max, c'est-à-dire une fatigue ≤ F_bound = (n - k_max) * f_max.
        Revient à self.F_max lorsque le discriminant est non positif (T est trop grand pour resserrer la borne) ou lorsque les statistiques du graphe sont dégénérées.
        """
        self.compute_F_max()

        rows = list(self.df.itertuples(index=False))
        d_min = min(int(r[2]) for r in rows) if rows else 1
        f_max = max(int(r[3]) for r in rows) if rows else 1

        if d_min <= 0 or f_max <= 0:
            return self.F_max

        n = self.n
        m = n - 1

        discriminant = (
            (2 * n - 1) ** 2
            - 4 * (
                2 * m / f_max
                + m ** 2 + m
                - 2 * T / (f_max * d_min)
            )
        )

        if discriminant <= 0:
            return self.F_max

        k_max = math.ceil((2 * n - 1 + math.sqrt(discriminant)) / 2)
        #k_max > m signifie que T est assez grand pour que n'importe quel chemin simple respecte le budget : la formule n'apporte aucune amélioration par rapport à F_max.
        if k_max > m:
            return self.F_max

        F_bound = (n - k_max) * f_max
        return max(1, min(int(F_bound), self.F_max))

    def _encode(self, v: int, F: int) -> int: # Question I-2 
        """Transforme un couple (nœud, fatigue) en un identifiant unique entier.
        
        Fonction de hachage qui permet de projeter l'espace bidimensionnel du graphe étendu sur une dimension unique, facilitant l'indexation dans les listes d'adjacence.

        Complexité
        ----------
        O(1)

        Paramètres
        ----------
        v 
            L'identifiant du nœud d'origine.
        F 
            Le niveau de fatigue accumulé au nœud v.

        Retours
        -------
        int
            L'indice unique correspondant dans le graphe étendu.

        """
        return v * (self.F_max + 1) + F

    def build_extended_graph(self) -> Graph: # Question I-2
        """Construit le graphe étendu (sommet, fatigue) à partir du DataFrame.
        
        Chaque sommet original est démultiplié en autant d'états qu'il existe de niveaux de fatigue possibles.
        Inclut un nœud objectif virtuel qui se connecte à tous les états (goal, F) avec un coût de 0, pour permettre un appel unique de shortest_path vers le meilleur achèvement.

        Complexité
        ----------
        Temporelle : O(m⋅F_max + mlogm)
        Spatiale : O((n+m)⋅F max)
        
        Retours
        -------
        Graph
            Une instance de Graph possédant (n * (F_max + 1)) + 1 sommets.
        
        """
        self.compute_F_max()
          
        # N_ext représente le nombre total d'états (nœud, fatigue).
        N_ext = self.n * (self.F_max + 1)
        
        # Le nœud virtuel d'arrivée est placé au dernier indice.
        virtual_goal_id = N_ext
        g_ext = Graph(N_ext + 1)
        
        # Pour chaque arc original dans le DataFrame.
        for line in self.df.itertuples(index=False):
            i = int(line[0])        # Sommet 1
            j = int(line[1])        # Sommet 2
            l_ij = int(line[2])     # Distance
            f_ij = int(line[3])     # Fatigue
            if f_ij < 0:
                f_ij = 0
            
            # Pour chaque niveau possible de fatigue accumulée.
            for F in range(1, self.F_max + 1):
                
                # Actualiser la fatigue après passage de l'arc.
                F_new = F + f_ij
                if F_new > self.F_max:
                    continue
                
                # Créer les états étendus et leur poids associé.
                u_ext = self._encode(i, F)
                v_ext = self._encode(j, F_new)
                weight = l_ij * F
                g_ext.add_edge(u_ext, v_ext, weight)
        
        # Connexion des états finaux (goal, F) au nœud virtuel avec un coût nul.
        for F in range(1, self.F_max + 1):
            goal_state = self._encode(self.goal, F)
            g_ext.add_edge(goal_state, virtual_goal_id, 0)
        return g_ext
 
    def build_extended_implicit_graph(self) -> GraphImplicit: # Question I-3
        """Initialise le graphe étendu sous forme implicite avec gestion de la fatigue.
        
        Définit les règles de transition entre les états (sommet, fatigue) et crée un sommet cible virtuel pour unifier les sorties. 
        L'utilisation d'un graphe implicite permet de simuler l'espace d'états sans allocation mémoire préalable des arêtes étendues.

        Complexité
        ----------
        Temporelle : O(m) + O(m log m)= O(m log m)
            Le pré-traitement du DataFrame s'effectue en une seule passe sur les m arêtes initiales : O(m)
            Compute_F_max: O(m log m)
            La définition de la fonction neighbor_fn est instantanée.
            
        Spatiale : O(n + m)
            La mémoire est limitée au stockage du réseau de base dans edges_dict. 
            Contrairement à une version explicite, l'espace n'est pas multiplié par F_max lors de cette phase de construction.

        Retours
        -------
        GraphImplicit
            Une structure capable de générer dynamiquement les transitions respectant la contrainte F <= F_max.

        """
        self.compute_F_max()
        N_ext = self.n * (self.F_max + 1)
        virtual_goal_id = N_ext

        # Indexation des données de base pour un accès rapide : O(m).
        edges_dict = {}
        for line in self.df.itertuples(index=False):
            i = int(line[0])
            j = int(line[1])
            l_ij = int(line[2])
            f_ij = int(line[3])
            edges_dict.setdefault(i, []).append((j, l_ij, f_ij))

        def neighbor_fn(label_id: int) -> list[tuple[int, float]]:
            """Calcule les transitions valides à partir d'un identifiant d'état.
            
            Paramètres
            ----------
            label_id 
                Identifiant unique de l'état source. 
                Cet entier est décodé selon la formule : label_id = v * (F_max + 1) + F, où :
                - v : Index du sommet dans le graphe original (0 à n-1).
                - F : Niveau de fatigue accumulée (0 à F_max).

            """
            v = label_id // (self.F_max + 1)
            F = label_id % (self.F_max + 1)
            neighbors = []

            # Connexion gratuite vers le but virtuel si le sommet actuel est la cible.
            if v == self.goal:
                neighbors.append((virtual_goal_id, 0))

            # Itération uniquement pour les arcs sortant de v
            for j, l_ij, f_ij in edges_dict.get(v, []):
                F_new = F + f_ij
                if F_new <= self.F_max:
                    v_ext = self._encode(j, F_new)
                    weight = l_ij * F
                    neighbors.append((v_ext, weight))
            return neighbors

        return GraphImplicit(n=N_ext + 1, neighbor_fn=neighbor_fn)
    
    def pruning(self, start: int, goal_node: int, time_limit: float = float('inf'), f_bound: Optional[int] = None) -> tuple[list[int], float]:  # Question II-1
        """Calcule le plus court chemin avec contrainte de fatigue via Dijkstra optimisé par Pruning de Pareto.

        Intègre des possibilités d'optimisation pour réduire l'espace de recherche via des bornes de temps et de fatigue.
        
        Complexité
        ----------
        Temporelle : O(mlog(m) + m⋅L⋅(L+log(n⋅L)))= O(m⋅L⋅log(n⋅L)))
        Spatiale : O(n⋅L+m)

        Extensions de pruning
        ---------------------
        f_bound 
            Facultatif plafond de fatigue plus petit depuis la fonction compute_F_bound().
        Si fournie et plus petite que F_max, elle réduit l’espace d’état.

        time_limit
            Borne supérieure sur la distance totale.

        Paramètres
        ----------
        start
            Point Initial, représenté par le numéro du sommet correspondant.
        goal_node 
            Point Final, représenté par le numéro du sommet correspondant.
        time_limit 
            Plafond de distance/temps. Tout chemin dépassant cette valeur est élagué.
        f_bound 
            Plafond de fatigue réduit (issu par exemple d'une solution trouvée par ACO).
            Si None, utilise la fatigue maximale théorique du graphe.

        Retours
        -------
        tuple[list[int], float]
            1) Il existe un chemin de start vers goal : le chemin optimal (liste de sommets) et le coût total (distance pondérée).
            2) Aucun chemin n'existe de start vers goal : ([], float('inf'))
        """
        self.compute_F_max()
        if f_bound is not None and f_bound < self.F_max:
            self.F_max = f_bound
        enc_start = self._encode(start, 1)
        
        # --- ÉTAPE 1 : Indexation des données ---
        # Stockage local pour un accès rapide aux voisins du réseau de base.
        adj = {}
        for line in self.df.itertuples(index=False):
            i, j, l, f = int(line[0]), int(line[1]), int(line[2]), int(line[3])
            adj.setdefault(i, []).append((j, l, f))
        
        # Initialisation des frontières de Pareto.
        pareto_fronts: dict[int, list[tuple[float, int]]] = {}
        for node in range(self.n):
            pareto_fronts[node] = []
        
        # --- ÉTAPE 2 : Recherche avec pruning Pareto ---
        pq: list[tuple[float, int]] = [(0.0, enc_start)]
        visited: set[int] = set()  # Visited : ensemble des états complètement explorés (Optimisation mémoire/temps).
        distances: dict[int, float] = {enc_start: 0.0}
        parent: dict[int, int] = {enc_start: -1}
        best_arrival_enc = None
        while pq:
            dist, u_enc = heapq.heappop(pq)
            if u_enc in visited:
                continue
            visited.add(u_enc)

            # Décodage du nœud étendu.
            node_u = u_enc // (self.F_max + 1)
            fatigue_u = u_enc % (self.F_max + 1)
            
            # --- TEST DE DOMINANCE DE PARETO ---
            # Un chemin est dominé s'il existe déjà un chemin plus court ET moins fatiguant vers node_u.
            new_frontier = [(dist, fatigue_u)]
            is_dominated = False
            for (d_old, f_old) in pareto_fronts[node_u]:
                if d_old <= dist and f_old <= fatigue_u:
                    is_dominated = True
                    break  # Sortie anticipée : on est dominé.
                else:
                    # Garder les anciens chemins que celui-ci ne domine pas.
                    if not (dist <= d_old and fatigue_u <= f_old):
                        new_frontier.append((d_old, f_old))
            if is_dominated: # Si ce chemin est dominé, on l'ignore
                continue
            pareto_fronts[node_u] = new_frontier  # Assigner la frontier mise à jour.
            if node_u == goal_node:  # Arrêt dès que la cible est atteinte (propriété de Dijkstra).
                best_arrival_enc = u_enc
                break
            
            # ========== EXPLORATION DES VOISINS ==========
            for (j, l_ij, f_ij) in adj.get(node_u, []):
                fatigue_v = fatigue_u + f_ij
                if fatigue_v > self.F_max:  # Respecter la limite de fatigue
                    continue
                v_enc = self._encode(j, fatigue_v)
                new_dist = dist + (l_ij * fatigue_u)
                if new_dist > time_limit:
                    continue
                if v_enc in visited: 
                    continue
                
                # Vérification de dominance avant même d'ajouter à la file (Pruning préventif).
                is_v_dominated = False
                for (d_frontier, f_frontier) in pareto_fronts[j]:
                    if d_frontier <= new_dist and f_frontier <= fatigue_v:
                        is_v_dominated = True
                        break
                if not is_v_dominated:
                    if v_enc not in distances or new_dist < distances[v_enc]:
                        distances[v_enc] = new_dist
                        parent[v_enc] = u_enc
                        heapq.heappush(pq, (new_dist, v_enc))
        
        # --- ÉTAPE 3 : Reconstruction du chemin ---
        if best_arrival_enc is None:
            return [], float('inf')
        path_enc = []
        cur = best_arrival_enc
        while cur != -1:
            path_enc.append(cur)
            cur = parent.get(cur, -1)
        path_enc.reverse()
        path = [node // (self.F_max + 1) for node in path_enc]
        best_arrival_dist = distances[best_arrival_enc]
        return path, best_arrival_dist
    
    def pruning_implicit(self, start: int, goal_node: int, time_limit: float = float('inf'), f_bound: Optional[int] = None) -> tuple[list[int], float]: # Question II-1
        """Même algorithme que pruning() (Dijkstra + Pareto Pruning) mais avec un graphe implicite pour réduire la mémoire.
        
        Intègre des possibilités d'optimisation pour réduire l'espace de recherche via des bornes de temps et de fatigue.
        
        Complexité
        ----------
        Temporelle : O(mlog(m) + m⋅L⋅(L+log(n⋅L)))= O(m⋅L⋅log(n⋅L)))
        Spatiale : O(n⋅L+m)
        
        Extensions de pruning
        ---------------------
        f_bound 
            Facultatif plafond de fatigue plus petit depuis la fonction compute_F_bound().
        Si fournie et plus petite que F_max, elle réduit l’espace d’état.

        time_limit
            Borne supérieure sur la distance totale.

        neighbor_fn récupère self.F_max dynamiquement. 
        Modifier cette valeur après la construction du graphe mais avant l'exploration garantit que l'encodage des IDs reste cohérent

        Paramètres
        ----------
        start
            Point Initial, représenté par le numéro du sommet correspondant.
        goal_node 
            Point Final, représenté par le numéro du sommet correspondant.
        time_limit 
            Borne de temps/distance. On élague toute branche dépassant ce coût.
        f_bound 
            Borne de fatigue supérieure. Si fournie, elle remplace F_max pour restreindre l'espace de recherche.

        Retours
        -------
        tuple[list[int], float]
            1) Il existe un chemin de start vers goal : le chemin optimal (liste de sommets) et le coût total (distance pondérée).
            2) Aucun chemin n'existe de start vers goal : ([], float('inf'))
        """
        # --- ÉTAPE 1 : CONFIGURATION DU GRAPHE IMPLICITE ---
        # On initialise le générateur d'arêtes. 
        # build_extended_implicit_graph calcule F_max par défaut via compute_F_max.
        graph = self.build_extended_implicit_graph()

        # Si on a une borne de fatigue plus précise (ex: via ACO), on l'applique ICI.
        # On le fait APRÈS la création du graphe pour que les calculs de voisins utilisent la borne la plus restrictive.
        if f_bound is not None and f_bound < self.F_max:
            self.F_max = f_bound
        enc_start = self._encode(start, 1)
        
        # Initialisation des frontières.
        pareto_fronts: dict[int, list[tuple[float, int]]] = {}
        for node in range(self.n):
            pareto_fronts[node] = []
        
        # --- ÉTAPE 2 : BOUCLE DIJKSTRA OPTIMISÉE ---
        pq: list[tuple[float, int]] = [(0.0, enc_start)]
        visited: set[int] = set()
        distances: dict[int, float] = {enc_start: 0.0}
        parent: dict[int, int] = {enc_start: -1}
        best_arrival_enc = None
        
        while pq:
            dist, u_enc = heapq.heappop(pq)
            if u_enc in visited: # Ignorer si l'état a déjà été traité.
                continue
            visited.add(u_enc)
            
            # Décodage du nœud étendu.
            node_u = u_enc // (self.F_max + 1)
            fatigue_u = u_enc % (self.F_max + 1)
            
            # --- PRUNING DU NŒUD COURANT ---
            # On vérifie si le chemin actuel vers node_u est dominé par un chemin trouvé précédemment.
            new_frontier = [(dist, fatigue_u)]
            is_dominated = False
            for (d_old, f_old) in pareto_fronts[node_u]:
                if d_old <= dist and f_old <= fatigue_u:
                    is_dominated = True
                    break
                else:
                    if not (dist <= d_old and fatigue_u <= f_old):
                        new_frontier.append((d_old, f_old))
            if is_dominated:
                continue
            pareto_fronts[node_u] = new_frontier
            if node_u == goal_node:  
                best_arrival_enc = u_enc
                break
            
            # --- EXPLORATION DES VOISINS (APPROCHE IMPLICITE) ---
            for edge in graph._neighbors(u_enc):
                v_enc = edge.to
                weight = edge.weight
                j = v_enc // (self.F_max + 1)
                fatigue_v = v_enc % (self.F_max + 1)
                new_dist = dist + weight
                if new_dist > time_limit:
                    continue
                if v_enc in visited:
                    continue
                is_v_dominated = False 
                # Vérifier si ce voisin est dominé
                for (d_frontier, f_frontier) in pareto_fronts[j]:
                    if d_frontier <= new_dist and f_frontier <= fatigue_v:
                        is_v_dominated = True
                        break
                
                # Si non dominé et mieux, ajouter à la file.
                if not is_v_dominated:
                    if v_enc not in distances or new_dist < distances[v_enc]:
                        distances[v_enc] = new_dist
                        parent[v_enc] = u_enc
                        heapq.heappush(pq, (new_dist, v_enc))
        
        # --- ÉTAPE 3 : RECONSTRUCTION DU CHEMIN ---
        if best_arrival_enc is None:
            return [], float('inf')
        path_enc = []
        cur = best_arrival_enc
        while cur != -1:
            path_enc.append(cur)
            cur = parent.get(cur, -1)
        path_enc.reverse()
        path = [node // (self.F_max + 1) for node in path_enc]
        best_arrival_dist = distances[best_arrival_enc]
        return path, best_arrival_dist

    def pruning_single_leg_explicit(self,source: int,target: int,current_fatigue: int,total_F_max: int) -> tuple[list[int], float, int]: # Extension
        """Adaptateur de l'algorithme de pruning explicite pour un trajet unique (leg) avec fatigue initiale arbitraire et F_max pré-calculé.
        
        Complexité
        ----------
        O(m⋅L⋅log(n⋅L))
        
        Paramètres
        ----------
        source        : nœud de départ (ID brut, non encodé)
        target        : nœud d'arrivée (ID brut, non encodé)
        current_fatigue : niveau de fatigue au début de ce trajet
        total_F_max   : valeur de F_max calculée globalement (pour tous les trajets)

        Retours
        -------
        (path, distance, final_fatigue)
        - path           : liste d'IDs bruts de source à target
        - distance       : coût total avec fatigue
        - final_fatigue  : niveau de fatigue en fin de trajet
        """
        def _enc(v: int, F: int) -> int:
            return v * (total_F_max + 1) + F

        # Pré-traitement adjacence
        adj: dict[int, list[tuple[int, int, int]]] = {}
        for line in self.df.itertuples(index=False):
            i, j, l, f = int(line[0]), int(line[1]), int(line[2]), int(line[3])
            adj.setdefault(i, []).append((j, l, f))

        enc_start = _enc(source, current_fatigue)

        # Frontières de Pareto par nœud original
        pareto_fronts: dict[int, list[tuple[float, int]]] = {node: [] for node in range(self.n)}

        pq: list[tuple[float, int]] = [(0.0, enc_start)]
        visited: set[int] = set()
        distances: dict[int, float] = {enc_start: 0.0}
        parent: dict[int, int] = {enc_start: -1}
        best_arrival_enc = None

        while pq:
            dist, u_enc = heapq.heappop(pq)

            if u_enc in visited:
                continue
            visited.add(u_enc)

            node_u = u_enc // (total_F_max + 1)
            fatigue_u = u_enc % (total_F_max + 1)

            # Test de dominance + mise à jour frontière
            new_frontier = [(dist, fatigue_u)]
            is_dominated = False

            for (d_old, f_old) in pareto_fronts[node_u]:
                if d_old <= dist and f_old <= fatigue_u:
                    is_dominated = True
                    break
                else:
                    if not (dist <= d_old and fatigue_u <= f_old):
                        new_frontier.append((d_old, f_old))

            if is_dominated:
                continue
            pareto_fronts[node_u] = new_frontier

            if node_u == target:
                best_arrival_enc = u_enc
                break

            for (j, l_ij, f_ij) in adj.get(node_u, []):
                fatigue_v = fatigue_u + f_ij
                if fatigue_v > total_F_max:
                    continue
                v_enc = _enc(j, fatigue_v)
                new_dist = dist + (l_ij * fatigue_u)

                if v_enc in visited:
                    continue

                is_v_dominated = False
                for (d_frontier, f_frontier) in pareto_fronts[j]:
                    if d_frontier <= new_dist and f_frontier <= fatigue_v:
                        is_v_dominated = True
                        break

                if not is_v_dominated:
                    if v_enc not in distances or new_dist < distances[v_enc]:
                        distances[v_enc] = new_dist
                        parent[v_enc] = u_enc
                        heapq.heappush(pq, (new_dist, v_enc))

        if best_arrival_enc is None:
            return [], float('inf'), current_fatigue

        # Reconstruction du chemin
        path_enc = []
        cur = best_arrival_enc
        while cur != -1:
            path_enc.append(cur)
            cur = parent.get(cur, -1)
        path_enc.reverse()

        path_decoded = [node // (total_F_max + 1) for node in path_enc]
        final_fatigue = best_arrival_enc % (total_F_max + 1)
        dist_total = distances[best_arrival_enc]

        return path_decoded, dist_total, final_fatigue

    def compute_admissible_heuristic(self, adj_reversed: dict = None) -> dict[int, float]: # Question II-2
        """Calcule la distance la plus courte (sans fatigue) de chaque ville vers le but.
        
        Cette valeur servira pour le calcul de l'heuristique dans l'algorithme A*. 
        
        Complexité 
        ----------
        Temporelle : O(m + nlogn + mlogn)= O((n+m)log(n))
            Dijkstra et construction du graphe inversé.
        Spatiale : O(n+m)
        
        Paramètres
        ----------
        adj_reversed : dict, optional
                Graphe inversé pré-construit. Si None, sera construit à partir du DataFrame.
        
        Retours
        -------
        dict[int, float]
            Dictionnaire h où h[node] = distance minimale de node à self.goal (h[self.goal] = 0).
        """
        # --- ÉTAPE 1 : Construction du graphe inversé (si non fourni) ---
        # Pour savoir à quelle distance chaque sommet est du but, le plus simple est de partir du but et de remonter les routes à l'envers.
        if adj_reversed is None:
            roads_reversed = {}
            for line in self.df.itertuples(index=False):
                i = int(line[0])
                j = int(line[1])
                l_ij = int(line[2])
                # Ajouter l'arête inversée j → i
                roads_reversed.setdefault(j, []).append((i, l_ij))
        else:
            roads_reversed = adj_reversed
        
        # --- ÉTAPE 2 : Dijkstra "à l'envers" ---
        # On lance un Dijkstra classique à partir de la destination (self.goal) pour calculer la distance vers tous les autres sommets du réseau.
        h = {self.goal: 0}
        pq: list[tuple[float, int]] = [(0, self.goal)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > h.get(u, float('inf')):
                continue
            
            # On explore les prédécesseurs de u (car le graphe est inversé).
            for v, length in roads_reversed.get(u, []):
                new_dist = d + length
                if new_dist < h.get(v, float('inf')):
                    h[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
        return h
    
    def astar(self, start: int, goal_node: int) -> tuple[list[int], float]: # Question II-2
        """Résout le problème de plus court chemin avec fatigue via l'algorithme A*.
        
        Complexité
        ----------
        Temporelle : O((n⋅F_max +m⋅F max)log(n⋅F max))
        Spatiale : O((n+m)⋅F max)

        Paramètres
        ----------
        start 
            Nœud de départ brut (identifiant dans [0, n)).
        goal_node 
            Nœud d'arrivée brut (identifiant dans [0, n), NON encodé).
        
        Retours
        -------
        tuple[list[int], float]
            1) Il existe un chemin de start vers goal : le chemin optimal (liste de sommets) et le coût total (distance pondérée).
            2) Aucun chemin n'existe de start vers goal : ([], float('inf')).
        """
        # --- ÉTAPE 1 : Pré-traitement et Heuristique ---
        # On encode le départ avec une fatigue initiale de 1.
        self.compute_F_max()
        assert 0 <= start < self.n and 0 <= goal_node < self.n, (
            f"start and goal_node must be raw node ids in [0, {self.n}), got {start} and {goal_node}"
        )
        enc_start = self._encode(start, 1)
        
        # Construire l'adjacence UNE SEULE FOIS au départ
        adj = {}
        adj_reversed = {}
        for line in self.df.itertuples(index=False):
            i, j, l, f = int(line[0]), int(line[1]), int(line[2]), int(line[3])
            adj.setdefault(i, []).append((j, l, f)) # Graphe direct (avec fatigue).
            adj_reversed.setdefault(j, []).append((i, l)) # Graphe inversé (sans fatigue, pour l'heuristique).
        h = self.compute_admissible_heuristic(adj_reversed=adj_reversed)

        # --- ÉTAPE 2 : Initialisation de A* ---
        d = {enc_start: 0.0}
        parent = {enc_start: -1}
        start_node = enc_start // (self.F_max + 1)
        start_fatigue = enc_start % (self.F_max + 1) 
        h_start = h.get(start_node, float('inf'))
        pq: list[tuple[float, float, int]] = [(h_start, 0.0, enc_start)]

        # --- ÉTAPE 3 : Boucle de recherche ---
        while pq:
            f_score, t, u_enc = heapq.heappop(pq)
            if t > d.get(u_enc, float('inf')):
                continue
            
            # Décoder l'état actuel.
            node_u = u_enc // (self.F_max + 1)
            fatigue_u = u_enc % (self.F_max + 1)
            
            if node_u == goal_node: 
                path_enc = []
                cur = u_enc
                while cur != -1:
                    path_enc.append(cur)
                    cur = parent.get(cur, -1)
                path_enc.reverse()
                path = [node // (self.F_max + 1) for node in path_enc]
                return path, t
            
            # Exploration des voisins.
            for (j, l_ij, f_ij) in adj.get(node_u, []):
                fatigue_v = fatigue_u + f_ij
                if fatigue_v > self.F_max:
                    continue
                t_new = t + (l_ij * fatigue_u)   # Calcul du coût g(v) = g(u) + (longueur * fatigue_actuelle).
                v_enc = self._encode(j, fatigue_v)
                if v_enc not in d or t_new < d[v_enc]:    # Mettre à jour si on a trouvé un meilleur chemin vers cet état.
                    d[v_enc] = t_new
                    parent[v_enc] = u_enc
                    # Calcul de f(v) = g(v) + h(v)
                    # On multiplie la distance brute par fatigue_v car la fatigue ne peut qu'augmenter par la suite (heuristique admissible).
                    # h(v) = distance_brute × fatigue_en_v)
                    h_v = h.get(j, float('inf')) * fatigue_v
                    f_new = t_new + h_v
                    heapq.heappush(pq, (f_new, t_new, v_enc))
        #Aucun chemin trouvé.
        return [], float('inf')

    def astar_implicit(self, start: int, goal_node: int) -> tuple[list[int], float]:  # Question II-2
        """Résout le problème de chemin optimal avec fatigue via A* sur un graphe implicite étendu.
        
        Même optimalité que astar() mais avec graphe implicite pour économiser la mémoire.

        Complexité
        ----------
        Temporelle : O(((N_ext + M_ext) * recalculs) log N_ext), où N_ext= nombre de sommet du graphe étendu, M_ext= nombre d’arcs du graphe étendu.
        Spatiale : O(n⋅F max + m).
        
        Paramètres
        ----------
        start
            Nœud de départ brut (identifiant dans [0, n)).
        goal_node 
            Nœud d'arrivée brut (identifiant dans [0, n), NON encodé).
        
        Retours
        -------
            tuple[list[int], float]
            1) Il existe un chemin de start vers goal : le chemin optimal (liste de sommets) et le coût total (distance pondérée).
            2) Aucun chemin n'existe de start vers goal : ([], float('inf')).
        """
        self.compute_F_max()
        assert 0 <= start < self.n and 0 <= goal_node < self.n, (
            f"start and goal_node must be raw node ids in [0, {self.n}), got {start} and {goal_node}"
        )
        enc_start = self._encode(start, 1)
        
        # Construire le graphe implicite une seule fois.
        graph = self.build_extended_implicit_graph()
        
        # Construire le graphe inversé pour l'heuristique.
        adj_reversed = {}
        for line in self.df.itertuples(index=False):
            i, j, l = int(line[0]), int(line[1]), int(line[2])
            adj_reversed.setdefault(j, []).append((i, l))
        
        # Pré-calculer l'heuristique une seule fois.
        h = self.compute_admissible_heuristic(adj_reversed=adj_reversed)
        
        # Initialisation
        d = {enc_start: 0.0}
        parent = {enc_start: -1}
        
        # Décoder l'état de départ.
        start_node = enc_start // (self.F_max + 1)
        h_start = h.get(start_node, float('inf'))
        pq: list[tuple[float, float, int]] = [(h_start, 0.0, enc_start)]
        while pq:
            f_score, t, u_enc = heapq.heappop(pq)
            
            # Staleness check AVANT goal check (ordre CRUCIAL pour optimalité).
            if t > d.get(u_enc, float('inf')):
                continue

            # Décoder l'état actuel.
            node_u = u_enc // (self.F_max + 1)
            fatigue_u = u_enc % (self.F_max + 1)
            
            # Vérifier si on atteint le goal (maintenant on est sûr que c'est optimal).
            if node_u == goal_node:
                
                # Reconstruction du chemin.
                path_enc = []
                cur = u_enc
                while cur != -1:
                    path_enc.append(cur)
                    cur = parent.get(cur, -1)
                path_enc.reverse()
                path = [node // (self.F_max + 1) for node in path_enc]
                return path, t
            
            # Explorer les voisins via le graphe implicite.
            # Attention : ceci recalcule les arêtes à chaque extraction !
            for edge in graph._neighbors(u_enc):
                v_enc = edge.to
                weight = edge.weight
                j = v_enc // (self.F_max + 1)
                fatigue_v = v_enc % (self.F_max + 1)
                
                # Calculer le nouveau coût réel.
                t_new = t + weight
                
                # Mettre à jour si on a trouvé un meilleur chemin vers cet état.
                if v_enc not in d or t_new < d[v_enc]:
                    d[v_enc] = t_new
                    parent[v_enc] = u_enc
                    
                    # Calculer f_score = g + h.
                    # Admissibilité garantie : coût_réel ≥ fatigue_v × h(j).
                    h_v = h.get(j, float('inf')) * fatigue_v
                    f_new = t_new + h_v
                    heapq.heappush(pq, (f_new, t_new, v_enc))
        
        # Aucun chemin trouvé
        return [], float('inf')

    def multi_mission_path(self, missions: list[tuple[int, int]], initial_fatigue: int = 1) -> tuple[list[int], float, int]: # Extension
        """Résout une séquence de missions dans un ordre imposé avec propagation de la fatigue.

        L'agent effectue les missions dans l'ordre fourni.
        Entre deux missions consécutives, si le point d'arrivée de la mission i diffère du point de départ de la mission i+1, l'agent effectue automatiquement un trajet intermédiaire en conservant sa fatigue.
        
        Complexité 
        ---------- 
        O(k * (N_ext + M_ext) * log(N_ext)) où k = nombre de trajets, N_ext = n * F_max_total, M_ext = m * F_max_total.

        Paramètres
        ----------
        missions 
            Liste de missions [(vs1, vt1), (vs2, vt2), ...].
            Les identifiants sont des entiers (obtenus via self.dict_matching).
        initial_fatigue 
            Fatigue au départ de la première mission (1 par défaut)

        Retours
        -------
        tuple[list[int], float, int]
            (path, total_time, final_fatigue) où :
            - path          : chemin complet en identifiants de nœuds originaux
            - total_time    : temps total cumulé jusqu'à la fin de la dernière mission
            - final_fatigue : niveau de fatigue à l'arrivée finale
            En cas d'infaisabilité, retourne ([], float('inf'), current_fatigue).

        Exemple
        -------
        >>> net = Network("examples/small.txt")
        >>> path, time, F = net.multi_mission_path([(net.start, net.goal)])
        >>> print(time)   # même résultat que Q3
        125.0
        """
        if not missions:
            return [], 0.0, initial_fatigue

        # --- Coefficient de fatigue maximal sur les arêtes ---
        max_f_coeff: int = max(
            (int(line[3]) for line in self.df.itertuples(index=False)), default=0
        )

        # --- Construction de la liste ordonnée des trajets ---
        # Inclut les déplacements inter-missions si vs_{i+1} ≠ vt_i
        legs: list[tuple[int, int]] = []
        for idx, (vs, vt) in enumerate(missions):
            if idx > 0:
                prev_vt = missions[idx - 1][1]
                if prev_vt != vs:
                    legs.append((prev_vt, vs))  # déplacement entre missions
            legs.append((vs, vt))

        # --- F_max suffisant pour couvrir la fatigue accumulée sur tous les trajets ---
        # Chaque trajet (chemin simple) traverse au plus n-1 arêtes avec f ≤ max_f_coeff.
        num_legs = len(legs)
        total_F_max: int = max(1, initial_fatigue + num_legs * max_f_coeff * self.n)

        # --- Pré-traitement du graphe original (une seule fois) ---
        edges_dict: dict[int, list[tuple[int, int, int]]] = {}
        for line in self.df.itertuples(index=False):
            i, j = int(line[0]), int(line[1])
            l_ij, f_ij = int(line[2]), int(line[3])
            edges_dict.setdefault(i, []).append((j, l_ij, f_ij))

        N_ext = self.n * (total_F_max + 1)
        virtual_goal_id = N_ext

        def _enc(v: int, F: int) -> int:
            return v * (total_F_max + 1) + F

        def _build_leg_graph(target_node: int) -> GraphImplicit:
            """Graphe implicite étendu avec nœud-but virtuel pour target_node."""
            _t = target_node

            def neighbor_fn(label_id: int) -> list[tuple[int, float]]:
                if label_id == virtual_goal_id:
                    return []
                v = label_id // (total_F_max + 1)
                F = label_id % (total_F_max + 1)
                nbrs: list[tuple[int, float]] = []
                if v == _t:
                    nbrs.append((virtual_goal_id, 0.0))
                for nj, l_ij, f_ij in edges_dict.get(v, []):
                    F_new = F + f_ij
                    if F_new <= total_F_max:
                        nbrs.append((_enc(nj, F_new), float(l_ij * F)))
                return nbrs

            return GraphImplicit(n=N_ext + 1, neighbor_fn=neighbor_fn)

        # --- Résolution séquentielle des trajets ---
        full_path: list[int] = []
        total_time = 0.0
        current_fatigue = initial_fatigue

        for source, target in legs:
            if source == target:
                # Trajet trivial : pas de déplacement nécessaire
                if not full_path:
                    full_path = [source]
                continue

            g = _build_leg_graph(target)
            enc_start = _enc(source, current_fatigue)

            path_enc, dist = g.shortest_path(enc_start, virtual_goal_id)

            if not path_enc or dist == float("inf"):
                return [], float("inf"), current_fatigue

            # Exclure le nœud-but virtuel en fin de chemin
            real_path_enc = path_enc[:-1]
            path_decoded = [node // (total_F_max + 1) for node in real_path_enc]

            # La fatigue finale = niveau encodé dans le dernier état réel
            current_fatigue = real_path_enc[-1] % (total_F_max + 1)

            # Concaténation (on évite de dupliquer le nœud de jonction)
            if full_path:
                full_path.extend(path_decoded[1:])
            else:
                full_path.extend(path_decoded)

            total_time += dist

        return full_path, total_time, current_fatigue

    def multi_mission_path_pruning(self, missions: list[tuple[int, int]], initial_fatigue: int = 1) -> tuple[list[int], float, int]: # Extension
        """Même fonctionnalité que multi_mission_path() mais utilise l'algorithme de Pruning explicite (Pareto dominance) au lieu de Dijkstra sur graphe implicite.

        Complexité
        ----------
        O(m+k⋅(m⋅L⋅log(n⋅L)))
        
        Paramètres
        ----------
        missions     
            liste de missions [(vs1, vt1), ...]
        initial_fatigue 
            fatigue initiale (1 par défaut)

        Retours
        -------
        (path, total_time, final_fatigue)
        """
        if not missions:
            return [], 0.0, initial_fatigue

        max_f_coeff: int = max(
            (int(line[3]) for line in self.df.itertuples(index=False)), default=0
        )

        legs: list[tuple[int, int]] = []
        for idx, (vs, vt) in enumerate(missions):
            if idx > 0:
                prev_vt = missions[idx - 1][1]
                if prev_vt != vs:
                    legs.append((prev_vt, vs))
            legs.append((vs, vt))

        num_legs = len(legs)
        total_F_max: int = max(1, initial_fatigue + num_legs * max_f_coeff * self.n)

        full_path: list[int] = []
        total_time = 0.0
        current_fatigue = initial_fatigue

        for source, target in legs:
            if source == target:
                if not full_path:
                    full_path = [source]
                continue

            path_decoded, dist, current_fatigue = self.pruning_single_leg_explicit(
                source, target, current_fatigue, total_F_max
            )

            if not path_decoded or dist == float("inf"):
                return [], float("inf"), current_fatigue

            if full_path:
                full_path.extend(path_decoded[1:])
            else:
                full_path.extend(path_decoded)

            total_time += dist

        return full_path, total_time, current_fatigue

    def optimal_mission_order(self, missions: list[tuple[int, int]], initial_fatigue: int = 1) -> tuple[list[tuple[int, int]], list[int], float, int]: # Extension
        """Trouve l'ordre de missions minimisant le temps total (ordre libre).

        Énumère toutes les permutations possibles et évalue chacune via multi_mission_path. 
        L'agent démarre au point de départ de la première mission de chaque permutation.

        Complexité 
        ----------
        O(k! * T_multi) où T_multi est le coût d'un appel à multi_mission_path. 
        Pratique pour k ≤ 6 environ (720 permutations).

        Paramètres
        ----------
        missions 
            Liste des missions dont l'ordre est libre.
        initial_fatigue 
            Fatigue initiale (1 par défaut)

        Retours
        -------
        tuple[list[tuple[int,int]], list[int], float, int]
            (best_order, path, total_time, final_fatigue)
            - best_order    : permutation optimale des missions
            - path          : chemin complet correspondant
            - total_time    : temps total minimal trouvé
            - final_fatigue : fatigue à la fin du meilleur trajet

        Exemple
        -------
        >>> net = Network("examples/small.txt")
        >>> missions = [(net.start, net.goal), (net.start, net.goal)]
        >>> order, path, time, F = net.optimal_mission_order(missions)
        """
        best_order: list[tuple[int, int]] = list(missions)
        best_path: list[int] = []
        best_time = float("inf")
        best_fatigue = initial_fatigue
        for perm in permutations(missions):
            perm_list = list(perm)
            path, time, fatigue = self.multi_mission_path(perm_list, initial_fatigue)
            if time < best_time:
                best_time = time
                best_order = perm_list
                best_path = path
                best_fatigue = fatigue
        return best_order, best_path, best_time, best_fatigue

if __name__ == "__main__":
    network = Network(filename='examples/small.txt')
    g = network.build_extended_graph()
    print(g.shortest_path(network.start, network.goal))
