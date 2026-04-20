# Partie 3 — Missions Multiples

Ce document explique les deux fonctions ajoutées à la classe `Network` dans [network.py](network.py) pour résoudre le problème de missions multiples avec propagation de fatigue.

---

## Contexte : la fatigue

Dans ce projet, traverser une arête `(i, j)` de longueur `l` avec un coefficient de fatigue `f` coûte :

```
temps = l × F_courant
F_nouveau = F_courant + f
```

où `F_courant` est le niveau de fatigue **au moment de l'emprunt**. La fatigue part de 1 et augmente à chaque arête fatiguante. Elle n'est **jamais réinitialisée** entre deux missions.

---

## `multi_mission_path(missions, initial_fatigue=1)`

**Signature :**
```python
net.multi_mission_path(
    missions: list[tuple[int, int]],
    initial_fatigue: int = 1,
) -> tuple[list[int], float, int]
```

**But :** résoudre une séquence ordonnée de missions `[(vs1, vt1), (vs2, vt2), ...]` en propageant la fatigue accumulée d'une mission à la suivante.

### Fonctionnement étape par étape

#### 1. Construction de la liste des trajets (`legs`)

On commence par construire la liste de tous les trajets à effectuer dans l'ordre :

- Pour chaque mission `(vs_i, vt_i)`, on ajoute le trajet `vs_i → vt_i`.
- Si la destination de la mission précédente `vt_{i-1}` **diffère** du départ de la mission `vs_i`, un **trajet intermédiaire** `vt_{i-1} → vs_i` est inséré automatiquement.

Exemple avec les missions `[(lozere, guichet), (ensae, saclay)]` :
```
legs = [(lozere → guichet), (guichet → ensae), (ensae → saclay)]
                                ↑ trajet intermédiaire inséré
```

#### 2. Calcul du `F_max` global

La fatigue peut s'accumuler sur tous les trajets. On calcule une borne supérieure :

```
total_F_max = initial_fatigue + num_legs × max_f_coeff × n
```

où `max_f_coeff` est le coefficient de fatigue maximal sur les arêtes et `n` le nombre de nœuds. Cela couvre le pire cas : chaque trajet emprunte `n-1` arêtes avec la fatigue maximale.

#### 3. Graphe étendu implicite

Pour chaque trajet `(source, target)`, on construit à la volée un **graphe implicite étendu** dont les états sont des paires `(nœud, fatigue)` encodées en un seul entier :

```
encode(v, F) = v × (total_F_max + 1) + F
```

Un **nœud-but virtuel** est ajouté, relié à tous les états `(target, F)` avec un coût de 0. Cela permet de lancer **un seul appel à Dijkstra** et de trouver automatiquement le niveau de fatigue optimal à l'arrivée (au lieu d'itérer sur tous les niveaux possibles).

Le graphe est reconstruit pour chaque trajet car la cible change — mais le dictionnaire d'arêtes `edges_dict` est pré-calculé **une seule fois** avant la boucle.

#### 4. Résolution séquentielle (approche gloutonne)

Pour chaque trajet `(source, target)` :

1. On encode le départ : `enc_start = encode(source, current_fatigue)`.
2. On lance Dijkstra vers le nœud-but virtuel.
3. On décode le chemin et on lit la **fatigue finale** dans l'encodage du dernier état réel.
4. `current_fatigue` est mis à jour → cette valeur devient la fatigue initiale du trajet suivant.
5. Le chemin partiel est concaténé au chemin global (sans dupliquer le nœud de jonction).

> **Limite :** l'approche est **gloutonne** — chaque trajet est optimisé indépendamment. La solution n'est pas nécessairement globalement optimale. Pour trouver l'optimum global sur l'ordre des missions, voir `optimal_mission_order`.

#### 5. Cas limites

| Situation | Comportement |
|---|---|
| Liste de missions vide | Retourne `([], 0.0, initial_fatigue)` |
| `source == target` pour un trajet | Trajet trivial ignoré (pas de déplacement) |
| Destination inatteignable | Retourne `([], float('inf'), current_fatigue)` |

### Retour

```python
(path, total_time, final_fatigue)
```

- `path` : liste des IDs de nœuds du chemin complet
- `total_time` : temps total cumulé
- `final_fatigue` : fatigue à la fin de la dernière mission

### Complexité

**O(k × (N_ext + M_ext) × log N_ext)**

où `k` = nombre de trajets, `N_ext = n × total_F_max`, `M_ext = m × total_F_max`.

### Exemple numérique (small.txt)

Réseau `small.txt` : ensae=0, guichet=1, lozere=2, saclay=3. `start=lozere=2`, `goal=saclay=3`.

```python
net = Network("examples/small.txt")

# k=1 : résultat identique à Q3
path, time, F = net.multi_mission_path([(net.start, net.goal)])
# → path=[2,1,0,3], time=125.0, F=2

# k=2 : fatigue propagée
path, time, F = net.multi_mission_path([(net.start, 0), (0, net.goal)])
# Mission 1 (lozere→ensae) : chemin 2→0, temps=10, F passe à 3
# Mission 2 (ensae→saclay) : chemin 0→3, temps=45×3=135, F=3
# → path=[2,0,3], time=145.0, F=3
```

---

## `optimal_mission_order(missions, initial_fatigue=1)`

**Signature :**
```python
net.optimal_mission_order(
    missions: list[tuple[int, int]],
    initial_fatigue: int = 1,
) -> tuple[list[tuple[int, int]], list[int], float, int]
```

**But :** trouver l'ordre des missions qui **minimise le temps total**, en testant toutes les permutations possibles.

### Fonctionnement

La fonction énumère exhaustivement toutes les `k!` permutations des missions via `itertools.permutations`. Pour chaque permutation, elle appelle `multi_mission_path` et conserve la permutation donnant le temps minimal.

```python
for perm in permutations(missions):
    path, time, fatigue = self.multi_mission_path(list(perm), initial_fatigue)
    if time < best_time:
        best_time = time
        best_order = list(perm)
        ...
```

> **Note :** cette recherche exhaustive est pratique pour `k ≤ 6` (720 permutations). Au-delà, une heuristique ou une méthode de branch-and-bound serait nécessaire.

### Retour

```python
(best_order, path, total_time, final_fatigue)
```

- `best_order` : permutation optimale des missions (liste de tuples)
- `path` : chemin complet correspondant à cet ordre optimal
- `total_time` : temps total minimal trouvé
- `final_fatigue` : fatigue finale pour cet ordre optimal

### Complexité

**O(k! × T_multi)**

où `T_multi = O(k × N_ext × log N_ext)` est le coût d'un appel à `multi_mission_path`.

### Exemple

```python
net = Network("examples/small.txt")
missions = [(net.start, 0), (0, net.goal)]  # lozere→ensae, ensae→saclay

best_order, path, time, F = net.optimal_mission_order(missions)
# Teste [(2,0),(0,3)] et [(0,3),(2,0)]
# → retourne l'ordre de temps minimal
```

---

## Fichiers associés

| Fichier | Rôle |
|---|---|
| [network.py](network.py) | Implémentation de `multi_mission_path` et `optimal_mission_order` |
| [main.py](main.py) | Fonction `Q_multi` et démonstration sur `small.txt` |
| [test_multi_mission.py](test_multi_mission.py) | 8 tests unitaires pytest couvrant tous les cas |
| [graph.py](graph.py) | `GraphImplicit` et algorithme de Dijkstra utilisés en interne |
| [examples/small.txt](examples/small.txt) | Réseau de test à 4 nœuds |
