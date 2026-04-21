# Résolution du problème de plus court chemin sous contrainte de fatigue cumulative


## Projet de programmation - 1A ENSAE 2026

Ce dépôt github contient l'implémentation du projet de programmation de 1A à l'ENSAE Paris, dédié à l'optimisation du plus court chemin sous contraintes de fatigue cumulative, via plusiseurs méthodes proposées dans description_projet.pdf

## Membre de l'équipe

- Bastien LEVY-GUINOT
- Niels SOUFFIR

## Description génerale du problème

Le projet consiste à résoudre un problème de recherche de chemin optimal dans un graphe orienté, en introduisant une variable dynamique majeure : la fatigue cumulative. Contrairement aux algorithmes de plus court chemin classiques où le coût des arêtes est statique, notre modèle simule un agent dont l'efficacité décroît à mesure qu'il progresse dans le réseau.

### 1. Modélisation du réseau
Le réseau est représenté par un graphe orienté G=(V,E). Chaque arête e ∈ E reliant un sommet i à un sommet j est définie par deux paramètres:

- La longueur (l_ij) : La distance physique entre les deux sommets.
- Le coefficient de fatigue (f_ij) : L'effort supplémentaire imposé par la traversée de cette arête.

### 2. Mécanique de déplacement et de fatigue
Le coût de traversée d'une arête n'est plus constant mais dépend de l'état interne de l'agent. On définit :

- La fatigue actuelle (F) : Une valeur initiale de 1 qui augmente à chaque déplacement.
- Le temps de trajet (T_ij) : Le coût temporel pour parcourir l'arête (i,j), calculé comme le produit de la longueur par la fatigue actuelle : T_ij=l_ij
x F
- Mise à jour de l'état : Après avoir franchi l'arête, la nouvelle fatigue devient F′=F+f_ij

### 3. Objectif d'optimisation
L'enjeu est de trouver un chemin reliant un sommet source à un sommet cible qui minimise le temps total de parcours.
Ce projet implémente des algorithmes de plus court chemin sur des graphes orientés pondérés, en se concentrant sur la résolution d'un problème de routage intégrant une mécanique de fatigue. L'implémentation comprend :

- Des structures de données de graphes standards (explicites) et implicites.

- L'algorithme de Dijkstra pour les graphes explicites et implicites.

- Optimisation algorithmiques: Pruning et A*

- Extension : Routage séquentiel multi-missions e

- Visualisation de graphes et analyse comparative complète des performances (benchmarking).

- Optimisations Avancées : Élagage par dominance de Pareto (Pareto dominance pruning) et Optimisation par Colonies de Fourmis (ACO).


## Utilisation
### Installation

**Requirements**:
- Python 3.8+
- pandas 
- pytest 
- networkx, matplotlib

Installer les dépendances:
```bash
pip install -r requirements.txt
```

### Tests

Tourner tous les tests:
```bash
pytest test_graph_network.py -v
```

Tourner un test spécifique:
```bash
pytest test_graph_network.py::test_name -v
```
Le projet est modulaire. Vous pouvez activer des questions spécifiques dans le fichier main.py: 
```python
# Dans main.py
main(filename, Q1_enabled=True, Q2_enabled=True, Q3_enabled=True, Q_pruning_enabled=False)
```
Vous pouvez choisir un exemple spécifique dans le fichier main.py: 
```python
# Dans main.py
filename = 'medium-smallfatigue'
```

Puis tourner le code: 
```python
python3 main.py
```

### Structure du projet

```    

├── graph.py                      # Structures de données fondamentales (Graph, GraphImplicit, Edge)
├── network.py                    # Cœur du projet : logique de fatigue et construction des graphes étendus
├── format_examples.py            # Script de conversion des fichiers .txt vers DataFrames/CSV
├── main.py                       # Point d'entrée principal (Exécution des questions Q1 à Q_multi)
├── graph_representation.py       # Génération de visuels des réseaux et trajets avec NetworkX
├── benchmark.py                  # Script d'analyse comparative (temps, mémoire, distance)
├── ants.py                       # Algorithme de Colonie de Fourmis (ACO) et Solveur Hybride
├── requirements.txt              # Liste des dépendances Python (pandas, networkx, matplotlib)
│
├── rapport.pdf                   # Rapport final : analyse des algorithmes et résultats
├── preuve.pdf                    # Démonstration mathématique de la borne de fatigue F_bound
├── description_projet.pdf        # Énoncé officiel et consignes du projet
│
├── benchmark_distance.csv        # Résultats du benchmark : Précision (optimalité)
├── benchmark_memory.csv          # Résultats du benchmark : Consommation RAM (Peak)
├── benchmark_time.csv            # Résultats du benchmark : Temps d'exécution (ms)
│
├── README.md                     # Documentation générale du projet et guide d'utilisation
├── README_ANTS.md                # Documentation spécifique à l'implémentation de l'ACO
├── README_MULTI.md                # Documentation spécifique à quelque fonction de l'extension multi missions
│
├── tests/                        # Dossier contenant les scripts de validation unitaire
│   ├── test_multi_mission.py     # Tests spécifiques pour l'enchaînement de missions
│   ├── test_network.py           # Tests spécifiques pour le fichier network
│   └── test_graph.txt            # Tests spécifiques pour le fichier graph
│
├── examples/                     # Graphes d'entrée au format texte brut (.txt)
│   ├── small.txt                 # Instance de test minimale (4 nœuds)
│   ├── medium-*.txt              # Instances de taille moyenne (réseaux urbains)
│   └── large-*.txt               # Instances complexes (réseaux régionaux)
├── formated_examples/            # Graphes convertis en format tabulaire (.csv)
│   ├── small.csv                 # Données structurées pour le graphe 'small'
│   ├── medium-*.csv              # Données structurées pour les graphes 'medium'
│   └── large-*.csv               # Données structurées pour les graphes 'large'
├── example_graph_visualisation/  # Rendus graphiques exportés sous forme d'images
│   ├── small-*.png               # Visualisation du trajet optimal sur petit graphe
│   └── medium-*.png              # Visualisation du trajet optimal sur graphe moyen
├── example_missions/             # Scénarios de missions multiples (points de passage)
│   ├── medium-*.txt              # Missions définies pour les graphes 'medium'
└── └── large-*.txt               # Missions définies pour les graphes 'large'

```

---



