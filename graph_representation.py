"""
Représentation visuelle d'un graphe (Network) avec longueurs et coefficients de fatigue.
"""

from __future__ import annotations

from typing import Optional

from network import Network
import networkx as nx
import matplotlib.pyplot as plt

def draw_network(
    net: Network,
    *,
    title: str = "Graphe",
    highlight_path: Optional[list[int]] = None,
    save_path : Optional[str] = None,
) -> None:
    
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Pour la visualisation, installez networkx et matplotlib: "
            "pip install networkx matplotlib"
        )

    G = nx.DiGraph()
    for i in range(net.n):
        G.add_node(i)

    edge_labels: dict[tuple[int, int], str] = {}
    # Iterate through the DataFrame to add edges
    for line in net.df.itertuples(index=False):
        i = int(line[0])        # Node 1
        j = int(line[1])        # Node 2
        l_ij = int(line[2])     # Distance
        f_ij = int(line[3])     # Fatigue
        
        if l_ij > 0:
            G.add_edge(i, j)
            edge_labels[(i, j)] = f"l={l_ij}\nf={f_ij}"

    pos = nx.spring_layout(G, seed=42, k=1.5)

    fig, ax = plt.subplots(figsize=(10, 8))

    node_colors = []
    for node in G.nodes():
        if highlight_path and node in highlight_path:
            node_colors.append("#E74C3C")  # rouge pour le chemin
        elif node == net.start:
            node_colors.append("#90EE90")  # vert clair pour départ
        elif node == net.goal:
            node_colors.append("#FFB6C1")  # rose clair pour arrivée
        else:
            node_colors.append("#87CEEB")  # bleu ciel pour les autres

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=800, ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)

    # Arêtes normales en gris
    nx.draw_networkx_edges(
        G, pos, edge_color="gray", arrows=True, arrowsize=20, ax=ax
    )
    # Arêtes du shortest_path en rouge (plus épaisses)
    if highlight_path and len(highlight_path) >= 2:
        path_edges = list(zip(highlight_path, highlight_path[1:]))
        nx.draw_networkx_edges(
            G, pos, edgelist=path_edges, edge_color="#E74C3C",
            width=3, arrows=True, arrowsize=24, ax=ax
        )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels, font_size=9, ax=ax
    )

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(str("example_graph_visualisation/") + save_path, dpi=150, bbox_inches="tight")
        print(f"Figure sauvegardée dans {save_path}")

    plt.close()


if __name__ == "__main__":
    # Exemple d'utilisation avec la nouvelle classe Network
    net = Network(filename='examples/small.txt')

    # Visualiser le graphe simple
    draw_network(
        net,
        title="Exemple de graphe",
        save_path="small.png"
    )
    
    # Optionnel : visualiser avec un chemin mis en évidence
    # path, dist = net.shortest_path_extented_graph(start_tiredness=1)
    # draw_network(
    #     net,
    #     title=f"Chemin optimal (temps = {dist})",
    #     highlight_path=path,
    #     save_path="graphe_avec_chemin.png"
    # )
