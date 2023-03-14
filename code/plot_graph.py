import networkx as nx
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
import sys


def plot_graph_test():
    z = [5, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1]
    print(nx.is_graphical(z))

    print("Configuration model")
    G = nx.configuration_model(z)  # configuration model
    degree_sequence = [d for n, d in G.degree()]  # degree sequence
    print(f"Degree sequence {degree_sequence}")
    print("Degree histogram")
    hist = {}
    for d in degree_sequence:
        if d in hist:
            hist[d] += 1
        else:
            hist[d] = 1
    print("degree #nodes")
    for d in hist:
        print(f"{d:4} {hist[d]:6}")

    nx.draw(G)
    plt.show()


def plot_3dgraph():
    # some graphs to try
    # H=nx.krackhardt_kite_graph()
    # H=nx.Graph();H.add_edge('a','b');H.add_edge('a','c');H.add_edge('a','d')
    # H=nx.grid_2d_graph(4,5)
    H = nx.cycle_graph(20)

    # reorder nodes from 0,len(G)-1
    G = nx.convert_node_labels_to_integers(H)
    # 3d spring layout
    pos = nx.spring_layout(G, dim=3)
    # numpy array of x,y,z positions in sorted node order
    xyz = np.array([pos[v] for v in sorted(G)])
    # scalar colors
    scalars = np.array(list(G.nodes())) + 5

    pts = mlab.points3d(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        scalars,
        scale_factor=0.1,
        scale_mode="none",
        colormap="Blues",
        resolution=20,
    )

    pts.mlab_source.dataset.lines = np.array(list(G.edges()))
    tube = mlab.pipeline.tube(pts, tube_radius=0.01)
    mlab.pipeline.surface(tube, color=(0.8, 0.8, 0.8))
    mlab.show()

    pass


def mayavi_test():
    x, y = np.ogrid[-2:2:160j, -2:2:160j]
    z = abs(x) * np.exp(-x ** 2 - (y / .75) ** 2)
    pl = mlab.surf(x, y, z, warp_scale=2)
    mlab.axes(xlabel='x', ylabel='y', zlabel='z')
    mlab.outline(pl)
    mlab.show()
    pass


def label_color():
    G = nx.cubical_graph()
    pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    options = {"node_size": 500, "alpha": 0.8}
    nx.draw_networkx_nodes(G, pos, nodelist=[0, 1, 2, 3], node_color="r", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=[4, 5, 6, 7], node_color="b", **options)

    # edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(0, 1), (1, 2), (2, 3), (3, 0)],
        width=8,
        alpha=0.5,
        edge_color="r",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(4, 5), (5, 6), (6, 7), (7, 4)],
        width=8,
        alpha=0.5,
        edge_color="b",
    )

    # some math labels
    labels = {}
    labels[0] = r"$a$"
    labels[1] = r"$b$"
    labels[2] = r"$c$"
    labels[3] = r"$d$"
    labels[4] = r"$\alpha$"
    labels[5] = r"$\beta$"
    labels[6] = r"$\gamma$"
    labels[7] = r"$\delta$"
    nx.draw_networkx_labels(G, pos, labels, font_size=16)

    plt.axis("off")
    plt.show()


if __name__ == "__main__":

    # label_color()

    # mayavi_test()

    # plot_graph_test()

    plot_3dgraph()

    pass
