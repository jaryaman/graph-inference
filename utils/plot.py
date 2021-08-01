import networkx as nx
import matplotlib.pyplot as plt

def plot_multigraph(G):
    pos = G.nodes(data="pos")
    nx.draw_networkx_nodes(G, pos, node_color='#ed7c6d', node_size=100, alpha=1)
    nx.draw_networkx_labels(G, pos, font_size=15)
    ax = plt.gca()
    for e in G.edges:
        ax.annotate("",
                    xy=pos[e[0]], xycoords='data',
                    xytext=pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="-", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr', str(0.3 * e[2])
                                                                           ),
                                    ),
                    )
    plt.axis('off')
    return ax