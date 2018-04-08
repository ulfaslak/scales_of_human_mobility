import networkx as nx
import matplotlib.pylab as plt
import community
import numpy as np

def draw(G, partition=False, colormap='rainbow', labels=None):
    """Draw graph G in my standard style.

    Uses graphviz. Do `conda install graphviz` if not installed.

    Input
    -----
    G : networkx graph
    partition : bool
    colormap : matplotlib colormap
    labels : dict (Node labels in a dictionary keyed by node of text labels)
    """

    def shuffle_list(l):
        l_out = list(l)[:]
        np.random.shuffle(l_out)
        return l_out
    
    def _get_cols(partition):
        return dict(
            zip(
                shuffle_list(set(partition.values())),
                np.linspace(0, 256, len(set(partition.values()))).astype(int)
            )
        )

    cmap = plt.get_cmap(colormap)
    if partition == True:
        partition = community.best_partition(G)
        cols = _get_cols(partition)
        colors = [cmap(cols[partition[n]]) for n in G.nodes()]
    elif type(partition) is dict and len(partition) >= len(G.nodes()):
        cols = _get_cols(partition)
        colors = [cmap(cols[partition[n]]) for n in G.nodes()]
    elif type(partition) in [list, tuple] and len(partition) == len(G.nodes()):
        colors = list(partition)
    else:
        try:
            colors = [n[1]['node_color'] for n in G.nodes(data=True)]
        except KeyError:
            # nodes do not have node_color attribute
            colors = "grey"
    
    pos = nx.nx_pydot.graphviz_layout(G, prog='neato')
    nx.draw_networkx_edges(G, pos=pos, width=2, alpha=.3, zorder=-10)
    nx.draw_networkx_nodes(G, pos=pos, node_size=120, alpha=1, linewidths=0, node_color=colors)
    
    if labels is not None:
        nx.draw_networkx_labels(G, pos=dict((k, (v[0]+15, v[1])) for k, v in pos.items()), labels=labels, font_size=16)

    plt.axis("off")