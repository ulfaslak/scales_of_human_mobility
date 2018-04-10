import networkx as nx

def multigraph_to_weighted_graph(M, digraph=False):
    """Convert a nx.MultiGraph into a weighted nx.Graph."""
    if digraph:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    for u, v, data in M.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if G.has_edge(u, v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
    return G