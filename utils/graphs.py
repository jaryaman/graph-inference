import math
from itertools import combinations

import networkx as nx
import numpy as np
from networkx.linalg.graphmatrix import adjacency_matrix
from networkx.utils import nodes_or_number, py_random_state



def make_inter_vertex_distances(G, p=2):
    pos = G.nodes(data="pos")
    n = len(pos)
    distance = np.zeros((n, n))

    for i in range(n)   :
        for j in range(n):
            distance[i, j] = (sum(abs(a - b) ** p for a, b in zip(pos[i], pos[j]))) ** (1 / p)

    return distance


def get_independent_components_rgg(G, distances):
    row, col = np.triu_indices_from(distances, k=1)
    distance_filter = distances[row, col]

    adj = adjacency_matrix(G)
    adj_filter = adj[row, col]

    adj_filter = np.array(adj_filter).ravel()

    return distance_filter, adj_filter


@py_random_state(6)
@nodes_or_number(0)
def soft_random_geometric_graph(
        n, radius, dim=2, pos=None, p=2, p_dist=None, seed=None
):
    n_name, nodes = n
    G = nx.Graph()
    G.name = f"soft_random_geometric_graph({n}, {radius}, {dim})"
    G.add_nodes_from(nodes)
    # If no positions are provided, choose uniformly random vectors in
    # Euclidean space of the specified dimension.
    if pos is None:
        pos = {v: [seed.random() for i in range(dim)] for v in nodes}
    nx.set_node_attributes(G, pos, "pos")

    # if p_dist function not supplied the default function is an exponential
    # distribution with rate parameter :math:`\lambda=1`.
    if p_dist is None:
        def p_dist(dist):
            return math.exp(-dist)

    def should_join(edge):
        u, v = edge
        dist = (sum(abs(a - b) ** p for a, b in zip(pos[u], pos[v]))) ** (1 / p)
        return seed.random() < p_dist(dist)

    G.add_edges_from(filter(should_join, geometric_edges(G, radius, p)))
    return G


def geometric_edges(G, radius, p):
    """Returns edge list of node pairs within `radius` of each other

    Radius uses Minkowski distance metric `p`.
    If scipy available, use scipy cKDTree to speed computation.
    """
    nodes_pos = G.nodes(data="pos")
    try:
        import scipy as sp
        import scipy.spatial  # call as sp.spatial
    except ImportError:
        # no scipy KDTree so compute by for-loop
        radius_p = radius ** p
        edges = [
            (u, v)
            for (u, pu), (v, pv) in combinations(nodes_pos, 2)
            if sum(abs(a - b) ** p for a, b in zip(pu, pv)) <= radius_p
        ]
        return edges
    # scipy KDTree is available
    nodes, coords = list(zip(*nodes_pos))
    kdtree = sp.spatial.cKDTree(coords)  # Cannot provide generator.
    edge_indexes = kdtree.query_pairs(radius, p)
    edges = [(nodes[u], nodes[v]) for u, v in sorted(edge_indexes)]
    return edges


@nodes_or_number(0)
def poissonian_random_geometric_graph(
        n, radius, rng, dim=2, pos=None, p=2, p_dist=None,
):
    n_name, nodes = n
    G = nx.MultiGraph()
    G.name = f"poissonian_random_geometric_graph({n}, {radius}, {dim})"
    G.add_nodes_from(nodes)

    if pos is None:
        pos = {v: [rng.random() for i in range(dim)] for v in nodes}
    nx.set_node_attributes(G, pos, "pos")

    if p_dist is None:
        def p_dist(dist):
            return math.exp(-dist)

    def multiply_edge_poisson(edge):
        u, v = edge
        dist = (sum(abs(a - b) ** p for a, b in zip(pos[u], pos[v]))) ** (1 / p)
        return rng.poisson(p_dist(dist)) * [(u, v)]

    for edge in geometric_edges(G, radius, p):
        G.add_edges_from(multiply_edge_poisson(edge))

    return G


@nodes_or_number(0)
def deg_corrected_poissonian_random_geometric_graph(
        n, radius, ki, p_dist, rng, dim=2, pos=None, p=2,
):
    n_name, nodes = n
    G = nx.MultiGraph()
    G.name = f"deg_corrected_poissonian_random_geometric_graph({n}, {radius}, {dim})"
    G.add_nodes_from(nodes)

    if pos is None:
        pos = {v: [rng.random() for i in range(dim)] for v in nodes}
    nx.set_node_attributes(G, pos, "pos")

    def multiply_edge_poisson(edge):
        u, v = edge
        dist = (sum(abs(a - b) ** p for a, b in zip(pos[u], pos[v]))) ** (1 / p)
        return rng.poisson(p_dist(dist, ki[u], ki[v])) * [(u, v)]

    for edge in geometric_edges(G, radius, p):
        G.add_edges_from(multiply_edge_poisson(edge))

    return G

