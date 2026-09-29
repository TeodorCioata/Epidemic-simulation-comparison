"""Per-edge rates (attribute "rate"), R_0 estimators and the matching protocols (i) and (ii)."""

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.sparse.linalg import eigs

from episim.networks import degrees, edge_array, edge_values


def set_rates(graph: nx.Graph, c: float, alpha: float = 0.0) -> None:
    """tau_ij = c w_ij^alpha = tau_M (w_ij / w_M)^alpha with tau_M = c w_M^alpha [10]."""
    rates = c * edge_values(graph, "weight") ** alpha
    nx.set_edge_attributes(graph, dict(zip(graph.edges, rates.tolist(), strict=True)), "rate")


def tau_max(graph: nx.Graph, c: float, alpha: float) -> float:
    """tau_M = c w_M^alpha."""
    return float(c * edge_values(graph, "weight").max() ** alpha)


def transmissibility(rate: np.ndarray, gamma: float, delta: float) -> np.ndarray:
    """T = tau / (tau + gamma + delta), the chance to transmit before recovering or dying."""
    rate = np.asarray(rate, dtype=float)
    return rate / (rate + gamma + delta)


def r0_newman(graph: nx.Graph, transmissibilities: np.ndarray) -> float:
    """R_0 = <T> (<k^2> - <k>) / <k> (Newman [8])."""
    k = degrees(graph).astype(float)
    return float(np.mean(transmissibilities) * ((k**2).mean() - k.mean()) / k.mean())


def nonbacktracking_pattern(graph: nx.Graph) -> sp.csr_matrix:
    """Sparsity of B[(i->j),(j->l)] = T_jl for l != i, with data holding each column's edge id."""
    n, edges = len(graph), edge_array(graph)
    n_edges = len(edges)
    size = 2 * n_edges
    if n_edges == n - nx.number_connected_components(graph):
        # A forest's B is nilpotent (radius 0) and ARPACK misreports it, so leave the pattern empty.
        return sp.csr_matrix((size, size), dtype=np.int64)

    # Half-edge h runs tail[h] -> head[h]; h and (h + E) mod 2E are the two directions of an edge.
    tail = np.concatenate([edges[:, 0], edges[:, 1]])
    head = np.concatenate([edges[:, 1], edges[:, 0]])
    by_tail = np.argsort(tail, kind="stable")
    start = np.searchsorted(tail[by_tail], np.arange(n + 1))
    position = np.empty(size, dtype=np.int64)
    position[by_tail] = np.arange(size)

    # Row h continues along every half-edge leaving head[h] except its own reverse.
    count = start[head + 1] - start[head] - 1
    offset = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    reverse_offset = position[(np.arange(size) + n_edges) % size] - start[head]
    offset += offset >= np.repeat(reverse_offset, count)
    rows = np.repeat(np.arange(size), count)
    cols = by_tail[np.repeat(start[head], count) + offset]
    return sp.csr_matrix((cols % n_edges, (rows, cols)), shape=(size, size))


def r0_nonbacktracking(
    graph: nx.Graph, transmissibilities: np.ndarray, pattern: sp.csr_matrix | None = None
) -> float:
    """R_0 = spectral radius of the T-weighted non-backtracking matrix B."""
    pattern = nonbacktracking_pattern(graph) if pattern is None else pattern
    t = np.asarray(transmissibilities, dtype=float)
    if pattern.nnz == 0 or not t.any():
        return 0.0
    matrix = sp.csr_matrix((t[pattern.data], pattern.indices, pattern.indptr), shape=pattern.shape)
    if matrix.shape[0] < 64:
        return float(np.abs(np.linalg.eigvals(matrix.toarray())).max())
    values = eigs(matrix, k=1, which="LM", return_eigenvectors=False, maxiter=10_000)
    return float(np.abs(values).max())


def match_mean_rate(graph: nx.Graph, alpha: float, mean_rate: float) -> float:
    """Protocol (i): c = <tau> / <w^alpha>, so that the mean per-edge rate is <tau>."""
    return float(mean_rate / (edge_values(graph, "weight") ** alpha).mean())


def match_r0(
    graph: nx.Graph,
    alpha: float,
    r0_target: float,
    gamma: float,
    delta: float,
    pattern: sp.csr_matrix | None = None,
) -> float:
    """Protocol (ii): c such that the non-backtracking R_0 of c w^alpha equals r0_target."""
    pattern = nonbacktracking_pattern(graph) if pattern is None else pattern
    shape = edge_values(graph, "weight") ** alpha

    def excess(log_c: float) -> float:
        t = transmissibility(np.exp(log_c) * shape, gamma, delta)
        return r0_nonbacktracking(graph, t, pattern) - r0_target

    ceiling = r0_nonbacktracking(graph, np.ones(len(shape)), pattern)
    if not 0 < r0_target < ceiling:
        raise ValueError(f"R_0 = {r0_target} is unreachable: it lies in (0, {ceiling:.4g}) here")

    k = degrees(graph)
    t_newman = min(r0_target * k.mean() / ((k**2).mean() - k.mean()), 0.99)
    low = high = float(np.log((gamma + delta) * t_newman / (1 - t_newman) / shape.mean()))
    while excess(low) > 0:
        low -= 1.0
    while excess(high) < 0:
        high += 1.0
    return float(np.exp(brentq(excess, low, high, xtol=1e-12)))
