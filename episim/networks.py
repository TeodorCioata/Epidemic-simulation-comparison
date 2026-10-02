"""Topologies at matched <k>, the weight schemes (edge attribute "weight") and their diagnostics."""

import random
from collections.abc import Iterator
from contextlib import contextmanager

import networkx as nx
import numpy as np
from scipy.stats import spearmanr

from episim import main

NETWORK_NAMES = {
    "E": "Erdős-Rényi",
    "B": "Barabási-Albert",
    "C": "configuration model (Poisson degrees)",
    "H": "configuration model (BA degrees)",
    "W": "BBV weighted growth",
}


def create_network(
    kind: str, n: int, mean_degree: int, rng: np.random.Generator, *, reinforcement: float = 1.0
) -> nx.Graph:
    if kind in ("E", "B", "C"):
        seed = _seed(rng)
        with _seeded_global_rngs(seed):
            return main.create_network(kind, n, mean_degree, seed)
    if kind == "H":
        ba_degrees = [k for _, k in create_network("B", n, mean_degree, rng).degree()]
        return _erased(nx.configuration_model(ba_degrees, seed=_seed(rng)))
    if kind == "W":
        return bbv_network(n, mean_degree // 2, reinforcement, rng)
    raise ValueError(f"unknown network kind {kind!r}; expected one of {', '.join(NETWORK_NAMES)}")


def _seed(rng: np.random.Generator) -> int:
    return int(rng.integers(2**32))


@contextmanager
def _seeded_global_rngs(seed: int) -> Iterator[None]:
    saved_numpy, saved_python = np.random.get_state(), random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(saved_numpy)
        random.setstate(saved_python)


def _erased(multigraph: nx.MultiGraph) -> nx.Graph:
    graph = nx.Graph(multigraph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def bbv_network(n: int, m: int, reinforcement: float, rng: np.random.Generator) -> nx.Graph:
    """BBV [1]: attach m edges with P(i) = s_i / sum_j s_j; w_ij += reinforcement w_ij / s_i."""
    graph = nx.complete_graph(m + 1)
    nx.set_edge_attributes(graph, 1.0, "weight")
    strength = np.zeros(n)
    strength[: m + 1] = m
    for new in range(m + 1, n):
        cumulative = np.cumsum(strength[:new])
        targets: set[int] = set()
        while len(targets) < m:
            draws = rng.random(m - len(targets)) * cumulative[-1]
            targets.update(np.searchsorted(cumulative, draws).tolist())
        for i in sorted(targets):
            for j, edge in graph.adj[i].items():
                increment = reinforcement * edge["weight"] / strength[i]
                edge["weight"] += increment
                strength[j] += increment
            graph.add_edge(i, new, weight=1.0)
            strength[i] += reinforcement + 1.0
            strength[new] += 1.0
    return _with_weights(graph, edge_values(graph, "weight"))


def iid_weights(graph: nx.Graph, cv: float, rng: np.random.Generator) -> nx.Graph:
    """Scheme (i) [10]: iid gamma weights, shape 1/CV_w^2, scale CV_w^2, normalised to <w> = 1."""
    size = graph.number_of_edges()
    return _with_weights(graph, rng.gamma(1 / cv**2, cv**2, size) if cv > 0 else np.ones(size))


def degree_weights(graph: nx.Graph, theta: float) -> nx.Graph:
    """Scheme (ii) [2]: w_ij = w_0 (k_i k_j)^theta, with w_0 fixed by <w> = 1."""
    k = degrees(graph).astype(float)
    edges = edge_array(graph)
    return _with_weights(graph, (k[edges[:, 0]] * k[edges[:, 1]]) ** theta)


def _with_weights(graph: nx.Graph, weights: np.ndarray) -> nx.Graph:
    weighted = graph.copy()
    unit_mean = (weights / weights.mean()).tolist()
    nx.set_edge_attributes(weighted, dict(zip(graph.edges, unit_mean, strict=True)), "weight")
    return weighted


def edge_array(graph: nx.Graph) -> np.ndarray:
    return np.array(graph.edges, dtype=np.int64).reshape(-1, 2)


def edge_values(graph: nx.Graph, name: str, default: float = 1.0) -> np.ndarray:
    values = (value for *_, value in graph.edges(data=name, default=default))
    return np.fromiter(values, float, graph.number_of_edges())


def degrees(graph: nx.Graph) -> np.ndarray:
    return np.fromiter((k for _, k in graph.degree()), np.int64, graph.number_of_nodes())


def strengths(graph: nx.Graph) -> np.ndarray:
    """s_i = sum_j w_ij."""
    weights = np.repeat(edge_values(graph, "weight"), 2)
    return np.bincount(edge_array(graph).ravel(), weights=weights, minlength=len(graph))


def weight_cv(graph: nx.Graph) -> float:
    """CV_w = sd(w) / <w>, with the population standard deviation."""
    w = edge_values(graph, "weight")
    return float(w.std() / w.mean())


def weight_degree_correlation(graph: nx.Graph) -> float:
    """Spearman rho(w_ij, k_i k_j); 0 when every weight ties."""
    w = edge_values(graph, "weight")
    if np.all(w == w[0]):
        return 0.0
    k = degrees(graph)
    edges = edge_array(graph)
    return float(spearmanr(w, k[edges[:, 0]] * k[edges[:, 1]]).statistic)


def strength_exponent(graph: nx.Graph) -> float:
    """beta in s(k) ~ k^beta: log mean strength per degree class on log k, weighted by size."""
    k, s = degrees(graph), strengths(graph)
    counts = np.bincount(k[k > 0])
    totals = np.bincount(k[k > 0], weights=s[k > 0])
    classes = np.flatnonzero(counts)
    if classes.size < 2:
        raise ValueError("beta is undefined on a regular graph: only one degree class")
    mean_strength = totals[classes] / counts[classes]
    # polyfit weights residuals, so sqrt(count) weights each class's squared error by its size.
    fit = np.polyfit(np.log(classes), np.log(mean_strength), 1, w=np.sqrt(counts[classes]))
    return float(fit[0])
