"""Per-edge rates (attribute "rate"), R_0, growth rate and final size from the non-backtracking
matrix, and the matching protocols (i), (ii) and (iii)."""

from collections.abc import Callable
from functools import cache

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.sparse.linalg import ArpackNoConvergence, eigs

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
    return _spectral_radius(pattern, transmissibilities)[0]


def _spectral_radius(
    pattern: sp.csr_matrix, transmissibilities: np.ndarray, start: np.ndarray | None = None
) -> tuple[float, np.ndarray | None]:
    """rho(B) and its eigenvector. A fixed start vector makes ARPACK deterministic, and a small
    Krylov space halves its cost; the default one is the fallback when that does not converge."""
    t = np.asarray(transmissibilities, dtype=float)
    if pattern.nnz == 0 or not t.any():
        return 0.0, None
    matrix = sp.csr_matrix((t[pattern.data], pattern.indices, pattern.indptr), shape=pattern.shape)
    if matrix.shape[0] < 64:
        return float(np.abs(np.linalg.eigvals(matrix.toarray())).max()), None
    start = np.ones(matrix.shape[0]) if start is None else start
    try:
        values, vectors = eigs(matrix, k=1, which="LM", v0=start, ncv=8, maxiter=1000)
    except ArpackNoConvergence:
        values, vectors = eigs(matrix, k=1, which="LM", v0=start, maxiter=10_000)
    return float(np.abs(values[0])), vectors[:, 0].real


def _radius_along_a_root_find(pattern: sp.csr_matrix) -> Callable[[np.ndarray], float]:
    """rho(B) as a function of T, each solve warm-started from the previous eigenvector."""
    vector: np.ndarray | None = None

    def radius(transmissibilities: np.ndarray) -> float:
        nonlocal vector
        value, vector = _spectral_radius(pattern, transmissibilities, vector)
        return value

    return radius


def unit_radius(graph: nx.Graph, pattern: sp.csr_matrix | None = None) -> float:
    """rho(B) of the unweighted graph: the R_0 ceiling (T = 1), and M1's R_0 is T(tau) times it."""
    return r0_nonbacktracking(graph, np.ones(graph.number_of_edges()), pattern)


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
    rho_unit: float | None = None,
) -> float:
    """Protocol (ii): c such that the non-backtracking R_0 of c w^alpha equals r0_target."""
    pattern = nonbacktracking_pattern(graph) if pattern is None else pattern
    shape = edge_values(graph, "weight") ** alpha
    radius = _radius_along_a_root_find(pattern)

    @cache
    def excess(log_c: float) -> float:
        return radius(transmissibility(np.exp(log_c) * shape, gamma, delta)) - r0_target

    ceiling = unit_radius(graph, pattern) if rho_unit is None else rho_unit
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


def discounted_transmissibility(
    rate: np.ndarray, sigma: float, gamma: float, delta: float, r: float
) -> np.ndarray:
    """T_r = sigma / (sigma + r) * tau / (tau + gamma + delta + r), the Laplace transform at r of
    the time from infection to transmission along an edge; T_0 = T."""
    return sigma / (sigma + r) * transmissibility(rate, gamma + r, delta)


def growth_rate_uniform(
    tau: float, rho_unit: float, sigma: float, gamma: float, delta: float
) -> float:
    """M1's r: T_r rho(B_unit) = 1, i.e. (sigma + r)(tau + gamma + delta + r) = sigma tau rho."""
    exit_rate = tau + gamma + delta
    discriminant = (sigma - exit_rate) ** 2 + 4 * sigma * tau * rho_unit
    return float((np.sqrt(discriminant) - sigma - exit_rate) / 2)


def growth_rate(
    graph: nx.Graph,
    sigma: float,
    gamma: float,
    delta: float,
    pattern: sp.csr_matrix | None = None,
    rho_unit: float | None = None,
) -> float:
    """Early growth rate r of the rates on the graph: rho(B(r)) = 1 with B built from T_r, so
    r = 0 exactly when R_0 = 1 and r < 0 below the threshold."""
    pattern = nonbacktracking_pattern(graph) if pattern is None else pattern
    rho_unit = unit_radius(graph, pattern) if rho_unit is None else rho_unit
    rates = edge_values(graph, "rate")
    radius = _radius_along_a_root_find(pattern)

    @cache
    def excess(r: float) -> float:
        return radius(discounted_transmissibility(rates, sigma, gamma, delta, r)) - 1.0

    def mean_field_excess(r: float) -> float:
        return discounted_transmissibility(rates, sigma, gamma, delta, r).mean() * rho_unit - 1.0

    # T_r increases with tau, so the uniform closed forms at the extreme rates bracket r.
    pole = -min(sigma, gamma + delta)
    low = max(growth_rate_uniform(rates.min(), rho_unit, sigma, gamma, delta), pole * (1 - 1e-9))
    high = growth_rate_uniform(rates.max(), rho_unit, sigma, gamma, delta)
    if not low < high:
        return high
    # <T_r> rho_unit = 1 is exact for rates independent of the structure: narrow the bracket to it.
    guess = brentq(mean_field_excess, low, high)
    step = 0.05 * (1 + abs(guess))
    if excess(below := max(low, guess - step)) >= 0:
        low = below
    if excess(above := min(high, guess + step)) <= 0:
        high = above
    return float(brentq(excess, low, high, xtol=1e-12))


def match_growth_rate(
    graph: nx.Graph,
    alpha: float,
    r_target: float,
    sigma: float,
    gamma: float,
    delta: float,
    pattern: sp.csr_matrix | None = None,
    rho_unit: float | None = None,
) -> float:
    """Protocol (iii): c such that c w^alpha grows at r_target. rho(B(r)) = 1 is protocol (ii)
    with gamma + r in place of gamma and the target (sigma + r) / sigma."""
    target = (sigma + r_target) / sigma
    return match_r0(graph, alpha, target, gamma + r_target, delta, pattern, rho_unit)


def match_protocol(
    protocol: str,
    graph: nx.Graph,
    alpha: float,
    tau: float,
    rho_unit: float,
    *,
    sigma: float,
    gamma: float,
    delta: float,
    pattern: sp.csr_matrix | None = None,
) -> float:
    """c that matches M2 to M1 (rate tau on every edge, unweighted radius rho_unit) in mean rate
    ("i"), in R_0 = T(tau) rho_unit ("ii") or in early growth rate ("iii")."""
    if protocol == "i":
        return match_mean_rate(graph, alpha, tau)
    if protocol == "ii":
        r0 = float(transmissibility(np.array(tau), gamma, delta)) * rho_unit
        return match_r0(graph, alpha, r0, gamma, delta, pattern, rho_unit)
    if protocol == "iii":
        r = growth_rate_uniform(tau, rho_unit, sigma, gamma, delta)
        return match_growth_rate(graph, alpha, r, sigma, gamma, delta, pattern, rho_unit)
    raise ValueError(f"unknown protocol {protocol!r}; expected i, ii or iii")


def attack_rate_message_passing(
    graph: nx.Graph, transmissibilities: np.ndarray, pattern: sp.csr_matrix | None = None
) -> float:
    """Final size without waning: u_{i<-j} = 1 - T_ij + T_ij prod_{l in N(j) - i} u_{j<-l} and
    attack rate = 1 - <prod_{j in N(i)} u_{i<-j}>, iterated up from u = 1 - T."""
    pattern = nonbacktracking_pattern(graph) if pattern is None else pattern
    t = np.tile(np.asarray(transmissibilities, dtype=float), 2)
    continues = sp.csr_matrix(
        (np.ones(pattern.nnz), pattern.indices, pattern.indptr), shape=pattern.shape
    )
    u = 1 - t
    for _ in range(100_000):
        previous, u = u, 1 - t + t * np.exp(continues @ np.log(u))
        if np.abs(u - previous).max() < 1e-12:
            break
    tail = edge_array(graph).T.ravel()
    escapes = np.exp(np.bincount(tail, weights=np.log(u), minlength=len(graph)))
    return float(1 - escapes.mean())
