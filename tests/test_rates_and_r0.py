import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from episim.calibration import (
    match_mean_rate,
    match_r0,
    nonbacktracking_pattern,
    r0_newman,
    r0_nonbacktracking,
    set_rates,
    tau_max,
    transmissibility,
)
from episim.networks import create_network, edge_array, edge_values, iid_weights

GAMMA, DELTA = 1.0, 0.1


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture(scope="module")
def graph():
    return create_network("E", 1000, 7, rng(0))


def rate_t(graph):
    return transmissibility(edge_values(graph, "rate"), GAMMA, DELTA)


def test_m2_rates_reduce_to_m1_exactly(graph):
    m1 = graph.copy()
    set_rates(m1, 0.3)
    for cv, alpha in ((0.0, 1.7), (2.0, 0.0)):
        m2 = iid_weights(graph, cv, rng(1))
        set_rates(m2, 0.3, alpha)
        assert edge_values(m2, "rate").tolist() == edge_values(m1, "rate").tolist()
    assert set(edge_values(m1, "rate")) == {0.3}


def test_rates_follow_the_plan_parameterisation(graph):
    """c w^alpha = tau_M (w / w_M)^alpha."""
    weighted = iid_weights(graph, 1.5, rng(1))
    set_rates(weighted, 0.37, 1.3)
    w = edge_values(weighted, "weight")
    expected = tau_max(weighted, 0.37, 1.3) * (w / w.max()) ** 1.3
    assert edge_values(weighted, "rate") == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("cv", [0.0, 1.0, 2.0])
def test_protocol_i_matches_the_mean_rate(graph, cv, alpha):
    weighted = iid_weights(graph, cv, rng(1))
    set_rates(weighted, match_mean_rate(weighted, alpha, 0.3), alpha)
    assert edge_values(weighted, "rate").mean() == pytest.approx(0.3, rel=1e-12)


def test_protocol_i_lowers_r0_as_dispersion_grows(graph):
    """T is concave in tau, so dispersion at fixed <tau> lowers <T> and R_0 (Jensen)."""
    r0 = []
    for cv in (0.0, 0.5, 1.0, 2.0):
        weighted = iid_weights(graph, cv, rng(1))
        set_rates(weighted, match_mean_rate(weighted, 1.0, 0.3), 1.0)
        r0.append(r0_nonbacktracking(weighted, rate_t(weighted)))
    assert r0 == sorted(r0, reverse=True)
    assert len(set(r0)) == len(r0)


def test_pattern_matches_the_definition():
    graph = create_network("E", 30, 4, rng(2))
    t = rng(3).random(graph.number_of_edges())
    directed = [(i, j, e) for e, (i, j) in enumerate(edge_array(graph))]
    directed += [(j, i, e) for i, j, e in directed]
    expected = np.zeros((len(directed), len(directed)))
    for row, (i, j, _) in enumerate(directed):
        for col, (tail, head, e) in enumerate(directed):
            if tail == j and head != i:
                expected[row, col] = t[e]
    pattern = nonbacktracking_pattern(graph)
    built = sp.csr_matrix((t[pattern.data], pattern.indices, pattern.indptr), shape=pattern.shape)
    assert np.array_equal(built.toarray(), expected)


@pytest.mark.parametrize("k", [4, 6, 8])
def test_r0_on_a_regular_graph_is_t_times_k_minus_one(k):
    ring = nx.circulant_graph(300, range(1, k // 2 + 1))
    t = np.full(ring.number_of_edges(), 0.3)
    assert r0_nonbacktracking(ring, t) == pytest.approx(0.3 * (k - 1))
    assert r0_newman(ring, t) == pytest.approx(0.3 * (k - 1))


def test_forest_has_r0_exactly_zero():
    tree = nx.balanced_tree(3, 4)
    assert r0_nonbacktracking(tree, np.full(tree.number_of_edges(), 0.4)) == 0.0


def test_newman_on_a_star():
    """Degrees 4, 1, 1, 1, 1: <k> = 1.6, <k^2> = 4."""
    assert r0_newman(nx.star_graph(4), np.full(4, 0.5)) == pytest.approx(0.5 * (4 - 1.6) / 1.6)


@pytest.mark.slow
def test_nonbacktracking_agrees_with_newman_on_a_cm_with_iid_weights():
    weighted = iid_weights(create_network("C", 10_000, 4, rng(0)), 1.0, rng(1))
    set_rates(weighted, 0.3, 1.0)
    t = rate_t(weighted)
    assert r0_nonbacktracking(weighted, t) == pytest.approx(r0_newman(weighted, t), rel=0.03)


def test_protocol_ii_hits_the_target(graph):
    weighted = iid_weights(graph, 1.5, rng(1))
    pattern = nonbacktracking_pattern(weighted)
    set_rates(weighted, match_r0(weighted, 1.2, 2.0, GAMMA, DELTA, pattern), 1.2)
    assert r0_nonbacktracking(weighted, rate_t(weighted), pattern) == pytest.approx(2.0, rel=1e-9)


def test_protocol_ii_rejects_an_unreachable_target(graph):
    with pytest.raises(ValueError, match="unreachable"):
        match_r0(graph, 1.0, 1e6, GAMMA, DELTA)
