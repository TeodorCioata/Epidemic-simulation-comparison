import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from episim.calibration import (
    attack_rate_message_passing,
    discounted_transmissibility,
    growth_rate,
    growth_rate_uniform,
    match_growth_rate,
    match_mean_rate,
    match_protocol,
    match_r0,
    nonbacktracking_pattern,
    r0_newman,
    r0_nonbacktracking,
    set_rates,
    tau_max,
    transmissibility,
    unit_radius,
)
from episim.networks import create_network, edge_array, edge_values, iid_weights

SIGMA, GAMMA, DELTA, TAU = 1.0, 1.0, 0.1, 1.0


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


@pytest.mark.parametrize("k", [4, 6, 8])
def test_growth_rate_on_a_regular_graph_is_the_root_of_a_quadratic(k):
    """(sigma + r)(tau + gamma + delta + r) = sigma tau (k - 1)."""
    tau = 0.8
    ring = nx.circulant_graph(300, range(1, k // 2 + 1))
    set_rates(ring, tau)
    exit_rate = tau + GAMMA + DELTA
    expected = np.roots([1, SIGMA + exit_rate, SIGMA * (exit_rate - tau * (k - 1))]).max()
    assert growth_rate_uniform(tau, k - 1, SIGMA, GAMMA, DELTA) == pytest.approx(expected)
    assert growth_rate(ring, SIGMA, GAMMA, DELTA) == pytest.approx(expected)


def test_growth_rate_changes_sign_at_r0_equal_to_one():
    """T_0 = T, so rho(B(0)) is R_0: on a 5-regular graph R_0 = 1 at tau = (gamma + delta) / 3."""
    critical = (GAMMA + DELTA) / 3
    rates = np.array([0.1, 1.0, 5.0])
    assert discounted_transmissibility(rates, SIGMA, GAMMA, DELTA, 0.0).tolist() == (
        transmissibility(rates, GAMMA, DELTA).tolist()
    )
    assert growth_rate_uniform(critical, 4, SIGMA, GAMMA, DELTA) == pytest.approx(0.0, abs=1e-12)
    assert growth_rate_uniform(critical / 2, 4, SIGMA, GAMMA, DELTA) < 0
    assert growth_rate_uniform(critical * 2, 4, SIGMA, GAMMA, DELTA) > 0


def test_protocol_iii_gives_m2_the_growth_rate_of_m1(graph):
    weighted = iid_weights(graph, 1.5, rng(1))
    pattern = nonbacktracking_pattern(weighted)
    rho_unit = unit_radius(graph, pattern)
    r = growth_rate_uniform(TAU, rho_unit, SIGMA, GAMMA, DELTA)
    set_rates(weighted, match_growth_rate(weighted, 1.2, r, SIGMA, GAMMA, DELTA, pattern), 1.2)

    discounted = discounted_transmissibility(edge_values(weighted, "rate"), SIGMA, GAMMA, DELTA, r)
    assert r0_nonbacktracking(weighted, discounted, pattern) == pytest.approx(1.0, abs=1e-10)
    assert growth_rate(weighted, SIGMA, GAMMA, DELTA, pattern) == pytest.approx(r, abs=1e-9)
    m1_r0 = TAU / (TAU + GAMMA + DELTA) * rho_unit
    assert r0_nonbacktracking(weighted, rate_t(weighted), pattern) < 0.95 * m1_r0


@pytest.mark.parametrize(("cv", "alpha"), [(0.0, 1.7), (2.0, 0.0)])
@pytest.mark.parametrize("protocol", ["i", "ii", "iii"])
def test_every_protocol_recovers_tau_in_the_degenerate_limits(graph, protocol, cv, alpha):
    weighted = iid_weights(graph, cv, rng(1))
    pattern = nonbacktracking_pattern(weighted)
    rho_unit = unit_radius(graph, pattern)
    c = match_protocol(
        protocol, weighted, alpha, TAU, rho_unit, sigma=SIGMA, gamma=GAMMA, delta=DELTA
    )
    assert c == pytest.approx(TAU, rel=2e-12)


def test_unknown_protocol_is_rejected(graph):
    with pytest.raises(ValueError, match="unknown protocol"):
        match_protocol("iv", graph, 1.0, TAU, 7.0, sigma=SIGMA, gamma=GAMMA, delta=DELTA)


def test_message_passing_on_a_regular_graph_is_the_scalar_fixed_point():
    """u = 1 - T + T u^(k-1) and attack rate = 1 - u^k, which is 0 below T (k - 1) = 1."""
    k, t = 4, 0.6
    regular = nx.random_regular_graph(k, 500, seed=0)
    u = 0.0
    for _ in range(1000):
        u = 1 - t + t * u ** (k - 1)
    supercritical = np.full(regular.number_of_edges(), t)
    predicted = attack_rate_message_passing(regular, supercritical)
    assert predicted == pytest.approx(1 - u**k, abs=1e-9)
    assert 0.5 < predicted < 1
    assert attack_rate_message_passing(regular, supercritical / 2) == pytest.approx(0, abs=1e-9)
