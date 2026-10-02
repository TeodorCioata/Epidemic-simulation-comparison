import random

import numpy as np
import pytest

from episim import main
from episim.networks import (
    NETWORK_NAMES,
    bbv_network,
    create_network,
    degree_weights,
    degrees,
    edge_values,
    iid_weights,
    strength_exponent,
    strengths,
    weight_cv,
    weight_degree_correlation,
)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def weight_list(graph):
    return [w for *_, w in graph.edges(data="weight")]


@pytest.mark.parametrize("kind", ["E", "B", "C"])
def test_main_kinds_are_main_create_network_and_leave_global_rngs_alone(kind):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    graph = create_network(kind, 200, 7, rng(1))
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert random.getstate() == python_state

    seed = int(rng(1).integers(2**32))
    np.random.seed(seed)
    random.seed(seed)
    assert list(graph.edges) == list(main.create_network(kind, 200, 7, seed).edges)


@pytest.mark.parametrize("kind", list(NETWORK_NAMES))
def test_generator_seed_fixes_the_graph(kind):
    first, again, other = (create_network(kind, 300, 6, rng(s)) for s in (3, 3, 4))
    assert list(first) == list(range(300))
    assert list(first.edges) == list(again.edges) != list(other.edges)
    assert weight_list(first) == weight_list(again)


def test_heavy_tailed_cm_keeps_ba_degree_distribution():
    def mean_square_degree(kind):
        return (degrees(create_network(kind, 5000, 6, rng(5))) ** 2).mean()

    assert mean_square_degree("H") == pytest.approx(mean_square_degree("B"), rel=0.05)
    assert mean_square_degree("H") > 1.5 * mean_square_degree("C")


def test_weight_schemes_return_weighted_copies():
    graph = create_network("E", 300, 6, rng(0))
    for weighted in (iid_weights(graph, 1.0, rng(1)), degree_weights(graph, 0.5)):
        assert weighted is not graph
        assert list(weighted.edges) == list(graph.edges)
        assert None not in weight_list(weighted)
    assert set(weight_list(graph)) == {None}


@pytest.mark.parametrize(("cv", "tolerance"), [(0.5, 0.05), (1.0, 0.05), (2.0, 0.10)])
def test_iid_weights_have_unit_mean_and_target_cv(cv, tolerance):
    weighted = iid_weights(create_network("C", 4000, 7, rng(0)), cv, rng(1))
    assert edge_values(weighted, "weight").mean() == pytest.approx(1.0, abs=1e-12)
    assert weight_cv(weighted) == pytest.approx(cv, rel=tolerance)


def test_degenerate_weight_limits_are_exactly_one():
    graph = create_network("E", 300, 6, rng(0))
    for weighted in (
        iid_weights(graph, 0.0, rng(1)),
        degree_weights(graph, 0.0),
        create_network("W", 300, 6, rng(2), reinforcement=0.0),
    ):
        assert weight_list(weighted) == [1.0] * weighted.number_of_edges()


def test_weight_degree_correlation_separates_the_schemes():
    graph = create_network("C", 3000, 7, rng(0))
    assert weight_degree_correlation(degree_weights(graph, 0.5)) == pytest.approx(1.0)
    assert abs(weight_degree_correlation(iid_weights(graph, 1.0, rng(1)))) < 0.05
    assert weight_degree_correlation(graph) == 0.0


@pytest.mark.parametrize("reinforcement", [0.5, 2.0])
def test_bbv_follows_its_strength_law(reinforcement):
    """s_i = (2 delta + 1) k_i - 2 delta m [1]; <w> = 1 rescales s, so compare intercept/slope."""
    m = 2
    graph = bbv_network(3000, m, reinforcement, rng(0))
    k, s = degrees(graph).astype(float), strengths(graph)
    slope, intercept = np.polyfit(k, s, 1)
    assert 1 - (s - slope * k - intercept).var() / s.var() > 0.99
    expected = -2 * reinforcement * m / (2 * reinforcement + 1)
    assert intercept / slope == pytest.approx(expected, rel=0.08)


@pytest.mark.slow
@pytest.mark.parametrize("cv", [1.0, 3.0])
def test_strength_exponent_is_one_for_iid_weights_at_high_dispersion(cv):
    """A node-level fit of log s_i on log k_i gives 1.23 (CV 1) and 3.81 (CV 3) on this graph."""
    graph = create_network("C", 20_000, 4, rng(0))
    assert strength_exponent(iid_weights(graph, cv, rng(1))) == pytest.approx(1.0, abs=0.1)


def test_strength_exponent_recovers_a_planted_exponent():
    graph = create_network("C", 5000, 4, rng(0))
    assert strength_exponent(graph) == pytest.approx(1.0)
    assert strength_exponent(degree_weights(graph, 0.5)) == pytest.approx(1.5, rel=0.05)
