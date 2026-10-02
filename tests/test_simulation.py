import itertools
import math

import networkx as nx
import numpy as np
import pytest

from episim.calibration import set_rates
from episim.networks import create_network, iid_weights
from episim.simulation import PREVIOUS_STATE, simulate, states_at

MAIN_PARAMETERS = {"sigma": 1.0, "gamma": 1.0, "delta": 0.1, "beta": 1.0}


def rng(*seed: int) -> np.random.Generator:
    return np.random.default_rng(list(seed))


def with_rate(graph: nx.Graph, rate: float) -> nx.Graph:
    nx.set_edge_attributes(graph, rate, "rate")
    return graph


def test_every_transition_is_legal_and_nodes_are_conserved():
    graph = create_network("E", 300, 7, rng(0))
    set_rates(graph, 1.0)
    initial, events = simulate(graph, [0, 1], rng(1), **MAIN_PARAMETERS)
    assert any(new_state == "S" for *_, new_state in events), "the run should include waning"
    times = [time for time, *_ in events]
    assert times == sorted(times)

    states = list(initial)
    for _, node, new_state in events:
        assert states[node] == PREVIOUS_STATE[new_state]
        states[node] = new_state
    for snapshot in states_at(initial, events, np.linspace(0, times[-1], 20)):
        assert len(snapshot) == 300
        assert set(snapshot) <= set("SEIRD")
    assert states_at(initial, events, [math.inf]) == [states]


def test_death_is_logged_at_its_own_time():
    pair = with_rate(nx.Graph([(0, 1)]), 0.0)
    initial, events = simulate(pair, [0], rng(2), sigma=1.0, gamma=0.0, delta=1.0, beta=0.0)
    [(time, node, new_state)] = events
    assert (node, new_state) == (0, "D")
    assert time > 0
    before, at = states_at(initial, events, [time / 2, time])
    assert (before[0], at[0]) == ("I", "D")


def test_two_node_infection_probability_is_t():
    """T = tau / (tau + gamma + delta), within a 99.9 % binomial interval."""
    tau, gamma, delta, runs = 0.5, 1.0, 0.1, 4000
    pair = with_rate(nx.Graph([(0, 1)]), tau)
    infected = sum(
        any(
            node == 1
            for _, node, _ in simulate(
                pair, [0], rng(3, run), sigma=1.0, gamma=gamma, delta=delta, beta=0.0
            )[1]
        )
        for run in range(runs)
    )
    t = tau / (tau + gamma + delta)
    assert abs(infected / runs - t) < 3.29 * math.sqrt(t * (1 - t) / runs)


def test_reinfection_while_the_neighbour_is_still_infectious():
    """Renewal over node 0's Exp(gamma) infectious period: E[N] = L_1 / (1 - L_c), where
    L_x = x / (x + gamma), L_1 = L_tau and L_c = L_tau L_sigma L_gamma L_beta."""
    tau = sigma = beta = 5.0
    gamma = 1.0
    pair = with_rate(nx.Graph([(0, 1)]), tau)
    exposures = []
    for run in range(4000):
        _, events = simulate(
            pair, [0], rng(4, run), sigma=sigma, gamma=gamma, delta=0.0, beta=beta, t_max=50.0
        )
        exit_of_0 = next(time for time, node, _ in events if node == 0)
        exposures.append(sum(n == 1 and s == "E" and t < exit_of_0 for t, n, s in events))

    def laplace(rate):
        return rate / (rate + gamma)

    first = laplace(tau)
    cycle = laplace(tau) * laplace(sigma) * laplace(gamma) * laplace(beta)
    mean, standard_error = np.mean(exposures), np.std(exposures) / math.sqrt(len(exposures))
    assert mean == pytest.approx(first / (1 - cycle), abs=4 * standard_error)
    assert mean > first + 10 * standard_error, "main.py's re-targeting quirk gives exactly L_1"


def exact_mean_exposures(graph, tau, sigma, gamma, beta, start):
    """Expected S->E count before extinction, from the SEIRS Markov chain on a tiny graph."""
    progress = {"E": ("I", sigma), "I": ("R", gamma), "R": ("S", beta)}
    states = [s for s in itertools.product("SEIR", repeat=len(graph)) if {"E", "I"} & set(s)]
    index = {s: i for i, s in enumerate(states)}
    outflow, gain = np.zeros((len(states), len(states))), np.zeros(len(states))
    for s, i in index.items():
        for node, own in enumerate(s):
            if own == "S":
                new, rate, exposure = "E", tau * sum(s[v] == "I" for v in graph[node]), 1
            else:
                (new, rate), exposure = progress[own], 0
            outflow[i, i] += rate
            gain[i] += rate * exposure
            target = (*s[:node], new, *s[node + 1 :])
            if target in index:
                outflow[i, index[target]] -= rate
    return np.linalg.solve(outflow, gain)[index[start]]


@pytest.mark.slow
def test_triangle_with_waning_matches_the_exact_markov_chain():
    """Several infectious periods per node: stale contacts from an earlier one would bias this."""
    triangle = with_rate(nx.complete_graph(3), 1.0)
    parameters = {"sigma": 50.0, "gamma": 1.0, "beta": 50.0}
    exposures = [
        sum(s == "E" for *_, s in simulate(triangle, [0], rng(5, run), delta=0.0, **parameters)[1])
        for run in range(10_000)
    ]
    exact = exact_mean_exposures(triangle, 1.0, **parameters, start=("I", "S", "S"))
    standard_error = np.std(exposures) / math.sqrt(len(exposures))
    assert np.mean(exposures) == pytest.approx(exact, abs=4 * standard_error)


@pytest.mark.parametrize("kind", ["E", "B", "H"])
@pytest.mark.parametrize(("cv", "alpha"), [(0.0, 1.7), (2.0, 0.0)])
def test_degenerate_m2_reproduces_m1_event_for_event(kind, cv, alpha):
    """Also on B and H, where a copy of the graph lists a node's neighbours in another order."""
    graph = create_network(kind, 300, 7, rng(0))
    m2 = iid_weights(graph, cv, rng(1))
    set_rates(graph, 1.0)
    set_rates(m2, 1.0, alpha)
    m1_run = simulate(graph, [0, 1], rng(0, 3), **MAIN_PARAMETERS)
    assert simulate(m2, [0, 1], rng(0, 3), **MAIN_PARAMETERS) == m1_run
    assert len(m1_run[1]) > 1000


def test_zero_rate_edge_never_transmits():
    path = nx.path_graph(3)
    nx.set_edge_attributes(path, {(0, 1): 50.0, (1, 2): 0.0}, "rate")
    for run in range(50):
        _, events = simulate(path, [0], rng(6, run), **MAIN_PARAMETERS)
        assert all(node != 2 for _, node, _ in events)
