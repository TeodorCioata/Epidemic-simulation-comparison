import math

import numpy as np
import pytest

from episim.calibration import set_rates
from episim.metrics import (
    STATES,
    bootstrap_ci,
    counts_on_grid,
    fitted_growth_rate,
    run_metrics,
    state_counts,
)
from episim.networks import create_network
from episim.simulation import simulate, states_at

SEIRD = {"sigma": 1.0, "gamma": 1.0, "delta": 0.1, "beta": 0.0}


def rng(*seed: int) -> np.random.Generator:
    return np.random.default_rng(list(seed))


@pytest.fixture(scope="module")
def graph():
    graph = create_network("E", 2000, 7, rng(0))
    set_rates(graph, 1.0)
    return graph


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_counts_sum_to_n_and_equal_a_replay_of_the_log(graph, beta):
    initial, events = simulate(graph, [0, 1], rng(1), **(SEIRD | {"beta": beta}), t_max=20.0)
    times, counts = state_counts(initial, events)
    assert len(times) == len(events) + 1
    assert (counts.sum(axis=1) == len(graph)).all()
    assert (counts >= 0).all()

    grid = np.linspace(0, 20, 41)
    replayed = [[states.count(s) for s in STATES] for states in states_at(initial, events, grid)]
    assert counts_on_grid(initial, events, grid).tolist() == replayed


def test_metrics_of_a_hand_written_log():
    """Ten nodes, node 0 infectious at t = 0; I(t) peaks at 2 during [3, 4)."""
    initial = ["I"] + ["S"] * 9
    events = [
        (1.0, 1, "E"),
        (2.0, 2, "E"),
        (3.0, 1, "I"),
        (4.0, 0, "R"),
        (5.0, 2, "I"),
        (6.0, 1, "D"),
        (7.0, 2, "R"),
    ]
    metrics = run_metrics(initial, events)
    assert metrics == {
        "major": True,
        "extinct": True,
        "peak_prevalence": 0.2,
        "peak_time": 3.0,
        "attack_rate": 0.3,
        "exposures": 0.2,
        "deaths": 0.1,
        "duration": 7.0,
        "growth_rate": pytest.approx(math.nan, nan_ok=True),
    }
    assert run_metrics(["S"] * 10, [])["duration"] == 0.0


def test_fitted_growth_rate_recovers_a_planted_exponential():
    """Cumulative incidence k at time log(k) / r grows as exp(r t)."""
    exposure_times = np.log(np.arange(1, 501)) / 0.7
    assert fitted_growth_rate(exposure_times, 20, 200) == pytest.approx(0.7)
    assert math.isnan(fitted_growth_rate(exposure_times, 20, 501))
    assert math.isnan(fitted_growth_rate(exposure_times, 200, 200))


def test_a_major_outbreak_has_a_growth_rate_and_a_minor_one_does_not(graph):
    major = run_metrics(*simulate(graph, [0, 1], rng(1), **SEIRD))
    assert major["major"] and major["extinct"]
    assert major["attack_rate"] > 0.9
    assert 0.5 < major["growth_rate"] < 2.0
    assert 0 < major["peak_time"] < major["duration"]

    set_rates(isolated := graph.copy(), 0.0)
    minor = run_metrics(*simulate(isolated, [0, 1], rng(1), **SEIRD))
    assert not minor["major"]
    assert minor["attack_rate"] == 2 / len(graph)
    assert math.isnan(minor["growth_rate"])


def test_waning_with_a_time_cap_stays_well_defined(graph):
    """Reinfections make exposures exceed the attack rate; duration is censored at the cap."""
    run = simulate(graph, [0, 1], rng(1), **(SEIRD | {"beta": 1.0}), t_max=30.0)
    metrics = run_metrics(*run, t_max=30.0)
    assert not metrics["extinct"]
    assert metrics["duration"] == 30.0
    assert metrics["exposures"] > metrics["attack_rate"] > 0.9
    assert all(math.isfinite(value) for value in metrics.values())


def test_identical_seeds_give_identical_metrics(graph):
    first, again, other = (
        run_metrics(*simulate(graph, [0, 1], rng(seed), **SEIRD)) for seed in (5, 5, 6)
    )
    assert first == again != other


def test_bootstrap_interval_brackets_the_median_and_is_seeded():
    values = rng(0).normal(3.0, 1.0, 400)
    low, high = bootstrap_ci(values, rng(1))
    assert low < np.median(values) < high
    assert high - low < 0.5
    assert bootstrap_ci(values, rng(1)) == (low, high)
    assert all(math.isnan(bound) for bound in bootstrap_ci(np.empty(0), rng(1)))
