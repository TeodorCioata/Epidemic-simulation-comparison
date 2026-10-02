"""Small seeded versions of the checks that validate.py runs on the pilot."""

import numpy as np
import pytest

from episim.calibration import (
    attack_rate_message_passing,
    growth_rate,
    match_protocol,
    nonbacktracking_pattern,
    r0_newman,
    r0_nonbacktracking,
    set_rates,
    transmissibility,
    unit_radius,
)
from episim.metrics import run_metrics
from episim.networks import create_network, edge_values, iid_weights
from episim.simulation import simulate

pytestmark = pytest.mark.slow

DISEASE = {"sigma": 1.0, "gamma": 1.0, "delta": 0.1}
TAU = 1.0


def rng(*seed: int) -> np.random.Generator:
    return np.random.default_rng(list(seed))


@pytest.fixture(scope="module")
def models():
    """M1, and M2 with CV_w = 1.5 matched in growth rate, on one configuration-model graph."""
    graph = create_network("C", 5000, 7, rng(0))
    pattern = nonbacktracking_pattern(graph)
    rho_unit = unit_radius(graph, pattern)
    set_rates(graph, TAU)
    weighted = iid_weights(graph, 1.5, rng(1))
    c = match_protocol("iii", weighted, 1.0, TAU, rho_unit, **DISEASE, pattern=pattern)
    set_rates(weighted, c, 1.0)
    return {"M1": graph, "M2": weighted}, pattern, rho_unit


@pytest.fixture(scope="module")
def major_outbreaks(models):
    networks, _, _ = models
    outbreaks = {}
    for model, network in networks.items():
        runs = []
        for run in range(24):
            generator = rng(2, run)
            initial_infected = generator.choice(len(network), 2, replace=False)
            runs.append(
                run_metrics(*simulate(network, initial_infected, generator, **DISEASE, beta=0.0))
            )
        outbreaks[model] = [metrics for metrics in runs if metrics["major"]]
    return outbreaks


def test_simulated_growth_rate_matches_the_analytic_one(models, major_outbreaks):
    networks, pattern, rho_unit = models
    analytic = {
        model: growth_rate(network, **DISEASE, pattern=pattern, rho_unit=rho_unit)
        for model, network in networks.items()
    }
    assert analytic["M2"] == pytest.approx(analytic["M1"], abs=1e-9)
    for model, outbreaks in major_outbreaks.items():
        assert len(outbreaks) >= 16
        simulated = np.median([metrics["growth_rate"] for metrics in outbreaks])
        assert simulated == pytest.approx(analytic[model], rel=0.1)


def test_simulated_attack_rate_matches_message_passing(models, major_outbreaks):
    """Equal growth rate is not equal final size: M2's is lower, and theory predicts both."""
    networks, pattern, _ = models
    predicted = {}
    for model, network in networks.items():
        t = transmissibility(edge_values(network, "rate"), DISEASE["gamma"], DISEASE["delta"])
        predicted[model] = attack_rate_message_passing(network, t, pattern)
        simulated = np.median([metrics["attack_rate"] for metrics in major_outbreaks[model]])
        assert simulated == pytest.approx(predicted[model], abs=0.01)
    assert predicted["M2"] < predicted["M1"] - 0.02


def test_newman_fails_on_ba_because_of_its_wiring_not_its_degrees():
    """H is a configuration model on B's degree sequence: same degrees, no growth correlations."""
    gap = {}
    for kind in "BH":
        graph = create_network(kind, 10_000, 4, rng(0))
        t = np.full(graph.number_of_edges(), TAU / (TAU + DISEASE["gamma"] + DISEASE["delta"]))
        gap[kind] = r0_nonbacktracking(graph, t) / r0_newman(graph, t) - 1
    assert gap["B"] < -0.2
    assert abs(gap["H"]) < abs(gap["B"]) / 2
