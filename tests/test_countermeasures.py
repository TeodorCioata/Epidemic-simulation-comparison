import numpy as np
import pytest

from countermeasures import (
    STRATEGIES,
    countermeasure_figure,
    edge_measure,
    immunised,
    parse,
    ranking,
    replicate,
    summarise,
)
from episim.calibration import set_rates
from episim.networks import create_network, degrees, edge_values, iid_weights


def rng(*seed: int) -> np.random.Generator:
    return np.random.default_rng(list(seed))


@pytest.fixture(scope="module")
def models():
    graph = create_network("B", 500, 6, rng(0))
    m2 = iid_weights(graph, 2.0, rng(1))
    set_rates(graph, 1.0)
    set_rates(m2, 0.8, 1.0)
    return graph, m2


@pytest.fixture(scope="module")
def rows():
    return replicate(("B", 0, parse(["--n", "600", "--coverage", "0", "0.1"])))


def find(rows, model, strategy, coverage):
    [row] = [
        r for r in rows if (r["model"], r["strategy"], r["coverage"]) == (model, strategy, coverage)
    ]
    return row


def outcome(row):
    """As text, without the strategy: a growth rate that is undefined is nan, and nan != nan."""
    return repr({name: value for name, value in row.items() if name != "strategy"})


def test_weight_free_strategies_immunise_the_same_nodes_in_both_models(models):
    graph, m2 = models
    for strategy in ("random", "degree", "acquaintance"):
        assert immunised(graph, strategy, 50, rng(2)) == immunised(m2, strategy, 50, rng(2))
    assert immunised(m2, "strength", 50, rng(2)) != immunised(m2, "degree", 50, rng(2))


def test_in_m1_strength_targeting_is_degree_targeting(models, rows):
    graph, _ = models
    assert immunised(graph, "strength", 50, rng(2)) == immunised(graph, "degree", 50, rng(2))
    assert outcome(find(rows, "M1", "strength", 0.1)) == outcome(find(rows, "M1", "degree", 0.1))


def test_strategies_pick_distinct_nodes_and_targeting_finds_contacts(models):
    graph, _ = models
    k = degrees(graph)
    chosen = {strategy: immunised(graph, strategy, 50, rng(3)) for strategy in STRATEGIES}
    assert all(len(set(nodes)) == 50 for nodes in chosen.values())
    assert k[chosen["degree"]].min() >= np.sort(k)[-50]
    assert k[chosen["acquaintance"]].mean() > 1.5 * k[chosen["random"]].mean()


def test_coverage_zero_reproduces_no_intervention(rows):
    assert len(rows) == 2 + 2 * len(STRATEGIES) * 2 + 2
    for model in ("M1", "M2"):
        none = outcome(find(rows, model, "none", 0.0))
        assert all(outcome(find(rows, model, s, 0.0)) == none for s in STRATEGIES)
        assert (
            find(rows, model, "degree", 0.1)["attack_rate"]
            < find(rows, model, "none", 0.0)["attack_rate"]
        )


def test_edge_measures_halve_the_strongest_rates_or_remove_random_contacts(models):
    _, m2 = models
    rates, weights = edge_values(m2, "rate"), edge_values(m2, "weight")
    strongest = weights >= np.sort(weights)[-round(0.1 * len(weights))]
    halved = edge_values(edge_measure(m2, "halve strongest", 0.1, rng(4)), "rate")
    assert np.array_equal(halved, np.where(strongest, rates / 2, rates))
    removed = edge_values(edge_measure(m2, "remove random", 0.1, rng(4)), "rate")
    assert (removed == 0).sum() == round(0.1 * len(rates))
    assert np.array_equal(removed[removed > 0], rates[removed > 0])


def test_summary_is_the_expected_burden_and_ties_share_a_rank():
    """Two replicates; in the second, degree (and strength) immunisation stops the outbreak."""
    rows = []
    for index, (none, targeted, random) in enumerate([(0.9, 0.3, 0.6), (0.9, 0.0, 0.6)]):
        for strategy, attack_rate in (
            ("none", none),
            ("degree", targeted),
            ("strength", targeted),
            ("random", random),
        ):
            coverage = 0.0 if strategy == "none" else 0.1
            rows.append(
                {
                    "network": "E",
                    "replicate": index,
                    "model": "M1",
                    "strategy": strategy,
                    "coverage": coverage,
                    "major": attack_rate >= 0.05,
                    "attack_rate": attack_rate,
                    "peak_prevalence": attack_rate / 4,
                }
            )
    summary = {entry["strategy"]: entry for entry in summarise(rows, seed=1)}
    assert summary["degree"]["attack_rate_reduction"] == pytest.approx(1 - 0.15 / 0.9)
    assert summary["degree"]["peak_prevalence_reduction"] == pytest.approx(1 - 0.15 / 0.9)
    assert summary["degree"]["major"] == 0.5
    assert summary["random"]["attack_rate_reduction"] == pytest.approx(1 / 3)
    assert summary["none"]["attack_rate_reduction"] == 0.0
    ranks = [summary[strategy]["rank"] for strategy in ("degree", "strength", "random", "none")]
    assert ranks == [1, 1, 3, "-"]
    entry = summary["degree"]
    assert entry["attack_rate_low"] <= entry["attack_rate_reduction"] <= entry["attack_rate_high"]
    entries = [summary[strategy] for strategy in ("random", "strength", "degree")]
    assert ranking(entries) in ("degree = strength > random", "strength = degree > random")


@pytest.mark.slow
def test_figure_of_a_small_run(tmp_path, rows):
    countermeasure_figure(summarise(rows, seed=113), tmp_path / "countermeasures.png")
    assert (tmp_path / "countermeasures.png").stat().st_size > 10_000
    assert (tmp_path / "countermeasures.pdf").exists()
