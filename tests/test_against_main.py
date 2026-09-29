import random

import numpy as np
import pytest

import main
from episim.calibration import set_rates
from episim.simulation import simulate


@pytest.mark.slow
def test_final_deaths_match_main_without_waning(monkeypatch):
    """Deaths are immune to main.py's death-at-onset quirk. Waning is switched off (beta ~ 0)
    because main.py never re-targets a neighbour that becomes susceptible again."""
    final_graphs = []
    monkeypatch.setattr(main, "plotting", final_graphs.append)
    monkeypatch.setattr(main, "beta", 1e-12)
    np.random.seed(0)
    random.seed(0)
    runs = 300
    for _ in range(runs):
        main.simulate_epidemic()

    for index, kind in enumerate("EBC"):
        graphs = final_graphs[index::3]
        main_deaths = [sum(data["State"] == "D" for _, data in g.nodes(data=True)) for g in graphs]
        our_deaths = []
        for run, graph in enumerate(graphs):
            set_rates(graph, main.tau)
            rng = np.random.default_rng([7, index, run])
            _, events = simulate(
                graph,
                rng.choice(len(graph), 2, replace=False),
                rng,
                sigma=main.sigma,
                gamma=main.gamma,
                delta=main.delta,
                beta=0.0,
            )
            our_deaths.append(sum(new_state == "D" for *_, new_state in events))
        standard_error = np.sqrt((np.var(main_deaths) + np.var(our_deaths)) / runs)
        z = (np.mean(main_deaths) - np.mean(our_deaths)) / standard_error
        assert abs(z) < 4, (
            f"{kind}: main {np.mean(main_deaths):.2f} vs ours {np.mean(our_deaths):.2f}"
        )
