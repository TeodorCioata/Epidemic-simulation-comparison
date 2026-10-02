import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from test_simulation import exact_mean_exposures

from episim import main
from episim.calibration import set_rates
from episim.simulation import simulate


@pytest.fixture
def main_histories(monkeypatch):
    """main.simulate_epidemic without its window: one (network, history) per network type and call,
    read from the animate callback it hands to FuncAnimation."""
    recorded = []

    def record(figure, animate, **_):
        closure = dict(zip(animate.__code__.co_freevars, animate.__closure__, strict=True))
        recorded.append((closure["network"].cell_contents, closure["history"].cell_contents))

    monkeypatch.setattr(main, "animation", SimpleNamespace(FuncAnimation=record))
    monkeypatch.setattr(
        main, "plt", SimpleNamespace(subplots=lambda **_: (None, None), show=lambda: None)
    )
    for drawing in ("spring_layout", "draw_networkx_edges", "draw_networkx_nodes"):
        monkeypatch.setattr(nx, drawing, lambda *args, **kwargs: None)

    def run(calls: int) -> list:
        np.random.seed(0)
        random.seed(0)
        for _ in range(calls):
            main.simulate_epidemic()
        return recorded

    return run


def count(events, state: str) -> int:
    return sum(new_state == state for *_, new_state in events)


def z_score(first, second) -> float:
    return (np.mean(first) - np.mean(second)) / np.sqrt(
        np.var(first) / len(first) + np.var(second) / len(second)
    )


@pytest.mark.slow
def test_deaths_and_exposures_match_main_without_waning(monkeypatch, main_histories):
    """Waning is switched off (beta ~ 0) because main.py double-counts a pending contact when a
    node becomes susceptible again (see the triangle test below)."""
    monkeypatch.setattr(main, "beta", 1e-12)
    histories = main_histories(300)
    for index, kind in enumerate("EBC"):
        ours = []
        for run, (graph, _) in enumerate(histories[index::3]):
            set_rates(graph, main.tau)
            rng = np.random.default_rng([7, index, run])
            ours.append(
                simulate(
                    graph,
                    rng.choice(len(graph), 2, replace=False),
                    rng,
                    sigma=main.sigma,
                    gamma=main.gamma,
                    delta=main.delta,
                    beta=0.0,
                )[1]
            )
        for state in "DE":
            in_main = [count(history, state) for _, history in histories[index::3]]
            in_ours = [count(events, state) for events in ours]
            assert abs(z_score(in_main, in_ours)) < 4, (
                f"{kind} {state}: main {np.mean(in_main):.2f} vs ours {np.mean(in_ours):.2f}"
            )


@pytest.mark.slow
@pytest.mark.xfail(reason="main.py redraws a contact that is still pending", strict=False)
def test_main_with_waning_matches_the_exact_chain_on_a_triangle(monkeypatch, main_histories):
    """Exact 4.77 exposures; main.py gives 5.21. Our simulator passes the same check."""
    parameters = {"tau": 1.0, "sigma": 50.0, "gamma": 1.0, "beta": 50.0}
    for name, value in (parameters | {"delta": 1e-12}).items():
        monkeypatch.setattr(main, name, value)
    monkeypatch.setattr(main, "create_network", lambda *_: nx.complete_graph(3))
    exposures = [count(history, "E") for _, history in main_histories(2000)]
    exact = exact_mean_exposures(nx.complete_graph(3), **parameters, start=("I", "I", "S"))
    standard_error = np.std(exposures) / np.sqrt(len(exposures))
    assert np.mean(exposures) == pytest.approx(exact, abs=4 * standard_error)


def test_importing_the_layer_works_headless_and_leaves_global_state_alone():
    """main.py calls matplotlib.use("TkAgg") and seeds both global RNGs at import."""
    code = (
        "import random, matplotlib, numpy as np\n"
        "use, python, numpy = matplotlib.use, random.getstate(), np.random.get_state()[1]\n"
        "import episim.plotting\n"
        "assert matplotlib.use is use\n"
        "assert random.getstate() == python\n"
        "assert (np.random.get_state()[1] == numpy).all()\n"
    )
    headless = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    repository = Path(__file__).parent.parent
    subprocess.run([sys.executable, "-c", code], check=True, env=headless, cwd=repository)
