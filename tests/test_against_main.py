import gc
import os
import random
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from test_simulation import exact_mean_exposures

from episim import main
from episim.calibration import set_rates
from episim.plotting import COLOURS
from episim.simulation import simulate

REPOSITORY = Path(__file__).parent.parent


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
    monkeypatch.setattr(main, "beta", 0)
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


def test_main_runs_without_waning_or_deaths(monkeypatch, main_histories):
    monkeypatch.setattr(main, "beta", 0)
    monkeypatch.setattr(main, "delta", 0)
    histories = main_histories(1)
    assert len(histories) == 3
    assert all({state for *_, state in history} <= set("EIR") for _, history in histories)
    assert max(len(history) for _, history in histories) > 100


@pytest.mark.slow
def test_main_with_waning_matches_the_exact_chain_on_a_triangle(monkeypatch, main_histories):
    """Exact 4.77 exposures; main.py gave 5.21 before it stopped redrawing pending contacts."""
    parameters = {"tau": 1.0, "sigma": 50.0, "gamma": 1.0, "beta": 50.0}
    for name, value in (parameters | {"delta": 0}).items():
        monkeypatch.setattr(main, name, value)
    monkeypatch.setattr(main, "create_network", lambda *_: nx.complete_graph(3))
    exposures = [count(history, "E") for _, history in main_histories(2000)]
    exact = exact_mean_exposures(nx.complete_graph(3), **parameters, start=("I", "I", "S"))
    standard_error = np.std(exposures) / np.sqrt(len(exposures))
    assert np.mean(exposures) == pytest.approx(exact, abs=4 * standard_error)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:Animation was deleted without rendering")
def test_running_main_builds_its_animation_and_each_pass_starts_from_the_initial_states(
    monkeypatch,
):
    """python main.py with the Tk backend and the window patched out; the animation is real."""

    def shown(artist, network) -> bool:
        states = [network.nodes[node]["State"] for node in network]
        return np.allclose(artist.get_facecolors(), [to_rgba(COLOURS[s]) for s in states])

    built, real = [], matplotlib.animation.FuncAnimation

    def build(figure, animate, **kwargs):
        closure = dict(zip(animate.__code__.co_freevars, animate.__closure__, strict=True))
        network, artist = closure["network"].cell_contents, closure["nodes"].cell_contents
        animation = real(figure, animate, **kwargs)
        initial = dict(network.nodes(data="State"))
        for _ in range(2):  # the first pass and a repeat
            kwargs["init_func"]()
            assert dict(network.nodes(data="State")) == initial and shown(artist, network)
            for frame in range(kwargs["frames"]):
                animate(frame)
            assert shown(artist, network)
        built.append(kwargs["frames"])
        plt.close(figure)
        return animation

    monkeypatch.setattr(matplotlib, "use", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(matplotlib.animation, "FuncAnimation", build)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        runpy.run_path(str(REPOSITORY / "main.py"), run_name="__main__")
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        gc.collect()  # the animations are never shown: let them go inside the warning filter
    assert len(built) == 3 and min(built) > 100


def test_importing_main_or_the_layer_works_headless_and_leaves_global_state_alone():
    """main.py selects TkAgg and seeds the global RNGs only when it runs as a script."""
    code = (
        "import random, matplotlib, numpy as np\n"
        "backend, python = matplotlib.get_backend(), random.getstate()\n"
        "numpy = np.random.get_state()[1]\n"
        "import main, episim.plotting\n"
        "assert matplotlib.get_backend() == backend\n"
        "assert random.getstate() == python\n"
        "assert (np.random.get_state()[1] == numpy).all()\n"
    )
    headless = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    subprocess.run([sys.executable, "-c", code], check=True, env=headless, cwd=REPOSITORY)
