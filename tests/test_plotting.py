import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgba

from episim.calibration import set_rates
from episim.networks import create_network, edge_values, iid_weights
from episim.plotting import COLOURS, draw_states, layout, save_snapshots
from episim.simulation import simulate, states_at


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_drawing_uses_main_colours_one_layout_and_rate_widths():
    graph = create_network("E", 40, 4, rng(0))
    m2 = iid_weights(graph, 1.0, rng(1))
    set_rates(graph, 1.0)
    set_rates(m2, 1.0, 1.0)
    pos = layout(graph, 0)
    initial, events = simulate(graph, [0], rng(2), sigma=1.0, gamma=1.0, delta=0.1, beta=1.0)

    figure, ax = plt.subplots()
    for model, time in ((graph, 0.0), (graph, 3.0), (m2, 3.0)):
        states = states_at(initial, events, [time])[0]
        draw_states(ax, model, pos, states)
        [nodes] = [c for c in ax.collections if isinstance(c, PathCollection)]
        [edges] = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert np.allclose(nodes.get_offsets(), [pos[node] for node in model])
        assert np.allclose(nodes.get_facecolors(), [to_rgba(COLOURS[s]) for s in states])
        rates = edge_values(model, "rate")
        assert np.allclose(edges.get_linewidths(), rates / rates.mean())
    plt.close(figure)


def test_snapshots_write_one_png_per_time(tmp_path):
    graph = create_network("B", 30, 4, rng(0))
    set_rates(graph, 1.0)
    initial, events = simulate(graph, [0], rng(1), sigma=1.0, gamma=1.0, delta=0.1, beta=1.0)
    open_figures = plt.get_fignums()
    paths = save_snapshots(graph, layout(graph, 0), initial, events, [0, 1, 2], tmp_path, "M1", "B")
    assert [p.name for p in paths] == ["M1_00_t0.0.png", "M1_01_t1.0.png", "M1_02_t2.0.png"]
    assert all(p.stat().st_size > 0 for p in paths)
    assert plt.get_fignums() == open_figures
