import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from PIL import Image

from episim.calibration import set_rates
from episim.networks import create_network, edge_values, iid_weights
from episim.plotting import (
    COLOURS,
    comparison_figure,
    draw_states,
    end_time,
    layout,
    save_animation,
    save_curves,
    save_snapshots,
)
from episim.simulation import simulate, states_at

MAIN_PARAMETERS = {"sigma": 1.0, "gamma": 1.0, "delta": 0.1, "beta": 1.0}


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture(scope="module")
def models():
    graph = create_network("E", 40, 4, rng(0))
    m2 = iid_weights(graph, 1.0, rng(1))
    set_rates(graph, 1.0)
    set_rates(m2, 1.0, 1.0)
    runs = {
        model: simulate(network, [0], rng(4), **MAIN_PARAMETERS, t_max=12.0)
        for model, network in (("M1", graph), ("M2", m2))
    }
    assert all(len(events) > 100 for _, events in runs.values())
    return graph, m2, layout(graph, 0), runs


def node_colours(states):
    return [to_rgba(COLOURS[state]) for state in states]


def test_drawing_uses_main_colours_one_layout_and_rate_widths(models):
    graph, m2, pos, runs = models
    initial, events = runs["M1"]
    ax = Figure().subplots()
    for model, time in ((graph, 0.0), (graph, 3.0), (m2, 3.0)):
        states = states_at(initial, events, [time])[0]
        nodes = draw_states(ax, model, pos, states)
        assert [nodes] == [c for c in ax.collections if isinstance(c, PathCollection)]
        [edges] = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert np.allclose(nodes.get_offsets(), [pos[node] for node in model])
        assert np.allclose(nodes.get_facecolors(), node_colours(states))
        rates = edge_values(model, "rate")
        assert np.allclose(edges.get_linewidths(), rates / rates.mean())


def test_snapshots_and_curves_are_written_without_pyplot(tmp_path, models):
    graph, _, pos, runs = models
    open_figures = plt.get_fignums()
    paths = save_snapshots(graph, pos, *runs["M1"], [0, 1, 2], tmp_path, "M1", "E")
    assert [p.name for p in paths] == ["M1_00_t0.0.png", "M1_01_t1.0.png", "M1_02_t2.0.png"]
    assert all(p.stat().st_size > 0 for p in paths)
    assert save_curves(runs, tmp_path / "curves.png").stat().st_size > 0
    assert plt.get_fignums() == open_figures


def test_frames_recolour_the_same_artists_in_place_and_move_the_cursor(models):
    """Frames are evenly spaced in time and show exactly the replayed state, also after a rewind
    (main.py's animation replays one event per frame and never resets)."""
    graph, m2, pos, runs = models
    frames = 25
    figure, show = comparison_figure(graph, m2, pos, runs, frames)
    m1_axes, m2_axes, curves = figure.axes
    artists = [
        next(c for c in ax.collections if isinstance(c, PathCollection))
        for ax in (m1_axes, m2_axes)
    ]
    assert np.allclose(artists[0].get_offsets(), artists[1].get_offsets())
    frame_times = np.linspace(0, end_time(runs), frames)
    assert frame_times[-1] == pytest.approx(max(runs["M1"][1][-1][0], runs["M2"][1][-1][0]))

    for frame in (*range(frames), 0, 7):
        show(frame)
        for artist, ax, run in zip(artists, (m1_axes, m2_axes), runs.values(), strict=True):
            [states] = states_at(*run, [frame_times[frame]])
            assert artist is next(c for c in ax.collections if isinstance(c, PathCollection))
            assert np.allclose(artist.get_facecolors(), node_colours(states))
        assert curves.lines[-1].get_xdata() == [frame_times[frame]] * 2


def test_animation_is_saved_as_a_gif_with_one_image_per_frame(tmp_path, models):
    open_figures = plt.get_fignums()
    path = save_animation(*models, tmp_path / "nested" / "M1_vs_M2.gif", frames=6, fps=10)
    with Image.open(path) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == 6
        assert gif.info["duration"] == 100
    assert plt.get_fignums() == open_figures
