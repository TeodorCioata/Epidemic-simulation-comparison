"""Snapshots and the M1-vs-M2 animation in main.py's style, on one fixed layout per graph.

Figures are built as matplotlib.figure.Figure, not through pyplot, so files are rendered with Agg
whatever backend is active and no display is needed."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from episim.metrics import CODE, STATES, event_arrays, state_counts
from episim.networks import edge_values
from episim.simulation import Event, states_at

Run = tuple[Sequence[str], Sequence[Event]]

COLOURS = {"S": (0, 1, 0), "E": (1, 1, 1), "I": (1, 0, 0), "R": (0, 0, 1), "D": (0, 0, 0)}
STATE_NAMES = {"S": "susceptible", "E": "exposed", "I": "infectious", "R": "recovered", "D": "dead"}
LINE_STYLES: tuple[Literal["-", "--"], ...] = ("-", "--")
MAX_DRAWN_NODES = 500

# Analysis charts: three series colours that stay apart under colour-vision deficiency.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def layout(graph: nx.Graph, seed: int) -> dict:
    return nx.spring_layout(graph, seed=seed)


def draw_states(
    ax: Axes, graph: nx.Graph, pos: dict, states: Sequence[str], title: str = ""
) -> PathCollection:
    """Edge width = tau_ij / <tau>, so M1 keeps nx's default width 1. Returns the node artist."""
    rates = edge_values(graph, "rate")
    ax.clear()
    nx.draw(
        graph,
        pos,
        ax=ax,
        node_color=[COLOURS[states[node]] for node in graph],
        edgecolors="black",
        linewidths=0.5,
        width=(rates / rates.mean()).tolist(),
    )
    ax.set_title(title)
    return next(artist for artist in ax.collections if isinstance(artist, PathCollection))


def save_snapshots(
    graph: nx.Graph,
    pos: dict,
    initial_states: Sequence[str],
    events: Sequence[Event],
    times: Sequence[float],
    folder: Path,
    model: str,
    network: str,
) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, (time, states) in enumerate(
        zip(times, states_at(initial_states, events, times), strict=True)
    ):
        figure = Figure(figsize=(10, 10))
        draw_states(figure.subplots(), graph, pos, states, f"{model} · {network} · t = {time:.1f}")
        paths.append(folder / f"{model}_{index:02d}_t{time:.1f}.png")
        figure.savefig(paths[-1])
    return paths


def end_time(runs: dict[str, Run]) -> float:
    return max((events[-1][0] for _, events in runs.values() if events), default=1.0)


def draw_curves(ax: Axes, runs: dict[str, Run], t_end: float) -> None:
    """Nodes in each state against time, in main.py's colours; the second model is dashed.
    The grey background keeps the white exposed curve visible."""
    ax.set_facecolor("0.8")
    styles = dict(zip(runs, LINE_STYLES, strict=False))
    for model, (initial_states, events) in runs.items():
        times, counts = state_counts(initial_states, events)
        times, counts = np.append(times, t_end), np.vstack([counts, counts[-1]])
        for state, code in CODE.items():
            ax.plot(
                times,
                counts[:, code],
                styles[model],
                color=COLOURS[state],
                linewidth=2,
                drawstyle="steps-post",
            )
    ax.set(xlim=(0, t_end), ylim=(0, None), xlabel="time", ylabel="nodes")
    keys = [Line2D([], [], color=COLOURS[s], linewidth=2, label=STATE_NAMES[s]) for s in STATES]
    keys += [Line2D([], [], color="0.3", linestyle=styles[m], label=m) for m in runs]
    ax.legend(
        handles=keys,
        ncols=len(keys),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        facecolor="0.8",
    )


def save_curves(runs: dict[str, Run], path: Path, title: str = "") -> Path:
    figure = Figure(figsize=(10, 4.5), layout="constrained")
    ax = figure.subplots()
    draw_curves(ax, runs, end_time(runs))
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    return path


@dataclass
class _Panel:
    artist: PathCollection
    initial: np.ndarray
    colours: np.ndarray
    nodes: np.ndarray
    new: np.ndarray
    stops: np.ndarray
    applied: int = 0


def comparison_figure(
    m1: nx.Graph, m2: nx.Graph, pos: dict, runs: dict[str, Run], frames: int = 120
) -> tuple[Figure, Callable[[int], None]]:
    """M1 and M2 side by side on one layout, above their epidemic curves with a time cursor, and
    the function that shows frame i. Frames are evenly spaced in time; each one recolours only the
    nodes whose events fall between it and the previous frame, so the event log is walked once."""
    t_end = end_time(runs)
    frame_times = np.linspace(0, t_end, frames)
    figure = Figure(figsize=(11, 8.5), layout="constrained")
    axes = figure.subplot_mosaic([[*runs], ["curves"] * len(runs)], height_ratios=[3, 1.2])
    palette = np.array([to_rgba(COLOURS[state]) for state in STATES])

    panels = []
    for (model, (initial_states, events)), graph in zip(runs.items(), (m1, m2), strict=True):
        times, nodes, new = event_arrays(events)
        initial = palette[[CODE[state] for state in initial_states]]
        panels.append(
            _Panel(
                draw_states(axes[model], graph, pos, initial_states, model),
                initial,
                initial.copy(),
                nodes,
                new,
                np.searchsorted(times, frame_times, side="right"),
            )
        )
    draw_curves(axes["curves"], runs, t_end)
    cursor = axes["curves"].axvline(0.0, color="black", linewidth=1)
    clock = figure.suptitle("")

    def show(frame: int) -> None:
        for panel in panels:
            stop = panel.stops[frame]
            if stop < panel.applied:
                panel.colours, panel.applied = panel.initial.copy(), 0
            # Within a slice the last event of a node wins: reverse, then keep first occurrences.
            nodes = panel.nodes[panel.applied : stop][::-1]
            changed, latest = np.unique(nodes, return_index=True)
            panel.colours[changed] = palette[panel.new[panel.applied : stop][::-1][latest]]
            panel.applied = stop
            panel.artist.set_facecolor(panel.colours)  # type: ignore[arg-type]
        cursor.set_xdata([frame_times[frame]] * 2)
        clock.set_text(f"t = {frame_times[frame]:.1f}")

    return figure, show


def save_animation(
    m1: nx.Graph,
    m2: nx.Graph,
    pos: dict,
    runs: dict[str, Run],
    path: Path,
    *,
    frames: int = 120,
    fps: int = 10,
) -> Path:
    figure, show = comparison_figure(m1, m2, pos, runs, frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(figure, show, frames=frames, repeat=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=80)
    return path


def chart(
    rows: int = 1, columns: int = 1, figsize: tuple[float, float] = (6.4, 4.2)
) -> tuple[Figure, np.ndarray]:
    """Axes for analysis charts: plain surface, hairline grid, no box."""
    figure = Figure(figsize=figsize, facecolor=SURFACE, layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0, labelsize=9)
        ax.spines[:].set_visible(False)
    return figure, axes
