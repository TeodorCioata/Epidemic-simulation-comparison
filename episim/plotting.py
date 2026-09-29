"""Snapshots in main.plotting's style, drawn with one fixed layout per graph."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes

from episim.networks import edge_values
from episim.simulation import Event, states_at

COLOURS = {"S": (0, 1, 0), "E": (1, 1, 1), "I": (1, 0, 0), "R": (0, 0, 1), "D": (0, 0, 0)}
MAX_DRAWN_NODES = 500


def layout(graph: nx.Graph, seed: int) -> dict:
    return nx.spring_layout(graph, seed=seed)


def draw_states(
    ax: Axes, graph: nx.Graph, pos: dict, states: Sequence[str], title: str = ""
) -> None:
    """Edge width = tau_ij / <tau>, so M1 keeps nx's default width 1."""
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
        figure, ax = plt.subplots(figsize=(10, 10))
        draw_states(ax, graph, pos, states, f"{model} · {network} · t = {time:.1f}")
        paths.append(folder / f"{model}_{index:02d}_t{time:.1f}.png")
        figure.savefig(paths[-1])
        plt.close(figure)
    return paths
