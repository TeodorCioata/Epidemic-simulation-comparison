"""Event-driven SEIR+D with waning and per-edge rates, generalising main.simulate_epidemic."""

import heapq
import math
from collections.abc import Iterator, Sequence

import networkx as nx
import numpy as np

Event = tuple[float, int, str]

PREVIOUS_STATE = {"E": "S", "I": "E", "R": "I", "D": "I", "S": "R"}


def simulate(
    graph: nx.Graph,
    initial_infected: Sequence[int],
    rng: np.random.Generator,
    *,
    sigma: float,
    gamma: float,
    delta: float,
    beta: float,
    t_max: float = math.inf,
) -> tuple[list[str], list[Event]]:
    # Sorted neighbours: the random stream is then read in the same order on a graph and its copies.
    adjacency = [
        [(v, graph.adj[u][v]["rate"]) for v in sorted(graph.adj[u])] for u in range(len(graph))
    ]
    draw = _exponentials(rng).__next__

    def waiting(rate: float) -> float:
        return draw() / rate if rate > 0 else math.inf

    infected = sorted({int(u) for u in initial_infected})
    state = ["S"] * len(graph)
    for u in infected:
        state[u] = "I"
    initial_states = state.copy()
    exit_time = [math.inf] * len(graph)
    next_contact: dict[tuple[int, int], float] = {}
    heap: list[Event] = []
    events: list[Event] = []

    def schedule(time: float, node: int, new_state: str) -> None:
        if time < t_max:
            heapq.heappush(heap, (time, node, new_state))

    def contact(u: int, v: int, rate: float, now: float) -> None:
        next_contact[u, v] = time = now + waiting(rate)
        if time < exit_time[u]:
            schedule(time, v, "E")

    def become_infectious(u: int, now: float) -> None:
        exit_time[u], outcome = min((now + waiting(gamma), "R"), (now + waiting(delta), "D"))
        schedule(exit_time[u], u, outcome)
        for v, rate in adjacency[u]:
            if state[v] == "S":
                contact(u, v, rate, now)
            else:
                next_contact[u, v] = now  # nothing sampled yet this infectious period

    for u in infected:
        become_infectious(u, 0.0)
    while heap:
        time, node, new_state = heapq.heappop(heap)
        if state[node] != PREVIOUS_STATE[new_state]:
            continue
        state[node] = new_state
        events.append((time, node, new_state))
        if new_state == "E":
            schedule(time + waiting(sigma), node, "I")
        elif new_state == "I":
            become_infectious(node, time)
        elif new_state == "R":
            schedule(time + waiting(beta), node, "S")
        elif new_state == "S":
            # Memorylessness: an infectious neighbour with no contact pending draws a fresh one now.
            for u, rate in adjacency[node]:
                if state[u] == "I" and next_contact[u, node] <= time:
                    contact(u, node, rate, time)
    return initial_states, events


def _exponentials(rng: np.random.Generator, batch: int = 4096) -> Iterator[float]:
    while True:
        yield from rng.standard_exponential(batch).tolist()


def states_at(
    initial_states: Sequence[str], events: Sequence[Event], times: Sequence[float]
) -> list[list[str]]:
    states, snapshots, next_event = list(initial_states), [], 0
    for t in times:
        while next_event < len(events) and events[next_event][0] <= t:
            _, node, new_state = events[next_event]
            states[node] = new_state
            next_event += 1
        snapshots.append(states.copy())
    return snapshots
