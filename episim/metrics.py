"""Counts over time and per-run metrics, replayed from the event log (initial_states, events)."""

import math
from collections.abc import Sequence

import numpy as np

from episim.simulation import PREVIOUS_STATE, Event

STATES = "SEIRD"
CODE = {state: code for code, state in enumerate(STATES)}
PREVIOUS_CODE = np.array([CODE[PREVIOUS_STATE[state]] for state in STATES])
MAJOR_ATTACK_RATE = 0.05
FIT_WINDOW = (0.005, 0.05)
FIT_MINIMUM_COUNT = 20


def event_arrays(events: Sequence[Event]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not events:
        return np.empty(0), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    times, nodes, states = zip(*events, strict=True)
    return np.array(times), np.array(nodes), np.array([CODE[state] for state in states])


def state_counts(
    initial_states: Sequence[str], events: Sequence[Event]
) -> tuple[np.ndarray, np.ndarray]:
    """counts[i] holds the numbers of S, E, I, R, D from times[i] until times[i + 1]."""
    times, _, new = event_arrays(events)
    steps = np.zeros((len(times) + 1, len(STATES)), dtype=np.int64)
    steps[0] = np.bincount([CODE[state] for state in initial_states], minlength=len(STATES))
    rows = np.arange(1, len(times) + 1)
    steps[rows, new] = 1
    steps[rows, PREVIOUS_CODE[new]] = -1
    return np.concatenate([[0.0], times]), steps.cumsum(axis=0)


def counts_on_grid(
    initial_states: Sequence[str], events: Sequence[Event], grid: Sequence[float]
) -> np.ndarray:
    times, counts = state_counts(initial_states, events)
    return counts[np.searchsorted(times, grid, side="right") - 1]


def fitted_growth_rate(exposure_times: np.ndarray, lower: int, upper: int) -> float:
    """r-hat: slope of log(cumulative incidence) on time, from the lower-th to the upper-th S->E."""
    if not 0 < lower < upper <= len(exposure_times):
        return math.nan
    cumulative = np.arange(lower, upper + 1)
    return float(np.polyfit(exposure_times[lower - 1 : upper], np.log(cumulative), 1)[0])


def run_metrics(
    initial_states: Sequence[str], events: Sequence[Event], t_max: float = math.inf
) -> dict[str, float]:
    """Fractions of N; duration is censored at t_max when the epidemic is not extinct by then."""
    n = len(initial_states)
    event_times, nodes, new = event_arrays(events)
    times, counts = state_counts(initial_states, events)
    infectious = counts[:, CODE["I"]]
    active = counts[:, CODE["E"]] + infectious
    exposed = new == CODE["E"]
    ever_infected = np.array([state != "S" for state in initial_states])
    ever_infected[nodes[exposed]] = True
    attack_rate = float(ever_infected.mean())
    peak = int(infectious.argmax())
    extinct = bool(active[-1] == 0)
    lower = max(FIT_MINIMUM_COUNT, math.ceil(FIT_WINDOW[0] * n))
    return {
        "major": attack_rate >= MAJOR_ATTACK_RATE,
        "extinct": extinct,
        "peak_prevalence": float(infectious[peak] / n),
        "peak_time": float(times[peak]),
        "attack_rate": attack_rate,
        "exposures": float(exposed.sum() / n),
        "deaths": float(counts[-1, CODE["D"]] / n),
        "duration": float(times[np.argmax(active == 0)]) if extinct else t_max,
        "growth_rate": fitted_growth_rate(event_times[exposed], lower, int(FIT_WINDOW[1] * n)),
    }


def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, resamples: int = 2000, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval of the median."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return math.nan, math.nan
    resampled = values[rng.integers(values.size, size=(resamples, values.size))]
    low, high = np.quantile(np.median(resampled, axis=1), [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)
