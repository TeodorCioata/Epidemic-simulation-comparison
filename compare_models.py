"""M1 (tau on every edge) vs M2 (weighted, matched to M1's R_0), mirroring main.simulate_epidemic.

python compare_models.py                          # main.py's setup: E, B, C; n 100, <k> 7, seed 113
python compare_models.py --networks EBCHW --cv 2 --alpha 1.5
python compare_models.py --n 10000 --networks E   # large n: summary only, no drawings
"""

import argparse
import time
from pathlib import Path

import numpy as np

from episim.calibration import (
    match_r0,
    nonbacktracking_pattern,
    r0_newman,
    r0_nonbacktracking,
    set_rates,
    tau_max,
    transmissibility,
)
from episim.networks import (
    NETWORK_NAMES,
    create_network,
    degrees,
    edge_values,
    iid_weights,
    strength_exponent,
    weight_cv,
    weight_degree_correlation,
)
from episim.plotting import MAX_DRAWN_NODES, layout, save_snapshots
from episim.simulation import simulate
from main import beta, delta, gamma, sigma, tau

GRAPH, WEIGHTS, SEEDING, DYNAMICS = range(4)
SNAPSHOT_TIMES = (0.0, 1.0, 2.0, 4.0, 8.0)
COLUMNS = (
    f"{'net':<4}{'<k>':>6}{'CV_w':>6}{'rho':>6}{'beta':>6}{'R0 NB':>7}{'Newman M1/M2':>14}{'c':>7}"
    f"{'tau_M':>7}{'<T> M1/M2':>12}{'infections':>14}{'deaths':>12}{'t_end':>13}{'sim s':>11}"
)


def compare(kind: str, args: argparse.Namespace) -> None:
    def generator(purpose: int) -> np.random.Generator:
        return np.random.default_rng([args.seed, purpose])

    graph = create_network(kind, args.n, args.k, generator(GRAPH))
    initial_infected = generator(SEEDING).choice(len(graph), args.initial, replace=False).tolist()
    m2 = graph.copy() if kind == "W" else iid_weights(graph, args.cv, generator(WEIGHTS))
    set_rates(graph, tau)
    pattern = nonbacktracking_pattern(graph)

    def t_of(model):
        return transmissibility(edge_values(model, "rate"), gamma, delta)

    r0_target = r0_nonbacktracking(graph, t_of(graph), pattern)
    c = match_r0(m2, args.alpha, r0_target, gamma, delta, pattern)
    set_rates(m2, c, args.alpha)

    runs, seconds = {}, {}
    for model, network in (("M1", graph), ("M2", m2)):
        started = time.perf_counter()
        runs[model] = simulate(
            network,
            initial_infected,
            generator(DYNAMICS),
            sigma=sigma,
            gamma=gamma,
            delta=delta,
            beta=beta,
        )
        seconds[model] = time.perf_counter() - started

    def pair(values, spec):
        return "/".join(format(value, spec) for value in values)

    def count(model, state):
        return sum(new_state == state for *_, new_state in runs[model][1])

    ends = [events[-1][0] if events else 0.0 for _, events in runs.values()]
    print(
        f"{kind:<4}{degrees(graph).mean():>6.2f}{weight_cv(m2):>6.2f}"
        f"{weight_degree_correlation(m2):>6.2f}{strength_exponent(m2):>6.2f}{r0_target:>7.3f}"
        f"{pair((r0_newman(graph, t_of(graph)), r0_newman(m2, t_of(m2))), '.3f'):>14}"
        f"{c:>7.3f}{tau_max(m2, c, args.alpha):>7.2f}"
        f"{pair((t_of(graph).mean(), t_of(m2).mean()), '.3f'):>12}"
        f"{pair((count('M1', 'E'), count('M2', 'E')), 'd'):>14}"
        f"{pair((count('M1', 'D'), count('M2', 'D')), 'd'):>12}"
        f"{pair(ends, '.1f'):>13}{pair(seconds.values(), '.2f'):>11}"
    )

    if len(graph) <= MAX_DRAWN_NODES:
        times = [*SNAPSHOT_TIMES, max(*ends, SNAPSHOT_TIMES[-1])]
        pos = layout(graph, args.seed)
        for model, network in (("M1", graph), ("M2", m2)):
            save_snapshots(
                network,
                pos,
                *runs[model],
                times,
                Path("snapshots") / kind,
                model,
                f"{NETWORK_NAMES[kind]} ({kind})",
            )


def run() -> None:
    parser = argparse.ArgumentParser(description="Compare M1 and M2 on main.py's networks.")
    parser.add_argument("--networks", default="EBC", help=f"kinds from {''.join(NETWORK_NAMES)}")
    parser.add_argument("--n", type=int, default=100, help="number of nodes")
    parser.add_argument("--k", type=int, default=7, help="mean degree")
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--initial", type=int, default=2, help="initially infected nodes")
    parser.add_argument("--cv", type=float, default=1.0, help="CV_w of the iid weights")
    parser.add_argument("--alpha", type=float, default=1.0, help="rate exponent in c w^alpha")
    args = parser.parse_args()

    print(f"main.py parameters: sigma={sigma} gamma={gamma} tau={tau} delta={delta} beta={beta}")
    print(f"M2: iid gamma weights CV_w={args.cv} (BBV's own for W), alpha={args.alpha}, R0 matched")
    print(COLUMNS)
    for kind in args.networks:
        compare(kind, args)
    if args.n <= MAX_DRAWN_NODES:
        print(f"snapshots at t = {', '.join(map(str, SNAPSHOT_TIMES))} and final: snapshots/")
    else:
        print(f"n > {MAX_DRAWN_NODES}: no network drawings")


if __name__ == "__main__":
    run()
