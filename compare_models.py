"""M1 (tau on every edge) vs M2 (weighted, matched to M1), mirroring main.simulate_epidemic.

python compare_models.py                          # main.py's setup: E, B, C; n 100, <k> 7, seed 113
python compare_models.py --networks EBCHW --cv 2 --alpha 1.5
python compare_models.py --protocol iii --beta 0  # equal growth rate, no waning
python compare_models.py --networks E --animate   # also M1_vs_M2.gif, both models side by side
python compare_models.py --n 10000 --networks E   # large n: summary and curves, no drawings
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np

from episim import main
from episim.calibration import (
    growth_rate,
    match_protocol,
    nonbacktracking_pattern,
    r0_newman,
    r0_nonbacktracking,
    set_rates,
    tau_max,
    transmissibility,
    unit_radius,
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
from episim.plotting import (
    MAX_DRAWN_NODES,
    end_time,
    layout,
    save_animation,
    save_curves,
    save_snapshots,
)
from episim.simulation import simulate

GRAPH, WEIGHTS, SEEDING, DYNAMICS = range(4)
SNAPSHOT_TIMES = (0.0, 1.0, 2.0, 4.0, 8.0)
MATCHED = {"i": "mean rate", "ii": "R0", "iii": "growth rate"}
COLUMNS = (
    f"{'net':<4}{'<k>':>6}{'CV_w':>6}{'rho':>6}{'beta':>6}{'R0 NB M1/M2':>13}{'Newman M1/M2':>14}"
    f"{'r* M1/M2':>14}{'c':>8}{'tau_M':>9}{'<T> M1/M2':>12}{'infections':>15}{'deaths':>12}"
    f"{'t_end':>13}{'sim s':>11}"
)


def compare(kind: str, args: argparse.Namespace) -> None:
    disease = {"sigma": main.sigma, "gamma": main.gamma, "delta": main.delta}

    def generator(purpose: int) -> np.random.Generator:
        return np.random.default_rng([args.seed, purpose])

    graph = create_network(kind, args.n, args.k, generator(GRAPH))
    initial_infected = generator(SEEDING).choice(len(graph), args.initial, replace=False).tolist()
    m2 = graph.copy() if kind == "W" else iid_weights(graph, args.cv, generator(WEIGHTS))
    set_rates(graph, main.tau)
    pattern = nonbacktracking_pattern(graph)
    rho_unit = unit_radius(graph, pattern)
    c = match_protocol(
        args.protocol, m2, args.alpha, main.tau, rho_unit, **disease, pattern=pattern
    )
    set_rates(m2, c, args.alpha)
    models = {"M1": graph, "M2": m2}

    runs, seconds = {}, {}
    for model, network in models.items():
        started = time.perf_counter()
        runs[model] = simulate(
            network,
            initial_infected,
            generator(DYNAMICS),
            **disease,
            beta=args.beta,
            t_max=args.t_max,
        )
        seconds[model] = time.perf_counter() - started

    def pair(values, spec):
        return "/".join(format(value, spec) for value in values)

    def count(model, state):
        return sum(new_state == state for *_, new_state in runs[model][1])

    t = {
        model: transmissibility(edge_values(network, "rate"), main.gamma, main.delta)
        for model, network in models.items()
    }
    ends = [events[-1][0] if events else 0.0 for _, events in runs.values()]
    print(
        f"{kind:<4}{degrees(graph).mean():>6.2f}{weight_cv(m2):>6.2f}"
        f"{weight_degree_correlation(m2):>6.2f}{strength_exponent(m2):>6.2f}"
        f"{pair((r0_nonbacktracking(models[m], t[m], pattern) for m in models), '.3f'):>13}"
        f"{pair((r0_newman(models[m], t[m]) for m in models), '.3f'):>14}"
        f"{pair((growth_rate(models[m], **disease, pattern=pattern) for m in models), '.3f'):>14}"
        f"{c:>8.3f}{tau_max(m2, c, args.alpha):>9.2f}"
        f"{pair((t[m].mean() for m in models), '.3f'):>12}"
        f"{pair((count(m, 'E') for m in models), 'd'):>15}"
        f"{pair((count(m, 'D') for m in models), 'd'):>12}"
        f"{pair(ends, '.1f'):>13}{pair(seconds.values(), '.2f'):>11}"
    )

    folder = args.out / kind
    name = f"{NETWORK_NAMES[kind]} ({kind})"
    save_curves(runs, folder / "curves.png", f"{name} · M2 matched in {MATCHED[args.protocol]}")
    if len(graph) <= MAX_DRAWN_NODES:
        times = [*SNAPSHOT_TIMES, max(end_time(runs), SNAPSHOT_TIMES[-1])]
        pos = layout(graph, args.seed)
        for model, network in models.items():
            save_snapshots(network, pos, *runs[model], times, folder, model, name)
        if args.animate:
            save_animation(graph, m2, pos, runs, folder / "M1_vs_M2.gif")


def run() -> None:
    parser = argparse.ArgumentParser(description="Compare M1 and M2 on main.py's networks.")
    parser.add_argument("--networks", default="EBC", help=f"kinds from {''.join(NETWORK_NAMES)}")
    parser.add_argument("--n", type=int, default=100, help="number of nodes")
    parser.add_argument("--k", type=int, default=7, help="mean degree")
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--initial", type=int, default=2, help="initially infected nodes")
    parser.add_argument("--cv", type=float, default=1.0, help="CV_w of the iid weights")
    parser.add_argument("--alpha", type=float, default=1.0, help="rate exponent in c w^alpha")
    parser.add_argument("--protocol", default="ii", choices=list(MATCHED), help="what M2 matches")
    parser.add_argument("--beta", type=float, default=main.beta, help="waning rate R -> S")
    parser.add_argument("--t-max", type=float, default=math.inf, help="stop the runs at this time")
    parser.add_argument("--out", type=Path, default=Path("snapshots"), help="output folder")
    parser.add_argument("--animate", action="store_true", help="save M1_vs_M2.gif (n <= 500)")
    args = parser.parse_args()

    print(
        f"main.py parameters: sigma={main.sigma} gamma={main.gamma} tau={main.tau} "
        f"delta={main.delta}; waning beta={args.beta}"
    )
    print(
        f"M2: iid gamma weights CV_w={args.cv} (BBV's own for W), alpha={args.alpha}, "
        f"protocol ({args.protocol}): equal {MATCHED[args.protocol]}"
    )
    print(COLUMNS)
    for kind in args.networks:
        compare(kind, args)
    drawn = f"snapshots at t = {', '.join(map(str, SNAPSHOT_TIMES))} and final, "
    if args.n > MAX_DRAWN_NODES:
        drawn = f"n > {MAX_DRAWN_NODES}: no network drawings; "
    print(f"{drawn}curves.png per network: {args.out}/")


if __name__ == "__main__":
    run()
