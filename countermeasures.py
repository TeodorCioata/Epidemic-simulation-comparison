"""Countermeasures: does the ranking of node immunisation strategies survive the move from M1 to M2?

python countermeasures.py                        # E and B, n 5000, 100 replicates: ~8 min
python countermeasures.py --networks E --n 2000 --replicates 8 --out results/try_cm

Each replicate builds one graph and matches M2 (iid weights) to M1 in R_0. Before seeding, a share
of the nodes is immunised: they start in R and are never seeded. The four strategies are
random, highest degree, highest strength, and acquaintance (a random neighbour of a random node).
Random, degree and acquaintance immunise the same nodes in M1 and M2; in M1 strength is degree.
On M2 alone, two edge measures: halve the rate of the strongest contacts, or remove random ones.
A reduction is in expected burden, 1 - mean(with) / mean(without) over all replicates, so a
strategy that stops outbreaks counts fully. Writes runs.csv and summary.csv to --out, the figure
to --figure.
"""

import argparse
import itertools
import os
import time
from pathlib import Path

import networkx as nx
import numpy as np

from episim import main
from episim.calibration import match_protocol, nonbacktracking_pattern, set_rates, unit_radius
from episim.metrics import run_metrics
from episim.networks import (
    NETWORK_NAMES,
    create_network,
    degrees,
    edge_values,
    iid_weights,
    strengths,
)
from episim.plotting import INK, MUTED, SERIES, chart, save_figure, series_legend, series_style
from episim.simulation import simulate
from run_experiments import run_parallel, write_csv

GRAPH, WEIGHTS, SEEDING, DYNAMICS, BOOTSTRAP, INTERVENTION = 0, 1, 2, 3, 4, 6
STRATEGIES = ("random", "degree", "strength", "acquaintance")
EDGE_MEASURES = ("halve strongest", "remove random")
# Random immunisation is the reference, in grey: three hues stay apart in every pair, four do not.
STYLE = {
    "random": (MUTED, "v"),
    "degree": (SERIES[0], "o"),
    "strength": (SERIES[1], "s"),
    "acquaintance": (SERIES[2], "D"),
}
REDUCED = {"attack_rate": "attack rate", "peak_prevalence": "peak prevalence"}


def immunised(graph: nx.Graph, strategy: str, count: int, rng: np.random.Generator) -> list[int]:
    """count distinct nodes: at random, of highest degree or strength (ties by index), or
    acquaintances, each a random neighbour of a random node."""
    if strategy == "random":
        return rng.choice(len(graph), count, replace=False).tolist()
    if strategy in ("degree", "strength"):
        score = degrees(graph) if strategy == "degree" else strengths(graph)
        return np.argsort(-score, kind="stable")[:count].tolist()
    neighbours = [sorted(graph.adj[u]) for u in range(len(graph))]
    chosen: dict[int, None] = {}
    while len(chosen) < count:
        friends = neighbours[rng.integers(len(graph))]
        if friends:
            chosen[friends[rng.integers(len(friends))]] = None
    return list(chosen)


def edge_measure(
    network: nx.Graph, measure: str, share: float, rng: np.random.Generator
) -> nx.Graph:
    """A copy with the rates halved on the share of strongest edges, or zero on a random share."""
    rates = edge_values(network, "rate")
    count = round(share * len(rates))
    if measure == "halve strongest":
        rates[np.argsort(-edge_values(network, "weight"), kind="stable")[:count]] *= 0.5
    else:
        rates[rng.choice(len(rates), count, replace=False)] = 0.0
    changed = network.copy()
    nx.set_edge_attributes(changed, dict(zip(network.edges, rates.tolist(), strict=True)), "rate")
    return changed


def replicate(task: tuple[str, int, argparse.Namespace]) -> list[dict]:
    kind, index, args = task
    disease = {"sigma": main.sigma, "gamma": main.gamma, "delta": main.delta}

    def generator(purpose: int) -> np.random.Generator:
        return np.random.default_rng([args.seed, index, purpose])

    graph = create_network(kind, args.n, args.k, generator(GRAPH))
    pattern = nonbacktracking_pattern(graph)
    m2 = iid_weights(graph, args.cv, generator(WEIGHTS))
    set_rates(graph, main.tau)
    rho_unit = unit_radius(graph, pattern)
    c = match_protocol("ii", m2, args.alpha, main.tau, rho_unit, **disease, pattern=pattern)
    set_rates(m2, c, args.alpha)
    order = generator(SEEDING).permutation(len(graph)).tolist()

    def run(model: str, network: nx.Graph, strategy: str, coverage: float, immune=()) -> dict:
        blocked = set(immune)
        initial = [u for u in order if u not in blocked][: args.initial]
        log = simulate(network, initial, generator(DYNAMICS), **disease, beta=0.0, recovered=immune)
        return {
            "network": kind,
            "replicate": index,
            "model": model,
            "strategy": strategy,
            "coverage": coverage,
            **run_metrics(*log),
        }

    models = {"M1": graph, "M2": m2}
    rows = [run(model, network, "none", 0.0) for model, network in models.items()]
    for strategy, coverage in itertools.product(STRATEGIES, args.coverage):
        for model, network in models.items():
            count = round(coverage * len(graph))
            immune = immunised(network, strategy, count, generator(INTERVENTION))
            rows.append(run(model, network, strategy, coverage, immune))
    for measure in EDGE_MEASURES if args.edge_share else ():
        changed = edge_measure(m2, measure, args.edge_share, generator(INTERVENTION))
        rows.append(run("M2", changed, measure, args.edge_share))
    return rows


def summarise(rows: list[dict], seed: int) -> list[dict]:
    """Per network, model, strategy and coverage over all replicates: P(major), the mean attack
    rate and peak, their reduction 1 - mean(with) / mean(without) with a 95 % paired bootstrap
    interval over replicates, and the rank of the immunisation strategies by attack-rate
    reduction at each coverage (ties share a rank)."""
    rng = np.random.default_rng([seed, BOOTSTRAP])
    untreated = {
        (row["network"], row["model"], row["replicate"]): row
        for row in rows
        if row["strategy"] == "none"
    }
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["network"], row["model"], row["strategy"], row["coverage"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (network, model, strategy, coverage), members in groups.items():
        entry = {"network": network, "model": model, "strategy": strategy, "coverage": coverage}
        entry["replicates"] = len(members)
        entry["major"] = float(np.mean([row["major"] for row in members]))
        picks = rng.integers(len(members), size=(2000, len(members)))
        for metric in REDUCED:
            treated = np.array([row[metric] for row in members])
            before = np.array([untreated[network, model, r["replicate"]][metric] for r in members])
            resampled = 1 - treated[picks].mean(axis=1) / before[picks].mean(axis=1)
            entry[metric] = float(treated.mean())
            entry[f"{metric}_reduction"] = float(1 - treated.mean() / before.mean())
            low, high = np.quantile(resampled, [0.025, 0.975])
            entry[f"{metric}_low"], entry[f"{metric}_high"] = float(low), float(high)
        summary.append(entry)
    for entry in summary:
        rivals = [rival["attack_rate_reduction"] for rival in rivals_of(summary, entry)]
        better = sum(value > entry["attack_rate_reduction"] for value in rivals)
        entry["rank"] = 1 + better if entry["strategy"] in STRATEGIES else "-"
    return summary


def rivals_of(summary: list[dict], entry: dict) -> list[dict]:
    """The immunisation strategies of entry's network, model and coverage."""
    cell = (entry["network"], entry["model"], entry["coverage"])
    return [
        rival
        for rival in summary
        if (rival["network"], rival["model"], rival["coverage"]) == cell
        and rival["strategy"] in STRATEGIES
    ]


def ranking(entries: list[dict]) -> str:
    """Strategies by attack-rate reduction, e.g. 'degree = strength > acquaintance > random'."""
    ordered = sorted(entries, key=lambda entry: -entry["attack_rate_reduction"])
    text = ordered[0]["strategy"]
    for before, after in itertools.pairwise(ordered):
        tie = after["attack_rate_reduction"] == before["attack_rate_reduction"]
        text += (" = " if tie else " > ") + after["strategy"]
    return text


def print_summary(summary: list[dict]) -> None:
    print(
        f"{'net':<4}{'strategy':<17}{'cov':>5}{'P(major) M1/M2':>16}"
        + "".join(f"{f'{name} reduction % M1 | M2':>40}" for name in REDUCED.values())
        + f"{'rank M1/M2':>12}"
    )
    by_cell = {(e["network"], e["strategy"], e["coverage"], e["model"]): e for e in summary}
    for network, strategy, coverage in dict.fromkeys(key[:3] for key in by_cell):
        pair = [by_cell.get((network, strategy, coverage, model)) for model in ("M1", "M2")]
        major = "/".join("-" if e is None else f"{e['major']:.2f}" for e in pair)
        line = f"{network:<4}{strategy:<17}{100 * coverage:>4.0f}%{major:>16}"
        for metric in REDUCED:
            line += f"{' | '.join(reduction(e, metric) for e in pair):>40}"
        print(line + f"{'/'.join('-' if e is None else str(e['rank']) for e in pair):>12}")
    print("\nranking by attack-rate reduction")
    for network, coverage in dict.fromkeys((e["network"], e["coverage"]) for e in summary):
        cells = [by_cell.get((network, "random", coverage, model)) for model in ("M1", "M2")]
        if all(cells):
            orders = [ranking(rivals_of(summary, cell)) for cell in cells]
            print(f"{network:<4}{100 * coverage:>4.0f}%  M1: {orders[0]:<45} M2: {orders[1]}")


def reduction(entry: dict | None, metric: str) -> str:
    if entry is None:
        return "-"
    low, high = 100 * entry[f"{metric}_low"], 100 * entry[f"{metric}_high"]
    return f"{100 * entry[f'{metric}_reduction']:.1f} [{low:.1f}, {high:.1f}]"


def countermeasure_figure(summary: list[dict], path: Path) -> None:
    """Reduction against coverage: a row per network, a column per metric, colour and marker per
    strategy, M1 solid and M2 dashed, with 95 % bootstrap bars."""
    networks = list(dict.fromkeys(entry["network"] for entry in summary))
    figure, axes = chart(
        len(networks), len(REDUCED), figsize=(4.6 * len(REDUCED), 2.6 * len(networks) + 1.0)
    )
    series = list(itertools.product(STRATEGIES, ("M1", "M2")))
    for row, network in enumerate(networks):
        for column, (metric, name) in enumerate(REDUCED.items()):
            ax = axes[row, column]
            for position, (strategy, model) in enumerate(series):
                entries = sorted(
                    (
                        entry
                        for entry in summary
                        if (entry["network"], entry["strategy"], entry["model"])
                        == (network, strategy, model)
                    ),
                    key=lambda entry: entry["coverage"],
                )
                at = 100 * np.array([entry["coverage"] for entry in entries])
                middle, low, high = (
                    100 * np.array([entry[f"{metric}{suffix}"] for entry in entries])
                    for suffix in ("_reduction", "_low", "_high")
                )
                ax.errorbar(
                    at + 0.3 * (position - (len(series) - 1) / 2),  # side by side
                    middle,
                    yerr=[middle - low, high - middle],
                    elinewidth=1.2,
                    capsize=2.5,
                    **series_style(*STYLE[strategy], dashed=model == "M2"),
                )
                ax.set_xticks(at)
            if row == 0:
                ax.set_title(f"reduction in {name}", color=INK, fontsize=11)
            if row == len(networks) - 1:
                ax.set_xlabel("coverage (% of nodes immunised)", color=MUTED)
            if column == 0:
                ax.set_ylabel(f"{NETWORK_NAMES[network].split(' (')[0]}\nreduction (%)", color=INK)
    styles = {strategy: series_style(*STYLE[strategy]) for strategy in STRATEGIES}
    styles["M1 (unweighted)"] = {"color": INK, "linewidth": 2}
    styles["M2 (weighted)"] = {"color": INK, "linewidth": 2, "linestyle": "--"}
    series_legend(figure, styles, columns=3)
    save_figure(figure, path)


def parse(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Node immunisation and edge measures, M1 vs M2.")
    parser.add_argument("--networks", default="EB", help=f"kinds from {''.join(NETWORK_NAMES)}")
    parser.add_argument("--n", type=int, default=5000, help="number of nodes")
    parser.add_argument("--k", type=int, default=7, help="mean degree")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--initial", type=int, default=2, help="initially infected nodes")
    parser.add_argument("--cv", type=float, default=2.0, help="CV_w of M2's iid weights")
    parser.add_argument("--alpha", type=float, default=1.0, help="rate exponent in c w^alpha")
    parser.add_argument(
        "--coverage", type=float, nargs="+", default=[0.05, 0.1, 0.2], help="shares immunised"
    )
    parser.add_argument("--edge-share", type=float, default=0.1, help="M2 edges changed; 0: none")
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=Path("results/countermeasures"))
    parser.add_argument("--figure", type=Path, help="default <out>/countermeasures.png (and .pdf)")
    return parser.parse_args(arguments)


def run(arguments: list[str] | None = None) -> None:
    args = parse(arguments)
    started = time.perf_counter()
    tasks = [(kind, index, args) for kind in args.networks for index in range(args.replicates)]
    rows = run_parallel(replicate, tasks, args.workers)
    summary = summarise(rows, args.seed)
    write_csv(rows, args.out / "runs.csv")
    write_csv(summary, args.out / "summary.csv")
    figure = args.figure or args.out / "countermeasures.png"
    countermeasure_figure(summary, figure)
    print(
        f"protocol (ii), iid weights CV_w={args.cv}, alpha={args.alpha}, beta=0 | n={args.n} "
        f"k={args.k} replicates={args.replicates} seed={args.seed}"
    )
    print_summary(summary)
    print(f"{len(rows)} runs in {time.perf_counter() - started:.0f} s -> {args.out}/, {figure}")


if __name__ == "__main__":
    run()
