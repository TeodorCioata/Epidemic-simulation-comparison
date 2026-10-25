"""Paired M1 vs M2 experiment over weights x level x alpha x protocol cells, for RQ1 and RQ2.

python run_experiments.py                    # RQ1: E, C, B x iid CV_w x protocol; 16-21 min
python run_experiments.py --networks EB --cv --theta -0.5 0 0.5 1 --shuffle --protocols ii iii
python run_experiments.py --networks EB --cv 1 --alpha 0 0.5 1 2 --protocols ii iii
python run_experiments.py --networks W --reinforcement 1 --shuffle --protocols ii iii
python run_experiments.py --networks E --beta 1 --t-max 50 --replicates 20 --out results/waning

Each replicate builds one graph, runs M1 once and M2 once per cell on the same graph, initial
infected and random streams. M2's weights are iid gamma at each --cv, (k_i k_j)^theta at each
--theta, or BBV's own on W; --shuffle adds each correlated one with its weights permuted across
edges. Writes runs.csv and summary.csv to --out, the difference figure to --figure.
"""

import argparse
import csv
import math
import os
import sys
import time
from collections.abc import Callable, Iterator
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from episim import main
from episim.calibration import (
    attack_rate_message_passing,
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
from episim.metrics import bootstrap_ci, run_metrics
from episim.networks import (
    NETWORK_NAMES,
    create_network,
    degree_weights,
    degrees,
    edge_values,
    iid_weights,
    shuffled_weights,
    weight_degree_correlation,
)
from episim.networks import weight_cv as realised_cv
from episim.plotting import INK, MUTED, SERIES, chart, save_figure, series_legend, series_style
from episim.simulation import simulate

GRAPH, WEIGHTS, SEEDING, DYNAMICS, BOOTSTRAP, SHUFFLE = range(6)
CELL = ("network", "weights", "level", "shuffled", "alpha", "protocol")
M1_CELL = {"weights": "-", "level": 0.0, "shuffled": False, "alpha": 0.0, "protocol": "-"}
PROTOCOLS = {"i": "(i) equal mean rate", "ii": "(ii) equal $R_0$", "iii": "(iii) equal growth rate"}
LEVELS = {
    "iid": "$CV_w$ (weight dispersion)",
    "degree": "$\\theta$ in $w_{ij} \\propto (k_i k_j)^\\theta$",
    "bbv": "BBV reinforcement",
}
ANALYTIC = {"r0": "R_0", "r_analytic": "analytic r", "attack_rate_mp": "attack rate (theory)"}
SIMULATED = {
    "peak_prevalence": "peak prevalence",
    "peak_time": "peak time",
    "attack_rate": "attack rate",
    "growth_rate": "growth rate $\\hat r$",
    "deaths": "deaths",
    "duration": "duration",
}
PLOTTED = ("peak_prevalence", "peak_time", "attack_rate", "growth_rate")
# One BLAS thread per worker: several workers each spinning a thread pool slow each other down.
BLAS_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def weight_cells(
    kind: str, graph: nx.Graph, args: argparse.Namespace, generator: Callable
) -> Iterator[tuple[str, float, bool, nx.Graph]]:
    """M2's weighted copies as (weights, level, shuffled, copy): BBV's own on W, otherwise iid at
    each CV_w and degree-correlated at each theta; --shuffle follows each correlated one by its
    shuffle null, from one permutation per replicate."""
    if kind == "W":
        cells = [("bbv", args.reinforcement, graph.copy())]
    else:
        cells = [("iid", cv, iid_weights(graph, cv, generator(WEIGHTS))) for cv in args.cv]
        cells += [("degree", theta, degree_weights(graph, theta)) for theta in args.theta]
    for weights, level, weighted in cells:
        yield weights, level, False, weighted
        if args.shuffle and weights != "iid":
            yield weights, level, True, shuffled_weights(weighted, generator(SHUFFLE))


def replicate(task: tuple[str, int, argparse.Namespace]) -> list[dict]:
    kind, index, args = task
    disease = {"sigma": main.sigma, "gamma": main.gamma, "delta": main.delta}

    def generator(purpose: int) -> np.random.Generator:
        return np.random.default_rng([args.seed, index, purpose])

    graph = create_network(kind, args.n, args.k, generator(GRAPH), reinforcement=args.reinforcement)
    initial_infected = generator(SEEDING).choice(len(graph), args.initial, replace=False).tolist()
    pattern = nonbacktracking_pattern(graph)
    rho_unit = unit_radius(graph, pattern)

    def row(model: str, network: nx.Graph, cell: dict, c: float) -> dict:
        rates = edge_values(network, "rate")
        t = transmissibility(rates, main.gamma, main.delta)
        run = simulate(
            network,
            initial_infected,
            generator(DYNAMICS),
            **disease,
            beta=args.beta,
            t_max=args.t_max,
        )
        return {
            "network": kind,
            "replicate": index,
            "model": model,
            **cell,
            "mean_degree": float(degrees(network).mean()),
            "cv_w": realised_cv(network) if model == "M2" else 0.0,
            "spearman": weight_degree_correlation(network) if model == "M2" else 0.0,
            "c": c,
            "tau_max": tau_max(network, c, cell["alpha"]) if model == "M2" else c,
            "mean_rate": float(rates.mean()),
            "mean_t": float(t.mean()),
            "r0": r0_nonbacktracking(network, t, pattern),
            "r0_newman": r0_newman(network, t),
            "r_analytic": growth_rate(network, **disease, pattern=pattern, rho_unit=rho_unit),
            "attack_rate_mp": attack_rate_message_passing(network, t, pattern),
            **run_metrics(*run, args.t_max),
        }

    set_rates(graph, main.tau)
    rows = [row("M1", graph, M1_CELL, float(main.tau))]
    for weights, level, shuffled, weighted in weight_cells(kind, graph, args, generator):
        for alpha in args.alpha:
            for protocol in args.protocols:
                c = match_protocol(
                    protocol, weighted, alpha, main.tau, rho_unit, **disease, pattern=pattern
                )
                set_rates(weighted, c, alpha)
                cell = {"weights": weights, "level": level, "shuffled": shuffled, "alpha": alpha}
                rows.append(row("M2", weighted, cell | {"protocol": protocol}, c))
    return rows


def run_parallel(function: Callable[[Any], list[dict]], tasks: list, workers: int) -> list[dict]:
    """The rows of every task in task order, from spawned workers with one BLAS thread each."""
    os.environ.update(dict.fromkeys(BLAS_THREADS, "1"))
    rows = []
    with get_context("spawn").Pool(workers) as pool:
        for done, task_rows in enumerate(pool.imap(function, tasks), start=1):
            rows += task_rows
            print(f"\r{done}/{len(tasks)} tasks", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return rows


def run_grid(args: argparse.Namespace) -> list[dict]:
    tasks = [(kind, index, args) for kind in args.networks for index in range(args.replicates)]
    return run_parallel(replicate, tasks, args.workers)


def quartiles(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.5, 0.25, 0.75]).tolist() if values.size else [math.nan] * 3


def summarise(rows: list[dict], seed: int) -> list[dict]:
    """Per cell and metric, over the replicates where M1 and M2 are both major outbreaks: medians
    and quartiles, and the median of (M2 - M1) / M1 with a 95 % bootstrap interval."""
    rng = np.random.default_rng([seed, BOOTSTRAP])
    m1 = {(row["network"], row["replicate"]): row for row in rows if row["model"] == "M1"}
    cells: dict[tuple, list[dict]] = {}
    for row in rows:
        if row["model"] == "M2":
            cells.setdefault(tuple(row[name] for name in CELL), []).append(row)

    summary = []
    for cell, m2_rows in cells.items():
        m1_rows = [m1[row["network"], row["replicate"]] for row in m2_rows]
        both = np.array([a["major"] and b["major"] for a, b in zip(m1_rows, m2_rows, strict=True)])
        for metric in (*ANALYTIC, *SIMULATED):
            a = np.array([row[metric] for row in m1_rows], dtype=float)[both]
            b = np.array([row[metric] for row in m2_rows], dtype=float)[both]
            defined = np.isfinite(a) & np.isfinite(b)
            a, b = a[defined], b[defined]
            relative = (b - a) / a
            low, high = bootstrap_ci(relative, rng)
            summary.append(
                {
                    **dict(zip(CELL, cell, strict=True)),
                    "metric": metric,
                    "replicates": len(m2_rows),
                    "cv_w": float(np.median([row["cv_w"] for row in m2_rows])),
                    "spearman": float(np.median([row["spearman"] for row in m2_rows])),
                    "major_m1": float(np.mean([row["major"] for row in m1_rows])),
                    "major_m2": float(np.mean([row["major"] for row in m2_rows])),
                    "pairs": int(defined.sum()),
                    **dict(zip(("m1_median", "m1_q1", "m1_q3"), quartiles(a), strict=True)),
                    **dict(zip(("m2_median", "m2_q1", "m2_q3"), quartiles(b), strict=True)),
                    "difference": quartiles(relative)[0],
                    "difference_low": low,
                    "difference_high": high,
                }
            )
    return summary


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    """Rows as written by write_csv, with numbers and booleans parsed back."""

    def parsed(text: str) -> object:
        if text in ("True", "False"):
            return text == "True"
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text

    with path.open(newline="") as file:
        return [{name: parsed(text) for name, text in row.items()} for row in csv.DictReader(file)]


def print_summary(summary: list[dict]) -> None:
    """One line per cell: weights, medians of the analytic quantities, relative differences in %."""
    cells: dict[tuple, dict[str, dict]] = {}
    for entry in summary:
        cells.setdefault(tuple(entry[name] for name in CELL), {})[entry["metric"]] = entry
    print(
        f"{'net':<4}{'weights':<8}{'level':>6}{'shuf':>5}{'alpha':>6}{'prot':>5}{'CV_w':>6}"
        f"{'rho':>6}{'major M1/M2':>13}{'pairs':>6}{'R0 M1/M2':>12}{'r* M1/M2':>12}"
        + "".join(f"{name + ' %':>24}" for name in PLOTTED)
    )
    for (network, weights, level, shuffled, alpha, protocol), metrics in cells.items():
        any_metric = metrics["r0"]
        line = (
            f"{network:<4}{weights:<8}{level:>6.2f}{'yes' if shuffled else '':>5}{alpha:>6.2f}"
            f"{protocol:>5}{any_metric['cv_w']:>6.2f}{any_metric['spearman']:>6.2f}"
            f"{any_metric['major_m1']:>8.2f}/{any_metric['major_m2']:.2f}{any_metric['pairs']:>6}"
            f"{metrics['r0']['m1_median']:>7.2f}/{metrics['r0']['m2_median']:.2f}"
            f"{metrics['r_analytic']['m1_median']:>7.2f}/{metrics['r_analytic']['m2_median']:.2f}"
        )
        for name in PLOTTED:
            entry = metrics[name]
            interval = (
                f"[{100 * entry['difference_low']:.1f}, {100 * entry['difference_high']:.1f}]"
            )
            line += f"{100 * entry['difference']:>9.1f} {interval:<14}"
        print(line)


def difference_figure(summary: list[dict], path: Path) -> None:
    """Median paired relative difference against the level of the weights (or alpha, when several
    were run): a row per network, a column per metric, a line per protocol with 95 % bootstrap
    bars; shuffle nulls are dashed with open markers."""
    x = "alpha" if len({entry["alpha"] for entry in summary}) > 1 else "level"
    others = [name for name in CELL[1:] if name != x]
    networks = list(dict.fromkeys(entry["network"] for entry in summary))
    series = list(dict.fromkeys(tuple(entry[name] for name in others) for entry in summary))
    figure, axes = chart(
        len(networks), len(PLOTTED), figsize=(3.4 * len(PLOTTED), 2.4 * len(networks) + 0.9)
    )
    styles: dict[str, dict] = {}
    for row, network in enumerate(networks):
        for column, metric in enumerate(PLOTTED):
            ax = axes[row, column]
            ax.axhline(0, color=MUTED, linewidth=0.8)
            for position, key in enumerate(series):
                cell = dict(zip(others, key, strict=True))
                entries = sorted(
                    (
                        entry
                        for entry in summary
                        if (entry["network"], entry["metric"]) == (network, metric)
                        and all(entry[name] == value for name, value in cell.items())
                    ),
                    key=lambda entry: entry[x],
                )
                at = np.array([entry[x] for entry in entries])
                middle, low, high = (
                    100 * np.array([entry[name] for entry in entries])
                    for name in ("difference", "difference_low", "difference_high")
                )
                index = list(PROTOCOLS).index(cell["protocol"])
                label = PROTOCOLS[cell["protocol"]] + (", shuffle null" if cell["shuffled"] else "")
                styles[label] = series_style(SERIES[index], "osD"[index], dashed=cell["shuffled"])
                ax.errorbar(
                    # side by side, so the intervals stay readable
                    at + 0.015 * (np.ptp(at) or 1.0) * (position - (len(series) - 1) / 2),
                    middle,
                    yerr=[middle - low, high - middle],
                    elinewidth=1.2,
                    capsize=2.5,
                    **styles[label],
                )
                ax.set_xticks(at)
            if row == 0:
                ax.set_title(SIMULATED[metric], color=INK, fontsize=11)
            if row == len(networks) - 1:
                label = "$\\alpha$ (exponent in $c\\,w^\\alpha$)" if x == "alpha" else None
                ax.set_xlabel(label or LEVELS[summary[0]["weights"]], color=MUTED)
            if column == 0:
                ax.set_ylabel(f"{NETWORK_NAMES[network].split(' (')[0]}\nM2 vs M1 (%)", color=INK)
    series_legend(figure, styles)
    save_figure(figure, path)


def parse(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired M1 vs M2 experiment over weight cells.")
    parser.add_argument("--networks", default="ECB", help=f"kinds from {''.join(NETWORK_NAMES)}")
    parser.add_argument("--n", type=int, default=5000, help="number of nodes")
    parser.add_argument("--k", type=int, default=7, help="mean degree")
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--initial", type=int, default=2, help="initially infected nodes")
    parser.add_argument(
        "--cv", type=float, nargs="*", default=[0.0, 0.5, 1.0, 2.0], help="iid CV_w levels"
    )
    parser.add_argument("--theta", type=float, nargs="*", default=[], help="(k_i k_j)^theta levels")
    parser.add_argument("--shuffle", action="store_true", help="add shuffle nulls (non-iid)")
    parser.add_argument("--reinforcement", type=float, default=1.0, help="BBV reinforcement on W")
    parser.add_argument("--alpha", type=float, nargs="+", default=[1.0], help="exponents in c w^a")
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    parser.add_argument("--beta", type=float, default=0.0, help="waning rate R -> S (0 = SEIR+D)")
    parser.add_argument(
        "--t-max", type=float, default=math.inf, help="time cap, needed with waning"
    )
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=Path("results/rq1"), help="folder for the CSVs")
    parser.add_argument("--figure", type=Path, help="default <out>/differences.png (and .pdf)")
    return parser.parse_args(arguments)


def run(arguments: list[str] | None = None) -> None:
    args = parse(arguments)
    started = time.perf_counter()
    rows = run_grid(args)
    summary = summarise(rows, args.seed)
    write_csv(rows, args.out / "runs.csv")
    write_csv(summary, args.out / "summary.csv")
    figure = args.figure or args.out / "differences.png"
    difference_figure(summary, figure)
    print(
        f"sigma={main.sigma} gamma={main.gamma} tau={main.tau} delta={main.delta} beta={args.beta}"
        f" | n={args.n} k={args.k} replicates={args.replicates} seed={args.seed}"
    )
    print_summary(summary)
    print(f"{len(rows)} runs in {time.perf_counter() - started:.0f} s -> {args.out}/, {figure}")


if __name__ == "__main__":
    run()
