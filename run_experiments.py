"""Pilot for RQ1: M1 against M2 in every (CV_w, protocol) cell, paired within replicates.

python run_experiments.py                    # E, C, B; n 5000; 60 replicates; no waning; ~10 min
python run_experiments.py --networks E --beta 1 --t-max 50 --replicates 20 --out results/waning

Each replicate builds one graph, runs M1 once and M2 once per cell on the same graph, initial
infected and random streams. Writes runs.csv and summary.csv to --out, the RQ1 figure to --figure.
"""

import argparse
import csv
import math
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

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
from episim.networks import NETWORK_NAMES, create_network, degrees, edge_values, iid_weights
from episim.networks import weight_cv as realised_cv
from episim.plotting import INK, MUTED, SERIES, chart
from episim.simulation import simulate

GRAPH, WEIGHTS, SEEDING, DYNAMICS, BOOTSTRAP = range(5)
PROTOCOLS = {"i": "(i) equal mean rate", "ii": "(ii) equal $R_0$", "iii": "(iii) equal growth rate"}
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


def replicate(task: tuple[str, int, argparse.Namespace]) -> list[dict]:
    kind, index, args = task
    disease = {"sigma": main.sigma, "gamma": main.gamma, "delta": main.delta}

    def generator(purpose: int) -> np.random.Generator:
        return np.random.default_rng([args.seed, index, purpose])

    graph = create_network(kind, args.n, args.k, generator(GRAPH))
    initial_infected = generator(SEEDING).choice(len(graph), args.initial, replace=False).tolist()
    pattern = nonbacktracking_pattern(graph)
    rho_unit = unit_radius(graph, pattern)

    def row(model: str, network, cv: float, protocol: str, c: float) -> dict:
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
            "cv": cv,
            "protocol": protocol,
            "mean_degree": float(degrees(network).mean()),
            "cv_w": realised_cv(network) if model == "M2" else 0.0,
            "c": c,
            "tau_max": tau_max(network, c, args.alpha) if model == "M2" else c,
            "mean_rate": float(rates.mean()),
            "mean_t": float(t.mean()),
            "r0": r0_nonbacktracking(network, t, pattern),
            "r0_newman": r0_newman(network, t),
            "r_analytic": growth_rate(network, **disease, pattern=pattern, rho_unit=rho_unit),
            "attack_rate_mp": attack_rate_message_passing(network, t, pattern),
            **run_metrics(*run, args.t_max),
        }

    set_rates(graph, main.tau)
    rows = [row("M1", graph, 0.0, "-", main.tau)]
    for cv in args.cv:
        weighted = iid_weights(graph, cv, generator(WEIGHTS))
        for protocol in args.protocols:
            c = match_protocol(
                protocol, weighted, args.alpha, main.tau, rho_unit, **disease, pattern=pattern
            )
            set_rates(weighted, c, args.alpha)
            rows.append(row("M2", weighted, cv, protocol, c))
    return rows


def run_grid(args: argparse.Namespace) -> list[dict]:
    tasks = [(kind, index, args) for kind in args.networks for index in range(args.replicates)]
    os.environ.update(dict.fromkeys(BLAS_THREADS, "1"))
    rows = []
    with get_context("spawn").Pool(args.workers) as pool:
        for done, replicate_rows in enumerate(pool.imap(replicate, tasks), start=1):
            rows += replicate_rows
            print(f"\r{done}/{len(tasks)} replicates", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return rows


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
            cells.setdefault((row["network"], row["cv"], row["protocol"]), []).append(row)

    summary = []
    for (network, cv, protocol), m2_rows in cells.items():
        m1_rows = [m1[network, row["replicate"]] for row in m2_rows]
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
                    "network": network,
                    "cv": cv,
                    "protocol": protocol,
                    "metric": metric,
                    "replicates": len(m2_rows),
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
    """One line per cell: medians of the analytic quantities, then relative differences in %."""
    cells: dict[tuple, dict[str, dict]] = {}
    for entry in summary:
        cells.setdefault((entry["network"], entry["cv"], entry["protocol"]), {})[
            entry["metric"]
        ] = entry
    print(
        f"{'net':<4}{'CV_w':>5}{'prot':>5}{'major M1/M2':>13}{'pairs':>6}{'R0 M1/M2':>12}"
        f"{'r* M1/M2':>12}" + "".join(f"{name + ' %':>24}" for name in PLOTTED)
    )
    for (network, cv, protocol), metrics in cells.items():
        any_metric = metrics["r0"]
        line = (
            f"{network:<4}{cv:>5.2f}{protocol:>5}"
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


def rq1_figure(summary: list[dict], path: Path) -> None:
    """Median paired relative difference against CV_w: a row per network, a column per metric,
    a line per protocol, bars for the 95 % bootstrap interval."""
    networks = list(dict.fromkeys(entry["network"] for entry in summary))
    protocols = list(dict.fromkeys(entry["protocol"] for entry in summary))
    figure, axes = chart(
        len(networks), len(PLOTTED), figsize=(3.4 * len(PLOTTED), 2.4 * len(networks) + 0.9)
    )
    for row, network in enumerate(networks):
        for column, metric in enumerate(PLOTTED):
            ax = axes[row, column]
            ax.axhline(0, color=MUTED, linewidth=0.8)
            for index, (protocol, colour) in enumerate(zip(protocols, SERIES, strict=False)):
                entries = [
                    entry
                    for entry in summary
                    if (entry["network"], entry["protocol"], entry["metric"])
                    == (network, protocol, metric)
                ]
                cv = np.array([entry["cv"] for entry in entries])
                middle = 100 * np.array([entry["difference"] for entry in entries])
                low = 100 * np.array([entry["difference_low"] for entry in entries])
                high = 100 * np.array([entry["difference_high"] for entry in entries])
                ax.errorbar(
                    cv + 0.03 * (index - 1),  # side by side, so the intervals stay readable
                    middle,
                    yerr=[middle - low, high - middle],
                    color=colour,
                    marker="osD"[index],
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    linewidth=2,
                    elinewidth=1.2,
                    capsize=2.5,
                    label=PROTOCOLS.get(protocol, protocol),
                )
            ax.set_xticks(cv)
            if row == 0:
                ax.set_title(SIMULATED[metric], color=INK, fontsize=11)
            if row == len(networks) - 1:
                ax.set_xlabel("$CV_w$ (weight dispersion)", color=MUTED)
            if column == 0:
                ax.set_ylabel(f"{NETWORK_NAMES[network].split(' (')[0]}\nM2 vs M1 (%)", color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=len(labels), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def parse(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired M1 vs M2 experiment over CV_w x protocol.")
    parser.add_argument("--networks", default="ECB", help=f"kinds from {''.join(NETWORK_NAMES)}")
    parser.add_argument("--n", type=int, default=5000, help="number of nodes")
    parser.add_argument("--k", type=int, default=7, help="mean degree")
    parser.add_argument("--replicates", type=int, default=60)
    parser.add_argument("--initial", type=int, default=2, help="initially infected nodes")
    parser.add_argument(
        "--cv", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0], help="CV_w levels"
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="rate exponent in c w^alpha")
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    parser.add_argument("--beta", type=float, default=0.0, help="waning rate R -> S (0 = SEIR+D)")
    parser.add_argument(
        "--t-max", type=float, default=math.inf, help="time cap, needed with waning"
    )
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--out", type=Path, default=Path("results/pilot"), help="folder for the CSVs"
    )
    parser.add_argument("--figure", type=Path, help="RQ1 figure; default <out>/rq1.png")
    return parser.parse_args(arguments)


def run(arguments: list[str] | None = None) -> None:
    args = parse(arguments)
    started = time.perf_counter()
    rows = run_grid(args)
    summary = summarise(rows, args.seed)
    write_csv(rows, args.out / "runs.csv")
    write_csv(summary, args.out / "summary.csv")
    figure = args.figure or args.out / "rq1.png"
    rq1_figure(summary, figure)
    print(
        f"sigma={main.sigma} gamma={main.gamma} tau={main.tau} delta={main.delta} beta={args.beta}"
        f" | n={args.n} k={args.k} alpha={args.alpha} replicates={args.replicates} seed={args.seed}"
    )
    print_summary(summary)
    print(f"{len(rows)} runs in {time.perf_counter() - started:.0f} s -> {args.out}/, {figure}")


if __name__ == "__main__":
    run()
