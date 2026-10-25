"""Validation against theory. Each check prints a table, writes it to --tables as a CSV and
(except eon) saves one figure.

python validate.py                      # all six checks, about 3 minutes
python validate.py growth final-size    # read RQ1's runs.csv (run_experiments.py first)
python validate.py r0-gap jensen        # theory only: no simulation, about 20 s

growth      simulated growth rate r-hat against the analytic r, where rho(B(r)) = 1
final-size  simulated attack rate against message passing (no waning)
r0-gap      non-backtracking vs Newman R_0 for iid weights on C, B and H (B's degrees, rewired)
jensen      <T> and R_0 against CV_w under protocol (i), equal mean rate
threshold   P(major) and attack rate near R_0 = 1, against the NB and Newman's R_0
eon         attack-rate distribution against EoN's fast_SIR [7] (dev dependency)
"""

import argparse
import math
import os
from pathlib import Path

import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from episim import main
from episim.calibration import (
    attack_rate_message_passing,
    match_mean_rate,
    match_protocol,
    match_r0,
    nonbacktracking_pattern,
    r0_newman,
    r0_nonbacktracking,
    set_rates,
    transmissibility,
    unit_radius,
)
from episim.metrics import MAJOR_ATTACK_RATE, run_metrics
from episim.networks import NETWORK_NAMES, create_network, degree_weights, edge_values, iid_weights
from episim.plotting import INK, MUTED, SERIES, chart, save_figure, series_legend, series_style
from episim.simulation import simulate
from run_experiments import read_csv, run_parallel, write_csv

CV_LEVELS = (0.0, 0.5, 1.0, 2.0)
R0_LEVELS = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0)


def rng(*seed: int) -> np.random.Generator:
    return np.random.default_rng(list(seed))


def short_name(kind: str) -> str:
    return NETWORK_NAMES[kind].split(" (")[0] if kind in "EB" else NETWORK_NAMES[kind]


def agreement(
    rows: list[dict], predicted: str, simulated: str, label: str, path: Path
) -> list[dict]:
    """Per network and cell, over major outbreaks: the prediction, the simulated median and the
    median relative gap. The figure puts simulation against prediction on the diagonal."""
    networks = list(dict.fromkeys(row["network"] for row in rows))
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        zero_dispersion_copy_of_m1 = row["model"] == "M2" and row["level"] == 0
        if row["major"] and np.isfinite(row[simulated]) and not zero_dispersion_copy_of_m1:
            key = (networks.index(row["network"]), row["model"], row["level"], row["protocol"])
            groups.setdefault(key, []).append(row)
    figure, axes = chart(figsize=(6.2, 5.6))
    ax = axes[0, 0]
    print(f"\n{label}: simulated (median over major outbreaks) against predicted")
    print(
        f"{'net':<4}{'model':<6}{'CV_w':>5}{'prot':>5}{'runs':>6}{'predicted':>11}{'simulated':>11}"
        f"{'gap %':>8}{'gap IQR %':>18}"
    )
    table = []
    for (index, model, cv, protocol), members in sorted(groups.items()):
        network = networks[index]
        theory = np.array([row[predicted] for row in members])
        observed = np.array([row[simulated] for row in members])
        gap, q1, q3 = np.quantile(observed / theory - 1, [0.5, 0.25, 0.75])
        table.append(
            {
                "network": network,
                "model": model,
                "level": cv,
                "protocol": protocol,
                "runs": len(members),
                "predicted": float(np.median(theory)),
                "simulated": float(np.median(observed)),
                "gap": gap,
                "gap_q1": q1,
                "gap_q3": q3,
            }
        )
        gap, q1, q3 = 100 * gap, 100 * q1, 100 * q3
        print(
            f"{network:<4}{model:<6}{cv:>5.2f}{protocol:>5}{len(members):>6}{np.median(theory):>11.4f}"
            f"{np.median(observed):>11.4f}{gap:>8.1f}{f'[{q1:.1f}, {q3:.1f}]':>18}"
        )
        low, middle, high = np.quantile(observed, [0.25, 0.5, 0.75])
        ax.errorbar(
            np.median(theory),
            middle,
            yerr=[[middle - low], [high - middle]],
            color=SERIES[index],
            marker="o" if model == "M1" else "s",
            markersize=9 if model == "M1" else 6,
            markeredgecolor="white",
            markeredgewidth=1.2,
            elinewidth=1.2,
        )
    limits = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(limits, limits, color=MUTED, linewidth=0.8, zorder=0)
    ax.set(xlim=limits, ylim=limits)
    ax.set_xlabel(f"predicted {label}", color=INK)
    ax.set_ylabel(f"simulated {label} (median, quartile bar)", color=INK)
    keys = [
        Line2D([], [], color=colour, marker="o", linestyle="", label=short_name(network))
        for network, colour in zip(networks, SERIES, strict=False)
    ]
    keys += [
        Line2D([], [], color=MUTED, marker="o", markersize=9, linestyle="", label="M1"),
        Line2D([], [], color=MUTED, marker="s", markersize=6, linestyle="", label="M2 cells"),
    ]
    ax.legend(handles=keys, frameon=False, loc="upper left")
    save(figure, path)
    return table


def r0_gap(path: Path, n: int = 10_000, k: int = 4, seeds: int = 5) -> list[dict]:
    """Newman's formula assumes no degree correlations. B has them; H has B's degrees without."""
    tau, gamma, delta = main.tau, main.gamma, main.delta
    figure, axes = chart(figsize=(6.4, 4.4))
    ax = axes[0, 0]
    ax.axhline(0, color=MUTED, linewidth=0.8)
    print(f"\nNB vs Newman R_0, iid weights, rate tau w; n={n}, <k>~{k}, mean of {seeds} graphs")
    print(f"{'net':<4}{'assortativity':>14}" + "".join(f"{f'CV_w {cv:g}':>10}" for cv in CV_LEVELS))
    table = []
    for kind, colour, marker in zip("CBH", SERIES, "osD", strict=True):
        gaps, assortativity = np.empty((seeds, len(CV_LEVELS))), []
        for seed in range(seeds):
            graph = create_network(kind, n, k, rng(seed, 0))
            pattern = nonbacktracking_pattern(graph)
            assortativity.append(nx.degree_assortativity_coefficient(graph))
            for column, cv in enumerate(CV_LEVELS):
                weighted = iid_weights(graph, cv, rng(seed, 1))
                set_rates(weighted, tau, 1.0)
                t = transmissibility(edge_values(weighted, "rate"), gamma, delta)
                gaps[seed, column] = (
                    r0_nonbacktracking(weighted, t, pattern) / r0_newman(weighted, t) - 1
                )
        mean = 100 * gaps.mean(axis=0)
        table += [
            {"network": kind, "assortativity": float(np.mean(assortativity)), "cv": cv, "gap": gap}
            for cv, gap in zip(CV_LEVELS, gaps.mean(axis=0).tolist(), strict=True)
        ]
        print(
            f"{kind:<4}{np.mean(assortativity):>14.3f}" + "".join(f"{gap:>9.1f}%" for gap in mean)
        )
        ax.plot(
            CV_LEVELS,
            mean,
            color=colour,
            marker=marker,
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.5,
            linewidth=2,
            label=short_name(kind),
        )
    ax.set_xticks(CV_LEVELS)
    ax.set_xlabel("$CV_w$ (weight dispersion)", color=INK)
    ax.set_ylabel("non-backtracking $R_0$ vs Newman's (%)", color=INK)
    ax.legend(frameon=False)
    save(figure, path)
    return table


def jensen(path: Path, n: int = 10_000, k: int = 4) -> list[dict]:
    """Protocol (i) holds <tau> fixed; T is concave in tau, so <T> and R_0 fall with CV_w."""
    tau, gamma, delta = main.tau, main.gamma, main.delta
    graph = create_network("C", n, k, rng(0, 0))
    pattern = nonbacktracking_pattern(graph)
    levels = np.arange(0, 3.01, 0.25)
    mean_t, r0 = [], []
    for cv in levels:
        weighted = iid_weights(graph, cv, rng(0, 1))
        set_rates(weighted, match_mean_rate(weighted, 1.0, tau), 1.0)
        t = transmissibility(edge_values(weighted, "rate"), gamma, delta)
        mean_t.append(t.mean())
        r0.append(r0_nonbacktracking(weighted, t, pattern))

    print(f"\nProtocol (i) on C, n={n}, <k>~{k}: equal mean rate tau={tau}")
    print(f"{'CV_w':>5}{'<T>':>8}{'R_0':>8}{'<T> %':>8}{'R_0 %':>8}")
    change_t, change_r0 = 100 * (np.array(mean_t) / mean_t[0] - 1), 100 * (np.array(r0) / r0[0] - 1)
    for row in zip(levels, mean_t, r0, change_t, change_r0, strict=True):
        print("{:>5.2f}{:>8.3f}{:>8.3f}{:>8.1f}{:>8.1f}".format(*row))

    figure, axes = chart(figsize=(6.4, 4.4))
    ax = axes[0, 0]
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.plot(
        levels,
        change_r0,
        color=SERIES[0],
        marker="o",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.5,
        linewidth=2,
        label="$R_0$ (non-backtracking)",
    )
    ax.plot(
        levels,
        change_t,
        color=SERIES[1],
        marker="s",
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=1,
        linewidth=1.2,
        label="mean transmissibility $\\langle T\\rangle$",
    )
    ax.set_xlabel("$CV_w$ (weight dispersion)", color=INK)
    ax.set_ylabel("change against $CV_w = 0$ at equal mean rate (%)", color=INK)
    ax.legend(frameon=False)
    save(figure, path)
    names = ("cv", "mean_t", "r0", "mean_t_change", "r0_change")
    rows = zip(levels, mean_t, r0, change_t / 100, change_r0 / 100, strict=True)
    return [dict(zip(names, map(float, row), strict=True)) for row in rows]


def threshold_runs(task: tuple[str, int, int, int, tuple[float, ...]]) -> list[dict]:
    """One graph: M1 with T = R_0 / rho(B_unit), and M2 with weights (k_i k_j)^0.5 scaled to the
    same NB R_0, at each level; each run starts from one random case, the same for both."""
    kind, index, n, runs, levels = task
    disease = {"sigma": main.sigma, "gamma": main.gamma, "delta": main.delta}
    graph = create_network(kind, n, 7, rng(2, index, 0))
    pattern = nonbacktracking_pattern(graph)
    rho_unit = unit_radius(graph, pattern)
    weighted = degree_weights(graph, 0.5)
    exit_rate = main.gamma + main.delta
    rows = []
    for level in levels:
        t = level / rho_unit
        set_rates(graph, exit_rate * t / (1 - t))
        c = match_r0(weighted, 1.0, level, main.gamma, main.delta, pattern, rho_unit)
        set_rates(weighted, c, 1.0)
        for model, network in (("M1", graph), ("M2", weighted)):
            t_edges = transmissibility(edge_values(network, "rate"), main.gamma, main.delta)
            cell = {"network": kind, "graph": index, "model": model, "level": level}
            cell["r0"] = r0_nonbacktracking(network, t_edges, pattern)
            cell["r0_newman"] = r0_newman(network, t_edges)
            # At R_0 = 1 exactly the iteration converges only algebraically: skip that one level.
            mp = attack_rate_message_passing(network, t_edges, pattern) if level != 1 else math.nan
            cell["attack_rate_mp"] = mp
            for run in range(runs):
                generator = rng(2, index, 1, run)
                case = [int(generator.integers(n))]
                metrics = run_metrics(*simulate(network, case, generator, **disease, beta=0.0))
                rows.append(cell | {"attack_rate": metrics["attack_rate"]})
    return rows


def threshold(
    path: Path, n: int = 5000, graphs: int = 10, runs: int = 20, workers: int | None = None
) -> list[dict]:
    """The epidemic threshold [9] on the actual graph: P(major), the mean attack rate and the
    message-passing final size rise from zero where the NB R_0 crosses 1, for M1 and for
    degree-correlated M2, while Newman's R_0 places the transition elsewhere on B and under
    weight-degree correlation."""
    tasks = [(kind, index, n, runs, R0_LEVELS) for kind in "EB" for index in range(graphs)]
    rows = run_parallel(threshold_runs, tasks, workers or os.cpu_count() or 1)
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["network"], row["model"], row["level"]), []).append(row)
    table = []
    for (kind, model, level), members in groups.items():
        attack_rate = np.array([row["attack_rate"] for row in members])
        major = attack_rate >= MAJOR_ATTACK_RATE
        table.append(
            {
                "network": kind,
                "model": model,
                "level": level,
                "runs": len(members),
                "r0": float(np.mean([row["r0"] for row in members])),
                "r0_newman": float(np.mean([row["r0_newman"] for row in members])),
                "major": float(major.mean()),
                "attack_rate": float(attack_rate.mean()),
                "attack_rate_major": float(np.median(attack_rate[major])) if major.any() else 0.0,
                "attack_rate_mp": float(np.mean([row["attack_rate_mp"] for row in members])),
            }
        )
    print(f"\nThreshold: n={n}, {graphs} graphs x {runs} runs per point, one initial case")
    print(
        f"{'net':<4}{'model':<6}{'NB R0':>7}{'Newman':>8}{'P(major)':>10}{'mean AR':>9}"
        f"{'AR | major':>12}{'theory':>8}"
    )
    for row in table:
        print(
            f"{row['network']:<4}{row['model']:<6}{row['r0']:>7.2f}{row['r0_newman']:>8.2f}"
            f"{row['major']:>10.3f}{row['attack_rate']:>9.3f}{row['attack_rate_major']:>12.3f}"
            f"{row['attack_rate_mp']:>8.3f}"
        )

    figure, axes = chart(2, 2, figsize=(9.6, 6.8))
    styles = {}
    for kind, model in dict.fromkeys((row["network"], row["model"]) for row in table):
        label = f"{short_name(kind)}, {'M1' if model == 'M1' else 'M2 ($\\theta$ = 0.5)'}"
        colour, marker = (SERIES[0], "o") if kind == "E" else (SERIES[2], "D")
        styles[label] = series_style(colour, marker, dashed=model == "M2")
        points = [row for row in table if (row["network"], row["model"]) == (kind, model)]
        for column, x in enumerate(("r0", "r0_newman")):
            for line, y in enumerate(("major", "attack_rate")):
                at = [row[x] for row in points]
                axes[line, column].plot(at, [row[y] for row in points], **styles[label])
    for (line, column), ax in np.ndenumerate(axes):
        ax.axvline(1, color=MUTED, linewidth=0.8)
        if line == 1:
            name = "non-backtracking $R_0$" if column == 0 else "Newman's $R_0$"
            ax.set_xlabel(name, color=INK)
        if column == 0:
            ax.set_ylabel("P(major outbreak)" if line == 0 else "mean attack rate", color=INK)
    series_legend(figure, styles, columns=2)
    save(figure, path)
    return table


def eon(n: int = 200, runs: int = 4000) -> list[dict]:
    """Our attack rates against EoN's fast_SIR [7], given the same per-edge rates, on small E and
    B graphs: SIR with removal rate gamma + delta, as the final size does not depend on sigma.
    M1 has rate 0.5 on every edge (minor and major outbreaks both occur), M2 iid weights with
    CV_w 1.5 matched in R_0. z of the difference in means, and a two-sample KS test."""
    import EoN  # dev dependency, used only here and in the tests
    from scipy.stats import ks_2samp

    disease = {"sigma": main.sigma, "gamma": main.gamma, "delta": main.delta}
    table = []
    for kind in "EB":
        graph = create_network(kind, n, 4, rng(3, 0))
        pattern = nonbacktracking_pattern(graph)
        weighted = iid_weights(graph, 1.5, rng(3, 1))
        set_rates(graph, 0.5)
        rho_unit = unit_radius(graph, pattern)
        set_rates(weighted, match_protocol("ii", weighted, 1.0, 0.5, rho_unit, **disease), 1.0)
        cases = rng(3, 2).integers(n, size=runs).tolist()
        for model, network in (("M1", graph), ("M2", weighted)):
            ours, theirs = np.empty(runs), np.empty(runs)
            for run, case in enumerate(cases):
                log = simulate(network, [case], rng(3, 3, run), **disease, beta=0.0)
                ours[run] = run_metrics(*log)["attack_rate"]
                *_, recovered = EoN.fast_SIR(
                    network,
                    1.0,
                    main.gamma + main.delta,
                    initial_infecteds=[case],
                    transmission_weight="rate",
                    rng=rng(3, 4, run),
                )
                theirs[run] = recovered[-1] / n
            error = np.sqrt(ours.var() / runs + theirs.var() / runs)
            table.append(
                {
                    "network": kind,
                    "model": model,
                    "runs": runs,
                    "attack_rate": float(ours.mean()),
                    "attack_rate_eon": float(theirs.mean()),
                    "z": float((ours.mean() - theirs.mean()) / error),
                    "major": float((ours >= MAJOR_ATTACK_RATE).mean()),
                    "major_eon": float((theirs >= MAJOR_ATTACK_RATE).mean()),
                    "ks_p": float(ks_2samp(ours, theirs).pvalue),
                }
            )
    print(f"\nAgainst EoN fast_SIR: n={n}, <k>~4, {runs} runs each, one random initial case")
    print(
        f"{'net':<4}{'model':<6}{'mean AR':>9}{'EoN':>8}{'z':>7}{'P(major)':>10}{'EoN':>7}"
        f"{'KS p':>8}"
    )
    for row in table:
        print(
            f"{row['network']:<4}{row['model']:<6}{row['attack_rate']:>9.4f}"
            f"{row['attack_rate_eon']:>8.4f}{row['z']:>7.2f}{row['major']:>10.3f}"
            f"{row['major_eon']:>7.3f}{row['ks_p']:>8.3f}"
        )
    return table


def save(figure, path: Path) -> None:
    print(f"-> {save_figure(figure, path)} (and .pdf)")


def run(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validation of the model layer against theory.")
    checks = ("growth", "final-size", "r0-gap", "jensen", "threshold", "eon")
    parser.add_argument("checks", nargs="*", choices=checks, help="default: all")
    parser.add_argument("--runs", type=Path, default=Path("results/rq1/runs.csv"))
    parser.add_argument("--figures", type=Path, default=Path("docs/figures"))
    parser.add_argument("--tables", type=Path, default=Path("results/validation"))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args(arguments)
    figures = args.figures

    def runs() -> list[dict]:
        return read_csv(args.runs)

    run_check = {
        "growth": lambda: agreement(
            runs(), "r_analytic", "growth_rate", "growth rate r", figures / "growth.png"
        ),
        "final-size": lambda: agreement(
            runs(), "attack_rate_mp", "attack_rate", "attack rate", figures / "final_size.png"
        ),
        "r0-gap": lambda: r0_gap(figures / "r0_gap.png"),
        "jensen": lambda: jensen(figures / "jensen.png"),
        "threshold": lambda: threshold(figures / "threshold.png", workers=args.workers),
        "eon": eon,
    }
    for check in args.checks or checks:
        write_csv(run_check[check](), args.tables / f"{check}.csv")


if __name__ == "__main__":
    run()
