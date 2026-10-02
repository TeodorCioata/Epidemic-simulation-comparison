"""Validation against theory. Each check prints a table and saves one figure.

python validate.py                      # all four checks
python validate.py growth final-size    # read the pilot's runs.csv (run_experiments.py first)
python validate.py r0-gap jensen        # theory only: no simulation, about 20 s

growth      simulated growth rate r-hat against the analytic r, where rho(B(r)) = 1
final-size  simulated attack rate against message passing (no waning)
r0-gap      non-backtracking vs Newman R_0 for iid weights on C, B and H (B's degrees, rewired)
jensen      <T> and R_0 against CV_w under protocol (i), equal mean rate
"""

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

from episim import main
from episim.calibration import (
    match_mean_rate,
    nonbacktracking_pattern,
    r0_newman,
    r0_nonbacktracking,
    set_rates,
    transmissibility,
)
from episim.networks import NETWORK_NAMES, create_network, edge_values, iid_weights
from episim.plotting import INK, MUTED, SERIES, chart
from run_experiments import read_csv

CV_LEVELS = (0.0, 0.5, 1.0, 2.0)


def rng(*seed: int) -> np.random.Generator:
    return np.random.default_rng(list(seed))


def short_name(kind: str) -> str:
    return NETWORK_NAMES[kind].split(" (")[0] if kind in "EB" else NETWORK_NAMES[kind]


def agreement(rows: list[dict], predicted: str, simulated: str, label: str, path: Path) -> None:
    """Per network and cell, over major outbreaks: the prediction, the simulated median and the
    median relative gap. The figure puts simulation against prediction on the diagonal."""
    networks = list(dict.fromkeys(row["network"] for row in rows))
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        zero_dispersion_copy_of_m1 = row["model"] == "M2" and row["cv"] == 0
        if row["major"] and np.isfinite(row[simulated]) and not zero_dispersion_copy_of_m1:
            key = (networks.index(row["network"]), row["model"], row["cv"], row["protocol"])
            groups.setdefault(key, []).append(row)
    figure, axes = chart(figsize=(6.2, 5.6))
    ax = axes[0, 0]
    print(f"\n{label}: simulated (median over major outbreaks) against predicted")
    print(
        f"{'net':<4}{'model':<6}{'CV_w':>5}{'prot':>5}{'runs':>6}{'predicted':>11}{'simulated':>11}"
        f"{'gap %':>8}{'gap IQR %':>18}"
    )
    for (index, model, cv, protocol), members in sorted(groups.items()):
        network = networks[index]
        theory = np.array([row[predicted] for row in members])
        observed = np.array([row[simulated] for row in members])
        gap, q1, q3 = 100 * np.quantile(observed / theory - 1, [0.5, 0.25, 0.75])
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


def r0_gap(path: Path, n: int = 10_000, k: int = 4, seeds: int = 5) -> None:
    """Newman's formula assumes no degree correlations. B has them; H has B's degrees without."""
    tau, gamma, delta = main.tau, main.gamma, main.delta
    figure, axes = chart(figsize=(6.4, 4.4))
    ax = axes[0, 0]
    ax.axhline(0, color=MUTED, linewidth=0.8)
    print(f"\nNB vs Newman R_0, iid weights, rate tau w; n={n}, <k>~{k}, mean of {seeds} graphs")
    print(f"{'net':<4}{'assortativity':>14}" + "".join(f"{f'CV_w {cv:g}':>10}" for cv in CV_LEVELS))
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


def jensen(path: Path, n: int = 10_000, k: int = 4) -> None:
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


def save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    print(f"-> {path}")


def run(arguments: list[str] | None = None) -> None:
    checks = ("growth", "final-size", "r0-gap", "jensen")
    parser = argparse.ArgumentParser(description="Validation of the model layer against theory.")
    parser.add_argument("checks", nargs="*", choices=checks, help="default: all")
    parser.add_argument("--runs", type=Path, default=Path("results/pilot/runs.csv"))
    parser.add_argument("--figures", type=Path, default=Path("docs/figures"))
    args = parser.parse_args(arguments)
    selected = args.checks or checks

    if {"growth", "final-size"} & set(selected):
        rows = read_csv(args.runs)
    if "growth" in selected:
        agreement(rows, "r_analytic", "growth_rate", "growth rate r", args.figures / "growth.png")
    if "final-size" in selected:
        agreement(
            rows, "attack_rate_mp", "attack_rate", "attack rate", args.figures / "final_size.png"
        )
    if "r0-gap" in selected:
        r0_gap(args.figures / "r0_gap.png")
    if "jensen" in selected:
        jensen(args.figures / "jensen.png")


if __name__ == "__main__":
    run()
