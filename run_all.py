"""Regenerate every result, figure and table from scratch: about 54 minutes on 4 cores.

python run_all.py                  # every step, in order, printing each step's time and the total
python run_all.py theta tables     # only some steps; `tables` reads the CSVs already in results/

CSVs go to results/, figures to docs/figures/ (PNG, and PDF for the report), and the tables of
docs/RESULTS.md, in Markdown and in LaTeX (booktabs), to results/tables.md.
"""

import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

import compare_models
import countermeasures
import run_experiments
import validate
from episim import main
from episim.networks import NETWORK_NAMES
from run_experiments import CELL, PLOTTED, read_csv

FIGURES = "docs/figures"
MATCHED = "--replicates 100 --protocols ii iii"
CURVES = "--networks E --n 5000 --beta 0 --cv 2 --seed 1 --protocol {0} --out {1}/protocol_{0}"
STEPS: dict[str, tuple[Callable, str]] = {
    "animation": (compare_models.run, f"--networks E --beta 0 --animate --out {FIGURES}/compare"),
    **{
        f"protocol-{p}": (compare_models.run, CURVES.format(p, FIGURES)) for p in ("i", "ii", "iii")
    },
    "rq1": (run_experiments.run, f"--replicates 100 --out results/rq1 --figure {FIGURES}/rq1.png"),
    "waning": (
        run_experiments.run,
        "--networks E --beta 1 --t-max 50 --replicates 20 --out results/waning"
        f" --figure {FIGURES}/rq1_waning.png",
    ),
    "theta": (
        run_experiments.run,
        f"{MATCHED} --networks EB --cv --theta -0.5 0 0.5 1 --shuffle --out results/theta"
        f" --figure {FIGURES}/theta.png",
    ),
    "alpha": (
        run_experiments.run,
        f"{MATCHED} --networks EB --cv 1 --alpha 0 0.5 1 2 --out results/alpha"
        f" --figure {FIGURES}/alpha.png",
    ),
    "bbv": (run_experiments.run, f"{MATCHED} --networks W --shuffle --out results/bbv"),
    "countermeasures": (countermeasures.run, f"--figure {FIGURES}/countermeasures.png"),
    "validation": (validate.run, ""),
}
DIFFERENCES = ["peak prevalence", "peak time", "attack rate", "$\\hat r$"]


def interval(middle: float, low: float, high: float) -> str:
    return f"{100 * middle:.1f} [{100 * low:.1f}, {100 * high:.1f}]"


def difference(entry: dict) -> str:
    """Median paired (M2 - M1) / M1 in %, with its 95 % bootstrap interval."""
    return interval(entry["difference"], entry["difference_low"], entry["difference_high"])


def reduction(entry: dict, metric: str) -> str:
    return interval(entry[f"{metric}_reduction"], entry[f"{metric}_low"], entry[f"{metric}_high"])


def quartiles(values: list[float], scale: float = 100) -> str:
    """Median (lower quartile, upper quartile)."""
    q1, median, q3 = scale * np.quantile(values, [0.25, 0.5, 0.75])
    return f"{median:.1f} ({q1:.1f}, {q3:.1f})"


def name(kind: str) -> str:
    return NETWORK_NAMES[kind].split(" (")[0] if kind in "EBC" else NETWORK_NAMES[kind]


def latex(text: str) -> str:
    """Markdown cell to LaTeX: escaped %, minus signs, `code` in typewriter."""
    text = re.sub(r"(?<![\w$])-(?=\d)", "$-$", text.replace("%", r"\%"))
    return re.sub(
        r"`([^`]*)`",
        lambda code: r"\texttt{" + code[1].replace("-", "-{}").replace("_", r"\_") + "}",
        text,
    )


def table(caption: str, header: list[str], rows: list[list[str]]) -> str:
    """One table in Markdown and, below it, the same table in LaTeX (booktabs)."""
    markdown = ["| " + " | ".join(cells) + " |" for cells in (header, ["---"] * len(header), *rows)]
    body = [" & ".join(map(latex, cells)) + r" \\" for cells in (header, *rows)]
    tex = [r"\begin{table}[ht]", r"\centering", rf"\caption{{{latex(caption)}}}"]
    tex += [rf"\begin{{tabular}}{{{'l' * len(header)}}}", r"\toprule", body[0], r"\midrule"]
    tex += [*body[1:], r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join([f"### {caption}", "", *markdown, "", "```latex", *tex, "```", ""])


def summary_of(folder: str) -> dict[tuple, dict]:
    """Summary entries by (network, weights, level, shuffled, alpha, protocol, metric)."""
    entries = read_csv(Path(folder) / "summary.csv")
    return {tuple(entry[key] for key in (*CELL, "metric")): entry for entry in entries}


def parameters() -> str:
    rows = [
        ["$\\sigma$", f"{main.sigma}", "rate E $\\to$ I", "`main.py`"],
        ["$\\gamma$", f"{main.gamma}", "rate I $\\to$ R", "`main.py`"],
        ["$\\delta$", f"{main.delta}", "rate I $\\to$ D", "`main.py`"],
        ["$\\tau$", f"{main.tau}", "rate S $\\to$ E per contact (M1)", "`main.py`"],
        ["$\\beta$", "0 (waning scenario: 1)", "rate R $\\to$ S", "`--beta`"],
        ["$N$", "5000 (EoN check: 200)", "nodes", "`--n`"],
        ["$\\langle k\\rangle$", "7 (B and W attach $m = 3$ edges)", "target mean degree", "`--k`"],
        ["initial infected", "2 (threshold, EoN: 1)", "random nodes", "`--initial`"],
        ["replicates", "100 (waning: 20)", "graphs per configuration", "`--replicates`"],
        ["seed", "113", "fixes every random stream", "`--seed`"],
        ["$CV_w$", "0, 0.5, 1, 2 (RQ1); 1 ($\\alpha$); 2 (immunisation)", "iid gamma", "`--cv`"],
        ["$\\theta$", "-0.5, 0, 0.5, 1", "$w_{ij} \\propto (k_i k_j)^\\theta$", "`--theta`"],
        ["$\\alpha$", "1 (sweep: 0, 0.5, 1, 2)", "$\\tau_{ij} = c\\, w_{ij}^\\alpha$", "`--alpha`"],
        ["reinforcement", "1", "BBV weight growth", "`--reinforcement`"],
        ["major outbreak", "attack rate $\\geq$ 5 %", "conditioning", "`metrics.py`"],
        ["bootstrap", "2000 resamples, 95 % percentile", "intervals", "`metrics.py`"],
        ["coverage", "5, 10, 20 % of nodes; 10 % of edges", "countermeasures", "`--coverage`"],
    ]
    return table("Parameters", ["Symbol", "Value", "Meaning", "Set by"], rows)


def baseline() -> str:
    runs = read_csv(Path("results/rq1/runs.csv"))
    rows = []
    for kind in dict.fromkeys(row["network"] for row in runs):
        m1 = [row for row in runs if (row["model"], row["network"]) == ("M1", kind)]
        major = [row for row in m1 if row["major"]]
        theory = [np.median([row[key] for row in m1]) for key in ("r0", "r0_newman", "r_analytic")]
        rows.append(
            [
                name(kind),
                f"{np.median([row['mean_degree'] for row in m1]):.2f}",
                *(f"{value:.2f}" for value in theory),
                f"{np.mean([row['major'] for row in m1]):.2f}",
                quartiles([row["peak_prevalence"] for row in major]),
                quartiles([row["peak_time"] for row in major], scale=1),
                quartiles([row["attack_rate"] for row in major]),
                quartiles([row["deaths"] for row in major]),
                quartiles([row["duration"] for row in major], scale=1),
            ]
        )
    header = ["Network", "$\\langle k\\rangle$", "NB $R_0$", "Newman $R_0$", "analytic $r$"]
    header += ["P(major)", "peak prevalence (%)", "peak time", "attack rate (%)", "deaths (%)"]
    caption = "M1 baseline (100 graphs; simulated metrics over major outbreaks: median (quartiles))"
    return table(caption, [*header, "duration"], rows)


def rq1() -> str:
    summary = summary_of("results/rq1")
    rows = []
    for kind in ("E", "C", "B"):
        for cv in (0.5, 1.0, 2.0):
            for protocol in ("i", "ii", "iii"):
                cell = (kind, "iid", cv, False, 1.0, protocol)
                entry = summary[(*cell, "attack_rate")]
                rows.append(
                    [
                        name(kind),
                        f"{cv:g}",
                        f"({protocol})",
                        f"{summary[(*cell, 'r0')]['m2_median']:.2f}",
                        f"{summary[(*cell, 'r_analytic')]['m2_median']:.2f}",
                        f"{entry['major_m1']:.2f}/{entry['major_m2']:.2f}",
                        f"{entry['pairs']}",
                        *(difference(summary[(*cell, metric)]) for metric in PLOTTED),
                    ]
                )
    header = ["Network", "$CV_w$", "protocol", "M2 $R_0$", "M2 $r$", "P(major) M1/M2", "pairs"]
    caption = "RQ1: M2 against M1, median paired difference in % [95 % bootstrap interval]"
    return table(caption, header + DIFFERENCES, rows)


def rq2(summary: dict, cells: list[tuple], labels: list[list[str]]) -> list[list[str]]:
    """Rows of an RQ2 table: labels, weights, outbreak probabilities, theory, differences."""
    rows = []
    for cell, label in zip(cells, labels, strict=True):
        entry = summary[(*cell, "attack_rate")]
        rows.append(
            [
                *label,
                f"{entry['cv_w']:.2f}",
                f"{entry['spearman']:.2f}",
                f"{entry['major_m1']:.2f}/{entry['major_m2']:.2f}",
                f"{100 * summary[(*cell, 'attack_rate_mp')]['difference']:.1f}",
                *(difference(summary[(*cell, metric)]) for metric in PLOTTED),
            ]
        )
    return rows


RQ2 = ["$CV_w$", "$\\rho$", "P(major) M1/M2", "attack rate, theory", *DIFFERENCES]


def theta() -> str:
    cells, labels = [], []
    for kind in ("E", "B"):
        for level in (-0.5, 0.5, 1.0):
            for shuffled in (False, True):
                for protocol in ("ii", "iii"):
                    cells.append((kind, "degree", level, shuffled, 1.0, protocol))
                    null = "shuffled" if shuffled else "-"
                    labels.append([name(kind), f"{level:g}", null, f"({protocol})"])
    rows = rq2(summary_of("results/theta"), cells, labels)
    caption = "RQ2, $w_{ij} \\propto (k_i k_j)^\\theta$: M2 against M1 in %"
    return table(caption, ["Network", "$\\theta$", "null", "protocol", *RQ2], rows)


def alpha() -> str:
    cells, labels = [], []
    for kind in ("E", "B"):
        for level in (0.5, 1.0, 2.0):
            for protocol in ("ii", "iii"):
                cells.append((kind, "iid", 1.0, False, level, protocol))
                labels.append([name(kind), f"{level:g}", f"({protocol})"])
    rows = rq2(summary_of("results/alpha"), cells, labels)
    caption = (
        "RQ2, $\\tau_{ij} = c\\, w_{ij}^\\alpha$ with iid weights ($CV_w = 1$): M2 against M1 in %"
    )
    return table(caption, ["Network", "$\\alpha$", "protocol", *RQ2], rows)


def bbv() -> str:
    summary = summary_of("results/bbv")
    cells = [("W", "bbv", 1.0, null, 1.0, p) for null in (False, True) for p in ("ii", "iii")]
    labels = [
        ["shuffled" if null else "own", f"({p})"] for null in (False, True) for p in ("ii", "iii")
    ]
    m1_r0 = summary[(*cells[0], "r0")]["m1_median"]
    caption = f"RQ2, BBV weights (reinforcement 1; M1 $R_0$ = {m1_r0:.2f}): M2 against M1 in %"
    return table(caption, ["Weights", "protocol", *RQ2], rq2(summary, cells, labels))


def immunisation() -> str:
    summary = read_csv(Path("results/countermeasures/summary.csv"))
    by = {(e["network"], e["model"], e["strategy"], e["coverage"]): e for e in summary}
    rows, rankings, edges = [], [], []
    for kind in ("E", "B"):
        for coverage in (0.05, 0.1, 0.2):
            for strategy in countermeasures.STRATEGIES:
                m1, m2 = (by[kind, model, strategy, coverage] for model in ("M1", "M2"))
                rows.append(
                    [
                        name(kind),
                        f"{100 * coverage:g} %",
                        strategy,
                        f"{m1['major']:.2f}/{m2['major']:.2f}",
                        *(
                            reduction(e, m)
                            for m in ("attack_rate", "peak_prevalence")
                            for e in (m1, m2)
                        ),
                        f"{m1['rank']}/{m2['rank']}",
                    ]
                )
            cells = [by[kind, model, "random", coverage] for model in ("M1", "M2")]
            orders = [countermeasures.ranking(countermeasures.rivals_of(summary, e)) for e in cells]
            rankings.append([name(kind), f"{100 * coverage:g} %", *orders])
        for measure in countermeasures.EDGE_MEASURES:
            e = by[kind, "M2", measure, 0.1]
            changes = (reduction(e, m) for m in ("attack_rate", "peak_prevalence"))
            edges.append([name(kind), measure, f"{e['major']:.2f}", *changes])
    header = ["Network", "coverage", "strategy", "P(major) M1/M2", "attack rate M1", "M2"]
    caption = "Node immunisation: reduction in expected attack rate and peak, % [95 % interval]"
    edge_header = ["Network", "measure", "P(major)", "attack rate", "peak"]
    return "\n".join(
        [
            table(caption, [*header, "peak M1", "M2", "rank M1/M2"], rows),
            table(
                "Ranking by attack-rate reduction", ["Network", "coverage", "M1", "M2"], rankings
            ),
            table("Edge measures on M2 (10 % of edges): reduction, %", edge_header, edges),
        ]
    )


def agreement(folder: Path) -> str:
    rows = []
    for check, quantity in (("growth", "growth rate"), ("final-size", "attack rate")):
        entries = read_csv(folder / f"{check}.csv")
        for kind in dict.fromkeys(e["network"] for e in entries):
            [m1] = [e for e in entries if (e["network"], e["model"]) == (kind, "M1")]
            m2 = [100 * e["gap"] for e in entries if (e["network"], e["model"]) == (kind, "M2")]
            rows.append(
                [
                    name(kind),
                    quantity,
                    f"{100 * m1['gap']:.1f}",
                    f"{min(m2):.1f} to {max(m2):.1f}",
                    f"{100 * m1['gap_q1']:.1f}, {100 * m1['gap_q3']:.1f}",
                ]
            )
    header = ["Network", "quantity", "M1", "M2 cells", "M1 per-run quartiles"]
    return table("Simulation against theory: median gap in %", header, rows)


def validation() -> str:
    folder = Path("results/validation")
    gaps = read_csv(folder / "r0-gap.csv")
    rows = []
    for kind in dict.fromkeys(e["network"] for e in gaps):
        entries = [e for e in gaps if e["network"] == kind]
        values = (f"{100 * e['gap']:.1f}" for e in entries)
        rows.append([name(kind), f"{entries[0]['assortativity']:.3f}", *values])
    header = ["Network", "assortativity", *(f"$CV_w$ = {e['cv']:g}" for e in gaps[:4])]
    newman = table("Non-backtracking against Newman's $R_0$ (iid weights), %", header, rows)

    rows = [
        [
            f"{e['cv']:g}",
            f"{e['mean_t']:.3f}",
            f"{e['r0']:.3f}",
            f"{100 * e['mean_t_change']:.1f}",
            f"{100 * e['r0_change']:.1f}",
        ]
        for e in read_csv(folder / "jensen.csv")
        if e["cv"] in (0.0, 0.5, 1.0, 2.0, 3.0)
    ]
    header = ["$CV_w$", "$\\langle T\\rangle$", "NB $R_0$", "$\\langle T\\rangle$ (%)", "$R_0$ (%)"]
    jensen = table("Protocol (i) on C: mean transmissibility and $R_0$", header, rows)

    points = read_csv(folder / "threshold.csv")
    levels = (0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0)
    rows = []
    for kind, model in dict.fromkeys((e["network"], e["model"]) for e in points):
        at = {e["level"]: e for e in points if (e["network"], e["model"]) == (kind, model)}
        label = "M1" if model == "M1" else "M2 ($\\theta$ = 0.5)"
        majors = (f"{at[level]['major']:.3f}" for level in levels)
        rows.append([name(kind), label, f"{at[1.0]['r0_newman']:.2f}", *majors])
    header = ["Network", "model", "Newman $R_0$ at NB 1", *(f"NB {v:g}" for v in levels)]
    threshold = table(
        "Threshold: P(major outbreak) from one case against the NB $R_0$", header, rows
    )
    rows = []
    for kind, model in dict.fromkeys((e["network"], e["model"]) for e in points):
        at = {e["level"]: e for e in points if (e["network"], e["model"]) == (kind, model)}
        label = "M1" if model == "M1" else "M2 ($\\theta$ = 0.5)"
        sizes = (f"{100 * at[level]['attack_rate_mp']:.1f}" for level in levels)
        rows.append([name(kind), label, *(size.replace("nan", "n/a") for size in sizes)])
    header = ["Network", "model", *(f"NB {v:g}" for v in levels)]
    caption = "Threshold: message-passing final size, % of nodes (n/a: not computed at 1)"
    threshold += "\n" + table(caption, header, rows)

    rows = [
        [
            name(e["network"]),
            e["model"],
            f"{e['attack_rate']:.4f}",
            f"{e['attack_rate_eon']:.4f}",
            f"{e['z']:.2f}",
            f"{e['major']:.3f}",
            f"{e['major_eon']:.3f}",
            f"{e['ks_p']:.2f}",
        ]
        for e in read_csv(folder / "eon.csv")
    ]
    header = ["Network", "model", "mean attack rate", "EoN", "$z$", "P(major)", "EoN", "KS $p$"]
    eon = table("Against EoN's `fast_SIR` ($N = 200$)", header, rows)
    return "\n".join([agreement(folder), newman, jensen, threshold, eon])


def tables(path: Path = Path("results/tables.md")) -> None:
    sections = [parameters(), baseline(), rq1(), theta(), alpha(), bbv(), immunisation()]
    intro = "# Tables\n\nWritten by `python run_all.py tables` from the CSVs in `results/`.\n\n"
    path.write_text(intro + "\n".join([*sections, validation()]))
    print(f"-> {path}")


def run(selected: list[str]) -> None:
    started = time.perf_counter()
    for step in selected or [*STEPS, "tables"]:
        print(f"\n=== {step}", flush=True)
        step_started = time.perf_counter()
        if step == "tables":
            tables()
        else:
            function, arguments = STEPS[step]
            function(arguments.split())
        print(f"=== {step}: {time.perf_counter() - step_started:.0f} s", flush=True)
    total = time.perf_counter() - started
    print(f"\ntotal: {total:.0f} s ({total / 60:.1f} min)")


if __name__ == "__main__":
    run(sys.argv[1:])
