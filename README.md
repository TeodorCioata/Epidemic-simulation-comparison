# Epidemic-simulation-comparison

Comparing epidemic model classes under matched assumptions: an **unweighted** network SEIR+D model
(**M1**) against a **weighted** one (**M2**), run on the same graph, the same initial infected and the
same random streams, so that any difference comes from contact-strength heterogeneity.

- **M1**: every edge transmits at rate `tau`.
- **M2**: edge `(i, j)` transmits at `tau_M (w_ij / w_M)^alpha`, with its scale set by one of three
  matching protocols: (i) equal mean rate, (ii) equal `R_0`, (iii) equal early growth rate.

M1 is the zero-dispersion limit of M2: with constant weights or `alpha = 0` the two produce identical
event logs. Results for
the report, with every table in LaTeX: [docs/RESULTS.md](docs/RESULTS.md).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

numpy is pinned below 2: numpy 2 wheels need a CPU feature (POPCNT) the development VM lacks.

## Run

```bash
python run_all.py                  # every experiment, figure and table from scratch
python main.py                     # Sarlote's unweighted model: E, B, C in a live window (needs tkinter)
python compare_models.py           # M1 vs M2 on the same networks: summary, snapshots, curves
python compare_models.py --networks E --protocol iii --beta 0 --animate   # plus M1_vs_M2.gif
python run_experiments.py          # RQ1: CV_w x protocol, 100 replicates, results/rq1/
python run_experiments.py --networks EB --cv --theta -0.5 0 0.5 1 --shuffle --protocols ii iii
python countermeasures.py          # node immunisation in M1 and M2, results/countermeasures/
python validate.py                 # theory vs simulation, threshold, EoN: docs/figures/*.png|pdf

pytest                             # full suite, under a minute; pytest -m "not slow" is faster
ruff check . && ruff format --check .
mypy
```

Everything except `python main.py` runs without a display. Experiments use main.py's parameters
(`sigma = gamma = tau = 1`, `delta = 0.1`), no waning (`beta = 0`), 5000 nodes, mean degree 7, two
initial infections, seed 113 and 100 replicates per configuration. Streamlit frontend is in
`frontend/`.

Network kinds: **E** Erdős-Rényi, **B** Barabási-Albert, **C** configuration model with Poisson degrees
(these three from `main.create_network`), **H** configuration model on BA's degree sequence, **W** BBV
weighted growth.

## Layout

| Path | Owner | Contents |
|---|---|---|
| `main.py` | Sarlote | unweighted model: networks, SEIR+D with waning, live animation |
| `frontend/` | Sarlote | Streamlit app on top of `episim` |
| `episim/networks.py` | Teodor | topologies, weight schemes, shuffle null, `CV_w`, weight-degree correlation, strength exponent |
| `episim/calibration.py` | Teodor | rates, `T`, `R_0` (Newman, non-backtracking), growth rate, final size, protocols (i)–(iii) |
| `episim/simulation.py` | Teodor | event-driven SEIR+D with per-edge rates and immunised nodes; event log and replay |
| `episim/metrics.py` | Teodor | counts over time and per-run metrics from the event log; bootstrap interval |
| `episim/plotting.py` | Teodor | snapshots in main.py's style, epidemic curves, the M1-vs-M2 animation, chart helpers |
| `compare_models.py` | Teodor | one M1 vs M2 run per network, mirroring `main.simulate_epidemic()` |
| `run_experiments.py` | Teodor | paired experiment over weights × level × `alpha` × protocol (RQ1, RQ2) |
| `countermeasures.py` | Teodor | random, degree, strength and acquaintance immunisation; edge measures |
| `validate.py` | Teodor | growth rate, final size, `R_0`, Jensen, threshold and EoN checks |
| `run_all.py` | Teodor | every step in order, and `results/tables.md` (Markdown + LaTeX) |
| `tests/` | Teodor | behaviour tests, including cross-checks against `main.py` and EoN |

For the frontend, it is needed to install streamlit as follows:
```python -m pip install streamlit```

To run the frontend, run:

```python -m streamlit run frontend/app.py```

Authors: Teodor Cioata, Sarlote Odzina. MIT licensed.
