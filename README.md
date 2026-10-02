# Epidemic-simulation-comparison

Comparing epidemic model classes under matched assumptions: an **unweighted** network SEIR+D model
(**M1**) against a **weighted** one (**M2**), run on the same graph, the same initial infected and the
same random streams, so that any difference comes from contact-strength heterogeneity.

- **M1**: every edge transmits at rate `tau`.
- **M2**: edge `(i, j)` transmits at `tau_M (w_ij / w_M)^alpha`, with its scale set by one of three
  matching protocols: (i) equal mean rate, (ii) equal `R_0`, (iii) equal early growth rate.

M1 is the zero-dispersion limit of M2: with constant weights or `alpha = 0` the two produce identical
event logs. Research plan: [docs/epidemic_simulation.pdf](docs/epidemic_simulation.pdf).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

numpy is pinned below 2: numpy 2 wheels need a CPU feature (POPCNT) the development VM lacks.

## Run

```bash
python main.py                     # Sarlote's unweighted model: E, B, C in a live window (needs tkinter)
python compare_models.py           # M1 vs M2 on the same networks: summary, snapshots, curves
python compare_models.py --networks E --protocol iii --beta 0 --animate   # plus M1_vs_M2.gif
python run_experiments.py          # the RQ1 pilot: about 10 minutes on 4 cores, results/pilot/
python validate.py                 # theory vs simulation: tables and docs/figures/*.png

pytest                             # full suite, about half a minute; pytest -m "not slow" is faster
ruff check . && ruff format --check .
mypy
```

Everything except `python main.py` runs without a display. `compare_models.py` uses main.py's
parameters (`sigma, gamma, tau, delta, beta`), 100 nodes, mean degree 7, seed 113 and 2 initial
infections. `run_experiments.py` uses 5000 nodes and no waning, and compares M2 with M1 in every cell of
`CV_w` in {0, 0.5, 1, 2} × protocol in {i, ii, iii}, with 60 replicates per network.

Network kinds: **E** Erdős-Rényi, **B** Barabási-Albert, **C** configuration model with Poisson degrees
(these three from `main.create_network`), **H** configuration model on BA's degree sequence, **W** BBV
weighted growth.

## Layout

| Path | Owner | Contents |
|---|---|---|
| `main.py` | Sarlote | unweighted model: networks, SEIR+D with waning, live animation |
| `episim/networks.py` | Teodor | topologies, weight schemes, `CV_w`, weight-degree correlation, strength exponent |
| `episim/calibration.py` | Teodor | rates, `T`, `R_0` (Newman, non-backtracking), growth rate, final size, protocols (i)–(iii) |
| `episim/simulation.py` | Teodor | event-driven SEIR+D with per-edge rates; event log and replay |
| `episim/metrics.py` | Teodor | counts over time and per-run metrics from the event log; bootstrap interval |
| `episim/plotting.py` | Teodor | snapshots in main.py's style, epidemic curves, the M1-vs-M2 animation |
| `compare_models.py` | Teodor | one M1 vs M2 run per network, mirroring `main.simulate_epidemic()` |
| `run_experiments.py` | Teodor | paired experiment over `CV_w` × protocol: CSV, summary, RQ1 figure |
| `validate.py` | Teodor | growth rate, final size and `R_0` against theory |
| `tests/` | Teodor | behaviour tests, including a statistical cross-check against `main.py` |

Authors: Teodor Cioata, Sarlote Odzina. MIT licensed.
