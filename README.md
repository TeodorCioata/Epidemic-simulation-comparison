# Epidemic-simulation-comparison

Comparing epidemic model classes under matched assumptions: an **unweighted** network SEIR+D model with
waning immunity (**M1**) against a **weighted** one (**M2**), run on the same graph, the same initial
infected and the same random stream, so that any difference comes from contact-strength heterogeneity.

- **M1**: every edge transmits at rate `tau`.
- **M2**: edge `(i, j)` transmits at `tau_M (w_ij / w_M)^alpha`, calibrated so that M2 has M1's `R_0`.

M1 is the zero-dispersion limit of M2: with constant weights or `alpha = 0` the two produce identical
event logs. Research plan: [docs/epidemic_simulation.pdf](docs/epidemic_simulation.pdf). Decisions and
findings: [docs/DESIGN.md](docs/DESIGN.md).

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

numpy is pinned below 2: numpy 2 wheels need a CPU feature (POPCNT) the development VM lacks.

## Run

```bash
python main.py                     # Sarlote's unweighted model: E, B, C; final state plot
python compare_models.py           # M1 vs M2 on the same networks, with snapshots in snapshots/
python compare_models.py --networks EBCHW --cv 2 --alpha 1.5
python compare_models.py --n 10000 --networks E      # large n: summary only, no drawings

pytest                             # full suite, ~15 s
pytest -m "not slow"
ruff check . && ruff format --check .
mypy
```

`compare_models.py` uses main.py's parameters (`sigma, gamma, tau, delta, beta`), 100 nodes, mean degree 7,
seed 113 and 2 initial infections. Per network it builds the graph, gives M2 iid gamma weights (or BBV's
own weights on W), matches M2's non-backtracking `R_0` to M1's, simulates both, prints one summary line
and saves PNGs at t = 0, 1, 2, 4, 8 and the final time as `snapshots/<network>/<model>_<i>_t<time>.png`.

Network kinds: **E** Erdős-Rényi, **B** Barabási-Albert, **C** configuration model with Poisson degrees
(these three from `main.create_network`), **H** configuration model on BA's degree sequence, **W** BBV
weighted growth.

## Layout

| Path | Owner | Contents |
|---|---|---|
| `main.py` | Sarlote | unweighted model: networks, SEIR+D with waning, final-state plot |
| `episim/networks.py` | Teo | topologies, weight schemes, `CV_w`, weight-degree correlation, strength exponent |
| `episim/calibration.py` | Teo | rates, `T`, Newman and non-backtracking `R_0`, protocols (i) and (ii) |
| `episim/simulation.py` | Teo | event-driven SEIR+D with per-edge rates; event log and replay |
| `episim/plotting.py` | Teo | snapshots in main.py's style, one fixed layout per graph |
| `compare_models.py` | Teo | the M1 vs M2 run, mirroring `main.simulate_epidemic()` |
| `tests/` | Teo | behaviour tests, including a statistical cross-check against `main.py` |

Authors: Teodor Cioata, Sarlote Odzina. MIT licensed.
