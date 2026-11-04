import base64
import io

import numpy as np
import streamlit as st
from episim.networks import create_network, iid_weights
from episim.calibration import set_rates, match_protocol, nonbacktracking_pattern, unit_radius
from episim.simulation import simulate, states_at
from episim.plotting import layout, draw_states, COLOURS, STATE_NAMES
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from animation import build_comparison_html
from countermeasures import immunised, edge_measure
from pathlib import Path
from episim.metrics import run_metrics

MEAN_DEGREE = 7
SEED = 113
MAX_FRAMES = 120
MIN_FRAMES = 30
FRAME_BUDGET = 300_000
FRAME_MS = 100
FRAME_DPI = 90
BG_COLOUR = "lightGray"

st.title("Epidemic Simulation")
st.write("Authors: Teodor Cioata, Šarlote Luīze Odziņa")

# Countermeasures apply to both models for vaccination; edge interventions are M2-only.
st.sidebar.header("Disease countermeasures")
category = st.sidebar.radio("Countermeasure category", ("No countermeasure", "Vaccination", "Contact reduction"))
vaccination_strategy = None
vaccination_coverage = 0.0
contact_strategy = None
edge_share = 0.0
if category == "Vaccination":
    vaccination_strategy = st.sidebar.selectbox("Vaccination strategy", ("random", "degree", "strength", "acquaintance"))
    vaccination_coverage = st.sidebar.slider("Vaccination coverage (%)", 0, 100, 20, 5) / 100
elif category == "Contact reduction":
    contact_strategy = st.sidebar.selectbox("Contact reduction strategy", ("halve strongest", "remove random"))
    edge_share = st.sidebar.slider("Contacts affected (%)", 0, 100, 10, 5) / 100
    st.sidebar.info("Contact reduction is applied to M2 only. M1 remains untreated as a reference.")

st.sidebar.header("Network settings")
topology_choice = st.sidebar.radio("Topology", ("Erdős–Rényi", "Barabási–Albert", "Configuration model"))
population_size = st.sidebar.slider("Population Size", 2, 500, 20, 1)
initially_infectious_people = st.sidebar.slider("Initially Infectious People", 1, population_size, 1, 1)

st.sidebar.header("Disease parameters")
tau = st.sidebar.slider("S->E", 0.1, 1.0, 1.0, 0.1)
sigma = st.sidebar.slider("E->I", 0.1, 1.0, 1.0, 0.1)
gamma = st.sidebar.slider("I->R", 0.1, 1.0, 1.0, 0.1)
delta = st.sidebar.slider("I->D", 0.0, 1.0, 0.1, 0.01)
beta = st.sidebar.slider("R->S", 0.0, 1.0, 0.0, 0.1)
t_max = st.sidebar.slider("Time cap", 10.0, 200.0, 50.0, 10.0) if beta > 0 else 100.0
protocol = st.sidebar.radio("Match M2 to M1 in", ("i", "ii", "iii"), index=1)
cv = st.sidebar.slider("CV", 0.0, 2.0, 0.5, 0.5)

#Everywhere else except in this file the network topologies are denoted by "E", "B", "C", so make it match
topology = {
    "Erdős–Rényi": "E",
    "Barabási–Albert": "B",
    "Configuration model": "C"
}

#Here the network plots
def render_frame(graph, pos, states, t):
    figure = Figure(figsize=(9, 7))
    ax = figure.subplots()
    draw_states(ax, graph, pos, states, title="Epidemic at t = " + str(round(t, 1)))

    legend_items = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=COLOURS[state],
            markeredgecolor="black",
            markersize=10,
            linestyle="None",
            label=STATE_NAMES[state].capitalize(),
        )
        for state in ["S", "E", "I", "R", "D"]
    ]
    ax.legend(handles=legend_items, loc="upper right", title="SEIRD States")
    return figure

#The figure must be turned into a JPEG so that it can be put into the animation page
def figure_to_data_uri(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="jpeg", dpi=FRAME_DPI, pil_kwargs={"quality": 80})
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded

#Here the curve plots
def render_progression(initial_states, events, max_time):
    times = np.linspace(0, max_time, 200)
    all_states = states_at(initial_states, events, times)
    figure = Figure(figsize=(10, 5))
    figure.patch.set_facecolor(BG_COLOUR)
    ax = figure.subplots()
    ax.set_facecolor(BG_COLOUR)

    for state in ["S", "E", "I", "R", "D"]:
        counts = [
            sum(s == state for s in states)
            for states in all_states
        ]
        ax.plot(
            times,
            counts,
            label=STATE_NAMES[state].capitalize(),
            color=COLOURS[state],
            linewidth=2,
        )
    ax.set_xlabel("Simulation time")
    ax.set_ylabel("Number of people")
    ax.set_title("SEIRD progression over time")
    ax.legend(edgecolor="gray")
    ax.grid(color="white", alpha=0.6)
    return figure


def progression_png(initial_states, events, max_time):
    figure = render_progression(initial_states, events, max_time)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100)
    return buffer.getvalue()


# Both networks are sampled at exactly the same times and controlled by one HTML player.
def build_comparison_animation(graphs, pos, runs, max_time):
    wanted = int(np.clip(FRAME_BUDGET // max(2 * len(graphs["M1"]), 1), MIN_FRAMES, MAX_FRAMES))
    n_frames = max(2, min(wanted, round(max_time / 0.1) + 1))
    times = np.linspace(0.0, max_time, n_frames)
    frames = {}
    for model in ("M1", "M2"):
        initial_states, events = runs[model]
        snapshots = states_at(initial_states, events, times)
        frames[model] = []
        for t, states in zip(times, snapshots):
            figure = render_frame(graphs[model], pos, states, float(t))
            frames[model].append(figure_to_data_uri(figure))
            figure.clear()
    return build_comparison_html(frames["M1"], frames["M2"], times.tolist(), FRAME_MS)


def show_animation(html):
    import streamlit.components.v1 as components
    components.html(html, height=300, scrolling=False)



def show_research_figures():
    st.header("Precalculated results")
    st.caption("These figures are precalculated results and they will not change when you change the simulation settings through the frontend")
    figures_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    figures = [
        ("Countermeasure effectiveness", "countermeasures.png"),
        ("Final epidemic size", "final_size.png"),
        ("Epidemic growth", "growth.png"),
        ("Model comparison experiments", "rq1.png"),
        ("Waning immunity experiments", "rq1_waning.png"),
        ("Epidemic threshold", "threshold.png"),
        ("Reproduction number differences", "r0_gap.png"),
        ("Weight degree correlation", "theta.png"),
        ("Transmission weight exponent", "alpha.png"),
        ("Analysis of Weighted Model M2", "jensen.png"),
    ]
    for title, filename in figures:
        path = figures_dir / filename
        if path.is_file():
            st.subheader(title)
            st.image(str(path), use_container_width=True)
    st.subheader("Calibration protocol comparisons")
    for protocol_name in ("i", "ii", "iii"):
        path = figures_dir / f"protocol_{protocol_name}" / "E" / "curves.png"
        if path.is_file():
            st.markdown(f"**Protocol {protocol_name}**")
            st.image(str(path), use_container_width=True)



if st.sidebar.button("Run simulation!"):
    status = st.sidebar.empty()
    status.markdown(":red[Simulation loading...]")
    disease = dict(sigma=sigma, gamma=gamma, delta=delta)
    base = create_network(topology[topology_choice], population_size,
                          MEAN_DEGREE, np.random.default_rng([SEED, 0]))
    m1 = base.copy()
    set_rates(m1, tau)
    m2 = iid_weights(base, cv, np.random.default_rng([SEED, 1]))
    pattern = nonbacktracking_pattern(base)
    rho_unit = unit_radius(base, pattern)
    c = match_protocol(protocol, m2, 1.0, tau, rho_unit, **disease, pattern=pattern)
    set_rates(m2, c, 1.0)
    graphs = {"M1": m1, "M2": m2}

    vaccinated = []
    if category == "Vaccination":
        count = round(vaccination_coverage * len(base))
        if vaccination_strategy == "acquaintance":
            reachable = {v for u in base for v in base.neighbors(u)}
            if count > len(reachable):
                st.error("Choose less vaccinated people!")
                st.stop()
        target_graph = m2 if vaccination_strategy == "strength" else base
        vaccinated = immunised(target_graph, vaccination_strategy, count,
                               np.random.default_rng([SEED, 2]))
    elif category == "Contact reduction":
        graphs["M2"] = edge_measure(m2, contact_strategy, edge_share,
                                    np.random.default_rng([SEED, 3]))

    eligible = sorted(set(base.nodes) - set(vaccinated))
    if initially_infectious_people > len(eligible):
        st.error("Reduce the initial infectious count or vaccination coverage.")
        st.stop()
    initial_infected = np.random.default_rng([SEED, 4]).choice(
        eligible, size=initially_infectious_people, replace=False
    ).tolist()
    runs = {
        model: simulate(graph, initial_infected, np.random.default_rng([SEED, 5]),
                        **disease, beta=beta, t_max=t_max,
                        recovered=vaccinated if category == "Vaccination" else [])
        for model, graph in graphs.items()
    }
    max_time = max(0.1, *(events[-1][0] if events else 0.0
                           for _, events in runs.values()))
    pos = layout(base, seed=SEED)
    st.session_state["m1_progression"] = progression_png(*runs["M1"], max_time)
    st.session_state["m2_progression"] = progression_png(*runs["M2"], max_time)
    st.session_state["model_metrics"] = {
        model: run_metrics(initial_states, events, t_max=t_max)
        for model, (initial_states, events) in runs.items()
    }
    st.session_state["comparison_html"] = build_comparison_animation(graphs, pos, runs, max_time)
    status.markdown(":green[Simulation was successful!]")


if "comparison_html" in st.session_state:
    st.subheader("Synchronized M1 and M2 animations")
    show_animation(st.session_state["comparison_html"])


    st.subheader("SEIRD progression")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**M1 Unweighted**")
        st.image(st.session_state["m1_progression"], use_container_width=True)
    with col2:
        st.markdown("**M2 Weighted**")
        st.image(st.session_state["m2_progression"], use_container_width=True)

show_research_figures()
