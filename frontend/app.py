import base64
import io

import numpy as np
import streamlit as st
from episim.networks import create_network
from episim.calibration import set_rates, match_protocol, nonbacktracking_pattern, unit_radius
from episim.networks import iid_weights
from episim.simulation import simulate, states_at
from episim.plotting import layout, draw_states, COLOURS, STATE_NAMES
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from animation import build_animation_html

MEAN_DEGREE = 7
SEED = 113
MAX_FRAMES = 120
MIN_FRAMES = 30
FRAME_BUDGET = 300_000
FRAME_MS = 100
FRAME_DPI = 90
ANIMATION_HEIGHT = 7 * FRAME_DPI + 90
BG_COLOUR = "lightGray"

st.title("Epidemic Simulation")
st.write("Authors: Teodor Cioata, Šarlote Luīze Odziņa")

#These are the options on the sidebar
model_choice = st.sidebar.radio("Model", ("Unweighted (M1)", "Weighted (M2)"))
topology_choice = st.sidebar.radio("Topology", ("Erdős–Rényi", "Barabási–Albert", "Configuration model"))
population_size = st.sidebar.slider("Population Size", min_value=2, max_value=5000, value=20, step=1)
initially_infectious_people = st.sidebar.slider("Initially Infectious People", min_value=1, max_value=population_size, value=1, step=1)

st.sidebar.header("Disease parameters")
tau   = st.sidebar.slider("S->E", 0.1, 1.0, 1.0, 0.1)
sigma = st.sidebar.slider("E->I", 0.1, 1.0, 1.0, 0.1)
gamma = st.sidebar.slider("I->R", 0.1, 1.0, 1.0, 0.1)
delta = st.sidebar.slider("I->D", 0.0, 1.0, 0.1, 0.01)
beta  = st.sidebar.slider("R->S", 0.0, 1.0, 0.0, 0.1)
if beta > 0:
    t_max = st.sidebar.slider("Time cap", 10.0, 200.0, 50.0, 10.0)
else:
    t_max = 100.0
if model_choice == "Weighted (M2)":
    protocol = st.sidebar.radio("Match M2 to M1 in", ("i", "ii", "iii"), index=1)

if model_choice == "Weighted (M2)":
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


#Here the actual animation is built
def build_animation(graph, pos, initial_states, events, max_time):
    #Bigger populations get fewer frames otherwise it will take forever to draw them
    wanted = int(np.clip(FRAME_BUDGET // max(len(graph), 1), MIN_FRAMES, MAX_FRAMES))
    n_frames = max(2, min(wanted, round(max_time / 0.1) + 1))
    times = np.linspace(0.0, max_time, n_frames)
    all_states = states_at(initial_states, events, times)
    frames = [
        figure_to_data_uri(render_frame(graph, pos, states, float(t)))
        for t, states in zip(times, all_states)
    ]
    return build_animation_html(frames, [round(float(t), 1) for t in times], FRAME_MS)


def show_animation(html):
    if hasattr(st, "iframe"):
        st.iframe(html, height=ANIMATION_HEIGHT)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=ANIMATION_HEIGHT)


#Here define what happens when the 'Run simulation' button is pressed
if st.sidebar.button("Run simulation!"):
    status = st.sidebar.empty()
    status.markdown(":red[Simulation loading...]")
    rng = np.random.default_rng(SEED)
    disease = dict(sigma=sigma, gamma=gamma, delta=delta)
    graph = create_network(kind=topology[topology_choice], n=population_size, mean_degree=MEAN_DEGREE, rng=rng)

    if model_choice == "Unweighted (M1)":
        set_rates(graph, tau)
    else:
        weighted = iid_weights(graph, cv, rng)
        pattern = nonbacktracking_pattern(graph)
        rho_unit = unit_radius(graph, pattern)
        c = match_protocol(protocol, weighted, 1.0, tau, rho_unit, **disease, pattern=pattern)
        set_rates(weighted, c, 1.0)
        graph = weighted
    initial_infected = rng.choice(len(graph), size=initially_infectious_people, replace=False).tolist()
    initial_states, events = simulate(graph, initial_infected, rng, **disease, beta=beta, t_max=t_max)
    max_time = max(events[-1][0] if events else 0.0, 0.1)
    pos = layout(graph, seed=SEED)
    st.session_state["curves_png"] = progression_png(initial_states, events, max_time)
    st.session_state["animation_html"] = build_animation(graph, pos, initial_states, events, max_time)
    status.markdown(":green[Simulation completed!]")


#Show the animation and the plot
if "animation_html" in st.session_state:
    st.subheader("Epidemic progression")
    show_animation(st.session_state["animation_html"])
    st.subheader("Population progression over time")
    st.image(st.session_state["curves_png"])