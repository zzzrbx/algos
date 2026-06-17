"""Mirollo-Strogatz pulse-coupled oscillators on a random geometric graph -- Manim.

Model
-----
Each node is an oscillator with a phase that climbs from 0 to 1 at a constant rate
(its clock hand sweeping toward "twelve o'clock"). When a node's phase reaches 1 it
*fires* (spikes) and resets to 0; every neighbour it is connected to jumps ahead by
a fixed amount ``EPSILON``. A kick that pushes a neighbour to >= 1 makes it fire too,
so spikes can cascade. This excitatory pulse-coupling drives the network toward
synchrony: clusters of nodes lock together until eventually all of them flash in
unison (the classic firefly-synchronisation result, Mirollo & Strogatz 1990).

Phase is shown as node brightness (dim when just reset, bright near firing); a node
flashes white when it spikes. A synchrony readout (the Kuramoto order parameter of
the phases) climbs toward 1 as the network locks.

The network is a sparse, connected random geometric graph (same construction as the
colour-copying script).

Render
------
    uv run python scripts/run_script.py scripts/pulse_coupled_oscillators.py --quality high
"""

# Standard library
import os

# Third-party
import networkx as nx
import numpy as np
from manim import (
    GREY,
    UP,
    WHITE,
    Circle,
    Dot,
    Line,
    ManimColor,
    Scene,
    ValueTracker,
    VGroup,
    config,
    interpolate_color,
    linear,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N = int(os.environ.get("PCO_N", 5000))   # number of oscillators (override via PCO_N)
K_NEIGHBORS = 6          # each node connects to its k nearest neighbours (sparse graph)
DRAW_EDGES = N <= 2000   # edges become invisible clutter (and too slow) for huge N
EPSILON = 0.06           # voltage kick a neighbour receives when a node fires
DISSIPATION_B = 3.0      # concavity of the Mirollo-Strogatz firing map (>0). The
                         # concavity makes the phase advance bigger when a neighbour
                         # is early in its cycle => gradual synchronisation.
DT = 0.02                # phase advance per simulation step (in units of the period)
STEPS_PER_PERIOD = int(round(1.0 / DT))
# Total simulation steps = total animation frames. Pick large enough that the
# network converges and then blinks in unison for a good while at the end.
# (At 60 fps: 2000 frames = 33 s; 5k fully locks by ~26 s, leaving a short tail.)
TOTAL_FRAMES = 2000
SEED = 3

# Wide (16:9) layout so the network fills a YouTube-style frame.
ASPECT = 16 / 9

CACHE_PATH = os.path.join(os.path.dirname(__file__), f"pulse_coupled_cache_{N}.npz")

# Phase -> colour: faint yellow that brightens a little near firing; the prominent
# event is the WHITE flash on a spike (kept subtle so the white pops).
DIM_COLOR = ManimColor("#3d2e09")   # inactive (just-reset) nodes are faint yellow
HOT_COLOR = ManimColor("#9c7a1e")   # muted yellow near firing
FLASH_DECAY = 0.82       # how fast a spike flash fades (per frame; higher = longer)


def phase_color(phase, flash):
    base = interpolate_color(DIM_COLOR, HOT_COLOR, float(np.clip(phase, 0, 1)))
    if flash > 0:
        return interpolate_color(base, WHITE, float(np.clip(flash, 0, 1)))
    return base


def precompute_rgb(phases, flash):
    """Vectorised (frames, N, 3) RGB for every node and frame.

    Nodes are visibly yellow (dim when just reset, brighter as they approach
    firing) and flash white at the spike instant.
    """
    dim = np.array(DIM_COLOR.to_rgb())     # dim yellow
    hot = np.array(HOT_COLOR.to_rgb())     # bright yellow
    white = np.array([1.0, 1.0, 1.0])
    p = np.clip(phases, 0.0, 1.0)[..., None]
    base = dim * (1 - p) + hot * p         # phase -> yellow brightness
    f = np.clip(flash, 0.0, 1.0)[..., None]
    rgb = base * (1 - f) + white * f       # spike -> white
    return rgb.astype(np.float32)


def build_graph(seed=SEED):
    """Sparse, connected k-nearest-neighbour spatial graph: (positions, graph).

    Each node links to its ``K_NEIGHBORS`` nearest neighbours (low, uniform
    degree), built with a KDTree so it scales to many thousands of nodes. If the
    graph is not connected, k is increased until it is.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    pos = rng.random((N, 2))
    pos[:, 0] *= ASPECT  # spread nodes across a wide (16:9) rectangle
    tree = cKDTree(pos)

    k = K_NEIGHBORS
    while True:
        _, idx = tree.query(pos, k=k + 1)  # first column is the node itself
        graph = nx.Graph()
        graph.add_nodes_from(range(N))
        for i in range(N):
            for j in idx[i, 1:]:
                graph.add_edge(int(i), int(j))
        if nx.is_connected(graph):
            return pos, graph
        k += 1  # rare: bump k until the kNN graph is connected


def simulate(graph, seed=SEED):
    """Run the pulse-coupled dynamics step by step.

    Returns:
      phases:  (steps, N) phase of every node at each frame.
      fired:   (steps, N) bool, which nodes spiked on that frame.
    """
    rng = np.random.default_rng(seed + 1)
    neighbors = {v: list(graph.neighbors(v)) for v in range(N)}
    phases = rng.random(N)

    # Mirollo-Strogatz concave firing map f and its inverse. A neighbour's voltage
    # x = f(phase) is bumped by EPSILON, then mapped back to a phase. Concavity
    # (DISSIPATION_B > 0) is what drives gradual synchronisation.
    expb1 = np.exp(DISSIPATION_B) - 1.0

    def kick_phase(p):
        x = np.log1p(expb1 * p) / DISSIPATION_B   # f(p)
        x = min(x + EPSILON, 1.0)
        return (np.exp(DISSIPATION_B * x) - 1.0) / expb1  # f^{-1}(x)

    phase_log, fired_log = [phases.copy()], [np.zeros(N, dtype=bool)]
    # Run a fixed number of steps (= animation frames). The network converges
    # partway through and then keeps blinking in unison until the end.
    for _step in range(TOTAL_FRAMES):
        phases += DT
        fired = np.zeros(N, dtype=bool)
        # Resolve firings and any cascades triggered by the kicks.
        for _ in range(N + 1):
            crossers = np.where((phases >= 1.0) & (~fired))[0]
            if crossers.size == 0:
                break
            for f in crossers:
                if fired[f]:
                    continue
                fired[f] = True
                phases[f] = 0.0
                for j in neighbors[f]:
                    if not fired[j]:
                        phases[j] = kick_phase(phases[j])
        phase_log.append(phases.copy())
        fired_log.append(fired.copy())

    return np.array(phase_log, dtype=np.float32), np.array(fired_log, dtype=bool)


def build_flash(fired):
    """Turn instantaneous spikes into a few frames of decaying white flash."""
    steps, n = fired.shape
    flash = np.zeros((steps, n), dtype=np.float32)
    level = np.zeros(n, dtype=np.float32)
    for t in range(steps):
        level = np.maximum(level * FLASH_DECAY, fired[t].astype(np.float32))
        flash[t] = level
    return flash


def coherence(phases):
    """Kuramoto order parameter of the phases at each frame (synchrony in [0, 1])."""
    z = np.exp(2j * np.pi * phases)
    return np.abs(z.mean(axis=1))


def load_or_simulate():
    sig = (f"N={N};knn={K_NEIGHBORS};eps={EPSILON};b={DISSIPATION_B};"
           f"dt={DT};total={TOTAL_FRAMES};decay={FLASH_DECAY};seed={SEED}")
    if os.path.exists(CACHE_PATH):
        data = np.load(CACHE_PATH)
        if "signature" in data and str(data["signature"]) == sig:
            return (data["pos"], data["edges"], data["phases"],
                    data["flash"], data["coh"])

    pos, graph = build_graph()
    phases, fired = simulate(graph)
    flash = build_flash(fired)
    coh = coherence(phases)
    edges = np.array(graph.edges())
    np.savez_compressed(CACHE_PATH, pos=pos, edges=edges, phases=phases,
                        flash=flash, coh=coh, signature=sig)
    return pos, edges, phases, flash, coh


LAYOUT_CENTER = np.array([0.0, 0.0, 0.0])
LAYOUT_SCALE = 7.0       # isotropic scale; x spans ASPECT, y spans 1 (fills the frame)


def to_point(xy):
    return LAYOUT_CENTER + LAYOUT_SCALE * np.array(
        [xy[0] - ASPECT / 2, xy[1] - 0.5, 0.0]
    )


class PulseCoupledOscillators(Scene):
    def construct(self):
        pos, edges, phases, flash, coh = load_or_simulate()
        points = np.array([to_point(p) for p in pos])
        n_frames = len(phases) - 1

        edge_lines = VGroup(*[
            Line(points[i], points[j], stroke_width=0.4,
                 stroke_color=GREY, stroke_opacity=0.12)
            for i, j in edges
        ]) if DRAW_EDGES else VGroup()
        rgb = precompute_rgb(phases, flash)  # (frames, N, 3)
        node_radius = 0.022 if N <= 2000 else 0.02
        node_dots = VGroup(*[
            Dot(points[i], radius=node_radius, fill_opacity=1.0)
            for i in range(N)
        ])

        tracker = ValueTracker(0)

        def idx():
            return max(0, min(int(round(tracker.get_value())), n_frames))

        def update_nodes(group):
            frame_rgb = rgb[idx()]
            for k, dot in enumerate(group.submobjects):
                dot.fill_rgbas[:, :3] = frame_rgb[k]

        node_dots.add_updater(update_nodes)

        self.add(edge_lines, node_dots)
        # One simulation step per frame; the last EXTRA_FRAMES (~5 s at 60 fps) are
        # the synchronised blinking after the network locks.
        self.play(
            tracker.animate.set_value(n_frames),
            run_time=n_frames / config.frame_rate,
            rate_func=linear,
        )
        self.wait(0.5)
