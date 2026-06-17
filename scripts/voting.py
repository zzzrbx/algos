"""Local color-copying (voter model) on a random geometric graph -- Manim only.

Setup
-----
* ``N`` nodes are placed at random locations in the unit square.
* Edges connect nodes within a small radius (a *random geometric graph*). The
  radius is the smallest value that keeps the graph connected, so the graph is
  sparse but connected.
* Every node starts with a distinct value 1..N, mapped through a continuous
  colour gradient (so values look like a smooth field you can watch spread).

Dynamics
--------
Nodes wake one at a time, in random order. A waking node copies the colour of one
randomly chosen neighbour. This is the voter model: colours coalesce until every
node shares the same colour (consensus), at which point the simulation stops.

Every wake-up is shown as one animation frame (no downsampling): run_time is set
to events / frame_rate, so the whole run plays back at one wake-up per frame.

    uv run python scripts/run_script.py scripts/color_copying_geometric_graph.py \
        --quality high
"""

# Standard library
import os
from collections import Counter

# Third-party
import networkx as nx
import numpy as np
from manim import (
    BLUE,
    GREEN,
    GREY,
    ORANGE,
    RED,
    TEAL,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Circle,
    Dot,
    Line,
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
N = 100                  # number of nodes
RADIUS_SLACK = 1.08      # how far above the connectivity threshold to set the radius
MAX_EVENTS = 2_000_000   # safety cap on wake-ups (consensus normally stops earlier)
SEED = 7

# Wide (16:9) layout so the network fills a YouTube-style frame.
ASPECT = 16 / 9

CACHE_PATH = os.path.join(os.path.dirname(__file__), "color_copying_cache.npz")

# Continuous, non-cyclic colour gradient (value 0..1 -> colour).
GRADIENT = [BLUE, TEAL, GREEN, YELLOW, ORANGE, RED]

# Wide rectangular layout inside the Manim frame.
LAYOUT_CENTER = np.array([0.0, 0.0, 0.0])
LAYOUT_SCALE = 7.0       # isotropic scale; x spans ASPECT, y spans 1 (fills the frame)


def value_to_color(v):
    """Map a scalar value in [0, 1] to a colour along the continuous gradient."""
    v = float(np.clip(v, 0.0, 1.0))
    x = v * (len(GRADIENT) - 1)
    i = min(int(np.floor(x)), len(GRADIENT) - 2)
    return interpolate_color(GRADIENT[i], GRADIENT[i + 1], x - i)


def build_graph(seed=SEED):
    """Return (positions, graph) for a sparse, connected random geometric graph."""
    rng = np.random.default_rng(seed)
    pos = rng.random((N, 2))
    pos[:, 0] *= ASPECT  # spread nodes across a wide (16:9) rectangle

    # Minimum radius that connects every node = the largest edge of the Euclidean
    # minimum spanning tree. A small slack above it keeps the graph sparse.
    complete = nx.Graph()
    for i in range(N):
        for j in range(i + 1, N):
            complete.add_edge(i, j, weight=float(np.hypot(*(pos[i] - pos[j]))))
    mst = nx.minimum_spanning_tree(complete)
    r_connect = max(d["weight"] for _, _, d in mst.edges(data=True))
    radius = r_connect * RADIUS_SLACK

    graph = nx.random_geometric_graph(
        N, radius, pos={i: tuple(pos[i]) for i in range(N)}
    )
    assert nx.is_connected(graph), "graph is not connected"
    return pos, graph


def simulate_to_consensus(graph, seed=SEED):
    """Run wake-ups until consensus, recording every intermediate state.

    Returns (one entry per recorded colour change; index 0 = initial state):
      history:  (n_events + 1, N) colour value of every node after each change.
      distinct: (n_events + 1,)   number of distinct colours after each change.
      wakers:   (n_events,)       node that woke and changed colour at each step.
      sources:  (n_events,)       neighbour it copied that colour from.
    """
    rng = np.random.default_rng(seed + 1)
    neighbors = {v: list(graph.neighbors(v)) for v in range(N)}

    colors = np.arange(N, dtype=float) / (N - 1)
    counts = Counter(colors.tolist())
    distinct = len(counts)

    history = [colors.copy()]
    distinct_log = [distinct]
    wakers, sources = [], []
    attempts = 0
    while distinct > 1 and attempts < MAX_EVENTS:
        attempts += 1
        v = int(rng.integers(N))
        nbrs = neighbors[v]
        u = int(nbrs[rng.integers(len(nbrs))])
        old, new = colors[v], colors[u]
        if old == new:
            continue  # node copied a neighbour of its own colour -> no frame
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
            distinct -= 1
        counts[new] += 1
        colors[v] = new
        history.append(colors.copy())
        distinct_log.append(distinct)
        wakers.append(v)
        sources.append(u)
    return (np.array(history, dtype=np.float32), np.array(distinct_log, dtype=int),
            np.array(wakers, dtype=int), np.array(sources, dtype=int))


def load_or_simulate():
    """Build the graph + simulate to consensus, with caching."""
    sig = f"N={N};slack={RADIUS_SLACK};seed={SEED}"
    if os.path.exists(CACHE_PATH):
        data = np.load(CACHE_PATH)
        if "signature" in data and str(data["signature"]) == sig:
            return (data["pos"], data["edges"], data["history"],
                    data["distinct"], data["wakers"], data["sources"])

    pos, graph = build_graph()
    history, distinct, wakers, sources = simulate_to_consensus(graph)
    edges = np.array(graph.edges())
    np.savez_compressed(
        CACHE_PATH, pos=pos, edges=edges, history=history,
        distinct=distinct, wakers=wakers, sources=sources, signature=sig,
    )
    return pos, edges, history, distinct, wakers, sources


def to_point(xy):
    """Map a unit-square coordinate to a Manim scene point."""
    return LAYOUT_CENTER + LAYOUT_SCALE * np.array(
        [xy[0] - ASPECT / 2, xy[1] - 0.5, 0.0]
    )


class ColorCopyingGraph(Scene):
    def construct(self):
        pos, edges, history, distinct, wakers, sources = load_or_simulate()
        points = np.array([to_point(p) for p in pos])
        n_events = len(history) - 1

        edge_lines = VGroup(*[
            Line(points[i], points[j], stroke_width=0.8,
                 stroke_color=GREY, stroke_opacity=0.3)
            for i, j in edges
        ])

        node_dots = VGroup(*[
            Dot(points[i], radius=0.09, color=value_to_color(history[0, i]))
            for i in range(N)
        ])

        # Highlight of the current wake-up: a ring on the waking node and an arrow
        # FROM the neighbour it copied (source) TO the waker.
        wake_ring = Circle(radius=0.18, color=WHITE, stroke_width=4,
                           fill_opacity=0).move_to(points[0])
        source_ring = Circle(radius=0.16, color=YELLOW, stroke_width=4,
                             fill_opacity=0).move_to(points[0])
        copy_arrow = Arrow(points[0], points[1], buff=0.18, stroke_width=5,
                           color=WHITE, max_tip_length_to_length_ratio=0.35)
        wake_ring.set_stroke(opacity=0.0)
        source_ring.set_stroke(opacity=0.0)
        copy_arrow.set_opacity(0.0)

        event_tracker = ValueTracker(0)

        def current_event():
            return max(0, min(int(round(event_tracker.get_value())), n_events))

        def update_nodes(group):
            for dot, v in zip(group, history[current_event()]):
                dot.set_color(value_to_color(v))

        def update_highlight(_):
            idx = current_event()
            if idx >= 1:
                w, s = int(wakers[idx - 1]), int(sources[idx - 1])
                wake_ring.move_to(points[w]).set_stroke(opacity=1.0)
                source_ring.move_to(points[s]).set_stroke(opacity=1.0)
                copy_arrow.put_start_and_end_on(points[s], points[w]).set_opacity(1.0)
            else:
                wake_ring.set_stroke(opacity=0.0)
                source_ring.set_stroke(opacity=0.0)
                copy_arrow.set_opacity(0.0)

        node_dots.add_updater(update_nodes)
        wake_ring.add_updater(update_highlight)

        self.add(edge_lines, node_dots, copy_arrow, source_ring, wake_ring)
        # One wake-up per rendered frame: run_time = events / frame_rate.
        self.play(
            event_tracker.animate.set_value(n_events),
            run_time=n_events / config.frame_rate,
            rate_func=linear,
        )
        # Clear the final highlight so the last frame shows the clean consensus state.
        wake_ring.clear_updaters()
        wake_ring.set_stroke(opacity=0.0)
        source_ring.set_stroke(opacity=0.0)
        copy_arrow.set_opacity(0.0)
        self.wait(1.0)
