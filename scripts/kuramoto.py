"""Coupled oscillators (Kuramoto model) simulated with Brian2, animated with Manim.

This reproduces the Brian2 "Coupled oscillators" example
(https://brian2.readthedocs.io/en/stable/examples/coupled_oscillators.html)
but replaces the matplotlib animation with a Manim scene.

Brian2 records the full phase trajectory of every oscillator via a
``StateMonitor``; those phases are cached to disk and then replayed as a Manim
animation (dots on the unit circle plus the mean-field "order parameter" arrow).

Run the simulation + render the animation:

    uv run manim render scripts/coupled_oscillators_manim.py CoupledOscillators

The first run executes the Brian2 simulations and caches the results to
``scripts/coupled_oscillators_cache.npz``; later renders reuse the cache.
Delete the cache file to force a re-simulation.
"""

# Standard library
import colorsys
import os

# Third-party
import numpy as np
from manim import (
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Circle,
    Dot,
    ManimColor,
    Scene,
    ValueTracker,
    VGroup,
    linear,
)


def phase_color(phase):
    """Map an oscillator phase (radians) to a vivid cyclic color (manim color).

    Uses an HSV wheel at full saturation but slightly varied value so the ring
    reads as a glowing neon rainbow rather than flat primaries.
    """
    hue = (phase % (2 * np.pi)) / (2 * np.pi)
    value = 0.85 + 0.15 * np.cos(phase)  # gentle brightness shimmer around the ring
    r, g, b = colorsys.hsv_to_rgb(hue, 0.95, float(value))
    return ManimColor.from_rgb((r, g, b))

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
N = 250                       # number of oscillators
SIM_DURATION_MS = 10_000      # total simulated time (ms), dt = 1 ms
FRAME_STRIDE_MS = 20          # sample the recording every 20 ms (smoother motion)
K_VALUES = [0, 3]             # coupling strengths: uncoupled vs. K=3
RANDOM_SEED = 214040893

# Intrinsic-frequency distribution (rad/s). Each oscillator's natural frequency
# is drawn from a Gaussian N(OMEGA_MEAN, OMEGA_SPREAD), clipped to be >= 0.
# Set OMEGA_SPREAD = 0 to give every oscillator the *same* frequency, so only the
# initial phases differ -- with coupling they then collapse to a single point.
OMEGA_MEAN = 0.5
OMEGA_SPREAD = 0.0

CACHE_PATH = os.path.join(os.path.dirname(__file__), "coupled_oscillators_cache.npz")


def run_sim(K, random_seed=RANDOM_SEED):
    """Run one Kuramoto simulation in Brian2, return subsampled phases.

    Returns an array of shape ``(N, n_frames)`` of phases (radians).
    """
    # Imported lazily so that re-renders that hit the cache don't pay the
    # Brian2 import / code-generation cost.
    from brian2 import (
        NeuronGroup,
        StateMonitor,
        Synapses,
        defaultclock,
        ms,
        run,
        seed,
    )

    seed(random_seed)
    defaultclock.dt = 1 * ms

    eqs = """
    dTheta/dt = omega + K/N*coupling : radian
    omega : radian/second (constant)  # intrinsic frequency
    coupling : 1
    """

    oscillators = NeuronGroup(N, eqs, method="euler")
    oscillators.Theta = "rand()*2*pi"  # random initial phase
    if OMEGA_SPREAD == 0:
        # Identical natural frequency for every oscillator.
        oscillators.omega = f"{OMEGA_MEAN}*radian/second"
    else:
        oscillators.omega = (
            f"clip({OMEGA_MEAN} + randn()*{OMEGA_SPREAD}, 0, inf)*radian/second"
        )

    connections = Synapses(
        oscillators,
        oscillators,
        "coupling_post = sin(Theta_pre - Theta_post) : 1 (summed)",
    )
    connections.connect()  # all-to-all

    mon = StateMonitor(oscillators, "Theta", record=True)
    run(SIM_DURATION_MS * ms)

    theta = mon.Theta[:]  # shape (N, n_timesteps), radians
    return theta[:, ::FRAME_STRIDE_MS]


def _cache_signature():
    """A string capturing every parameter that affects the simulation output."""
    return (
        f"N={N};dur={SIM_DURATION_MS};stride={FRAME_STRIDE_MS};"
        f"K={K_VALUES};seed={RANDOM_SEED};"
        f"omega_mean={OMEGA_MEAN};omega_spread={OMEGA_SPREAD}"
    )


def load_or_run_simulations():
    """Return ``(thetas, K_values)`` where ``thetas`` has shape (n_K, N, n_frames)."""
    if os.path.exists(CACHE_PATH):
        data = np.load(CACHE_PATH)
        if "signature" in data and str(data["signature"]) == _cache_signature():
            return data["thetas"], list(data["K_values"])

    thetas = []
    for K in K_VALUES:
        print(f"Running Brian2 simulation for K={K:.1f} ...")
        thetas.append(run_sim(K / _seconds()))
    thetas = np.stack(thetas)  # (n_K, N, n_frames)
    np.savez_compressed(
        CACHE_PATH,
        thetas=thetas,
        K_values=np.array(K_VALUES),
        signature=_cache_signature(),
    )
    return thetas, K_VALUES


def _seconds():
    """K is a rate (1/second) in the model; return the ``second`` unit."""
    from brian2 import second

    return second


def order_parameter(phases):
    """Kuramoto order parameter: coherence ``r`` in [0, 1] and mean phase ``phi``."""
    x = np.mean(np.cos(phases))
    y = np.mean(np.sin(phases))
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return r, phi


# ---------------------------------------------------------------------------
# Manim scene
# ---------------------------------------------------------------------------
PANEL_RADIUS = 2.2            # radius of each unit-circle panel in scene units
PANEL_CENTERS = [             # two panels side by side (matches K_VALUES order)
    np.array([-3.6, -0.3, 0.0]),
    np.array([3.6, -0.3, 0.0]),
]
PLAYBACK_FPS = 60             # animation frames per second of video


class CoupledOscillators(Scene):
    def construct(self):
        thetas, k_values = load_or_run_simulations()
        n_frames = thetas.shape[2]

        panels = []
        for thetas_k, K, center in zip(thetas, k_values, PANEL_CENTERS):
            panels.append(self._build_panel(thetas_k, K, center))

        frame_tracker = ValueTracker(0)

        for panel in panels:
            self._attach_updater(panel, frame_tracker)

        # Add static elements (circles) and the dynamic groups.
        for panel in panels:
            self.add(panel["circle"],
                     panel["dots"], panel["arrow"], panel["mean_dot"])

        self.play(
            frame_tracker.animate.set_value(n_frames - 1),
            run_time=n_frames / PLAYBACK_FPS,
            rate_func=linear,
        )
        self.wait(0.5)

    def _build_panel(self, thetas_k, K, center):
        """Create the mobjects for one oscillator panel."""
        circle = Circle(radius=PANEL_RADIUS, color=WHITE, stroke_opacity=0.3)
        circle.move_to(center)

        initial = thetas_k[:, 0]
        dots = VGroup()
        for phase in initial:
            dot = Dot(
                point=center + PANEL_RADIUS * np.array([np.cos(phase), np.sin(phase), 0.0]),
                radius=0.07,
                color=phase_color(phase),
            )
            dots.add(dot)

        r0, phi0 = order_parameter(initial)
        arrow = self._make_arrow(center, r0, phi0)
        mean_dot = Dot(
            point=center + PANEL_RADIUS * r0 * np.array([np.cos(phi0), np.sin(phi0), 0.0]),
            radius=0.09,
            color=YELLOW,
        )

        return {
            "thetas": thetas_k,
            "center": center,
            "circle": circle,
            "dots": dots,
            "arrow": arrow,
            "mean_dot": mean_dot,
        }

    def _make_arrow(self, center, r, phi):
        end = center + PANEL_RADIUS * max(r, 1e-3) * np.array([np.cos(phi), np.sin(phi), 0.0])
        return Arrow(start=center, end=end, color=YELLOW, buff=0, stroke_width=6)

    def _attach_updater(self, panel, frame_tracker):
        thetas_k = panel["thetas"]
        center = panel["center"]
        n_frames = thetas_k.shape[1]

        def update_dots(group):
            frame = int(round(frame_tracker.get_value()))
            frame = max(0, min(frame, n_frames - 1))
            phases = thetas_k[:, frame]
            for dot, phase in zip(group, phases):
                dot.move_to(
                    center + PANEL_RADIUS * np.array([np.cos(phase), np.sin(phase), 0.0])
                )
                # Recolor by current phase so the ring flows like a neon rainbow.
                dot.set_color(phase_color(phase))

        def update_arrow(arrow):
            frame = int(round(frame_tracker.get_value()))
            frame = max(0, min(frame, n_frames - 1))
            r, phi = order_parameter(thetas_k[:, frame])
            end = center + PANEL_RADIUS * max(r, 1e-3) * np.array(
                [np.cos(phi), np.sin(phi), 0.0]
            )
            arrow.put_start_and_end_on(center, end)

        def update_mean_dot(dot):
            frame = int(round(frame_tracker.get_value()))
            frame = max(0, min(frame, n_frames - 1))
            r, phi = order_parameter(thetas_k[:, frame])
            dot.move_to(
                center + PANEL_RADIUS * r * np.array([np.cos(phi), np.sin(phi), 0.0])
            )

        panel["dots"].add_updater(update_dots)
        panel["arrow"].add_updater(update_arrow)
        panel["mean_dot"].add_updater(update_mean_dot)
