"""Render a Manim scene file with configurable output settings.

A thin wrapper around Manim's programmatic API so you don't have to memorize the
``manim`` CLI flags. Point it at any module containing a ``Scene`` subclass.

Examples
--------
Render the default (auto-detected) scene at low quality::

    uv run python scripts/run_script.py scripts/coupled_oscillators_manim.py

Pick a scene, quality, output dir and filename::

    uv run python scripts/run_script.py scripts/coupled_oscillators_manim.py \
        --scene CoupledOscillators \
        --quality high \
        --media-dir out \
        --output kuramoto

Render a GIF instead of mp4::

    uv run python scripts/run_script.py scripts/coupled_oscillators_manim.py --format gif
"""

# Standard library
import argparse
import importlib.util
import inspect
import os
import sys

# Third-party
from manim import Scene, tempconfig

QUALITY_CHOICES = {
    "low": "low_quality",        # 480p15
    "medium": "medium_quality",  # 720p30
    "high": "high_quality",      # 1080p60
    "production": "production_quality",  # 1440p60
    "fourk": "fourk_quality",    # 2160p60
}


def load_module(script_path):
    """Import a Python file by path and return the module object."""
    script_path = os.path.abspath(script_path)
    if not os.path.exists(script_path):
        sys.exit(f"error: script not found: {script_path}")

    module_name = os.path.splitext(os.path.basename(script_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    # Make the module importable by name and let it resolve sibling imports.
    sys.modules[module_name] = module
    sys.path.insert(0, os.path.dirname(script_path))
    spec.loader.exec_module(module)
    return module


def find_scenes(module):
    """Return ``{name: class}`` for Scene subclasses defined in ``module``."""
    return {
        name: obj
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, Scene) and obj is not Scene and obj.__module__ == module.__name__
    }


def select_scene(scenes, requested):
    """Choose which Scene class to render."""
    if requested:
        if requested not in scenes:
            available = ", ".join(sorted(scenes)) or "(none)"
            sys.exit(f"error: scene '{requested}' not found. Available: {available}")
        return requested, scenes[requested]

    if len(scenes) == 1:
        return next(iter(scenes.items()))
    if not scenes:
        sys.exit("error: no Scene subclasses found in the given script.")
    sys.exit(
        "error: multiple scenes found; pass --scene NAME. "
        f"Available: {', '.join(sorted(scenes))}"
    )


def build_config(args):
    """Translate CLI args into a Manim config override dict."""
    overrides = {
        "quality": QUALITY_CHOICES[args.quality],
        "media_dir": args.media_dir,
        # Write videos straight into <media_dir>/videos rather than the default
        # per-quality subfolder (e.g. .../videos/1080p60).
        "video_dir": os.path.join(args.media_dir, "videos"),
        "format": args.format,
        "disable_caching": args.disable_caching,
        "preview": args.preview,
    }
    if args.output:
        overrides["output_file"] = args.output
    if args.fps is not None:
        overrides["frame_rate"] = args.fps
    if args.transparent:
        overrides["transparent"] = True
    return overrides


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render a Manim scene file with configurable output settings."
    )
    parser.add_argument("script", help="Path to the Python file containing the scene.")
    parser.add_argument(
        "-s", "--scene", default=None,
        help="Scene class name (auto-detected if the file has exactly one).",
    )
    parser.add_argument(
        "-q", "--quality", default="low", choices=QUALITY_CHOICES,
        help="Render quality (default: low).",
    )
    parser.add_argument(
        "--media-dir", default="media",
        help="Base directory for rendered output (default: media).",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output filename stem (default: the scene's class name).",
    )
    parser.add_argument(
        "-f", "--format", default="mp4", choices=["mp4", "mov", "gif", "png", "webm"],
        help="Output format (default: mp4).",
    )
    parser.add_argument("--fps", type=int, default=None, help="Override frame rate.")
    parser.add_argument(
        "--transparent", action="store_true",
        help="Render with a transparent background (mov/webm/png).",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Open the rendered file when done.",
    )
    parser.add_argument(
        "--no-cache", dest="disable_caching", action="store_true",
        help="Disable Manim's partial-movie caching.",
    )
    args = parser.parse_args(argv)

    module = load_module(args.script)
    scenes = find_scenes(module)
    scene_name, scene_class = select_scene(scenes, args.scene)

    overrides = build_config(args)
    print(f"Rendering scene '{scene_name}' from {args.script} ...")
    with tempconfig(overrides):
        scene = scene_class()
        scene.render()

    out = scene.renderer.file_writer.movie_file_path
    print(f"Done. Output: {out}")


if __name__ == "__main__":
    main()
