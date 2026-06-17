"""Render every scene script via run_script.py.

Runs each scene in its own process (clean Manim state) with a simple output name.
Any extra flags are passed through to run_script.py.

Examples:
    uv run python scripts/run_all.py                      # all at high quality
    uv run python scripts/run_all.py --quality fourk      # all at 4K
    uv run python scripts/run_all.py --media-dir out
"""

# Standard library
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_script.py")

# (scene script, output name)
SCENES = [
    ("kuramoto.py", "kuramoto"),
    ("voting.py", "voting"),
    ("strogatz_mirollo.py", "strogatz-mirollo"),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render all scene scripts.")
    parser.add_argument("-q", "--quality", default="high",
                        help="Render quality passed to run_script (default: high).")
    parser.add_argument("--media-dir", default="media",
                        help="Base output directory (default: media).")
    args, passthrough = parser.parse_known_args(argv)

    for script, output in SCENES:
        cmd = [sys.executable, RUNNER, os.path.join(HERE, script),
               "--output", output, "--quality", args.quality,
               "--media-dir", args.media_dir, *passthrough]
        print(f"\n=== Rendering {script} -> {output} ({args.quality}) ===")
        subprocess.run(cmd, check=True)

    print("\nAll scenes rendered.")


if __name__ == "__main__":
    main()
