"""shared bits of the grid-eval entry points: repo root on sys.path, env parsing."""
import os
import sys

ROOT = os.environ.get("SLURM_SUBMIT_DIR") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def env_flag(name):
    return os.environ.get(name, "0") == "1"


def env_batch(name="BATCH"):
    """semicolon-separated entries, each ``<a>|<b>|...``; returns a list of tuples."""
    return [tuple(e.strip().split("|")) for e in os.environ[name].split(";") if e.strip()]
