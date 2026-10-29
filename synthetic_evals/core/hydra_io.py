"""Hydra plumbing shared by every synthetic eval.

Composes the project config for a given ``(experiment, dataset)`` pair, locates
the matching trained-model output directory by config hash, and reloads the
pool/split modules. This is the only place that touches Hydra, so the eval
loops stay pure Python.
"""

from __future__ import annotations

import glob
import logging
import subprocess
import sys
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from utils.hash import get_output_dir

ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger("synth_eval")


# --------------------------------------------------------------------------
#  Config composition
# --------------------------------------------------------------------------


def compose_cfg(experiment: str, dataset: str | None = None,
                overrides: list[str] | None = None):
    """Resolved DictConfig for an experiment (optionally on a specific dataset).

    ``dataset=None`` uses the experiment's own default dataset -- the right
    choice when the experiment fully specifies its data (e.g. the per-sigma
    context configs). Opens its own ``initialize_config_dir`` context per call
    (the mechanism that composes cleanly outside an ``@hydra.main`` app).
    """
    ov = list(overrides or []) + [f"experiment={experiment}"]
    if dataset is not None:
        ov.append(f"dataset={dataset}")
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        return compose(config_name="config", overrides=ov)


def output_dir_for(
    experiment: str, dataset: str | None = None, overrides: list[str] | None = None,
) -> str:
    cfg = compose_cfg(experiment, dataset, overrides)
    return get_output_dir(cfg, base_dir=str(ROOT / "outputs"), create=False)


def latest_step(d: Path) -> int | None:
    """Largest non-EMA ``pool_step{N}.pt`` checkpoint step in *d*, or None."""
    steps: list[int] = []
    for f in glob.glob(str(Path(d) / "pool_step*.pt")):
        name = Path(f).name
        if "ema" in name:
            continue
        try:
            steps.append(int(name.replace("pool_step", "").replace(".pt", "")))
        except ValueError:
            pass
    return max(steps) if steps else None


# --------------------------------------------------------------------------
#  Pool-split training (Hydra multirun over dataset variants)
# --------------------------------------------------------------------------


def run_pool_split_multirun(
    experiment: str,
    datasets: list[str],
    *,
    skip_if_done: bool = True,
    n_steps: int | None = None,
    extra_overrides: list[str] | None = None,
    launcher: str = "local",
):
    """Train pool/split for *experiment* across *datasets* in one Hydra multirun.

    ``launcher='local'`` runs the variants in-process, strictly one at a time
    (the right default when this driver is itself inside one GPU job). To pack
    several variants onto one GPU at once, pass ``launcher='joblib'`` (needs the
    hydra-joblib-launcher plugin; see config/hydra/launcher/joblib.yaml). A no-op
    when ``skip_if_done`` and every variant already has a checkpoint.
    """
    overrides = list(extra_overrides or [])
    if n_steps is not None:
        overrides.append(f"training.n_steps={n_steps}")

    if skip_if_done:
        if all(latest_step(Path(output_dir_for(experiment, d, overrides))) is not None
               for d in datasets):
            logger.info("All pool-split checkpoints present; skipping training")
            return
        for d in datasets:
            step = latest_step(Path(output_dir_for(experiment, d, overrides)))
            logger.info("[%s] checkpoint step=%s", d, step)

    cmd = [
        sys.executable, str(ROOT / "main.py"), "-m",
        f"experiment={experiment}",
        f"dataset={','.join(datasets)}",
        f"hydra/launcher={launcher}",
    ] + overrides
    logger.info("Pool-split multirun: %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


# --------------------------------------------------------------------------
#  Checkpoint reload
# --------------------------------------------------------------------------


def load_pool_split(
    experiment: str, dataset: str, overrides: list[str] | None = None,
    *, step: int | None = None,
):
    """Reload trained pool & splitter for one ``(experiment, dataset)``.

    ``step`` pins which checkpoint to load; the default (None) takes the latest
    on disk. Pin it when several runs share one output dir at different step
    counts and you want a fixed training budget across them.

    Honors ``share_pool_split`` (splitter is the pooler instance) and prefers
    EMA weights when present. Returns ``(pool, split, cfg)``.
    """
    out_dir = Path(output_dir_for(experiment, dataset, overrides))
    cfg = OmegaConf.load(out_dir / "config.yaml")
    if step is None:
        step = latest_step(out_dir)
        if step is None:
            raise FileNotFoundError(f"No checkpoints found in {out_dir}")
    elif not (out_dir / f"pool_step{step}.pt").exists():
        raise FileNotFoundError(f"No step-{step} checkpoint in {out_dir}")

    pool = hydra.utils.instantiate(cfg.pooler).to(DEVICE)
    if cfg.get("share_pool_split", False):
        split = pool
    else:
        split = hydra.utils.instantiate(cfg.splitter).to(DEVICE)

    pool_ckpt = out_dir / f"pool_ema_step{step}.pt"
    split_ckpt = out_dir / f"split_ema_step{step}.pt"
    if not pool_ckpt.exists():
        pool_ckpt = out_dir / f"pool_step{step}.pt"
        split_ckpt = out_dir / f"split_step{step}.pt"

    pool.load_state_dict(torch.load(pool_ckpt, map_location=DEVICE))
    if split is not pool:
        split.load_state_dict(torch.load(split_ckpt, map_location=DEVICE))
    pool.eval()
    split.eval()
    logger.info(
        "[%s] loaded pool/split step %d (ema=%s, shared=%s)",
        dataset, step, "ema" in str(pool_ckpt), split is pool,
    )
    return pool, split, cfg
