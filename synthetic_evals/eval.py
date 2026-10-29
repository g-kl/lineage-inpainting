"""Single config-driven entrypoint for the synthetic pool-split evals.

The benchmark is selected by an ``eval`` config group, the model by the
``--experiment`` (Hydra experiment config). One command covers every
(benchmark x model) combination:

    # crossing family (model = experiment); extra Hydra overrides pass straight
    # through, so the ambient dim is one per run (the figure uses 1, 4, 16, 64)
    python synthetic_evals/eval.py --eval high_dim_crossing \
        --experiment high_dim_crossing/unshared_vpsde_idnn_leaf dataset.proj_dim=4

    # context family (fate-bias sigma axis of the eval config)
    python synthetic_evals/eval.py --eval context \
        --experiment synbench_fb_unshared_vpsde_idnn_leaf

Model-agnostic baselines are trained once and cached under
``outputs/_synth_baselines/`` (see synthetic_evals.baselines.cache), so
evaluating several model variants of one benchmark reuses a single baseline
fit instead of retraining it per variant.

The eval config group is appended with ``+eval=`` rather than declared in the
base defaults, so it never enters the training-config hash and leaves model
checkpoint hashes unchanged.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", required=True,
                   help="eval config group (config/eval/<name>.yaml)")
    p.add_argument("--experiment", default=None,
                   help="Hydra experiment (model) for crossing-family evals")
    p.add_argument("--plot-only", action="store_true",
                   help="skip compute (context: re-render from the cached results pickle)")
    p.add_argument("--no-skip-train", action="store_true",
                   help="crossing: run the pool-split multirun even if checkpoints exist")
    p.add_argument("--retrain-pool-split", action="store_true",
                   help="context: train the per-sigma checkpoints before evaluating")
    p.add_argument("--pool-split-steps", type=int, default=None,
                   help="context: override training.n_steps in the multirun")
    p.add_argument("--launcher", default="local",
                   help="Hydra launcher for the pool-split multirun")
    p.add_argument("overrides", nargs="*",
                   help="extra Hydra overrides for crossing-family per-variant "
                        "composition, e.g. dataset.proj_dim=200")
    args = p.parse_args()

    # Split passthrough overrides: ``eval.*`` configure the benchmark (applied
    # to the top-level eval config); everything else (dataset.*, training.*, ...)
    # configures the model run and is forwarded to the per-variant composition.
    eval_overrides = [o for o in args.overrides if o.split("=", 1)[0].startswith("eval.")]
    model_overrides = [o for o in args.overrides if o not in eval_overrides]

    # Append the eval group (never in the base defaults -> never hashed).
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        base = compose(
            config_name="config", overrides=[f"+eval={args.eval}", *eval_overrides],
        )
    ecfg = base.eval

    if args.experiment is None:
        p.error("evals require --experiment (the model config)")

    if ecfg.kind == "crossing":
        from synthetic_evals.core import crossing_eval
        crossing_eval.run(
            ecfg, args.experiment,
            plot_only=args.plot_only,
            skip_pool_split_train=not args.no_skip_train,
            launcher=args.launcher,
            extra_overrides=model_overrides,
        )
    elif ecfg.kind == "context":
        from synthetic_evals.core import context_eval
        context_eval.run(
            ecfg, args.experiment,
            plot_only=args.plot_only,
            skip_pool_split_train=not args.no_skip_train,
            retrain_pool_split=args.retrain_pool_split,
            pool_split_steps=args.pool_split_steps,
            launcher=args.launcher,
            extra_overrides=model_overrides,
        )
    else:
        raise ValueError(f"unknown eval kind {ecfg.kind!r}")


if __name__ == "__main__":
    main()
