import argparse
import glob
import importlib
import inspect
import os
import re
import sys

# Python 3.14 added argparse._check_help which rejects Hydra's
# LazyCompletionHelp object. Patch it out so Hydra can initialize.
if sys.version_info >= (3, 14) and hasattr(argparse.ArgumentParser, "_check_help"):
    argparse.ArgumentParser._check_help = lambda self, action: None

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from utils.hash import get_output_dir


def _latest_checkpoint_step(output_dir: str) -> int | None:
    """Return the largest N for which pool_step{N}.pt exists, or None."""
    if not os.path.isdir(output_dir):
        return None
    steps = []
    for f in glob.glob(os.path.join(output_dir, "pool_step*.pt")):
        if "ema" in os.path.basename(f):
            continue
        m = re.search(r"pool_step(\d+)\.pt$", os.path.basename(f))
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    seed = cfg.get("seed", None)
    if seed is not None:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    # Output directory from config hash
    output_dir = get_output_dir(cfg, base_dir="outputs")
    print(f"Output directory: {output_dir}")

    # dataset
    dataset = hydra.utils.instantiate(cfg.dataset)

    print('dataset instantiated')
    # pooler
    pooler = hydra.utils.instantiate(cfg.pooler).to(cfg.device)
    print('pooler instantiated')
    # splitter -- when share_pool_split is set, pool and split are the same
    # instance so a single transformer's weights are updated by both losses.

    if cfg.get("share_pool_split", False):
        splitter = pooler
    else:
        splitter = hydra.utils.instantiate(cfg.splitter).to(cfg.device)
    print('splitter instantiated')

    # aligner
    aligner = hydra.utils.instantiate(cfg.aligner)
    print('aligner instantiated')

    # pairing between stage-one synthetic states and stage-two split targets.
    # ``cfg.pairing`` is a single module by default; a list of pairings makes
    # the trainer alternate across steps (step t uses pairings[t % len]).
    pairing = hydra.utils.instantiate(cfg.pairing)
    if isinstance(pairing, (list, ListConfig)):
        pairing = [p.to(cfg.device) for p in pairing]
    else:
        pairing = pairing.to(cfg.device)

    # extender grafts late-record topology onto early-record subtrees so the
    # batch reaches a common target leaf_time. Default is NullExtender.
    extender = hydra.utils.instantiate(cfg.extender)

    # Hand the dataset to every object that wants to register something off
    # it once at startup -- snap poolers / marginal aligners pull a real-cell
    # bank, bank extenders precompute donor topology. Each implements
    # ``bind_dataset(dataset)`` and takes only the slice it needs (see
    # utils.reference). Dedup by id() so a shared pool/split instance
    # (share_pool_split) is bound once.
    bound: set[int] = set()
    pairing_objs = pairing if isinstance(pairing, list) else [pairing]
    for obj in (aligner, pooler, splitter, *pairing_objs, extender):
        if id(obj) in bound:
            continue
        if hasattr(obj, "bind_dataset"):
            obj.bind_dataset(dataset)
            bound.add(id(obj))

    # record sampler
    record_sampler = hydra.utils.instantiate(cfg.record_sampler)
    print('record_sampler instantiated')

    # split the training config into constructor kwargs and fit kwargs.
    # the trainer's _target_ chooses the class; it exposes
    # fit(n_steps, log_every, save_every, output_dir, start_step).
    training_init = OmegaConf.to_container(cfg.training, resolve=True)
    n_steps = training_init.pop("n_steps")
    log_every = training_init.pop("log_every")
    save_every = training_init.pop("save_every", None)
    target = training_init.pop("_target_")

    # import + call the trainer class directly. hydra.utils.instantiate
    # would re-walk and OmegaConf-merge the runtime kwargs (including
    # dataset.records, which can be tuple-keyed), both slow and prone to
    # KeyValidationError. plain importlib + a Python call avoids both.
    module_name, class_name = target.rsplit(".", 1)
    TrainerClass = getattr(importlib.import_module(module_name), class_name)

    # The objective (loss) and optimizer (lr/wd/etc.) configs are separate,
    # hashed groups -- merge each into the trainer kwargs. Only forward keys
    # this trainer's constructor accepts; any other key is skipped.
    accepted = set(inspect.signature(TrainerClass).parameters)
    for group_name in ("objective", "optimizer"):
        group_init = OmegaConf.to_container(getattr(cfg, group_name), resolve=True)
        for k, v in group_init.items():
            if k in accepted:
                training_init[k] = v
            else:
                print(f"{group_name} key '{k}' not accepted by {class_name}; skipping")

    trainer = TrainerClass(
        pool=pooler,
        split=splitter,
        aligner=aligner,
        pairing=pairing,
        extender=extender,
        records=dataset.records,
        leaf_states=dataset.leaf_states,
        record_sampler=record_sampler,
        **training_init,
    )

    print(f"Trainer instantiated: {trainer.__class__.__name__}")
    # resume from the latest checkpoint in output_dir, if any.
    start_step = 0
    latest = _latest_checkpoint_step(output_dir)
    if latest is not None and hasattr(trainer, "load_checkpoint"):
        print(f"Resuming from checkpoint at step {latest}")
        trainer.load_checkpoint(output_dir, latest)
        start_step = latest

    trainer.fit(
        n_steps=n_steps,
        log_every=log_every,
        save_every=save_every,
        output_dir=output_dir,
        start_step=start_step,
    )

    print(f"Training complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
