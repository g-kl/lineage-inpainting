"""
Trainer for pool + split models.
"""

from __future__ import annotations

import abc
import contextlib
import inspect
import itertools
import logging
import os
import torch
import torch.nn as nn
import networkx as nx
import numpy as np
from collections import deque, defaultdict
from datasets.tree_record import TreeBatch, collate_subtrees
from extenders.graft import NullExtender
from pairings import IdentityPairing
from record_samplers.record_samplers import RecordSampler, PairedOffsetSampler

logger = logging.getLogger(__name__)

def _tree_root(tree: nx.DiGraph):
    roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
    assert len(roots) == 1, f"Expected 1 root, got {len(roots)}"
    return roots[0]

class WindowMetrics:
    """Simple sliding-window average over the last ``window`` steps."""

    def __init__(self, window: int = 50):
        self.window = window
        self._buffers: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def update(self, metrics: dict[str, float]):
        for k, v in metrics.items():
            self._buffers[k].append(v)

    def get(self) -> dict[str, float]:
        return {k: sum(buf) / len(buf) for k, buf in self._buffers.items()}

    def reset(self):
        self._buffers.clear()


class EMA:
    """
    Maintains shadow copies on the same device as the originals.
    Call :meth:`update` after every ``optimizer.step()``.
    Use :meth:`swap` (or the :meth:`averaged_parameters` context manager)
    to temporarily replace live weights with EMA weights for eval / checkpointing.
    """

    def __init__(self, params, decay: float = 0.999):
        self.decay = decay
        self.params = list(params)
        self.shadow = [p.data.clone() for p in self.params]
        self.num_updates = 0

    @torch.no_grad()
    def update(self):
        self.num_updates += 1
        # Warm-up decay so early updates matter more
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        # Single fused kernel launch for all params (avoids Python-loop overhead)
        torch._foreach_lerp_(self.shadow, [p.data for p in self.params], 1.0 - decay)

    def swap(self):
        """Swap live and shadow parameters in-place."""
        param_data = [p.data for p in self.params]
        tmp = [t.clone() for t in param_data]
        torch._foreach_copy_(param_data, self.shadow)
        torch._foreach_copy_(self.shadow, tmp)

    @contextlib.contextmanager
    def averaged_parameters(self):
        """Context manager that temporarily loads EMA weights."""
        self.swap()
        try:
            yield
        finally:
            self.swap()

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay, "num_updates": self.num_updates}

    def load_state_dict(self, state):
        self.shadow = state["shadow"]
        self.decay = state["decay"]
        self.num_updates = state["num_updates"]


class Trainer:
    """
    Each step:
      1. Delegate record selection + time sampling to ``record_sampler``.
      2. Draw subtrees from each chosen record.
      3. Pool every tree bottom-up  -> ``(B, N, dim)`` states.
      4. Compute split loss (batched) over all records.
      5. Compute alignment loss over all unique record pairs via ``aligner``.
      6. Backprop and update.
    """

    def __init__(
        self,
        pool: nn.Module,
        split: nn.Module,
        aligner: nn.Module,
        records: dict,
        leaf_states: dict[str, torch.Tensor],
        *,
        pairing: nn.Module | None = None,
        extender=None,
        lr: float = 1e-3,
        n_trees: int = 32,
        n_records: int = 2,
        split_weight: float = 1.0,
        align_weight: float = 1.0,
        split_align_weight: float = 0.0,
        align_only_steps: int = 0,
        leaf_reconstruction_only: bool | list[bool] = False,
        record_sampler: "RecordSampler | None" = None,
        device: str | torch.device = "cuda",
        max_grad_norm: float = 4.0,
        log_window: int = 50,
        ema_decay: float | None = 0.999,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        adam_eps: float = 1e-8,
    ):
        self.device = torch.device(device)
        self.pool = pool.to(self.device)
        self.split = split.to(self.device)
        self.aligner = aligner.to(self.device)
        # Pairing(s): a single module, or a list to alternate across steps --
        # step t uses pairings[t % len]. Alternating is an unbiased minibatch
        # estimate of the summed objective (each step samples one term), at one
        # split-forward per step rather than one per pairing.
        if pairing is None:
            pairings = [IdentityPairing()]
        elif isinstance(pairing, (list, tuple, nn.ModuleList)):
            pairings = list(pairing)
        else:
            pairings = [pairing]
        self.pairings = nn.ModuleList(p.to(self.device) for p in pairings)
        self._step_count = 0
        self.extender = extender or NullExtender()
        self.records = records
        self.leaf_states = leaf_states
        self.n_trees = n_trees
        self.n_records = n_records
        self.split_weight = split_weight
        self.align_weight = align_weight
        # Second-stage alignment: reuse the aligner on the splitter's one-step
        # predictions, not just the pooled states. Off (weight 0) by default,
        # and only available when the splitter can hand back its prediction.
        self.split_align_weight = float(split_align_weight)
        # Phased training: for the first ``align_only_steps`` steps, only the
        # first stage (pool -> aligner) is trained; the split stage is skipped.
        self.align_only_steps = int(align_only_steps)
        # ``leaf_reconstruction_only`` may be a single bool (used every step) or
        # a list aligned with ``self.pairings`` -- step t uses entry t % len, so
        # the split-loss scope can be indexed on the active pairing (e.g.
        # leaves-only under IdentityPairing, full tree under NN pairing).
        if isinstance(leaf_reconstruction_only, (list, tuple)):
            if len(leaf_reconstruction_only) != len(self.pairings):
                raise ValueError(
                    f"leaf_reconstruction_only list (len "
                    f"{len(leaf_reconstruction_only)}) must match the number of "
                    f"pairings ({len(self.pairings)})"
                )
            leaf_reconstruction_only = [bool(x) for x in leaf_reconstruction_only]
        self.leaf_reconstruction_only = leaf_reconstruction_only
        self.record_sampler = record_sampler or PairedOffsetSampler()
        self.max_grad_norm = max_grad_norm
        split_loss_params = inspect.signature(self.split.batch_loss).parameters
        self._split_accepts_anchor_mask = "anchor_mask" in split_loss_params
        # Second-stage alignment needs the splitter to hand back its prediction
        # and to accept the shared antichain (so anchors clamp consistently).
        # Gated on the weight, not on split_weight: with split_weight=0 and
        # split_align_weight>0 the MSE is off but the prediction still feeds
        # the aligner.
        self._split_returns_pred = (
            self.split_align_weight != 0.0
            and self._split_accepts_anchor_mask
            and "return_pred" in split_loss_params
        )

        # Dedupe by tensor identity so a unified pool/split instance (where
        # `pool is split`) doesn't double-count parameters in optimizer/EMA.
        seen: set[int] = set()
        self._params = []
        for p in list(pool.parameters()) + list(split.parameters()):
            if id(p) not in seen:
                seen.add(id(p))
                self._params.append(p)
        # Optimizer: AdamW when weight_decay>0 (standard transformer setup),
        # Adam otherwise. Larger eps (e.g. 1e-6) prevents Adam's adaptive lr
        # from blowing up after a quiet stretch followed by one big gradient.
        self.lr = float(lr)
        self.warmup_steps = int(warmup_steps)
        if weight_decay and weight_decay > 0:
            self.optimizer = torch.optim.AdamW(
                self._params, lr=lr, weight_decay=weight_decay, eps=adam_eps,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self._params, lr=lr, eps=adam_eps,
            )
        # Linear warmup: 0 -> lr over warmup_steps, then flat. Step counter
        # advances on each optimizer.step(); the trainer's start_step is
        # respected via load_checkpoint.
        if self.warmup_steps > 0:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: min(1.0, (step + 1) / self.warmup_steps),
            )
        else:
            self.scheduler = None
        self.ema = EMA(self._params, decay=ema_decay) if ema_decay is not None else None
        self.history: list[dict] = []
        self.metrics = WindowMetrics(window=log_window)

        n_params = sum(p.numel() for p in self._params)
        logger.info(
            "Trainer initialized: device=%s, n_records=%d, n_trees=%d, "
            "lr=%g, max_grad_norm=%g, trainable_params=%d",
            self.device, n_records, n_trees, lr, max_grad_norm, n_params,
        )
        logger.info(
            "Loss weights: split=%.4f, align=%.4f, split_align=%.4f "
            "(stage2_align=%s), leaf_reconstruction_only=%s",
            split_weight, align_weight, self.split_align_weight,
            self._split_returns_pred, leaf_reconstruction_only,
        )
        logger.info(
            "Record sampler: %s, pool size: %d records, ema_decay=%s",
            type(self.record_sampler).__name__, len(records),
            ema_decay,
        )
        logger.info(
            "Pairing(s): %s (alternating per step), explicit_split_anchors=%s",
            [type(p).__name__ for p in self.pairings], self._split_accepts_anchor_mask,
        )
        logger.info("Extender: %s", type(self.extender).__name__)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _make_batch(self, tree_recs: list[dict]) -> "TreeBatch":
        return collate_subtrees(tree_recs, self.leaf_states, device=self.device)

    def _pool_batch(self, batch: "TreeBatch") -> torch.Tensor:
        return self.pool.pool_batch(batch)

    def _leaves_only_for(self, idx: int) -> bool:
        """Split-loss scope for the pairing active at step ``idx``."""
        lro = self.leaf_reconstruction_only
        if isinstance(lro, (list, tuple)):
            return bool(lro[idx % len(lro)])
        return bool(lro)

    def _split_loss_batch(
        self,
        batch: "TreeBatch",
        states: torch.Tensor,
        anchor_mask: torch.BoolTensor | None = None,
        *,
        leaves_only: bool,
        return_pred: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        kwargs = {"leaves_only": leaves_only}
        if anchor_mask is not None:
            kwargs["anchor_mask"] = anchor_mask
        if return_pred:
            kwargs["return_pred"] = True
        return self.split.batch_loss(batch, states, **kwargs)

    def _sample_stage2_anchor_mask(
        self,
        batch: "TreeBatch",
    ) -> torch.BoolTensor | None:
        """Sampled antichain used as anchor for the split loss.

        The same mask is handed to the pairing module so it leaves anchor
        positions alone (it independently excludes ``batch.observed_mask``).
        """
        if not self._split_accepts_anchor_mask:
            return None
        sample_anchor_mask = getattr(self.split, "sample_anchor_mask", None)
        if sample_anchor_mask is None:
            return None
        return sample_anchor_mask(batch) & batch.mask

    def _pair_batch(
        self,
        batch: "TreeBatch",
        states: torch.Tensor,
        stage2_anchor_mask: torch.BoolTensor | None,
        pairing: nn.Module,
    ) -> torch.Tensor:
        return pairing(
            batch,
            states,
            stage2_anchor_mask=stage2_anchor_mask,
        )

    def _grad_norm(self) -> float:
        """Total L2 norm of gradients across all tracked parameters."""
        return torch.nn.utils.clip_grad_norm_(self._params, float("inf")).item()

    def _checkpoint(self, output_dir: str, step: int):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.pool.state_dict(), os.path.join(output_dir, f"pool_step{step}.pt"))
        torch.save(self.split.state_dict(), os.path.join(output_dir, f"split_step{step}.pt"))
        torch.save(self.aligner.state_dict(), os.path.join(output_dir, f"aligner_step{step}.pt"))
        if self.ema is not None:
            with self.ema.averaged_parameters():
                torch.save(self.pool.state_dict(), os.path.join(output_dir, f"pool_ema_step{step}.pt"))
                torch.save(self.split.state_dict(), os.path.join(output_dir, f"split_ema_step{step}.pt"))
        # Trainer state for clean resume: optimizer momentum, EMA shadow,
        # LR scheduler progress, step counter.
        trainer_state = {
            "step": step,
            "optimizer": self.optimizer.state_dict(),
        }
        if self.ema is not None:
            trainer_state["ema"] = self.ema.state_dict()
        if self.scheduler is not None:
            trainer_state["scheduler"] = self.scheduler.state_dict()
        torch.save(trainer_state, os.path.join(output_dir, f"trainer_step{step}.pt"))
        logger.info("Checkpoint saved at step %d to %s", step, output_dir)

    def load_checkpoint(self, output_dir: str, step: int) -> int:
        """Restore weights and (when present) optimizer/EMA state from ``step``.

        If ``trainer_step{N}.pt`` is absent, the optimizer / EMA start fresh
        from the loaded weights.
        Returns the loaded step number.
        """
        map_loc = self.device
        self.pool.load_state_dict(torch.load(
            os.path.join(output_dir, f"pool_step{step}.pt"), map_location=map_loc,
        ))
        if self.split is not self.pool:
            self.split.load_state_dict(torch.load(
                os.path.join(output_dir, f"split_step{step}.pt"), map_location=map_loc,
            ))
        self.aligner.load_state_dict(torch.load(
            os.path.join(output_dir, f"aligner_step{step}.pt"), map_location=map_loc,
        ))
        trainer_path = os.path.join(output_dir, f"trainer_step{step}.pt")
        if os.path.exists(trainer_path):
            ts = torch.load(trainer_path, map_location=map_loc)
            self.optimizer.load_state_dict(ts["optimizer"])
            if self.ema is not None and "ema" in ts:
                self.ema.load_state_dict(ts["ema"])
            if self.scheduler is not None and "scheduler" in ts:
                self.scheduler.load_state_dict(ts["scheduler"])
            logger.info("Resumed full trainer state from step %d", step)
        else:
            logger.info(
                "Loaded weights from step %d; no trainer_step%d.pt present, "
                "optimizer/EMA will warm up from weights.", step, step,
            )
        # Keep alternating-pairing parity consistent across resume.
        self._step_count = step
        return step

    # ------------------------------------------------------------------ #
    #  Core step
    # ------------------------------------------------------------------ #

    def step(self, align_only: bool = False, _max_resample: int = 10) -> dict[str, float]:
        # --- delegate record selection + time to sampler ---
        for _attempt in range(_max_resample):
            chosen_recs, sample_time = self.record_sampler(self.records, self.n_records)

            # Extender grafts each earlier record forward to the common
            # target_leaf_time using donor topology precomputed at startup
            # (SubtreeBank) or a synthetic skeleton. With NullExtender this is a
            # no-op. The target record itself is never extended.
            target_leaf_time = max(r.leaf_time for r in chosen_recs)
            late_rec = max(chosen_recs, key=lambda r: r.leaf_time)

            # --- draw subtrees & pool for every chosen record ---
            batches, states_list, n_leaves_per_rec = [], [], []
            bad = False
            for rec in chosen_recs:
                trees = rec.get_subtrees(time=sample_time, n=self.n_trees)
                if len(trees) == 0:
                    logger.warning(
                        "record %s returned 0 subtrees at time %.4f; resampling records (attempt %d/%d)",
                        getattr(rec, "name", rec), sample_time, _attempt + 1, _max_resample,
                    )
                    bad = True
                    break
                if rec is not late_rec:
                    trees = self.extender(
                        trees,
                        target_leaf_time=target_leaf_time,
                        leaf_states=self.leaf_states,
                    )
                n_leaves_per_rec.append(float(np.mean([t["size"] for t in trees])))
                batch = self._make_batch(trees)
                states = self._pool_batch(batch)
                batches.append(batch)
                states_list.append(states)
            if not bad:
                break
        else:
            raise RuntimeError(
                f"Could not find records with enough subtrees after {_max_resample} attempts"
            )
        n_pick = len(chosen_recs)

        # --- alignment loss over all record pairs ---
        pair_indices = list(itertools.combinations(range(n_pick), 2))
        n_align_pairs = len(pair_indices)

        # --- first-stage alignment loss on raw synthetic states ---
        align_loss = self.aligner.forward_many(
            batches, states_list, pair_indices=pair_indices,
        )
        # Snapshot per-feature losses now: a second forward_many (stage 2)
        # would overwrite the aligner's last_feature_losses.
        align_feat = dict(getattr(self.aligner, "last_feature_losses", {}) or {})

        # --- second-stage split loss on optionally paired states ---
        # Phased warmup: skip the split stage entirely so only the first stage
        # (pool -> aligner) is trained.
        n_paired = 0
        split_align_loss = torch.zeros((), device=self.device)
        split_align_feat: dict[str, float] = {}
        if align_only:
            split_loss = torch.zeros((), device=self.device)
            loss = self.align_weight * align_loss
        else:
            # Alternate pairings across steps: step t uses pairings[t % len].
            pairing_idx = self._step_count % len(self.pairings)
            pairing = self.pairings[pairing_idx]
            leaves_only = self._leaves_only_for(pairing_idx)
            split_states_list, stage2_masks = [], []
            for batch, states in zip(batches, states_list):
                stage2_mask = self._sample_stage2_anchor_mask(batch)
                split_states = self._pair_batch(batch, states, stage2_mask, pairing)
                n_paired += int(pairing.last_n_paired)
                split_states_list.append(split_states)
                stage2_masks.append(stage2_mask)

            if self._split_returns_pred:
                # One forward pass per record yields both the split loss and a
                # predicted tree; reuse the aligner on the predictions. A
                # distributional splitter returns an (B, M, N, dim) sample set;
                # the set-based aligner can't take all M samples as sibling trees
                # (too memory-heavy at the real-data batch size), so we align on
                # a single posterior slice per element (slice 0, a fresh draw
                # each step since z is resampled per forward). A point splitter
                # returns (B, N, dim) and passes through unchanged.
                split_loss = torch.zeros((), device=self.device)
                pred_states_list = []
                for b, s, m in zip(batches, split_states_list, stage2_masks):
                    l, pred = self._split_loss_batch(
                        b, s, m, leaves_only=leaves_only, return_pred=True,
                    )
                    split_loss = split_loss + l
                    pred_states_list.append(pred[:, 0] if pred.dim() == 4 else pred)
                # Realign against real data: align each record's predicted tree
                # against every *other* record's tree with its real observed
                # cells masked back in. ``b.observed_states`` is the original
                # input embedding at observed positions (not a pooled / stage-1
                # prediction), so this restores ground-truth cells; the stage-2
                # ``pred`` re-predicts any observed leaf that fell below the
                # sampled antichain, so the overwrite changes those positions.
                # Internal/unobserved nodes stay predicted, as the aligner
                # always sees a real+predicted mix. Gradient flows through both
                # sides. Ordered pairs (focal i -> other j, i != j); the aligner
                # averages within each stratum, so the loss scale is
                # independent of the pair count.
                real_masked_list = [
                    torch.where(
                        b.observed_mask.unsqueeze(-1), b.observed_states, pred
                    )
                    for b, pred in zip(batches, pred_states_list)
                ]
                cross_pairs = [
                    (i, n_pick + j)
                    for i in range(n_pick)
                    for j in range(n_pick)
                    if i != j
                ]
                split_align_loss = self.aligner.forward_many(
                    batches + batches,
                    pred_states_list + real_masked_list,
                    pair_indices=cross_pairs,
                )
                split_align_feat = dict(
                    getattr(self.aligner, "last_feature_losses", {}) or {}
                )
            else:
                split_loss = sum(
                    self._split_loss_batch(b, s, m, leaves_only=leaves_only)
                    for b, s, m in zip(batches, split_states_list, stage2_masks)
                )

            loss = (
                self.split_weight * split_loss
                + self.align_weight * align_loss
                + self.split_align_weight * split_align_loss
            )

        self.optimizer.zero_grad()
        loss.backward()

        # --- gradient clipping ---
        grad_norm_before = self._grad_norm()
        torch.nn.utils.clip_grad_norm_(self._params, self.max_grad_norm)
        grad_norm_after = self._grad_norm()

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # --- EMA update (single pass over params, no grad) ---
        if self.ema is not None:
            self.ema.update()

        metrics = {
            "split_loss": split_loss.item(),
            "align_loss": align_loss.item(),
            "split_align_loss": split_align_loss.item(),
            "total_loss": loss.item(),
            "grad_norm": grad_norm_before,
            "grad_norm_clipped": grad_norm_after,
            "n_records": n_pick,
            "n_align_pairs": n_align_pairs,
            "n_paired": n_paired,
            "pairing_idx": float(self._step_count % len(self.pairings)),
            "sample_time": sample_time,
        }
        # Per-feature align losses (stage 1 on pooled states, stage 2 on
        # split predictions) if the aligner exposed them.
        for k, v in align_feat.items():
            metrics[f"align_{k}"] = float(v)
        for k, v in split_align_feat.items():
            metrics[f"split_align_{k}"] = float(v)
        for i, v in enumerate(n_leaves_per_rec):
            metrics[f"n_leaves_rec{i}"] = v
        self.history.append(metrics)
        self.metrics.update(metrics)
        self._step_count += 1
        return metrics

    # ------------------------------------------------------------------ #
    #  Training loop
    # ------------------------------------------------------------------ #

    def fit(
        self,
        n_steps: int,
        log_every: int = 10,
        output_dir: str | None = None,
        save_every: int | None = None,
        start_step: int = 0,
    ):
        if start_step >= n_steps:
            logger.info(
                "Already at step %d >= target %d, nothing to do.", start_step, n_steps,
            )
            return
        logger.info(
            "Starting training from step %d to %d (log_every=%d)",
            start_step, n_steps, log_every,
        )
        for step in range(start_step, n_steps):
            self.step(align_only=step < self.align_only_steps)
            if step % log_every == 0 or step == n_steps - 1:
                avg = self.metrics.get()

                def _feat_str(prefix: str, label: str) -> str:
                    keys = sorted(
                        k for k in avg
                        if k.startswith(prefix) and k != f"{prefix}loss"
                    )
                    if not keys:
                        return ""
                    parts = [f"{k[len(prefix):]}={avg[k]:.3f}" for k in keys]
                    return f" | {label}: " + " ".join(parts)

                # "align_" naturally excludes "split_align_*" (different prefix).
                feat_str = _feat_str("align_", "feats")
                split_feat_str = _feat_str("split_align_", "split_feats")
                n_leaves_str = ", ".join(
                    f"{avg[k]:.1f}"
                    for k in sorted(avg) if k.startswith("n_leaves_rec")
                )
                logger.info(
                    "Step %4d/%d | Split: %.4f | Align: %.4f | SAlign: %.4f | "
                    "Total: %.4f | Grad: %.3f (clipped -> %.3f) | "
                    "n_leaves: %s%s%s",
                    step, n_steps,
                    avg.get("split_loss", 0.0),
                    avg.get("align_loss", 0.0),
                    avg.get("split_align_loss", 0.0),
                    avg.get("total_loss", 0.0),
                    avg.get("grad_norm", 0.0),
                    avg.get("grad_norm_clipped", 0.0),
                    n_leaves_str,
                    feat_str,
                    split_feat_str,
                )
            if output_dir and save_every and (step + 1) % save_every == 0:
                self._checkpoint(output_dir, step + 1)
        if output_dir:
            self._checkpoint(output_dir, n_steps)
        logger.info("training complete. final avg metrics: %s", self.metrics.get())
