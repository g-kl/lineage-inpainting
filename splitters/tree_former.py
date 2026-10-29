"""Pure one-step tree predictors -- the flow-free counterpart of
``BridgeTreeFlow`` / ``BridgeDeltaTreeFlow``.

Same network and tree machinery as the bridge models, but with **no flow**: no
flow-time, no interpolant, no multi-step rollout, no bridge. The model takes the
observed anchors (clamped to their real values) plus a Gaussian draw at every
masked node and predicts the whole tree in a **single forward pass**; training
is plain MSE on the masked nodes for a sampled anchor slice. In other words it
is exactly the bridge models' ``t=0`` head, trained on its own rather than as
one slice of a flow -- so it cannot roll out, but spends all its capacity on the
one-shot prediction.

``pool_batch`` (stage-1) and ``batch_loss`` (stage-2) are both single one-step
passes, so the trainer drives pool and split as two one-step models.
``DeltaTreeFormer`` adds the residual (delta-from-baseline) frame and
the observed/masked role embedding, identical to ``BridgeDeltaTreeFlow``.

The anchor slice is the same shared :func:`~datasets.tree_record.cross_section`
the slice-based aligner uses.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from datasets.tree_record import TreeBatch, cross_section
from splitters.bridge_tree_flow import _BiasedEncoderLayer
from utils.energy_scores import energy_score


class TreeFormer(nn.Module):
    def __init__(
        self,
        dim: int,
        d_internal: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        hidden: int = 512,
        attention_bias_init: float = 0.0,
        slice_direction: str = "up",
    ):
        super().__init__()
        if slice_direction not in {"up", "down"}:
            raise ValueError("slice_direction must be 'up' or 'down'")
        self.dim = dim
        self.d_internal = d_internal
        self.n_heads = n_heads
        self.slice_direction = slice_direction

        self.input_proj = nn.Linear(dim, d_internal)
        self.out_proj = nn.Sequential(
            nn.Linear(d_internal, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )
        self.node_time_embed = nn.Sequential(
            nn.Linear(1, d_internal), nn.GELU(), nn.Linear(d_internal, d_internal),
        )
        self.encoder_layers = nn.ModuleList([
            _BiasedEncoderLayer(d_internal, n_heads, hidden) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_internal)
        # Per-layer/head scale on the patristic-distance attention bias.
        self.attention_bias_scale = nn.Parameter(
            torch.full((n_layers, n_heads), float(attention_bias_init))
        )

    # -- hooks specialised by the Delta variant ----------------------------

    def _output_baseline(self, batch, anchor_states, anchor_mask):
        """Per-node additive baseline. Base model: none (predict x_1 outright)."""
        return None

    def _residual(self, x, baseline):
        """Endpoint in the model's frame: ``x`` (base) or ``x - baseline`` (delta)."""
        return x if baseline is None else x - baseline

    def _role_features(self, observed_mask):
        """Per-node observed/masked role embedding. Base adds none."""
        return None

    def _attention_bias(self, batch, dist, pad_add):
        """Per-(layer, head) pre-softmax attention bias ``(L, B, n_heads, N, N)``.

        Base model: a single learnable scalar per (layer, head) times the
        symmetric patristic distance ``dist = t_i + t_j - 2 t_mrca`` (so the
        bias is direction- and absolute-time-blind), plus the ``-inf`` pad mask.
        """
        L = len(self.encoder_layers)
        scale = self.attention_bias_scale.view(L, 1, self.n_heads, 1, 1)
        return scale * dist[None, :, None] + pad_add[None]

    # -- network: (x_in) -> x_1 residual prediction ------------------------

    def encode(
        self,
        x_in: torch.Tensor,                # (B, N, dim): anchors clamped, masked = noise
        *,
        batch: TreeBatch,
        observed_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """Transformer stack -> per-node context ``final_norm(h)`` (B, N, d_internal).

        Everything up to (but not including) the output head; ``forward``
        composes ``decode(encode(...))``.
        """
        B, N = batch.mask.shape
        h = self.input_proj(x_in)
        h = h + self.node_time_embed(batch.node_times.unsqueeze(-1))
        role = self._role_features(observed_mask)
        if role is not None:
            h = h + role

        # Pad columns get -inf (keep one open per row to avoid all-(-inf)
        # softmax); the patristic distance is added, scaled per layer/head.
        valid = batch.mask
        pad = ~valid
        pad[~valid.any(dim=-1), 0] = False
        pad_add = torch.zeros(B, N, dtype=h.dtype, device=h.device)
        pad_add.masked_fill_(pad, float("-inf"))
        pad_add = pad_add.view(B, 1, 1, N)

        nt = batch.node_times
        dist = nt.unsqueeze(-1) + nt.unsqueeze(-2) - 2.0 * batch.mrca_times   # (B,N,N)
        bias_all = self._attention_bias(batch, dist, pad_add)                 # (L,B,n_heads,N,N)
        for li, layer in enumerate(self.encoder_layers):
            attn_mask = bias_all[li].expand(B, self.n_heads, N, N).reshape(B * self.n_heads, N, N)
            h = layer(h, attn_mask)

        return self.final_norm(h)

    def decode(self, h: torch.Tensor, batch: TreeBatch) -> torch.Tensor:
        """Output head: context -> ``x_1`` residual prediction. Base: no noise."""
        out = self.out_proj(h)
        return out * batch.mask.unsqueeze(-1).to(out.dtype)

    def forward(
        self,
        x_in: torch.Tensor,                # (B, N, dim): anchors clamped, masked = noise
        *,
        batch: TreeBatch,
        observed_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        return self.decode(
            self.encode(x_in, batch=batch, observed_mask=observed_mask), batch,
        )

    # -- shared one-step prediction (pool + loss) --------------------------

    def _predict(
        self,
        batch: TreeBatch,
        states: torch.Tensor,              # (B, N, dim) anchor values at anchors
        anchor_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """One forward: anchors clamped to their value, masked nodes a Gaussian
        draw; returns the absolute ``x_1`` prediction at every node."""
        anc = anchor_mask.unsqueeze(-1)
        baseline = self._output_baseline(batch, states, anchor_mask)
        noise = torch.randn_like(states)
        x_in = torch.where(anc, states, noise)
        resid_pred = self(x_in, batch=batch, observed_mask=anchor_mask)
        return resid_pred if baseline is None else resid_pred + baseline

    def pool_batch(self, batch: TreeBatch) -> torch.Tensor:
        """One-step prediction of the tree from the observations."""
        anchor_mask = batch.observed_mask & batch.mask
        x1 = self._predict(batch, batch.observed_states, anchor_mask)
        out = torch.where(anchor_mask.unsqueeze(-1), batch.observed_states, x1)
        return out * batch.mask.unsqueeze(-1).to(out.dtype)

    def split_layer_batch(
        self,
        batch: TreeBatch,
        anchor_states: torch.Tensor,
        anchor_mask: torch.BoolTensor,
        n_steps: int | None = None,   # ignored: one-step model
    ) -> torch.Tensor:
        """One-step prediction of the masked nodes from an anchor slice."""
        anchor_mask = anchor_mask & batch.mask
        x1 = self._predict(batch, anchor_states, anchor_mask)
        out = torch.where(anchor_mask.unsqueeze(-1), anchor_states, x1)
        return out * batch.mask.unsqueeze(-1).to(out.dtype)

    # -- training: one-step MSE on the masked nodes ------------------------

    def batch_loss(
        self,
        batch: TreeBatch,
        target_states: torch.Tensor,
        *,
        anchor_mask: torch.BoolTensor,
        leaves_only: bool = False,
        return_pred: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        dtype = target_states.dtype
        anchor_mask = anchor_mask & batch.mask
        x1 = self._predict(batch, target_states, anchor_mask)

        loss_mask = (~anchor_mask) & batch.mask
        if leaves_only:                                 # the observed layer only
            loss_mask = loss_mask & batch.observed_mask
        denom = loss_mask.sum().clamp_min(1).to(dtype) * self.dim
        diff = (x1 - target_states) * loss_mask.unsqueeze(-1).to(dtype)
        loss = (diff * diff).sum() / denom
        if not return_pred:
            return loss
        pred = torch.where(
            anchor_mask.unsqueeze(-1), target_states, x1,
        ) * batch.mask.unsqueeze(-1).to(dtype)
        return loss, pred

    # -- antichain anchor sampler (tree utility, used by the trainer) ------

    def sample_anchor_mask(self, batch: TreeBatch) -> torch.BoolTensor:
        """Anchors = the slice cross-section at a random per-tree cut time
        (same shared :func:`~datasets.tree_record.cross_section` the aligner
        scores)."""
        B = batch.mask.shape[0]
        device = batch.mask.device
        dtype = batch.node_times.dtype

        lo = batch.node_times.masked_fill(~batch.mask, float("inf")).amin(dim=1)
        hi = batch.node_times.masked_fill(~batch.mask, float("-inf")).amax(dim=1)
        span = (hi - lo).clamp_min(1e-6)
        slack = span / float(max(batch.max_depth, 1))
        cut = lo + torch.rand(B, device=device, dtype=dtype) * (span + slack)
        return cross_section(batch, cut, direction=self.slice_direction)


class DeltaTreeFormer(TreeFormer):
    """Residual-frame one-step predictor: the net predicts ``x_1 - baseline``
    and the baseline is added back; an observed/masked role embedding lets the
    transformer tell the absolute anchors from the noise-filled masked nodes.
    Baseline construction is identical to ``BridgeDeltaTreeFlow``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role_embed = nn.Embedding(2, self.d_internal)  # index 1 == observed

    def _role_features(self, observed_mask):
        return self.role_embed(observed_mask.long())

    def _output_baseline(self, batch, anchor_states, anchor_mask):
        dtype = anchor_states.dtype
        anchor_mask = anchor_mask & batch.mask
        am = anchor_mask.unsqueeze(-1).to(dtype)
        par = batch.parent_idx.clamp(min=0)
        par_d = par.unsqueeze(-1).expand(-1, -1, self.dim)

        # below: carry each anchor's value down to its descendants
        below = anchor_states * am
        has_below = anchor_mask.clone()
        for d in range(1, batch.max_depth + 1):
            nodes = (batch.depth == d) & batch.mask & ~anchor_mask
            fill = nodes & has_below.gather(1, par)
            if not fill.any():
                continue
            below = torch.where(fill.unsqueeze(-1), below.gather(1, par_d), below)
            has_below = has_below | fill

        # above: hierarchical mean of subtree anchors (mean of children)
        above = batch.mask & ~has_below
        up = anchor_states * am
        n_kids = batch.adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        for d in range(batch.max_depth - 1, -1, -1):
            targets = (batch.depth == d) & above
            if not targets.any():
                continue
            up = torch.where(targets.unsqueeze(-1), torch.bmm(batch.adj, up) / n_kids, up)

        baseline = torch.where(above.unsqueeze(-1), up, below)
        return baseline * batch.mask.unsqueeze(-1).to(dtype)


class _EnergyDeltaTreeFormerBase(DeltaTreeFormer):
    """Shared core for the distributional one-step tree predictors: the residual
    baseline and role embedding of ``DeltaTreeFormer`` with the conditional
    **energy score** (a strictly proper rule) in place of MSE. Where MSE on a
    stochastic head just drives the prediction to the conditional **mean** (and so
    collapses at branch points), the energy score lets the one-step head learn the
    multimodal ``p(x_tgt | anchors)`` and split. No flow-time, no
    interpolant, no bridge -- the noise enters at the masked nodes; only the
    objective is distributional.

    The subclass decides *where the per-sample noise enters* by implementing
    :meth:`_predict_samples`: :class:`FullEnergyDeltaTreeFormer` draws ``M``
    independent forward passes on freshly resampled masked-node input noise, so
    the spread is built by the whole transformer (attention included).
    """

    def __init__(
        self,
        *args,
        n_energy_samples: int = 8,
        energy_beta: float = 1.0,
        energy_lam: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if n_energy_samples < 2:
            raise ValueError(
                "n_energy_samples must be >= 2 (energy score needs a population)"
            )
        self.n_energy_samples = int(n_energy_samples)
        self.energy_beta = float(energy_beta)
        self.energy_lam = float(energy_lam)

    def _predict_samples(self, batch, states, anchor_mask, n_samples):
        """``n_samples`` endpoint draws stacked as ``(n_samples, B, N, dim)``."""
        raise NotImplementedError

    # -- training: one-step energy score on the masked nodes ---------------

    def batch_loss(
        self,
        batch: TreeBatch,
        target_states: torch.Tensor,
        *,
        anchor_mask: torch.BoolTensor,
        leaves_only: bool = False,
        return_pred: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Conditional energy score of ``M`` noise draws against the target
        (lower better), replacing ``TreeFormer.batch_loss``'s MSE.

        ``leaves_only`` restricts the score to observed, non-anchor nodes --
        the observed layer -- so the loss ignores the model's internal-node
        predictions. ``return_pred`` returns the whole ``M``-sample set
        ``(B, M, N, dim)`` (anchors clamped, pad zeroed) for the trainer's
        split-align step.
        """
        dtype = target_states.dtype
        anchor_mask = anchor_mask & batch.mask
        x1 = self._predict_samples(
            batch, target_states, anchor_mask, self.n_energy_samples,
        )                                                   # (M, B, N, dim)
        x1 = x1.movedim(0, 1)                               # (B, M, N, dim)

        loss_mask = (~anchor_mask) & batch.mask             # (B, N)
        if leaves_only:                                     # observed, non-anchor nodes only
            loss_mask = loss_mask & batch.observed_mask
        loss = energy_score(
            x1, target_states,
            beta=self.energy_beta, lam=self.energy_lam,
            mask=loss_mask.unsqueeze(-1),
        )
        if not return_pred:
            return loss
        anc = anchor_mask[:, None, :, None]                 # (B, 1, N, 1)
        pred = torch.where(anc, target_states[:, None], x1)
        pred = pred * batch.mask[:, None, :, None].to(dtype)  # (B, M, N, dim)
        return loss, pred



class FullEnergyDeltaTreeFormer(_EnergyDeltaTreeFormerBase):
    """Full distributional predictor: the ``M`` energy samples come from ``M``
    *independent full forward passes* on freshly resampled masked-node input
    noise. The noise therefore routes through the whole transformer stack
    (attention included), so sibling symmetry is broken by the full network --
    at the cost of ``M`` encodes per step. The output head is the deterministic
    ``out_proj(h)`` inherited from ``TreeFormer``, and the single-sample
    ``_predict`` used by pool / split / eval is inherited unchanged (one full
    forward with fresh masked-node noise).
    """

    def _predict_samples(self, batch, states, anchor_mask, n_samples):
        """``n_samples`` full forward passes, each re-drawing the masked-node
        input noise so it flows through attention -> ``(M, B, N, dim)``."""
        return torch.stack(
            [self._predict(batch, states, anchor_mask) for _ in range(n_samples)],
            dim=0,
        )
