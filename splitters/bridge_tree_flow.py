"""Bridge-backed tree flow models -- flow matching over a tree, in ambient space.

A small model: the transformer predicts ``x_1`` (the data) from a noised
tree, and a :class:`~splitters.bridges.flow_matching.CFMBridge` owns all
the flow geometry (the prior, the interpolant, the ODE step, the rollout). The
model owns only the network and the tree structure.

Design:

- **Ambient flow.** The flow runs in ``dim`` (data) space.
  ``forward(x_t, t, ...) -> x_1`` is exactly the callable the bridge
  integrates, so the prior is ``N(0, I)`` in ``dim`` and anchors are clamped
  to their observed data values.
- **Structure from a patristic-distance attention bias.** The tree enters
  through a single additive attention bias, the patristic distance
  ``t_i + t_j - 2 t_mrca(i, j)`` scaled per layer/head.
- **Flow geometry lives in the bridge.** Training builds the point with
  ``bridge.prior`` + ``bridge.interpolate``; the multi-step rollout is a single
  ``bridge.sample`` call.

A single :meth:`_bridge_step` -- interpolate to time ``t``, run the net once,
reconstruct ``x_1`` -- is shared by training and the one-step pool, so the
one-step prediction the trainer consumes is ``batch_loss`` at ``t=0`` with
the observations as anchors (just without the loss).

``BridgeDeltaTreeFlow`` adds the residual (delta-from-baseline) frame: masked
nodes flow ``noise -> (x_1 - baseline)`` and the baseline is added back at the
output; an observed/masked role embedding lets the transformer tell the frames
apart.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from datasets.tree_record import TreeBatch, cross_section
from splitters.bridges.flow_matching import CFMBridge


class _BiasedEncoderLayer(nn.Module):
    """Pre-norm transformer layer taking an additive ``(B*n_heads, L, L)`` mask.

    Hand-written to bypass ``nn.TransformerEncoderLayer``'s eval-mode fast
    path, which produces NaN at pad rows when the additive ``src_mask`` has
    ``-inf`` columns.
    """

    def __init__(self, dim: int, n_heads: int, hidden: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        return x + self.ff(self.norm2(x))


class BridgeTreeFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        d_internal: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        hidden: int = 512,
        n_flow_steps: int = 8,
        attention_bias_init: float = 0.0,
        slice_direction: str = "up",
        bridge: CFMBridge | None = None,
    ):
        super().__init__()
        if slice_direction not in {"up", "down"}:
            raise ValueError("slice_direction must be 'up' or 'down'")
        self.dim = dim
        self.d_internal = d_internal
        self.n_heads = n_heads
        self.n_flow_steps = int(n_flow_steps)
        self.slice_direction = slice_direction
        self.bridge = bridge if bridge is not None else CFMBridge()

        self.input_proj = nn.Linear(dim, d_internal)
        self.out_proj = nn.Sequential(
            nn.Linear(d_internal, hidden), nn.GELU(), nn.Linear(hidden, dim),
        )
        self.node_time_embed = nn.Sequential(
            nn.Linear(1, d_internal), nn.GELU(), nn.Linear(d_internal, d_internal),
        )
        self.flow_time_embed = nn.Sequential(
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
        """Endpoint in the flow's frame: ``x`` (base) or ``x - baseline`` (delta)."""
        return x if baseline is None else x - baseline

    def _role_features(self, observed_mask):
        """Per-node observed/masked role embedding. Base adds none."""
        return None

    # -- bridge callable: (x_t, t) -> x_1 residual prediction --------------

    def forward(
        self,
        x_t: torch.Tensor,                 # (B, N, dim) flow state in data space
        t: torch.Tensor,                   # (B,) flow time
        *,
        batch: TreeBatch,
        observed_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        B, N = batch.mask.shape
        h = self.input_proj(x_t)
        h = h + self.node_time_embed(batch.node_times.unsqueeze(-1))
        h = h + self.flow_time_embed(t.view(B, 1, 1))
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
        dist = (nt.unsqueeze(-1) + nt.unsqueeze(-2) - 2.0 * batch.mrca_times).unsqueeze(1)
        for li, layer in enumerate(self.encoder_layers):
            bias = self.attention_bias_scale[li].view(1, -1, 1, 1) * dist + pad_add
            attn_mask = bias.expand(B, self.n_heads, N, N).reshape(B * self.n_heads, N, N)
            h = layer(h, attn_mask)

        out = self.out_proj(self.final_norm(h))
        return out * batch.mask.unsqueeze(-1).to(out.dtype)

    # -- shared one-shot prediction (training + one-step pool) -------------

    def _bridge_step(
        self,
        batch: TreeBatch,
        states: torch.Tensor,              # (B, N, dim) anchor values; endpoint at masked
        anchor_mask: torch.BoolTensor,
        t: torch.Tensor,                   # (B,) flow time
    ) -> torch.Tensor:
        """Interpolate to time ``t``, run the net once, return the absolute
        ``x_1`` prediction at every node.

        Anchors are clamped to their absolute value; masked nodes sit on the
        noise -> residual path. ``batch_loss`` calls this at a random ``t``
        (then takes MSE on masked nodes); the one-step ``pool_batch`` calls it
        at ``t=0`` (where the masked input is pure noise, independent of the
        masked endpoint) with the observations as anchors.
        """
        anc = anchor_mask.unsqueeze(-1)
        baseline = self._output_baseline(batch, states, anchor_mask)
        x_0 = self.bridge.prior(states.shape, states.device, states.dtype)
        x_t = self.bridge.interpolate(x_0, self._residual(states, baseline), t)
        main = torch.where(anc, states, x_t)
        resid_pred = self(main, t, batch=batch, observed_mask=anchor_mask)
        return resid_pred if baseline is None else resid_pred + baseline

    # -- training: one random-t interpolant --------------------------------

    def batch_loss(
        self,
        batch: TreeBatch,
        target_states: torch.Tensor,
        *,
        anchor_mask: torch.BoolTensor,
        leaves_only: bool = False,
        return_pred: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B = batch.mask.shape[0]
        device, dtype = batch.mask.device, target_states.dtype
        anchor_mask = anchor_mask & batch.mask

        t = self.bridge.sample_t(B, device).to(dtype)
        x1 = self._bridge_step(batch, target_states, anchor_mask, t)

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

    # -- inference + the one-step training head ----------------------------

    def pool_batch(self, batch: TreeBatch, n_steps: int = 1) -> torch.Tensor:
        """Predict the tree from the observations.

        ``n_steps=1`` is the one-step denoising head the trainer optimises --
        the shared :meth:`_bridge_step` at ``t=0``, identical to ``batch_loss``
        minus the loss. ``n_steps>1`` is the inference-only multi-step rollout.
        """
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        anchor_mask = batch.observed_mask & batch.mask
        anc = anchor_mask.unsqueeze(-1)
        if n_steps == 1:
            t = torch.zeros(batch.mask.shape[0], device=batch.mask.device,
                            dtype=batch.observed_states.dtype)
            x1 = self._bridge_step(batch, batch.observed_states, anchor_mask, t)
            out = torch.where(anc, batch.observed_states, x1)
            return out * batch.mask.unsqueeze(-1).to(out.dtype)
        return self.split_layer_batch(
            batch, batch.observed_states, anchor_mask, n_steps=n_steps,
        )

    def split_layer_batch(
        self,
        batch: TreeBatch,
        anchor_states: torch.Tensor,
        anchor_mask: torch.BoolTensor,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """Multi-step flow rollout (inference): integrate masked nodes from
        noise to data with the bridge, anchors pinned throughout."""
        n_steps = int(n_steps) if n_steps is not None else self.n_flow_steps
        anchor_mask = anchor_mask & batch.mask
        anc = anchor_mask.unsqueeze(-1)
        masked = (batch.mask & ~anchor_mask).unsqueeze(-1)

        baseline = self._output_baseline(batch, anchor_states, anchor_mask)
        x_0 = torch.where(
            anc, anchor_states,
            self.bridge.prior(anchor_states.shape, anchor_states.device, anchor_states.dtype),
        )
        x1_resid = self.bridge.sample(
            self, x_0, n_steps=n_steps, mask=masked, return_x1=True,
            batch=batch, observed_mask=anchor_mask,
        )
        x1 = x1_resid if baseline is None else x1_resid + baseline
        out = torch.where(anc, anchor_states, x1)
        return out * batch.mask.unsqueeze(-1).to(out.dtype)

    # -- antichain anchor sampler (tree utility, used by the trainer) ------

    def sample_anchor_mask(self, batch: TreeBatch) -> torch.BoolTensor:
        """Anchors = the slice cross-section at a random per-tree cut time.

        ``cut ~ U[lo, hi + slack]`` with ``slack = span / max_depth`` so each of
        the ``max_depth + 1`` layers (incl. the leaf layer, reached once the cut
        runs past ``hi``) is sampleable with roughly equal mass. The frontier is
        the same :func:`~datasets.tree_record.cross_section` the slice-based
        aligner scores, so the splitter conditions on exactly those slices.
        """
        B = batch.mask.shape[0]
        device = batch.mask.device
        dtype = batch.node_times.dtype

        lo = batch.node_times.masked_fill(~batch.mask, float("inf")).amin(dim=1)
        hi = batch.node_times.masked_fill(~batch.mask, float("-inf")).amax(dim=1)
        span = (hi - lo).clamp_min(1e-6)
        slack = span / float(max(batch.max_depth, 1))
        cut = lo + torch.rand(B, device=device, dtype=dtype) * (span + slack)
        return cross_section(batch, cut, direction=self.slice_direction)


class BridgeDeltaTreeFlow(BridgeTreeFlow):
    """Residual-frame variant: flow ``noise -> (x_1 - baseline)``.

    The baseline is built from the anchor states and tree topology (hierarchical
    mean of subtree anchors above the antichain; nearest anchor ancestor's value
    below it). An observed/masked role embedding lets the transformer tell the
    absolute anchors from the residual-frame masked nodes.
    """

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
