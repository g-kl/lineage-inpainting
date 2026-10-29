"""the staged per-model eval driver: couple -> assignment metrics -> distortion.

one pattern for every grid run. each stage saves its artifact next to the checkpoint
under the canonical filename and skips if it already exists (unless ``overwrite``),
so a stage can be re-run on its own:

    stage 1  run_assignments         pooler (or splitter) -> couple_all_trees -> assignments_nn_*.pkl
    stage 2  run_assignment_metrics  metrics that need ONLY the coupling      -> metrics_*.csv
    stage 3  run_model_metrics       clonal + marginal + descendant distortion (splitter), marginal_pool (imputing model) -> <which>_distortion_*.csv
    eval_model(...)                  compose 1 + 2

the metric code is the main-text figure's: stage 2 goes through
``analysis.lib.eval_core.metrics`` (summary_qc's compute_assignment_metrics /
compute_hox) and stage 3 calls summary_qc's compute_distortion directly with its
style constants, on the same seeded subtree / cell samples the figure scores, so the
main-text model's grid rows are the figure's numbers. this module owns only the staging and
the on-disk schema (metrics: columns ``[metric, tree, value]``; distortion: the
figure's own row dicts, ``method == "model"``).
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from analysis.lib.eval_core.annotations import load_annotations
from analysis.lib.eval_core.checkpoints import resolve_checkpoint
from analysis.lib.eval_core.metrics import assignment_metrics


# --------------------------------------------------------------------------
# canonical filenames
# --------------------------------------------------------------------------

def _suffix(n, use_splitter_as_pooler):
    return f"_adj_n{n}{'_splitter' if use_splitter_as_pooler else ''}"


def assignments_path(model_dir, cfg, step, n, use_splitter_as_pooler=False):
    return os.path.join(
        model_dir, f"assignments_nn_{cfg}_step{step}{_suffix(n, use_splitter_as_pooler)}.pkl")


def metrics_path(model_dir, cfg, step, n, use_splitter_as_pooler=False):
    return os.path.join(
        model_dir, f"metrics_{cfg}_step{step}{_suffix(n, use_splitter_as_pooler)}.csv")


def distortion_path(model_dir, cfg, step, which):
    return os.path.join(model_dir, f"{which}_distortion_{cfg}_step{step}.csv")


def _write_csv(df, path):
    """same-dir tmp then atomic replace: Lustre sometimes leaves trailing NULs when
    to_csv overwrites in place."""
    tmp = f"{path}.tmp.{os.getpid()}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# stage 1: assignments (needs the model -- only if not already cached)
# --------------------------------------------------------------------------

def run_assignments(model_dir, dataset=None, *, step=None, n=1,
                    use_splitter_as_pooler=False, device="cuda",
                    overwrite=False, cfg=None):
    """couple the trees with this run's pool (or split) model; save + return.

    reuses an existing pkl unless ``overwrite`` -- checked BEFORE loading the model,
    so a reuse needs no model / dataset. returns ``(assignments, path, step)``.
    """
    name = os.path.basename(model_dir.rstrip("/"))
    which = "split" if use_splitter_as_pooler else "pool"
    if step is None:
        _, step = resolve_checkpoint(model_dir, None, which=which)

    path = assignments_path(model_dir, name, step, n, use_splitter_as_pooler)
    if os.path.exists(path) and not overwrite:
        print(f"[eval_core] reuse {os.path.basename(path)}")
        with open(path, "rb") as f:
            return pickle.load(f), path, step

    if dataset is None:
        raise ValueError(
            f"assignments missing ({os.path.basename(path)}); pass dataset to compute")

    from analysis.lib.eval_core import models  # torch/hydra -- imported lazily
    cfg = cfg or models.load_config(model_dir)
    load = models.load_splitter if use_splitter_as_pooler else models.load_pooler
    model, step = load(model_dir, step, device=device, cfg=cfg)
    print(f"[eval_core] coupling ({n} pool steps) -> {os.path.basename(path)}")
    assignments = models.couple(model, dataset, pool_n_steps=n)
    with open(path, "wb") as f:
        pickle.dump(assignments, f)
    return assignments, path, step


# --------------------------------------------------------------------------
# stage 2: assignment metrics (needs only the coupling -- if not already cached)
# --------------------------------------------------------------------------

def run_assignment_metrics(assignments, model_dir, trees=None, ann=None, *,
                           step, n, use_splitter_as_pooler=False,
                           dataset=None, genot=None, expr=None, overwrite=False):
    """assignment-only metrics -> grid-schema CSV (``[metric, tree, value]``).

    germ_layer_crossing (+ per bin), lineage_oscillation_fraction,
    linkage_correlation_normalized + linkage_distortion (per cut) and, when
    ``dataset`` + ``genot`` are given, hox_correlation + hox_distortion per
    (cell type, Hox gene). skips if the CSV exists (unless ``overwrite``), checked
    before touching any input. returns the path.
    """
    name = os.path.basename(model_dir.rstrip("/"))
    path = metrics_path(model_dir, name, step, n, use_splitter_as_pooler)
    if os.path.exists(path) and not overwrite:
        print(f"[eval_core] reuse {os.path.basename(path)}")
        return path
    if trees is None or ann is None:
        raise ValueError(
            f"metrics missing ({os.path.basename(path)}); pass trees+ann to compute")
    rows = assignment_metrics(trees, assignments, ann, dataset=dataset, genot=genot,
                              expr=expr)
    _write_csv(pd.DataFrame(rows, columns=["metric", "tree", "value"]), path)
    print(f"[eval_core] wrote {os.path.basename(path)}  ({len(rows)} rows)")
    return path


# --------------------------------------------------------------------------
# stage 3: distortion (needs the splitter) -- the figure's compute_distortion
# --------------------------------------------------------------------------

def _compute_distortion():
    """the figure's compute_distortion module. importing it pulls in
    analysis.pipelines.context._context_common, which chdirs to the repo root on
    import, so the working directory is restored afterwards."""
    cwd = os.getcwd()
    try:
        from analysis.figures.summary_qc import compute_distortion as CD
    finally:
        os.chdir(cwd)
    return CD


class LabelScorer:
    """cell-type readouts on the distortion evals' own generations: a generated cloud
    is snapped to the nearest real cells observed at its time (the coupling's own
    snap-to-real) and its cell-type composition is compared with a real composition
    by total variation. built once per dataset (the faiss time indices are the
    expensive part); ``on_transport`` / ``on_split`` are the hooks
    compute_distortion.model_marginal / model_clonal call."""

    def __init__(self, dataset):
        from analysis.pipelines.perturb.perturb import build_time_indices_for
        self.dataset = dataset
        self.label = load_annotations(dataset).name_to_cell_type
        self.time_indices = build_time_indices_for(dataset)
        self.row_of = {n: i for i, n in enumerate(dataset.states.obs_names)}
        self._observed = {}

    def state_of(self, cell):
        """a real cell's state in the splitter's feature space."""
        return np.asarray(self.dataset.reference_features[self.row_of[cell]], np.float32)

    def labels_of(self, names):
        from collections import Counter
        return Counter(self.label[n] for n in names if n in self.label)

    def composition(self, states, t):
        """the cell types of the nearest real cells observed at t to ``states``."""
        from analysis.pipelines.perturb.projection import snap_to_nearest_batch
        key = min(self.time_indices, key=lambda k: abs(k - t))
        _, _, names, _ = snap_to_nearest_batch(np.asarray(states, np.float32),
                                               self.time_indices[key])
        return self.labels_of(names)

    def observed_composition(self, t):
        """every tree leaf observed within the marginal distortion's tolerance of t."""
        from analysis.figures.summary_qc import style as SQ
        if t not in self._observed:
            self._observed[t] = self.labels_of(
                node for tree in self.dataset.trees.values() for node in tree.nodes()
                if tree.out_degree(node) == 0
                and abs(tree.nodes[node].get("time", np.nan) - t) <= SQ.MARGINAL_TIME_TOL)
        return self._observed[t]

    @staticmethod
    def tv(a, b):
        na, nb = sum(a.values()), sum(b.values())
        if na == 0 or nb == 0:
            return float("nan")
        return float(sum(abs(a[k] / na - b[k] / nb) for k in a.keys() | b.keys()) / 2)

    def on_transport(self, t_late, t_early, transported):
        """marginal: the pooled-back ancestors against the cells observed at t_early."""
        return {"cell_type_tv": self.tv(self.composition(transported, t_early),
                                        self.observed_composition(t_early))}

    def on_split(self, row, desc_names, recon_leaves):
        """descendant: the generated leaves against the subtree's true leaves."""
        return {"cell_type_tv": self.tv(self.composition(recon_leaves, row["leaf_time"]),
                                        self.labels_of(desc_names))}


def distortion_samples(dataset):
    """the figure's seeded clonal candidates + marginal samples (dataset-determined,
    so identical for every run and for the main-text figure), and the label scorer
    built on the same dataset."""
    import torch
    from analysis.figures.summary_qc import style as SQ
    CD = _compute_distortion()
    trees, leaf_states = dataset.trees, dataset.leaf_states
    leaf_np = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
               for k, v in leaf_states.items()}
    candidates = CD.collect_clonal_candidates(
        trees, leaf_states, offset=SQ.CLONAL_OFFSET, time_tol=SQ.CLONAL_TIME_TOL,
        n=SQ.N_SUBTREE_LEAVES, seed=SQ.SEED, min_leaves=SQ.MIN_CLONE_LEAVES)
    samples = CD.collect_marginal_samples(
        trees, leaf_np, time_tol=SQ.MARGINAL_TIME_TOL, n_cells=SQ.N_MARGINAL_CELLS,
        seed=SQ.SEED)
    print(f"[eval_core] distortion samples: {len(candidates)} clonal subtrees, "
          f"{len(samples)} adjacent-timepoint pairs")
    return candidates, samples, LabelScorer(dataset)


DISTORTIONS = ("clonal", "marginal", "descendant", "marginal_pool")
SPLITTER_PASSES = ("clonal", "marginal", "descendant")   # one noise stream: run together


def _pooler_root_states(pooler, n_steps=1):
    """the imputing model's pool-back on a collated subtree batch: ``pool_batch``
    (as the coupling applies it) read at the root. ``pool_back(batch) -> (B, dim)``."""
    import torch
    from analysis.lib.coupling.nearest_neighbor_coupling import _pool_call

    @torch.no_grad()
    def pool_back(batch):
        pooled = _pool_call(pooler, batch, n_steps)
        return pooled[torch.arange(pooled.shape[0], device=pooled.device), batch.root_idx]

    return pool_back


def run_model_metrics(model_dir, dataset, *, step=None, samples=None, device="cuda",
                      overwrite=False, cfg=None, coupling_n=8):
    """the distortion CSVs of one run (DISTORTIONS):
    clonal + marginal exactly as the main-text figure scores its model rows
    (compute_distortion.model_clonal / model_marginal with summary_qc.style's
    SPLIT_STEPS / MAX_NODES / BATCH_SIZE); descendant = the figure's clonal subtrees
    split from the root's COUPLED real cell (this run's splitter-as-pooler coupling
    at ``coupling_n`` steps) instead of the pooled-back root, against the true
    leaves; marginal_pool = the marginal distortion with the IMPUTING model (the pool
    model applied once, as the imputing-side coupling) doing the pool-back, on the
    same samples. the marginal, descendant and marginal_pool rows also carry the
    cell-type TV of the same generation (LabelScorer). writes
    ``<which>_distortion_<cfg>_step{step}.csv``; computes only the missing ones
    unless ``overwrite`` (the three splitter passes share one seeded noise stream
    and are always run together)."""
    from analysis.figures.summary_qc import style as SQ
    from analysis.lib.eval_core import models
    CD = _compute_distortion()

    name = os.path.basename(model_dir.rstrip("/"))
    if step is None:
        _, step = resolve_checkpoint(model_dir, None, which="split")
    paths = {w: distortion_path(model_dir, name, step, w) for w in DISTORTIONS}
    todo = {w for w, p in paths.items() if overwrite or not os.path.exists(p)}
    if todo & set(SPLITTER_PASSES):
        todo |= set(SPLITTER_PASSES)
    if not todo:
        print(f"[eval_core] reuse {', '.join(os.path.basename(p) for p in paths.values())}")
        return {**paths, "step": step}

    import torch
    candidates, marg_samples, scorer = samples or distortion_samples(dataset)
    cfg = cfg or models.load_config(model_dir)
    trees, leaf_states = dataset.trees, dataset.leaf_states
    rollout = dict(n_steps=SQ.SPLIT_STEPS, max_nodes=SQ.MAX_NODES, batch_size=SQ.BATCH_SIZE)
    written = []

    if "clonal" in todo:
        assignments, _, _ = run_assignments(model_dir, None, step=step, n=coupling_n,
                                            use_splitter_as_pooler=True)
        splitter, step = models.load_splitter(model_dir, step, device=device, cfg=cfg)
        splitter.eval()
        # the bridge rollouts draw noise; the figure's script leaves torch unseeded,
        # so two runs differ by a few percent. seed with the figure's SEED so every
        # run sees the same draws and a recompute reproduces itself.
        torch.manual_seed(SQ.SEED)
        clonal_rows = CD.model_clonal(splitter, candidates, trees, leaf_states, **rollout)
        CD.on_transport = scorer.on_transport
        try:
            marginal_rows = CD.model_marginal(splitter, marg_samples, trees, leaf_states,
                                              time_tol=SQ.MARGINAL_TIME_TOL, **rollout)
        finally:
            CD.on_transport = None
        # descendant: the same subtrees split from the coupled real root. after the
        # figure's two passes, so their noise draws are unchanged.
        coupled = [c for c in candidates if c[3] in assignments.get(c[0], {})]
        CD.root_state_of = lambda tname, anc: scorer.state_of(assignments[tname][anc])
        CD.on_split = scorer.on_split
        try:
            descendant_rows = CD.model_clonal(splitter, coupled, trees, leaf_states, **rollout)
        finally:
            CD.root_state_of = CD.on_split = None
        _write_csv(pd.DataFrame(clonal_rows), paths["clonal"])
        _write_csv(pd.DataFrame(marginal_rows), paths["marginal"])
        _write_csv(pd.DataFrame(descendant_rows), paths["descendant"])
        written += [f"clonal ({len(clonal_rows)} rows)", f"marginal ({len(marginal_rows)} rows)",
                    f"descendant ({len(descendant_rows)} rows; {len(coupled)}/{len(candidates)} "
                    f"subtrees have a coupled root)"]

    if "marginal_pool" in todo:
        pooler, step = models.load_pooler(model_dir, step, device=device, cfg=cfg)
        pooler.eval()
        torch.manual_seed(SQ.SEED + 1)        # its own stream: this pass can run alone
        CD.pool_back = _pooler_root_states(pooler, n_steps=1)
        CD.on_transport = scorer.on_transport
        try:
            pool_rows = CD.model_marginal(None, marg_samples, trees, leaf_states,
                                          time_tol=SQ.MARGINAL_TIME_TOL, **rollout)
        finally:
            CD.pool_back = CD.on_transport = None
        _write_csv(pd.DataFrame(pool_rows), paths["marginal_pool"])
        written.append(f"marginal_pool ({len(pool_rows)} rows)")

    print(f"[eval_core] wrote {name} step {step}: " + ", ".join(written))
    return {**paths, "step": step}


# --------------------------------------------------------------------------
# compose
# --------------------------------------------------------------------------

def eval_model(model_dir, dataset=None, ann=None, *, step=None, n=1,
               use_splitter_as_pooler=False, device="cuda", overwrite=False,
               genot=None, expr=None):
    """stages 1 + 2 for one run. both skip if their artifact exists, so a fully
    cached run needs ``dataset=None``. returns the artifact paths."""
    assignments, apath, step = run_assignments(
        model_dir, dataset, step=step, n=n,
        use_splitter_as_pooler=use_splitter_as_pooler, device=device,
        overwrite=overwrite)
    trees = dataset.trees if dataset is not None else None
    if ann is None and dataset is not None:
        ann = load_annotations(dataset)
    mpath = run_assignment_metrics(
        assignments, model_dir, trees, ann, step=step, n=n,
        use_splitter_as_pooler=use_splitter_as_pooler, dataset=dataset, genot=genot,
        expr=expr, overwrite=overwrite)
    return {"assignments": apath, "metrics": mpath, "step": step}
