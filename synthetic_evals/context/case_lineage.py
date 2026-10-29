"""Add the case clone's shared lineage to a context results pickle.

The eval records only the clone's sibling STATES at t_in, not their lineage back
to the common ancestor -- but the eval sim is deterministic in (seed, dataset
config), so we re-simulate the case's sim on CPU and extract the subtree from the
clone's common ancestor down to its t_in siblings. This lets the figure show the
n=8 clone as one family tree (ancestor -> 2 -> 4 -> 8) across three time points,
making "these cells are siblings" visually obvious. No model / GPU needed.

Usage:
    python -m synthetic_evals.context.case_lineage          # patch all 3 pkls
"""

from __future__ import annotations

import glob
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets.synthetic import make_fate_bias  # noqa: E402

# results pickle -> an output-dir glob whose config gives the sim params.
PKL_TO_STEM = {
    "results.pkl": "synbench_fb_attnbias_sig0*",
    "results_dbtf.pkl": "synbench_fb_dbtf_sig0*",
    "results_dbtf_vpsde.pkl": "synbench_fb_dbtf_vpsde_sig0*",
}


def _sim_kw_for(sigma: float, stem_glob: str) -> dict:
    """Read sim params from any matching output dir's config (sim depends only
    on the dataset, identical across model variants), with sigma pinned. Falls
    back to any fate-bias output dir if the preferred stem is unavailable."""
    d = sorted(glob.glob(str(ROOT / "outputs" / stem_glob)))
    if not d:
        d = sorted(glob.glob(str(ROOT / "outputs" / "synbench_fb_*")))
    if not d:
        raise FileNotFoundError(f"no fate-bias output dir for {stem_glob}")
    ds = OmegaConf.load(Path(d[0]) / "config.yaml").dataset
    # MUST match eval_sigma's sim_kw exactly, or make_fate_bias produces a
    # different realization (its defaults differ from the dataset's theta / mu /
    # fate_prob) and the re-simulated lineage won't line up with the recorded
    # descendant trajectories.
    return dict(
        n_generations=int(ds.n_generations),
        split_gen=int(ds.split_gen),
        n_primed_generations=int(ds.n_primed_generations),
        sigma=float(sigma),
        theta=float(ds.theta),
        mu_1_primed=float(ds.mu_1_primed),
        mu_2_primed=float(ds.mu_2_primed),
        mu_1=float(ds.mu_1),
        mu_2=float(ds.mu_2),
        fate_prob=float(ds.fate_prob),
        branch_interval=float(ds.branch_interval),
        steps_per_interval=int(ds.steps_per_interval),
    )


def record_clone_lineage(res, ancestor, root_t: float, t_in: float) -> dict:
    """Extract the clone's lineage from a simulation ``res`` and its clone
    ancestor node. This is the clean, consistent path: the eval calls it with
    its OWN ``res`` so the lineage shares the exact trajectories of the recorded
    descendants (no re-simulation, no mismatch).

    Returns {"root_t", "ancestor_state", "segments", "ancestry"}:
      segments  (steps, 2) [time, state] arrays for every subtree node clipped
                to [.., t_in] -- the family tree from the common ancestor down to
                the t_in cousins (1 -> 2 -> 4 -> 8).
      ancestry  segments for the single lineage from the root down to the common
                ancestor -- the starting cell's history before the clone fans out.

    Trajectories are clipped to ``<= t_in`` (NOT to ``>= root_t``) so the
    ancestor's own pre-root segment is kept and the ancestry and family tree
    connect with no gap.
    """
    def clip(n):
        raw = res.tree.nodes[n].get("trajectory")
        if raw is None:
            return None
        arr = np.asarray(raw, dtype=np.float32)
        m = arr[:, 0] <= t_in + 1e-6
        return arr[m] if int(m.sum()) >= 2 else None

    segments = [s for s in (clip(n) for n in
                            (nx.descendants(res.tree, ancestor) | {ancestor}))
                if s is not None]
    ancestry = [s for s in (clip(n) for n in nx.ancestors(res.tree, ancestor))
                if s is not None]

    # ancestor state AT root_t (where the diamond sits), not at its birth.
    anc_traj = np.asarray(res.tree.nodes[ancestor]["trajectory"], dtype=np.float32)
    idx = int(np.argmin(np.abs(anc_traj[:, 0] - root_t)))
    return {"root_t": float(root_t), "ancestor_state": float(anc_traj[idx, 1]),
            "segments": segments, "ancestry": ancestry}


def clone_lineage(case: dict, sim_kw: dict) -> dict:
    """Re-simulate the case's sim and extract its clone lineage, for result pickles
    that lack it (the eval records it via :func:`record_clone_lineage`)."""
    seed = int(case["seed"])
    t_in = float(case["t_in"])
    anchors = set(case["n8_clone_anchors"])
    res = make_fate_bias(seed=seed, **sim_kw)
    snap_in, _ = res.snapshot(t_in)

    root_t = t_in - np.log2(len(anchors))
    ancestor = None
    for n in snap_in.nodes():
        if abs(snap_in.nodes[n].get("time", -1e9) - root_t) > 1e-3:
            continue
        leaves = {m for m in (nx.descendants(snap_in, n) | {n})
                  if snap_in.out_degree(m) == 0}
        if leaves == anchors:
            ancestor = n
            break
    if ancestor is None:
        raise RuntimeError(f"common ancestor not found for seed={seed}")
    return record_clone_lineage(res, ancestor, root_t, t_in)


def patch_pickle(pkl: Path, stem_glob: str) -> None:
    with open(pkl, "rb") as f:
        results = pickle.load(f)
    case = results.get("case")
    if case is None:
        print(f"  {pkl.name}: no case, skipped")
        return
    sim_kw = _sim_kw_for(float(case["sigma"]), stem_glob)
    lin = clone_lineage(case, sim_kw)
    case["clone_lineage"] = lin
    with open(pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"  {pkl.name}: added clone_lineage "
          f"(root_t={lin['root_t']:g}, {len(lin['segments'])} segments)")


def main() -> None:
    base = ROOT / "synthetic_evals" / "context"
    for name, stem_glob in PKL_TO_STEM.items():
        pkl = base / name
        if pkl.exists():
            patch_pickle(pkl, stem_glob)


if __name__ == "__main__":
    main()
