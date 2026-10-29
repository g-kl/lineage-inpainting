"""distil the grid's per-run eval artefacts into the numerical_results caches.

    python -m analysis.figures.grid_summary.compute_grid            # style.STEP
    python -m analysis.figures.grid_summary.compute_grid --step 5000

for every outputs/mouse_embryo_grid_bridge_unshared_* dir whose name decodes into the
grid (parse_dirname; graft / selfgraft / parameter-free-pool / LineageOT runs are not
part of it) and that has a checkpoint at --step, reads
  metrics_<cfg>_step{S}_adj_n{n}[_splitter].csv        coupling metrics per side
  clonal_distortion_<cfg>_step{S}.csv                  splitter clonal distortion
  marginal_distortion_<cfg>_step{S}.csv                splitter marginal distortion
  descendant_distortion_<cfg>_step{S}.csv              splitter descendant distortion
  marginal_pool_distortion_<cfg>_step{S}.csv           the imputing model's marginal distortion
  cell_type_map_<cfg>_step{S}_adj_n{n}_splitter.pkl    sampling-side transition graphs
and writes numerical_results/<step>/
  configs.csv        one row per run: cfg + its axes (bridge, pairing, pool, aligner)
  coverage.csv       runs scored per (eval kind, n) -- the sanity table, also printed
  metrics_wide.csv   one row per run: the axes + every metric column of style.METRICS
                     (sampling side) and style.IMPUTING_METRICS (`cpl_*`, pool side)
  similarity.csv     run x run cosine similarity of the sampling-side transition graphs

every aggregation is the main-text metrics row's (summary_qc/plot_metrics): clonal
and marginal = mean over the figure's seeded subtrees / adjacent-timepoint pairs,
germ-layer crossing = pooled over every link, linkage = mean of the distortion over
the two well-conditioned cuts, Hox = mean over the (cell type, gene) pairs, the
marginal cell-type TV = mean over the same adjacent-timepoint pairs as the marginal
distortion (its ``cell_type_tv`` column, eval_core.runner.LabelScorer), the
descendant distortion and its cell-type TV = mean over the figure's clonal subtrees
split from the root's coupled real cell (eval_core.runner.run_model_metrics); the
imputing model's marginal distortion and TV (`cpl_marginal`, `cpl_label_tv`) = the
same pool-back samples with the pool model applied once. the coupling side is read
at its canonical n (style.coupling_n; pool side n = 1). a clonal CSV without a
`method` column is ignored.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import re

import numpy as np
import pandas as pd

from analysis.figures.grid_summary import style as S

HASH_RE = re.compile(rf"^{re.escape(S.GRID_PREFIX)}(.+)_[a-f0-9]{{32}}$")
DIR_BRIDGE = {"fullen": "one_step", "cfm": "cfm", "vpsde": "vpsde"}
PAIRINGS_KNOWN = ["idnn_leaf", "id_sg", "nn_sg", "id", "nn"]     # longest first
# suffix after `<bridge>_<pairing>_nosa_std` -> (pool, aligner). the two Tree Tx pools
# with the full aligner, plus the aligner-feature ablation on the main-text run. the
# Flat Tx pool (`_ft_none`), the parameter-free pools (`_meanpool`, `_fgwpool`,
# `_w2pool`) and the LineageOT pairing (`_ot_`) are not part of the grid.
SUFFIXES = {
    "none": ("dtfm", "full"), "nondelta": ("nondelta", "full"),
    "alignmarg": ("dtfm", "marg"), "alignpath": ("dtfm", "marg_path"),
    "alignca": ("dtfm", "marg_ca"),
}
AXES = ["bridge", "pairing", "pool", "aligner"]
KEY = ["cfg", "eval", "n", "metric", "source", "target", "method"]
# linkage distortion is averaged over the two well-conditioned cuts only; at the
# earlier cuts the real-real linkage baseline is near zero (main-text convention).
LINKAGE_TAGS = ("t9.5_to_t9.0", "t9.0_to_t8.5")


def parse_dirname(d):
    """decode a run-dir name into its axes; None if it is not a grid run."""
    m = HASH_RE.match(d)
    if not m:
        return None
    rest = m.group(1)
    for tag, bridge in DIR_BRIDGE.items():
        if rest.startswith(tag + "_"):
            rest = rest[len(tag) + 1:]
            break
    else:
        return None
    pairing = next((p for p in PAIRINGS_KNOWN if rest.startswith(p + "_")), None)
    if pairing is None:
        return None
    rest = rest[len(pairing) + 1:]
    if not rest.startswith("nosa_std"):
        return None
    spec = SUFFIXES.get(rest[len("nosa_std"):].lstrip("_"))
    if spec is None:
        return None
    return {"cfg": d, "bridge": bridge, "pairing": pairing, "pool": spec[0],
            "aligner": spec[1]}


def grid_configs(step):
    rows = [parse_dirname(d) for d in sorted(os.listdir(S.OUTPUTS))
            if d.startswith(S.GRID_PREFIX)]
    rows = [r for r in rows if r is not None
            and os.path.exists(os.path.join(S.OUTPUTS, r["cfg"], f"pool_ema_step{step}.pt"))]
    return pd.DataFrame(rows, columns=["cfg", *AXES])


# --- loaders: one tidy row per (cfg, eval, n, metric, source, target, method) -----

def _read(f):
    try:
        if os.path.getsize(f) < 4:
            return None
        return pd.read_csv(f)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def _files(d, pattern):
    return sorted(glob.glob(os.path.join(d, pattern)))


def load_coupling(d, cfg, step, splitter, n_ok):
    """the coupling metrics CSV of one side at the n values in ``n_ok``."""
    suffix = "_splitter" if splitter else ""
    kind = "cplspl" if splitter else "cpl"
    rows = []
    for f in _files(d, f"metrics_*_step{step}_adj_n*{suffix}.csv"):
        if not splitter and "_splitter" in os.path.basename(f):
            continue
        n = int(re.search(r"_adj_n(\d+)(?:_splitter)?\.csv$", f).group(1))
        df = _read(f)
        if n not in n_ok or df is None or "tree" not in df.columns:
            continue
        for metric, tag, value in zip(df["metric"], df["tree"], df["value"]):
            rows.append((cfg, kind, n, metric, tag if pd.notna(tag) else np.nan,
                         np.nan, "na", float(value)))
    return rows


def load_distortion(d, cfg, step, which):
    """the figure's model rows of ``{which}_distortion_<cfg>_step{S}.csv``."""
    rows = []
    for f in _files(d, f"{which}_distortion_*_step{step}.csv"):
        df = _read(f)
        if df is None or "method" not in df.columns:      # not the figure's definition
            continue
        df = df[df["method"] == "model"]
        keys = ("t_src", "t_tgt") if which.startswith("marginal") else ("leaf", "tree")
        metrics = [m for m in ("energy_distance", "cell_type_tv") if m in df.columns]
        for metric in metrics:
            for a, b, value in zip(df[keys[0]], df[keys[1]], df[metric]):
                rows.append((cfg, which, 1, metric, a, b, "model", float(value)))
    return rows


def load_all(configs, step):
    rows = []
    for r in configs.itertuples(index=False):
        d = os.path.join(S.OUTPUTS, r.cfg)
        rows += load_coupling(d, r.cfg, step, splitter=False, n_ok=(1,))
        rows += load_coupling(d, r.cfg, step, splitter=True, n_ok=(S.coupling_n(r.bridge),))
        rows += load_distortion(d, r.cfg, step, "clonal")
        rows += load_distortion(d, r.cfg, step, "marginal")
        rows += load_distortion(d, r.cfg, step, "descendant")
        rows += load_distortion(d, r.cfg, step, "marginal_pool")
    data = pd.DataFrame(rows, columns=[*KEY, "value"])
    return data.drop_duplicates(subset=KEY, keep="last")


# --- the wide table ---------------------------------------------------------------

def _agg(sub, agg):
    """per config, the ``agg`` of ``value`` over the rows at that config's largest n."""
    if sub.empty:
        return pd.Series(dtype=float, name="value").rename_axis("cfg")
    sub = sub[sub["n"] == sub.groupby("cfg")["n"].transform("max")]
    return sub.groupby("cfg")["value"].agg(agg)


def _assignment_columns(side, prefix):
    """the four assignment metrics of one coupling side, aggregated as the figure does."""
    tagged = side["source"].notna()
    return [
        _agg(side[(side["metric"] == "lineage_oscillation_fraction") & ~tagged], "first")
        .rename(f"{prefix}osc"),
        _agg(side[(side["metric"] == "linkage_distortion")
                  & side["source"].isin(LINKAGE_TAGS)], "mean").rename(f"{prefix}linkage"),
        _agg(side[(side["metric"] == "germ_layer_crossing") & ~tagged], "first")
        .rename(f"{prefix}germ"),
        _agg(side[(side["metric"] == "hox_distortion")], "mean").rename(f"{prefix}hox"),
    ]


def metrics_wide(data, configs):
    dist = lambda ev, metric: data[(data["eval"] == ev) & (data["metric"] == metric)]
    parts = [_agg(dist("clonal", "energy_distance"), "mean").rename("clonal"),
             _agg(dist("marginal", "energy_distance"), "mean").rename("marginal"),
             _agg(dist("marginal", "cell_type_tv"), "mean").rename("label_tv")]
    parts += _assignment_columns(data[data["eval"] == "cplspl"], "")
    parts += [_agg(dist("marginal_pool", "energy_distance"), "mean").rename("cpl_marginal"),
              _agg(dist("marginal_pool", "cell_type_tv"), "mean").rename("cpl_label_tv")]
    parts += _assignment_columns(data[data["eval"] == "cpl"], "cpl_")
    parts += [_agg(dist("descendant", "energy_distance"), "mean").rename("descendant"),
              _agg(dist("descendant", "cell_type_tv"), "mean").rename("descendant_tv")]
    wide = pd.concat(parts, axis=1).reset_index()
    cols = [*S.METRICS, *S.IMPUTING_METRICS]
    return configs.merge(wide, on="cfg", how="left")[["cfg", *AXES, *cols]]


# --- cell-type transition-graph similarity -----------------------------------------

def similarity(configs, step):
    """run x run cosine similarity of the sampling-side transition graphs over the
    heatmap's runs (style.in_heatmap): the full-aligner runs in canonical (bridge,
    pool, pairing) order, then the aligner ablation in its ladder order.

    edge weight = the mean of the edge's in- and out-composition, so the cosine sees
    both which (type, t) -> (type, t + dt) transitions survived and their mass.
    """
    keep = configs[[S.in_heatmap(r.bridge, r.pool, r.pairing, r.aligner)
                    for r in configs.itertuples(index=False)]]
    keep = pd.concat([S.sort_rows(keep[keep["aligner"] == "full"], ["bridge", "pool", "pairing"]),
                      S.sort_rows(keep[keep["aligner"] != "full"], ["aligner"])])
    graphs = {}
    for r in keep.itertuples(index=False):
        p = os.path.join(S.OUTPUTS, r.cfg,
                         f"cell_type_map_{r.cfg}_step{step}_adj_n{S.coupling_n(r.bridge)}_splitter.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                graphs[r.cfg] = pickle.load(f)
    cfgs = [c for c in keep["cfg"] if c in graphs]
    edges = sorted({e for G in graphs.values() for e in G.edges()})
    idx = {e: i for i, e in enumerate(edges)}
    M = np.zeros((len(cfgs), len(edges)))
    for i, cfg in enumerate(cfgs):
        for s, t, d in graphs[cfg].edges(data=True):
            M[i, idx[(s, t)]] = 0.5 * (d.get("in_composition", 0) + d.get("out_composition", 0))
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1
    Mn = M / norms
    return pd.DataFrame(Mn @ Mn.T, index=cfgs, columns=cfgs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, default=S.STEP)
    args = ap.parse_args()
    nr = S.nr_dir(args.step)

    configs = grid_configs(args.step)
    hero = configs[configs["cfg"] == S.HERO_CFG]
    assert len(hero) == 1 and all(hero.iloc[0][k] == v for k, v in S.HERO.items()), \
        f"the main-text hero {S.HERO_CFG} is not in the grid at step {args.step}"
    print(f"{len(configs)} grid runs with a step-{args.step} checkpoint")
    for a in AXES:
        print(f"  {a:8s}", dict(configs[a].value_counts().sort_index()))
    configs.to_csv(os.path.join(nr, "configs.csv"), index=False)

    data = load_all(configs, args.step)
    coverage = (data.groupby(["eval", "n"])["cfg"].nunique()
                .unstack("n", fill_value=0).reset_index())
    print("runs scored per (eval, n):")
    print(coverage.to_string(index=False))
    coverage.to_csv(os.path.join(nr, "coverage.csv"), index=False)

    wide = metrics_wide(data, configs)
    wide.to_csv(os.path.join(nr, "metrics_wide.csv"), index=False)
    missing = wide[[*S.METRICS, *S.IMPUTING_METRICS]].isna().sum()
    print("missing cells per metric column:", dict(missing[missing > 0]) or "none")

    sim = similarity(configs, args.step)
    sim.to_csv(os.path.join(nr, "similarity.csv"))
    print(f"similarity: {len(sim)} runs")
    print(f"wrote caches -> {nr}")


if __name__ == "__main__":
    main()
