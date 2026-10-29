"""Align-loss feature dissociation on the fate-bias (context) process.

Shows, over *mixtures of distributions* scored by the slice-aligner energy
distance, which lineage feature is "operative" under three perturbations:

  fate mixture            real fate-1 vs fate-2 clones (the signal we care about)
  sibling-scramble        regroup siblings randomly -> breaks ONLY common-ancestor
  trajectory-decorrelate  resample each lineage's intermediate states from the
                          per-time pool (keep t=9 + ancestor) -> breaks ONLY path

The marginal (single-cell state) is blind to both scrambles by construction, so
it stays flat -- the point being that the align loss reads lineage *structure*
the marginal cannot. Renders a full 3x3 (scenario x feature) heatmap and a
compact one-row scatter for the paper. Run:  python synthetic_evals/aligner_dissociation.py
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from synthetic_evals.core.hydra_io import compose_cfg
from synthetic_evals import viz
from datasets.synthetic import make_fate_bias

PRIME = [5.0, 6.0, 7.0, 8.0, 9.0]
K = 4
SEEDS = list(range(8))
FRACS = np.linspace(0, 1, 6)
N_MIX, R = 100, 4
RNG = np.random.default_rng(0)
ED = viz.energy_distance
FEATS = ["marginal", "path", "ca"]
FEAT_TITLE = {"marginal": "marginal", "path": "path", "ca": "common ancestor"}
OUT = ROOT / "synthetic_evals" / "figures" / "aligner_dissociation"


# --------------------------------------------------------------------------
#  Data: fate-labelled clones (ancestor@7 + K descendants, priming trajectories)
# --------------------------------------------------------------------------

def collect_clones():
    ds = compose_cfg("synbench_fb_unshared_vpsde_idnn_leaf", "fate_bias_sig02").dataset
    KW = dict(n_generations=int(ds.n_generations), split_gen=int(ds.split_gen),
              n_primed_generations=int(ds.n_primed_generations), sigma=float(ds.sigma),
              theta=float(ds.theta), mu_1_primed=float(ds.mu_1_primed),
              mu_2_primed=float(ds.mu_2_primed), mu_1=float(ds.mu_1), mu_2=float(ds.mu_2),
              fate_prob=float(ds.fate_prob), branch_interval=float(ds.branch_interval),
              steps_per_interval=int(ds.steps_per_interval))
    ist = lambda r, n, t: float(np.asarray(r._interp_state(n, float(t))).reshape(-1)[0])
    fat = lambda r, n: 1 if r.tree.nodes[n]["params"].get("mu", 0.0) > 0 else 0
    bir = lambda r, n: int(round(r.tree.nodes[n]["time"]))
    clones = []
    for s in SEEDS:
        res = make_fate_bias(seed=s, **KW)
        d9 = set(res._alive_at(9.0)); bb = {}
        for n in res.tree.nodes():
            bb.setdefault(bir(res, n), []).append(n)
        def traj(leaf):
            ch, cur = {}, leaf
            while cur is not None:
                ch[bir(res, cur)] = cur
                ps = list(res.tree.predecessors(cur)); cur = ps[0] if ps else None
            return [ist(res, ch.get(int(t)) or ch.get(int(t) - 1), t) for t in PRIME]
        for a in bb.get(6, []):
            d = [n for n in nx.descendants(res.tree, a) if n in d9]
            if len(d) == K:
                clones.append({"anc7": ist(res, a, 7.0), "trajs": [traj(n) for n in d],
                               "fate": fat(res, a)})
    return clones


def reps(cl):
    return {"marginal": np.array([t[-1] for c in cl for t in c["trajs"]])[:, None],
            "path": np.array([t for c in cl for t in c["trajs"]]),
            "ca": np.array([[c["anc7"]] + sorted(t[-1] for t in c["trajs"]) for c in cl])}


def _scrambles(clones, trajs_all):
    def sib(cl):
        pool = [t for c in cl for t in c["trajs"]]; ancs = [c["anc7"] for c in cl]
        idx = RNG.permutation(len(pool)); out = []
        for i in range(0, len(idx) - K + 1, K):
            out.append({"anc7": ancs[RNG.integers(len(ancs))],
                        "trajs": [pool[j] for j in idx[i:i + K]], "fate": -1})
        return out
    def decorr(cl):
        cols = {ti: trajs_all[:, ti] for ti in range(len(PRIME) - 1)}
        return [{"anc7": c["anc7"], "fate": c["fate"],
                 "trajs": [[float(RNG.choice(cols[ti])) for ti in range(len(PRIME) - 1)]
                           + [t[-1]] for t in c["trajs"]]} for c in cl]
    return sib, decorr


def scenarios(clones):
    trajs_all = np.array([t for c in clones for t in c["trajs"]])
    sib, decorr = _scrambles(clones, trajs_all)
    f1 = [c for c in clones if c["fate"] == 1]; f0 = [c for c in clones if c["fate"] == 0]
    return {"fate mixture": (f0, f1),
            "sibling-scramble": (clones, sib(clones)),
            "trajectory-decorrelate": (clones, decorr(clones))}


def _mix(A, B, f):
    nb = int(round(f * N_MIX))
    a = [A[i] for i in RNG.choice(len(A), N_MIX - nb, replace=False)]
    b = [B[i] for i in RNG.choice(len(B), nb, replace=False)]
    return a + b


def compute(clones):
    """Per scenario: full pairwise grids (3x3 fig) and ED-vs-fraction curves (compact)."""
    SCN = scenarios(clones)
    grids, curves = {}, {}
    for name, (A, B) in SCN.items():
        G = {f: np.zeros((len(FRACS), len(FRACS))) for f in FEATS}
        cur = {f: np.zeros(len(FRACS)) for f in FEATS}
        for _ in range(R):
            mr = [reps(_mix(A, B, f)) for f in FRACS]
            for i in range(len(FRACS)):
                for j in range(len(FRACS)):
                    for f in FEATS:
                        G[f][i, j] += ED(mr[i][f], mr[j][f])
            for i in range(len(FRACS)):                  # ED from the f=0 reference
                for f in FEATS:
                    cur[f][i] += ED(mr[0][f], mr[i][f])
        for f in FEATS:
            G[f] /= R; cur[f] /= R
        grids[name] = G; curves[name] = cur
    return grids, curves


# --------------------------------------------------------------------------
#  Figures
# --------------------------------------------------------------------------

FEAT_STYLE = {"marginal": (viz.PRESPLIT, "o"), "path": (viz.MODEL, "s"),
              "ca": (viz.GS, "^")}


def render_full(grids):
    viz.apply_rc()
    labs = [f"{f:.0%}" for f in FRACS]
    fig, axes = plt.subplots(3, 3, figsize=(7.6, 7.4), constrained_layout=True)
    for r, name in enumerate(grids):
        vmax = max(grids[name][f].max() for f in FEATS) or 1.0
        norm = Normalize(0, vmax)
        for c, f in enumerate(FEATS):
            ax = axes[r, c]
            im = ax.imshow(grids[name][f], origin="lower", cmap=viz.SEQ_CMAP, norm=norm,
                           aspect="equal")
            ax.set_xticks(range(len(FRACS))); ax.set_yticks(range(len(FRACS)))
            ax.set_xticklabels(labs if r == 2 else [], fontsize=5)
            ax.set_yticklabels(labs if c == 0 else [], fontsize=5)
            ax.tick_params(length=0)
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color("0.8")
            if r == 0: ax.set_title(FEAT_TITLE[f], fontsize=viz.FONT_TITLE)
            if c == 0: ax.set_ylabel(f"{name}\nmixture A", fontsize=viz.FONT_LABEL)
            if r == 2: ax.set_xlabel("mixture B", fontsize=viz.FONT_LABEL)
        fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.025, pad=0.01)
    fig.suptitle("align-loss energy distance over mixtures (full grid)",
                 fontsize=viz.FONT_TITLE + 1, y=1.02)
    _save(fig, "aligner_dissociation")


def render_compact(curves):
    """One row: ED vs perturbation fraction, marginal/path/CA per scenario."""
    viz.apply_rc()
    names = list(curves)
    fig, axes = plt.subplots(1, len(names), figsize=(7.2, 2.25), constrained_layout=True)
    xlab = {"fate mixture": "fraction fate-1",
            "sibling-scramble": "fraction siblings scrambled",
            "trajectory-decorrelate": "fraction trajectories decorrelated"}
    for ax, name in zip(axes, names):
        for f in FEATS:
            col, mk = FEAT_STYLE[f]
            ax.plot(FRACS, curves[name][f], color=col, marker=mk, ms=4.0, lw=1.4)
        ax.set_title(name, fontsize=viz.FONT_TITLE)
        ax.set_xlabel(xlab[name], fontsize=viz.FONT_LABEL)
        ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "", "1"])
        ax.set_ylim(bottom=0); ax.margins(x=0.04)
    axes[0].set_ylabel("align-loss\nenergy distance", fontsize=viz.FONT_LABEL)
    handles = [Line2D([], [], color=FEAT_STYLE[f][0], marker=FEAT_STYLE[f][1], ms=4, lw=1.4)
               for f in FEATS]
    fig.legend(handles, [FEAT_TITLE[f] for f in FEATS], loc="upper center",
               ncol=3, bbox_to_anchor=(0.5, 1.16), frameon=False)
    _save(fig, "aligner_dissociation_compact")


def render_modality(curves):
    """Tree-consistent dissociation: real fate-1 vs fate-2 clones, the feature each
    real observation modality affords. Snapshot lineage tracing (a clone at one
    time) affords common-ancestor; live single-cell imaging (one lineage over
    time) affords path; both beat the marginal, which barely separates the fates."""
    viz.apply_rc()
    c = curves["fate mixture"]
    grey = FEAT_STYLE["marginal"][0]; teal = FEAT_STYLE["path"][0]; pur = FEAT_STYLE["ca"][0]
    ymax = max(c["path"].max(), c["ca"].max()) * 1.08
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(FRACS, c["marginal"], color=grey, marker="o", ms=4, lw=1.4, label="marginal")
    ax.plot(FRACS, c["ca"], color=pur, marker="^", ms=4, lw=1.4, label="common ancestor")
    ax.set_title("snapshot lineage tracing\n(clone, one time)", fontsize=viz.FONT_TITLE)
    ax.legend(fontsize=viz.FONT_LEGEND, loc="upper left")

    ax = axes[1]
    ax.plot(FRACS, c["marginal"], color=grey, marker="o", ms=4, lw=1.4, label="marginal")
    ax.plot(FRACS, c["path"], color=teal, marker="s", ms=4, lw=1.4, label="path")
    ax.set_title("live single-cell imaging\n(one lineage, trajectory)", fontsize=viz.FONT_TITLE)
    ax.legend(fontsize=viz.FONT_LEGEND, loc="upper left")

    for ax in axes:
        ax.set_xlabel("fraction fate-1", fontsize=viz.FONT_LABEL)
        ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "", "1"])
        ax.set_ylim(0, ymax); ax.margins(x=0.04)
    axes[1].tick_params(labelleft=False)
    axes[0].set_ylabel("align-loss\nenergy distance", fontsize=viz.FONT_LABEL)
    _save(fig, "aligner_dissociation_modality")


def _save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)


def main():
    cache = OUT / ".grids.pkl"; OUT.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        grids, curves = pickle.load(open(cache, "rb")); print("loaded cache")
    else:
        clones = collect_clones(); print(f"clones={len(clones)}")
        grids, curves = compute(clones); pickle.dump((grids, curves), open(cache, "wb"))
    render_full(grids)            # SI: 3x3 scenarios x features (incl. synthetic scrambles)
    render_compact(curves)        # SI: one-row scramble curves
    render_modality(curves)       # MAIN: tree-consistent modality dissociation
    print("wrote", OUT)


if __name__ == "__main__":
    main()
