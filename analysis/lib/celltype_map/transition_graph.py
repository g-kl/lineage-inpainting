"""Cell-type transition graph via continuous-space pointwise mutual information.

Pipeline:
  1. ``collect_pairs``  -- walk every tree backward and emit one (parent, child)
     pair per adjacent-timepoint transition, as indices into the state matrix.
  2. ``pmi_knn``        -- per-pair PMI = log[ p(x,y) / (p(x) p(y)) ] estimated
     from three FAISS k-NN densities (joint, x-marginal, y-marginal). This is
     the continuous-space observed/expected ratio; >0 means the parent and
     child states co-occur more than chance, <0 means the assignment is no
     better than coincidence (a likely-spurious transition).
  3. keep pairs above a PMI cutoff, label each endpoint at the resolution
     chosen for its timepoint, and apply a simple in/out composition cutoff on
     the discrete (cell_type, time) edges (``_build_from_buckets``).
"""
from collections import defaultdict

import numpy as np
import networkx as nx
from scipy.special import digamma, gammaln


def quantize_time(x, dt=0.5):
    """Snap a continuous time to the dt grid, clamped at the 7.5 floor."""
    return max(7.5, round(round(x / dt) * dt, 6))


# --------------------------------------------------------------------------
# discrete-graph assembly (simple composition cutoff)
# --------------------------------------------------------------------------

def _build_from_buckets(buckets, observed_counts, min_composition, min_count,
                        composition='max'):
    """Digraph from per-target source tallies, kept by count + composition.

    Keep an edge source -> target when it carries >= ``min_count`` cells and is
    at least ``min_composition`` of either the target's incoming cells or the
    source's outgoing cells (the out view keeps edges from small types that
    never dominate a larger target). With ``composition='sum'`` the two shares
    are added instead, so an edge moderate on both sides passes while one held
    up by a single view needs a larger share.
    """
    in_total = defaultdict(int)
    out_total = defaultdict(int)
    for target, sources in buckets.items():
        for source, c in sources.items():
            in_total[target] += c
            out_total[source] += c

    G = nx.DiGraph()
    for target, sources in buckets.items():
        for source, c in sources.items():
            in_comp = c / in_total[target]
            out_comp = c / out_total[source]
            comp = (in_comp + out_comp if composition == 'sum'
                    else max(in_comp, out_comp))
            if c >= min_count and comp >= min_composition:
                G.add_edge(source, target, count=int(c),
                           in_composition=float(in_comp),
                           out_composition=float(out_comp))
    for node, count in observed_counts.items():
        G.add_node(node, observed_count=int(count))
    return G


# --------------------------------------------------------------------------
# pair extraction
# --------------------------------------------------------------------------

def _resolve_nn_cutoff(assignments, nn_cutoff, nn_quantile):
    if nn_cutoff is not None:
        return nn_cutoff
    ds = np.fromiter(
        (d for x in assignments['nn_distances'].values() for d in x.values()),
        dtype=float,
    )
    return float(np.quantile(ds, nn_quantile))


def collect_pairs(trees, assignments, name_to_idx, labelable,
                  time_key='time', dt=0.5, nn_cutoff=None, nn_quantile=0.8,
                  tol=1e-8):
    """One (parent, child) record per adjacent-timepoint transition.

    Returns parallel arrays ``(pi, ci, qtp, qtc)``: parent / child indices into
    the state matrix (via ``name_to_idx``) and their quantized times. Internal
    nodes are represented by their assigned observed cell (dropped if the
    assignment distance exceeds ``nn_cutoff``); leaves by themselves. Endpoints
    must be in ``labelable`` (a name -> ... mapping) to be emitted.
    """
    nn_cutoff = _resolve_nn_cutoff(assignments, nn_cutoff, nn_quantile)

    pi, ci, qtp, qtc = [], [], [], []
    for tree_name, g in trees.items():
        amap = assignments[tree_name]
        dmap = assignments['nn_distances'][tree_name]

        roots = [n for n in g if g.in_degree(n) == 0]
        stack = [(r, None) for r in roots]  # last = (idx, qt)

        while stack:
            n, last = stack.pop()
            qt = quantize_time(g.nodes[n][time_key], dt)

            if g.out_degree(n) == 0:
                obs = n
            else:
                obs = amap.get(n)
                if obs is None or dmap[n] > nn_cutoff:
                    obs = None

            cur = None
            if obs is not None and obs in labelable:
                cur = (name_to_idx[obs], qt)

            if cur is not None:
                if last is not None and abs(cur[1] - last[1] - dt) < tol:
                    pi.append(last[0]); ci.append(cur[0])
                    qtp.append(last[1]); qtc.append(cur[1])
                last = cur

            for c in g.successors(n):
                stack.append((c, last))

    return (np.asarray(pi, dtype=np.int64), np.asarray(ci, dtype=np.int64),
            np.asarray(qtp, dtype=float), np.asarray(qtc, dtype=float))


# --------------------------------------------------------------------------
# pointwise mutual information (continuous O/E)
# --------------------------------------------------------------------------

def _knn_rk2(Z, k, use_gpu=True):
    """Squared L2 distance to the k-th nearest neighbour (self excluded)."""
    import faiss
    Z = np.ascontiguousarray(Z, dtype=np.float32)
    n, d = Z.shape
    k = min(k, max(1, n - 1))
    index = faiss.IndexFlatL2(d)
    if use_gpu and faiss.get_num_gpus() > 0:
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
    index.add(Z)
    D, _ = index.search(Z, k + 1)
    return np.maximum(D[:, k], 1e-30)


def _log_ball_volume(D):
    """log volume of the unit L2 ball in R^D."""
    return (D / 2.0) * np.log(np.pi) - gammaln(D / 2.0 + 1.0)


def pmi_knn(X, Y, k=10, use_gpu=True, jitter=0.0, add_const=True):
    """Per-pair PMI = log p(x,y) - log p(x) - log p(y) via k-NN densities.

    Each log density is the Kozachenko-Leonenko estimate
    ``log p(z) = -D log r_k + digamma(k) - digamma(N) - log c_D``; the ratio
    leaves ``0.5 d (log r2_x + log r2_y) - d log r2_xy`` plus a global constant
    (digamma terms cancel to ``digamma(N)-digamma(k)``, volumes to
    ``2 log c_d - log c_2d``). With ``add_const`` the constant is included so
    PMI ~ 0 marks independence; otherwise only relative ordering is meaningful.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    if jitter > 0:
        rng = np.random.default_rng(0)
        X = X + rng.normal(0, jitter, X.shape).astype(np.float32)
        Y = Y + rng.normal(0, jitter, Y.shape).astype(np.float32)

    n, d = X.shape
    rx2 = _knn_rk2(X, k, use_gpu)
    ry2 = _knn_rk2(Y, k, use_gpu)
    rxy2 = _knn_rk2(np.concatenate([X, Y], axis=1), k, use_gpu)

    pmi = 0.5 * d * (np.log(rx2) + np.log(ry2)) - d * np.log(rxy2)
    if add_const:
        const = (digamma(n) - digamma(k)
                 + 2 * _log_ball_volume(d) - _log_ball_volume(2 * d))
        pmi = pmi + const
    return pmi


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def compute_pairs_pmi(trees, assignments, dataset, name_to_idx, name_to_labels,
                      time_key='time', dt=0.5, nn_cutoff=None, nn_quantile=0.8,
                      embedding_key='X_scvi', k=10, n_max=500_000, seed=0,
                      jitter=0.0, use_gpu=True):
    """Collect transition pairs (subsampled to ``n_max``; None keeps every pair)
    and score their PMI.

    Returns a dict with parent/child indices, quantized times, and per-pair
    PMI -- the expensive part. Thresholding + relabelling are cheap, so a
    cutoff scan reuses this once.
    """
    nn_cutoff = _resolve_nn_cutoff(assignments, nn_cutoff, nn_quantile)
    pi, ci, qtp, qtc = collect_pairs(
        trees, assignments, name_to_idx, name_to_labels,
        time_key=time_key, dt=dt, nn_cutoff=nn_cutoff)

    n = len(pi)
    if n_max is not None and n > n_max:
        sel = np.random.default_rng(seed).choice(n, n_max, replace=False)
        pi, ci, qtp, qtc = pi[sel], ci[sel], qtp[sel], qtc[sel]

    emb = dataset.states.obsm[embedding_key]
    X = np.asarray(emb[pi], dtype=np.float32)
    Y = np.asarray(emb[ci], dtype=np.float32)
    pmi = pmi_knn(X, Y, k=k, use_gpu=use_gpu, jitter=jitter)

    return dict(pi=pi, ci=ci, qtp=qtp, qtc=qtc, pmi=pmi, n_total=n)


def _observed_counts(trees, name_to_labels, level_for, time_key='time', dt=0.5):
    counts = defaultdict(int)
    for g in trees.values():
        for n in g.nodes():
            if g.out_degree(n) != 0:
                continue
            labs = name_to_labels.get(n)
            if labs is None:
                continue
            tval = g.nodes[n].get(time_key)
            if tval is None:
                continue
            qt = quantize_time(tval, dt)
            lab = labs.get(level_for(qt))
            if lab is not None:
                counts[(lab, qt)] += 1
    return counts


def graph_from_pairs(pp, trees, dataset, name_to_labels, resolution_by_time,
                     default_level='cell_type', pmi_cutoff=0.0, pmi_quantile=None,
                     min_composition=0.05, min_count=5, time_key='time', dt=0.5,
                     composition='max'):
    """Threshold pairs by PMI, relabel by per-time resolution, build the graph.

    ``pmi_quantile`` (if given) sets the cutoff at that quantile of the PMI
    distribution -- more robust than an absolute ``pmi_cutoff`` given the noisy
    high-dimensional estimate. Resolution is a ``{qtime: level}`` dict; any
    timepoint not listed falls back to ``default_level``. ``composition`` is
    the edge rule's ``max`` or ``sum`` of the in / out shares (``_build_from_buckets``).
    """
    pmi = pp['pmi']
    cutoff = (np.quantile(pmi, pmi_quantile) if pmi_quantile is not None
              else pmi_cutoff)
    keep = pmi >= cutoff
    pi, ci = pp['pi'][keep], pp['ci'][keep]
    qtp, qtc = pp['qtp'][keep], pp['qtc'][keep]

    names = dataset.states.obs_names.to_numpy()

    def level_for(qt):
        return resolution_by_time.get(qt, default_level)

    buckets = defaultdict(lambda: defaultdict(int))
    for a, b, ta, tb in zip(pi, ci, qtp, qtc):
        la = name_to_labels[names[a]][level_for(ta)]
        lb = name_to_labels[names[b]][level_for(tb)]
        buckets[(lb, tb)][(la, ta)] += 1

    observed = _observed_counts(trees, name_to_labels, level_for,
                                time_key=time_key, dt=dt)
    return _build_from_buckets(buckets, observed, min_composition, min_count,
                               composition=composition)
