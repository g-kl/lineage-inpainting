import bisect
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_MAX_NODES = 128

_TENSOR_FIELDS = (
    "adj", "node_times", "mrca_times", "mask", "leaf_mask",
    "depth", "child_index", "parent_idx", "root_idx", "n_nodes",
    "leaf_states", "observed_mask", "observed_states",
)


@dataclass
class TreeBatch:
    """padded batch of B trees, each with at most N nodes.

    per-node tensors have shape (B, N, ...) and are zero-padded
    (or -1 for index tensors) beyond each tree's real node count.
    """

    adj: torch.Tensor               # (B, N, N) float, adj[b,i,j]=1 means i->j
    node_times: torch.Tensor        # (B, N)
    mrca_times: torch.Tensor        # (B, N, N)
    mask: torch.BoolTensor          # (B, N) true for real nodes
    leaf_mask: torch.BoolTensor     # (B, N) true for leaves
    depth: torch.LongTensor         # (B, N) depth from root (root=0)
    child_index: torch.LongTensor   # (B, N) arbitrary sibling slot, 0-indexed
    parent_idx: torch.LongTensor    # (B, N) parent index, -1 for root/pad
    root_idx: torch.LongTensor      # (B,) root node index per tree
    n_nodes: torch.LongTensor       # (B,) real node count per tree
    max_depth: int
    node_names: list[list[str]]     # per-tree node names in topo order
    leaf_states: torch.Tensor       # (B, N, dim) leaf embeddings, 0 elsewhere
    observed_mask: torch.BoolTensor # (B, N) true for supplied real states
    observed_states: torch.Tensor   # (B, N, dim) observed embeddings, 0 elsewhere
    subtree_records: list[dict]     # original subtree dicts for aligner

    def to(self, device):
        """move all tensors to device, return self."""
        for f in _TENSOR_FIELDS:
            setattr(self, f, getattr(self, f).to(device))
        return self

    def to_pooled_tree_list(self, states):
        """unpack (B, N, dim) states into per-tree dicts for aligner."""
        result = []
        for i in range(len(self.node_names)):
            ni = self.n_nodes[i].item()
            names = self.node_names[i][:ni]
            state_dict = {names[j]: states[i, j] for j in range(ni)}
            leaf_dict = {
                names[j]: states[i, j]
                for j in range(ni)
                if self.leaf_mask[i, j]
            }
            observed_nodes = [
                names[j]
                for j in range(ni)
                if self.observed_mask[i, j]
            ]
            observed_state_dict = {
                names[j]: self.observed_states[i, j]
                for j in range(ni)
                if self.observed_mask[i, j]
            }
            result.append({
                "graph": self.subtree_records[i]["graph"],
                "states": state_dict,
                "root": self.subtree_records[i]["root"],
                "leaf_states": leaf_dict,
                "observed_nodes": observed_nodes,
                "observed_states": observed_state_dict,
            })
        return result


# slice / cross-section primitive
#
# The "active slice" at a time t is the set of branches (edges) the slice would
# cut -- every edge (v, w) with tau_v <= t < tau_w -- each represented by an
# adjacent node: 'up' takes the node immediately above the cut (the parent /
# floor), 'down' the node immediately below (the child / ceil). This is the
# single operation both the slice-based aligner features and the splitter's
# anchor sampler are defined on, so it lives with TreeBatch.


def max_child_time(batch: "TreeBatch") -> torch.Tensor:
    """``(B, N)`` max child node-time per node (``-inf`` for leaves)."""
    B, N = batch.mask.shape
    nt = batch.node_times
    out = torch.full((B, N), float("-inf"), device=nt.device, dtype=nt.dtype)
    has_parent = batch.mask & (batch.parent_idx >= 0)
    src = torch.where(has_parent, nt, torch.full_like(nt, float("-inf")))
    out.scatter_reduce_(
        1, batch.parent_idx.clamp(min=0), src, reduce="amax", include_self=True,
    )
    return out


def cross_section(
    batch: "TreeBatch",
    t,
    direction: str = "up",
    *,
    max_child: torch.Tensor | None = None,
    parent_time: torch.Tensor | None = None,
) -> torch.BoolTensor:
    """Nodes representing the branches the slice at time ``t`` would cut.

    ``t`` is a scalar (same slice time for every tree) or a ``(B,)`` tensor (a
    per-tree time, broadcast over nodes). A node ``v`` is included iff it sits
    immediately on one side of a cut branch:

    - ``'up'``   (floor): ``tau_v <= t`` and ``v`` has a child after ``t`` (or
      is a leaf) -- the node just *above* a crossing branch.
    - ``'down'`` (ceil):  ``tau_v >= t`` and ``v``'s parent is before ``t`` (or
      ``v`` is the root) -- the node just *below* a crossing branch.

    ``max_child`` / ``parent_time`` may be passed precomputed (see
    :func:`max_child_time`) to avoid recompute when slicing at many times.
    """
    nt = batch.node_times
    if torch.is_tensor(t) and t.dim() >= 1:
        t = t.view(-1, 1)                    # (B, 1): per-tree time over nodes
    if direction == "up":
        if max_child is None:
            max_child = max_child_time(batch)
        is_leaf = batch.mask & batch.leaf_mask
        return batch.mask & (nt <= t) & (is_leaf | (max_child > t))
    if direction == "down":
        if parent_time is None:
            parent_time = nt.gather(1, batch.parent_idx.clamp(min=0))
        is_root = batch.parent_idx < 0
        return batch.mask & (nt >= t) & (is_root | (parent_time < t))
    raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")


# tree helpers


def _path_to_root(graph, node):
    """walk from node to root, returning the path as a list (root last)."""
    path = []
    cur = node
    while cur is not None:
        path.append(cur)
        preds = list(graph.predecessors(cur))
        cur = preds[0] if preds else None
    return path


def _leaf_counts(tree):
    counts = {}
    for node in reversed(list(nx.topological_sort(tree))):
        children = list(tree.successors(node))
        counts[node] = sum(counts[c] for c in children) if children else 1
    return counts


def _node_depths(tree):
    depths = {}
    for node in nx.topological_sort(tree):
        parents = list(tree.predecessors(node))
        depths[node] = (depths[parents[0]] + 1) if parents else 0
    return depths


def _all_roots(tree, sizes, max_size):
    """nodes whose subtree has at most max_size leaves (includes leaves themselves)."""
    return [
        n for n in tree.nodes()
        if 0 < sizes[n] <= max_size
    ]


def _snapshot_at(tree, bin_time, *, eps=1e-6):
    """snapshot of ``tree`` at ``bin_time`` -- the cross-section at the cut.

    A node n is in the snapshot iff ``n.time <= bin_time`` AND either n is
    a leaf or at least one child of n has ``time > bin_time``. Every leaf
    whose path to the root crosses ``bin_time`` is reached by exactly one
    snapshot ancestor, so using these as subtree roots guarantees that
    each crossing lineage is represented at this bin (subject to the
    later ``max_size`` filter). ``eps`` absorbs floating-point error so
    that a child whose stored time is bin_time + 1e-15 is treated as "at"
    the cut, not past it.
    """
    result = []
    for n in tree.nodes:
        t_n = tree.nodes[n].get("time")
        if t_n is None or t_n > bin_time + eps:
            continue
        children = list(tree.successors(n))
        if not children or any(
            tree.nodes[c].get("time", float("inf")) > bin_time + eps
            for c in children
        ):
            result.append(n)
    return result

def _extract_subtree(tree, root, sizes, depths, tree_name, bin_time=None, eps=1e-6):
    """Extract the subgraph rooted at ``root``.

    If ``bin_time`` is None, includes every descendant of ``root``.
    Otherwise, includes only ``root`` plus descendants reachable through
    children of ``root`` whose ``time > bin_time + eps`` -- i.e. the
    clade descending past the cut at ``bin_time``. This makes the
    snapshot subtrees at a bin leaf-disjoint: each leaf belongs to
    exactly the subtree of its latest snapshot ancestor.
    """
    if bin_time is None:
        nodes = nx.descendants(tree, root) | {root}
    else:
        nodes = {root}
        for c in tree.successors(root):
            if tree.nodes[c].get("time", float("inf")) > bin_time + eps:
                nodes.add(c)
                nodes.update(nx.descendants(tree, c))
    sg = tree.subgraph(nodes).copy()
    leaves = [n for n in sg if sg.out_degree(n) == 0]
    root_depth = depths.get(root, 0)
    max_leaf_depth = max(
        (depths.get(l, 0) for l in leaves), default=root_depth
    )
    return {
        "graph": sg,
        "root": root,
        "root_time": tree.nodes[root].get("time", float("nan")),
        "leaf_time": max(
            (tree.nodes[l].get("time", float("nan")) for l in leaves),
            default=float("nan"),
        ),
        "leaves": leaves,
        "tree_name": tree_name,
        "depth": max_leaf_depth - root_depth,
        "size": len(leaves),
    }

def _subsample_graph(graph, root, max_nodes=DEFAULT_MAX_NODES, rng=None):
    """if a tree exceeds max_nodes, randomly drop leaves and keep ancestors."""
    if graph.number_of_nodes() <= max_nodes:
        return graph

    rng = rng or np.random.default_rng()
    leaves = [n for n in graph if graph.out_degree(n) == 0]

    # iteratively reduce leaf count until total nodes fit
    n_keep = max(1, int(len(leaves) * max_nodes / graph.number_of_nodes()))
    keep = None

    for _ in range(20):
        chosen = (
            rng.choice(leaves, size=min(n_keep, len(leaves)), replace=False)
            if n_keep < len(leaves)
            else leaves
        )
        keep = {root}
        for leaf in chosen:
            for node in _path_to_root(graph, leaf):
                keep.add(node)
        if len(keep) <= max_nodes:
            break
        n_keep = max(1, int(n_keep * max_nodes / len(keep)))

    return graph.subgraph(keep).copy()


# -- collation: list[subtree_dict] -> treebatch ------------------------


def _mrca_times(graph, node_list, idx):
    """compute (N, N) mrca-time matrix for a single tree."""
    N = len(node_list)
    root = next(n for n in node_list if graph.in_degree(n) == 0)
    paths = {
        n: list(reversed(_path_to_root(graph, n))) for n in node_list
    }

    mrca = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i, N):
            lca = root
            for a, b in zip(paths[node_list[i]], paths[node_list[j]]):
                if a == b:
                    lca = a
                else:
                    break
            t = graph.nodes[lca].get("time", 0.0)
            mrca[i, j] = mrca[j, i] = t
    return mrca


def _process_tree(sd, leaf_states, dim, max_nodes, rng, randomize_child_index):
    """convert a single subtree dict into numpy arrays."""
    graph = _subsample_graph(sd["graph"], sd["root"], max_nodes, rng)
    node_list = list(nx.topological_sort(graph))
    N = len(node_list)
    idx = {n: i for i, n in enumerate(node_list)}

    adj = np.zeros((N, N), dtype=np.float32)
    for u, v in graph.edges():
        adj[idx[u], idx[v]] = 1.0

    times = np.array(
        [graph.nodes[n].get("time", 0.0) for n in node_list], dtype=np.float32
    )
    mrca = _mrca_times(graph, node_list, idx)
    leaf_mask = np.array(
        [graph.out_degree(n) == 0 for n in node_list], dtype=bool
    )

    depth = np.zeros(N, dtype=np.int64)
    parent_idx = np.full(N, -1, dtype=np.int64)
    for n in node_list:
        preds = list(graph.predecessors(n))
        if preds:
            depth[idx[n]] = depth[idx[preds[0]]] + 1
            parent_idx[idx[n]] = idx[preds[0]]

    child_index = np.zeros(N, dtype=np.int64)
    for n in node_list:
        children = list(graph.successors(n))
        slots = (
            rng.permutation(len(children))
            if randomize_child_index
            else np.arange(len(children))
        )
        for ci, child in zip(slots, children):
            child_index[idx[child]] = ci

    ls = np.zeros((N, dim), dtype=np.float32)
    obs = np.zeros((N, dim), dtype=np.float32)
    obs_mask = np.zeros(N, dtype=bool)
    for n in node_list:
        if n not in leaf_states:
            continue
        value = leaf_states[n].detach().cpu().numpy()
        obs[idx[n]] = value
        obs_mask[idx[n]] = True
        if graph.out_degree(n) == 0:
            ls[idx[n]] = value

    return {
        "adj": adj, "times": times, "mrca": mrca, "leaf_mask": leaf_mask,
        "depth": depth, "parent_idx": parent_idx, "child_index": child_index,
        "root_idx": idx.get(sd["root"], 0), "n_nodes": N,
        "node_names": node_list, "ls": ls, "obs": obs, "obs_mask": obs_mask,
    }


def collate_subtrees(subtree_dicts, leaf_states, device="cpu",
                     max_nodes=DEFAULT_MAX_NODES, rng=None,
                     randomize_child_index=True):
    """pack a list of subtree record dicts into a TreeBatch.

    parameters
    ----------
    subtree_dicts :
        each dict must have at least "graph" (nx.DiGraph) and "root".
    leaf_states :
        {node_name: (dim,) tensor} for observed leaf embeddings.
    device :
        target device for all output tensors.
    max_nodes :
        trees exceeding this node count are subsampled by dropping leaves.
    rng :
        numpy random generator for reproducible subsampling.
    randomize_child_index :
        if true, sibling slot labels are randomly permuted at collation time.
        This prevents slot-headed splitters from reading graph successor order
        as a stable branch label.  The tree topology itself is unchanged.
    """
    if not subtree_dicts:
        raise ValueError("cannot collate an empty list of subtrees")

    B = len(subtree_dicts)
    dim = next(iter(leaf_states.values())).shape[-1]
    rng = rng or np.random.default_rng()

    trees = [
        _process_tree(
            sd, leaf_states, dim, max_nodes, rng,
            randomize_child_index=randomize_child_index,
        )
        for sd in subtree_dicts
    ]
    max_N = max(t["n_nodes"] for t in trees)

    # padded batch arrays
    adj_b   = np.zeros((B, max_N, max_N), dtype=np.float32)
    times_b = np.zeros((B, max_N), dtype=np.float32)
    mrca_b  = np.zeros((B, max_N, max_N), dtype=np.float32)
    mask_b  = np.zeros((B, max_N), dtype=bool)
    lm_b    = np.zeros((B, max_N), dtype=bool)
    dep_b   = np.zeros((B, max_N), dtype=np.int64)
    par_b   = np.full((B, max_N), -1, dtype=np.int64)
    ci_b    = np.zeros((B, max_N), dtype=np.int64)
    ri_b    = np.zeros(B, dtype=np.int64)
    nn_b    = np.zeros(B, dtype=np.int64)
    ls_b    = np.zeros((B, max_N, dim), dtype=np.float32)
    obs_b   = np.zeros((B, max_N, dim), dtype=np.float32)
    om_b    = np.zeros((B, max_N), dtype=bool)
    names   = []
    max_depth = 0

    for i, t in enumerate(trees):
        n = t["n_nodes"]
        adj_b[i, :n, :n]  = t["adj"]
        times_b[i, :n]    = t["times"]
        mrca_b[i, :n, :n] = t["mrca"]
        mask_b[i, :n]     = True
        lm_b[i, :n]       = t["leaf_mask"]
        dep_b[i, :n]      = t["depth"]
        par_b[i, :n]      = t["parent_idx"]
        ci_b[i, :n]       = t["child_index"]
        ri_b[i]           = t["root_idx"]
        nn_b[i]           = n
        ls_b[i, :n]       = t["ls"]
        obs_b[i, :n]      = t["obs"]
        om_b[i, :n]       = t["obs_mask"]
        names.append(t["node_names"])
        max_depth = max(max_depth, int(t["depth"].max()) if n else 0)

    def _t(arr):
        return torch.from_numpy(arr).to(device)

    return TreeBatch(
        adj=_t(adj_b), node_times=_t(times_b), mrca_times=_t(mrca_b),
        mask=_t(mask_b), leaf_mask=_t(lm_b), depth=_t(dep_b),
        child_index=_t(ci_b), parent_idx=_t(par_b), root_idx=_t(ri_b),
        n_nodes=_t(nn_b), max_depth=max_depth, node_names=names,
        leaf_states=_t(ls_b), observed_mask=_t(om_b),
        observed_states=_t(obs_b), subtree_records=subtree_dicts,
    )


# -- treerecord: cache and index subtrees from a treedata file ---------


class TreeRecord:
    """cache and index subtrees from a single treedata file."""

    def __init__(self, tdata=None, time_bins=None, max_size=128,
                 min_size=1, time_tolerance=None, tree_names=None):
        self.time_bins = sorted(time_bins) if time_bins else []
        self.max_size = max_size
        self.min_size = min_size
        self.time_tolerance = (
            time_tolerance
            if time_tolerance is not None
            else (
                min(b - a for a, b in zip(self.time_bins, self.time_bins[1:])) / 2.0
                if len(self.time_bins) > 1
                else 0.5
            )
        )

        self.subtree_index = defaultdict(list)
        self._n_total = 0
        self.leaf_time = None
        self.trees = {}

        if tdata is not None:
            self._build_index(tdata, tree_names=tree_names)
            if tdata.obst:
                first_key = tree_names[0] if tree_names else next(iter(tdata.obst))
                first_tree = tdata.obst[first_key]
                self.leaf_time = self._get_leaf_time(first_tree)

    @staticmethod
    def _get_leaf_time(tree):
        for node in reversed(list(tree.nodes())):
            if tree.out_degree(node) == 0:
                return np.round(tree.nodes[node].get("time", None), 3)
        return None

    def _build_index(self, tdata, tree_names=None):
        items = tdata.obst.items()
        if tree_names is not None:
            items = [(k, v) for k, v in items if k in set(tree_names)]
        for name, tree in items:
            self.trees[name] = tree
            logger.info("indexing tree %s (%d nodes)", name, tree.number_of_nodes())
            sizes = _leaf_counts(tree)
            depths = _node_depths(tree)

            eps = 1e-6
            for bin_time in self.time_bins:
                for rnode in _snapshot_at(tree, bin_time, eps=eps):
                    children = list(tree.successors(rnode))
                    if not children:
                        restricted_size = 1
                    else:
                        restricted_size = sum(
                            sizes[c] for c in children
                            if tree.nodes[c].get("time", float("inf"))
                            > bin_time + eps
                        )
                        if restricted_size == 0:
                            continue  # not actually in the snapshot
                    if not (self.min_size <= restricted_size <= self.max_size):
                        continue
                    st = _extract_subtree(
                        tree, rnode, sizes, depths, name,
                        bin_time=bin_time, eps=eps,
                    )
                    self.subtree_index[bin_time].append(st)
                    self._n_total += 1

        logger.info(
            "built index: %d subtrees across %d time bins",
            self._n_total, len(self.subtree_index),
        )

    def _nearest_bin(self, t):
        if not self.time_bins:
            return None
        idx = bisect.bisect_left(self.time_bins, t)
        cands = []
        if idx > 0:
            cands.append(self.time_bins[idx - 1])
        if idx < len(self.time_bins):
            cands.append(self.time_bins[idx])
        return min(cands, key=lambda b: abs(b - t))

    def get_subtrees(self, time, n=None, rng=None):
        """return subtrees whose root time is nearest to the given time.

        if n is given, sample n subtrees without replacement.
        """
        nearest = self._nearest_bin(time)
        if nearest is not None and abs(time - nearest) > self.time_tolerance:
            logger.warning(
                "get_subtrees(time=%.4f) snapped to bin %.4f (offset %.4f > "
                "indexing tolerance %.4f); returned pool was indexed at a "
                "different time than requested.",
                time, nearest, abs(time - nearest), self.time_tolerance,
            )
        pool = self.subtree_index.get(nearest, [])
        if n is None or n >= len(pool):
            return list(pool)
        rng = rng or np.random.default_rng()
        idxs = rng.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in idxs]


    def get_batch(self, times, n_per_time, rng=None):
        """sample n_per_time subtrees for each time in the list."""
        rng = rng or np.random.default_rng()
        batch = []
        for t in times:
            batch.extend(self.get_subtrees(t, n=n_per_time, rng=rng))
        return batch

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "time_bins": self.time_bins,
            "max_size": self.max_size,
            "min_size": self.min_size,
            "leaf_time": self.leaf_time,
            "time_tolerance": self.time_tolerance,
            "subtree_index": dict(self.subtree_index),
            "trees": self.trees,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("saved TreeRecord to %s (%.1f MB)", path, path.stat().st_size / 1e6)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.time_bins = data["time_bins"]
        obj.max_size = data["max_size"]
        obj.min_size = data["min_size"]
        obj.time_tolerance = data["time_tolerance"]
        obj.subtree_index = defaultdict(list, data["subtree_index"])
        obj.leaf_time = data["leaf_time"]
        obj.trees = data.get("trees", {})
        obj._n_total = sum(len(v) for v in obj.subtree_index.values())
        logger.info("loaded TreeRecord from %s (%d subtrees)", path, obj._n_total)
        return obj

    def summary(self):
        lines = [
            f"TreeRecord  max_size={self.max_size}  "
            f"min_size={self.min_size}  tol={self.time_tolerance}",
            f"  {len(self.time_bins)} time bins: {self.time_bins}",
        ]
        for t in self.time_bins:
            pool = self.subtree_index.get(t, [])
            if not pool:
                lines.append(f"  t={t:.1f}:  (empty)")
                continue
            sizes = [s["size"] for s in pool]
            lines.append(
                f"  t={t:.1f}:  {len(pool):>5d} subtrees  |  "
                f"leaves  min={min(sizes)}  max={max(sizes)}  "
                f"mean={np.mean(sizes):.1f}  median={np.median(sizes):.0f}"
            )
        out = "\n".join(lines)
        print(out)
        return out

    def __repr__(self):
        return f"TreeRecord(trees {list(self.trees.keys())}, \n {self._n_total} subtrees, bins={self.time_bins})"
