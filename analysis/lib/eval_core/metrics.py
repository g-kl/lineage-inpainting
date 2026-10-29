"""assignment-only coupling metrics, scored by the main-text figure's own code.

the grid scores every configuration's coupling with the functions the summary/QC
figure (analysis/figures/summary_qc) uses for its metrics row, so a grid number and
a main-figure number are the same quantity:

  germ-layer crossing, lineage oscillation,    compute_assignment_metrics.compute_for
  linkage distortion (+ normalised correlation)
  Hox-code distortion (+ correlation)          compute_hox.collect_pairs on the figure's
                                               common support -- ancestors assigned by
                                               the coupling AND both GENOT couplings --
                                               then the figure's per-(cell type, gene)
                                               Pearson loop

rows are tidy ``(metric, tag, value)`` triples in the figure's vocabulary. the figure
modules chdir to the repo root when imported, which is where every grid job runs;
they are imported lazily so importing this module stays light.
"""
from __future__ import annotations

import os
import pickle
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr

from literature.hox_consistency import static_hox_pairs

# the two GENOT couplings that define the Hox common support: the `assignments` field
# of the summary_qc GENOT model pickles (summary_qc.style.GENOT_PKLS), extracted so no
# JAX unpickling is needed, keyed as summary_qc.style.METHODS names them.
GENOT_ASSIGNMENTS = {
    "genot_sinkhorn": "outputs/e_baselines/assignments_w2_unified.pkl",
    "genot_cbfgw": "outputs/e_baselines/assignments_fgw_unified.pkl",
}


def load_genot_assignments(root="."):
    out = {}
    for method, path in GENOT_ASSIGNMENTS.items():
        with open(os.path.join(root, path), "rb") as f:
            out[method] = pickle.load(f)
    return out


@dataclass
class HoxExpression:
    """the dense Hox expression the figure correlates, built once per dataset."""
    expr: np.ndarray            # (n_cells, n_present)
    gene_col: dict              # gene -> column
    genes_by_ct: dict           # cell type -> [shortlisted genes]
    name_to_idx: dict           # cell name -> row


def hox_expression(dataset) -> HoxExpression:
    """mirrors compute_hox.main's expression pull (shortlisted, HVG-present genes)."""
    genes_by_ct = defaultdict(list)
    for ct, gene in static_hox_pairs:
        genes_by_ct[ct].append(gene)
    var_names = set(dataset.states.var_names)
    present = [g for g in sorted({g for _, g in static_hox_pairs}) if g in var_names]
    expr = np.asarray(dataset.states[:, present].X.todense(), dtype=np.float32)
    return HoxExpression(expr, {g: j for j, g in enumerate(present)}, dict(genes_by_ct),
                         dataset.name_to_idx)


def hox_rows(dataset, assignments, genot, expr):
    """``(hox_correlation | hox_distortion, "<ct>__<gene>", value)`` rows for one
    coupling, on the common support with the GENOT couplings (compute_hox.collect_pairs)
    and with compute_hox.main's scoring rule (n >= 3, both sides non-constant)."""
    from analysis.figures.summary_qc import compute_hox as H
    methods = {"model": assignments, **genot}
    pairs, *_ = H.collect_pairs(dataset, methods, set(expr.genes_by_ct))
    rows = []
    for ct in sorted(expr.genes_by_ct):
        plist = pairs[("model", ct)]
        for gene in expr.genes_by_ct[ct]:
            if gene not in expr.gene_col:
                continue
            j = expr.gene_col[gene]
            desc, anc = [], []
            for leaf, anc_cell in plist:
                li, ai = expr.name_to_idx.get(leaf), expr.name_to_idx.get(anc_cell)
                if li is None or ai is None:
                    continue
                desc.append(expr.expr[li, j])
                anc.append(expr.expr[ai, j])
            if len(desc) >= 3 and np.std(desc) > 0 and np.std(anc) > 0:
                r, _ = pearsonr(desc, anc)
                rows.append(("hox_correlation", f"{ct}__{gene}", float(r)))
                rows.append(("hox_distortion", f"{ct}__{gene}", float(1.0 - r)))
    return rows


def assignment_metrics(trees, assignments, ann, *, dataset=None, genot=None, expr=None):
    """tidy ``(metric, tag, value)`` rows for one coupling.

    ``ann`` is an :class:`analysis.lib.eval_core.annotations.Annotations`. the Hox
    rows need ``dataset`` (expression + cell types), ``genot`` (the two couplings
    defining the common support) and ``expr`` (:func:`hox_expression`); leave them
    None to emit only the germ / oscillation / linkage rows.
    """
    from analysis.figures.summary_qc.compute_assignment_metrics import compute_for
    rows = list(compute_for(assignments, trees, ann.name_to_gl, ann.name_to_lineage,
                            ann.lineages, ann.time_bins))
    if dataset is not None and genot is not None:
        rows += hox_rows(dataset, assignments, genot, expr or hox_expression(dataset))
    return rows
