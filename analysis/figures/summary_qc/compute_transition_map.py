"""row-1 cache: the cell-type transition map (an alluvial DiGraph) built from the
lineage-inpainting coupling.

thin wrapper over analysis.lib.celltype_map.eval.build_cell_type_map: takes every
transition pair in the model coupling (style.MODEL_ASSIGNMENTS) within the nn snap
cutoff, keeps a (cell_type, time) edge when its in + out composition is >=
style.MIN_COMPOSITION (no pmi cut, no subsample), and caches the networkx graph plus
the germ-layer colour dict so the plotting script needs no dataset.

writes numerical_results/transition_map.pkl = {graph, color_dict, name_to_labels}.
loads the scvi embedding via MouseEmbryoDataset (faiss kNN for the PMI, which the
quantile-0 cut keeps whole);
run with the lt conda env from the repo root.
"""
import os
import sys
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from analysis.figures.summary_qc import style as S
from analysis.lib.celltype_map.eval import build_cell_type_map, build_label_maps
from datasets.mouse_embryo import MouseEmbryoDataset


def main():
    print("loading dataset...", flush=True)
    dataset = MouseEmbryoDataset(max_size=S.N_TREES)

    # build the graph but keep its artifacts inside this figure folder.
    res = build_cell_type_map(
        S.MODEL_ASSIGNMENTS,
        dataset=dataset,
        out_dir=S.NR,       # keep any side-effect artifacts inside the figure folder
        pmi_quantile=S.PMI_QUANTILE,
        n_max=None,         # every pair: deterministic
        resolution_by_time=S.RESOLUTION_BY_TIME,
        min_composition=S.MIN_COMPOSITION,
        composition=S.COMPOSITION,
        palette=S.TRANSITION_PALETTE,
        save_graph=False,   # we re-pickle (graph + colours) into numerical_results
        save_png=False,     # plotting is a separate phase
    )
    graph = res["graph"]
    name_to_labels, color_dict = build_label_maps(dataset, S.TRANSITION_PALETTE)

    out = os.path.join(S.NR, "transition_map.pkl")
    with open(out, "wb") as f:
        pickle.dump({"graph": graph, "color_dict": color_dict,
                     "name_to_labels": name_to_labels}, f)
    n_types = len({ct for ct, _ in graph.nodes()})
    print(f"wrote {out}  (types={n_types}, edges={graph.number_of_edges()}, "
          f"pmi_cutoff={res['pmi_cutoff']:.2f})", flush=True)


if __name__ == "__main__":
    main()
