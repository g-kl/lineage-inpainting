"""lineage inpainting's cell type map, fig 3's recipe -> numerical_results/cell_type_map.pkl.
one gpu, ~7 min, most of it the dataset load.

    python -m analysis.figures.supp_summary.compute_map
"""
import pickle

from analysis.figures.supp_summary import style as S
from analysis.lib.celltype_map.eval import build_cell_type_map, build_label_maps
from datasets.mouse_embryo import MouseEmbryoDataset


def main():
    print("loading dataset...", flush=True)
    dataset = MouseEmbryoDataset(max_size=S.N_TREES)
    g = build_cell_type_map(S.MODEL_ASSIGNMENTS, dataset=dataset, out_dir=S.NR,
                            pmi_quantile=S.PMI_QUANTILE, n_max=None,
                            resolution_by_time=S.RESOLUTION_BY_TIME,
                            min_composition=S.MIN_COMPOSITION, composition=S.COMPOSITION,
                            save_graph=False, save_png=False)["graph"]
    name_to_labels, _ = build_label_maps(dataset)
    with open(S.MAP_PKL, "wb") as f:
        pickle.dump({"graph": g, "name_to_labels": name_to_labels}, f)
    print(f"types={len({ct for ct, _ in g.nodes()})} edges={g.number_of_edges()} "
          f"-> {S.MAP_PKL}")


if __name__ == "__main__":
    main()
