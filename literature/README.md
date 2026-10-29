# literature

external gene sets, orthology maps, curated transitions and phenotype maps that the analyses read.

| file | contents | source | read by |
|---|---|---|---|
| `human_to_mouse_hcop.csv` | human to mouse orthologs | [HGNC HCOP](https://www.genenames.org/tools/hcop/) | [utils/human_to_mouse_gene_names.py](../utils/human_to_mouse_gene_names.py) |
| `human_to_mouse_biomart.csv` | human to mouse orthologs with homology type and confidence | [Ensembl BioMart](https://www.ensembl.org/biomart/) | [utils/human_to_mouse_gene_names.py](../utils/human_to_mouse_gene_names.py) |
| `gene_age.csv` | per-gene evolutionary age (Myr) | Kijima et al., 2026 (biorxiv) | [analysis/figures/recipe/tf_reuse.py](../analysis/figures/recipe/tf_reuse.py) |
| `tf_list.txt` | mouse transcription factors | [Aerts lab list](https://resources.aertslab.org/cistarget/tf_lists/) | `analysis/pipelines/perturb/`, `analysis/figures/{perturb,tbx5}/` |
| `Qiu24_mouse_tree_edges.csv`, `Qiu24_mouse_tree_edges_full.csv` | cell-type developmental tree | Qiu et al. 2024 (Nature) | [analysis/figures/supp_summary/style.py](../analysis/figures/supp_summary/style.py) |
| `transitions/*.csv` | curated cell-type transitions per lineage | one citation per row, in the `reference` column | [analysis/lib/celltype_map/eval.py](../analysis/lib/celltype_map/eval.py) |
| `phenotype_to_celltype_mapping.csv` | MGI phenotype to cell type map | built by `mgi_phenotype_mapping.py` from the MGI mouse phenotype gene-attribute table (`../data/mouse_phenotypes_gene_attribute_edges.txt`, from [Harmonizome](https://maayanlab.cloud/Harmonizome/)) | [analysis/figures/perturb/compute_mgi_box.py](../analysis/figures/perturb/compute_mgi_box.py) |
| `hox_consistency.py` | curated (cell type, Hox gene) pairs | literature curation | [analysis/figures/summary_qc/compute_hox.py](../analysis/figures/summary_qc/compute_hox.py) |

## not included

`gnomad_constraints.tsv` is gitignored. download the gnomAD constraint metrics (per-transcript TSV with `gene`, `canonical`, `transcript_type` and `lof.oe_ci.upper` columns) from [gnomad downloads page](https://gnomad.broadinstitute.org/downloads).

it is read by [analysis/figures/perturb/compute_gnomad_box.py](../analysis/figures/perturb/compute_gnomad_box.py) (the LoF-constraint panel of the perturbation figure).
