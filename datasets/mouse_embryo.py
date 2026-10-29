
import logging
import os
from typing import Optional

import pandas as pd
import torch
import treedata

from datasets.tree_record import TreeRecord

logger = logging.getLogger(__name__)


class MouseEmbryoDataset:
    """Loads mouse embryo trees, scVI embeddings, and cell-type annotations.

    Parameters
    ----------
    data_dir : str
        Directory containing ``.h5td`` tree files, the h5ad expression file,
        and the cell-type annotation CSV.
    embedding_key : str
        Key in ``adata.obsm`` for the embedding matrix.
    tree_filenames : list of str, optional
        Unused; trees are read from ``trees_filename``.
    time_bins : list of float, optional
        Time bins for ``TreeRecord`` construction.
    cache_dir : str
        Directory for caching ``TreeRecord`` pickles.
    max_size : int
        Maximum number of nodes per ``TreeRecord``.
    """

    def __init__(
        self,
        data_dir: str = "../data/embryos",
        states_filename: str = "qiu24_integrated_hvgs_and_scvi.h5ad",
        trees_filename: str = "final_topology.h5td",
        annotation_filename: str = "final_subtype_v2_lower.csv",
        umap_filename: str = "final_umap.csv",
        embedding_key: str = "X_scvi",
        tree_filenames: Optional[list[str]] = None,
        time_bins: Optional[list[float]] = None,
        cache_dir: str = "cache",
        max_size: int = 128,
        devmap_only: bool = True,
        type_info_filename: Optional[str] = 'final_type_info_v2_lower.csv',
        # State source. ``state_source`` selects where leaf features live in
        # the AnnData object: "obsm" (default -- pulls obsm[state_key], where
        # state_key defaults to embedding_key), "X" (pulls adata.X), or
        # "layers" (pulls adata.layers[state_key]).
        # ``state_transform`` applies an element-wise transform after read.
        # Defaults read the scVI embedding (obsm[embedding_key]).
        state_source: str = "obsm",
        state_key: Optional[str] = None,
        state_transform: Optional[str] = None,
    ):
        if time_bins is None:
            time_bins = [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

        # trees
        trees = {}
        records = {}
        td = None  # lazy-load treedata only if no cached records found

        cache_paths = {}
        for fname in os.listdir(cache_dir) if os.path.isdir(cache_dir) else []:
            prefix = f"{trees_filename}_"
            if fname.startswith(prefix) and fname.endswith(".pkl"):
                tree_name = fname[len(prefix):-len(".pkl")]
                cache_paths[tree_name] = os.path.join(cache_dir, fname)

        # Load cached records
        for tree_name, cache_path in cache_paths.items():
            records[tree_name] = TreeRecord.load(cache_path)

        # If no cached records found, we need to load the treedata
        if not records:
            td = treedata.read_h5td(os.path.join(data_dir, trees_filename))
            logger.info("loaded treedata from %s", trees_filename)
            for tree_name in td.obst:
                cache_path = os.path.join(cache_dir, f"{trees_filename}_{tree_name}.pkl")
                records[tree_name] = TreeRecord(
                    td, time_bins=time_bins, max_size=max_size,
                    tree_names=[tree_name],
                )
                logger.info("created TreeRecord for %s", tree_name)
                records[tree_name].save(cache_path)
                logger.info("saved TreeRecord for %s", tree_name)

        # states
        states = treedata.read_h5ad(
            os.path.join(data_dir, states_filename)
        )

        if devmap_only:
            states = states[states.obs['dataset'] == 'devmap']

        # annotations
        cell_types_df = pd.read_csv(
            os.path.join(data_dir, annotation_filename), index_col=0
        )

        # leaf states -- read from the configured source, optionally transformed.
        obs = self._load_state_features(
            states,
            source=state_source,
            key=(state_key if state_key is not None else embedding_key),
            transform=state_transform,
        )
        # Memory-efficient build: only convert to tensors for cells that
        # actually appear as leaves in some subtree. At HVG dim (~3956)
        # this saves tens of GB vs. building a tensor per cell.
        referenced = self._collect_referenced_cell_names(records)
        if referenced:
            leaf_states = {}
            for i, name in enumerate(states.obs_names):
                if name in referenced:
                    leaf_states[name] = torch.tensor(obs[i])
        else:
            leaf_states = {
                states.obs_names[i]: torch.tensor(obs[i])
                for i in range(len(states))
            }

        self.records = records
        self.trees = {}
        
        for record in records.values():
            self.trees.update(record.trees)

        self.states = states
        self.leaf_states = leaf_states
        # ``reference_features`` is the (n_cells, x_dim) matrix used to build
        # leaf_states; aligners that need full-dataset references read from
        # here (so they automatically follow whatever state source / transform
        # the dataset is configured to use).
        self.reference_features = obs
        self.state_source = state_source
        self.state_key = state_key if state_key is not None else embedding_key
        self.state_transform = state_transform
        self.cell_types_df = cell_types_df
        self.x_dim = obs.shape[1]

        if type_info_filename is not None:
            self.type_info_df = pd.read_csv(
                os.path.join(data_dir, type_info_filename), index_col=0
            )
            name_to_idx = {n: i for i, n in enumerate(states.obs_names)}
            # Keys are matched case-insensitively: the *_lower type tables have a
            # lowercased index while obs['cell_subtype'] keeps its original case.
            subtype_index = self.type_info_df.index.astype(str).str.lower()
            subtype_to_type = dict(zip(subtype_index, self.type_info_df['cell_type']))
            subtype_to_gl = dict(zip(subtype_index, self.type_info_df['germ_layer']))
            subtype_to_lineage = dict(zip(subtype_index, self.type_info_df['lineage']))

            subtype_key = self.states.obs['cell_subtype'].astype(str).str.lower()
            self.states.obs['cell_type'] = subtype_key.map(subtype_to_type)
            self.states.obs['germ_layer'] = subtype_key.map(subtype_to_gl)
            self.states.obs['lineage'] = subtype_key.map(subtype_to_lineage)

            self.name_to_idx = name_to_idx

        self.umap_coords = pd.read_csv(os.path.join(data_dir, umap_filename), index_col=0)

    @staticmethod
    def _collect_referenced_cell_names(records) -> set:
        """Cell names appearing as a leaf in any cached subtree."""
        referenced: set = set()
        for rec in records.values():
            for subtrees_at_t in rec.subtree_index.values():
                for st in subtrees_at_t:
                    graph = st["graph"]
                    for n in graph.nodes:
                        if graph.out_degree(n) == 0:
                            referenced.add(n)
        return referenced

    @staticmethod
    def _load_state_features(adata, source: str, key: str, transform: Optional[str]):
        """Pull a (n_cells, dim) feature matrix from adata according to source.

        Returns a numpy array with the optional element-wise transform applied.
        """
        import numpy as np
        from scipy.sparse import issparse

        if source == "obsm":
            arr = adata.obsm[key]
        elif source == "X":
            arr = adata.X
        elif source == "layers":
            arr = adata.layers[key]
        else:
            raise ValueError(
                f"state_source must be 'obsm' | 'X' | 'layers'; got {source!r}"
            )

        if issparse(arr):
            arr = arr.toarray()
        arr = np.asarray(arr, dtype=np.float32)

        if transform is None:
            return arr
        if transform == "log1p":
            return np.log1p(arr)
        raise ValueError(
            f"state_transform must be None | 'log1p'; got {transform!r}"
        )
