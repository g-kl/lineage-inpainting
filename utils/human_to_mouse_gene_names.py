import os

import pandas as pd
from tqdm import tqdm

path = os.path.abspath(os.path.dirname(__file__))
biomart_csv = os.path.join(path, '../literature/human_to_mouse_biomart.csv')
hcop_csv = os.path.join(path, '../literature/human_to_mouse_hcop.csv')


def load_one2one(source='biomart'):
    """Load the full human to mouse 1-to-1 ortholog dictionary from a reference csv."""
    source = source.lower()
    print('Loading {} ortholog table...'.format(source))

    if source == 'biomart':
        tab = pd.read_csv(biomart_csv)
        tab = tab[tab['Mouse homology type'] == 'ortholog_one2one']
        tab = tab.dropna(subset=['Gene name', 'Mouse gene name'])
        return dict(zip(tab['Gene name'], tab['Mouse gene name']))

    elif source == 'hcop':
        tab = pd.read_csv(hcop_csv)
        tab = tab[(tab['Gene name'] != '-') & (tab['Mouse gene name'] != '-')]
        tab = tab.dropna(subset=['Gene name', 'Mouse gene name'])
        tab = tab.drop_duplicates(subset=['Gene name', 'Mouse gene name'])
        # keep only genes that are unique on both the human and the mouse side
        unique_human = tab['Gene name'].value_counts()
        unique_mouse = tab['Mouse gene name'].value_counts()
        tab = tab[tab['Gene name'].map(unique_human) == 1]
        tab = tab[tab['Mouse gene name'].map(unique_mouse) == 1]
        return dict(zip(tab['Gene name'], tab['Mouse gene name']))

    else:
        raise ValueError('source must be either "biomart" or "hcop"')


def human_to_mouse(human_genes, source='biomart'):
    """Map a list of human genes to their mouse orthologs.

    Returns a dictionary of human gene : mouse ortholog, including only genes
    that have a 1-to-1 ortholog in the reference table.
    """
    one2one = load_one2one(source=source)

    print('Mapping {} human genes to mouse orthologs...'.format(len(human_genes)))
    result = {}
    for gene in tqdm(human_genes):
        if gene in one2one:
            result[gene] = one2one[gene]

    print('Found 1-to-1 orthologs for {} of {} genes.'.format(len(result), len(human_genes)))
    return result
