from __future__ import annotations
import argparse
import re
from collections import OrderedDict

import pandas as pd

# reference copy of the cell-type names; the run uses those read by load_inputs()
CELL_TYPES = [
    'rostral ectoderm', 'amniotic ectoderm', 'extraembryonic ectoderm',
    'cranial neural crest', 'frontonasal mesenchyme', 'pharyngeal arch mesenchyme',
    'trigeminal ganglion neuron', 'autonomic neuron', 'autonomic progenitor',
    'dorsal root ganglion neuron', 'dorsal root ganglion progenitor',
    'sacral neural crest', 'trunk neural crest', 'vagal neural crest',
    'forebrain', 'hindbrain', 'spinal cord', 'diencephalon', 'eye field',
    'floor plate', 'hypothalamus', 'midbrain', 'roof plate', 'telencephalon',
    'hindbrain (isthmic organizer)', 'hindbrain (r1)', 'hindbrain (r2-r4)',
    'hindbrain (r5-r8)', 'cranial motor neuron', 'hypothalamic neuron',
    'midbrain neuron', 'somatic motor neuron', 'spinal cord interneuron',
    'spinal cord (cervical)', 'spinal cord (lumbar)', 'spinal cord (sacral)',
    'spinal cord (thoracic)', 'epibranchial placode', 'epibranchial sensory neuron',
    'epidermal progenitor', 'lens placode', 'olfactory placode',
    'olfactory sensory neuron', 'otic placode', 'otic sensory neuron',
    'pre-placodal ectoderm', 'surface ectoderm', 'trigeminal placode',
    'trigeminal sensory neuron', 'definative endoderm', 'gut',
    'enteroendocrine progenitor', 'foregut', 'hepatoblast', 'hindgut', 'midgut',
    'parietal endoderm', 'pharyngeal endoderm', 'visceral endoderm', 'epiblast',
    'primitive streak', 'axial progenitor', 'macrophage', 'megakaryocyte',
    'primitive erythrocyte', 'allantoic endothelial progenitor', 'endothelial cell',
    'allantois', 'extraembryonic mesoderm', 'intermediate mesoderm',
    'lateral plate mesoderm', 'epicardium', 'first heart field', 'juxtacardiac field',
    'second heart field', 'forelimb field', 'hindlimb field', 'somatic mesoderm',
    'mesothelium', 'splanchnic mesoderm', 'anterior mesoderm', 'nascent mesoderm',
    'arterial endothelial cell', 'endocardial cell', 'endothelial progenitor',
    'hematoendothelial progenitor', 'notochord', 'pharyngeal mesoderm',
    'dermomyotome', 'head paraxial mesoderm', 'hypaxial myogenic progenitor',
    'myotome', 'nascent somite', 'periocular mesoderm', 'presomitic mesoderm',
    'sclerotome', 'primordial germ cell',
]

## coarse cell type groupings

FOREBRAIN   = ['forebrain', 'telencephalon', 'diencephalon', 'hypothalamus',
               'hypothalamic neuron']
MIDBRAIN    = ['midbrain', 'midbrain neuron']
HINDBRAIN   = ['hindbrain', 'hindbrain (isthmic organizer)', 'hindbrain (r1)',
               'hindbrain (r2-r4)', 'hindbrain (r5-r8)']
BRAIN       = FOREBRAIN + MIDBRAIN + HINDBRAIN
SOMITE      = ['nascent somite', 'presomitic mesoderm', 'dermomyotome',
               'myotome', 'sclerotome']
HEART       = ['first heart field', 'second heart field', 'juxtacardiac field',
               'epicardium', 'endocardial cell']
VASC_ENDO   = ['endothelial cell', 'endothelial progenitor', 'arterial endothelial cell',
               'allantoic endothelial progenitor', 'hematoendothelial progenitor']
GUT         = ['gut', 'foregut', 'midgut', 'hindgut']
CRANIOFACIAL = ['cranial neural crest', 'frontonasal mesenchyme',
                'pharyngeal arch mesenchyme']
PHARYNGEAL  = ['pharyngeal arch mesenchyme', 'cranial neural crest',
               'pharyngeal endoderm', 'pharyngeal mesoderm']
OTIC        = ['otic placode', 'otic sensory neuron']
EXE_MESO    = ['extraembryonic mesoderm', 'allantois', 'allantoic endothelial progenitor']


def _entry(cell_types, scope, note):
    return {'cell_types': cell_types, 'scope': scope, 'note': note}



## curated phenotype-concept -> cell-type mapping

CONCEPT_MAP = OrderedDict({
    # CNS
    'brain':        _entry(BRAIN, 'broad', 
                           None),
    'forebrain':    _entry(FOREBRAIN, 'direct', None),
    'midbrain':     _entry(MIDBRAIN, 'direct', None),
    'hindbrain':    _entry(HINDBRAIN, 'direct', None),
    'midbrain-hindbrain boundary': _entry(['hindbrain (isthmic organizer)', 'midbrain',
                                           'hindbrain (r1)'], 'direct',
                                          'isthmic organizer region'),
    'telencephalon': _entry(['telencephalon'], 'direct', None),
    'pallium':      _entry(['telencephalon'], 'direct',
                           'pallium = dorsal telencephalon (regional synonym, in-window)'),

    # eye
    'eye':          _entry(['eye field', 'lens placode'], 'direct', None),
    'lens':         _entry(['lens placode'], 'direct', None),
    'lens vesicle': _entry(['lens placode'], 'progenitor', 'lens vesicle from lens placode'),
    'retina':       _entry(['eye field'], 'progenitor', 'eye field = prospective neural retina'),

    # otic
    'ear':          _entry(OTIC, 'progenitor', None),
    'inner ear':    _entry(OTIC, 'progenitor', None),
    'otic vesicle': _entry(['otic placode'], 'progenitor', 'otic vesicle from otic placode'),

    # somitogenesis
    'somite':       _entry(SOMITE, 'direct', None),
    'dermomyotome': _entry(['dermomyotome'], 'direct', None),
    'myotome':      _entry(['myotome'], 'direct', None),

    # heart
    'heart':        _entry(HEART, 'direct', None),

    # vasculature
    'vascular':     _entry(VASC_ENDO + ['endocardial cell'], 'direct', None),
    'vascular endothelial cell': _entry(VASC_ENDO, 'direct',  None),
    'vascular remodeling': _entry(VASC_ENDO, 'direct', None),
    'artery':       _entry(['arterial endothelial cell', 'endothelial cell'], 'direct', None),
    'vein':         _entry(['endothelial cell', 'endothelial progenitor'], 'direct',
                           None),

    # gut
    'digestive system': _entry(GUT, 'direct', None),
    'intestine':    _entry(['midgut', 'hindgut', 'gut'], 'direct',
                           'mid/hindgut only'),

    # hepatoblast related
    'liver':            _entry(['hepatoblast'], 'progenitor', None),
    'hepatic':          _entry(['hepatoblast'], 'progenitor',  None),
    'hepatobiliary system': _entry(['hepatoblast'], 'progenitor', None),

    # pharyngeal/craniofacial
    'craniofacial':  _entry(CRANIOFACIAL, 'progenitor',
                            None),
    'face':          _entry(CRANIOFACIAL, 'progenitor', None),
    'pharyngeal arch': _entry(PHARYNGEAL, 'direct', None),

    # extraembryonic
    'allantois':    _entry(['allantois', 'allantoic endothelial progenitor'], 'direct', None),
    'extraembryonic mesoderm': _entry(EXE_MESO, 'direct', None),
})


SYNONYMS = {}
_VASC_REMODEL_RE = re.compile(r'developmental vascular remodeling')
_LEADING = re.compile(r'^(abnormal|delayed|impaired|increased|decreased)\s+')
_ARREST  = re.compile(r'^arrest of\s+')


def normalize(phenotype: str) -> str:
    """Reduce an MGI phenotype string to a canonical anatomical core."""
    p = phenotype.strip().lower()

    # phrase special case mapped to a concept
    if _VASC_REMODEL_RE.search(p):
        return 'vascular remodeling'

    p = p.replace(' morphology/development', '')
    p = p.replace(' embryonic development', ' development')
    p = re.sub(r'\bdevelopmental\b', '', p)
    p = re.sub(r'\s+development\b', '', p)
    p = re.sub(r'\s+delay\b', '', p)

    p = _ARREST.sub('', p)
    p = _LEADING.sub('', p)
    p = re.sub(r'\s+', ' ', p).strip()
    return SYNONYMS.get(p, p)

def _auto_substring_hits(core: str, cell_types):
    """Conservative word-boundary phrase containment of `core` in cell-type names."""
    if not core or len(core) < 3:
        return []
    pat = re.compile(r'\b' + re.escape(core) + r'\b')
    return [ct for ct in cell_types if pat.search(ct.lower())]


def build_mapping(phenotypes, cell_types, restrict_to=None, include_no_match=False,
                  use_auto=False):
    """Long-format DataFrame, one row per confident (phenotype, cell_type) edge.

    Columns: phenotype, anatomical_core, cell_type, concept_key, rule, scope, note
    `restrict_to`     : optional set of scopes to keep, e.g. {'direct'}.
    `include_no_match`: if True, append one NA row per unmapped phenotype.
    `use_auto`        : off by default, so every edge comes from CONCEPT_MAP. The
                        word-boundary lexical layer re-expands germ-layer terms
                        (ectoderm/mesoderm/endoderm) to their whole layer and lets
                        'head'/'motor' match compound names. Useful for spotting
                        lexical gaps.
    """
    cell_set = set(cell_types)
    rows, unmapped = [], []

    for ph in phenotypes:
        core = normalize(ph)
        seen = {}

        entry = CONCEPT_MAP.get(core)
        if entry is not None:
            for ct in entry['cell_types']:
                if ct in cell_set:
                    seen[ct] = dict(phenotype=ph, anatomical_core=core, cell_type=ct,
                                    concept_key=core, rule='curated', scope=entry['scope'],
                                    note=entry['note'])

        if use_auto:
            for ct in _auto_substring_hits(core, cell_types):
                if ct not in seen:
                    seen[ct] = dict(phenotype=ph, anatomical_core=core, cell_type=ct,
                                    concept_key=core, rule='auto_substring', scope='direct',
                                    note='lexical phrase match')

        if seen:
            rows.extend(seen.values())
        else:
            unmapped.append(ph)
            if include_no_match:
                rows.append(dict(phenotype=ph, anatomical_core=core, cell_type=pd.NA,
                                 concept_key=core, rule='no_match', scope=pd.NA,
                                 note='no confident in-window mapping'))

    df = pd.DataFrame(rows, columns=['phenotype', 'anatomical_core', 'cell_type',
                                     'concept_key', 'rule', 'scope', 'note'])
    if restrict_to is not None and len(df):
        keep = df['scope'].isin(set(restrict_to)) | (df['rule'] == 'no_match')
        df = df[keep].reset_index(drop=True)
    df.attrs['unmapped_phenotypes'] = unmapped
    return df


def to_wide(long_df):
    """phenotype -> sorted list of mapped cell types (real edges only)."""
    m = long_df.dropna(subset=['cell_type'])
    return (m.groupby('phenotype')['cell_type']
              .apply(lambda s: sorted(set(s))).to_dict())


def dead_concept_keys(phenotypes):
    """CONCEPT_MAP keys never triggered by any phenotype (should be empty)."""
    used = {normalize(p) for p in phenotypes}
    return [k for k in CONCEPT_MAP if k not in used]


def load_inputs(cell_types_csv, mgi_txt):
    ct = pd.read_csv(cell_types_csv, index_col=0)['cell_type'].unique().tolist()
    mgi = pd.read_csv(mgi_txt, sep='\t', index_col=0)
    dev = [p for p in mgi['Phenotype'].unique() if 'development' in p.lower()]
    return ct, dev



cell_types, phenotypes = load_inputs('../data/embryos/final_type_info.csv',
                                     '../data/mouse_phenotypes_gene_attribute_edges.txt')

long_df = build_mapping(phenotypes, cell_types)

wide = to_wide(long_df)
unmapped = long_df.attrs['unmapped_phenotypes']

print(f'phenotypes: {len(phenotypes)} | mapped: {len(wide)} | '
        f'unmapped: {len(unmapped)} | edges: {len(long_df)}')

print('\nedges by scope:')
print(long_df.groupby(['scope']).size().to_string())

dead = dead_concept_keys(phenotypes)
print(f'\ndead concept keys (never triggered): {dead if dead else "none"}')

mapped_cts = sorted(long_df['cell_type'].dropna().unique())
print(f'\ncell types receiving >=1 edge: {len(mapped_cts)} / {len(cell_types)}')

out_long = f'literature/phenotype_to_celltype_mapping.csv'
long_df.to_csv(out_long, index=False)

print(f'\nwrote {out_long} ({len(long_df)} edges)')