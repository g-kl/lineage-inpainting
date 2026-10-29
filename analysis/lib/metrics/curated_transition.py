"""fraction of literature-curated cell-type transitions present in a transition graph.

the curated (source_type, target_type) pairs are read from every .csv in a
directory; the graph is the cell-type transition graph from the sankey code. the
fraction is not normalised by the graph's edge count (that is controlled for
separately).
"""

import os
import pandas as pd


def curated_transition_fraction(transition_dir,
                                transition_graph):
    
    # loop through all .csv files in transition_dir
    curated_transitions = set()
    for file in os.listdir(transition_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(transition_dir, file))
            for _, row in df.iterrows():
                curated_transitions.add((row['source_type'], row['target_type']))
    print(f"Number of curated transitions: {len(curated_transitions)}")

    # compute fraction of curated transitions that are present in the graph
    graph_transitions = set([(x[0][0], x[1][0]) for x in transition_graph.edges()])
    num_curated_transitions = len(curated_transitions)

    print(f"Number of transitions in graph: {len(graph_transitions)}")
    num_curated_transitions_in_graph = len(curated_transitions.intersection(graph_transitions))
    fraction = num_curated_transitions_in_graph / num_curated_transitions if num_curated_transitions > 0 else 0.0
    return fraction

            