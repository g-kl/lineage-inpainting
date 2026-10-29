"""shared spine for the grid's model evaluation.

one place for the driver / glue that turns a trained run into the tidy metric rows
the grid-summary figures read, with the metric code itself borrowed from the
main-text figure (``analysis/figures/summary_qc``) so the two never drift.
"""
from analysis.lib.eval_core.annotations import Annotations, load_annotations
from analysis.lib.eval_core.checkpoints import resolve_checkpoint
from analysis.lib.eval_core.metrics import (
    GENOT_ASSIGNMENTS, assignment_metrics, hox_expression, load_genot_assignments,
)

# NB: model loading lives in analysis.lib.eval_core.models (imports torch/hydra) and
# is not re-exported here, so importing the spine stays light.

__all__ = [
    "Annotations", "load_annotations", "resolve_checkpoint",
    "GENOT_ASSIGNMENTS", "assignment_metrics", "hox_expression", "load_genot_assignments",
]
