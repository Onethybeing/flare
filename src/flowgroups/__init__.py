"""flowgroups: leakage-safe, case-level review queues for illicit-flow detection on transaction graphs.

The package contains the reusable algorithms behind the BTP study
"From Transactions to Cases" (Elliptic++):

* :mod:`flowgroups.asof`           point-in-time (as-of) address features and a lifetime-leakage audit
* :mod:`flowgroups.graph`          per-step transaction graph with money-flow and shared-address links
* :mod:`flowgroups.groups`         flow-constrained group construction, baselines, group features and scores
* :mod:`flowgroups.queues`         fixed-budget review queues and case-fragmentation statistics
* :mod:`flowgroups.metrics`        threshold selection and paired bootstrap confidence intervals
* :mod:`flowgroups.models`         direction-aware GraphSAGE (PyTorch, optional)
* :mod:`flowgroups.counterfactual` gradient-selected evidence edges and deletion tests (PyTorch, optional)
"""

__version__ = "1.0.0"

from .graph import StepGraph
from .groups import flow_groups, cc_groups, ego_groups, group_features, group_score, FEATURE_NAMES
from .queues import run_queue, queue_stats, pooled, QUEUE_STAT_NAMES
from .metrics import best_f1_threshold, paired_bootstrap_ci

__all__ = [
    "StepGraph", "flow_groups", "cc_groups", "ego_groups", "group_features", "group_score", "FEATURE_NAMES",
    "run_queue", "queue_stats", "pooled", "QUEUE_STAT_NAMES", "best_f1_threshold", "paired_bootstrap_ci",
]
