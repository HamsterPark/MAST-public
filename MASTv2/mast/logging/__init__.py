"""mast.logging — day-scale SQLite experiment store (Experiment → Sample → Action).

Phase 6 port:
  storage.py        K from v1 (3-layer schema preserved)
  action_record.py  K from v1 (ActionRecord <-> dict marshalling)
  experiment_log.py K from v1 (per-experiment append-only log)
  report.py         K from v1 (ReportGenerator: markdown report writer)
"""

from mast.logging.action_record import action_to_dict, dict_to_action
from mast.logging.experiment_log import ExperimentLog
from mast.logging.report import ReportGenerator
from mast.logging.storage import ExperimentStorage

__all__ = [
    "ExperimentStorage",
    "ExperimentLog",
    "ReportGenerator",
    "action_to_dict",
    "dict_to_action",
]
