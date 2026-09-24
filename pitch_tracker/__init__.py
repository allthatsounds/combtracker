"""pitch_tracker, reduced to what this paper's results depend on.

The full package's ``__init__`` also imports ``fine_contour`` and ``ridge``,
which pull in the cool-frames filterbank library. Nothing reported in the paper
uses them: every result goes through ``comb_f0``, ``shrp``, ``evaluate`` and
``groundtruth``, so only those four ship here.

Importing the package needs numpy only. ``groundtruth.parse_raw_sheet`` (reading
the annotation workbooks) additionally needs openpyxl, imported when called.
"""
from . import comb_f0, evaluate, groundtruth, shrp   # noqa: F401

__all__ = ["comb_f0", "evaluate", "groundtruth", "shrp"]
