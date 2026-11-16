"""Neural statistics and descriptors of addendum 1 (docs/MFM_DESIGN_ADDENDUM_1_NEURAL_SCIENCE.md, sections 2.3 and
2.5) under the neural rule file docs/rules/NEURAL_RULES_DRAFT.md.

Status: DRAFT. The rule file is not hashed; every default that fills one of its OPEN items is a Draft value naming
that item, and every result lists the draft defaults it used. Tested on synthetic signals only; no function reads a
record. See mdfm/neural/README.md.
"""
from .stats import Draft, draft_parameters, resolve  # noqa: F401
from . import descriptors, stats  # noqa: F401
