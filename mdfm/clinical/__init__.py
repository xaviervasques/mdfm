"""Fixed-formula clinical descriptors on raw sensor streams, outside the body model (DRAFT).

image_plane: the S1 image-plane descriptors on raw 2D keypoints, both rest postures
(docs/MFM_DESIGN_NOTE_CLINICAL_QUESTIONS_2026-09-29.md, section 3 priority 4, sections 4.3 and 4.5;
docs/rules/DESCRIPTOR_RULES_DRAFT.md, DR-13 and DR-14). Nothing in this package reads or writes a file:
callers pass arrays. Nothing is imported here, so that importing the package loads no submodule.
"""
