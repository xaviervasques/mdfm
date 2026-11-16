# What is in this public tree, and what is not

Generated from the program's working tree. This repository is a curated export of the code, the
tests, the scripts and the design documents meant for readers; the working tree also holds session
notes, internal logs, rule drafts and decision notes that are not documentation, and its history
is not the history a reader needs.

## Included

| area | files | bytes |
|---|---|---|
| configs | 3 | 20,649 |
| docs | 19 | 502,989 |
| env | 3 | 4,291 |
| mdfm | 143 | 5,122,916 |
| reports | 19 | 507,622 |
| scripts | 185 | 7,214,646 |
| tests | 118 | 2,412,832 |
| generated (README.md, LICENSE, CITATION.cff, .gitignore, this file) | 5 | |
| **total copied** | **490** | **15,785,945** |

## Deliberately not included

- the session notes, handoffs, planning files and decision logs of the working tree;
- the rule drafts and proposals: a gate rule ships once it is registered by its SHA-256;
- the held-out membership lists, the person maps, and every per-window or per-stride table keyed
  by AMASS take names (the AMASS terms do not allow redistribution);
- the reports, except the run-time inputs without any person (RUN_INPUTS of the builder): a
  report summary ships when it backs a published number;
- the stores, run folders and checkpoints;
- the v1 code of the SMPL+H body and of the AMASS readers, the v1 configurations and the scripts
  that cannot start without them: their inputs are distributed only after registration;
- the video round trip, until its first result.

## The gate

The export is scanned after it is built. The build refuses to finish on a software vendor's or
product's name, a credential-shaped string, French prose in a document, a held-out membership
list, a person map, an AMASS take name in a data file or a binary data file; it reports local
paths, third-party email addresses, provenance wording and dashes for review. Its word lists are
stored as sha256 digests of the words.
