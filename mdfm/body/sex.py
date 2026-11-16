"""The sex codes of the program, independent of the body model: the rows of a skeleton bank and the sex covariate.

0 female, 1 male, 2 neutral (also for an unknown sex): the codes of the sex covariate of mdfm.data.windows and the
gender rows of the skeleton banks (the v1 SMPL+H SkeletonBank and the open body's LatentSkeletonBank, which reads no
code). Moved here from mdfm/body/smplh.py on 2026-11-15 (plan phase E step 2, docs/OPEN_BODY_PORT_DESIGN.md section
22), which re-exports both names, so that the shipped code that reads a sex label (kernels/force.py, eval/infer.py)
imports no SMPL+H module: the same objects, the same codes.
"""

# SkeletonBank gender codes: 0 female, 1 male, 2 neutral (also for an unknown sex), the codes
# of the sex covariate of mdfm.data.windows
GENDERS = ("female", "male", "neutral")

_GENDER_CODES = {"female": 0, "f": 0, "male": 1, "m": 1, "neutral": 2, "unknown": 2, "": 2}


def gender_code(gender):
    """SkeletonBank code of a sex label, case-insensitive: 'female' or 'F' -> 0, 'male' or 'M'
    -> 1 (the AMASS and the source-table spellings), 'neutral', 'unknown', '' or None -> 2.
    Any other label raises ValueError rather than falling back to the neutral skeleton."""
    if gender is None:
        return 2
    g = gender.decode() if isinstance(gender, bytes) else str(gender)
    code = _GENDER_CODES.get(g.strip().lower())
    if code is None:
        raise ValueError(f"unknown sex label {gender!r}")
    return code
