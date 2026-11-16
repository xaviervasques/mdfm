"""HSIC strata of stage 2 (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.7 and section 9 item 10, task S2.8).

A strata table maps a person id to an opaque non-negative integer, the person's HSIC stratum. losses.hsic then
stratifies on (task, stratum) instead of the task alone, so that its source-invariance pressure acts between sources
within one stratum and never across strata. The table is written by scripts/build_hsic_strata.py into
cache/hsic_strata/strata.csv (columns person_id and stratum, non-lockbox persons only, asserted there and here); the
class each integer stands for is written by that script into its reports, whose folder no module of mdfm/ names
(tests/test_hsic_strata.py greps for it), so that the code reads integers and never a diagnosis name; a run config
pins the table by its SHA-256 instead.
Stratum 0 is the stratum of every window that carries none (the synthetic windows, the stage-1 replay and every
stage-1 run, Window.hsic_stratum's default); the script gives it to the healthy class, so that the healthy persons of
the stage-2 units share their strata with the replay and the synthetic windows, as in stage 1.

load(path, sha256=None) reads a table: the file's SHA-256 must equal sha256 when given (the run config pins it,
train/config.py, losses.hsic_strata_sha256); a header other than (person_id, stratum), an empty or repeated person
id, a stratum that is not a non-negative integer, a CODY name or a lockbox person (window_store.lockbox_persons)
raise ValueError (canonical.Refused for the last two). The window store attaches the stratum of a window's person
(window_store.WindowStore, argument strata) and refuses a store holding a person the table lacks.
"""
import csv
import hashlib
import os
from dataclasses import dataclass, field

from mdfm import paths

HEADER = ("person_id", "stratum")
DEFAULT_STRATUM = 0                 # Window.hsic_stratum of a window without a table entry (module docstring)
TABLE = "cache/hsic_strata/strata.csv"     # tree-relative; scripts/build_hsic_strata.py writes it


@dataclass(frozen=True)
class StrataTable:
    """person id -> stratum (module docstring); path, the file read; sha256, the SHA-256 of its bytes."""
    path: str
    sha256: str
    strata: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.strata)

    def __contains__(self, person_id):
        return str(person_id) in self.strata

    def stratum(self, person_id):
        """The stratum of a person; KeyError for a person the table lacks."""
        try:
            return self.strata[str(person_id)]
        except KeyError:
            raise KeyError(f"person {person_id!r} is not in the HSIC strata table {self.path}") from None

    def missing(self, person_ids):
        """Sorted person ids of person_ids that the table lacks."""
        return sorted({str(p) for p in person_ids} - set(self.strata))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def resolve(path):
    """An absolute path as given, a relative one under the tree (joined part by part, as train/names.resolve)."""
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.join(paths.TREE, *[p for p in path.replace("\\", "/").split("/") if p])


def load(path=TABLE, sha256=None):
    """The StrataTable of a strata file (module docstring for the checks)."""
    from mdfm.data import canonical
    from mdfm.data import window_store as WS
    p = resolve(str(path))
    got = sha256_file(p)
    if sha256 is not None and got != str(sha256):
        raise ValueError(f"{p}: SHA-256 {got}, the configuration names {sha256}")
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows or tuple(rows[0]) != HEADER:
        raise ValueError(f"{p}: header {rows[0] if rows else None}, expected {list(HEADER)}")
    out = {}
    lock = WS.lockbox_persons()
    for i, r in enumerate(rows[1:], 2):
        if len(r) != 2 or not r[0]:
            raise ValueError(f"{p} line {i}: {r}")
        pid, s = r
        if pid in out:
            raise ValueError(f"{p} line {i}: person {pid} repeated")
        if not s.isdigit():
            raise ValueError(f"{p} line {i}: stratum {s!r} is not a non-negative integer")
        if canonical.is_cody(pid.split(":", 1)[0], pid, pid):
            raise canonical.Refused(f"{p} line {i}: a CODY name ({pid})")
        if pid in lock:
            raise canonical.Refused(f"{p} line {i}: lockbox person {pid}")
        out[pid] = int(s)
    return StrataTable(p, got, out)
