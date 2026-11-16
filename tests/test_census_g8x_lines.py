"""Tests of scripts/census_g8x_lines.py (task G8X): the sheet census keeps classes, never values, skips lockbox
rows on the ID cell, refuses a changed header or an unknown child; the child folds are probes.person_folds; the
Spearman interval from counts behaves. Synthetic files only (tmp_path); nothing in the tree or in MoveAtlas is read."""
import importlib.util
import math
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("census_g8x_lines", os.path.join(HERE, "scripts", "census_g8x_lines.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def _sheet(path, rows, header=None):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(list(header or C.HEADER))
    for r in rows:
        ws.append(list(r))
    wb.save(path)
    return path


def _row(child_id, video, time, scorer, items):
    out = [child_id, video, time, scorer]
    for v in items:
        out += [v, 4, None]                  # item, _max, _pscore (the census never reads the last two)
    return out


def test_header_and_items():
    assert len(C.ITEMS) == 20 and len(C.HEADER) == 64
    assert C.HEADER[:4] == ("ID", "video", "time", "scorer")
    assert C.HEADER[4:7] == ("D_TRUNK_R_tD", "D_TRUNK_R_tD_max", "D_TRUNK_R_tD_pscore")
    assert C.HEADER[-3:] == ("CA_LLP_R_tA", "CA_LLP_R_tA_max", "CA_LLP_R_tA_pscore")


@pytest.mark.parametrize("v,cls", [(0, "scored"), (4, "scored"), (2.0, "scored"), ("3", "scored"), (999, "not_scored"),
                                   ("999", "not_scored"), (None, "blank"), ("  ", "blank"), (5, "other"),
                                   (1.5, "other"), (-1, "other"), ("x", "other"), (True, "other"),
                                   (float("nan"), "other")])
def test_classify(v, cls):
    assert C.classify(v) == cls


class _SpyRow:
    """A sheet row that records every cell index read and whether it was iterated."""

    def __init__(self, cells):
        self.cells = list(cells)
        self.indexed = []
        self.iterated = False

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, k):
        self.indexed.append(k)
        return self.cells[k]

    def __iter__(self):
        self.iterated = True
        return iter(self.cells)


def test_scan_rows_reads_only_the_id_of_a_lockbox_row():
    lock = _SpyRow(_row(3, 5, "T12", 1, ["not a grade"] * 20))
    train = _SpyRow(_row(7, 1, "T0", 1, [2] * 20))
    counts, per_video, n_lock, n_empty = C.scan_rows([lock, train], lockbox={"modys:child03"},
                                                     training={"modys:child07"})
    assert lock.indexed == [0] and not lock.iterated          # the ID cell, nothing else
    assert train.indexed[0] == 0                              # the ID cell before any other cell
    assert (n_lock, n_empty, set(per_video)) == (1, 0, {1})
    assert dict(counts[C.ITEMS[0]]["all"]) == {"scored": 1}


def test_scan_rows_empty_rows_and_rows_without_an_id():
    _, per_video, n_lock, n_empty = C.scan_rows([(), (None,) * 64, (None,) * 70], lockbox=set(),
                                                training={"modys:child07"})
    assert (n_empty, n_lock, per_video) == (3, 0, {})
    with pytest.raises(ValueError, match="no ID"):
        C.scan_rows([[None, 4, "T0", 1] + [0] * 60], lockbox=set(), training={"modys:child07"})


def test_read_sheet_skips_lockbox_and_keeps_classes(tmp_path):
    items_a = [1] * 10 + [999] * 10
    items_b = [0] * 19 + [None]
    garbage = ["not a grade"] * 20               # a lockbox row's item cells are never touched: no error, no count
    p = _sheet(tmp_path / "s.xlsx", [_row(7, 1, "T0", 1, items_a), _row(7, 1, "T0", 2, items_b),
                                     _row(3, 5, "T12", 1, garbage), _row(9, 2, "T3", 3, [999] * 20)])
    counts, per_video, n_lock = C.read_sheet(str(p), lockbox={"modys:child03"},
                                             training={"modys:child07", "modys:child09"}, log=lambda m: None)
    assert n_lock == 1
    assert set(per_video) == {1, 2}
    assert per_video[1] == {"person_ns": "modys:child07", "time": "T0", "rows": 2, "scorers": {1, 2}, "scored_rows": 2}
    assert per_video[2]["scored_rows"] == 0
    first, last = C.ITEMS[0], C.ITEMS[-1]
    assert dict(counts[first]["all"]) == {"scored": 2, "not_scored": 1}
    assert dict(counts[last]["all"]) == {"not_scored": 2, "blank": 1}
    assert sum(counts[first]["T12"].values()) == 0          # the lockbox row's visit holds nothing
    assert all(counts[i]["all"]["other"] == 0 for i in C.ITEMS)


def test_read_sheet_refuses_changed_header(tmp_path):
    h = list(C.HEADER)
    h[5] = "D_TRUNK_R_tD_MAX"
    p = _sheet(tmp_path / "s.xlsx", [], header=h)
    with pytest.raises(ValueError, match="header"):
        C.read_sheet(str(p), lockbox=set(), training={"modys:child07"}, log=lambda m: None)


def test_read_sheet_refuses_unknown_child(tmp_path):
    p = _sheet(tmp_path / "s.xlsx", [_row(11, 4, "T0", 1, [0] * 20)])
    with pytest.raises(ValueError, match="neither"):
        C.read_sheet(str(p), lockbox={"modys:child03"}, training={"modys:child07"}, log=lambda m: None)


def test_read_sheet_refuses_child_in_both_sets(tmp_path):
    p = _sheet(tmp_path / "s.xlsx", [])
    with pytest.raises(AssertionError):
        C.read_sheet(str(p), lockbox={"modys:child07"}, training={"modys:child07"}, log=lambda m: None)


def test_child_folds_are_person_folds():
    from mdfm.eval import probes
    kids = [f"modys:child{i:02d}" for i in range(1, 28)]
    f1 = C.child_folds(list(reversed(kids)))
    f2 = C.child_folds(kids)
    assert f1 == f2
    ids = np.array(sorted(kids))
    assert [f1[k] for k in ids] == list(probes.person_folds(ids, C.FOLDS, C.FOLD_SEED))
    sizes = sorted(np.bincount(list(f1.values())).tolist())
    assert sizes == [5, 5, 5, 6, 6]


def test_spearman_interval():
    lo7, hi7, w7 = C.spearman_interval(0.5, 7)
    lo27, hi27, w27 = C.spearman_interval(0.5, 27)
    assert lo7 < lo27 < 0.5 < hi27 < hi7 and w7 > w27
    assert math.isclose(math.atanh(0.5) - math.atanh(lo27), w27) and math.isclose(math.atanh(hi27) - math.atanh(0.5), w27)
    assert math.isclose(w27, C.Z95 * math.sqrt(1.06 / 24))
    with pytest.raises(ValueError):
        C.spearman_interval(0.5, 3)


def test_ast_constant(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("import os\nSTEMS = ('a', 'b')\nX = {'y'}\n", encoding="utf-8")
    assert C.ast_constant(str(p), "STEMS") == ("a", "b")
    assert C.ast_constant(str(p), "X") == {"y"}
    with pytest.raises(KeyError):
        C.ast_constant(str(p), "Z")
