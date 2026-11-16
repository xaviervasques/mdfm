"""The shipped package imports without the never-shipped SMPL+H and AMASS modules (plan phase E step 2,
docs/OPEN_BODY_PORT_DESIGN.md section 22; plan decision P10 of docs/MODEL_OPEN_PORT_PLAN.md).

A fresh interpreter with MDFM_SMPLH_REFERENCE and MDFM_BODY_MODEL removed from its environment imports every module
of the package (pkgutil.walk_packages) but the five that never ship, and none of the five enters sys.modules; the same
walk with the five made unimportable (a finder that raises ModuleNotFoundError, as in the public tree where the files
are absent) imports every module too. The rules the shipped code took out of those modules: the lockbox flag
(mdfm/data/lockbox_flag.gates_override, read by g1.G1Set and the open G1 sets, by open_mocap_raw.load_take and,
re-exported, by the AMASS readers) and the sex codes (mdfm/body/sex.py, read by kernels/force.py and eval/infer.py);
in the working tree the v1 modules re-export the same objects, and mdfm.model.mfm's N_BETAS equals smplh's. In the
public tree the checks that need a v1 module skip."""
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import types

import pytest

from mdfm.data.canonical import Refused

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the five never-shipped modules (scripts/build_public_tree.py NEVER_SHIP; tests/test_synth_reference_isolation.py
# checks the list against it in the working tree)
NEVER_SHIPPED = ("mdfm.body.smplh", "mdfm.synth.smplh_reference", "mdfm.data.amass", "mdfm.data.amass_index",
                 "mdfm.data.body_store")

WALK = """
import importlib, importlib.abc, json, pkgutil, sys
NEVER = {never!r}
class Absent(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name in NEVER:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        return None
if {block!r}:
    sys.meta_path.insert(0, Absent())
import mdfm
names = sorted(m.name for m in pkgutil.walk_packages(mdfm.__path__, "mdfm.") if m.name not in NEVER)
fails = {{}}
for n in names:
    try:
        importlib.import_module(n)
    except BaseException as e:
        fails[n] = type(e).__name__ + ": " + str(e)
print(json.dumps({{"modules": len(names), "fails": fails, "loaded": sorted(m for m in NEVER if m in sys.modules)}}))
"""


def _walk(block):
    env = {k: v for k, v in os.environ.items() if k not in ("MDFM_SMPLH_REFERENCE", "MDFM_BODY_MODEL")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = TREE
    out = subprocess.run([sys.executable, "-c", WALK.format(never=set(NEVER_SHIPPED), block=block)], cwd=TREE,
                         env=env, capture_output=True, text=True, encoding="utf-8", timeout=1800)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_every_shipped_module_imports_and_loads_no_never_shipped_module():
    r = _walk(block=False)
    assert r["modules"] >= 130, r["modules"]
    assert r["fails"] == {}, r["fails"]
    assert r["loaded"] == [], r["loaded"]


def test_every_shipped_module_imports_with_the_never_shipped_modules_absent():
    r = _walk(block=True)
    assert r["modules"] >= 130, r["modules"]
    assert r["fails"] == {}, r["fails"]


def test_the_lockbox_flag_is_granted_to_mdfm_gates_callers_only():
    from mdfm.data import lockbox_flag as LF
    frame = types.SimpleNamespace
    assert LF.gates_override(False, frame(f_globals={"__name__": "tests.x"})) is False
    assert LF.gates_override(False, frame(f_globals={"__name__": "mdfm.gates.g1"})) is False
    for name in ("mdfm.gates", "mdfm.gates.g1", "mdfm.gates.g1_open"):
        assert LF.gates_override(True, frame(f_globals={"__name__": name})) is True
    for name in ("mdfm.gatesx", "mdfm.data.open_mocap_raw", "mdfm.train.loop", "__main__", "test_public_imports", ""):
        with pytest.raises(Refused, match="mdfm/gates"):
            LF.gates_override(True, frame(f_globals={"__name__": name}))


def test_the_g1_set_and_the_open_reader_refuse_the_flag_outside_mdfm_gates(tmp_path):
    """g1.G1Set (and the open G1 sets, which inherit its __init__) and open_mocap_raw.load_take read the flag through
    lockbox_flag before they open any file: from this test module the flag is refused."""
    from mdfm.data import open_mocap_raw as OMR
    from mdfm.gates import g1 as G
    with pytest.raises(Refused, match="mdfm/gates"):
        G.G1Set("dev", root=str(tmp_path), allow_lockbox=True)
    with pytest.raises(Refused, match="mdfm/gates"):
        OMR.load_take({"source": OMR.GAIT120, "subject_ns": "x", "person_id": "x", "key": "x"}, allow_lockbox=True)
    assert OMR._gates_granted(False, sys._getframe()) is False


def test_the_sex_codes():
    from mdfm.body import sex as SX
    from mdfm.eval import infer as I
    from mdfm.kernels import force as F
    assert SX.GENDERS == ("female", "male", "neutral")
    for label, code in (("female", 0), ("F", 0), (" f ", 0), (b"F", 0), ("male", 1), ("M", 1), (b"male", 1),
                        ("neutral", 2), ("unknown", 2), ("", 2), (None, 2)):
        assert SX.gender_code(label) == code, label
    with pytest.raises(ValueError):
        SX.gender_code("other")
    assert I.gender_code is SX.gender_code
    assert F.sex_codes("M") == 1 and F.sex_codes("female") == 0 and F.sex_codes(2) == 2


def test_the_v1_modules_reexport_the_shipped_rules():
    """In the working tree: smplh's sex codes, amass_index's flag rule and mfm's N_BETAS are the shipped objects."""
    SM = pytest.importorskip("mdfm.body.smplh")
    I = pytest.importorskip("mdfm.data.amass_index")
    from mdfm.body import sex as SX
    from mdfm.data import lockbox_flag as LF
    from mdfm.model import mfm as M
    assert SM.gender_code is SX.gender_code and SM.GENDERS is SX.GENDERS and SM._GENDER_CODES is SX._GENDER_CODES
    assert I.gates_override is LF.gates_override
    assert M.N_BETAS == SM.N_BETAS == 16 and M.GENDERS is SM.GENDERS


def test_provenance_reads_the_amass_index_only_where_it_exists():
    from mdfm.train import provenance as PV
    present = importlib.util.find_spec("mdfm.data.amass_index") is not None
    got = PV._amass_index()
    assert (got is not None) == present
    if present:
        assert got is importlib.import_module("mdfm.data.amass_index")
