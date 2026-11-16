import importlib.util
import os
import re
import sys

import pytest

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TREE not in sys.path:
    sys.path.insert(0, TREE)
import mdfm  # noqa: E402,F401  (sets the torch cache folder before torch is imported)

# The GPU queue runner writes cache/runs/queue/SLOW while a long training runs slower than its baseline, and several
# probe scripts wait while that file exists. A test must never wait on the live queue: inside the tests the real
# tree's SLOW file reads as absent. Tests of the SLOW logic use their own temporary trees and are not affected.
_REAL_SLOW = os.path.normcase(os.path.normpath(os.path.join(TREE, "cache", "runs", "queue", "SLOW")))
_real_exists = os.path.exists


def _exists_without_live_slow(path):
    try:
        if os.path.normcase(os.path.normpath(os.fspath(path))) == _REAL_SLOW:
            return False
    except TypeError:
        pass
    return _real_exists(path)


@pytest.fixture(autouse=True)
def _ignore_live_slow_flag(monkeypatch):
    monkeypatch.setattr(os.path, "exists", _exists_without_live_slow)
    yield


# The v1 model (MFM with the SMPL+H skeleton) is built only behind the reference flag of mdfm/body/backend.py (plan
# docs/MODEL_OPEN_PORT_PLAN.md, step M4). The tests of v1 run it as the reference, so the flag is set for the session
# (module and session fixtures build v1 models too); in the test functions of a module that names the flag itself and
# builds no MFM (it tests the refusal or sets the flag where it needs it), of the synthetic patient's tests
# (test_synth_*) and of the open model's tests (test_*open*), it is removed, so that they run with the environment as
# a user's shell holds it. The flag only allows the reference; it never selects it.
_FLAG = "MDFM_SMPLH_REFERENCE"
_FLAG_OUTSIDE = os.environ.get(_FLAG)
os.environ[_FLAG] = "1"
_OWN_FLAG = {}


def _module_owns_flag(path):
    key = os.fspath(path)
    if key not in _OWN_FLAG:
        name = os.path.basename(key)
        own = name.startswith("test_synth_") or "open" in name
        if not own:
            try:
                with open(key, encoding="utf-8") as f:
                    text = f.read()
                names_flag = _FLAG in text or ".REFERENCE_ENV" in text
                own = names_flag and "MFM(" not in text and "mfm_config(" not in text
            except OSError:
                own = True
        _OWN_FLAG[key] = own
    return _OWN_FLAG[key]


@pytest.fixture(autouse=True)
def _v1_reference_flag(request, monkeypatch):
    path = getattr(request.node, "path", None) or getattr(request.node, "fspath", None)
    if path is not None and _module_owns_flag(path):
        if _FLAG_OUTSIDE is None:
            monkeypatch.delenv(_FLAG, raising=False)
        else:
            monkeypatch.setenv(_FLAG, _FLAG_OUTSIDE)
    yield


# The marker "live" (task QROB): tests whose outcome follows files that queue jobs, readings or the MoveAtlas tree
# write or move, listed with their reasons in tests/live_tests.txt (helpers: tests/live_marker.py).
# scripts/run_suite.py decides its green marker with -m "not live" and runs -m live in a second pass that never
# blocks the queue.
_spec = importlib.util.spec_from_file_location("mdfm_tests_live_marker",
                                               os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_marker.py"))
LM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LM)


def pytest_configure(config):
    config.addinivalue_line("markers", LM.LIVE_DOC)


# The public tree (a curated export, recognised by its generated PUBLIC_TREE.md) carries the code and the tests but
# not the run-time inputs of the working tree: the reports, the rule drafts, the proposals and the stores. There, and
# only there, a test that stops on one of those missing inputs is reported as skipped with the input's name, and a test
# that reads this tree's git history is skipped while the copy has none. In the working tree nothing changes: a
# missing input fails as before.
PUBLIC_COPY = os.path.isfile(os.path.join(TREE, "PUBLIC_TREE.md"))
if PUBLIC_COPY:
    # the public copy's machine defaults are placeholders: the synthetic patient's clip store goes to a folder of the
    # system's temporary directory, outside the tree, unless MDFM_SYNTH_VIEWER names one
    import tempfile
    os.environ.setdefault("MDFM_SYNTH_VIEWER", os.path.join(tempfile.gettempdir(), "mdfm_public_synth_viewer"))
_INPUT_DIRS = tuple(os.path.normcase(os.path.join(TREE, *d.split("/"))) + os.sep
                    for d in ("reports", "docs/rules", "docs/proposals", "docs/notes", "cache", "configs"))
# the v1 SMPL+H and AMASS modules, which the public copy never carries (plan decision P10, plan phase E step 2): a
# test that reaches one of them there is reported as skipped with the module's name
_NEVER_SHIPPED = ("mdfm.body.smplh", "mdfm.synth.smplh_reference", "mdfm.data.amass", "mdfm.data.amass_index",
                  "mdfm.data.body_store")


def _missing_input(exc):
    """The tree-relative path of a missing run-time input named by a FileNotFoundError, or None."""
    names = [getattr(exc, "filename", None)] + [m.group(0) for m in re.finditer(r"[A-Za-z]:[\\/][^'\"\s]+", str(exc))]
    for n in names:
        if not n:
            continue
        p = os.path.normcase(os.path.normpath(os.path.join(TREE, os.fspath(n))))
        if p.startswith(_INPUT_DIRS) and not os.path.exists(p):
            return os.path.relpath(p, os.path.normcase(TREE)).replace(os.sep, "/")
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if not PUBLIC_COPY or call.excinfo is None:
        return
    rep = outcome.get_result()
    exc = call.excinfo.value
    why = None
    if isinstance(exc, FileNotFoundError):
        rel = _missing_input(exc)
        if rel:
            why = f"a run-time input of the working tree that this public copy does not carry: {rel}"
    elif isinstance(exc, ImportError) and (getattr(exc, "name", None) in _NEVER_SHIPPED
                                           or any(f"'{m.rsplit('.', 1)[1]}' from '{m.rsplit('.', 1)[0]}'" in str(exc)
                                                  for m in _NEVER_SHIPPED)):
        why = f"the v1 SMPL+H and AMASS code, which this public copy does not carry: {exc}"
    elif "not a git repository" in str(exc):
        why = "this public copy has no git history yet"
    if why and rep.failed:
        rep.outcome = "skipped"
        rep.longrepr = (str(item.path), item.location[1] or 0, why)


@pytest.hookimpl(tryfirst=True)            # before the deselection of -m reads the markers
def pytest_collection_modifyitems(config, items):
    LM.mark_items(items, TREE, pytest.mark.live)
