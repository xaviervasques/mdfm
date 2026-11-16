"""The pytest marker "live" (task QROB): tests whose outcome follows files that queue jobs, readings or the MoveAtlas
tree write or move, listed in tests/live_tests.txt with the reason of each. tests/conftest.py applies the marker with
mark_items; scripts/run_suite.py decides its green marker with -m "not live" and runs -m live in a second pass that
never blocks the queue.

No import of mdfm or of a test library here, so that a synthetic test tree can copy this file and use it as is.
"""
import fnmatch
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LIVE_LIST = os.path.join(HERE, "live_tests.txt")
MARKER = "live"
LIVE_DOC = (f"{MARKER}: the outcome follows files that jobs, readings or the MoveAtlas tree write or move "
            "(tests/live_tests.txt); scripts/run_suite.py runs these in a second pass that never blocks the queue")


def read_live_list(path=LIVE_LIST):
    """[(node-id pattern, reason)] of a live list: one entry per line, the pattern, whitespace, the reason; blank
    lines and lines starting with '#' left out; backslashes of the pattern read as '/'. A missing file lists nothing."""
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return out
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split(None, 1)
        out.append((parts[0].replace("\\", "/"), parts[1].strip() if len(parts) > 1 else ""))
    return out


def tree_node_id(path, nodeid, tree):
    """The node id with its file path relative to `tree` (forward slashes), whatever pytest's rootdir: `path` is the
    item's file, `nodeid` pytest's own id ('<path>::<name>[...]')."""
    rel = os.path.relpath(os.fspath(path), tree).replace("\\", "/")
    rest = nodeid.split("::", 1)[1] if "::" in nodeid else ""
    return f"{rel}::{rest}" if rest else rel


def is_live(node_id, patterns):
    """Whether node_id, or node_id without its parameters ('[...]'), matches one of the fnmatch patterns."""
    base = node_id.split("[", 1)[0]
    return any(fnmatch.fnmatchcase(node_id, p) or fnmatch.fnmatchcase(base, p) for p in patterns)


def mark_items(items, tree, mark, path=LIVE_LIST):
    """Add `mark` (pytest.mark.live) to every item whose tree-relative node id the live list matches; returns the
    node ids marked."""
    patterns = [p for p, _ in read_live_list(path)]
    marked = []
    if not patterns:
        return marked
    for item in items:
        nid = tree_node_id(item.path, item.nodeid, tree)
        if is_live(nid, patterns):
            item.add_marker(mark)
            marked.append(nid)
    return marked
