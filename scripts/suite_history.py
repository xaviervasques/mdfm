"""Which tests turned the runner's suite runs red, and which of them the live list now keeps out of the gating pass.

Usage: python scripts/suite_history.py [--logs cache/runs/queue/logs] [--live-list tests/live_tests.txt] [--out F]

Reads every run_suite_<YYYYmmdd_HHMMSS>.log of --logs (AppleDouble sidecars left out), takes the FAILED and ERROR
ids of its gating pass (the lines before the line 'run_suite: live pass'; the whole log for the logs written before
the live pass existed, task QROB), and prints: the number of logs, of logs with a failed test, of logs whose failed
tests all match the live list (tests/live_tests.txt, read with tests/live_marker.py), and one row per failed test
id (live or gate, the number of logs naming it, the first and last log). Read-only; with --out the same text is
also written to F (LF, UTF-8).
"""
import argparse
import glob
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RS = _load("run_suite_for_history", os.path.join(HERE, "run_suite.py"))
LM = _load("live_marker_for_history", os.path.join(TREE, "tests", "live_marker.py"))


def gating_failures(text):
    """The FAILED and ERROR ids of a suite log before its live pass, in order, each once."""
    out = []
    for line in text.splitlines():
        if line.startswith(RS.LIVE_LINE):
            break
        m = RS.FAILED_ID.match(line.strip())
        if m and m.group(2).strip() not in out:
            out.append(m.group(2).strip())
    return out


def history(logs_dir, live_list):
    """(lines of the report, {test id: [log stamps]})."""
    pats = [p for p, _ in LM.read_live_list(live_list)]
    logs = sorted(p for p in glob.glob(os.path.join(logs_dir, "run_suite_*.log"))
                  if not os.path.basename(p).startswith("._"))
    hits, red, all_live = {}, 0, 0
    for p in logs:
        with open(p, encoding="utf-8", errors="replace") as f:
            ids = gating_failures(f.read())
        stamp = os.path.basename(p)[len("run_suite_"):-len(".log")]
        if ids:
            red += 1
            all_live += all(LM.is_live(i, pats) for i in ids)
        for i in ids:
            hits.setdefault(i, []).append(stamp)
    try:
        shown = os.path.relpath(live_list, TREE)
    except ValueError:                          # another drive
        shown = os.path.abspath(live_list)
    lines = [f"suite logs: {len(logs)}; with a failed test: {red}; whose failed tests all match the live list: "
             f"{all_live}; live list: {len(pats)} entries ({shown.replace(os.sep, '/')})",
             "kind  logs  first            last             test"]
    for i, stamps in sorted(hits.items()):
        lines.append(f"{'live' if LM.is_live(i, pats) else 'gate'}  {len(stamps):4d}  {stamps[0]}  {stamps[-1]}  {i}")
    return lines, hits


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--logs", default=os.path.join(TREE, "cache", "runs", "queue", "logs"))
    p.add_argument("--live-list", default=LM.LIVE_LIST)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    lines, _ = history(a.logs, a.live_list)
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
