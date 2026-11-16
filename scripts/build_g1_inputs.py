"""B4.11: truth and inputs of the G1 harness for one split, through mdfm.gates.g1.build, and their report.

dev and calib are written under cache/g1/g1_v1/<split>/ (gitignored). The lockbox goes to the gates-only
store and needs docs/rules/G1_RULE.md approved by the owner and registered: mdfm.gates refuses it before
any lockbox row is listed otherwise (task B4.21). This script passes no lockbox flag: only mdfm/gates
does. Report: reports/gates/g1/inputs_<split>/summary.md, build.json and _provenance.json (shard counts,
dropped short takes, valid shares of the IMU samples and camera detections, hashes).

Run: <conda>/envs/mdfm/python.exe scripts/build_g1_inputs.py --split dev
     <conda>/envs/mdfm/python.exe scripts/build_g1_inputs.py --split calib
Options: --no-inputs (truth only), --limit N (first N takes per archive), --archives A,B, --force,
--device (the float64 reference IMU renderer: cuda when available, else cpu), --threads (torch CPU threads).
"""
import argparse
import json
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)


def parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"),
                   help="MoveAtlas working tree (read-only)")
    p.add_argument("--split", required=True, choices=["dev", "calib", "lockbox"])
    p.add_argument("--root", default=None, help="input root (default: mdfm.gates.g1.default_root(split))")
    p.add_argument("--archives", default=None, help="comma-separated archives of the split (default: all)")
    p.add_argument("--limit", type=int, default=None, help="first N takes of each archive (smoke runs)")
    p.add_argument("--no-inputs", action="store_true", help="truth only, no rendered inputs")
    p.add_argument("--force", action="store_true", help="rebuild shards that are up to date")
    p.add_argument("--device", default=None, help="reference IMU renderer device, float64 (default cuda when available)")
    p.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    p.add_argument("--out", default=None, help="report folder (default reports/gates/g1/inputs_<split>)")
    a = p.parse_args()
    a.moveatlas = a.moveatlas.replace("\\", "/").rstrip("/")
    a.out = a.out or os.path.join(TREE, "reports", "gates", "g1", f"inputs_{a.split}")
    return a


def main():
    args = parse()
    os.environ["MOVEATLAS_ROOT"] = args.moveatlas
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    import mdfm  # noqa: F401
    from mdfm.data import amass_index
    from mdfm.gates import common, g1
    sys.path.insert(0, os.path.join(TREE, "scripts"))
    from _common import moveatlas_head
    torch.set_num_threads(args.threads)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    archives = args.archives.split(",") if args.archives else None
    man = g1.build(args.split, root=args.root, archives=archives, limit=args.limit, render=not args.no_inputs,
                   force=args.force, log=lambda *m: print(*m, flush=True), device=device)
    root = g1.default_root(args.split) if args.root is None else args.root
    shards = {}
    for a in sorted(man["shards"]):
        with open(os.path.join(root, args.split, a, "shard.json"), encoding="utf-8") as f:
            shards[a] = json.load(f)
    wall = time.time() - t0
    with open(os.path.join(TREE, "reports", "amass_split", "hashes.json"), encoding="utf-8") as f:
        hashes = json.load(f)
    want = hashes["person_lists"].get(args.split, {})
    complete = args.limit is None and archives is None
    rows = []
    tot = {k: 0 for k in ("n_takes", "n_dropped_short", "n_windows", "n_knots", "n_cam", "imu_valid", "cam_valid",
                          "n_treadmill_takes", "n_persons")}
    for a, m in shards.items():
        for k in tot:
            tot[k] += m.get(k, 0)
        rows.append(m)
    lines = [f"# G1 inputs: {args.split} (B4.11)", "",
             f"Built by scripts/build_g1_inputs.py with mdfm/gates/g1.py (rule draft docs/rules/G1_RULE.md, "
             f"parameters g1.PARAMS); provenance in _provenance.json. Root: {common.rel(root)}/{args.split} "
             f"({'gates-only store' if args.split == 'lockbox' else 'gitignored cache'}). Truth: 22 body joints and the "
             "root yaw at 100 Hz in each window's frame. Inputs: six IMUs (the wrist_L column is the single-wrist input) "
             "after the band copy, and one camera per window at 30 fps through the tracker process.", "",
             f"Manifest SHA-256: {man['manifest_sha256']}. Input configuration SHA-256: {man['config_sha256']}. "
             f"Persons: {man['n_persons']}, list SHA-256 {man['persons_sha256']}"
             + (f" (equal to reports/amass_split/hashes.json: {man['persons_sha256'] == want.get('sha256')})"
                if complete and want else " (subset build)") + ".", "",
             "| archive | takes | dropped under 2 s | treadmill takes | persons | windows | knots | hours | IMU samples valid | "
             "camera frames | camera detections valid | build s |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in rows:
        imu_share = m["imu_valid"] / (m["n_knots"] * len(g1.SITES)) if m["render"] and m["n_knots"] else None
        cam_share = m["cam_valid"] / (m["n_cam"] * 15) if m["render"] and m["n_cam"] else None
        lines.append(f"| {m['archive']} | {m['n_takes']:,} | {m['n_dropped_short']:,} | {m.get('n_treadmill_takes', 0):,} | "
                     f"{m['n_persons']} | {m['n_windows']:,} | {m['n_knots']:,} | {m['n_knots'] / 100 / 3600:.2f} | "
                     + (f"{imu_share:.4f}" if imu_share is not None else "not built") + f" | {m['n_cam']:,} | "
                     + (f"{cam_share:.4f}" if cam_share is not None else "not built") + f" | {m['seconds']:.0f} |")
    render = all(m["render"] for m in rows)
    lines.append(f"| all | {tot['n_takes']:,} | {tot['n_dropped_short']:,} | {tot['n_treadmill_takes']:,} | "
                 f"{man['n_persons']} | {tot['n_windows']:,} | {tot['n_knots']:,} | {tot['n_knots'] / 100 / 3600:.2f} | "
                 + (f"{tot['imu_valid'] / (tot['n_knots'] * len(g1.SITES)):.4f}" if render else "not built")
                 + f" | {tot['n_cam']:,} | "
                 + (f"{tot['cam_valid'] / (tot['n_cam'] * 15):.4f}" if render and tot['n_cam'] else "not built")
                 + f" | {sum(m['seconds'] for m in rows):.0f} |")
    lines += ["", "Hours count knots / 100 Hz inside kept windows. A camera detection is a joint the tracker process "
              "reports (in the image, not dropped, its frame's pelvis detected); the static camera of sample_camera can "
              "lose a walking subject in a long window (open item of B4.8 and item 6 of the rule draft).", "",
              f"Code SHA-256 of the input path (every shard): "
              f"{', '.join(sorted({m['code_sha256'] for m in rows}))}. Wall time of this run: "
              f"{wall:.0f} s (shards already built are kept)."]
    common.write_text(os.path.join(args.out, "summary.md"), "\n".join(lines) + "\n")
    common.write_text(os.path.join(args.out, "build.json"),
                      json.dumps({"manifest": man, "shards": {a: {k: v for k, v in m.items() if k != "config"}
                                                              for a, m in shards.items()},
                                  "config": rows[0]["config"] if rows else None}, indent=1) + "\n")
    prov = {"script": "scripts/build_g1_inputs.py", "moveatlas_root": args.moveatlas,
            "moveatlas_head": moveatlas_head(args.moveatlas), "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": sys.version.split()[0], "torch": torch.__version__, "split": args.split, "limit": args.limit,
            "archives": archives, "render": not args.no_inputs, "render_device": device, "threads": args.threads,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "index_sha256": hashes["index_csv_sha256"],
            "lockbox_file_sha256": amass_index.LOCKBOX_SHA256}
    common.write_text(os.path.join(args.out, "_provenance.json"), json.dumps(prov, indent=1) + "\n")
    print(f"{args.split}: {tot['n_windows']} windows, {tot['n_knots']} knots, {man['n_persons']} persons, "
          f"manifest {man['manifest_sha256'][:12]}, {wall:.0f} s")


if __name__ == "__main__":
    main()
