"""B4.11: the G1 gate harness (mdfm/gates/g1.py) run by hand: the population-prior trivial baseline on the dev
archives now; the lockbox read at B4.21, once the owner has approved and registered docs/rules/G1_RULE.md.

--split dev (default): the population prior of reports/gates/g1/population_prior/prior.json (built from the
train body store and given its conformal factor on the calib inputs when absent or with --rebuild-prior)
scored on the dev inputs of scripts/build_g1_inputs.py; writes reports/gates/g1/population_prior_dev/:
g1.json, g1.md, per_person.csv, per_archive.csv, windows.csv and _provenance.json. The lockbox read log
reports/gates/lockbox_reads.csv is created when absent; a dev run never writes a row into it.
--split lockbox: mdfm.gates refuses it until the G1 rule is registered; after that the read is logged
before any lockbox array opens and the outputs go to reports/gates/g1/<checkpoint_sha256>/. A second read of
the same checkpoint and rule needs --reread with --reason, which the log keeps.
--register-rule --approved-on YYYY-MM-DD: the owner's registration of the approved rule file (after the
owner has changed its status line); a session never runs it.

Run: <conda>/envs/mdfm/python.exe scripts/gate_g1.py
"""
import argparse
import json
import math
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)


def parse():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"),
                   help="MoveAtlas working tree (read-only)")
    p.add_argument("--split", default="dev", choices=["dev", "lockbox"])
    p.add_argument("--root", default=None, help="G1 input root (default: mdfm.gates.g1.default_root(split))")
    p.add_argument("--prior", default=os.path.join(TREE, "reports", "gates", "g1", "population_prior", "prior.json"))
    p.add_argument("--rebuild-prior", action="store_true", help="recompute the prior from the train body store")
    p.add_argument("--out", default=None, help="output folder (default reports/gates/g1/population_prior_dev)")
    p.add_argument("--reread", action="store_true",
                   help="lockbox only: a second read of the same checkpoint and rule, logged with --reason")
    p.add_argument("--reason", default="", help="lockbox only: the logged reason of a --reread")
    p.add_argument("--register-rule", action="store_true", help="owner only: register the approved G1 rule")
    p.add_argument("--approved-on", default=None, help="owner only: the date of the status line, YYYY-MM-DD")
    a = p.parse_args()
    a.moveatlas = a.moveatlas.replace("\\", "/").rstrip("/")
    return a


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "not evaluated"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{x:.{nd}f}"


def build_prior(args, common, g1):
    """The population prior from the train body store, its conformal factor on the calib inputs, and its
    report (reports/gates/g1/population_prior/)."""
    from mdfm.data.body_store import BodyStore
    t0 = time.time()
    store = BodyStore(split="train")
    prior = g1.population_prior(store)
    t_prior = time.time() - t0
    calib = g1.G1Set("calib", args.root)
    fit = g1.fit_conformal(calib, prior)
    prior.c_wrist = fit["c"]
    prior.meta.update({"conformal": fit, "calib_manifest_sha256": calib.manifest["manifest_sha256"],
                       "calib_persons_sha256": calib.manifest["persons_sha256"], "seconds": round(time.time() - t0, 1)})
    prior.save(args.prior)
    mp = prior.mean_pose
    lines = ["# G1 population prior (B4.11)", "",
             "The trivial baseline of G1 (spec section 6; rule draft docs/rules/G1_RULE.md section 7), from the train "
             "persons only: built by scripts/gate_g1.py with mdfm.gates.g1.population_prior on the train body store "
             f"({prior.meta['store_root']}) and given its conformal factor on the calib inputs. prior.json holds the "
             "arrays (statistics of AMASS fits, not SMPL+H arrays).", "",
             f"- Mean pose: pelvis-relative joints in the pelvis-heading frame, averaged over every "
             f"{prior.meta['stride']}th knot of the {prior.meta['n_takes']:,} train takes: {prior.meta['n_knots_mean']:,} "
             f"knots of {len(prior.meta['train_persons'])} persons (list SHA-256 {prior.meta['train_persons_sha256']}).",
             f"- Wrist draws: {prior.meta['n_draws']} left-wrist positions (SMPL joint {g1.WRIST_JOINT}) in the same frame, "
             f"one uniform knot of each of {prior.meta['n_draws']} subject-balanced train windows (seed "
             f"{prior.meta['seed']}), from {len(prior.meta['draw_persons'])} distinct persons.",
             f"- Conformal factor on calib: c = {fit['c']:.4f} over {fit['n_frames']:,} calib knots "
             f"({calib.manifest['n_persons']} persons, manifest SHA-256 {calib.manifest['manifest_sha256']}); coverage "
             f"at c {fit['coverage_at_c']:.4f}; degenerate predictives {fit['n_degenerate']}.",
             f"- Height of the mean pose (head joint above the pelvis): {1000 * (mp[15, 1] - mp[0, 1]):.1f} mm; wrist draw "
             f"spread (sd per axis x, y, z): "
             + ", ".join(f"{1000 * s:.1f}" for s in prior.draws.std(0, ddof=1)) + " mm.",
             f"- Prior SHA-256 (the checkpoint hash of a trivial-baseline run): {prior.sha256}.",
             f"- Time: mean pose and draws {t_prior:.0f} s, all {prior.meta['seconds']:.0f} s."]
    common.write_text(os.path.join(os.path.dirname(args.prior), "summary.md"), "\n".join(lines) + "\n")
    return prior


def markdown(args, g1, res, sec, pa, gset, prior, rule_sha, rule_status, reads, head, wall, cfg_sha):
    L = [f"# G1 dry run: population prior on the {args.split} archives (B4.11)", "",
         "The G1 harness (mdfm/gates/g1.py) run end to end with the trivial baseline of spec section 6, before any "
         "model exists: this is the stage-0 number of each line on the development archives. Written by "
         "scripts/gate_g1.py; the numbers are in g1.json, per_person.csv, per_archive.csv and windows.csv.", "",
         f"Rule: docs/rules/G1_RULE.md, status {rule_status!r}, SHA-256 {rule_sha} (not registered; every choice "
         "below may change when the owner reads it). Folds: dev and calib are the proposal of spec question 5. "
         f"Lockbox reads logged in reports/gates/lockbox_reads.csv: {reads} (this run reads no lockbox row). "
         f"MoveAtlas: {head}.", "",
         f"Set: {len(gset)} windows, {int(gset.windows['n'].sum()):,} knots ({gset.windows['n'].sum() / 100 / 3600:.2f} h) "
         f"of {len(gset.persons)} persons in {gset.windows['archive'].nunique()} archives; input manifest SHA-256 "
         f"{gset.manifest['manifest_sha256']}, configuration {cfg_sha}.", "",
         "## Lines", "",
         "Point: the mean over persons of the frame-weighted person means; interval: person-clustered bootstrap, "
         f"{g1.PARAMS['bootstrap_n']} resamples, seed {g1.PARAMS['bootstrap_seed']}, 95 percent. The trivial baseline "
         "is the predictor itself here.", "",
         "| line | point | 95% interval | pass | kill | verdict |", "|---|---|---|---|---|---|"]
    for r in res:
        nd = 3 if r.unit == "" else 1
        unit = f" {r.unit}" if r.unit else ""
        L.append(f"| {r.line} | {fmt(r.point, nd)}{unit} | {fmt(r.interval[0], nd)} to {fmt(r.interval[1], nd)} | "
                 f"{'at least' if r.higher_is_better else 'at most'} {r.pass_line:g} | "
                 f"{'below' if r.higher_is_better else 'above'} {r.kill_line:g} | {r.verdict} |")
    b = res[1].extra
    L += ["", f"G1b details: band width {fmt(b['band_width_mm'])} mm (conformal factor c = {prior.c_wrist:.4f} from the calib "
          f"archives); energy score {fmt(b['energy_score_mm'])} mm, which is the reference of the 20 percent bound "
          f"(ratio {fmt(b['energy_ratio_to_prior'], 3)}, so the prior cannot pass G1b by construction); width ratio to the "
          f"six-IMU predictive: {fmt(b['width_ratio_to_six_imu'], 3)} (the prior has no six-IMU predictive); degenerate "
          f"knots {b['degenerate_frames']} of {b['n_frames']:,}.", "",
          "## Secondary rows (no kill)", "",
          "| set | row | point | 95% interval | persons | windows |", "|---|---|---|---|---|---|"]
    for r in sec:
        nd = 3 if "coverage" in r["row"] else 1
        unit = "" if "coverage" in r["row"] else " mm"
        L.append(f"| {r['set']} | {r['row']} | {fmt(r['point'], nd)}{unit} | {fmt(r['lo'], nd)} to {fmt(r['hi'], nd)} | "
                 f"{r['n_persons']} | {r['n_windows']:,} |")
    L += ["", "Not evaluated for the prior (no model): zero refinement steps, acc-only wrist, truth re-rendered and "
          "re-scored, analysis by synthesis.", "", "## Per archive", "",
          "| archive | persons | windows | G1a mm | G1a 15 joints mm | G1b coverage | G1b width mm | G1b energy mm | "
          "G1c mm | G1c 22 joints mm |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in pa.iterrows():
        L.append(f"| {r['archive']} | {r['n_persons']} | {r['n_windows']:,} | {r['g1a_mpjpe_mm']:.1f} | "
                 f"{r['g1a_mpjpe15_mm']:.1f} | {r['g1b_coverage']:.3f} | {r['g1b_width_mm']:.1f} | {r['g1b_energy_mm']:.1f} | "
                 f"{r['g1c_pa_mpjpe_mm']:.1f} | {r['g1c_pa_mpjpe22_mm']:.1f} |")
    L += ["", "## Hashes", "",
          f"- predictor (population prior) SHA-256: {prior.sha256}",
          f"- training persons of the predictor: {len(prior.train_persons)}, list SHA-256 {prior.train_persons_sha256} "
          "(disjoint from the scored persons, asserted)",
          f"- scored persons list SHA-256: {gset.manifest['persons_sha256']}",
          f"- scoring code SHA-256 ({', '.join(g1.SCORE_CODE)}): {g1.code_sha256(g1.SCORE_CODE)}",
          f"- input code SHA-256 at build: {', '.join(sorted({s['code_sha256'] for s in gset.shards}))}",
          "", f"Wall time of the scoring: {wall:.0f} s."]
    return "\n".join(L) + "\n"


def main():
    args = parse()
    if args.split != "lockbox" and (args.reread or args.reason):
        sys.exit("--reread and --reason apply to a lockbox read only")
    os.environ["MOVEATLAS_ROOT"] = args.moveatlas
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    import mdfm  # noqa: F401
    from mdfm.data.canonical import Refused
    from mdfm.gates import common, g1
    sys.path.insert(0, os.path.join(TREE, "scripts"))
    from _common import moveatlas_head
    common.ensure_read_log()
    if args.register_rule:
        if not args.approved_on:
            sys.exit("--register-rule needs --approved-on YYYY-MM-DD, the date of the owner's status line")
        print(common.register_rule(g1.GATE, g1.RULE_PATH, args.approved_on, approved_by="owner"))
        return
    if args.rebuild_prior or not os.path.exists(args.prior):
        prior = build_prior(args, common, g1)
    else:
        prior = g1.PopulationPrior.load(args.prior)
    print(f"prior {prior.sha256[:12]}, c {prior.c_wrist}")
    t0 = time.time()
    try:
        gset, table, rule_sha, read = g1.run(args.split, prior, root=args.root, reread=args.reread, reason=args.reason,
                                             log=lambda *m: print(*m, flush=True))
    except Refused as e:
        sys.exit(f"refused: {e}")
    wall = time.time() - t0
    res = g1.lines(table, table, prior, rule_sha)
    sec = g1.secondary(table)
    pp, pa = g1.per_group(table, "person_id"), g1.per_group(table, "archive")
    out = args.out or os.path.join(TREE, "reports", "gates", "g1",
                                   "population_prior_dev" if args.split == "dev" else prior.sha256)
    with open(g1.RULE_PATH, encoding="utf-8") as f:
        status = common.rule_status(f.read())[0]
    head = moveatlas_head(args.moveatlas)
    reads = len(common.read_log())
    cfg_sha = gset.manifest["config_sha256"]
    results = {"gate": g1.GATE, "split": args.split, "predictor": prior.name, "rule_path": common.rel(g1.RULE_PATH),
               "rule_sha256": rule_sha, "rule_status": status, "rule_registered": common.registered(g1.GATE, rule_sha),
               "lines": [r.to_dict() for r in res], "secondary": sec,
               "not_evaluated": ["zero refinement steps", "acc-only wrist", "truth re-rendered and re-scored",
                                 "analysis by synthesis", "G1b width ratio to the six-IMU predictive"],
               "set": {"windows": len(gset), "knots": int(gset.windows["n"].sum()), "persons": len(gset.persons),
                       "persons_sha256": gset.manifest["persons_sha256"], "manifest_sha256": gset.manifest["manifest_sha256"],
                       "config_sha256": cfg_sha},
               "hashes": {"checkpoint_sha256": prior.sha256, "train_persons_sha256": prior.train_persons_sha256,
                          "scoring_code_sha256": g1.code_sha256(g1.SCORE_CODE),
                          "input_code_sha256": sorted({s["code_sha256"] for s in gset.shards}),
                          "params_sha256": common.sha256_json(g1.PARAMS)},
               "conformal": {"c_wrist": prior.c_wrist, **prior.meta.get("conformal", {})},
               "lockbox_reads_logged": reads, "read": read, "moveatlas_head": head, "seconds": round(wall, 1)}
    md = markdown(args, g1, res, sec, pa, gset, prior, rule_sha, status, reads, head, wall, cfg_sha)
    common.write(results, out, "g1", md)
    common.write_csv(pp, os.path.join(out, "per_person.csv"))
    common.write_csv(pa, os.path.join(out, "per_archive.csv"))
    common.write_csv(table, os.path.join(out, "windows.csv"))
    prov = {"script": "scripts/gate_g1.py", "moveatlas_root": args.moveatlas, "moveatlas_head": head,
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0], "torch": torch.__version__,
            "split": args.split, "prior": common.rel(args.prior)}
    common.write_text(os.path.join(out, "_provenance.json"), json.dumps(prov, indent=1) + "\n")
    for r in res:
        print(f"{r.line}: {r.point:.4g} [{r.interval[0]:.4g}, {r.interval[1]:.4g}] {r.unit} -> {r.verdict}")
    print(f"written {common.rel(out)}; {wall:.0f} s")


if __name__ == "__main__":
    main()
