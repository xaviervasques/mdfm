"""B4.10: the layout families of the synthetic factory, read from real records, and the renderer bench.

Layouts (mdfm/factory/layouts.py, FAMILY_RULES): for each source a family names, one admitted, non-lockbox
recording is read by layouts.record_streams (pilot sources: the first recording in (person, recording)
order of mdfm.data.sources.rows() that pairs with the streams the families need, through sources.pair
and windows.cut; capture24 and li_taati: the first row in (subject_ns, key) order of
canonical._select(tier RELEASE, lane body), no lockbox, EVAL or CODY row; records read by canonical.load
with its flags at False). Sites, rates, channel masks, acc_content and force units come from these
records; every family stream is then compared with the record of each source the family names (sites,
rate, channel mask), and a difference stops the script.

Bench (mdfm/factory/synth.py, float32 on the GPU unless stated), on train windows of the body store
(mdfm.data.body_store, cache/body_store unless MDFM_BODY_STORE):
- precision: the IMU of all 14 sites at 100 Hz before band_copy, noise off, mounts drawn from the generic
  prior, against the reference renderer factory/imu_render.render (float64 BodyTrajectory on the CPU,
  same knots, rest joints, mounts and times), for the renderer in float64 and in float32;
- throughput: per family, macro-batches of about 4 micro-steps of 16,384 tokens (at most 512 windows),
  rendered windows per second with band copies, without the body draw (store.sample, on the CPU, timed
  apart), after one warm-up batch; the time per render phase (SynthSampler.timing) and the machine's CPU
  use during the timed batches (psutil);
- memory: xsens13 windows totalling at least 262,144 IMU timestamps, peak CUDA memory above the memory
  allocated before the call.
Writes reports/layouts/: layouts.json, checks.csv, bench.json, summary.md, _provenance.json.
Run: <conda>/envs/mdfm/python.exe scripts/build_layouts.py [--no-bench]

Stage 2 (--stage2, task S2.6; spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.5): the S2a families of
layouts.STAGE2_FAMILY_RULES (pads_wrists, multipatho4, pocket_acc) built from every stored input record of the S2
store (MDFM_WINDOW_STORE_S2, else cache/window_store_s2, read under mdfm.data.sources.s2a_names() at tier RELEASE;
built by scripts/build_window_store.py --stage2 s2a), and every stored input record of the five S2a units compared
with its family (layouts.compare_record; weargait_pd:PD and the stroke patients with their stage-1 families
xsens13 and mocap3d of layouts.json). Writes only reports/layouts/layouts_s2.json, checks_s2.csv, summary_s2.md
and _provenance_s2.json; layouts.json and the other stage-1 files are not opened for writing, and no bench runs.
Run: <conda>/envs/mdfm/python.exe scripts/build_layouts.py --stage2
"""
import argparse
import csv
import json
import os
import sys
import time

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "scripts"))

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--moveatlas", default=os.environ.get("MOVEATLAS_ROOT", "<moveatlas>"),
               help="MoveAtlas working tree (read-only)")
p.add_argument("--out", default=os.path.join(TREE, "reports", "layouts"))
p.add_argument("--no-bench", action="store_true", help="layouts only")
p.add_argument("--seed", type=int, default=20260929)
p.add_argument("--reps", type=int, default=3, help="timed macro-batches per family")
p.add_argument("--precision-windows", type=int, default=16)
p.add_argument("--stage2", action="store_true", help="the S2a families from the S2 store (docstring)")
p.add_argument("--store", default=None, help="--stage2: the S2 store root (default MDFM_WINDOW_STORE_S2, else "
                                              "cache/window_store_s2)")
ARGS = p.parse_args()
ARGS.moveatlas = ARGS.moveatlas.replace("\\", "/").rstrip("/")
if not os.path.isfile(os.path.join(ARGS.moveatlas, "INDEX.md")):
    sys.exit(f"MoveAtlas tree not found at {ARGS.moveatlas} (no INDEX.md)")
os.environ["MOVEATLAS_ROOT"] = ARGS.moveatlas
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

import mdfm  # noqa: E402,F401
from mdfm import paths  # noqa: E402
from mdfm.factory import camera_render as CR  # noqa: E402
from mdfm.factory import layouts as L  # noqa: E402

paths.assert_not_moveatlas(ARGS.out)
os.makedirs(ARGS.out, exist_ok=True)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- bench
_PROC = []


def cpu_load(reset=False):
    """Mean CPU use since the last reset, in percent of the whole machine: {cpu_machine_pct, cpu_this_pct}
    (every process, and this process alone); {} without psutil."""
    try:
        import psutil
    except ImportError:
        return {}
    if not _PROC:
        _PROC.append(psutil.Process())       # cpu_percent(None) measures from the previous call on this object
    proc = _PROC[0]
    if reset:
        psutil.cpu_percent(None)
        proc.cpu_percent(None)
        return {}
    return {"cpu_machine_pct": psutil.cpu_percent(None), "cpu_this_pct": proc.cpu_percent(None) / psutil.cpu_count()}


def bench(layouts):
    import torch
    from mdfm.body.smplh import SMPLHBody
    from mdfm.body.trajectory import BodyTrajectory
    from mdfm.data.body_store import BodyStore
    from mdfm.factory import synth as SY
    from mdfm.factory.imu_render import SITES, render
    out = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    t0 = time.time()
    store = BodyStore(split="train")
    out["store"] = {"root": os.path.relpath(store.root, TREE).replace("\\", "/"), "takes": len(store),
                    "persons": int(len(store.persons)), "open_s": round(time.time() - t0, 2)}
    gen = np.random.default_rng(ARGS.seed)

    # precision against the reference renderer
    full = L.Layout("all_sites", (L.StreamLayout("imu", "input", (100.0,), tuple(SITES), ("111111000",) * 14,
                                                 "with_gravity"),), mount="generic", device="generic")
    bodies = store.sample(gen, ARGS.precision_windows)
    specs = [SY.RenderSpec(full, None, True) for _ in range(len(bodies.lengths))]
    seed = int(gen.integers(2 ** 62))
    prec = {}
    ref = None
    bank64 = SY.SkeletonBank(torch.float64)
    bodies64 = {g: SMPLHBody(n, dtype=torch.float64) for g, n in ((0, "female"), (1, "male"))}
    for name, dt in (("float64", torch.float64), ("float32", torch.float32)):
        sy = SY.SynthSampler(store, layouts=layouts, dtype=dt)
        ws = sy.render(bodies, specs, seed, band=False)
        if ref is None:                  # the reference on the float64 render's times and mounts
            ref = []
            for i, w in enumerate(ws):
                st = w.streams[0]
                g = int(bodies.gender[i])
                rest = bank64.rest(bodies.betas[i:i + 1].double(), torch.tensor([g]))[0]
                n = int(bodies.lengths[i])
                traj = BodyTrajectory(bodies64[g], bodies.rot6d[i, :n].double(), bodies.transl[i, :n].double(),
                                      bodies.betas[i].double(), rest=rest)
                mount = {s: torch.as_tensor(st.truth["mount"][k]).double() for k, s in enumerate(SITES)}
                t64 = st.truth["t_first_s"] + np.arange(len(st.t), dtype=np.float64) / st.fps
                sig, _, _ = render(traj, torch.from_numpy(t64), sites=SITES, mount=mount)
                ref.append(sig.numpy())
        ea = max(float(np.abs(w.streams[0].values[..., 0:3] - r[..., 0:3]).max()) for w, r in zip(ws, ref))
        eg = max(float(np.abs(w.streams[0].values[..., 3:6] - r[..., 3:6]).max()) for w, r in zip(ws, ref))
        pa = float(np.percentile(np.concatenate([np.abs(w.streams[0].values[..., 0:3] - r[..., 0:3]).ravel()
                                                 for w, r in zip(ws, ref)]), 99.9))
        pg = float(np.percentile(np.concatenate([np.abs(w.streams[0].values[..., 3:6] - r[..., 3:6]).ravel()
                                                 for w, r in zip(ws, ref)]), 99.9))
        prec[name] = {"device": str(sy.device), "max_acc_g": ea, "max_gyro_dps": eg, "p999_acc_g": pa,
                      "p999_gyro_dps": pg}
        log(f"precision {name}: acc {ea:.3g} g, gyro {eg:.3g} deg/s")
    prec["windows"] = len(ref)
    prec["samples"] = int(sum(r.shape[0] for r in ref))
    prec["max_ref_gyro_dps"] = float(max(np.abs(r[..., 3:6]).max() for r in ref))
    prec["max_ref_acc_g"] = float(max(np.abs(r[..., 0:3]).max() for r in ref))
    out["precision"] = prec

    sy = SY.SynthSampler(store, layouts=layouts, dtype=torch.float32)
    # memory: xsens13 windows totalling at least 262,144 IMU timestamps
    if torch.cuda.is_available():
        x = L.concrete(layouts["xsens13"], gen)
        n = 0
        bl = []
        while n < 262144:
            b = store.sample(gen, 64)
            bl.append(b)
            n += int((b.lengths - 1).sum())          # about one 100 Hz sample per knot interval
        bodies = bl[0]
        for b in bl[1:]:
            bodies = SY.concat_bodies(bodies, b)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t1 = time.time()
        ws = sy.render(bodies, [x] * len(bodies.lengths), gen)
        torch.cuda.synchronize()
        pts = int(sum(len(w.streams[0].t) for w in ws))
        out["memory"] = {"windows": len(ws), "imu_timestamps": pts, "chunk": sy.chunk,
                         "peak_gb": (torch.cuda.max_memory_allocated() - base) / 1e9, "seconds": time.time() - t1}
        log(f"memory: {pts} timestamps, peak {out['memory']['peak_gb']:.3f} GB")

    # throughput per family, then the default mixture
    thr = {}
    for name, lay in list(layouts.items()) + [("mixture", None)]:
        if lay is None:
            mix, tok = L.DEFAULT_MIXTURE, None
            nb = 128
        else:
            mix = {name: 1.0}
            tok = L.token_count(L.concrete(lay, np.random.default_rng(0)) if lay.k_sites else lay, 10.0)
            nb = int(min(512, max(8, round(4 * 16384 / tok))))
        b0, s0 = sy.draw(gen, nb, mix)
        sy.render(b0, s0, gen)                                          # warm-up
        sy.reset_timing()
        t_draw, reps, nwin, npts, ntok, fval = 0.0, [], 0, 0, 0, [0, 0]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        cpu_load(reset=True)
        for _ in range(ARGS.reps):
            t1 = time.time()
            b, s = sy.draw(gen, nb, mix)
            t2 = time.time()
            ws = sy.render(b, s, gen)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            reps.append(time.time() - t2)
            t_draw += t2 - t1
            nwin += len(ws)
            npts += sum(len(st.t) for w in ws for st in w.streams)
            ntok += sum(L.window_tokens(w) for w in ws)
            for st in (st for w in ws for st in w.streams if st.modality == "force"):
                v = st.valid[:, st.chan_mask[:, 8]]
                fval[0] += int(v.sum())
                fval[1] += v.size
        t_render = sum(reps)
        load = cpu_load()
        thr[name] = {**load, "macro_batch": nb, "tokens_10s": tok, "windows": nwin, "render_s": [round(x, 3) for x in reps],
                     "draw_s": round(t_draw, 3), "windows_per_s": nwin / t_render,
                     "windows_per_s_fastest": nb / min(reps),
                     "windows_per_s_with_draw": nwin / (t_render + t_draw), "kept_samples_per_window": npts / nwin,
                     "tokens_per_window": ntok / nwin, "render_ms_per_16384_tokens": 1000 * t_render * 16384 / ntok,
                     "split_ms_per_16384_tokens": {k: 1000 * v * 16384 / ntok for k, v in sy.timing.items()},
                     "peak_gb": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None}
        if fval[1]:
            thr[name]["force_valid_share"] = fval[0] / fval[1]
        log(f"{name}: {nb} windows per batch, {thr[name]['windows_per_s']:.1f} windows/s "
            f"({thr[name]['windows_per_s_with_draw']:.1f} with the body draw), "
            f"{thr[name]['render_ms_per_16384_tokens']:.0f} ms per 16,384 tokens")
    out["throughput"] = thr
    t1 = time.time()
    b, s = sy.draw(gen, 64)
    t2 = time.time()
    pairs = sy.render_pair(b, s, gen)
    out["pair"] = {"bodies": len(pairs), "windows": 2 * len(pairs), "draw_s": t2 - t1, "render_s": time.time() - t2}
    return out


def _cell_sites(s):
    if s.modality == "imu":
        return " ".join(s.sites)
    return "15 joints" if s.modality in ("pose2d", "pose3d") else "feet"


def _cell_channels(s):
    if s.modality == "imu":
        return s.channels
    if s.modality == "force":
        return "totals" + (", one plate" if s.plate else "")
    return "x, y" if s.mask and s.mask[0] == "110" else "x, y, z"


# ---------------------------------------------------------------- report
def summary(doc, checks, bench_doc, head):
    lines = ["# Layout families and the synthetic renderer (plan task B4.10)", "",
             "Generated by scripts/build_layouts.py; every number below is read from layouts.json, checks.csv "
             f"and bench.json of this directory. MoveAtlas commit: {head}. Rules in mdfm/factory/layouts.py "
             "and mdfm/factory/synth.py.", "", "## Families", "",
             "Sites, rates and channel masks read from one admitted, non-lockbox record per source (record "
             "column); tokens of a 10 s window (40 latent patches, spec 2.2).", "",
             "| family | stream | sites | fps | channels | mount | device | tokens (10 s) | record |",
             "|---|---|---|---|---|---|---|---|---|"]
    lay = L.load_layouts(os.path.join(ARGS.out, "layouts.json"))
    for name, fam in lay.items():
        tok = L.token_count(fam, 10.0) if fam.k_sites is None else "81 k + 41"
        for s in fam.streams:
            rec = s.record or {}
            where = f"{rec['source']} {rec['subject_ns']} {rec['key']}" if rec else "drawn"
            lines.append(f"| {name} | {s.modality} ({s.role}) | {_cell_sites(s)} | "
                         f"{', '.join(f'{f:g}' for f in s.fps)} | {_cell_channels(s)} | {fam.mount or ''} | "
                         f"{fam.device or ''} | {tok} | {where} |")
    k = lay["random_sites"].k_sites
    lines += ["", f"random_sites draws k uniformly in {k[0]} to {k[1]} and its rate among the IMU rates read; camera1 "
              "takes the rates of the camera family (the li_taati records carry no fps).", "",
              "## Checks against the records", "",
              f"{sum(r['verdict'] == 'equal' for r in checks)} of {len(checks)} family streams equal their records "
              "in sites, rate and channel mask (checks.csv).", ""]
    if not bench_doc:
        return "\n".join(lines) + "\n"
    pr = bench_doc["precision"]
    lines += ["## IMU against the reference renderer", "",
              f"{pr['windows']} train windows, {pr['samples']} samples, 14 sites at 100 Hz before band_copy, noise "
              f"off, generic mounts drawn (largest reference |acc| {pr['max_ref_acc_g']:.3g} g, |gyro| "
              f"{pr['max_ref_gyro_dps']:.4g} deg/s); errors of the float32 stream values against the float32 output "
              "of imu_render.render (float64 inside):", "",
              "| renderer | device | max acc g | p99.9 acc g | max gyro deg/s | p99.9 gyro deg/s |",
              "|---|---|---|---|---|---|"]
    for key in ("float64", "float32"):
        x = pr[key]
        lines.append(f"| {key} | {x['device']} | {x['max_acc_g']:.3g} | {x['p999_acc_g']:.3g} | "
                     f"{x['max_gyro_dps']:.3g} | {x['p999_gyro_dps']:.3g} |")
    lines += ["", "The spec's bound (1e-4 g, 1e-2 deg/s) holds in float64; float32 is stated as measured: the Hermite "
              "second derivative weighs knot values by about 6 rate^2, so float32 rounding of positions of a few "
              "metres reaches the 1e-3 g range.", ""]
    if "memory" in bench_doc:
        m = bench_doc["memory"]
        lines += ["## Memory", "", f"{m['imu_timestamps']:,} IMU timestamps ({m['windows']} xsens13 windows, chunks "
                  f"of {m['chunk']:,} points): peak {m['peak_gb']:.3f} GB above the memory allocated before the call "
                  f"(bound 4 GB), {m['seconds']:.2f} s.", ""]
    reps = len(next(iter(bench_doc["throughput"].values()))["render_s"])
    lines += ["## Rendered windows per second", "",
              f"{bench_doc['gpu']}, torch {bench_doc['torch']}, float32; per family, macro-batches of about 4 "
              f"micro-steps of 16,384 tokens (at most 512 windows), {reps} timed batches after a warm-up; mixture: "
              "batches of 128 windows of the default mixture. Render time covers the kinematics, the host-side "
              "noise, tracker, normalization and band copies; the body draw (BodyStore.sample, CPU) is timed apart. "
              "windows/s is over all timed batches, fastest over the quickest one. CPU columns: mean use of the whole "
              "machine during the timed batches, by every process and by this one; the difference is other jobs "
              "sharing the machine, which slow the host-side phases (the torch CPU work of the camera and pose "
              "streams most).", "",
              "| family | windows per batch | tokens per window | kept samples per window | windows/s | fastest | "
              "windows/s with the body draw | render ms per 16,384 tokens | peak GB | CPU all % | CPU this % |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, x in bench_doc["throughput"].items():
        cpu = " | ".join(f"{x[k]:.0f}" if k in x else "" for k in ("cpu_machine_pct", "cpu_this_pct"))
        fast = f"{x['windows_per_s_fastest']:.1f}" if "windows_per_s_fastest" in x else ""
        lines.append(f"| {name} | {x['macro_batch']} | {x['tokens_per_window']:.1f} | "
                     f"{x['kept_samples_per_window']:.0f} | {x['windows_per_s']:.1f} | {fast} | "
                     f"{x['windows_per_s_with_draw']:.1f} | {x['render_ms_per_16384_tokens']:.0f} | "
                     f"{x['peak_gb']:.2f} | {cpu} |")
    if all("split_ms_per_16384_tokens" in x for x in bench_doc["throughput"].values()):
        phases = list(next(iter(bench_doc["throughput"].values()))["split_ms_per_16384_tokens"])
        lines += ["", "Where the render time goes, ms per 16,384 tokens (SynthSampler.timing over the timed batches): plan "
                  "(clocks, nuisance and mount draws), kinematics (GPU passes with their copies to the host), streams "
                  "(host-side device noise, camera, tracker and pose normalization of factory/camera_render.py, force, "
                  "Window assembly), band_copy (data/window_store.py, on the host). The spec's step-time estimate counts "
                  "about 75 ms of rendering per micro-step (section 5).", "",
                  "| family | " + " | ".join(phases) + " |", "|---|" + "---|" * len(phases)]
        for name, x in bench_doc["throughput"].items():
            sp = x["split_ms_per_16384_tokens"]
            lines.append(f"| {name} | " + " | ".join(f"{sp[k]:.0f}" for k in phases) + " |")
    fv = {n: x["force_valid_share"] for n, x in bench_doc["throughput"].items() if "force_valid_share" in x}
    if fv:
        lines += ["", "Force totals are invalid where the body is in flight (both contact probabilities below 0.5, "
                  "where newton_bw leaves the split undefined), and band_copy masks the valid runs too short to "
                  "filter. Valid share of the present-foot samples after band_copy over the timed batches: "
                  + "; ".join(f"{n} {100 * v:.1f} percent" for n, v in fv.items()) + "."]
    pa = bench_doc["pair"]
    lines += ["", f"render_pair: {pa['bodies']} bodies of the default mixture rendered twice ({pa['windows']} windows) "
              f"in {pa['render_s']:.2f} s after a body draw of {pa['draw_s']:.2f} s.", ""]
    return "\n".join(lines) + "\n"


def stage2():
    """The S2a families and the check of every stored input record (module docstring)."""
    from _common import moveatlas_head
    from mdfm.data import sources as S
    from mdfm.data import window_store as WS
    t0 = time.time()
    head = moveatlas_head(ARGS.moveatlas)
    names = S.s2a_names()
    root = ARGS.store or WS.store_root(kind="s2")
    store = WS.WindowStore(root, tier="RELEASE", names=names)
    recs = store.records
    inputs = recs[recs["role"] == "input"]
    rows = inputs.to_dict("records")
    fams = L.build_stage2_families(rows)
    stage1 = L.load_layouts()
    family_of = {n: S.STAGE2_S2A[n].layout for n in S.STAGE2_S2A}
    every = {**{k: stage1[k] for k in L.STAGE2_REUSED.values()}, **fams}
    checks, counts = [], {}
    for r in rows:
        source, modality, _, fps, cm, meta, keys = L.stored_record(r)
        lay = every[family_of[source]]
        verdict, bad = L.compare_record(lay, modality, fps, cm, meta)
        c = counts.setdefault((source, lay.name, modality), {})
        c[verdict] = c.get(verdict, 0) + 1
        checks.append({"source": source, "family": lay.name, "stream": modality, "record": f"{keys['subject_ns']} "
                       f"{keys['key']}", "fps": fps, "sites": "|".join(L.SITES[i] for i in meta.get("site_ids", [])),
                       "mask": "|".join(L.mask_rows(cm)), "verdict": verdict, "differences": "; ".join(bad)})
    store.close()
    bad = [c for c in checks if c["verdict"] == "differs"]
    doc = {"format": L.FORMAT, "moveatlas_head": head, "store_root": os.path.relpath(root, TREE).replace("\\", "/"),
           "names_sha256": names.sha256(), "mount_priors": "reports/mount_priors/priors.json",
           "device_noise": "reports/g2_renderer/device_noise.json",
           "families": {n: L.layout_to_json(x) for n, x in fams.items()}}
    with open(os.path.join(ARGS.out, "layouts_s2.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, default=float)
        f.write("\n")
    with open(os.path.join(ARGS.out, "checks_s2.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(checks[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(checks)
    back = L.load_layouts(os.path.join(ARGS.out, "layouts_s2.json"), L.STAGE2_FAMILY_RULES)
    lines = ["# Layout families of run S2a (task S2.6)", "",
             "Generated by scripts/build_layouts.py --stage2; every number below is read from layouts_s2.json and "
             f"checks_s2.csv of this directory. MoveAtlas commit: {head}. Families built from every stored input "
             f"record of the S2 store {doc['store_root']} (name tables {doc['names_sha256']}); rules in "
             "mdfm/factory/layouts.py (STAGE2_FAMILY_RULES). layouts.json and the stage-1 files of this folder are "
             "not written by this run.", "", "## Families", "",
             "| family | sites | k sites | rates (Hz) | channels | acc_content | mount | device | record |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n, lay in back.items():
        s = lay.streams[0]
        rates = ", ".join(f"{x:g}" for x in s.fps[:6]) + (f", ... ({len(s.fps)} rates, {min(s.fps):g} to "
                                                            f"{max(s.fps):g})" if len(s.fps) > 6 else "")
        rec = s.record or {}
        lines.append(f"| {n} | {' '.join(s.sites)} | {list(lay.k_sites) if lay.k_sites else ''} | {rates} | "
                     f"{s.channels} | {s.acc_content} | {lay.mount} | {lay.device} | {rec.get('subject_ns')} "
                     f"{rec.get('key')} |")
    lines += ["", "Tokens of a 10 s window: " + "; ".join(
        f"{n} {L.token_count(L.concrete(lay, np.random.default_rng(0)), 10.0)}" for n, lay in back.items()) + ".", "",
        "## Every stored input record against its family", "",
        "equal: rate a candidate of the family, same acc_content, the family's sites in its order with its channel "
        "rows (a drawn-subset family: a site count in its range); site_subset: as equal with some of a fixed "
        "family's sites absent; differs: anything else (checks_s2.csv lists the differences). What the check "
        "covers: the reused families (xsens13, mocap3d) come from layouts.json of stage 1, independent of these "
        "records. The three S2a families take their rates, channel row and acc_content from these same stored "
        "records (layouts.build_stage2_families; a second channel row or acc_content, or a second rate in a "
        "fixed-rate family, stops the build), so for them those three agree by construction and the check tests "
        "only the sites, their order and the site count against the rules of layouts.STAGE2_FAMILY_RULES.", "",
        "| source | family | stream | equal | site_subset | differs |", "|---|---|---|---|---|---|"]
    for (source, fam, modality), c in counts.items():
        lines.append(f"| {source} | {fam} | {modality} | {c.get('equal', 0)} | {c.get('site_subset', 0)} | "
                     f"{c.get('differs', 0)} |")
    lines += ["", f"{len(checks)} stored input records; {len(bad)} differ.", ""]
    with open(os.path.join(ARGS.out, "summary_s2.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    prov = {"script": "build_layouts.py --stage2", "moveatlas_root": ARGS.moveatlas, "moveatlas_head": head,
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
            "store_root": doc["store_root"], "names_sha256": names.sha256(), "runtime_s": round(time.time() - t0, 1),
            "outputs": ["_provenance_s2.json", "checks_s2.csv", "layouts_s2.json", "summary_s2.md"]}
    with open(os.path.join(ARGS.out, "_provenance_s2.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
        f.write("\n")
    log(f"S2a layouts: {len(fams)} families, {len(checks)} records checked, {len(bad)} differ")
    if bad:
        sys.exit(f"stored records differ from their families: {bad[:5]}")


def main():
    if ARGS.stage2:
        return stage2()
    from _common import moveatlas_head
    t0 = time.time()
    head = moveatlas_head(ARGS.moveatlas)
    real = L.read_real()
    for (source, kind), (fps, _, _, keys) in sorted(real.items()):
        log(f"{source} {kind}: {keys['subject_ns']} {keys['key']}, {fps:g} Hz")
    layouts = L.build_families(real, CR.FAMILIES["generic"].fps)
    checks = L.check_families(layouts, real)
    bad = [r for r in checks if r["verdict"] != "equal"]
    if bad:
        sys.exit(f"layouts differ from their records: {bad}")
    doc = {"format": L.FORMAT, "moveatlas_head": head, "device_noise": "reports/g2_renderer/device_noise.json",
           "mount_priors": "reports/mount_priors/priors.json", "camera_family": "generic",
           "families": {n: L.layout_to_json(l) for n, l in layouts.items()}}
    with open(os.path.join(ARGS.out, "layouts.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, default=float)
        f.write("\n")
    with open(os.path.join(ARGS.out, "checks.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(checks[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(checks)
    log(f"layouts: {len(layouts)} families, {len(checks)} checks equal")
    bench_doc, bench_from = None, None
    outputs = ["layouts.json", "checks.csv", "summary.md", "_provenance.json"]
    if not ARGS.no_bench:
        bench_doc = bench(L.load_layouts(os.path.join(ARGS.out, "layouts.json")))
        with open(os.path.join(ARGS.out, "bench.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(bench_doc, f, indent=1)
            f.write("\n")
        outputs.append("bench.json")
    elif os.path.exists(os.path.join(ARGS.out, "bench.json")):
        with open(os.path.join(ARGS.out, "bench.json"), encoding="utf-8") as f:
            bench_doc = json.load(f)
        try:                                     # the run that wrote the reused bench.json
            with open(os.path.join(ARGS.out, "_provenance.json"), encoding="utf-8") as f:
                old = json.load(f)
            bench_from = old.get("bench_from") or old.get("run_at")
        except (OSError, ValueError):
            bench_from = "unknown"
    with open(os.path.join(ARGS.out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(summary(doc, checks, bench_doc, head))
    prov = {"script": "build_layouts.py", "moveatlas_root": ARGS.moveatlas, "moveatlas_head": head,
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0], "seed": ARGS.seed,
            "runtime_s": round(time.time() - t0, 1), "outputs": sorted(outputs)}
    if bench_from:
        prov["bench_from"] = bench_from           # --no-bench: summary.md quotes the bench.json of that run
    with open(os.path.join(ARGS.out, "_provenance.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(prov, f, indent=1)
        f.write("\n")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
