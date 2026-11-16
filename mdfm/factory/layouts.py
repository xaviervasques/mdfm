"""Layout families of the synthetic factory (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.4, task
B4.10): which streams a synthetic window carries, at which sites, rates and channels, with which mount
prior and device model.

A Layout is a tuple of StreamLayouts (modality imu, pose2d, pose3d or force; role input or target;
candidate rates in Hz; IMU sites in record order; channel mask (G, C); acc_content; force unit; camera
family) plus the mount family of mdfm.kernels.nuisance.MountPriors, the device family (DeviceFamily)
and, for the random-subset family, the range of site counts. Sites, rates and channel masks are read
from one admitted, non-lockbox record per source by scripts/build_layouts.py (canonical.load, through
mdfm.data.sources for the pilot sources) and written to reports/layouts/layouts.json; this module holds
only the family rules (FAMILY_RULES: which source, which site subset, which prior and device). Families:
- xsens13: the 13 Xsens sites of weargait_pd, acc and gyro, 100 Hz; xsens13_insoles adds the insole
  totals (force target, unit unknown) of the same recording;
- nonan3: head, lower_back, trunk of nonan_gaitprint, acc only, 200 Hz (nonan_gaitprint_young checked);
- lower_back: sixmwt_lifespan, acc and gyro, 100 Hz;
- g1_six (head, lower_back, wrist_L, wrist_R, shank_L, shank_R) and wrist_L: the G1a and G1b inputs of
  spec section 6, the rows of those sites in the weargait_pd record, so xsens13 mounts and device;
- wrist_acc: the wrist accelerometer of capture24, acc only, 100 Hz (no gyro);
- random_sites: k sites drawn uniformly among the 14 (k uniform in 1 to 14), acc and gyro (the row
  pattern of the xsens13 record), a rate drawn among the IMU rates read above;
- camera1: one synthetic camera (pose2d); joints and coordinate mask of a li_taati record, rates of
  the camera family of factory/camera_render.py (the li_taati records carry no fps);
- mocap3d: canonical 3D pose of gait120 at 100 Hz (stroke_gait_vancriekinge able-bodied checked);
  mocap3d_plate adds the plate force totals (1,000 Hz, N) of a stroke able-bodied trial, one plate
  under one foot (plate=True): the drawn plate foot takes the record's row pattern of a present foot,
  the other foot is absent, as in a stroke trial where one foot never meets its plates.
Families without a measured prior use the generic mount family (identity, 10 degrees) and the generic
device (imu_render.Device defaults, noise levels scaled by a log-uniform factor in GENERIC_NOISE_SCALE
[cfg]); xsens13, xsens13_insoles, g1_six and wrist_L use the xsens13 mount prior and the per-site device
noise of reports/g2_renderer/device_noise.json (quiet standing of the G2 ALIGN half; accelerometer bias
spread not measured, Device default). Saturation: the Device ranges.

Tokens (spec 2.2, collate): a window of duration d holds P = min(40, ceil(d / 0.25)) latent tokens, one
subject token, one measurement token per channel group and 0.25 s patch of each input stream (an IMU
site has an acc group and a gyro group when those channels are present; a pose stream has one group)
and one nuisance token per IMU site and per pose stream. Force and EMG are targets, without tokens.

Stage 2, run S2a (spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.5, task S2.6): STAGE2_FAMILY_RULES names the
layout families of the new S2a units, each with a synthetic counterpart in the replay; they live in their own file,
reports/layouts/layouts_s2.json (LAYOUTS_S2_PATH), so that layouts.json, the stage-1 families and the rates of
random_sites stay as they were. Families (sites [cfg]; rates, channel rows and acc_content read from every input
record of the S2a store, scripts/build_layouts.py --stage2):
- pads_wrists: wrist_L or wrist_R, one per window (k_sites (1, 1): pads enters as two single-site windows, its two
  watches share no clock), acc without gravity (linear_no_gravity) and gyro, 100 Hz;
- multipatho4: head, lower_back, foot_L, foot_R of clinical_gait_multipatho, acc and gyro (the magnetometer masked),
  100 Hz; records turned per site (mdfm.data.mounting);
- pocket_acc: the pocket accelerometer of imu_inthewild_iprognosis, the records' own rates (every distinct rate of
  the stored records is a candidate); records turned.
The units weargait_pd:PD and stroke_gait_vancriekinge:stroke keep their stage-1 families (STAGE2_REUSED: xsens13,
mocap3d). Mount and device: the generic families until the S2a mount priors are measured (spec section 4.5: from
non-lockbox, non-dev records of the source; pads a broad tilt). compare_record checks one stored record against
its family: 'equal', 'site_subset' (a fixed family's sites minus some, as on the stage-1 controls) or the
differences.
"""
import json
import math
import os
from dataclasses import dataclass, field, replace

import numpy as np

from .. import paths
from .imu_render import SITES, Device

LAYOUTS_PATH = os.path.join(paths.TREE, "reports", "layouts", "layouts.json")
DEVICE_NOISE_PATH = os.path.join(paths.TREE, "reports", "g2_renderer", "device_noise.json")
FORMAT = "mdfm-layouts/1"
PATCH_S = 0.25
MAX_PATCHES = 40
MODALITIES = ("imu", "pose2d", "pose3d", "force")
GENERIC_NOISE_SCALE = (0.5, 2.0)      # [cfg] log-uniform factor on the generic device noise levels
G1_SIX = ("head", "lower_back", "wrist_L", "wrist_R", "shank_L", "shank_R")   # spec section 6, G1a
# family -> streams (modality, role, source of the record, site subset or None for the record's sites),
# sources whose records must match too, mount family, device family [cfg]
FAMILY_RULES = {
    "xsens13": {"streams": (("imu", "input", "weargait_pd", None),), "mount": "xsens13", "device": "xsens13"},
    "xsens13_insoles": {"streams": (("imu", "input", "weargait_pd", None), ("force", "target", "weargait_pd", None)),
                        "mount": "xsens13", "device": "xsens13"},
    "nonan3": {"streams": (("imu", "input", "nonan_gaitprint", None),), "also": ("nonan_gaitprint_young",),
               "mount": "generic", "device": "generic"},
    "lower_back": {"streams": (("imu", "input", "sixmwt_lifespan", None),), "mount": "generic", "device": "generic"},
    "g1_six": {"streams": (("imu", "input", "weargait_pd", G1_SIX),), "mount": "xsens13", "device": "xsens13"},
    "wrist_L": {"streams": (("imu", "input", "weargait_pd", ("wrist_L",)),), "mount": "xsens13", "device": "xsens13"},
    "wrist_acc": {"streams": (("imu", "input", "capture24", None),), "mount": "generic", "device": "generic"},
    "random_sites": {"streams": (("imu", "input", None, None),), "mount": "generic", "device": "generic",
                     "k_sites": (1, len(SITES))},
    "camera1": {"streams": (("pose2d", "input", "li_taati", None),), "camera": "generic"},
    "mocap3d": {"streams": (("pose3d", "input", "gait120", None),), "also": ("stroke_gait_vancriekinge",)},
    "mocap3d_plate": {"streams": (("pose3d", "input", "stroke_gait_vancriekinge", None),
                                  ("force", "target", "stroke_gait_vancriekinge", None)), "plate": True},
}
# default share of each family among synthetic windows [cfg]
DEFAULT_MIXTURE = {"xsens13": 0.15, "xsens13_insoles": 0.10, "nonan3": 0.10, "lower_back": 0.10, "g1_six": 0.10,
                   "wrist_L": 0.10, "wrist_acc": 0.05, "random_sites": 0.15, "camera1": 0.05, "mocap3d": 0.05,
                   "mocap3d_plate": 0.05}
# stage 2, run S2a (module docstring): family -> source, sites [cfg], k_sites (a drawn subset), fps ('records': every
# rate of the stored records), mount and device families
LAYOUTS_S2_PATH = os.path.join(paths.TREE, "reports", "layouts", "layouts_s2.json")
STAGE2_FAMILY_RULES = {
    "pads_wrists": {"source": "pads", "sites": ("wrist_L", "wrist_R"), "k_sites": (1, 1), "mount": "generic",
                    "device": "generic"},
    "multipatho4": {"source": "clinical_gait_multipatho", "sites": ("head", "lower_back", "foot_L", "foot_R"),
                    "mount": "generic", "device": "generic"},
    "pocket_acc": {"source": "imu_inthewild_iprognosis", "sites": ("pocket",), "fps": "records", "mount": "generic",
                   "device": "generic"},
}
STAGE2_REUSED = {"weargait_pd": "xsens13", "stroke_gait_vancriekinge": "mocap3d"}


# ---------------------------------------------------------------- layouts
@dataclass(frozen=True)
class StreamLayout:
    """One stream of a layout: modality, role, fps (candidate rates Hz; one entry once drawn), sites (IMU
    site names in record order, () otherwise), mask ((G, C) channel mask as rows of '0'/'1' strings),
    acc_content (IMU), unit (force), camera (camera family of a pose2d stream), plate (a force stream
    of one plate under one foot), record (source, modality, subject_ns and key of the record it was
    read from)."""
    modality: str
    role: str
    fps: tuple
    sites: tuple = ()
    mask: tuple = ()
    acc_content: str = None
    unit: str = None
    camera: str = None
    plate: bool = False
    record: dict = field(default=None, compare=False, hash=False)

    @property
    def chan_mask(self):
        """(G, C) bool array."""
        return np.array([[c == "1" for c in row] for row in self.mask], dtype=bool).reshape(len(self.mask), -1)

    @property
    def site_ids(self):
        return [SITES.index(s) for s in self.sites]

    @property
    def channels(self):
        """'acc', 'gyro', 'acc+gyro' or '' for an IMU stream (from the mask), None otherwise."""
        if self.modality != "imu":
            return None
        m = self.chan_mask
        return "+".join(n for n, sl in (("acc", slice(0, 3)), ("gyro", slice(3, 6))) if m[:, sl].any())


@dataclass(frozen=True)
class Layout:
    """A layout family, or one draw of it (concrete: every stream has one rate and, for random_sites, the
    drawn sites). mount: mount family of MountPriors (None without IMU); device: DeviceFamily name;
    k_sites: (k_min, k_max) for a random-subset family."""
    name: str
    streams: tuple
    mount: str = None
    device: str = None
    k_sites: tuple = None

    @property
    def concrete(self):
        return self.k_sites is None and all(len(s.fps) == 1 for s in self.streams)

    def stream(self, modality):
        for s in self.streams:
            if s.modality == modality:
                return s
        raise KeyError(f"{self.name}: no {modality} stream")

    @property
    def imu(self):
        return next((s for s in self.streams if s.modality == "imu"), None)


def mask_rows(chan_mask):
    """(G, C) bool -> tuple of '0'/'1' strings."""
    return tuple("".join("1" if x else "0" for x in row) for row in np.asarray(chan_mask, dtype=bool))


def stream_to_json(s):
    return {"modality": s.modality, "role": s.role, "fps": [float(f) for f in s.fps], "sites": list(s.sites),
            "mask": list(s.mask), "acc_content": s.acc_content, "unit": s.unit, "camera": s.camera,
            "plate": s.plate, "record": s.record}


def stream_from_json(d):
    return StreamLayout(modality=d["modality"], role=d["role"], fps=tuple(float(f) for f in d["fps"]),
                        sites=tuple(d.get("sites") or ()), mask=tuple(d.get("mask") or ()),
                        acc_content=d.get("acc_content"), unit=d.get("unit"), camera=d.get("camera"),
                        plate=bool(d.get("plate", False)), record=d.get("record"))


def layout_to_json(lay):
    return {"name": lay.name, "streams": [stream_to_json(s) for s in lay.streams], "mount": lay.mount,
            "device": lay.device, "k_sites": list(lay.k_sites) if lay.k_sites else None}


def layout_from_json(d):
    return Layout(name=d["name"], streams=tuple(stream_from_json(s) for s in d["streams"]), mount=d.get("mount"),
                  device=d.get("device"), k_sites=tuple(d["k_sites"]) if d.get("k_sites") else None)


def load_layouts(path=LAYOUTS_PATH, rules=None):
    """{name: Layout} of reports/layouts/layouts.json (scripts/build_layouts.py), in FAMILY_RULES order; rules: the
    family rules of the file (None: FAMILY_RULES; STAGE2_FAMILY_RULES for layouts_s2.json)."""
    rules = FAMILY_RULES if rules is None else rules
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: format {doc.get('format')!r}, expected {FORMAT}")
    fams = doc["families"]
    unknown = set(fams) - set(rules)
    if unknown:
        raise ValueError(f"{path}: families without a rule {sorted(unknown)}")
    return {n: layout_from_json(fams[n]) for n in rules if n in fams}


def load_stage2_layouts(path=LAYOUTS_PATH, s2_path=LAYOUTS_S2_PATH):
    """The stage-1 families followed by the S2a families (a phase C replay renders both); a name in both files
    raises."""
    a, b = load_layouts(path), load_layouts(s2_path, STAGE2_FAMILY_RULES)
    both = sorted(set(a) & set(b))
    if both:
        raise ValueError(f"families in both files: {both}")
    return {**a, **b}


# ---------------------------------------------------------------- reading the records
def family_sources():
    """{source: set of stream kinds} that the families read (FAMILY_RULES streams and 'also' sources)."""
    out = {}
    for rule in FAMILY_RULES.values():
        for modality, _, source, _ in rule["streams"]:
            if source is not None:
                out.setdefault(source, set()).add(modality)
        for source in rule.get("also", ()):
            out.setdefault(source, set()).update(m for m, _, s, _ in rule["streams"] if s is not None)
    return out


def record_streams(source, kinds, last=False):
    """{kind: (fps, chan_mask, meta, row keys)} of one admitted, non-lockbox recording of `source` holding
    every stream kind in `kinds`, the first in (person, recording) order, or the last with last=True. A
    pilot source goes through mdfm.data.sources (rows(), pair(): records read by canonical.load with its
    flags at False; pair() supplies the force unit and the pose rotation) and windows.cut of the whole
    recording; another source through canonical._select (tier RELEASE, lane body, no lockbox, EVAL or
    CODY row) and canonical.load, in (subject_ns, key) order. A CODY or EVAL source raises
    canonical.Refused before any record is read; this reader has no evaluation flag."""
    from ..data import canonical as C
    from ..data import sources as S
    from ..data import windows as W
    if C.is_cody(source):
        raise C.Refused(f"CODY is evaluation-only and never shapes a synthetic layout: {source}")
    kinds = set(kinds)
    if source in S.SOURCES:
        df = S.rows(source)
        groups = list(df.groupby(["person_id", "subject_ns", "recording"], sort=True))
        for _, g in (reversed(groups) if last else groups):
            if not kinds <= set(g["stream"]):
                continue
            rows = g.to_dict("records")
            try:
                pr = S.pair(source, S.load(rows))
            except S.PairingError:
                continue
            keys = {}
            for r in rows:
                keys.setdefault(r["stream"], {k: r[k] for k in ("source", "modality", "subject_ns", "key")})
            win = W.cut(pr, win_s=1e9, stride_s=1e9, min_s=0.0)[0]
            return {st.modality: (st.fps, st.chan_mask, st.meta, keys[st.modality]) for st in win.streams
                    if st.modality in kinds}
        raise LookupError(f"{source}: no recording with {sorted(kinds)}")
    if len(kinds) != 1:
        raise ValueError(f"{source}: one stream kind per record outside the pilot sources, got {sorted(kinds)}")
    kind = kinds.pop()
    modality = "pose" if kind in ("pose2d", "pose3d") else kind
    sel = C._select(modality=modality, tier="RELEASE", lane="body", sources=[source])
    sel = sel.sort_values(["subject_ns", "key"], kind="stable", ascending=not last)
    for _, row in sel.iterrows():
        rec = C.load(row)
        if kind in ("pose2d", "pose3d") and (int(rec.get("dim", 3)) == 2) != (kind == "pose2d"):
            continue
        _, cm, _, meta = W.arrays(kind, W.whitelist(rec))
        fps = rec.get("fps")
        fps = float(fps) if fps is not None else float("nan")
        return {kind: (fps, cm, meta, {k: row[k] for k in ("source", "modality", "subject_ns", "key")})}
    raise LookupError(f"{source}: no {kind} record")


def read_real(last=False):
    """{(source, kind): (fps, chan_mask, meta, row keys)} of every source the families name
    (record_streams, first or last recording)."""
    real = {}
    for source, kinds in sorted(family_sources().items()):
        for kind, v in record_streams(source, kinds, last=last).items():
            real[(source, kind)] = v
    return real


def check_families(layouts, real):
    """Rows (family, stream, source, record, fps, sites, mask, verdict) comparing each family stream with
    the record of every source the family names (compare_stream); verdict 'equal' or the differences."""
    rows = []
    for name, lay in layouts.items():
        rule = FAMILY_RULES[name]
        for (modality, _, source, _), s in zip(rule["streams"], lay.streams):
            for src in ([source] if source else []) + list(rule.get("also", ())):
                fps, cm, meta, keys = real[(src, modality)]
                bad = compare_stream(s, fps, cm, meta)
                rows.append({"family": name, "stream": modality, "source": src,
                             "record": f"{keys['subject_ns']} {keys['key']}", "fps_record": fps,
                             "fps_layout": "|".join(f"{f:g}" for f in s.fps), "sites": "|".join(s.sites),
                             "mask": "|".join(s.mask), "verdict": "; ".join(bad) or "equal"})
    return rows


# ---------------------------------------------------------------- building from real streams
def real_stream_layout(modality, role, fps, chan_mask, meta, record=None, sites=None):
    """StreamLayout of a real stream (windows.Stream fields: fps, chan_mask, meta), restricted to `sites`
    (IMU site names; None keeps the record's). A site absent from the record raises KeyError."""
    kw = {"modality": modality, "role": role, "fps": (float(fps),) if fps is not None and math.isfinite(fps) else (),
          "record": record}
    cm = np.asarray(chan_mask, dtype=bool)
    if modality == "imu":
        rec_sites = [SITES[i] for i in meta["site_ids"]]
        want = rec_sites if sites is None else list(sites)
        missing = [s for s in want if s not in rec_sites]
        if missing:
            raise KeyError(f"sites {missing} absent from the record ({rec_sites})")
        rows = [rec_sites.index(s) for s in want]
        kw.update(sites=tuple(want), mask=mask_rows(cm[rows]), acc_content=meta.get("acc_content") or "with_gravity")
    else:
        kw["mask"] = mask_rows(cm)
        if modality == "force":
            kw["unit"] = meta.get("unit")
    return StreamLayout(**kw)


def build_families(real, camera_fps):
    """{name: Layout} from real streams: real maps (source, modality) to (fps, chan_mask, meta, record key
    dict) of one record (scripts/build_layouts.py reads them); camera_fps, the rates of the camera family.
    random_sites takes the acc and gyro row pattern of the xsens13 record and the rates of the IMU
    families read here."""
    out = {}
    for name, rule in FAMILY_RULES.items():
        streams = []
        for modality, role, source, subset in rule["streams"]:
            if source is None:
                continue
            fps, cm, meta, rec = real[(source, modality)]
            s = real_stream_layout(modality, role, fps, cm, meta, record=rec, sites=subset)
            if modality == "pose2d":
                s = replace(s, fps=tuple(float(f) for f in camera_fps), camera=rule.get("camera", "generic"))
            if modality == "force" and rule.get("plate"):
                s = replace(s, plate=True)
            streams.append(s)
        out[name] = Layout(name, tuple(streams), mount=rule.get("mount"), device=rule.get("device"),
                           k_sites=rule.get("k_sites"))
    x = out["xsens13"].imu
    row = x.mask[0]
    if any(r != row for r in x.mask):
        raise ValueError("the xsens13 record's sites carry different channels")
    rates = sorted({f for lay in out.values() for s in lay.streams if s.modality == "imu" for f in s.fps})
    out["random_sites"] = Layout("random_sites", (StreamLayout("imu", "input", tuple(rates), tuple(SITES),
                                                               (row,) * len(SITES), acc_content=x.acc_content,
                                                               record=None),),
                                 mount=FAMILY_RULES["random_sites"]["mount"],
                                 device=FAMILY_RULES["random_sites"]["device"],
                                 k_sites=FAMILY_RULES["random_sites"]["k_sites"])
    return out


def compare_stream(layout_stream, fps, chan_mask, meta):
    """Differences (list of str, empty when equal) between a layout stream and a real stream in sites,
    rate and channel mask; the layout's sites are looked up in the record, so a site subset (g1_six,
    wrist_L) compares with its rows. A plate stream compares the row pattern of the present feet (a
    stroke trial holds one or two feet with plates). A record without fps compares its mask and sites
    only."""
    bad = []
    cm = np.asarray(chan_mask, dtype=bool)
    if layout_stream.modality == "imu":
        rec_sites = [SITES[i] for i in meta["site_ids"]]
        missing = [s for s in layout_stream.sites if s not in rec_sites]
        if missing:
            return [f"sites {missing} not in the record"]
        cm = cm[[rec_sites.index(s) for s in layout_stream.sites]]
        acc = meta.get("acc_content") or "with_gravity"
        if acc != layout_stream.acc_content:
            bad.append(f"acc_content {acc} vs {layout_stream.acc_content}")
    if layout_stream.plate:
        got = {r for r in mask_rows(cm) if "1" in r}
        want = {r for r in layout_stream.mask if "1" in r}
        if len(want) != 1 or not got or got != want or len(cm) != len(layout_stream.mask):
            bad.append(f"present feet {sorted(got)} vs {sorted(want)}")
    elif mask_rows(cm) != tuple(layout_stream.mask):
        bad.append(f"channel mask {mask_rows(cm)} vs {layout_stream.mask}")
    if fps is not None and math.isfinite(fps) and layout_stream.camera is None and float(fps) not in layout_stream.fps:
        bad.append(f"fps {fps} not in {layout_stream.fps}")
    return bad


# ---------------------------------------------------------------- stage 2 (task S2.6)
def stored_record(row):
    """(source, modality, role, fps native, chan_mask (G, C) bool, meta dict, record key dict) of a row of a window
    store's records (mdfm.data.window_store: chan_mask as '0'/'1' rows joined by '|', meta as JSON text or a dict)."""
    meta = row["meta"]
    meta = json.loads(meta) if isinstance(meta, str) else dict(meta or {})
    cm = np.array([[c == "1" for c in r] for r in str(row["chan_mask"]).split("|")], dtype=bool)
    keys = {"source": row["source"], "modality": row["stream"], "subject_ns": row["subject_ns"], "key": row["key"]}
    return row["source"], row["stream"], row["role"], float(row["fps_native"]), cm, meta, keys


def build_stage2_families(rows, rules=None):
    """{name: Layout} of STAGE2_FAMILY_RULES (or `rules`) from stored records (rows of a window store's records,
    stored_record): per family, the input IMU rows of its source; the sites of the rule; the channel row and
    acc_content that every stored site carries (one each, else ValueError); the rate of the records (one, else
    ValueError) or, with fps 'records', every distinct rate in increasing order; the record key of the first row."""
    rules = STAGE2_FAMILY_RULES if rules is None else rules
    recs = [stored_record(r) for r in rows]
    out = {}
    for name, rule in rules.items():
        mine = [r for r in recs if r[0] == rule["source"] and r[1] == "imu" and r[2] == "input"]
        if not mine:
            raise LookupError(f"{name}: no stored input IMU record of {rule['source']}")
        chans = {mask_rows(r[4])[g] for r in mine for g in range(r[4].shape[0])}
        acc = {r[5].get("acc_content") or "with_gravity" for r in mine}
        rates = sorted({r[3] for r in mine})
        if len(chans) != 1 or len(acc) != 1:
            raise ValueError(f"{name}: stored sites carry channel rows {sorted(chans)} and acc_content {sorted(acc)}")
        if rule.get("fps") != "records" and len(rates) != 1:
            raise ValueError(f"{name}: records at rates {rates[:5]}; a fixed-rate family needs one")
        row = chans.pop()
        sites = tuple(rule["sites"])
        s = StreamLayout("imu", "input", tuple(float(f) for f in rates), sites, (row,) * len(sites),
                         acc_content=acc.pop(), record=mine[0][6])
        out[name] = Layout(name, (s,), mount=rule.get("mount"), device=rule.get("device"),
                           k_sites=tuple(rule["k_sites"]) if rule.get("k_sites") else None)
    return out


def compare_record(layout, modality, fps, chan_mask, meta):
    """(verdict, differences) of one stored input stream against its family `layout`: differences in rate (not a
    candidate of the family), acc_content, sites outside the family, site order, a site's channel row, the number
    of sites of a drawn-subset family, or the mask of a pose stream; verdict 'equal' without a difference and, for a
    family of fixed sites, all of them; 'site_subset' when some of the family's sites are absent; 'differs'
    otherwise."""
    s = next((x for x in layout.streams if x.modality == modality and x.role == "input"), None)
    if s is None:
        return "differs", [f"{layout.name} has no {modality} input stream"]
    bad = []
    if float(fps) not in s.fps:
        bad.append(f"fps {fps:g} not a rate of {layout.name}")
    cm = np.asarray(chan_mask, dtype=bool)
    subset = False
    if modality == "imu":
        rec_sites = [SITES[i] for i in meta.get("site_ids", [])]
        acc = meta.get("acc_content") or "with_gravity"
        if acc != s.acc_content:
            bad.append(f"acc_content {acc} vs {s.acc_content}")
        out = [x for x in rec_sites if x not in s.sites]
        if out:
            bad.append(f"sites {out} outside {layout.name}")
        inside = [x for x in rec_sites if x in s.sites]
        if inside != [x for x in s.sites if x in inside]:
            bad.append(f"site order {inside} vs {list(s.sites)}")
        rows = mask_rows(cm)
        for g, x in enumerate(rec_sites):
            if x in s.sites and rows[g] != s.mask[list(s.sites).index(x)]:
                bad.append(f"{x} channels {rows[g]} vs {s.mask[list(s.sites).index(x)]}")
        if layout.k_sites is not None:
            if not layout.k_sites[0] <= len(rec_sites) <= layout.k_sites[1]:
                bad.append(f"{len(rec_sites)} sites for k in {list(layout.k_sites)}")
        else:
            subset = len(set(rec_sites)) < len(s.sites)
    elif mask_rows(cm) != tuple(s.mask):
        bad.append(f"mask {mask_rows(cm)} vs {s.mask}")
    if bad:
        return "differs", bad
    return ("site_subset" if subset else "equal"), []


# ---------------------------------------------------------------- draws
def _rng(gen):
    return gen if isinstance(gen, np.random.Generator) else np.random.default_rng(gen)


def concrete(layout, gen):
    """One draw of a family: each stream's rate uniformly among its candidates; for random_sites, k
    uniform in k_sites and k distinct sites uniformly, kept in SITES order. A concrete layout is returned
    unchanged."""
    if layout.concrete:
        return layout
    gen = _rng(gen)
    streams = []
    for s in layout.streams:
        fps = (s.fps[int(gen.integers(len(s.fps)))],) if len(s.fps) > 1 else s.fps
        s = replace(s, fps=fps)
        if layout.k_sites is not None and s.modality == "imu":
            k = int(gen.integers(layout.k_sites[0], layout.k_sites[1] + 1))
            pick = np.sort(gen.choice(len(s.sites), size=k, replace=False))
            s = replace(s, sites=tuple(s.sites[i] for i in pick), mask=tuple(s.mask[i] for i in pick))
        streams.append(s)
    return replace(layout, streams=tuple(streams), k_sites=None)


def sample_layout(gen, mixture=None, layouts=None):
    """A concrete Layout: a family drawn with the probabilities of `mixture` ({name: weight}, default
    DEFAULT_MIXTURE, normalized), then concrete(). gen: numpy Generator or int seed."""
    gen = _rng(gen)
    layouts = load_layouts() if layouts is None else layouts
    mixture = DEFAULT_MIXTURE if mixture is None else mixture
    names = [n for n in mixture if mixture[n] > 0]
    missing = [n for n in names if n not in layouts]
    if missing:
        raise KeyError(f"mixture names without a layout: {missing}")
    p = np.array([float(mixture[n]) for n in names])
    return concrete(layouts[names[int(gen.choice(len(names), p=p / p.sum()))]], gen)


# ---------------------------------------------------------------- device families
@dataclass(frozen=True)
class DeviceFamily:
    """Device model per site (SITES order): noise_acc (14, 3) g and noise_gyro (14, 3) deg/s white noise
    per sample, bias_acc_sd (14, 3) g and bias_gyro_sd (14, 3) deg/s constant-bias spreads, noise_scale
    (lo, hi) the range of a log-uniform factor on the noise levels drawn per window, and the saturation
    ranges acc_range_g, gyro_range_dps."""
    name: str
    noise_acc: np.ndarray
    noise_gyro: np.ndarray
    bias_acc_sd: np.ndarray
    bias_gyro_sd: np.ndarray
    noise_scale: tuple = (1.0, 1.0)
    acc_range_g: float = Device.acc_range_g
    gyro_range_dps: float = Device.gyro_range_dps


def device_families(noise_path=DEVICE_NOISE_PATH):
    """{'generic', 'xsens13'}: generic from imu_render.Device with GENERIC_NOISE_SCALE; xsens13 from the
    per-site levels of device_noise.json (pocket and the accelerometer bias spread: Device defaults)."""
    d = Device()
    full = lambda v: np.full((len(SITES), 3), float(v))  # noqa: E731
    generic = DeviceFamily("generic", full(d.acc_noise_g), full(d.gyro_noise_dps), full(d.acc_bias_g),
                           full(d.gyro_bias_dps), GENERIC_NOISE_SCALE)
    with open(noise_path, encoding="utf-8") as f:
        doc = json.load(f)
    na, ng, bg = full(d.acc_noise_g), full(d.gyro_noise_dps), full(d.gyro_bias_dps)
    for row in doc["per_site"]:
        k = SITES.index(row["site"])
        na[k], ng[k], bg[k] = row["acc_sigma_g"], row["gyro_sigma_dps"], row["gyro_bias_sd_dps"]
    xsens = DeviceFamily("xsens13", na, ng, full(d.acc_bias_g), bg, (1.0, 1.0))
    return {"generic": generic, "xsens13": xsens}


# ---------------------------------------------------------------- tokens
def n_patches(duration, patch_s=PATCH_S):
    """Latent patches of a window of `duration` s: min(40, ceil(duration / patch_s)), at least 1."""
    return int(min(MAX_PATCHES, max(1, math.ceil(duration / patch_s - 1e-9))))


def _groups(modality, chan_mask):
    """(channel groups, nuisance tokens) of an input stream."""
    cm = np.asarray(chan_mask, dtype=bool)
    if modality == "imu":
        return int(cm[:, 0:3].any(1).sum() + cm[:, 3:6].any(1).sum()), int(cm.shape[0])
    if modality in ("pose2d", "pose3d"):
        return 1, 1
    return 0, 0


def token_count(layout, duration=10.0):
    """Tokens of a window of `duration` s under a concrete layout (module docstring)."""
    if layout.k_sites is not None:
        raise ValueError(f"{layout.name}: draw the sites first (concrete)")
    P = n_patches(duration)
    n = P + 1
    for s in layout.streams:
        if s.role == "input":
            g, nuis = _groups(s.modality, s.chan_mask)
            n += g * P + nuis
    return n


def window_tokens(window):
    """Tokens of a windows.Window: P from its duration; per input stream, its channel groups times the
    patches holding at least one of its samples (at most P), plus its nuisance tokens."""
    P = n_patches(window.duration)
    n = P + 1
    for s in window.streams:
        if s.role != "input":
            continue
        g, nuis = _groups(s.modality, s.chan_mask)
        patches = min(P, len(np.unique(np.floor(np.asarray(s.t, dtype=np.float64) / PATCH_S))))
        n += g * patches + nuis
    return n
