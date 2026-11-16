"""Mixture sampler and bucket batcher of the stage-1 pilot (spec docs/MFM_MODEL_CODE_SPEC_2026-09-28.md, 2.2
and section 5, task B4.17): which windows fill a micro-step, and how they are packed into rows.

A micro-step holds about tokens_per_micro tokens (16,384) in at most max_windows windows (512) [cfg,
train/config.MixtureSection]:
- Shares of its tokens: synthetic, synthetic_share (1.0 in phase A, 0.45 in phase B); the rest (phase B)
  goes to the real sources in proportion to the square root of each source's subject-hours (the hours of the
  input recordings of its training persons in the window store), each capped at source_cap (15 percent) of
  all tokens of the micro-step [D§5]; what a capped source cannot take goes to the others (capped_shares).
  A real share the caps cannot absorb is refused (ValueError). The 60 percent cap on synthetic tokens is read
  cumulatively over the run (OD9, decided 2026-09-28; train/config.py checks it), so phase A is synthetic
  only.
- Budgets (plan): real sources first, each with its share of tokens_per_micro plus the budget it left unused
  in earlier micro-steps (its credit: a weargait_pd window of 10 s holds 1,094 tokens, so a budget of about
  1,600 takes one window and the credit lets a later micro-step take two), at most source_cap of
  tokens_per_micro, within the real share of max_windows; then the synthetic budget, synthetic_share / (1 -
  synthetic_share) times the real tokens drawn and at most tokens_per_micro minus them, so that the
  synthetic share of the micro-step's tokens stays at or under synthetic_share when the real budgets fill
  short (a fixed synthetic budget would lift the share above 0.45 and the cumulative share above 0.60) and a
  micro-step never exceeds tokens_per_micro.
- Units: the budgets fill with units drawn in turn across the sources (round robin): a person drawn
  uniformly among the source's persons (subject-balanced), then windows_per_person windows of that person
  (2; one when the person's data hold a single distinct window). A real window is a random span of
  min(win_s, span) s of one of the person's recordings, drawn with probability proportional to its span (the
  whole record when shorter than win_s; recordings under min_s are left out). A synthetic window is a
  body-store window of win_s s (masked after the end of a shorter take) of one of the person's takes, drawn
  with probability proportional to its knots, first knot uniform among those that fit, with a layout drawn
  from the family weights (factory/layouts.sample_layout). A unit that does not fit the source's remaining
  budget or the window cap is shrunk to its first body; max_misses draws in a row that do not fit close the
  source for the micro-step. Render pairs: each synthetic body is rendered a second time under resampled
  nuisance (factory/synth.py) with probability pair_share (25 percent), both renders in the same unit and
  never split by a drop. A synthetic body is charged against the budget as its single render times (1 +
  pair_share), whatever its pairing, so fitting the budget does not select against pairs (charging the
  actual tokens let paired units, twice as long, fit the end of a budget less often: 0.222 of the bodies
  paired over 1,000 simulated steps of the pilot configuration); the synthetic tokens are then trimmed to
  the budget.
- Caps after the draw (enforce_caps): while a real source holds more than source_cap of the micro-step's
  tokens its last body is dropped (one window, or both renders of a pair, never one render alone), and while
  the synthetic tokens exceed synthetic_share of them the last synthetic body is dropped; this runs on the
  planned counts (the real bodies first against the total the synthetic budget will complete) and again on
  the exact counts of the read and rendered windows. A plan or a step left empty raises EmptyStep, a ValueError
  (tokens_per_micro too small for the window sizes).
- Redraws (opt-in, task T15B; redraws=0, the default, is the behaviour above, plan for plan): with redraws=k > 0, a
  plan that the caps leave empty is drawn again from the continuing generator (new seed, new units, the same
  credits), and next() draws a new plan when the caps on the exact counts leave a step empty, each at most k times
  before EmptyStep. Why: at 8,192 tokens per micro-step (the fallback micro-batch of the 200M bench,
  scripts/bench_stage1.py) a phase B micro-step holds few bodies, some of them coarse (a 10 s weargait_pd window of
  1,094 tokens, a synthetic render pair of two long windows); when the synthetic fill stays short of its budget, a
  real source can sit just over source_cap, and the greedy drops of enforce_caps cascade (each drop shrinks the
  total and pushes another source, then the synthetic share, over its bound) down to an empty plan
  (reports/bench/stage1_200m_plan_check.md traces one). A redraw drops such a draw instead of stopping the run; the
  caps hold on every returned step as before. The draws before the first empty plan are those of the default
  sampler. state_dict() carries redraws when it is not 0, and a state of another redraws value is refused (the
  sequences part at the first empty plan).

Token counts (model/tokenize.window_tokens, factory/layouts.window_tokens): P + 1 + groups x patches holding a
sample + nuisance tokens, P = min(40, ceil(duration / 0.25)). The plan computes a real window's count exactly
from the store's sample grid (times offset + j k / fps_native, cast to float32 from t0 as WindowStore.stream
does); a synthetic window's count is layouts.token_count(layout, duration), an upper bound that the render
can undercut by its last patch; the exact count replaces it after rendering.

next() plans, reads the real windows (WindowStore.window), renders the synthetic ones in one call
(synth.render, generator seeded by the step's seed), enforces the caps on the exact counts and runs the guard
(train/provenance.assert_clean with the held-out persons: the window store outside the train fold and the
amass dev, calibration and excluded splits) on every window before returning them. The step's seed also
seeds collate and the model draws of that micro-step. state_dict() holds the numpy generator state, the step
counter, the real-source credits, the token and window totals per source and the SHA-256 of the configuration
and of the store tables the draws read; load_state_dict() resumes the sequence exactly and refuses a state of
another phase, seed, configuration or store content. Every recording the sampler draws must hold exactly one
input stream (ValueError otherwise), since the planned token count reads one.

BucketBatcher packs one micro-step's windows into rows of one bucket: the smallest row length of rows (128,
640 and 1,536 tokens [R bench]; B4.18 re-tunes them) that holds the longest window, first-fit decreasing.

Phase C (task S2.7; spec docs/MFM_STAGE2_CODE_SPEC_DRAFT.md, section 4.6): the stage-2 phase. cfg is then the run's
train.config.RunConfig (its mixture and its stage2 section) and window_store a Stage2Stores: the stage-1 root (the
healthy replay, read as it is) and the stage-2 root (the pathological units). Two groups share the tokens of a
micro-step:
- replay, replay_share R of them (0.30 [D§5]), drawn as the config's phase B mixture with its rules unchanged: the
  synthetic share R x synthetic_share[1], and the real healthy sources of mixture.sources at R times their phase B
  shares (capped_shares of the square roots of their subject-hours, each at most source_cap, over 1 -
  synthetic_share[1]), each capped at source_cap of all tokens;
- pathological, 1 - R, shared by the units of stage2.units in proportion to the square root of each unit's
  subject-hours, each capped at stage2.unit_cap of all tokens [D§5; S1 2.2 reads the cap against all tokens]
  (capped_shares; a share the caps cannot absorb raises ValueError). A unit's subject-hours are the recording time
  of its training persons with one recording counted once whatever the number of its single-site records: the two
  wrist records of one pads task (keys <task>_wrist_L and <task>_wrist_R, the pairs of
  reports/stage2_sources/items/pads_pairs.csv, equal in length on all 4,125 pairs) count as one recording of the
  longer one's span (unit_hours, ONE_RECORDING); every other record is a recording of its own. A unit's windows come
  from the stage-2 store's folder of its source, its persons drawn uniformly (subject-balanced) as in phase B.
- the window cap (task S2B, the recipe of run S2b; stage2.window_cap_per_person, 0.0 off by default): each unit's
  share of the phase C windows is at most window_cap_per_person times its training persons (the persons of its
  table, those with a training recording of at least min_s; window_caps), 1 at most. The tally (wtally: the windows
  of the phase so far, all keys, and of each capped unit) moves with every plan by its planned windows, and
  materialize replaces a plan's planned windows by the windows it keeps, so a sequence of plans alone (a plan check)
  and the training's materialized steps read the same quantity. enforce_window_caps drops a unit's windows, its last
  one (largest body id) first, while the unit's windows of the phase, the tally before the micro-step plus its own,
  exceed its cap times the phase's windows, the tally plus the micro-step's: once on the real items before the
  synthetic budget is set, then jointly with the token caps (enforce_caps) on the whole plan, and again on the exact
  counts. What fills the windows it drops (the refill, _cap_and_refill): the tokens they held go, within the same
  micro-step, to the replay's real sources (refill_keys), one budget that they fill in proportion to their shares
  (_fill_pool: each draw from the source farthest below its part; persons uniform, windows_per_person as the first
  fill), checked against the real token caps again; the
  synthetic budget then follows the real tokens as before, so the synthetic share of a micro-step keeps its bound
  (replay_share x synthetic_share[1]) and every source and unit keeps its token cap. Without a refill a micro-step
  short of the removed tokens leaves the other units above unit_cap of its smaller total, and enforce_caps cascades:
  under the S2b configuration without the refill, 143 of 2,000 plans stayed empty after 8 redraws and a plan held
  0.3493 of tokens_per_micro (reports/bench/stage2_s2b_recipe/iterations/0_no_refill.md); the other units, near
  unit_cap already, cannot take the tokens. Credits (_credit_tokens): a capped unit's
  credit reads its tokens before the window caps (the budget the caps leave unused is not the lumpiness the credit
  carries, and would otherwise saturate its budget and the refill), and a replay source's credit reads its first
  fill only (the refill carries no credit), so that the replay takes the tokens the units drew as in a run without
  the cap. state_dict() carries the tally while the cap is on.
The plan, the budgets, the credits, the caps after the draw and the redraws are those of phase B with one cap per
budget key (caps: source_cap for a replay source, unit_cap for a unit); a plan item of a unit carries the unit name
as its source (the budget key; its window carries the store's source name), so the token tallies count units apart
from the replay's sources of the same name (weargait_pd:PD apart from the weargait_pd controls). The guard reads
the held-out persons of both stores. The configuration SHA-256 of the state covers the mixture and the stage2
section; phases A and B read the mixture section alone, as before.
"""
import copy
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .. import paths
from ..factory import layouts as LY

SYNTHETIC = "synthetic"
PHASES = ("A", "B")
PHASE_C = "C"                    # the stage-2 phase (module docstring, Phase C)
RATE = 100.0                     # body clock, Hz (data/body_store.RATE)
FORMAT = "mdfm-sampler/1"
# single-site records of one recording (module docstring, Phase C): source -> the key suffix that names the site
ONE_RECORDING = {"pads": re.compile(r"_wrist_[LR]$")}
S2_REPORT = os.path.join(paths.TREE, "reports", "window_store_s2", "hashes.json")   # the S2 store's person hashes


def recording_id(source, key):
    """The recording a record key belongs to: the key without its site suffix for a source of ONE_RECORDING (a pads
    key CrossArms_wrist_L gives CrossArms), the key itself otherwise."""
    rx = ONE_RECORDING.get(source)
    return rx.sub("", str(key)) if rx is not None else str(key)


def unit_hours(source, persons, keys, spans):
    """Subject-hours of a unit (module docstring, Phase C): the sum over its recordings, (person, recording_id), of
    the longest span (s) among the recording's records, in hours."""
    best = {}
    for p, k, sp in zip(persons, keys, spans):
        r = (str(p), recording_id(source, k))
        best[r] = max(best.get(r, 0.0), float(sp))
    return float(sum(best.values())) / 3600.0


def _cap_of(cap, source):
    """The cap of a budget key: cap itself, or cap[source] when cap is a mapping (phase C)."""
    return cap[source] if isinstance(cap, dict) else cap


# ---------------------------------------------------------------- shares
def capped_shares(weights, total, cap):
    """{source: share} splitting `total` in proportion to weights {source: w > 0}, each share at most `cap`,
    the excess of a capped source spread over the others in proportion to their weights (water-filling).
    ValueError when total exceeds len(weights) x cap."""
    free = {s: float(w) for s, w in weights.items() if float(w) > 0}
    if total <= 0 or not free:
        if total > 1e-12:
            raise ValueError(f"a real share of {total} and no real source to take it")
        return {s: 0.0 for s in weights}
    if total > cap * len(free) + 1e-12:
        raise ValueError(f"a real share of {total:.4f} exceeds {len(free)} sources x cap {cap}")
    out, left = {}, float(total)
    while free:
        tw = sum(free.values())
        prop = {s: left * w / tw for s, w in free.items()}
        over = [s for s, p in prop.items() if p > cap + 1e-12]
        if not over:
            out.update(prop)
            break
        for s in over:
            out[s] = float(cap)
            left -= cap
            del free[s]
    return {s: out.get(s, 0.0) for s in weights}


def _mask(text):
    """(G, C) bool of a records.csv chan_mask cell ('111000000|...')."""
    return np.array([[c == "1" for c in row] for row in str(text).split("|")], dtype=bool)


def stream_groups(modality, chan_mask):
    """(channel groups, nuisance tokens) of an input stream (tokenize.patchify): per IMU site an acc group
    and a gyro group when those channels are present and one nuisance token; a pose stream one group and one
    nuisance token."""
    cm = np.asarray(chan_mask, dtype=bool)
    if modality == "imu":
        return int(cm[:, 0:3].any(1).sum() + cm[:, 3:6].any(1).sum()), int(cm.shape[0])
    if modality in ("pose2d", "pose3d"):
        return 1, 1
    return 0, 0


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class EmptyStep(ValueError):
    """A plan or a materialized step that the caps leave without a window (module docstring, redraws)."""


# ---------------------------------------------------------------- plan
@dataclass
class Item:
    """One window of a plan: source ('synthetic' or a real source), person id, planned tokens; unit (the unit
    of its person in the micro-step), body (its window, shared by the two renders of a pair), pair (render-pair
    id, -1 unpaired), second (the second render of a pair); real: recording (WindowStore.recordings row), t0
    and dur (s, recording clock); synthetic: take (body-store take row), start (first knot), dur ((knots - 1)
    / 100 s), layout (a concrete Layout)."""
    source: str
    person: str
    tokens: int
    unit: int = -1
    body: int = -1
    pair: int = -1
    second: bool = False
    recording: int = -1
    t0: float = 0.0
    dur: float = 0.0
    take: int = -1
    start: int = 0
    layout: object = None


@dataclass
class Plan:
    """The items of one micro-step with its index, phase, seed and the token budgets {source: tokens}."""
    step: int
    phase: str
    seed: int
    items: list
    budgets: dict

    def tokens(self):
        return source_tokens(self.items)


def source_tokens(items):
    """{source: tokens} of items."""
    out = {}
    for it in items:
        out[it.source] = out.get(it.source, 0) + int(it.tokens)
    return out


def cap_violations(items, cap, synthetic=SYNTHETIC):
    """Real sources holding more than cap of all the items' tokens: {source: share}. cap: a share, or {key: share}
    per budget key (phase C)."""
    by = source_tokens(items)
    total = sum(by.values())
    return {s: t / total for s, t in by.items() if s != synthetic and total and t > _cap_of(cap, s) * total + 1e-9}


def enforce_caps(items, cap, synthetic_share=1.0, synthetic=SYNTHETIC):
    """(items, bodies dropped): while a real source holds more than cap of all tokens, its last body (largest
    body id: one window, or both renders of a render pair) is dropped, the source farthest over the cap
    first; then, when synthetic_share is under 1 and the synthetic tokens exceed synthetic_share of all tokens,
    the last synthetic body is dropped; until both hold. cap: a share, or {key: share} per budget key (phase C,
    module docstring)."""
    items = list(items)
    dropped = 0
    while True:
        by = source_tokens(items)
        total = sum(by.values())
        over = {s: t - _cap_of(cap, s) * total for s, t in by.items()
                if s != synthetic and t > _cap_of(cap, s) * total + 1e-9}
        if over:
            s = max(sorted(over), key=lambda x: over[x])
        elif synthetic_share < 1.0 and by.get(synthetic, 0) > synthetic_share * total + 1e-9:
            s = synthetic
        else:
            return items, dropped
        u = max(it.body for it in items if it.source == s)
        items = [it for it in items if not (it.source == s and it.body == u)]
        dropped += 1


def window_cap_excess(items, caps, base_all=0, base=None):
    """{key: windows over the cap} of the keys of caps (phase C window caps, module docstring) whose windows of the
    phase so far, base[key] before the micro-step plus the items', exceed caps[key] times the phase's windows, base_all
    plus the items' count; only keys that hold an item."""
    base = base or {}
    n = len(items)
    cnt = {}
    for it in items:
        if it.source in caps:
            cnt[it.source] = cnt.get(it.source, 0) + 1
    out = {}
    for k, c in cnt.items():
        over = int(base.get(k, 0)) + c - float(caps[k]) * (int(base_all) + n)
        if over > 1e-9:
            out[k] = over
    return out


def enforce_window_caps(items, caps, base_all=0, base=None):
    """(items, bodies dropped): while a key of caps holds more than its cap of the phase's windows (window_cap_excess),
    its last body (largest body id; both renders of a pair) is dropped, the key farthest over its cap first."""
    items = list(items)
    dropped = 0
    while True:
        over = window_cap_excess(items, caps, base_all, base)
        if not over:
            return items, dropped
        s = max(sorted(over), key=lambda x: over[x])
        u = max(it.body for it in items if it.source == s)
        items = [it for it in items if not (it.source == s and it.body == u)]
        dropped += 1


@dataclass
class Step:
    """A materialized micro-step: plan (items after the exact caps, tokens exact), windows (windows.Window, in
    item order), tokens {source: exact tokens}, dropped (bodies dropped by the exact caps), mismatched (real
    windows whose exact count differed from the plan: 0 unless the store's grid is not what the plan read)."""
    plan: Plan
    windows: list
    tokens: dict
    dropped: int = 0
    mismatched: int = 0

    @property
    def seed(self):
        return self.plan.seed


class MixtureSampler:
    """The mixture sampler (module docstring). body_store: data.body_store.BodyStore of split train (or an
    object with takes (person_id, n_knots), split and windows(takes, starts, win_s)); window_store:
    data.window_store.WindowStore (phase B; None in phase A); synth: the renderer (factory.synth.SynthSampler
    or an object with render(bodies, specs, gen)), None for planning only; phase 'A' or 'B'; seed: int; cfg:
    train.config.MixtureSection or a RunConfig (its mixture); layouts: {name: Layout} (default
    layouts.load_layouts(), in phase C layouts.load_stage2_layouts()); guard: run provenance.assert_clean on every materialized step; redraws: draws again
    a plan or a step the caps leave empty, at most that many times (opt-in, module docstring; 0 by default, the
    former behaviour). redrawn counts the draws made again. tier: the run's tier, passed to the guard (default
    RELEASE, the guard's default). Phase 'C' (module docstring, Phase C): cfg the RunConfig, window_store a
    Stage2Stores; stage2 holds the config's stage2 section, caps the cap of each budget key and store_of the store
    each real key reads (both empty in phases A and B, where every real source reads window_store under
    source_cap)."""

    def __init__(self, body_store, window_store, synth, phase, seed, cfg=None, layouts=None, guard=True,
                 redraws=0, tier=None):
        from ..train import config as C
        if phase not in PHASES + (PHASE_C,):
            raise ValueError(f"phase must be one of {PHASES + (PHASE_C,)}, got {phase!r}")
        run = cfg
        cfg = C.MixtureSection() if cfg is None else cfg
        cfg = getattr(cfg, "mixture", cfg)
        self.cfg, self.phase, self.seed = cfg, phase, int(seed)
        self.synth, self.body_store, self.window_store = synth, body_store, window_store
        self.guard_on = bool(guard)
        self.tier = "RELEASE" if tier is None else str(tier)
        if layouts is None:          # phase C renders the stage-1 and the S2a families (layouts.load_stage2_layouts)
            layouts = LY.load_stage2_layouts() if phase == PHASE_C else LY.load_layouts()
        self.layouts = layouts
        self.mixture = {k: float(v) for k, v in cfg.layouts.items() if float(v) > 0}
        self.K = int(round(float(cfg.win_s) * RATE))
        self.stage2, self.caps, self.store_of = None, None, {}
        self.window_caps, self.wtally = None, None      # the phase C window cap (module docstring), off
        self.refill_keys = ()
        if phase == PHASE_C:
            self._init_phase_c(run, body_store, window_store)
        else:
            syn_share = float(cfg.synthetic_share[PHASES.index(phase)])
            self._init_bodies(body_store, syn_share)
            self._init_real(window_store, 1.0 - syn_share)
            self.shares = {SYNTHETIC: syn_share}
            self.shares.update(capped_shares(self.weights, 1.0 - syn_share, float(cfg.source_cap)))
            self.order = [SYNTHETIC] + [s for s in cfg.sources if s in self.real]
        self.held_out = None
        self.rng = np.random.default_rng(self.seed)
        self.step = 0
        self.credit = {}
        self.totals = {"tokens": dict.fromkeys(self.order, 0), "windows": dict.fromkeys(self.order, 0)}
        self.redraws = redraws
        self.redrawn = 0

    @property
    def redraws(self):
        """Draws made again, at most, of a plan or a step the caps leave empty (module docstring)."""
        return self._redraws

    @redraws.setter
    def redraws(self, n):
        if isinstance(n, bool) or int(n) != n or int(n) < 0:
            raise ValueError(f"redraws must be a non-negative integer, got {n!r}")
        self._redraws = int(n)

    # ------------------------------------------------------------ sources
    def _init_bodies(self, store, share):
        self.bodies = None
        if share <= 0:
            return
        if store is None:
            raise ValueError("a synthetic share needs a body store")
        split = getattr(store, "split", "train")
        if split != "train":
            from .canonical import Refused
            raise Refused(f"the body store of split {split!r} never trains (train only)")
        t = store.takes
        persons = np.asarray(t["person_id"]).astype(str)
        nk = np.asarray(t["n_knots"], dtype=np.int64)
        names, inv = np.unique(persons, return_inverse=True)
        takes = [np.flatnonzero(inv == i) for i in range(len(names))]
        self.bodies = {"persons": names, "takes": takes, "n_knots": nk,
                       "p": [nk[x] / nk[x].sum() for x in takes]}

    def _init_real(self, store, share):
        """Per real source of cfg.sources present in the store: its train-fold recordings of at least min_s
        (recording row, person, span, input grid, groups) and subject-hours."""
        self.real, self.weights, self.hours = {}, {}, {}
        if share <= 0:
            return
        if store is None:
            raise ValueError("a real share needs a window store")
        for s, table, hours in self._real_tables(store, [(s, s) for s in self.cfg.sources]):
            self.real[s] = table
            self.hours[s] = hours
            self.weights[s] = math.sqrt(self.hours[s])

    def _real_tables(self, store, keys, grouped=False):
        """[(key, table, subject-hours)] for the (key, source) pairs of keys whose source holds train-fold
        recordings of at least min_s in the store (_init_real's table of each); subject-hours: the spans summed, or
        with grouped, unit_hours (one recording counted once, module docstring, Phase C)."""
        rec, recs = store.records, store.recordings
        inp = rec[rec["role"] == "input"]
        n_in = inp["recording"].astype(int).value_counts().to_dict()
        by_rec = {int(r): i for i, r in zip(inp.index, inp["recording"])}
        out = []
        for key, s in keys:
            ok = (recs["source"] == s) & (recs["fold"] == "train") & (recs["span_s"] >= float(self.cfg.min_s) - 1e-9)
            g = recs[ok]
            if not len(g):
                continue
            rows = np.asarray(g.index, dtype=np.int64)
            bad = [int(i) for i in rows if n_in.get(int(i), 0) != 1]
            if bad:
                raise ValueError(f"{s}: {len(bad)} recordings without exactly one input stream (the planned token "
                                 f"count reads one), first {bad[:5]}")
            ir = [by_rec[int(i)] for i in rows]
            r = inp.loc[ir]
            groups = [stream_groups(m, _mask(c)) for m, c in zip(r["stream"], r["chan_mask"])]
            person = np.asarray(g["person_id"]).astype(str)
            names, inv = np.unique(person, return_inverse=True)
            span = np.asarray(g["span_s"], dtype=np.float64)
            per = [np.flatnonzero(inv == i) for i in range(len(names))]
            table = {"rows": rows, "persons": names, "per": per, "p": [span[x] / span[x].sum() for x in per],
                     "span_t0": np.asarray(g["span_t0"], dtype=np.float64), "span": span,
                     "offset": np.asarray(r["offset"], dtype=np.float64),
                     "step": np.asarray(r["k"], np.float64) / np.asarray(r["fps_native"], np.float64),
                     "n": np.asarray(r["n"], dtype=np.int64),
                     "g": np.array([x[0] for x in groups], np.int64),
                     "nuis": np.array([x[1] for x in groups], np.int64)}
            hours = unit_hours(s, person, g["key"], span) if grouped else float(span.sum()) / 3600.0
            out.append((key, table, hours))
        return out

    def _init_phase_c(self, run, body_store, stores):
        """The shares, caps and stores of phase C (module docstring, Phase C). ValueError: no enabled stage2 section,
        a window store that is not a Stage2Stores, a unit whose name is not its source's stage-2 unit
        (sources.STAGE2_S2A) or without a training recording of at least min_s, a pathological share the unit caps
        cannot absorb."""
        st = getattr(run, "stage2", None)
        if st is None or not bool(st.enabled):
            raise ValueError("phase C reads the run config's stage2 section (enabled): give the RunConfig as cfg")
        if not isinstance(stores, Stage2Stores):
            raise ValueError("phase C reads a Stage2Stores (the stage-1 replay root and the stage-2 root)")
        cfg = self.cfg
        R, s_b = float(st.replay_share), float(cfg.synthetic_share[PHASES.index("B")])
        syn = R * s_b
        self._init_bodies(body_store, syn)
        self._init_real(stores.replay, R * (1.0 - s_b))
        replay = capped_shares(self.weights, 1.0 - s_b, float(cfg.source_cap)) if self.weights else {}
        units = self._init_units(stores.stage2, tuple(st.units))
        unit = capped_shares({u: self.weights[u] for u in units}, 1.0 - R, float(st.unit_cap))
        self.stage2 = st
        self.shares = {SYNTHETIC: syn}
        self.shares.update({s: R * v for s, v in replay.items()})
        self.shares.update(unit)
        self.caps = {**{s: float(cfg.source_cap) for s in replay}, **{u: float(st.unit_cap) for u in units}}
        self.store_of = {**{s: stores.replay for s in replay}, **{u: stores.stage2 for u in units}}
        self.order = [SYNTHETIC] + [s for s in cfg.sources if s in replay] + list(units)
        wc = float(getattr(st, "window_cap_per_person", 0.0) or 0.0)
        if wc > 0.0:                                     # task S2B (module docstring, the window cap)
            self.window_caps = {u: min(1.0, wc * len(self.real[u]["persons"])) for u in units}
            self.wtally = {"all": 0, **{u: 0 for u in units}}
            self.refill_keys = tuple(s for s in self.order if s in replay)    # the replay's real sources

    def _init_units(self, store, units):
        """The units' tables, subject-hours (unit_hours) and weights added to real, hours and weights; returns the
        units in order (_init_phase_c)."""
        from . import sources as SRC
        keys = []
        for u in units:
            src = str(u).split(":", 1)[0]
            spec = SRC.STAGE2_S2A.get(src)
            if spec is not None and spec.unit != u:
                raise ValueError(f"unit {u!r}: the stage-2 store's {src} folder holds the unit {spec.unit!r}")
            keys.append((u, src))
        tabs = self._real_tables(store, keys, grouped=True)
        got = [k for k, _, _ in tabs]
        missing = [u for u in units if u not in got]
        if missing:
            raise ValueError(f"units without a training recording of at least {self.cfg.min_s} s in the stage-2 "
                             f"store: {missing}")
        for u, table, hours in tabs:
            self.real[u] = table
            self.hours[u] = hours
            self.weights[u] = math.sqrt(hours)
        return tuple(units)

    def _cap(self, key):
        """The cap of a real budget key: source_cap, or caps[key] in phase C."""
        return float(self.cfg.source_cap) if self.caps is None else self.caps[key]

    def _caps(self, scale=None):
        """The cap argument of enforce_caps: source_cap (phases A and B) or {key: cap} (phase C), each mapped through
        scale when given."""
        f = (lambda c: c) if scale is None else scale
        if self.caps is None:
            return f(float(self.cfg.source_cap))
        return {k: f(c) for k, c in self.caps.items()}

    # ------------------------------------------------------------ token counts
    def real_tokens(self, source, j, t0, dur):
        """Exact token count of the real window of recording j (index into the source's table) between t0 and
        t0 + dur s (module docstring)."""
        R = self.real[source]
        off, step, n = R["offset"][j], R["step"][j], int(R["n"][j])

        def first_at_or_after(t):
            i = min(max(int(math.ceil((t - off) / step)), 0), n)
            while i > 0 and off + (i - 1) * step >= t:
                i -= 1
            while i < n and off + i * step < t:
                i += 1
            return i

        P = LY.n_patches(dur)
        j0, j1 = first_at_or_after(t0), first_at_or_after(t0 + dur)
        count = 0
        if j1 > j0:
            a = math.floor(float(np.float32(off + j0 * step - t0)) / LY.PATCH_S)
            b = math.floor(float(np.float32(off + (j1 - 1) * step - t0)) / LY.PATCH_S)
            count = max(0, min(b, P - 1) - max(a, 0) + 1)
        return P + 1 + int(R["g"][j]) * count + int(R["nuis"][j])

    # ------------------------------------------------------------ draws
    def _real_unit(self, s, rng):
        R = self.real[s]
        p = int(rng.integers(len(R["persons"])))
        cand, w = R["per"][p], R["p"][p]
        out, seen = [], set()
        for _ in range(int(self.cfg.windows_per_person)):
            j = int(cand[int(rng.choice(len(cand), p=w))]) if len(cand) > 1 else int(cand[0])
            span, st0 = float(R["span"][j]), float(R["span_t0"][j])
            dur = min(float(self.cfg.win_s), span)
            t0 = st0 + (float(rng.uniform(0.0, span - dur)) if span > dur else 0.0)
            key = (j, t0)
            if key in seen:
                continue
            seen.add(key)
            out.append([Item(s, str(R["persons"][p]), self.real_tokens(s, j, t0, dur), recording=int(R["rows"][j]),
                             t0=t0, dur=dur)])
        return out

    def _synthetic_unit(self, rng):
        B = self.bodies
        p = int(rng.integers(len(B["persons"])))
        cand = B["takes"][p]
        out, seen = [], set()
        for _ in range(int(self.cfg.windows_per_person)):
            i = int(cand[int(rng.choice(len(cand), p=B["p"][p]))]) if len(cand) > 1 else int(cand[0])
            nk = int(B["n_knots"][i])
            start = int(rng.integers(max(nk - self.K, 0) + 1))
            lay = LY.sample_layout(rng, self.mixture, self.layouts)
            twice = bool(rng.random() < float(self.cfg.pair_share))
            if (i, start) in seen:
                continue
            seen.add((i, start))
            dur = (min(self.K, nk - start) - 1) / RATE
            first = Item(SYNTHETIC, str(B["persons"][p]), LY.token_count(lay, dur), take=i, start=start, dur=dur,
                         layout=lay)
            body = [first]
            if twice:
                body.append(Item(SYNTHETIC, first.person, first.tokens, take=i, start=start, dur=dur, layout=lay,
                                 second=True))
            out.append(body)
        return out

    def _charge(self, s, bodies):
        """Budget charge of drawn bodies: their tokens for a real source; for synthetic bodies, each body's single
        render times (1 + pair_share), whatever its pairing, so that fitting a budget never selects against
        render pairs."""
        if s != SYNTHETIC:
            return sum(it.tokens for b in bodies for it in b)
        return sum(b[0].tokens for b in bodies) * (1.0 + float(self.cfg.pair_share))

    @staticmethod
    def _trim(items, budget):
        """Drop the last synthetic bodies (both renders of a pair) while the synthetic tokens exceed budget."""
        while items and sum(it.tokens for it in items if it.source == SYNTHETIC) > budget:
            u = max(it.body for it in items if it.source == SYNTHETIC)
            items[:] = [it for it in items if not (it.source == SYNTHETIC and it.body == u)]
        return items

    def _fill(self, sources, budgets, max_w, items, rng, ids):
        """Units of `sources` drawn in turn (round robin) into items until each budget (in charges, _charge)
        closes or items holds max_w windows; ids: [next unit id, next pair id, next body id]. Returns {source:
        charge added}."""
        used = dict.fromkeys(sources, 0)
        misses = dict.fromkeys(sources, 0)
        open_ = [s for s in sources if budgets[s] > 0]
        while open_ and len(items) < max_w:
            for s in list(open_):
                bodies = self._synthetic_unit(rng) if s == SYNTHETIC else self._real_unit(s, rng)
                room_t, room_w = budgets[s] - used[s], max_w - len(items)
                chosen = None
                for cand in (bodies, bodies[:1]):
                    charge = self._charge(s, cand)
                    if cand and charge <= room_t + 1e-9 and sum(len(b) for b in cand) <= room_w:
                        chosen = cand
                        break
                if chosen is None:
                    misses[s] += 1
                    if misses[s] >= int(self.cfg.max_misses):
                        open_.remove(s)
                    continue
                misses[s] = 0
                used[s] += charge
                for b in chosen:
                    if len(b) > 1:
                        for it in b:
                            it.pair = ids[1]
                        ids[1] += 1
                    for it in b:
                        it.unit, it.body = ids[0], ids[2]
                        items.append(it)
                    ids[2] += 1
                ids[0] += 1
                if len(items) >= max_w:
                    break
        return used

    def _fill_pool(self, sources, weights, pool, max_w, items, rng, ids):
        """Units of real `sources` drawn into items against one shared token budget `pool` (the refill of the window
        cap, _cap_and_refill), so that each source's tokens approach pool x its weight over the weights' sum: each draw
        takes the open source farthest below that target (the first in `sources` order on a tie), draws one of its
        units (_real_unit; shrunk to its first window when the whole does not fit) and charges its tokens to the pool;
        max_misses draws of a source in a row that do not fit close it; until the pool or the sources are exhausted or
        items holds max_w windows. Returns the tokens drawn."""
        open_ = [s for s in sources if float(weights.get(s, 0.0)) > 0]
        tw = sum(float(weights[s]) for s in open_)
        target = {s: float(pool) * float(weights[s]) / tw for s in open_}
        got = dict.fromkeys(open_, 0.0)
        misses = dict.fromkeys(open_, 0)
        used = 0
        while open_ and used < pool and len(items) < max_w:
            s = max(open_, key=lambda k: (target[k] - got[k], -open_.index(k)))
            bodies = self._real_unit(s, rng)
            room_t, room_w = pool - used, max_w - len(items)
            chosen = None
            for cand in (bodies, bodies[:1]):
                charge = self._charge(s, cand)
                if cand and charge <= room_t + 1e-9 and sum(len(b) for b in cand) <= room_w:
                    chosen = cand
                    break
            if chosen is None:
                misses[s] += 1
                if misses[s] >= int(self.cfg.max_misses):
                    open_.remove(s)
                continue
            misses[s] = 0
            used += charge
            got[s] += charge
            for b in chosen:
                for it in b:
                    it.unit, it.body = ids[0], ids[2]
                    items.append(it)
                ids[2] += 1
            ids[0] += 1
        return used

    def plan(self):
        """The Plan of the next micro-step (_plan_once); advances the generator and the step counter. With redraws
        k > 0, a draw that the caps leave empty (EmptyStep) is made again, at most k times."""
        for attempt in range(self.redraws + 1):
            try:
                return self._plan_once()
            except EmptyStep:
                if attempt >= self.redraws:
                    raise
                self.redrawn += 1

    def _plan_once(self):
        """One draw of the Plan of the next micro-step; advances the generator, and the step counter unless the
        caps leave the plan empty (EmptyStep, the credits then unchanged). Real sources first:
        each budget is its share of tokens_per_micro plus the budget it left unused in earlier micro-steps
        (credit), at most source_cap of tokens_per_micro, filled in turn within the real share of max_windows.
        Then the synthetic budget: synthetic_share / (1 - synthetic_share) times the real tokens drawn, at most
        tokens_per_micro minus them (all of tokens_per_micro in phase A), filled in charges (_charge), trimmed
        to the budget in tokens (_trim), and what is left filled and trimmed once more. Then enforce_caps. A
        micro-step never exceeds tokens_per_micro tokens."""
        rng, cfg = self.rng, self.cfg
        seed = int(rng.integers(0, 2 ** 62))
        N = int(cfg.tokens_per_micro)
        s_syn = self.shares[SYNTHETIC]
        real = [s for s in self.order if s != SYNTHETIC]
        budgets = {s: int(min(self._cap(s) * N, math.floor(self.shares[s] * N + 1e-9) + self.credit.get(s, 0)))
                   for s in real}
        items, ids = [], [0, 0, 0]
        max_w = int(cfg.max_windows)
        real_w = max_w if s_syn <= 0 else max(1, int(math.floor(max_w * (1.0 - s_syn))))
        self._fill(real, budgets, real_w, items, rng, ids)
        if s_syn < 1.0 and items:
            # the cap against the total the synthetic budget will complete: real / (1 - synthetic_share)
            items, _ = enforce_caps(items, self._caps(lambda c: min(1.0, c / (1.0 - s_syn))))
        refill = None
        if self.window_caps:                 # the window cap on the real items, before the synthetic budget (S2B)
            items, refill = self._cap_and_refill(items, real_w, rng, ids, s_syn)
        r_tok = sum(it.tokens for it in items)
        if s_syn >= 1.0:
            budgets[SYNTHETIC] = N
        elif s_syn > 0:
            budgets[SYNTHETIC] = int(min(math.floor(s_syn / (1.0 - s_syn) * r_tok + 1e-9), N - r_tok))
        B = budgets.get(SYNTHETIC, 0)
        if B > 0:
            self._fill([SYNTHETIC], budgets, max_w, items, rng, ids)
            self._trim(items, B)
            left = B - sum(it.tokens for it in items if it.source == SYNTHETIC)
            if left > 0:
                self._fill([SYNTHETIC], {SYNTHETIC: left}, max_w, items, rng, ids)
                self._trim(items, B)
        items, _ = enforce_caps(items, self._caps(), s_syn)
        if self.window_caps:
            items, _ = self._enforce_windows(items, s_syn)
        if not items:
            raise EmptyStep(f"tokens_per_micro {N} leaves no window under the caps: budgets {budgets}")
        got = source_tokens(items) if refill is None else self._credit_tokens(items, refill)
        self.credit = {s: int(min(self._cap(s) * N, max(0, budgets[s] - got.get(s, 0)))) for s in real}
        plan = Plan(self.step, self.phase, seed, items, budgets)
        self.step += 1
        if self.window_caps:
            self._tally(plan.items, 1)
        return plan

    def _cap_and_refill(self, items, real_w, rng, ids, s_syn):
        """The window caps on the real items of a plan (enforce_window_caps against the tally before this micro-step),
        then the refill (module docstring, the window cap): the tokens the caps took from the units form one budget
        that the replay's real sources fill in proportion to their shares (_fill_pool; new units, ids from refill_from
        on), and the real token caps are checked again. Returns (items, {pre: the tokens
        per key before the window caps, refill_from: the first unit id of the refill or None, freed, drawn})."""
        before = source_tokens(items)
        items, _ = enforce_window_caps(items, self.window_caps, self.wtally["all"], self.wtally)
        after = source_tokens(items)
        freed = sum(before.get(u, 0) - after.get(u, 0) for u in self.window_caps)
        keys = [s for s in self.refill_keys if self.shares.get(s, 0.0) > 0]
        info = {"pre": before, "refill_from": None, "freed": int(freed), "drawn": 0}
        if freed <= 0 or not keys:
            return items, info
        info["refill_from"] = int(ids[0])
        info["drawn"] = int(self._fill_pool(keys, {s: self.shares[s] for s in keys}, freed, real_w, items, rng, ids))
        if s_syn < 1.0 and items:
            items, _ = enforce_caps(items, self._caps(lambda c: min(1.0, c / (1.0 - s_syn))))
        return items, info

    def _credit_tokens(self, items, refill):
        """{key: tokens} that the credits of a plan under the window cap read: a capped unit's tokens before the window
        caps (the budget the caps leave unused is not lumpiness and carries no credit), a replay source's tokens of its
        first fill (its refill draws, unit ids from refill_from on, neither use its budget nor carry credit), every
        other key's tokens in the plan."""
        rf = refill.get("refill_from")
        base = items if rf is None else [it for it in items if not (it.source in self.refill_keys and it.unit >= rf)]
        got = source_tokens(base)
        for u in self.window_caps:
            got[u] = int(refill["pre"].get(u, 0))
        return got

    def _enforce_windows(self, items, s_syn):
        """(items, bodies dropped): the window caps (enforce_window_caps against the tally before this micro-step) and
        the token caps (enforce_caps with the synthetic share s_syn) in turn until the token caps drop nothing (module
        docstring, the window cap)."""
        dropped = 0
        while True:
            items, a = enforce_window_caps(items, self.window_caps, self.wtally["all"], self.wtally)
            items, b = enforce_caps(items, self._caps(), s_syn)
            dropped += a + b
            if not b:
                return items, dropped

    def _tally(self, items, sign):
        """Add (sign 1) or remove (sign -1) the windows of items to or from the window tally."""
        self.wtally["all"] += int(sign) * len(items)
        for it in items:
            if it.source in self.window_caps:
                self.wtally[it.source] += int(sign)

    # ------------------------------------------------------------ materialize
    def materialize(self, plan):
        """Step of a plan: real windows read from the window store, synthetic ones rendered in one call, the
        exact token counts, the caps on them, the guard."""
        n = len(plan.items)
        wins = [None] * n
        mism = 0
        syn = [k for k, it in enumerate(plan.items) if it.source == SYNTHETIC]
        for k, it in enumerate(plan.items):
            if it.source != SYNTHETIC:
                wins[k] = self.store_of.get(it.source, self.window_store).window(it.recording, it.t0, it.dur)
        if syn:
            if self.synth is None:
                raise ValueError("synthetic windows need a renderer (synth)")
            its = [plan.items[k] for k in syn]
            bodies = self.body_store.windows([it.take for it in its], [it.start for it in its], float(self.cfg.win_s))
            out = self.synth.render(bodies, [it.layout for it in its], np.random.default_rng([plan.seed, 1]))
            for k, w in zip(syn, out):
                wins[k] = w
        items = []
        for it, w in zip(plan.items, wins):
            exact = LY.window_tokens(w)
            if it.source != SYNTHETIC and exact != it.tokens:
                mism += 1
            items.append(copy.copy(it))
            items[-1].tokens = int(exact)
        keep_items, dropped = enforce_caps(items, self._caps(), self.shares[SYNTHETIC])
        if self.window_caps:                 # the tally back to before this plan, then the caps on the exact counts
            self._tally(plan.items, -1)
            keep_items, more = self._enforce_windows(keep_items, self.shares[SYNTHETIC])
            dropped += more
        if not keep_items:
            raise EmptyStep(f"step {plan.step}: the caps on the exact token counts leave no window")
        kept = {id(it) for it in keep_items}
        windows = [w for it, w in zip(items, wins) if id(it) in kept]
        if self.guard_on:
            self.guard(windows)
        step = Step(Plan(plan.step, plan.phase, plan.seed, keep_items, plan.budgets), windows,
                    source_tokens(keep_items), dropped, mism)
        if self.window_caps:
            self._tally(keep_items, 1)
        for it in keep_items:
            self.totals["tokens"][it.source] = self.totals["tokens"].get(it.source, 0) + it.tokens
            self.totals["windows"][it.source] = self.totals["windows"].get(it.source, 0) + 1
        return step

    def guard(self, windows):
        """provenance.assert_clean on the windows' (source, person_id) at the run's tier, with the held-out persons
        (module docstring; in phase C those of both stores); raises canonical.Refused."""
        from ..train import provenance as PV
        if self.held_out is None:
            self.held_out = PV.held_out_persons(self.window_store)
        PV.assert_clean(windows, held_out=self.held_out, tier=self.tier)
        return windows

    def next(self):
        """The next materialized micro-step (Step). With redraws k > 0, a step that the caps on the exact counts
        leave empty (EmptyStep) is replaced by a new plan, at most k times; nothing of the empty step is counted
        (its plan has moved the credits)."""
        for attempt in range(self.redraws + 1):
            plan = self.plan()
            try:
                return self.materialize(plan)
            except EmptyStep:
                if attempt >= self.redraws:
                    raise
                self.redrawn += 1

    __next__ = next

    def __iter__(self):
        return self

    # ------------------------------------------------------------ state
    def config_sha256(self):
        """SHA-256 of the mixture section (phases A and B), or of the mixture and the stage2 sections (phase C)."""
        from ..train import config as C
        if self.stage2 is None:
            return _sha(C.to_dict(self.cfg))
        return _sha({"mixture": C.to_dict(self.cfg), "stage2": C.to_dict(self.stage2)})

    def data_sha256(self):
        """SHA-256 of the tables the draws read: the body-store persons, their takes and knot counts, and per
        real source its persons, recordings, spans and sample grids. A store rebuilt with other content moves
        it, and the same generator state would then draw other windows."""
        h = hashlib.sha256()
        B = self.bodies
        if B is not None:
            h.update(("bodies\n" + "\n".join(B["persons"]) + "\n").encode("utf-8"))
            h.update(np.ascontiguousarray(B["n_knots"], np.int64).tobytes())
            for x in B["takes"]:
                h.update(np.ascontiguousarray(x, np.int64).tobytes() + b"|")
        for s in sorted(self.real):
            R = self.real[s]
            h.update((f"real {s}\n" + "\n".join(R["persons"]) + "\n").encode("utf-8"))
            for x in R["per"]:
                h.update(np.ascontiguousarray(x, np.int64).tobytes() + b"|")
            for k in ("rows", "span_t0", "span", "offset", "step", "n", "g", "nuis"):
                h.update(k.encode("utf-8") + np.ascontiguousarray(R[k]).tobytes())
        return h.hexdigest()

    def state_dict(self):
        """Resumable state: format, phase, seed, config SHA-256, data SHA-256, step, generator state,
        real-source credits, totals; redraws when it is not 0 (opt-in); window_tally while the phase C window cap is
        on (task S2B)."""
        state = {"format": FORMAT, "phase": self.phase, "seed": self.seed, "config_sha256": self.config_sha256(),
                 "data_sha256": self.data_sha256(), "step": int(self.step),
                 "rng": copy.deepcopy(self.rng.bit_generator.state), "credit": dict(self.credit),
                 "totals": copy.deepcopy(self.totals)}
        if self.redraws:
            state["redraws"] = int(self.redraws)
        if self.wtally is not None:
            state["window_tally"] = dict(self.wtally)
        return state

    def load_state_dict(self, state):
        """Resume from state_dict(); a state of another phase, seed, mixture configuration, store content
        (data_sha256) or redraws value (absent: 0) is refused, and under a window cap a state without a fitting
        window_tally."""
        mine = self.state_dict()
        for k in ("format", "phase", "seed", "config_sha256", "data_sha256"):
            if state.get(k) != mine[k]:
                raise ValueError(f"sampler state {k} {state.get(k)!r} differs from this sampler's {mine[k]!r}")
        if int(state.get("redraws", 0)) != self.redraws:
            raise ValueError(f"sampler state redraws {state.get('redraws', 0)!r} differs from this sampler's "
                             f"{self.redraws!r}")
        if self.wtally is not None:
            got = state.get("window_tally")
            if not isinstance(got, dict) or set(got) != set(self.wtally):
                raise ValueError(f"sampler state window_tally {got!r} does not fit this sampler's window caps "
                                 f"{sorted(self.wtally)}")
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.step = int(state["step"])
        self.credit = {str(k): int(v) for k, v in state["credit"].items()}
        self.totals = copy.deepcopy(state["totals"])
        if self.wtally is not None:
            self.wtally = {str(k): int(v) for k, v in state["window_tally"].items()}
        return self


# ---------------------------------------------------------------- the stores of phase C
class Stage2Stores:
    """The two window stores of a phase C run (module docstring, Phase C; spec section 4.4): replay, the stage-1 root
    read as it is (WindowStore under names.STAGE1, tier RELEASE: the healthy replay); stage2, the stage-2 root of
    the run's tier under the run's name tables (the units). The sampler reads each through its own key; the
    provenance reads their union: records (both tables, with a 'store' column), person_lists (per source, the union
    of the two stores' persons), person_hashes and sources (per store), report_mismatches (each store against its
    own report: reports/window_store/hashes.json, reports/window_store_s2/hashes.json). tier is the stage-2 store's.
    ValueError: a replay store that is not RELEASE or that carries a stage-2 marker."""

    def __init__(self, replay, stage2):
        import pandas as pd
        if getattr(replay, "tier", "RELEASE") != "RELEASE" or getattr(replay, "marker", None) is not None:
            raise ValueError("the replay store is the stage-1 root (RELEASE, no stage-2 marker)")
        self.replay, self.stage2 = replay, stage2
        self.tier = getattr(stage2, "tier", "RELEASE")
        self.records = pd.concat([replay.records.assign(store="replay"), stage2.records.assign(store="stage2")],
                                 ignore_index=True)
        self.sources = {**{f"replay/{k}": v for k, v in (getattr(replay, "sources", None) or {}).items()},
                        **{f"stage2/{k}": v for k, v in (getattr(stage2, "sources", None) or {}).items()}}

    def person_lists(self, fold=None):
        """{source: sorted person ids} of both stores (of one fold when given), the union per source."""
        out = {}
        for st in (self.replay, self.stage2):
            for s, ids in st.person_lists(fold).items():
                out[s] = sorted(set(out.get(s, [])) | set(ids))
        return out

    def person_hashes(self):
        """{'replay': ..., 'stage2': ...}, WindowStore.person_hashes of each store."""
        return {"replay": self.replay.person_hashes(), "stage2": self.stage2.person_hashes()}

    def report_mismatches(self, body_store=None, fold="train", report=None):
        """The person lists that differ from their reports (list of str): the body store and the replay store
        through provenance.report_mismatches, each source of the stage-2 store against `report` (default
        reports/window_store_s2/hashes.json, person_sha256[source][fold])."""
        from ..train import provenance as PV
        out = PV.report_mismatches(PV.person_lists(body_store, self.replay, fold), fold)
        path = S2_REPORT if report is None else report
        want = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                want = json.load(f).get("person_sha256", {})
        for s, ids in self.stage2.person_lists(fold).items():
            got, w = PV.person_hash(ids), (want.get(s) or {}).get(fold)
            if got != w:
                out.append(f"stage2 {s}/{fold}: {got} against reports/window_store_s2 {w}")
        return out

    def close(self):
        for st in (self.replay, self.stage2):
            if hasattr(st, "close"):
                st.close()


# ---------------------------------------------------------------- bucket batcher
@dataclass
class Packing:
    """Rows of one micro-step: row_len (tokens), placement [(row, offset)] per window, n_rows, useful tokens,
    padded tokens (n_rows x row_len)."""
    row_len: int
    placement: list
    n_rows: int
    useful: int
    padded: int = field(init=False)

    def __post_init__(self):
        self.padded = int(self.n_rows) * int(self.row_len)


class BucketBatcher:
    """Packs the windows of one micro-step into rows of one bucket (module docstring): rows, the bucket row
    lengths in tokens (increasing); max_windows per micro-step."""

    def __init__(self, rows=(128, 640, 1536), tokens_per_micro=16384, max_windows=512):
        self.rows = tuple(int(r) for r in rows)
        if not self.rows or any(b <= a for a, b in zip(self.rows, self.rows[1:])):
            raise ValueError(f"rows {rows}: increasing row lengths")
        self.tokens_per_micro, self.max_windows = int(tokens_per_micro), int(max_windows)

    def bucket(self, counts):
        """The smallest row length that holds the longest window (ValueError when none does)."""
        m = max(int(c) for c in counts)
        for r in self.rows:
            if m <= r:
                return r
        raise ValueError(f"a window of {m} tokens exceeds the largest row of {self.rows[-1]}")

    def pack(self, counts, row_len=None):
        """Packing of windows of `counts` tokens: first-fit decreasing into rows of row_len (default
        bucket(counts)); ties in order of the windows."""
        counts = [int(c) for c in counts]
        if not counts:
            raise ValueError("no window to pack")
        if len(counts) > self.max_windows:
            raise ValueError(f"{len(counts)} windows exceed {self.max_windows} per micro-step")
        L = self.bucket(counts) if row_len is None else int(row_len)
        if max(counts) > L:
            raise ValueError(f"a window of {max(counts)} tokens exceeds the row of {L}")
        order = sorted(range(len(counts)), key=lambda i: (-counts[i], i))
        free, place = [], [None] * len(counts)
        for i in order:
            c = counts[i]
            r = next((k for k, f in enumerate(free) if f >= c), None)
            if r is None:
                free.append(L)
                r = len(free) - 1
            place[i] = (r, L - free[r])
            free[r] -= c
        return Packing(L, place, len(free), sum(counts))

    def __call__(self, windows):
        """Packing of windows.Window objects by their token counts (layouts.window_tokens)."""
        return self.pack([LY.window_tokens(w) for w in windows])
