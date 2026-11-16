"""The symptom catalogue of the synthetic patient (2026-11-02): every clip of the first exam set (exam.exam_set) and
of the catalogue modules, in one list of ClipSpec.

The catalogue modules (CATALOGUE_MODULES) each expose specs() -> [ClipSpec], with since="catalogue_v1"; a module
whose specs() takes a 'store' argument receives the AMASS body store of the exam set. catalogue_set checks that the
clip ids are unique over the whole set and that every level-0 partner is in the set, as exam_set does for its own
clips. The rules of every clip (SYNTHETIC labels, joint limits, sourced parameters, adult body only) are those of
docs/proposals/SYNTH_PATIENT_DRAFT.md and docs/proposals/SYMPTOM_CATALOGUE_DRAFT.md.

Physiological background (shared change SH1, mdfm/synth/physio.py): with background=True (the default) every spec's
build is wrapped (physio.wrap_specs) so that the clip carries the background layer of its series (breathing, quiet
sway standing, the drifts of the head, of the seated trunk and of the resting fingers), the same layer for a clip and
its level 0 (the series is the root of the level-0 chain); the series of physio.SKIP keep their bodies and say why in
their truth's 'background' block (the walks take it on their still spans since physio-v7). exam.exam_set() itself is not wrapped (the frozen exam page keeps its
bodies); background=False gives the bodies of the catalogue before SH1.
"""
import importlib
import inspect

from .exam import exam_set

CATALOGUE_MODULES = ("tremors", "jerks", "involuntary", "gait", "postures", "parkinsonism", "ataxia")


def module_specs(name, store=None):
    """[ClipSpec] of one catalogue module (its specs(), given the store when it asks for one)."""
    mod = importlib.import_module(f".{name}", __package__)
    fn = getattr(mod, "specs")
    if "store" in inspect.signature(fn).parameters:
        return list(fn(store=store))
    return list(fn())


def available_modules(modules=CATALOGUE_MODULES):
    """The names of `modules` that exist in this package (a family still being built is left out)."""
    import importlib.util
    return tuple(m for m in modules if importlib.util.find_spec(f"{__package__}.{m}") is not None)


def catalogue_set(store=None, modules=CATALOGUE_MODULES, with_exam=True, allow_missing=False, background=True):
    """[ClipSpec]: the exam set (with_exam) then each module of `modules` in order; unique ids, level 0 in the set.
    allow_missing: leave out the modules that do not exist yet instead of failing. background: each build wrapped with
    the series' physiological background (module docstring; physio.wrap_specs)."""
    specs = list(exam_set(store)) if with_exam else []
    for name in (available_modules(modules) if allow_missing else modules):
        specs += module_specs(name, store)
    ids = [s.clip_id for s in specs]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise RuntimeError(f"duplicate clip ids in the catalogue: {dup}")
    known = set(ids)
    for s in specs:
        if s.level0 is not None and s.level0 not in known:
            raise RuntimeError(f"{s.clip_id}: level 0 {s.level0} is not in the catalogue")
    if background:
        from .physio import wrap_specs
        specs = wrap_specs(specs)
    return specs
