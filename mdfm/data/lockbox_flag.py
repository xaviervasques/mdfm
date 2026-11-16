"""The lockbox flag of the readers: allow_lockbox=True is honoured only for a caller in mdfm/gates.

Every reader that can open a lockbox row or set (mdfm/gates/g1.G1Set and the open G1 sets of mdfm/gates/g1_open.py
that inherit it, mdfm/data/open_mocap_raw.load_take, and the v1 AMASS readers) passes the frame it was called from
to gates_override, which grants the flag when the calling module is mdfm.gates or one of its submodules and refuses
it otherwise. The rule was amass_index.gates_override; it lives here since 2026-11-15 (plan phase E step 2,
docs/OPEN_BODY_PORT_DESIGN.md section 22) so that the shipped readers import no AMASS module: amass_index re-exports
this function, the same object, until plan decision P10 removes that module.
"""
from mdfm.data.canonical import Refused


def gates_override(allow_lockbox, caller_frame):
    """allow_lockbox as granted: False when not asked; True when the calling module (the frame the
    public reader received its call from) is mdfm.gates or a submodule; Refused otherwise."""
    if not allow_lockbox:
        return False
    name = str(caller_frame.f_globals.get("__name__", ""))
    if name == "mdfm.gates" or name.startswith("mdfm.gates."):
        return True
    raise Refused(f"allow_lockbox=True is honoured for mdfm/gates code only (caller {name!r})")
