"""The open body store's reader (mdfm/data/open_body_store.py; plan docs/MODEL_OPEN_PORT_PLAN.md, steps M3 and M4), on a
small store written here in the packed layout, and on the built store when it is present."""
import csv
import hashlib
import json
import os

import numpy as np
import pytest

from mdfm.data import open_body_store as S

COLS = ["take_id", "source", "subject_ns", "person_id", "key", *S.PHENOTYPE_KEYS, "fps_native", "n_native", "offset",
        "n_knots", "person_weight", "take_share", "treadmill", "belt_x", "belt_z", "floor_m", "contact_l",
        "contact_r", "rms_m", "up_axis"]


def _write(root):
    d = os.path.join(root, "train", "src")
    os.makedirs(d)
    n = (5, 3)
    r6 = np.tile(np.array([1, 0, 0, 0, 1, 0], np.float32), (sum(n), 22, 1))
    tr = np.zeros((sum(n), 3), np.float32)
    ct = np.ones((sum(n), 2), np.uint8)
    for name, a in (("rot6d.f32", r6), ("transl.f32", tr), ("contact.u1", ct)):
        a.tofile(os.path.join(d, name))
    with open(os.path.join(d, "takes.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLS)
        for k, (off, m, tread) in enumerate(((0, 5, 0), (5, 3, 1))):
            w.writerow([f"src/p/{k}", "src", "src:p", "src:p", str(k), 0.5, 0.75, 0.5, 0.5, 0.5, 0.5, 100.0, m, off,
                        m, 1.0, m / 8, tread, 1.0 if tread else 0.0, 0.0, 0.0, 1.0, 1.0, 0.005, "+y"])
    files = {}
    for name in ("rot6d.f32", "transl.f32", "contact.u1", "takes.csv"):
        with open(os.path.join(d, name), "rb") as f:
            files[f"train/src/{name}"] = hashlib.sha256(f.read()).hexdigest()
    with open(os.path.join(root, "hashes.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"files": files}, f)


def test_the_reader_serves_takes_in_the_belt_frame_and_weights_by_person(tmp_path):
    _write(str(tmp_path))
    st = S.OpenBodyStore(str(tmp_path))
    assert len(st.takes) == 2 and st.persons() == {("src", "src:p"): [0, 1]}
    t1 = st.take(1)
    assert t1["rot6d"].shape == (3, 22, 6) and np.allclose(t1["transl"][:, 0], [0.0, -0.01, -0.02])
    assert np.isclose(sum(st.knot_weights(i) * st.takes[i]["n_knots"] for i in range(2)), 1.0)
    with open(os.path.join(str(tmp_path), "train", "src", "transl.f32"), "r+b") as f:
        f.write(b"\x01")
    with pytest.raises(ValueError):
        S.OpenBodyStore(str(tmp_path))


@pytest.mark.skipif(not os.path.exists(os.path.join(S.store_root(), "hashes.json")), reason="open body store absent")
def test_the_built_store_reads():
    st = S.OpenBodyStore(check=False)
    t = st.take(0)
    assert t["rot6d"].shape[1:] == (22, 6) and st.rest(0).shape == (22, 3)
