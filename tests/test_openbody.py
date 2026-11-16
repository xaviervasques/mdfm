"""The open body adapter (mdfm/synth/openbody.py): the licence guard, the joint map, the canonical rest and the
posing against the program's forward kinematics."""
import importlib.util
import os

import numpy as np
import pytest
import torch

from mdfm.synth import openbody as OB

ANNY = importlib.util.find_spec("anny") is not None
CACHED = os.path.isdir(os.path.join(OB.CACHE_DIR, "v14"))
needs_model = pytest.mark.skipif(not (ANNY and CACHED), reason="anny or its pinned cache is not available here")


# ---------------------------------------------------------------- licence guard (no model needed)
@pytest.mark.parametrize("rig,topology", [("anny", "smplx"), ("anny", "smpl"), ("anny", "soma"),
                                          ("anny", "anny_from_soma"), ("soma", "anny"), ("makehuman", "anny"),
                                          ("anny", "anny-full")])
def test_loader_refuses_names_outside_the_allow_list(rig, topology):
    with pytest.raises(OB.OpenBodyRefused):
        OB.check_names(rig, topology)
    with pytest.raises(OB.OpenBodyRefused):
        OB.model(rig, topology)


def test_allow_list_is_the_free_topology_and_rig_only():
    assert OB.ALLOWED_TOPOLOGIES == ("anny",) and OB.ALLOWED_RIGS == ("anny",)


def test_cache_audit_refuses_a_nonfree_entry(tmp_path):
    (tmp_path / "v14").mkdir()
    (tmp_path / "v14" / "build_model_data_x.safetensors").write_bytes(b"0")
    assert [p for p, _ in OB.cache_audit(str(tmp_path))] == ["v14/build_model_data_x.safetensors"]
    (tmp_path / "noncommercial").mkdir()
    (tmp_path / "noncommercial" / "anny2smplx.pth").write_bytes(b"0")
    with pytest.raises(OB.OpenBodyRefused):
        OB.cache_audit(str(tmp_path))


def test_the_pinned_cache_holds_no_nonfree_file():
    for rel, _ in OB.cache_audit():
        assert not any(m in rel.lower() for m in OB.NONFREE_MARKS)


def test_no_nonfree_package_is_installed():
    assert OB.nonfree_packages() == []


@pytest.mark.skipif(not ANNY, reason="anny is not installed here")
def test_guard_blocks_the_nonfree_code_paths():
    OB.guard()
    import anny.paths as AP
    assert str(AP.get_anny_cache_path()).replace("\\", "/") == OB.CACHE_DIR
    for fn in (AP.download_noncommercial_data, AP.get_anny2smpl_data_path, AP.get_anny2smplx_data_path):
        with pytest.raises(OB.OpenBodyRefused):
            fn()
    for name in ("anny.models.smpl", "anny.models.retopology", "smplx", "chumpy"):
        with pytest.raises(ImportError):
            __import__(name)


def test_selector_defaults_to_the_open_body_and_gates_the_reference(monkeypatch):
    monkeypatch.delenv(OB.REFERENCE_ENV, raising=False)
    monkeypatch.delenv(OB.BACKEND_ENV, raising=False)
    with pytest.raises(OB.OpenBodyRefused):
        OB.backend(OB.SMPLH_REFERENCE)
    with pytest.raises(ValueError):
        OB.backend("smplx")


# ---------------------------------------------------------------- joint map (no model needed)
def test_joint_map_is_one_to_one_and_mirrored():
    bones = [b for _, b in OB.JOINT_MAP]
    assert len(OB.JOINT_MAP) == OB.N_JOINTS == 52 and len(set(bones)) == 52
    names = dict(OB.JOINT_MAP)
    for n, b in OB.JOINT_MAP:
        if n.startswith("left_"):
            assert names["right_" + n[5:]] == b[:-2] + ".R" and b.endswith(".L")
    assert all(p < j for j, p in enumerate(OB.PARENTS52) if p >= 0)


def test_identity_specs_are_fourteen_adults():
    assert len(OB.IDENTITY_SPECS) == 14 and len({s[0] for s in OB.IDENTITY_SPECS}) == 14
    for name, *_ in OB.IDENTITY_SPECS:
        t = OB.identity_targets(name)
        assert 25 <= t["age_years"] <= 85 and 18.5 <= t["bmi"] < 30 and 1.5 < t["height_m"] < 1.9


def test_age_phenotype_reads_the_makehuman_targets():
    assert OB.age_phenotype(25) == pytest.approx(2 / 3) and OB.age_phenotype(90) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        OB.age_phenotype(17)


# ---------------------------------------------------------------- the model
@pytest.fixture(scope="module")
def ob():
    return OB.OpenBody()


@needs_model
def test_owner_table_covers_every_bone(ob):
    assert ob.owner.shape[0] == ob.m.bone_count == 104 and int(ob.owner.min()) >= 0
    assert ob.m.template_vertices.shape[0] == 13718


@needs_model
def test_canonical_rest_is_a_t_pose_with_palms_down(ob):
    r = ob.rest_joints52()
    J = dict(zip(OB.JOINT_NAMES, r))
    for side, sx in (("left", 1.0), ("right", -1.0)):
        for a, b in (("shoulder", "elbow"), ("elbow", "wrist")):
            d = J[f"{side}_{b}"] - J[f"{side}_{a}"]
            assert torch.allclose(d / d.norm(), torch.tensor([sx, 0.0, 0.0], dtype=OB.F64), atol=1e-9)
        for a, b in (("hip", "knee"), ("knee", "ankle")):
            d = J[f"{side}_{b}"] - J[f"{side}_{a}"]
            assert torch.allclose(d / d.norm(), torch.tensor([0.0, -1.0, 0.0], dtype=OB.F64), atol=1e-9)
        for f in ("index", "middle", "ring", "pinky"):
            d = J[f"{side}_{f}3"] - J[f"{side}_{f}1"]
            assert torch.allclose(d / d.norm(), torch.tensor([sx, 0.0, 0.0], dtype=OB.F64), atol=1e-9)
    # palms face the floor: the canonical palmar normal of each hand points down (ours)
    heads = ob.canonical_heads
    for side in ("left", "right"):
        n = ob.C @ OB.palm_normal(side, heads, ob.m)
        assert float(n[1]) < -0.999
    # the feet point forward and the face looks toward +z
    assert float(J["left_foot"][2] - J["left_ankle"][2]) > 0.1
    v = ob.canonical_vertices()
    assert float(v[:, 2].max()) > 0.15 > -float(v[:, 2].min()) > 0.0


@needs_model
def test_pose_matches_the_program_forward_kinematics(ob):
    from mdfm.body.rotations import axis_angle_to_matrix
    from mdfm.body.smplh import fk
    torch.manual_seed(0)
    R = axis_angle_to_matrix(0.5 * torch.randn(4, 52, 3, dtype=OB.F64))
    tr = 0.2 * torch.randn(4, 3, dtype=OB.F64)
    J, V = ob.pose(R[:, :22], tr, R[:, 22:])
    rest = ob.rest_joints52()
    P, G = fk(R[:, :22], tr, rest[:22])
    pos, rot = list(P.unbind(-2)), list(G.unbind(-3))
    for i in range(22, 52):
        p = OB.PARENTS52[i]
        pos.append(pos[p] + (rot[p] @ (rest[i] - rest[p])[:, None])[..., 0])
        rot.append(rot[p] @ R[:, i])
    assert float((torch.stack(pos, -2) - J).abs().max()) < 1e-12
    assert V.shape == (4, 13718, 3) and bool(torch.isfinite(V).all())


@needs_model
def test_identity_pose_is_the_canonical_rest_and_rigid_motion_moves_it_rigidly(ob):
    eye = torch.eye(3, dtype=OB.F64).expand(1, 22, 3, 3)
    J0, V0 = ob.pose(eye, torch.zeros(1, 3, dtype=OB.F64))
    assert torch.allclose(J0[0], ob.rest_joints52(), atol=1e-12)
    from mdfm.body.rotations import axis_angle_to_matrix
    Rr = axis_angle_to_matrix(torch.tensor([0.0, 0.7, 0.0], dtype=OB.F64))
    R = eye.clone()
    R[:, 0] = Rr
    t = torch.tensor([[0.3, 0.0, -0.2]], dtype=OB.F64)
    J1, V1 = ob.pose(R, t)
    pel = ob.rest_joints52()[0]
    expect = (V0[0] - pel) @ Rr.T + pel + t
    assert float((V1[0] - expect).abs().max()) < 1e-10


@needs_model
def test_identity_solver_hits_height_and_bmi():
    ph = OB.solve_phenotype("female", 48, 1.62, 29.0, muscle=OB.BUILD_MUSCLE["heavier"], rounds=4)
    a = OB.anthropometry(ph)
    assert abs(a["height_m"] - 1.62) < 0.005 and abs(a["bmi"] - 29.0) < 0.2
    assert 0.0 <= ph.height <= 1.0 and 0.0 <= ph.weight <= 1.0 and ph.gender == 1.0


def test_no_vertex_array_is_written_by_the_module():
    src = open(OB.__file__, encoding="utf-8").read()
    for word in ("np.save", "torch.save", "savez", ".tofile(", "save_safetensors"):
        assert word not in src
    assert np.array(OB.TO_OURS).dot(np.array([0.0, 0.0, 1.0])).tolist() == [0.0, 1.0, 0.0]


@needs_model
def test_vertex_sets_follow_their_rules(ob):
    sets = OB.vertex_sets(ob)
    v = ob.canonical_vertices().numpy()
    J = dict(zip(OB.JOINT_NAMES, ob.rest_joints52().numpy()))
    for side, sx in (("left", 1.0), ("right", -1.0)):
        sole = sets[f"sole_{side}"]
        assert len(sole) > 100 and float(v[sole, 1].min()) == float(v[:, 1].min())
        assert np.all(sx * v[sole, 0] > 0)
        assert float(v[sets[f"heel_{side}"], 2].mean()) < float(v[sets[f"toe_{side}"], 2].mean())
        palm = sets[f"palm_{side}"]
        assert len(palm) > 20 and np.all(v[palm, 1] < J[f"{side}_wrist"][1] + 0.01)
        tip = int(sets[f"tip_{side}_middle"][0])
        assert sx * v[tip, 0] > sx * J[f"{side}_middle3"][0]
    assert int(sets["nose"][0]) == int(sets["head"][np.argmax(v[sets["head"], 2])])


@pytest.mark.skipif(not os.path.isfile(__import__("mdfm.paths", fromlist=["x"]).smplh_model("neutral")),
                    reason="the SMPL+H reference model is not on this machine")
def test_reference_matches_the_program_skeleton(monkeypatch):
    from mdfm.synth import base, skeleton, viewer
    monkeypatch.setenv(OB.BACKEND_ENV, OB.SMPLH_REFERENCE)          # the core on the reference, behind its flag
    monkeypatch.setenv(OB.REFERENCE_ENV, "1")
    ref = OB.backend(OB.SMPLH_REFERENCE, allow_reference=True)
    body = base.static_body("sit", 0.5)
    t = viewer.frame_times(body.duration, 30.0)
    R, tr = viewer.pose_at(body, t)
    J, _ = ref.pose(R, tr, vertices=False)
    assert np.abs(J.numpy() - skeleton.joints_at(body, t)).max() < 1e-9


# ---------------------------------------------------------------- wave 1: the core on the open body
@needs_model
def test_the_twist_split_moves_no_joint_and_spreads_the_elbow_turn(ob):
    from mdfm.body.rotations import axis_angle_to_matrix
    rigid = OB.OpenBody(ob.phenotype, ob.m, twist_split=False)
    shares = sorted(ob.twist_shares().values())
    assert len(shares) == 4 and 0.0 < shares[0] < 0.5 < shares[-1] < 1.0
    eye = torch.eye(3, dtype=OB.F64).repeat(2, 22, 1, 1)
    eye[1, 18] = axis_angle_to_matrix(torch.tensor([1.2, 0.0, 0.0], dtype=OB.F64))   # a pure forearm turn
    tr = torch.zeros(2, 3, dtype=OB.F64)
    Js, Vs = ob.pose(eye, tr)
    Jr, Vr = rigid.pose(eye, tr)
    assert float((Js - Jr).abs().max()) < 1e-12                       # the program joints never move
    assert float((Vs[0] - Vr[0]).abs().max()) < 1e-12                 # no turn: the same mesh
    dom = ob.dominant_joint()
    hand = torch.as_tensor(np.flatnonzero(dom >= 22))
    fore = torch.as_tensor(np.flatnonzero(dom == 18))
    assert float((Vs[1, hand] - Vr[1, hand]).abs().max()) < 1e-3      # the hand keeps the full turn
    assert float((Vs[1, fore] - Vr[1, fore]).norm(dim=-1).max()) > 0.01   # the forearm's skin turns less


@needs_model
def test_the_skinning_call_on_a_subset_is_the_full_mesh_there(ob):
    from mdfm.body.rotations import axis_angle_to_matrix
    torch.manual_seed(1)
    R = axis_angle_to_matrix(0.4 * torch.randn(3, 52, 3, dtype=OB.F64))
    tr = 0.1 * torch.randn(3, 3, dtype=OB.F64)
    full = ob.skin(R[:, :22], tr, R[:, 22:])
    ids = np.array([5, 900, 4321, 13717])
    np.testing.assert_allclose(ob.skin(R[:, :22], tr, R[:, 22:], ids=ids), full[:, ids], atol=1e-12)
    assert ob.dominant_joint().shape == (13718,) and int(ob.dominant_joint().max()) <= 51


def test_the_core_selector_defaults_to_the_open_body_and_gates_the_reference(monkeypatch):
    from mdfm.synth import body as B
    monkeypatch.delenv(OB.BACKEND_ENV, raising=False)
    monkeypatch.delenv(OB.REFERENCE_ENV, raising=False)
    assert B.model_name() == OB.OPEN and B.is_open()
    monkeypatch.setenv(OB.BACKEND_ENV, OB.SMPLH_REFERENCE)
    with pytest.raises(OB.OpenBodyRefused):
        B.model_name()
    monkeypatch.setenv(OB.REFERENCE_ENV, "1")
    assert B.model_name() == OB.SMPLH_REFERENCE
    monkeypatch.setenv(OB.BACKEND_ENV, "smplx")
    with pytest.raises(ValueError):
        B.model_name()


@needs_model
def test_the_open_body_refuses_the_smplh_mesh_and_other_shapes(monkeypatch):
    from mdfm.synth import body as B
    from mdfm.synth import viewer
    monkeypatch.delenv(OB.BACKEND_ENV, raising=False)
    with pytest.raises(OB.OpenBodyRefused):
        viewer.mesh_model("neutral")
    with pytest.raises(ValueError):
        B.rest_joints(torch.ones(16, dtype=OB.F64), B.NEUTRAL)
    with pytest.raises(ValueError):
        B.rest_joints(None, 1)
    r = B.rest_joints()
    ob = B.body_model()
    assert torch.equal(r, ob.rest_joints()) and ob.name == OB.OPEN
    assert ob.phenotype == OB.catalogue_phenotype()


@needs_model
def test_the_catalogue_body_meets_its_targets():
    ph = OB.catalogue_phenotype()
    a = OB.anthropometry(ph)
    t = OB.CATALOGUE_TARGETS
    assert abs(a["height_m"] - t["height_m"]) < 1e-3 and abs(a["bmi"] - t["bmi"]) < 0.05
    assert ph.gender == 0.5 and ph.age == pytest.approx(OB.age_phenotype(40.0)) and ph.muscle == t["muscle"]


def test_the_anatomical_table_types_jarm2022_once_and_maps_it_by_the_signs():
    import json
    from mdfm import paths
    with open(os.path.join(paths.TREE, "reports", "openbody", "joint_limits.json"), encoding="utf-8") as f:
        d = json.load(f)
    J = d["jarm2022"]
    lo, hi = np.asarray(d["body"]["lo_deg"]), np.asarray(d["body"]["hi_deg"])
    n = d["body"]["joints"].index
    assert lo[n("left_hip"), 0] == -J["hip"]["flexion"] and hi[n("left_hip"), 0] == J["hip"]["extension"]
    assert hi[n("left_hip"), 2] == J["hip"]["abduction"] and lo[n("right_hip"), 2] == -J["hip"]["abduction"]
    assert lo[n("left_ankle"), 2] == -J["foot_and_ankle"]["inversion"]          # limits_ext's sign
    assert lo[n("left_elbow"), 1] == -J["elbow"]["flexion"] and hi[n("right_elbow"), 1] == J["elbow"]["flexion"]
    assert hi[n("left_shoulder"), 2] == J["shoulder"]["abduction"] - 90.0       # from the hanging arm to the T pose
    assert lo[n("left_wrist"), 2] == -J["wrist"]["flexion"] and hi[n("right_wrist"), 2] == J["wrist"]["flexion"]
    sag = d["shares"]["thoracolumbar_sagittal"]
    assert sum(sag.values()) == pytest.approx(1.0)
    assert sum(hi[n(j), 0] for j in sag) == pytest.approx(J["thoracolumbar"]["flexion"])
    assert hi[n("neck"), 1] + hi[n("head"), 1] == pytest.approx(J["cervical"]["rotation_left"])
    assert (lo <= hi).all() and (lo[1:] < 0).sum() + (hi[1:] > 0).sum() >= 120
    for r in d["rows"]:
        assert r["source"] in ("jarm2022", "jarm2022+design", "module")
    assert d["jarm2022_read"]["sha256"] and "priors.json" in d["against_amass"]["table"]
    for name, row in d["hands"]["joints"].items():
        assert row["flex_lo_deg"] < 0 < row["flex_hi_deg"], name
