"""The MFM model on the open body (plan docs/MODEL_OPEN_PORT_PLAN.md, step M4): the open configuration, the latent
skeleton, the phenotype head and prior, the open priors, the store's training view and the provenance key, and the v1
model reachable only behind the reference flag."""
import os

import numpy as np
import pytest
import torch

from mdfm.body import backend as BK
from mdfm.body import openbody as BO
from mdfm.model import mfm as M
from mdfm.train import bodies as BD
from mdfm.train import config as CF

TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN_CONFIG = os.path.join(TREE, "configs", "stage1_open.yaml")


@pytest.fixture(scope="module")
def open_tiny():
    return M.MFM(M.OPEN_TINY)


def test_open_model_shape_and_inputs(open_tiny):
    m = open_tiny
    assert m.body == "open" and m.n_shape == BO.N_PHENO == 6
    assert isinstance(m.shape_prior, M.PhenotypePrior) and m.shape_prior.k == 6
    assert m.shape_head.k == 6 and m.decoder.film.in_features == 6
    # the pooled row in all three sex rows (decided 2026-11-13)
    assert torch.equal(m.shape_prior.mean[0], m.shape_prior.mean[2])
    assert torch.equal(m.shape_prior.cov[1], m.shape_prior.cov[2])
    # the decoder's mean pose is the open priors' one
    from mdfm.model.decoder import mean_pose_from_report
    mp = mean_pose_from_report(BD.inputs(BD.OPEN)["priors"]).float()
    assert torch.allclose(m.decoder.mean_pose, mp)
    assert isinstance(m.skeleton, BO.LatentSkeletonBank)


def test_open_state_dict_holds_no_body_array(open_tiny):
    sd = open_tiny.state_dict()
    assert not [k for k in sd if k.startswith("skeleton.")]
    assert M.audit_state_dict(sd, arrays={}) == []


def test_latent_skeleton_is_the_basis_on_the_phenotype(open_tiny):
    u = torch.tensor([[0.3, -0.2, 0.1, 0.0, 0.5, -1.0], [1.0, 0.4, -0.3, 0.2, -0.5, 0.0]])
    rest = open_tiny.skeleton.rest(u, torch.tensor([2, 2]))
    want = open_tiny.skeleton.basis.rest(BO.to_phenotype(u))
    assert rest.shape == (2, 22, 3) and torch.allclose(rest, want)
    assert torch.allclose(open_tiny.skeleton.rest(u), rest)      # the gender code is not read


def test_v1_model_needs_the_reference_flag(monkeypatch):
    monkeypatch.delenv(BK.REFERENCE_ENV, raising=False)
    with pytest.raises(BK.ReferenceRefused):
        M.MFM(M.TINY)
    from dataclasses import replace
    m = M.MFM(replace(M.TINY, skeleton=False))     # no body file read: no flag needed
    assert m.body == "smplh" and m.skeleton is None and m.n_shape == 16


def test_open_configuration():
    cfg = CF.check(CF.load(OPEN_CONFIG))
    assert BD.body_of(cfg) == BD.OPEN and CF.model_config(cfg).body == "open"
    assert cfg.losses.plausibility_table == BD.OPEN_INPUTS["plausibility_table"]
    v1 = CF.load(os.path.join(TREE, "configs", "stage1_pilot.yaml"))
    assert BD.body_of(v1) == BD.SMPLH
    # the v1 recipe kept (decision P1): only the name, the status, the body and the plausibility table differ
    assert CF.step_counts(cfg) == CF.step_counts(v1)
    assert cfg.mixture == v1.mixture and cfg.schedule == v1.schedule and cfg.seed == v1.seed
    assert cfg.losses.weights == v1.losses.weights and cfg.losses.sigmas == v1.losses.sigmas


def test_body_problems_catch_a_mixed_configuration():
    from dataclasses import replace
    cfg = CF.load(OPEN_CONFIG)
    bad = replace(cfg, losses=replace(cfg.losses, plausibility_table=CF.PLAUSIBILITY_TABLE,
                                      plausibility_table_sha256=""))
    assert any("an open run reads" in p for p in CF.body_problems(bad))
    v1 = CF.load(os.path.join(TREE, "configs", "stage1_pilot.yaml"))
    mixed = replace(v1, losses=replace(v1.losses, plausibility_table=BD.OPEN_INPUTS["plausibility_table"]))
    assert any("open body's table" in p for p in CF.body_problems(mixed))


def test_open_plausibility_table_reads_as_the_loss_reads_it():
    from mdfm.train import losses as LS
    lo, hi = LS.joint_limits(path=BD.inputs(BD.OPEN)["plausibility_table"], dtype=torch.float64)
    assert lo.shape == (22, 3) and bool((lo <= hi).all())
    assert torch.allclose(lo[0], torch.full((3,), -np.pi, dtype=torch.float64))      # the root never binds


def test_open_inputs_name_no_v1_file():
    for k, v in BD.OPEN_INPUTS.items():
        assert v != BD.V1_INPUTS[k] and "body_store/" not in v and "mount_priors/" not in v.replace("open_mount", "")


@pytest.fixture(scope="module")
def store():
    from mdfm.data import open_body_store as OS
    if not os.path.exists(os.path.join(OS.store_root(), "hashes.json")):
        pytest.skip("the open body store is not on this machine")
    return OS.OpenTrainStore(check=False)


def test_train_view_refuses_takes_over_the_bound(store):
    assert (store.takes["rms_m"] <= store.bound_m).all()
    assert len(store.takes) + store.n_refused_bound == len(store.store.takes)
    assert store.split == "train" and len(store.persons) > 0


def test_train_view_windows(store):
    bw = store.sample(np.random.default_rng(5), 3, 4.0)
    assert bw.rot6d.shape == (3, 400, 22, 6) and bw.betas.shape == (3, 6)
    assert (bw.gender == 2).all()
    ph = BO.to_phenotype(bw.betas.double())
    for i, t in enumerate(bw.take.tolist()):
        assert np.allclose(ph[i].numpy(), store.phenotype[t], atol=1e-5)
    # window frame: the pelvis at x = z = 0 at the centre knot
    rest = torch.as_tensor(store.rest_root[bw.take.numpy()], dtype=torch.float32)
    c = bw.lengths // 2
    pel = bw.transl[torch.arange(3), c] + rest
    assert torch.allclose(pel[:, [0, 2]], torch.zeros(3, 2), atol=1e-4)


def test_provenance_key_of_the_open_store(store):
    from mdfm.train import provenance as PV
    lists = PV.person_lists(store)
    assert list(lists) == [PV.OPEN_BODIES] and lists[PV.OPEN_BODIES] == store.person_list()
    assert PV.report_mismatches(lists) == []
    h = store.hashes()
    assert h["n_takes"] == len(store.takes) and len(h["train_persons_sha256"]) == 64
