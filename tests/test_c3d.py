"""mdfm.data.c3d: the in-tree C3D reader. Synthetic files written by c3d.write in the three processor
formats (Intel, DEC, MIPS) and both data formats (float, int16) round trip; the reader is checked against
the values a canonical MoveAtlas record holds (one non-lockbox able-bodied stroke_gait_vancriekinge trial,
read from the raw zip); CODY paths are refused before any file is opened."""
import os
import zipfile

import numpy as np
import pytest

from mdfm import paths
from mdfm.data import c3d
from mdfm.data import canonical as C

RAW = f"{paths.MOVEATLAS}/datasets/pose/stroke_gait_vancriekinge/raw/138_HealthyPiG.zip"
HAVE_RAW = os.path.exists(RAW) and os.path.exists(C.index_path())


def _synthetic(T=50, P=4, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(0.0, 400.0, size=(T, P, 3)).round(2)
    pts[3, 1] = np.nan                                   # an invalid point
    pts[10:13, 2] = np.nan
    res = np.full((T, P), 0.5)
    analog = rng.normal(0.0, 2.0, size=(T * 10, 3))
    return pts, res, analog


@pytest.mark.parametrize("processor", ["intel", "dec", "mips"])
@pytest.mark.parametrize("scale", [-0.01, 0.1])
def test_round_trip(processor, scale):
    pts, res, analog = _synthetic()
    if scale > 0:                                        # int16 words: coordinates on the scale grid
        pts = np.round(pts / scale) * scale
        analog = np.round(analog / 0.01) * 0.01
    raw = c3d.write(pts, 100.0, ["PELO", "LASI", "RASI", "LTIO"], residual=np.where(np.isnan(pts[..., 0]), 0, 0.5),
                    analog=analog, analog_labels=["Force.Fz1", "Voltage.RTA", "Voltage.LTA"], analog_scale=0.01,
                    analog_offset=3, processor=processor, scale=scale, first_frame=651,
                    extra={"PROCESSING": {"BODYMASS": np.array([63.6], np.float32)},
                           "SUBJECTS": {"NAMES": ["someone"], "USED": np.int16(1)}})
    f = c3d.read(raw)
    assert f.processor == processor
    assert f.labels == ["PELO", "LASI", "RASI", "LTIO"]
    assert f.units == "mm" and f.rate == pytest.approx(100.0) and f.first_frame == 651
    assert f.points.shape == pts.shape and f.n_frames == 50
    bad = np.isnan(pts).any(-1)
    assert np.isnan(f.points[bad]).all() and (f.residual[bad] == -1).all()
    tol = 1e-3 if scale < 0 else 1e-4                   # mm: float32 words, or the float32 POINT:SCALE
    np.testing.assert_allclose(f.points[~bad], pts[~bad], atol=tol, rtol=0)
    assert f.analog.shape == (500, 3) and f.analog_rate == pytest.approx(1000.0)
    np.testing.assert_allclose(f.analog, analog, atol=1e-4)
    assert f.analog_labels == ["Force.Fz1", "Voltage.RTA", "Voltage.LTA"]
    assert f.param("processing", "bodymass")[0] == pytest.approx(63.6, abs=1e-5)
    assert f.param("SUBJECTS", "NAMES") == ["someone"] and f.param("SUBJECTS", "USED") == 1
    np.testing.assert_allclose(f.point("PELO")[~bad[:, 0]], pts[~bad[:, 0], 0], atol=tol, rtol=0)


def test_dec_floats_are_vax_f():
    x = np.array([1.0, -2.5, 1234.5678, 1e-3, 0.0], dtype=np.float32)
    raw = c3d.float_to_dec(x)
    # VAX F: 1.0 is exponent 129 with a zero fraction, sign bit clear: first word 0x4080, second 0x0000
    assert raw[:4] == bytes([0x80, 0x40, 0x00, 0x00])
    np.testing.assert_array_equal(c3d.dec_to_float(raw), x)


def test_parameter_arrays_keep_the_file_order():
    cal = np.arange(12, dtype=np.float32).reshape(3, 4)
    raw = c3d.write(np.zeros((2, 1, 3)), 50.0, ["A"], extra={"FORCE_PLATFORM": {"CORNERS": cal,
                                                                               "USED": np.int16(4)}})
    f = c3d.read(raw)
    np.testing.assert_array_equal(f.param("FORCE_PLATFORM", "CORNERS"), cal)
    assert f.param("FORCE_PLATFORM", "USED") == 4
    assert f.param("POINT", "missing", "x") == "x"


def test_unsigned_analog_words():
    """ANALOG:FORMAT UNSIGNED: integer analog words read as uint16 and an int16 OFFSET of -32768 as 32768,
    so words 0 to 65535 give -32768 to 32767 (words patched into an int16 file after writing)."""
    T = 5
    raw = bytearray(c3d.write(np.zeros((T, 1, 3)), 100.0, ["A"], scale=0.1, analog=np.zeros((T, 1)),
                              analog_scale=1.0, analog_offset=-32768, extra={"ANALOG": {"FORMAT": "UNSIGNED"}}))
    start = (int(np.frombuffer(bytes(raw[16:18]), "<u2")[0]) - 1) * 512
    words = np.array([0, 1000, 32768, 40000, 65535], dtype="<u2")
    for i, w in enumerate(words):                          # frame i: 4 point words, then 1 analog word
        at = start + (i * 5 + 4) * 2
        raw[at:at + 2] = w.tobytes()
    f = c3d.read(bytes(raw))
    np.testing.assert_array_equal(f.analog[:, 0], words.astype(np.float64) - 32768.0)
    assert np.allclose(f.points, 0.0)


def test_truncated_file_raises():
    raw = c3d.write(np.zeros((20, 3, 3)), 100.0, ["A", "B", "C"])
    with pytest.raises(c3d.C3DError):
        c3d.read(raw[:-600])
    with pytest.raises(c3d.C3DError):
        c3d.read(raw[:100])


def test_cody_path_refused_before_open(tmp_path):
    """A path through a CODY dataset folder is refused before the file is opened (it does not exist, so an
    open would raise FileNotFoundError instead); with the evaluation flag the open is attempted."""
    missing = os.path.join(str(tmp_path), "cody_3", "trial.c3d")
    with pytest.raises(C.Refused):
        c3d.read(missing)
    with pytest.raises(C.Refused):
        c3d.read(os.path.join(str(tmp_path), "Cody_1", "x", "trial.c3d"))
    with pytest.raises(FileNotFoundError):
        c3d.read(missing, allow_eval=True)


def test_reads_bytes_path_and_file_object(tmp_path):
    raw = c3d.write(np.ones((3, 2, 3)), 120.0, ["A", "B"])
    p = tmp_path / "x.c3d"
    p.write_bytes(raw)
    with open(p, "rb") as fh:
        for src in (raw, str(p), p, fh):
            assert c3d.read(src).points.shape == (3, 2, 3)


@pytest.mark.skipif(not HAVE_RAW, reason="MoveAtlas raw stroke zip or record index not found")
def test_matches_the_canonical_record():
    """PiG model outputs of a raw C3D trial, mapped as MoveAtlas ingest_stroke_gait.py maps them, equal the
    canonical record's coords x scale (root-centred on PELO), and the invalid frames agree."""
    member = "138_HealthyPiG_10.05/SUBJ01/SUBJ1 (2).c3d"
    with zipfile.ZipFile(RAW) as z:
        f = c3d.read(z.read(member))
    assert f.processor == "intel" and f.rate == pytest.approx(100.0) and f.units == "mm"
    rec = C.load({"source": "stroke_gait_vancriekinge", "modality": "pose",
                  "subject_ns": "stroke_gait:able_bodied:S001", "key": "gait_SUBJ1_2"})
    assert rec["n_frames"] == f.n_frames
    labels = [x.split(":", 1)[-1] for x in f.labels]

    def point(n):
        x = f.points[:, labels.index(n)].copy()
        x[x == 0.0] = np.nan
        return x / 1000.0

    names = {"pelvis": "PELO", "left_hip": "LFEP", "right_knee": "RFEO", "left_ankle": "LTIO", "head": "HEDO"}
    from mdfm.kernels.camera import CANONICAL_15
    y = np.asarray(rec["coords"], np.float64) * rec["scale"]
    for j, n in names.items():
        k = CANONICAL_15.index(j)
        d = point(n) - point("PELO")
        v = rec["valid_mask"][:, k]
        assert np.array_equal(v, np.isfinite(d).all(1))
        assert np.abs(d[v] - y[v, k]).max() < 1e-6
    neck = 0.5 * (point("LSHO") + point("RSHO")) - point("PELO")
    assert np.abs(neck - y[:, CANONICAL_15.index("neck")]).max() < 1e-6
    assert f.param("PROCESSING", "BODYMASS") is not None
