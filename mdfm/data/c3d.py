"""Minimal reader of the C3D format, written from its public specification (c3d.org, "The C3D File
Format User Guide", Motion Lab Systems); numpy only, read-only.

Layout: 512-byte blocks. Block 1 is the header: byte 0 the block number (1-based) of the parameter
section, then 16-bit words: 2 the number of 3D points, 3 the analog words per 3D frame, 4 and 5 the
first and last frame, 7 and 8 the scale factor (float), 9 the first block of the data section, 10 the
analog samples per 3D frame, 11 and 12 the frame rate (float). The parameter section starts with 4
bytes (byte 3 the processor type: 84 Intel, 85 DEC, 86 MIPS), then groups and parameters, each: name
length (signed byte, negative when locked, 0 ends the section), group id (signed byte, negative for a
group definition), the name, a signed 16-bit offset to the next entry counted from the offset itself,
then for a parameter its type (-1 char, 1 byte, 2 int16, 4 float), its dimension count and dimensions
(unsigned bytes, first dimension fastest) and data, and for both a description. The data section holds,
per frame, 4 words per point (x, y, z and the residual word) then the analog words (channel fastest),
as int16 when POINT:SCALE is positive (coordinates = word x POINT:SCALE) and as float32 when it is
negative (coordinates as stored).

Byte order: Intel little-endian, MIPS big-endian, DEC little-endian integers with VAX F floats
(16-bit words swapped, value / 4). Point validity: a negative residual word marks a point invalid; its
coordinates read NaN and its residual -1. The residual is the low byte of the word times |POINT:SCALE|.
Analog values are (word - ANALOG:OFFSET) x ANALOG:GEN_SCALE x ANALOG:SCALE per channel; ANALOG:FORMAT
UNSIGNED reads integer words as uint16.

read() takes a path, bytes or a binary file object. A path with a component that names a CODY dataset
(a component starting with 'cody') is refused unless allow_eval=True: CODY is evaluation-only.
"""
import io
import os
import re
from dataclasses import dataclass, field

import numpy as np

BLOCK = 512
KEY = 0x50
PROCESSORS = {84: "intel", 85: "dec", 86: "mips"}


class C3DError(ValueError):
    """A file that does not follow the C3D layout."""


@dataclass
class C3D:
    """One C3D file. points (T, P, 3) float64 in POINT:UNITS, NaN where invalid; residual (T, P) float64,
    -1 where invalid; labels (P,); rate Hz; first_frame (the header's, 1-based); analog (T x n, A) float64
    scaled, n = analog samples per frame; params {GROUP: {NAME: value}} with upper-case keys (char arrays
    as lists of stripped strings, a single string for a 1-D char parameter, numeric scalars as Python
    numbers, arrays as numpy in the file's dimension order); processor 'intel', 'dec' or 'mips'."""
    points: np.ndarray
    residual: np.ndarray
    labels: list
    units: str
    rate: float
    first_frame: int
    analog: np.ndarray
    analog_labels: list
    analog_units: list
    analog_rate: float
    params: dict = field(default_factory=dict)
    processor: str = "intel"
    header: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return int(self.points.shape[0])

    def param(self, group, name, default=None):
        """Value of GROUP:NAME (case-insensitive), default when absent."""
        return self.params.get(str(group).upper(), {}).get(str(name).upper(), default)

    def point(self, label):
        """(T, 3) of the point with this label (exact, after stripping); KeyError when absent, ValueError
        when two points carry it."""
        hits = [i for i, x in enumerate(self.labels) if x == label]
        if not hits:
            raise KeyError(label)
        if len(hits) > 1:
            raise ValueError(f"label {label!r} on {len(hits)} points")
        return self.points[:, hits[0]]


# ---------------------------------------------------------------- numbers
def _dtypes(processor):
    if processor == "mips":
        return np.dtype(">i2"), np.dtype(">u2"), np.dtype(">f4")
    return np.dtype("<i2"), np.dtype("<u2"), np.dtype("<f4")


def dec_to_float(raw):
    """float32 array of VAX F floats given as bytes (4 per value, as stored in a DEC C3D file)."""
    w = np.frombuffer(raw, dtype="<u2").reshape(-1, 2)[:, ::-1].copy()
    return (w.view("<f4").ravel() / 4.0).astype(np.float32)


def float_to_dec(x):
    """Bytes of float32 values as VAX F floats (the inverse of dec_to_float, for tests and writers)."""
    w = (np.asarray(x, dtype="<f4").ravel() * 4.0).astype("<f4").view("<u2").reshape(-1, 2)[:, ::-1].copy()
    return w.tobytes()


def _floats(raw, processor):
    if processor == "dec":
        return dec_to_float(raw)
    return np.frombuffer(raw, dtype=_dtypes(processor)[2]).astype(np.float32)


def _ints(raw, processor, unsigned=False):
    i16, u16, _ = _dtypes(processor)
    return np.frombuffer(raw, dtype=u16 if unsigned else i16)


# ---------------------------------------------------------------- parameters
def _s8(b):
    """Signed value of one byte."""
    return b - 256 if b > 127 else b


def _char_value(raw, dims):
    text = raw.decode("latin-1")
    if len(dims) <= 1:
        return text.rstrip(" \x00")
    n = dims[0]
    count = int(np.prod(dims[1:]))
    return [text[i * n:(i + 1) * n].rstrip(" \x00") for i in range(count)]


def _value(kind, dims, raw, processor):
    if kind == -1:
        return _char_value(raw, dims)
    if kind == 1:
        arr = np.frombuffer(raw, dtype=np.int8)
    elif kind == 2:
        arr = _ints(raw, processor)
    elif kind == 4:
        arr = _floats(raw, processor)
    else:
        raise C3DError(f"parameter type {kind}")
    if not dims:
        return arr[0].item()
    return arr.reshape(dims, order="F")


def parse_parameters(buf, start):
    """(params {GROUP: {NAME: value}}, processor, n_blocks) of the parameter section at byte `start`."""
    if len(buf) < start + 4:
        raise C3DError("file ends before the parameter section")
    n_blocks = buf[start + 2]
    ptype = buf[start + 3]
    if ptype not in PROCESSORS:
        raise C3DError(f"processor type {ptype}")
    processor = PROCESSORS[ptype]
    i16 = _dtypes(processor)[0]
    groups, pending = {}, []
    p = start + 4
    end = len(buf)
    while p + 2 <= end:
        nchar = _s8(buf[p])
        gid = _s8(buf[p + 1])
        if nchar == 0 or gid == 0:
            break
        n = abs(nchar)
        name = buf[p + 2:p + 2 + n].decode("latin-1").strip().upper()
        q = p + 2 + n
        off = int(np.frombuffer(buf[q:q + 2], dtype=i16)[0])
        body = q + 2
        if gid < 0:
            groups[-gid] = name
        else:
            kind = _s8(buf[body])
            nd = buf[body + 1]
            dims = tuple(int(d) for d in buf[body + 2:body + 2 + nd])
            count = int(np.prod(dims)) if dims else 1
            d0 = body + 2 + nd
            raw = bytes(buf[d0:d0 + count * abs(kind)])
            if len(raw) != count * abs(kind):
                raise C3DError(f"parameter {name} runs past the end of the file")
            pending.append((gid, name, _value(kind, dims, raw, processor)))
        if off == 0:
            break
        p = q + off
    params = {}
    for gid, name, value in pending:
        params.setdefault(groups.get(gid, f"GROUP{gid}"), {})[name] = value
    for g in groups.values():
        params.setdefault(g, {})
    return params, processor, n_blocks


# ---------------------------------------------------------------- reading
def _refuse_cody(path, allow_eval):
    parts = re.split(r"[\\/]+", os.fspath(path))
    if not allow_eval and any(p.strip().lower().startswith("cody") for p in parts):
        from mdfm.data.canonical import Refused
        raise Refused(f"CODY is evaluation-only: {path} (pass allow_eval=True in an evaluation stage)")


def _bytes(source, allow_eval):
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    if isinstance(source, (str, os.PathLike)):
        _refuse_cody(source, allow_eval)
        with open(source, "rb") as f:
            return f.read()
    if isinstance(source, io.IOBase) or hasattr(source, "read"):
        return source.read()
    raise TypeError(f"a path, bytes or a binary file object, got {type(source).__name__}")


def _labels(group, key, used):
    out = []
    for name in [key] + [f"{key}{k}" for k in range(2, 100)]:
        v = group.get(name)
        if v is None:
            break
        out += [v] if isinstance(v, str) else list(v)
    return [str(x).strip() for x in out[:used]] + [""] * max(0, used - len(out))


def _num(v, default):
    if v is None:
        return default
    a = np.asarray(v).ravel()
    return a[0].item() if a.size else default


def _uint(v):
    return int(v) + 65536 if int(v) < 0 else int(v)


def read(source, allow_eval=False):
    """C3D of a file (module docstring). source: path, bytes or binary file object."""
    buf = _bytes(source, allow_eval)
    if len(buf) < BLOCK:
        raise C3DError(f"{len(buf)} bytes, shorter than the header block")
    pblock = buf[0]
    if pblock < 1:
        raise C3DError(f"parameter block {pblock}")
    params, processor, _ = parse_parameters(buf, (pblock - 1) * BLOCK)
    i16, u16, _ = _dtypes(processor)
    words = np.frombuffer(buf[:BLOCK], dtype=u16)
    head_scale = float(_floats(buf[12:16], processor)[0])
    head_rate = float(_floats(buf[20:24], processor)[0])
    header = {"parameter_block": int(pblock), "n_points": int(words[1]), "analog_words": int(words[2]),
              "first_frame": int(words[3]), "last_frame": int(words[4]), "scale": head_scale,
              "data_block": int(words[8]), "analog_per_frame": int(words[9]), "rate": head_rate}
    P = params.get("POINT", {})
    A = params.get("ANALOG", {})
    n_points = _uint(_num(P.get("USED"), header["n_points"]))
    scale = float(_num(P.get("SCALE"), head_scale))
    rate = float(_num(P.get("RATE"), head_rate))
    n_frames = header["last_frame"] - header["first_frame"] + 1
    if P.get("FRAMES") is not None and header["last_frame"] == 65535:
        n_frames = _uint(_num(P.get("FRAMES"), n_frames))
    n_analog = _uint(_num(A.get("USED"), 0)) if A else 0
    per_frame = header["analog_per_frame"]
    if n_analog and per_frame == 0 and rate > 0:
        per_frame = int(round(float(_num(A.get("RATE"), 0.0)) / rate))
    analog_words = n_analog * per_frame
    if header["analog_words"] and header["analog_words"] != analog_words:
        raise C3DError(f"header analog words {header['analog_words']} for {n_analog} channels x {per_frame}")
    data_block = _uint(_num(P.get("DATA_START"), header["data_block"]))
    is_float = scale < 0
    wsize = 4 if is_float else 2
    per = 4 * n_points + analog_words
    start = (data_block - 1) * BLOCK
    need = start + n_frames * per * wsize
    if n_frames < 0 or need > len(buf):
        raise C3DError(f"data section of {n_frames} frames x {per} words needs {need} bytes, file has {len(buf)}")
    raw = buf[start:need]
    if is_float:
        data = _floats(raw, processor).astype(np.float64).reshape(n_frames, per)
    else:
        unsigned = str(A.get("FORMAT", "")).strip().upper() == "UNSIGNED"
        ints = _ints(raw, processor).reshape(n_frames, per)
        data = ints.astype(np.float64)
        if unsigned and analog_words:
            data[:, 4 * n_points:] = ints[:, 4 * n_points:].view(u16).astype(np.float64)
    pts = data[:, :4 * n_points].reshape(n_frames, n_points, 4)
    word = pts[..., 3]
    invalid = word < 0
    res = (np.floor(np.abs(word)).astype(np.int64) & 0xFF) * abs(scale)
    xyz = pts[..., :3] if is_float else pts[..., :3] * scale
    xyz = np.where(invalid[..., None], np.nan, xyz)
    residual = np.where(invalid, -1.0, res)
    analog = np.zeros((n_frames * per_frame, n_analog))
    if analog_words:
        a = data[:, 4 * n_points:].reshape(n_frames * per_frame, n_analog)
        off = np.resize(np.asarray(A.get("OFFSET", 0), dtype=np.float64).ravel(), n_analog) if n_analog else 0
        if not is_float and str(A.get("FORMAT", "")).strip().upper() == "UNSIGNED" and A.get("OFFSET") is not None:
            off = np.where(off < 0, off + 65536, off)
        sc = np.resize(np.asarray(A.get("SCALE", 1.0), dtype=np.float64).ravel(), n_analog)
        gen = float(_num(A.get("GEN_SCALE"), 1.0))
        analog = (a - off[None]) * gen * sc[None]
    units = P.get("UNITS", "")
    return C3D(points=xyz, residual=residual, labels=_labels(P, "LABELS", n_points),
               units=units if isinstance(units, str) else (units[0] if units else ""), rate=rate,
               first_frame=header["first_frame"], analog=analog, analog_labels=_labels(A, "LABELS", n_analog),
               analog_units=_labels(A, "UNITS", n_analog), analog_rate=float(_num(A.get("RATE"), rate * per_frame)),
               params=params, processor=processor, header=header)


# ---------------------------------------------------------------- writing (tests and fixtures)
def _param_entry(gid, name, kind, dims, payload, processor, last=False):
    i16 = _dtypes(processor)[0]
    nm = name.encode("latin-1")
    body = bytes([kind & 0xFF, len(dims)]) + bytes(dims) + payload + bytes([0])
    off = 0 if last else 2 + len(body)
    return bytes([len(nm), gid & 0xFF]) + nm + np.array([off], dtype=i16).tobytes() + body


def _group_entry(gid, name, processor):
    i16 = _dtypes(processor)[0]
    nm = name.encode("latin-1")
    return bytes([len(nm), (-gid) & 0xFF]) + nm + np.array([3], dtype=i16).tobytes() + bytes([0])


def _encode(value, processor):
    """(kind, dims, payload) of a parameter value: str, list of str, int, float or a numpy array."""
    i16 = _dtypes(processor)[0]
    if isinstance(value, str):
        b = value.encode("latin-1")
        return -1, (len(b),), b
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        n = max(1, max(len(v) for v in value))
        return -1, (n, len(value)), b"".join(v.encode("latin-1").ljust(n, b" ") for v in value)
    arr = np.asarray(value)
    dims = () if arr.ndim == 0 else tuple(int(d) for d in arr.shape)
    flat = arr.ravel(order="F")
    if np.issubdtype(arr.dtype, np.integer):
        return 2, dims, flat.astype(i16).tobytes()
    data = float_to_dec(flat) if processor == "dec" else flat.astype(_dtypes(processor)[2]).tobytes()
    return 4, dims, data


def write(points, rate, labels, *, units="mm", scale=-1.0, residual=None, analog=None, analog_labels=None,
          analog_rate=None, analog_scale=1.0, analog_offset=0, processor="intel", extra=None, first_frame=1):
    """Bytes of a C3D file (the layout of the module docstring), for tests and fixtures: points (T, P, 3)
    with NaN for invalid points, float (scale < 0) or int16 (scale > 0) data; analog (T x n, A) written as
    word = value / (GEN_SCALE 1 x ANALOG:SCALE) + ANALOG:OFFSET; extra {GROUP: {NAME: value}}."""
    if processor not in ("intel", "dec", "mips"):
        raise ValueError(processor)
    pts = np.asarray(points, dtype=np.float64)
    T, Pn, _ = pts.shape
    A = 0 if analog is None else np.asarray(analog).shape[1]
    per_frame = 0 if analog is None else np.asarray(analog).shape[0] // T
    i16, u16, f4 = _dtypes(processor)
    groups = {"POINT": {"USED": np.int16(Pn), "SCALE": np.float32(scale), "RATE": np.float32(rate),
                        "DATA_START": np.int16(0), "FRAMES": np.int16(T), "LABELS": list(labels) or [""],
                        "UNITS": units}}
    if A:
        groups["ANALOG"] = {"USED": np.int16(A), "RATE": np.float32(analog_rate or rate * per_frame),
                            "GEN_SCALE": np.float32(1.0),
                            "SCALE": np.resize(np.asarray(analog_scale, np.float32), A),
                            "OFFSET": np.resize(np.asarray(analog_offset, np.int16), A),
                            "LABELS": list(analog_labels or [f"A{i}" for i in range(A)])}
    for g, d in (extra or {}).items():
        groups.setdefault(g.upper(), {}).update({k.upper(): v for k, v in d.items()})

    def section(data_start):
        groups["POINT"]["DATA_START"] = np.int16(data_start)
        out = b""
        gids = {g: i + 1 for i, g in enumerate(groups)}
        entries = []
        for g, d in groups.items():
            entries.append(_group_entry(gids[g], g, processor))
            for k, v in d.items():
                entries.append((gids[g], k, _encode(v, processor)))
        for j, e in enumerate(entries):
            if isinstance(e, bytes):
                out += e
            else:
                gid, k, (kind, dims, payload) = e
                out += _param_entry(gid, k, kind, dims, payload, processor, last=(j == len(entries) - 1))
        return out

    body = section(0)
    n_pblocks = (4 + len(body)) // BLOCK + 1
    data_start = 2 + n_pblocks
    body = section(data_start)
    ptype = {v: k for k, v in PROCESSORS.items()}[processor]
    psec = (bytes([1, KEY, n_pblocks, ptype]) + body).ljust(n_pblocks * BLOCK, b"\x00")
    head = bytearray(BLOCK)
    head[0], head[1] = 2, KEY
    w = np.zeros(BLOCK // 2, dtype=np.int64)
    w[1], w[2], w[3], w[4], w[8], w[9] = Pn, A * per_frame, first_frame, first_frame + T - 1, data_start, per_frame
    for k in (1, 2, 3, 4, 8, 9):
        head[2 * k:2 * k + 2] = np.array([w[k]], dtype=u16).tobytes()
    head[12:16] = float_to_dec([scale]) if processor == "dec" else np.array([scale], dtype=f4).tobytes()
    head[20:24] = float_to_dec([rate]) if processor == "dec" else np.array([rate], dtype=f4).tobytes()
    res = np.zeros((T, Pn)) if residual is None else np.asarray(residual, dtype=np.float64)
    bad = ~np.isfinite(pts).all(-1)
    frame = np.zeros((T, Pn, 4))
    if scale < 0:
        frame[..., :3] = np.where(bad[..., None], 0.0, pts)
        frame[..., 3] = np.where(bad, -1.0, np.round(res / abs(scale)))
    else:
        frame[..., :3] = np.where(bad[..., None], 0.0, np.round(pts / scale))
        frame[..., 3] = np.where(bad, -1.0, np.round(res / abs(scale)))
    rows = [frame.reshape(T, 4 * Pn)]
    if A:
        sc = np.resize(np.asarray(analog_scale, np.float64), A)
        off = np.resize(np.asarray(analog_offset, np.float64), A)
        a = np.asarray(analog, np.float64) / sc[None] + off[None]
        rows.append((a if scale < 0 else np.round(a)).reshape(T, per_frame * A))
    flat = np.concatenate(rows, axis=1).ravel()
    if scale < 0:
        data = float_to_dec(flat) if processor == "dec" else flat.astype(f4).tobytes()
    else:
        data = flat.astype(i16).tobytes()
    data = data.ljust(-(-len(data) // BLOCK) * BLOCK, b"\x00")
    return bytes(head) + psec + data
