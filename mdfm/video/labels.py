"""The SYNTHETIC label rules of the proposal's section 6, and where clips may be written.

Every file of a clip carries the word SYNTHETIC in its file name (synthetic_name, check_name) and in its
metadata: a PNG text chunk (png_info), the container metadata of an MP4 (ffmpeg_metadata), a field of every
JSON and NPZ written. A provenance record goes with every clip (mdfm/video/render.py).

The visible mark burned into every frame (burn_mark) applies to generated clips, the generator's output: a
driver pose video is the generator's input, and a mark drawn into it would be fed to the generator and could be
reproduced in the generated frames, so driver videos carry the word in their names and metadata only. The mark
sits in a corner of the band that the full-body framing keeps free of drawn points (mdfm.factory.synth.
keep_in_view keeps every point inside the central 0.9 of the image, so a band of 5 percent of the width and
height stays free); its box is returned, so that an analysis can treat keypoints inside it as missing.

Clips are written only under VIDEO_ROOT (<store>/video); check_out_dir refuses this tree, MoveAtlas and the
gates-only store in any case.
"""
import os

from .. import paths

LABEL = "SYNTHETIC"
VIDEO_ROOT = "<store>/video"
NOTICE = ("SYNTHETIC: rendered from a body model by the MD_FM video round trip; not a recording of any person; "
          "internal research use only (proposal VIDEO_SYNTH_ROUNDTRIP_DRAFT section 6)")


def synthetic_name(stem, ext=""):
    """File name with the label first: SYNTHETIC_<stem><ext> (unchanged when stem already starts with it)."""
    stem = str(stem)
    return (stem if stem.startswith(LABEL + "_") else f"{LABEL}_{stem}") + ext


def check_name(path):
    """path unchanged when its base name holds SYNTHETIC; ValueError otherwise."""
    if LABEL not in os.path.basename(os.fspath(path)):
        raise ValueError(f"{path}: a clip file must carry {LABEL} in its name")
    return path


def check_out_dir(path, allowed_root=VIDEO_ROOT):
    """path unchanged when it lies inside allowed_root (default <store>/video) and outside this tree, the
    MoveAtlas tree and the gates-only store; PermissionError otherwise."""
    paths.assert_not_moveatlas(path)
    paths.assert_not_gates_store(path)
    if paths._inside(path, paths.TREE):
        raise PermissionError(f"clips are never written in the MD_FM tree: {path}")
    if not paths._inside(path, allowed_root):
        raise PermissionError(f"clips are written only under {allowed_root}: {path}")
    return path


def png_info(extra=None):
    """PIL PngInfo with the label: text chunks SYNTHETIC = 1, Description = NOTICE, and extra {key: text}."""
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()
    info.add_text(LABEL, "1")
    info.add_text("Description", NOTICE)
    for k, v in (extra or {}).items():
        info.add_text(str(k), str(v))
    return info


def ffmpeg_metadata(title):
    """ffmpeg arguments that write the label into the container metadata (title and comment)."""
    t = title if LABEL in title else f"{LABEL} {title}"
    return ["-metadata", f"title={t}", "-metadata", f"comment={NOTICE}"]


def mark_box(width, height, band=0.05, corner="top_left"):
    """(x0, y0, x1, y1) px of the burned-in mark: inside the outer band (share of width and height) at a corner,
    text height 70 percent of the band, width up to 4.5 times the band height."""
    bh = max(int(band * height), 8)
    th = max(int(0.7 * bh), 6)
    tw = min(int(4.5 * th), width // 4)
    pad = max((bh - th) // 2, 1)
    x0 = pad if corner.endswith("left") else width - pad - tw
    y0 = pad if corner.startswith("top") else height - pad - th
    return x0, y0, x0 + tw, y0 + th


def burn_mark(frame, band=0.05, corner="top_left", text=LABEL):
    """A copy of frame (H, W, 3) uint8 with the word drawn in white on black inside mark_box; returns (frame,
    box). For generated clips (module docstring)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = mark_box(w, h, band, corner)
    img = Image.fromarray(np.ascontiguousarray(frame))
    d = ImageDraw.Draw(img)
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(0, 0, 0))
    size = max(y1 - y0 - 2, 6)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:                     # Pillow before 10.1: a fixed bitmap font
        font = ImageFont.load_default()
    d.text((x0 + 1, y0), text, fill=(255, 255, 255), font=font)
    out = np.asarray(img).copy()
    # the box bounds every change: text overflowing it is cut back to the input
    keep = np.ones(out.shape[:2], bool)
    keep[y0:y1, x0:x1] = False
    out[keep] = frame[keep]
    return out, (x0, y0, x1, y1)
