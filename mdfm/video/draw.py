"""The pose drawing of Wan2.2-Animate (draw_aapose_by_meta_new of wan/modules/animate/preprocess/
human_visualization.py), re-implemented in numpy: the mdfm environment has no OpenCV and must not gain one.

What the original does (read on 2026-10-05, github.com/Wan-Video/Wan2.2 branch main; the V0 fetch output, [F3]
and [R1] of reports/weekend_2026-10-01/tasks/V0.md, confirms the stick widths, the ellipse call and the circle
call): on a black canvas of the frame's size, with threshold 0.5, stickwidth 'v2' and the hands drawn,
1. each of the 19 limbs of LIMB_SEQ whose two points both score at least the threshold is a filled ellipse
   (cv2.ellipse2Poly then cv2.fillConvexPoly) centred at the truncated midpoint, half-axes (int(length / 2),
   stickwidth), angle int(degrees(atan2(dy, dx))), colour 0.6 times the limb colour;
2. each of the 20 body points above the threshold is a filled circle (cv2.circle, thickness -1) of radius
   stickwidth at the truncated point, in the point's colour;
3. each hand (left then right): each of the 20 edges between two points above the threshold, both with
   truncated coordinates above 0.01, is a cv2.line of thickness max(stickwidth_v2 // 2, 1) coloured by
   matplotlib hsv_to_rgb([edge / 20, 1, 1]) * 255; then each point above the threshold with truncated
   coordinates above 0.01 is a filled circle of that radius in (0, 0, 255).
Coordinates are truncated toward zero (Python int), so a motion below one pixel is lost or becomes a one-pixel
step, and limb angles are truncated to whole degrees (proposal section 3.1). Colours are written in the canvas's
channel order, RGB here as in the preprocessing (frames read as RGB).

The OpenCV primitives are ported from OpenCV's drawing.cpp (4.x): ellipse2Poly from its whole-degree SinTable
(float32 of the sines rounded to 7 decimals, as the table is written; points rounded half to even, consecutive
duplicates removed), fillConvexPoly as its 8-connected outline plus the fixed-point (16 fractional bits) edge
walk of FillConvexPoly, circle as the integer midpoint algorithm of Circle() for a filled circle drawn with
LINE_8 (a plus at radius 1, a diamond at radius 2), a one-pixel line as the 8-connected LineIterator from the left
end after clipLine to the image. These four, and frames composed of them, are pixel-exact against cv2 4.9 and
4.11 (tests/test_video_driver.py compares when cv2 is importable and is skipped otherwise; the mdfm environment has
no cv2). A line thicker than one pixel
is a filled capsule, not OpenCV's ThickLine: it is drawn only for the hands of frames whose shorter side is at
least 1000 px (stick_widths), never at the default 1280 x 720 or at 832 x 480. The backend name goes into every
clip's provenance (BACKEND).
"""
import math

import numpy as np

BACKEND = "mdfm.video.draw numpy port of OpenCV drawing, v2"
THRESHOLD = 0.5             # draw_aapose_by_meta_new default
LIMB_SHADE = 0.6            # limb colour factor of the ellipses
HAND_EPS = 0.01
# draw_aapose_new limbSeq (1-based indices into the 20 body points) and colors (RGB), in drawing order
LIMB_SEQ = ((2, 3), (2, 6), (3, 4), (4, 5), (6, 7), (7, 8), (2, 9), (9, 10), (10, 11), (2, 12), (12, 13),
            (13, 14), (2, 1), (1, 15), (15, 17), (1, 16), (16, 18), (14, 19), (11, 20))
BODY_COLORS = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
               (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
               (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
               (200, 200, 0), (100, 100, 0))
HAND_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
              (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20))
HAND_POINT_COLOR = (0, 0, 255)


def stick_widths(width, height):
    """(body, hand) stick widths of stickwidth 'v2': max(int(min(H, W) / 200) - 1, 1) and max(body // 2, 1);
    2 and 1 at 1280 x 720."""
    base = int(min(int(height), int(width)) / 200)
    body = max(base - 1, 1)
    return body, max(body // 2, 1)


def _hand_colors():
    from matplotlib.colors import hsv_to_rgb
    n = len(HAND_EDGES)
    # cv2 converts a float colour by saturate_cast (rounding half to even, as numpy rint)
    return tuple(tuple(int(c) for c in np.clip(np.rint(hsv_to_rgb([ie / float(n), 1.0, 1.0]) * 255), 0, 255))
                 for ie in range(n))


HAND_COLORS = _hand_colors()

# OpenCV's SinTable: float literals of the sines of whole degrees 0 to 450, written with 7 decimals
_SIN = np.round(np.sin(np.deg2rad(np.arange(451, dtype=np.float64))), 7).astype(np.float32)
XY_SHIFT = 16               # fixed-point fraction bits of FillConvexPoly
XY_ONE = 1 << XY_SHIFT


def ellipse2poly(center, axes, angle, arc_start=0, arc_end=360, delta=1):
    """cv2.ellipse2Poly with integer arguments -> (P, 2) int64 points."""
    cx, cy = int(center[0]), int(center[1])
    aw, ah = int(axes[0]), int(axes[1])
    angle = int(angle)
    while angle < 0:
        angle += 360
    while angle > 360:
        angle -= 360
    if arc_start > arc_end:
        arc_start, arc_end = arc_end, arc_start
    while arc_start < 0:
        arc_start, arc_end = arc_start + 360, arc_end + 360
    while arc_end > 360:
        arc_start, arc_end = arc_start - 360, arc_end - 360
    if arc_end - arc_start > 360:
        arc_start, arc_end = 0, 360
    alpha, beta = float(_SIN[450 - angle]), float(_SIN[angle])          # cos, sin of the angle
    pts = []
    for i in range(arc_start, arc_end + delta, delta):
        a = min(i, arc_end)
        if a < 0:
            a += 360
        x = aw * float(_SIN[450 - a])
        y = ah * float(_SIN[a])
        pts.append((cx + x * alpha - y * beta, cy + x * beta + y * alpha))
    if len(pts) == 1:
        pts = [(cx, cy), (cx, cy)]
    out, prev = [], None
    for x, y in pts:
        p = (int(np.rint(x)), int(np.rint(y)))
        if p != prev:
            out.append(p)
            prev = p
    if len(out) == 1:
        out = [(cx, cy), (cx, cy)]
    return np.asarray(out, dtype=np.int64)


def _put(img, xs, ys, color):
    xs, ys = np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.int64)
    ok = (xs >= 0) & (xs < img.shape[1]) & (ys >= 0) & (ys < img.shape[0])
    img[ys[ok], xs[ok]] = color


def clip_line(width, height, p0, p1):
    """OpenCV clipLine (int64 version): the segment p0-p1 clipped to the image [0, width) x [0, height) ->
    (inside, p0, p1), the ends moved onto the border with truncated intercepts."""
    x1, y1, x2, y2 = int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1])
    right, bottom = int(width) - 1, int(height) - 1
    if width <= 0 or height <= 0:
        return False, (x1, y1), (x2, y2)

    def code(x, y):
        return (x < 0) + (x > right) * 2 + (y < 0) * 4 + (y > bottom) * 8

    c1, c2 = code(x1, y1), code(x2, y2)
    if (c1 & c2) == 0 and (c1 | c2) != 0:
        if c1 & 12:
            a = 0 if c1 < 8 else bottom
            x1 += int(float(a - y1) * (x2 - x1) / (y2 - y1))
            y1 = a
            c1 = (x1 < 0) + (x1 > right) * 2
        if c2 & 12:
            a = 0 if c2 < 8 else bottom
            x2 += int(float(a - y2) * (x2 - x1) / (y2 - y1))
            y2 = a
            c2 = (x2 < 0) + (x2 > right) * 2
        if (c1 & c2) == 0 and (c1 | c2) != 0:
            if c1:
                a = 0 if c1 == 1 else right
                y1 += int(float(a - x1) * (y2 - y1) / (x2 - x1))
                x1 = a
                c1 = 0
            if c2:
                a = 0 if c2 == 1 else right
                y2 += int(float(a - x2) * (y2 - y1) / (x2 - x1))
                x2 = a
                c2 = 0
    return (c1 | c2) == 0, (x1, y1), (x2, y2)


def line_points(p0, p1, size=None):
    """Pixels of an 8-connected one-pixel line between integer points, both ends included (OpenCV's
    LineIterator with connectivity 8, iterated from the left end) -> (P, 2) int64. size (width, height): the
    image, to which the line is first clipped as LineIterator does when an end lies outside it (no pixel when
    the line misses the image)."""
    (x0, y0), (x1, y1) = (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1]))
    if size is not None:
        w, h = int(size[0]), int(size[1])
        if not (0 <= x0 < w and 0 <= x1 < w and 0 <= y0 < h and 0 <= y1 < h):
            inside, (x0, y0), (x1, y1) = clip_line(w, h, (x0, y0), (x1, y1))
            if not inside:
                return np.zeros((0, 2), dtype=np.int64)
    if x1 < x0:
        x0, y0, x1, y1 = x1, y1, x0, y0
    dx, dy = x1 - x0, y1 - y0
    sy = -1 if dy < 0 else 1
    dy = abs(dy)
    steep = dy > dx
    if steep:
        dx, dy = dy, dx
    err, plus, minus = dx - 2 * dy, 2 * dx, -2 * dy
    x, y = x0, y0
    out = np.empty((dx + 1, 2), dtype=np.int64)
    for i in range(dx + 1):
        out[i] = (x, y)
        diag = err < 0
        err += minus + (plus if diag else 0)
        if steep:
            y += sy
            x += 1 if diag else 0
        else:
            x += 1
            y += sy if diag else 0
    return out


def draw_line(img, p0, p1, color, thickness=1):
    """cv2.line, LINE_8: one-pixel Bresenham for thickness 1 (exact), a filled capsule of that diameter otherwise
    (not OpenCV's ThickLine)."""
    if thickness <= 1:
        pts = line_points(p0, p1, (img.shape[1], img.shape[0]))
        _put(img, pts[:, 0], pts[:, 1], color)
        return img
    r = thickness / 2.0
    (x0, y0), (x1, y1) = (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))
    xa, xb = int(math.floor(min(x0, x1) - r)), int(math.ceil(max(x0, x1) + r))
    ya, yb = int(math.floor(min(y0, y1) - r)), int(math.ceil(max(y0, y1) + r))
    X, Y = np.meshgrid(np.arange(xa, xb + 1), np.arange(ya, yb + 1))
    vx, vy = x1 - x0, y1 - y0
    L2 = vx * vx + vy * vy
    s = np.clip(((X - x0) * vx + (Y - y0) * vy) / L2, 0.0, 1.0) if L2 > 0 else np.zeros(X.shape)
    d2 = (X - x0 - s * vx) ** 2 + (Y - y0 - s * vy) ** 2
    m = d2 <= r * r
    _put(img, X[m], Y[m], color)
    return img


def _cdiv(a, b):
    """C integer division (truncation toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def fill_convex_poly(img, pts, color):
    """cv2.fillConvexPoly, LINE_8, shift 0 (FillConvexPoly of drawing.cpp): the polygon's 8-connected outline
    (each edge from the previous vertex, clipped to the image), then from the top vertex down, a left and a right
    edge walked in 16-bit fixed point (per-row step rounded to nearest, truncated in C), each row filled between
    them, inclusive, the ends rounded half up."""
    v = [(int(x), int(y)) for x, y in np.asarray(pts, dtype=np.int64).reshape(-1, 2)]
    npts = len(v)
    if npts == 0:
        return img
    H, W = img.shape[:2]
    half = XY_ONE >> 1
    imin = 0
    xmin = xmax = v[0][0]
    ymin = ymax = v[0][1]
    p0 = v[-1]
    for i, p in enumerate(v):
        if p[1] < ymin:
            ymin, imin = p[1], i
        ymax, xmax, xmin = max(ymax, p[1]), max(xmax, p[0]), min(xmin, p[0])
        lp = line_points(p0, p, (W, H))
        _put(img, lp[:, 0], lp[:, 1], color)
        p0 = p
    if npts < 3 or xmax < 0 or ymax < 0 or xmin >= W or ymin >= H:
        return img
    ymax = min(ymax, H - 1)
    edge = [{"idx": imin, "di": 1, "x": -XY_ONE, "dx": 0, "ye": ymin},
            {"idx": imin, "di": npts - 1, "x": -XY_ONE, "dx": 0, "ye": ymin}]
    edges = npts
    y = ymin
    while True:
        for e in edge:
            if y >= e["ye"]:
                idx0, di = e["idx"], e["di"]
                idx = idx0 + di
                if idx >= npts:
                    idx -= npts
                while True:                                   # for (; edges-- > 0; )
                    more = edges > 0
                    edges -= 1
                    if not more:
                        break
                    ty = v[idx][1]
                    if ty > y:
                        xs, xe = v[idx0][0] << XY_SHIFT, v[idx][0] << XY_SHIFT
                        e["ye"] = ty
                        e["dx"] = _cdiv((xe - xs) * 2 + (ty - y), 2 * (ty - y))
                        e["x"] = xs
                        e["idx"] = idx
                        break
                    idx0 = idx
                    idx += di
                    if idx >= npts:
                        idx -= npts
        if edges < 0:
            break
        if y >= 0:
            left, right = (1, 0) if edge[0]["x"] > edge[1]["x"] else (0, 1)
            x1 = (edge[left]["x"] + half) >> XY_SHIFT
            x2 = (edge[right]["x"] + half) >> XY_SHIFT
            if x2 >= 0 and x1 < W:
                x1, x2 = max(x1, 0), min(x2, W - 1)
                if x2 >= x1:
                    img[y, x1:x2 + 1] = color
        edge[0]["x"] += edge[0]["dx"]
        edge[1]["x"] += edge[1]["dx"]
        y += 1
        if y > ymax:
            break
    return img


def circle_rows(radius):
    """[(dy, half width)] of OpenCV's filled circle of an integer radius (LINE_8, shift 0): the integer
    midpoint algorithm of drawing.cpp, Circle()."""
    rows = {}
    err, dx, dy, plus, minus = 0, int(radius), 0, 1, 2 * int(radius) - 1
    while dx >= dy:
        for yy, hw in ((dy, dx), (-dy, dx), (dx, dy), (-dx, dy)):
            rows[yy] = max(rows.get(yy, -1), hw)
        dy += 1
        err += plus
        plus += 2
        if err > 0:
            err -= minus
            dx -= 1
            minus -= 2
    return sorted(rows.items())


def filled_circle(img, center, radius, color):
    """cv2.circle with thickness -1 (filled), LINE_8, at an integer centre."""
    cx, cy = int(center[0]), int(center[1])
    for dy, hw in circle_rows(radius):
        y = cy + dy
        if 0 <= y < img.shape[0]:
            l, r = max(cx - hw, 0), min(cx + hw, img.shape[1] - 1)
            if r >= l:
                img[y, l:r + 1] = color
    return img


def _trunc(v):
    return int(v)                     # Python int: toward zero, as the original


def draw_pose(img, body, lhand=None, rhand=None, threshold=THRESHOLD, draw_hand=True, draw_head=True):
    """draw_aapose_new on img (H, W, 3) uint8, in place: body (20, 3) [x, y, score] px, lhand and rhand (21, 3)
    or None. Returns img."""
    H, W = img.shape[:2]
    stick, hand_stick = stick_widths(W, H)
    body = np.array(body, dtype=np.float64)
    if not draw_head:
        body[[0, 14, 15, 16, 17], 2] = 0
    for (i1, i2), color in zip(LIMB_SEQ, BODY_COLORS):
        k1, k2 = body[i1 - 1], body[i2 - 1]
        if k1[2] < threshold or k2[2] < threshold:
            continue
        mx, my = (k1[0] + k2[0]) / 2, (k1[1] + k2[1]) / 2
        length = math.hypot(k1[0] - k2[0], k1[1] - k2[1])
        angle = math.degrees(math.atan2(k1[1] - k2[1], k1[0] - k2[0]))
        poly = ellipse2poly((_trunc(mx), _trunc(my)), (_trunc(length / 2), stick), _trunc(angle))
        fill_convex_poly(img, poly, [int(float(c) * LIMB_SHADE) for c in color])
    for k, color in zip(body, BODY_COLORS):
        if k[2] < threshold:
            continue
        filled_circle(img, (_trunc(k[0]), _trunc(k[1])), stick, color)
    if draw_hand:
        for hand in (lhand, rhand):
            if hand is None:
                continue
            hand = np.asarray(hand, dtype=np.float64)
            for (e1, e2), color in zip(HAND_EDGES, HAND_COLORS):
                k1, k2 = hand[e1], hand[e2]
                if k1[2] < threshold or k2[2] < threshold:
                    continue
                x1, y1, x2, y2 = _trunc(k1[0]), _trunc(k1[1]), _trunc(k2[0]), _trunc(k2[1])
                if x1 > HAND_EPS and y1 > HAND_EPS and x2 > HAND_EPS and y2 > HAND_EPS:
                    draw_line(img, (x1, y1), (x2, y2), color, hand_stick)
            for k in hand:
                if k[2] < threshold:
                    continue
                x, y = _trunc(k[0]), _trunc(k[1])
                if x > HAND_EPS and y > HAND_EPS:
                    filled_circle(img, (x, y), hand_stick, HAND_POINT_COLOR)
    return img


def draw_frame(width, height, uv, score):
    """One pose frame (height, width, 3) uint8 RGB of the drawn set: uv (62, 2) px, score (62,) (1 drawn, 0
    not), on a black canvas."""
    from .points import BODY, LHAND, RHAND
    img = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    kp = np.concatenate([np.asarray(uv, dtype=np.float64), np.asarray(score, dtype=np.float64)[:, None]], 1)
    return draw_pose(img, kp[BODY], kp[LHAND], kp[RHAND])
