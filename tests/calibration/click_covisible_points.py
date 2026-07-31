"""
Fit the floor homography between the two CO-VISIBLE cameras, by clicking.

    # on the GPU box (no display needed) -- pull one frame per camera
    python tests/calibration/click_covisible_points.py --export

    # on a machine with a screen, after copying the two PNGs down
    python tests/calibration/click_covisible_points.py --click

    # or skip clicking entirely and type the correspondences in
    python tests/calibration/click_covisible_points.py --from-json my_points.json

WHY THIS IS SMALLER THAN IT SOUNDS. ADR-003 asks for WORLD coordinates -- metres
on a shared floor map -- which needs a tape measure, a floor plan or a tile grid
to fix the origin, the axes and the scale. That is real work and it is required
for the mining bands (0.5 m / 3.0 m), for BEV, and for pulling cam_206 / cam_213
into one frame.

None of it is required to answer the FIRST question, which is only ever asked
about the one co-visible pair: are these two tracks standing in the same place?
That is a CAMERA-TO-CAMERA homography

    H : cam_a floor pixels  ->  cam_b floor pixels

fitted from >= 4 floor points visible in BOTH images. No world frame, no metres,
no origin. Residuals come out in cam_b pixels, which the measurement script then
reports against the observed width of a person in the same image -- so the answer
reads as "a fifth of a person" or "three people apart" without anyone inventing a
scale they did not measure.

WHAT TO CLICK. Points ON THE FLOOR -- where an object MEETS the ground, never the
top of it. Good: tile corners, the base of a door frame, the feet of a table or
chair, corners of a rug or a floor marking, where a wall line meets the floor.
Bad: anything at table height, anything on a wall, anything on a person.
A homography is only valid for one plane, and the plane we care about is the one
people stand on.

HOW MANY. Four is the minimum and the worst possible choice: with exactly 4
points the fit is EXACT by construction (8 equations, 8 unknowns), the residual
is zero, and the error becomes unmeasurable. 8-12 is what makes error visible --
this is ADR-003A section 1.1, and it is the reason this tool refuses to report a
quality number below 6 points.

The record it writes carries `image_size` for both cameras. That is not
decoration: a homography is tied to the pixel space it was authored in, and
`source.resize_width` (config.yaml, 0 today) resizes frames BEFORE detection, so
a homography clicked on one resolution silently misreads boxes at another. The
measurement script checks the sizes match and refuses rather than guess.

READ-ONLY on everything except its own output file under calibration/.
"""

import json
import os
import sys
import time

import cv2
import numpy as np

from _common import arg, bootstrap, flag, header, project_root

bootstrap()

KNOWN_FLAGS = {"--export", "--click", "--from-json", "--cam-a", "--cam-b",
               "--frame", "--out", "--view-height", "--ransac-px"}

# The co-visible pair (REMEDIATION_PLAN.md section 2: "Co-visibility: only
# 224<->219. No other pair can see one person at once, even briefly.")
DEFAULT_CAM_A = "cam_219"
DEFAULT_CAM_B = "cam_224"


def strict_flags():
    """Unknown --flag is a hard error, for the reason _common.KNOWN_FLAGS exists:
    a flag silently ignored by an older deployment returns a measurement of the
    wrong thing, which is worse than a crash."""
    unknown = [t for t in sys.argv[1:]
               if t.startswith("--") and t not in KNOWN_FLAGS]
    if unknown:
        raise SystemExit(f"[calib] unknown flag(s) {unknown}.\n"
                         f"        Known: {', '.join(sorted(KNOWN_FLAGS))}")


def require_display():
    """Refuse to open a window when there is nowhere to put it.

    This has to be a PRE-FLIGHT check, not an exception handler. On a headless
    box Qt does not raise anything Python can catch -- it prints
    "could not connect to display" and calls abort(), so the process is gone
    before any `except cv2.error` can run. Checking the environment first is the
    only way to fail with a useful message instead of a core dump.
    """
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    cdir = calib_dir()
    raise SystemExit(
        "[calib] no display on this machine (DISPLAY and WAYLAND_DISPLAY are "
        "both unset).\n"
        "        Clicking needs a screen; the capture box does not have one.\n\n"
        "        The frames are already exported. From your LOCAL machine:\n\n"
        f"          scp <user>@<this-host>:{os.path.join(cdir, 'frame_cam_*.png')} "
        f"calibration/\n"
        "          python tests/calibration/click_covisible_points.py --click\n\n"
        "        then copy calibration/homography_*.json back here and run\n"
        "        measure_covisible_geometry.py.\n\n"
        "        `ssh -X` also works if X11 forwarding is set up, but it is slow\n"
        "        for a 2560x1440 frame and the round trip above is usually less\n"
        "        trouble.")


def calib_dir():
    d = os.path.join(project_root(), "calibration")
    os.makedirs(d, exist_ok=True)
    return d


def frame_png(cam):
    return os.path.join(calib_dir(), f"frame_{cam}.png")


def out_path(cam_a, cam_b):
    return arg("--out", os.path.join(calib_dir(),
                                     f"homography_{cam_a}_to_{cam_b}.json"))


# ------------------------------------------------------------------ frame export

def export_frame(cam, frame_frac=0.25):
    """Pull ONE frame out of ._live_src_<cam>.mp4 and write it as a PNG.

    Any frame will do -- the floor does not move -- so this defaults to a quarter
    of the way in rather than frame 0, which on a live capture is often the first
    frame after connect and can be a partial or dark decode.
    """
    clip = f"._live_src_{cam}.mp4"
    if not os.path.exists(clip):
        raise SystemExit(
            f"[calib] {clip} not found in {project_root()}.\n"
            f"        Clips are kept only when live.reconcile.keep_frames is true "
            f"(it is, by default, since 2026-07-30). Run this on the box that "
            f"captured the run.")
    cap = cv2.VideoCapture(clip)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    want = int(arg("--frame", -1))
    if want < 0:
        want = int(total * frame_frac) if total > 0 else 0
    if total > 0 and want >= total:
        raise SystemExit(f"[calib] {clip} has {total} frames; --frame {want} is "
                         f"past the end.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, want)
    ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        raise SystemExit(f"[calib] could not read frame {want} of {clip}.")
    path = frame_png(cam)
    cv2.imwrite(path, img)
    print(f"  {cam}: frame {want}/{total}  {img.shape[1]}x{img.shape[0]}  -> {path}")
    return path


# ------------------------------------------------------------------ the homography

def apply_h(H, pts):
    """(N,2) points through a 3x3 homography -> (N,2), NaN where the point maps
    to or beyond the horizon.

    The guard is not theoretical. A floor homography sends the plane's vanishing
    line to infinity, and a foot point estimated slightly too high in the image
    (a long coat, a clipped box) can land on the wrong side of it. Without this
    check that becomes a huge finite number instead of a missing value, and a
    missing value is the honest answer.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    out = hom @ np.asarray(H, dtype=np.float64).T
    w = out[:, 2:3]
    bad = np.abs(w[:, 0]) < 1e-9
    w = np.where(bad[:, None], np.nan, w)
    return out[:, :2] / w


def fit_homography(pts_a, pts_b, ransac_px=3.0):
    """-> (H, n_inliers, leave-one-out errors in cam_b pixels).

    Least-squares + RANSAC, per ADR-003A section 1.1: annotation error, not
    mathematics, is the risk here, and RANSAC is what keeps one mis-clicked point
    from bending the whole fit.

    Leave-one-out is the honest error estimate. Reporting the residual of the
    points the fit was computed FROM measures how well the fit memorised them; at
    4 points that number is exactly zero and means nothing at all.
    """
    src = np.asarray(pts_a, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.asarray(pts_b, dtype=np.float64).reshape(-1, 1, 2)
    if len(src) < 4:
        raise SystemExit(f"[calib] a homography needs >= 4 correspondences; "
                         f"got {len(src)}.")
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, float(ransac_px))
    if H is None:
        raise SystemExit(
            "[calib] findHomography failed. The usual cause is points that are "
            "not all on ONE plane, or three or more points on a straight line.")
    inliers = int(mask.sum()) if mask is not None else len(src)

    loo = []
    if len(src) >= 5:                       # need >= 4 left after removing one
        for i in range(len(src)):
            keep = [j for j in range(len(src)) if j != i]
            Hi, _ = cv2.findHomography(src[keep], dst[keep], cv2.RANSAC,
                                       float(ransac_px))
            if Hi is None:
                continue
            proj = apply_h(Hi, src[i, 0])[0]
            if np.isnan(proj).any():
                continue
            loo.append(float(np.linalg.norm(proj - dst[i, 0])))
    return H, inliers, np.array(loo, dtype=np.float64)


def report_fit(H, inliers, loo, n_points, cam_a, cam_b):
    header("THE FIT")
    print(f"  {n_points} correspondence(s), {inliers} RANSAC inlier(s)")
    if n_points < 6:
        print(f"  !! {n_points} points is too few to MEASURE the error. With 4 the")
        print(f"     fit is exact by construction and the residual is zero for")
        print(f"     reasons that have nothing to do with accuracy. Click 8-12.")
    if loo.size:
        print(f"  held-out (leave-one-out) reprojection error, in {cam_b} pixels:")
        print(f"     median {np.median(loo):7.1f}   p95 {np.percentile(loo, 95):7.1f}"
              f"   max {loo.max():7.1f}")
        print(f"  Read this against how wide a person is in {cam_b}: if a person")
        print(f"  is ~60 px across, a 10 px median is a sixth of a person and this")
        print(f"  calibration is good enough to separate two people standing apart.")
    else:
        print("  held-out error: not computable (needs >= 5 points)")


def write_record(path, cam_a, cam_b, pts_a, pts_b, size_a, size_b,
                 H, inliers, loo, source):
    """The calibration record. Versioned as a whole, per ADR-003A section 1.2 --
    `calib_version` alone cannot answer "why did v7 beat v6" six months later."""
    rec = {
        "calib_version": time.strftime("%Y%m%d_%H%M%S"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kind": "camera_to_camera_floor_homography",
        "cam_a": cam_a,
        "cam_b": cam_b,
        # NOT optional -- ADR-003A section 1.3. A homography belongs to the pixel
        # space it was authored in, and source.resize_width would silently change
        # that space with no error message anywhere.
        "image_size_a": list(size_a),
        "image_size_b": list(size_b),
        "n_points": len(pts_a),
        "ransac_inliers": int(inliers),
        "heldout_px_median": (float(np.median(loo)) if loo.size else None),
        "heldout_px_p95": (float(np.percentile(loo, 95)) if loo.size else None),
        "heldout_px_max": (float(loo.max()) if loo.size else None),
        "points_a": [[float(x), float(y)] for x, y in pts_a],
        "points_b": [[float(x), float(y)] for x, y in pts_b],
        "H_a_to_b": [[float(v) for v in row] for row in np.asarray(H)],
        "source": source,
        "notes": "",
    }
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"\n  -> {path}")
    print(f"     calib_version {rec['calib_version']}")
    return rec


# ------------------------------------------------------------------ the click UI

class Clicker:
    """Two frames side by side; click a floor point in the left, then the SAME
    floor point in the right, repeat. Alternation is enforced so a pair can never
    be half-recorded."""

    def __init__(self, img_a, img_b, cam_a, cam_b, view_height=760):
        self.img_a, self.img_b = img_a, img_b
        self.cam_a, self.cam_b = cam_a, cam_b
        self.sa = view_height / img_a.shape[0]
        self.sb = view_height / img_b.shape[0]
        self.va = cv2.resize(img_a, None, fx=self.sa, fy=self.sa)
        self.vb = cv2.resize(img_b, None, fx=self.sb, fy=self.sb)
        self.split = self.va.shape[1]
        self.pts_a, self.pts_b = [], []      # ORIGINAL-resolution coordinates
        self.win = f"{cam_a} (left)  |  {cam_b} (right)   [u]ndo [f]it [q]uit"

    def _on_mouse(self, event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        want_a = len(self.pts_a) == len(self.pts_b)
        in_a = x < self.split
        if want_a != in_a:
            side = self.cam_a if want_a else self.cam_b
            print(f"    next click must be in {side} -- pairs are recorded in order")
            return
        if in_a:
            self.pts_a.append((x / self.sa, y / self.sa))
        else:
            self.pts_b.append(((x - self.split) / self.sb, y / self.sb))
        n = min(len(self.pts_a), len(self.pts_b))
        if want_a:
            print(f"    {self.cam_a} point {len(self.pts_a)} recorded")
        else:
            print(f"    pair {n} complete")

    def _canvas(self):
        canvas = np.hstack([self.va.copy(), self.vb.copy()])
        for i, (x, y) in enumerate(self.pts_a):
            self._mark(canvas, int(x * self.sa), int(y * self.sa), i + 1)
        for i, (x, y) in enumerate(self.pts_b):
            self._mark(canvas, int(x * self.sb) + self.split, int(y * self.sb), i + 1)
        for i in range(min(len(self.pts_a), len(self.pts_b))):
            pa = (int(self.pts_a[i][0] * self.sa), int(self.pts_a[i][1] * self.sa))
            pb = (int(self.pts_b[i][0] * self.sb) + self.split,
                  int(self.pts_b[i][1] * self.sb))
            cv2.line(canvas, pa, pb, (90, 90, 90), 1)
        n = min(len(self.pts_a), len(self.pts_b))
        cv2.putText(canvas, f"{n} pair(s) -- aim for 8-12", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 220, 0) if n >= 8 else (0, 200, 255), 2)
        return canvas

    @staticmethod
    def _mark(canvas, x, y, label):
        cv2.drawMarker(canvas, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(canvas, str(label), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    def run(self):
        """-> (pts_a, pts_b) or (None, None) if the operator quit."""
        try:
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.win, self._on_mouse)
        except cv2.error as e:
            raise SystemExit(
                f"[calib] no display available ({e}).\n"
                f"        Run --export on the capture box, copy the two PNGs in "
                f"calibration/ to a machine with a screen, and run --click there.")
        while True:
            cv2.imshow(self.win, self._canvas())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return None, None
            if key == ord("u"):
                if len(self.pts_b) == len(self.pts_a) and self.pts_b:
                    self.pts_b.pop()
                elif self.pts_a:
                    self.pts_a.pop()
                print(f"    undo -> {min(len(self.pts_a), len(self.pts_b))} pair(s)")
            if key == ord("f"):
                n = min(len(self.pts_a), len(self.pts_b))
                if n < 4:
                    print(f"    need >= 4 pairs to fit, have {n}")
                    continue
                cv2.destroyAllWindows()
                return self.pts_a[:n], self.pts_b[:n]


# ------------------------------------------------------------------ entry points

def load_frames(cam_a, cam_b):
    pa, pb = frame_png(cam_a), frame_png(cam_b)
    for p in (pa, pb):
        if not os.path.exists(p):
            raise SystemExit(f"[calib] {p} not found -- run --export first "
                             f"(on the box that has the clips).")
    ia, ib = cv2.imread(pa), cv2.imread(pb)
    if ia is None or ib is None:
        raise SystemExit("[calib] could not read the exported PNGs.")
    return ia, ib


def from_json(path, cam_a, cam_b):
    """Correspondences typed in by hand, for when clicking is impractical.

    Expected: {"points_a": [[x,y], ...], "points_b": [[x,y], ...]} in ORIGINAL
    image pixels, plus optional "image_size_a" / "image_size_b" as [w, h].
    """
    with open(path) as f:
        blob = json.load(f)
    pts_a = blob.get("points_a") or []
    pts_b = blob.get("points_b") or []
    if len(pts_a) != len(pts_b):
        raise SystemExit(f"[calib] {path}: points_a has {len(pts_a)} entries and "
                         f"points_b has {len(pts_b)} -- they must pair up.")
    size_a = blob.get("image_size_a")
    size_b = blob.get("image_size_b")
    if not (size_a and size_b):
        # Fall back to the exported frames, which is the only other place the
        # authoring resolution is knowable. Guessing it would defeat section 1.3.
        ia, ib = load_frames(cam_a, cam_b)
        size_a = [ia.shape[1], ia.shape[0]]
        size_b = [ib.shape[1], ib.shape[0]]
        print(f"  image_size taken from the exported frames: {size_a} / {size_b}")
    return pts_a, pts_b, size_a, size_b


def main():
    strict_flags()
    cam_a = arg("--cam-a", DEFAULT_CAM_A)
    cam_b = arg("--cam-b", DEFAULT_CAM_B)
    ransac_px = float(arg("--ransac-px", "3.0"))

    if flag("--export"):
        header(f"EXPORT ONE FRAME PER CAMERA  ({cam_a}, {cam_b})")
        export_frame(cam_a)
        export_frame(cam_b)
        host = os.environ.get("HOSTNAME") or "this-host"
        headless = not (os.environ.get("DISPLAY")
                        or os.environ.get("WAYLAND_DISPLAY"))
        print(f"\n  Clicking needs a SCREEN"
              f"{' -- this box has none' if headless else ''}. From your local machine:")
        print(f"\n    scp <user>@{host}:{calib_dir()}/frame_cam_*.png calibration/")
        print(f"    python tests/calibration/click_covisible_points.py --click")
        print(f"\n  then copy calibration/homography_*.json back here.")
        print(f"\n  Click FLOOR points only -- where things MEET the ground.")
        print(f"  Tile corners, door-frame bases, table/chair feet, rug corners.")
        print(f"  Aim for 8-12. Pick a different frame with --frame N if these")
        print(f"  two are crowded with people standing on the landmarks.")
        return

    source = None
    if arg("--from-json") is not None:
        path = arg("--from-json")
        pts_a, pts_b, size_a, size_b = from_json(path, cam_a, cam_b)
        source = f"from-json:{os.path.basename(path)}"
        header(f"FIT FROM {path}")
    elif flag("--click"):
        require_display()
        img_a, img_b = load_frames(cam_a, cam_b)
        size_a = [img_a.shape[1], img_a.shape[0]]
        size_b = [img_b.shape[1], img_b.shape[0]]
        header(f"CLICK MATCHING FLOOR POINTS  ({cam_a} -> {cam_b})")
        print("  Left image, then the SAME spot in the right image. Repeat.")
        print("  Floor points ONLY -- where an object meets the ground.")
        print("  [u] undo   [f] fit and save   [q] quit without saving\n")
        pts_a, pts_b = Clicker(img_a, img_b, cam_a, cam_b,
                               int(arg("--view-height", "760"))).run()
        if pts_a is None:
            print("  quit -- nothing written.")
            return
        source = "clicked"
    else:
        raise SystemExit(__doc__.strip().split("\n\n")[1])

    H, inliers, loo = fit_homography(pts_a, pts_b, ransac_px)
    report_fit(H, inliers, loo, len(pts_a), cam_a, cam_b)
    path = out_path(cam_a, cam_b)
    write_record(path, cam_a, cam_b, pts_a, pts_b, size_a, size_b,
                 H, inliers, loo, source)
    print(f"\n  Next:")
    print(f"    python tests/calibration/measure_covisible_geometry.py <run_id>")


if __name__ == "__main__":
    main()
