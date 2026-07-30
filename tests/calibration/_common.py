"""
Shared helpers for the calibration harness.

These scripts EXIST TO PRODUCE NUMBERS, not to pass. Their output is what
REMEDIATION_PLAN.md Part H records, and re-running them on new footage is how
that section stays honest.

The one methodological rule that matters, and the one that was got wrong first
time round: a "different person" pair is ONLY valid when the two tracks
CO-OCCUR IN THE SAME FRAME. A person cannot be two simultaneous detections, so
co-occurrence proves distinctness. Comparing tracks that never co-occur silently
mixes in fragments of the SAME person (which is a known defect here), and that
inflates every different-person statistic. `proven_distinct_pairs` enforces this.
"""

from __future__ import annotations

import os
import sys
from itertools import combinations

import cv2
import numpy as np

# ---------------------------------------------------------------- project paths

def project_root() -> str:
    """Repo root, derived from this file's location (tests/calibration/_common.py)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bootstrap() -> str:
    """Put `src` on sys.path the same way main.py does, and cd to the repo root
    so relative model paths (yolo11n.pt, src/reid/weights/...) resolve."""
    root = project_root()
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    os.chdir(root)
    return root


REID_WEIGHTS = "src/reid/weights/osnet_ain_x1_0.pth"


def reid_tap() -> str:
    """The feature tap the pipeline would use (#39), overridable for comparisons
    with CALIB_REID_TAP. Read from config for the same reason DETECT_WEIGHTS is:
    a tap change must not leave the harness measuring the old feature space."""
    override = os.environ.get("CALIB_REID_TAP")
    if override:
        return override
    try:
        import yaml
        with open(os.path.join(project_root(), "config.yaml")) as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get("reid") or {}).get("tap") or "post_relu"
    except Exception:                                          # noqa: BLE001
        return "post_relu"
POSE_WEIGHTS = "yolo11n-pose.pt"


def detect_weights() -> str:
    """The detector the pipeline would actually load.

    Read from config.yaml (`detector.model`) rather than hardcoded, so a model swap
    cannot leave the calibration harness silently measuring the OLD model and
    reporting it as the shipped behaviour. `CALIB_DETECT_WEIGHTS` overrides, which
    is how compare_detector_models.py measures more than one.
    """
    override = os.environ.get("CALIB_DETECT_WEIGHTS")
    if override:
        return override
    try:
        import yaml
        with open(os.path.join(project_root(), "config.yaml")) as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get("detector") or {}).get("model") or "yolo11n.pt"
    except Exception:                                          # noqa: BLE001
        return "yolo11n.pt"


# Kept as a module-level name because the existing scripts read it directly. It is
# resolved at import time, AFTER bootstrap() has chdir'd to the repo root in the
# scripts that call it first -- project_root() is path-derived, so it does not care.
DETECT_WEIGHTS = detect_weights()


def pick_video(preferred: str | None = None) -> str:
    """Resolve a clip to measure on.

    Defaults to register_file.avi: it is 2560x1440 (production resolution) and
    decodes with ZERO H.265 reference errors, unlike test_file.avi (294 broken
    frames) and test_v2.avi (207). Measuring feature quality on a clip with
    corrupted frames measures the corruption.
    """
    if preferred and os.path.exists(preferred):
        return preferred
    for cand in ("register_file.avi", "test_v2.avi", "test_file.avi",
                 "recorded_1_2.mp4"):
        if os.path.exists(cand):
            if preferred:
                print(f"[calib] {preferred!r} not found; using {cand!r}")
            return cand
    raise SystemExit(
        "[calib] No footage found in the repo root. Pass a path as the first "
        "argument, e.g.  python tests/calibration/<script>.py myclip.mp4")


# ---------------------------------------------------------------- frame loading

def sample_frames(video: str, count: int, stride: int) -> list[np.ndarray]:
    """Every `stride`-th frame, up to `count`. Use for appearance statistics,
    where temporal continuity does not matter."""
    cap = cv2.VideoCapture(video)
    frames, i = [], 0
    while len(frames) < count:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(f)
        i += 1
    cap.release()
    if not frames:
        raise SystemExit(f"[calib] could not decode any frames from {video!r}")
    return frames


def consecutive_frames(video: str, count: int) -> list[np.ndarray]:
    """The first `count` frames, unskipped. Use for anything involving TRACKING
    -- ByteTrack needs real frame-to-frame continuity, so a strided sample
    measures a different (and much harder) problem than the pipeline faces."""
    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) < count:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f"[calib] could not decode any frames from {video!r}")
    return frames


# ---------------------------------------------------------------- box geometry

def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def containment(a, b) -> float:
    """Intersection over the area of the SMALLER box. Catches the
    torso-inside-full-body case, whose IoU is modest but which is still one
    person wearing two boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def det_box(d) -> tuple:
    return (d.x1, d.y1, d.x2, d.y2)


# ------------------------------------------------- embedding collection helpers

MIN_CROP_H, MIN_CROP_W = 64, 24     # mirrors reid.quality.min_height / min_width


def collect_track_embeddings(frames, detector, embed_fn, min_h=MIN_CROP_H,
                             min_w=MIN_CROP_W):
    """Run detect+track over `frames`, embed every usable crop, and return

        by_track : {track_id: [embedding, ...]}   in frame order
        cooccur  : {(tid_a, tid_b), ...}          pairs seen in the SAME frame
        per_frame: [{track_id: embedding}, ...]   one dict per frame

    `embed_fn(list_of_bgr_crops) -> (N, D) array` so callers can swap the
    feature tap without touching this function.
    """
    from detector import crop_person

    by_track: dict[int, list] = {}
    per_frame: list[dict] = []
    cooccur: set[tuple] = set()

    for frame in frames:
        dets = [d for d in detector.track(frame) if d.track_id is not None]
        crops, tids = [], []
        for d in dets:
            crop = crop_person(frame, d)
            if crop is None or crop.size == 0:
                continue
            h, w = crop.shape[:2]
            if h < min_h or w < min_w:
                continue
            crops.append(crop)
            tids.append(d.track_id)
        if not crops:
            per_frame.append({})
            continue
        embs = embed_fn(crops)
        this_frame = {}
        for tid, e in zip(tids, embs):
            by_track.setdefault(tid, []).append(e)
            this_frame[tid] = e
        per_frame.append(this_frame)
        for a, b in combinations(sorted(set(tids)), 2):
            cooccur.add((a, b))

    return by_track, cooccur, per_frame


def proven_distinct_pairs(by_track, cooccur, min_obs=6):
    """Track pairs that CO-OCCUR in at least one frame and both have enough
    observations. Co-occurrence is what proves they are different people.

    Also returns the pairs it had to EXCLUDE, because those are exactly the
    pairs that may be one fragmented person -- worth printing so the reader can
    see how much of the candidate space was discarded and why.
    """
    usable = {t: v for t, v in by_track.items() if len(v) >= min_obs}
    pairs = [(a, b) for (a, b) in cooccur if a in usable and b in usable]
    excluded = [p for p in combinations(sorted(usable), 2) if p not in cooccur]
    return usable, pairs, excluded


# ---------------------------------------------------------------- reporting

def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def describe(arr, label: str, width: int = 34) -> str:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return f"  {label:<{width}} (no samples)"
    return (f"  {label:<{width}} n={a.size:<6d} mean={a.mean():.3f} "
            f"p5={np.percentile(a, 5):.3f} median={np.median(a):.3f} "
            f"p95={np.percentile(a, 95):.3f} min={a.min():.3f} max={a.max():.3f}")


def operating_points(same, other, thresholds=(0.50, 0.60, 0.70, 0.80, 0.85, 0.90)):
    """correct-accept% / wrong-accept% at each threshold."""
    same, other = np.asarray(same), np.asarray(other)
    out = []
    for t in thresholds:
        tp = 100.0 * (same >= t).mean() if same.size else float("nan")
        fp = 100.0 * (other >= t).mean() if other.size else float("nan")
        out.append((t, tp, fp))
    return out


def print_operating_points(same, other, indent="    "):
    print(f"{indent}{'thr':>5}{'correct%':>10}{'WRONG%':>9}")
    for t, tp, fp in operating_points(same, other):
        print(f"{indent}{t:>5.2f}{tp:>10.1f}{fp:>9.1f}")


def margin(same, other) -> float:
    """Worst-case separation: same-person 5th percentile minus different-person
    95th percentile. Positive means a threshold exists that mostly works."""
    same, other = np.asarray(same), np.asarray(other)
    if same.size == 0 or other.size == 0:
        return float("nan")
    return float(np.percentile(same, 5) - np.percentile(other, 95))


def header(title: str, width: int = 76):
    print("\n" + "=" * width + f"\n{title}\n" + "=" * width)


def footnote_sample_size(usable, pairs, excluded):
    print(f"\n  SAMPLE: {len(usable)} tracks, {len(pairs)} proven-distinct pairs.")
    if excluded:
        print(f"  EXCLUDED {len(excluded)} never-co-visible pair(s) {sorted(excluded)}")
        print("  -- those may be ONE person fragmented, so including them would")
        print("     inflate the different-person statistics.")
    print("  Treat small samples as hypotheses. See REMEDIATION_PLAN.md Part H.")
