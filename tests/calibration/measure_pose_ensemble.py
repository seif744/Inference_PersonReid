"""
Pose ensemble effect on REAL frames: does it add duplicate boxes?

BATCH PATH ONLY. `live.inference.pose_ensemble: false` keeps this off the product
path, so this script is here for completeness and in case the batch path returns.
`characterize_known_defects.py` proves the mechanism synthetically; this measures
how often it actually fires on footage.

    python tests/calibration/measure_pose_ensemble.py [video] [frames] [stride]

Relevant even though the batch path is unused, because the live toggle's default
is `True` (`_g("inference", "pose_ensemble", True)`) -- deleting one config line
re-enables the duplicate generator on the live path. That is issue #66.
"""

import sys

from _common import (bootstrap, pick_video, sample_frames, DETECT_WEIGHTS,
                     POSE_WEIGHTS, iou, det_box, header)

bootstrap()

from detector import PersonDetector

VIDEO = pick_video(sys.argv[1] if len(sys.argv) > 1 else None)
NFRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 40
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 20

POSE_CFG = {"enabled": True, "pose_model": POSE_WEIGHTS,
            "min_containment": 0.6, "conf": 0.25}

frames = sample_frames(VIDEO, NFRAMES, STRIDE)
print(f"[calib] {VIDEO}: {len(frames)} frames @ stride {STRIDE}, "
      f"{frames[0].shape[1]}x{frames[0].shape[0]}\n")


def run(pose_cfg, label):
    det = PersonDetector(model_path=DETECT_WEIGHTS, confidence_threshold=0.4,
                         person_class_id=0, tracker_config="bytetrack.yaml",
                         pose_ensemble=pose_cfg, iou=0.60)
    counts, dup_pairs, synthetic = [], 0, set()
    for f in frames:
        dets = det.track(f)
        counts.append(len(dets))
        for d in dets:
            if d.track_id is not None and d.track_id >= 100000:
                synthetic.add(d.track_id)
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                if iou(det_box(dets[i]), det_box(dets[j])) >= 0.5:
                    dup_pairs += 1
    print(f"  {label:<22} boxes={sum(counts):5d}  overlapping pairs IoU>=0.5="
          f"{dup_pairs:3d}  synthetic ids={len(synthetic)}")
    return counts, dup_pairs, synthetic


header("LIVE config (pose OFF) vs BATCH config (pose ON)")
off_counts, off_dup, _ = run(None, "pose OFF (live)")
on_counts, on_dup, synthetic = run(POSE_CFG, "pose ON  (batch)")

added = sum(on_counts) - sum(off_counts)
fired = sum(1 for a, b in zip(on_counts, off_counts) if a > b)

header("RESULT")
print(f"  boxes ADDED by the pose ensemble          : {added}")
print(f"  frames where it split something           : {fired}/{len(frames)}")
print(f"  overlapping pairs introduced (IoU>=0.5)   : {on_dup - off_dup}")
print(f"  synthetic track_id + 100000*k ids created : {len(synthetic)}")
print()
if on_dup > off_dup:
    print("  The pose ensemble is the only measured source of overlapping duplicate")
    print("  boxes in this pipeline: a merged box splits into both people while a")
    print("  clean box already covering the second person passes through unchanged.")
else:
    print("  No duplicate pairs introduced on this footage. The mechanism is still")
    print("  real (see characterize_known_defects.py) -- this clip just does not")
    print("  produce the merged-box case often enough to trigger it.")
print()
print("  Synthetic ids are recomputed per frame and their 'primary' assignment")
print("  follows pose-model confidence order, so the real track_id can point at a")
print("  DIFFERENT person frame to frame -- which feeds two people into one bank.")
