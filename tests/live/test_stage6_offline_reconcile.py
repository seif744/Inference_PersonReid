"""
Stage 6 -- offline-reconcile live flow (the "correct-answer-at-the-end" path).

Validates the NEW glue added for RTSP + offline reconciliation, WITHOUT recording
the raw feed, using deterministic synthetic data (no GPU/models/threads):

  A. IdentityStage persists ONLY fresh embeddings, subsampled by sample_stride,
     with the metadata reconcile + audit need (camera/track_id/frame/run_id/ts +
     bbox/confidence/crop_quality) and NO live reid_id/global_id (identity is
     rebuilt from scratch offline).
  B. RenderStage capture mode records each CLEAN processed frame to a temp clip
     and its box geometry, index-aligned (the shape render_final_videos re-renders
     from).
  C. The reused tail (reconcile_tracklets -> build_gid_map -> render_final_videos)
     merges two same-person cross-camera tracklets to ONE id and writes the final
     output_<cam>.mp4 from the captured clips.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

from _synth import Check, person, observe

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_ROOT, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402

from live.frame import Frame  # noqa: E402
from live.render import RenderStage  # noqa: E402
from live.identity_stage import IdentityStage  # noqa: E402
from live.queues import DropOldestQueue  # noqa: E402
from database.store import PersonVectorStore  # noqa: E402


_CQ = {"accepted": True, "reason": "ok", "blur": 200.0, "brightness": 120.0}


def _det(tid, emb, x1=10, y1=20, x2=40, y2=120, conf=0.9, cq=None):
    return SimpleNamespace(track_id=tid, embedding=emb,
                           crop_quality=(_CQ if cq is None else cq),
                           x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf,
                           reid_id=None, global_id=None)


def _frame(cam, fidx, dets, fresh_ids, ts=None):
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    return Frame(cam=cam, ts=(ts if ts is not None else 1.0 + fidx),
                 frame_index=fidx, image=img, device="cpu",
                 detections=dets, meta={"fresh_track_ids": set(fresh_ids)})


def part_a(c):
    store = PersonVectorStore(path=":memory:")
    stage = IdentityStage(DropOldestQueue(8), {}, __import__("threading").Event(),
                          store=store, run_id="runA", sample_stride=2)
    p = person(1)

    # track 10: 4 FRESH obs -> stride 2 -> stored at counts 2 and 4 => 2 stored.
    for k in range(4):
        d = _det(10, observe(p, seed=10 + k))
        stage._resolve_frame(_frame("cam_a", k, [d], fresh_ids={10}))
    # track 11: 1 fresh obs -> count 1, 1 % 2 != 0 => NOT stored yet.
    stage._resolve_frame(_frame("cam_a", 99, [_det(11, observe(p, seed=200))],
                                fresh_ids={11}))
    # a NON-fresh detection (track present but not in fresh set) is never stored.
    stage._resolve_frame(_frame("cam_a", 100, [_det(12, observe(p, seed=300))],
                                fresh_ids=set()))

    c.eq(store.count(), 2, "only sampled fresh observations were persisted")
    c.eq(stage.stored, 2, "stage.stored counter matches store")

    pts, _ = store.client.scroll(store.collection, limit=10,
                                 with_payload=True, with_vectors=False)
    pl = (pts[0].payload or {})
    c.ok(pl.get("camera") == "cam_a" and pl.get("track_id") == 10,
         "payload carries camera + track_id")
    c.ok(pl.get("run_id") == "runA" and "frame" in pl and "ts" in pl,
         "payload carries run_id + frame + ts")
    c.ok(pl.get("bbox") == [10.0, 20.0, 40.0, 120.0], "payload carries bbox")
    c.ok(abs(pl.get("confidence") - 0.9) < 1e-6
         and isinstance(pl.get("crop_quality"), dict)
         and pl["crop_quality"].get("accepted") is True,
         "payload carries confidence + crop_quality dict")
    c.ok("reid_id" not in pl and "global_id" not in pl,
         "payload has NO live identity (reconcile assigns it from scratch)")


def part_b(c):
    clip = os.path.join(_HERE, "._test_capture_cam.mp4")
    if os.path.exists(clip):
        os.remove(clip)
    import threading
    r = RenderStage("cam_a", DropOldestQueue(8), None, threading.Event(),
                    capture_mode=True, clip_path=clip, clip_fps=20.0)
    for f in range(3):
        r._process(_frame("cam_a", f, [_det(10, None), _det(11, None)], fresh_ids=set()))
    if r._clip is not None:
        r._clip.release()

    c.eq(len(r.annotations), 3, "one geometry list captured per processed frame")
    c.eq(len(r.annotations[0]), 2, "both boxes captured for a 2-person frame")
    g = r.annotations[0][0]
    c.ok({"x1", "y1", "x2", "y2", "track_id", "confidence"} <= set(g),
         "captured geometry has the render_final_videos fields")
    c.ok(os.path.exists(clip) and os.path.getsize(clip) > 0,
         "temp processed-frame clip was written")
    os.remove(clip)


def part_c(c):
    store = PersonVectorStore(path=":memory:")
    p = person(7)
    for cam, tid, base in (("cam_a", 10, 10), ("cam_b", 20, 40)):
        embs = [observe(p, noise=0.05, seed=base + k) for k in range(3)]
        payloads = [{"camera": cam, "track_id": tid, "frame": f, "run_id": "runC"}
                    for f in (0, 10, 20)]
        store.add_many(embs, payloads)

    from identity.reconcile import reconcile_tracklets
    from main import build_gid_map, render_final_videos

    reconcile_tracklets(store, threshold=0.63, run_id="runC",
                        same_camera_threshold=0.90, require_reciprocal_best=True,
                        min_tracklet_observations=3, log=lambda *a: None)
    gmap = build_gid_map(store, "runC")
    c.ok(gmap.get(("cam_a", 10)) is not None
         and gmap.get(("cam_a", 10)) == gmap.get(("cam_b", 20)),
         "same person across two cameras reconciled to ONE id")

    # Build the captured clips + geometry the way the live render stage would,
    # then run the SAME re-render the pipeline calls.
    clips, outs = {}, {}
    shared = {"annotations": {}}
    for cam, tid in (("cam_a", 10), ("cam_b", 20)):
        clip = os.path.join(_HERE, f"._test_src_{cam}.mp4")
        w = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 64))
        annos = []
        for _ in range(3):
            w.write(np.full((64, 64, 3), 60, dtype=np.uint8))
            annos.append([{"x1": 5, "y1": 5, "x2": 30, "y2": 60,
                           "track_id": tid, "confidence": 0.9}])
        w.release()
        clips[cam] = clip
        shared["annotations"][cam] = annos
        outs[cam] = f"output_{cam}.mp4"
        if os.path.exists(outs[cam]):
            os.remove(outs[cam])

    jobs = [(cam, clips[cam]) for cam in clips]
    render_cfg = {"source": {"resize_width": 0}, "display": {"output_fps": 20.0}}
    render_final_videos(jobs, render_cfg, shared, store, "runC")

    for cam in clips:
        c.ok(os.path.exists(outs[cam]) and os.path.getsize(outs[cam]) > 0,
             f"final reconciled video written for {cam}")
    for f in list(clips.values()) + list(outs.values()):
        if os.path.exists(f):
            os.remove(f)


def main():
    c = Check("Stage6 offline-reconcile live flow (persistence + capture + tail)")
    part_a(c)
    part_b(c)
    part_c(c)
    c.done()


if __name__ == "__main__":
    main()
