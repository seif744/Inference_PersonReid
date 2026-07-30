"""
render.py  --  STAGE: draw the annotated frame (per camera).

Takes identity-stamped frames and draws boxes + reid labels, then hands the
annotated image to that camera's writer queue. The Render stage NEVER performs
inference (strict stage separation, v5 §2) -- it only draws, so it can run on
plain CPU threads independent of the (GPU) inference stage.

Device rule: if the frame's image is GPU-resident it is downloaded to CPU ONCE
here (the single GPU->CPU copy per frame) before drawing. In Stage 1 / CPU mode
frames are already CPU numpy arrays, so this is a no-op. Reuses the existing
`draw_detections`/`draw_hud` so overlays match the file pipeline exactly (incl.
the negative-provisional-id "REID ..." handling).

CAPTURE MODE (offline-reconcile flow, live.reconcile.enabled): instead of drawing
the live (provisional) ids and feeding the wall-clock paced writer, the stage
writes each camera's CLEAN processed frame to a transient temp clip and records
that frame's box geometry (x1,y1,x2,y2,track_id,confidence). Pixels AND geometry
come from the SAME Frame object here, so they stay index-aligned no matter what
upstream drops -- the exact `shared["annotations"]` shape the file path's
render_final_videos() re-renders from. Final ids are settled by the offline
reconcile at shutdown; this stage just preserves the material to re-render with.
"""

import json
import os
import threading

import cv2

from drawing import draw_detections, draw_hud


def _to_cpu_bgr(image, device):
    """Single GPU->CPU download point (no-op for CPU frames). Kept tiny and
    guarded so nothing imports torch unless a GPU frame actually arrives."""
    if device == "cpu" or image is None:
        return image
    try:
        import torch
        if isinstance(image, torch.Tensor):
            arr = image.detach().to("cpu").numpy()
            return arr
    except Exception:
        pass
    return image


class RenderStage(threading.Thread):
    def __init__(self, cam, render_queue, writer_queue, stop_event,
                 capture_mode=False, clip_path=None, clip_fps=20.0,
                 run_id=None):
        super().__init__(name=f"render-{cam}", daemon=True)
        self.cam = cam
        self.run_id = run_id
        self.in_q = render_queue
        self.writer_queue = writer_queue
        self.stop_event = stop_event
        self.rendered = 0
        # ---- capture mode (offline-reconcile flow) ----
        self.capture_mode = bool(capture_mode)
        self.clip_path = clip_path
        self.clip_fps = float(clip_fps) if clip_fps and clip_fps > 0 else 20.0
        # Per-frame box geometry, index-aligned with the temp clip. Consumed by
        # render_final_videos() as shared["annotations"][cam] after reconcile.
        self.annotations = []
        self._clip = None
        self._clip_disabled = False
        # Wall-clock of the first/last captured frame -> this camera's MEASURED
        # rate (#45/#46). A single global output.fps_default tagged every camera at
        # 20, so cam_224 at a real 14.8 fps played 1.35x FAST and cam_219 at 24.2
        # played 1.21x SLOW -- which the operator noticed immediately and which
        # makes cross-camera comparison by eye unreliable.
        self._first_ts = None
        self._last_ts = None

    failed = None

    def run(self):
        # #55: no worker had an exception guard. An unhandled error killed the
        # THREAD while the pipeline kept running -- a dead InferenceStage just
        # produces no detections, and a dead RenderStage writes nothing, with no
        # error anywhere. Record it loudly and set `failed` so shutdown can say
        # which stage died instead of leaving an empty output to explain.
        _FATAL_GUARD = True
        try:
            while not self.stop_event.is_set():
                frame = self.in_q.get(timeout=0.1)
                if frame is None:
                    continue
                self._process(frame)
            # Capture mode: drain whatever is still queued so trailing frames make it
            # into the clip (real-time-first still applies -- we don't block forever),
            # then finalize the temp clip so it is re-decodable by the render pass.
            if self.capture_mode:
                frame = self.in_q.get(timeout=0.0)
                while frame is not None:
                    self._process(frame)
                    frame = self.in_q.get(timeout=0.0)
                if self._clip is not None:
                    self._clip.release()
                    print(f"[render:{self.cam}] captured {self.rendered} processed "
                          f"frames -> {self.clip_path}")
                    self._write_annotations()
        except BaseException as e:                              # noqa: BLE001
            import traceback
            self.failed = e
            print(f"[RenderStage] FATAL: this stage has DIED ({type(e).__name__}: {e}). "
                  f"The run will continue but this stage produces nothing from now on.")
            traceback.print_exc()
            raise
    def _process(self, frame):
        img = _to_cpu_bgr(frame.image, frame.device)
        if img is None:
            return
        dets = frame.detections or []
        if self.capture_mode:
            ts = getattr(frame, "ts", None)
            if ts is not None:
                if self._first_ts is None:
                    self._first_ts = float(ts)
                self._last_ts = float(ts)
            self._capture(img, dets)
            return
        img = draw_detections(img, dets)
        img = draw_hud(img, person_count=len(dets))
        # Hand the annotated pixels + capture ts to the writer (pacing uses ts).
        self.writer_queue.put((img, frame.ts))
        self.rendered += 1

    def annotations_path(self):
        """Sidecar path for this camera's box geometry."""
        return os.path.splitext(self.clip_path)[0] + ".annotations.json"

    def _write_annotations(self):
        """Persist the per-frame box geometry next to the clip.

        Without this the clip alone is useless for re-rendering: `annotations`
        lives only in this thread's memory and dies with the process, and the
        store holds a bbox only every `reid.interval` frames. Clip + sidecar +
        the stored embeddings together are a COMPLETE record of the run, so any
        reconcile setting can be re-rendered offline afterwards
        (tests/calibration/rerender_from_clips.py) instead of costing another
        live capture with people walking the room.

        Fail-soft: this runs during finalization, where nothing is allowed to
        cost the run its ids or its video.
        """
        path = self.annotations_path()
        try:
            with open(path, "w") as f:
                json.dump({
                    "camera": self.cam,
                    "run_id": self.run_id,
                    "clip": os.path.basename(self.clip_path),
                    "clip_fps": self.clip_fps,
                    # The rate to PLAY this back at. clip_fps is only what the
                    # container was tagged with (a global default); this is what
                    # the frames were actually produced at.
                    "measured_fps": self.measured_fps(),
                    "frames": len(self.annotations),
                    "annotations": self.annotations,
                }, f)
            print(f"[render:{self.cam}] box geometry -> {path}")
        except (OSError, TypeError, ValueError) as e:
            print(f"[render:{self.cam}] could not write {path}: {e} "
                  f"(the run is unaffected; offline re-render will not be "
                  f"possible for this camera)")

    def measured_fps(self):
        """Frames actually captured per wall-clock second, or None if unknowable.

        This is the rate the clip's frames were really produced at, so it is the
        rate that plays them back in real time -- unlike the nominal camera fps
        (dropped frames make the real rate lower) or a global config default.
        """
        if (self._first_ts is None or self._last_ts is None
                or self.rendered < 2):
            return None
        elapsed = self._last_ts - self._first_ts
        if elapsed <= 0:
            return None
        return (self.rendered - 1) / elapsed

    def _capture(self, img, dets):
        """Record the CLEAN frame + its box geometry (both from this one Frame,
        so they stay aligned). No drawing here -- the final ids aren't known until
        the offline reconcile runs at shutdown."""
        self.annotations.append([
            {
                "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                "track_id": d.track_id,
                "confidence": getattr(d, "confidence", 0.0),
            }
            for d in dets
        ])
        if self._clip is None and not self._clip_disabled:
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._clip = cv2.VideoWriter(self.clip_path, fourcc, self.clip_fps, (w, h))
            if not self._clip.isOpened():
                print(f"[render:{self.cam}] ERROR: could not open temp clip "
                      f"{self.clip_path}; final re-render for this camera DISABLED.")
                self._clip.release()
                self._clip = None
                self._clip_disabled = True
        if self._clip is not None:
            self._clip.write(img)
        self.rendered += 1
