"""
identity_stage.py  --  STAGE: assign a stable reid to every detection.

Thin threading wrapper (Stage 3). All decision logic lives in the thread-free,
deterministic `IdentityEngine` (identity_engine.py); this class only:

  1. drains its single input queue into a per-camera fair buffer
     (CameraFairQueue) so no camera starves under a burst (priority.py),
  2. pops frames round-robin and, for each detection, calls the engine to stamp
     `reid_id` / `global_id`, then forwards the frame to that camera's render
     queue,
  3. periodically sweeps the engine (evict cold identities + stale track state).

Serial single thread on purpose: the id space and the active set stay consistent
without locks. The engine ports the PROVEN policies -- evidence gate, same-camera
overlap HARD veto, same-camera cold-reactivation lane, cross-camera
reciprocal-best lane with a pluggable (fail-open) topology veto, and
mint-when-uncertain -- against an in-memory active set (NO Qdrant). Provisional
ids are NEGATIVE and never leave this process (drawing.py renders them as
"REID ..." pending). N-camera generic: everything is keyed by frame.cam.
"""

import threading
import time
from collections import defaultdict

from live.identity_engine import IdentityEngine
from live.priority import CameraFairQueue
from live.topology import FailOpenTopology


class IdentityStage(threading.Thread):
    def __init__(self, identity_queue, render_queues, stop_event,
                 min_evidence_obs=3, same_camera_threshold=0.90,
                 cross_camera_threshold=0.63, accept_margin=0.03,
                 bank_size=20, active_ttl_sec=300.0, max_active_identities=200,
                 topology=None, max_per_lane=64, sweep_interval_sec=2.0,
                 metrics=None, store=None, run_id=None, sample_stride=1):
        super().__init__(name="identity", daemon=True)
        self.in_q = identity_queue
        self.render_queues = render_queues          # {cam: DropOldestQueue}
        self.stop_event = stop_event
        self.metrics = metrics
        self.sweep_interval = float(sweep_interval_sec)

        # ---- offline-reconcile persistence (live.reconcile flow) ----
        # This single-threaded stage is the sole writer, so no store lock is
        # needed. We persist only FRESH embeddings (already throttled to every
        # reid.interval frames by TrackEmbedder -- i.e. NOT per-frame), optionally
        # subsampled again by sample_stride, with the metadata the offline
        # reconcile + audit need. Identity is left UNASSIGNED in the payload: the
        # offline reconcile rebuilds it from scratch (it does not trust live ids).
        self._store = store
        self._run_id = run_id
        self._sample_stride = max(1, int(sample_stride))
        self._sample_count = defaultdict(int)
        self.stored = 0
        self._store_failed = False

        self.engine = IdentityEngine(
            min_evidence_obs=min_evidence_obs,
            same_camera_threshold=same_camera_threshold,
            cross_camera_threshold=cross_camera_threshold,
            accept_margin=accept_margin,
            bank_size=bank_size,
            active_ttl_sec=active_ttl_sec,
            max_active_identities=max_active_identities,
            topology=topology or FailOpenTopology(),
        )
        self.fair = CameraFairQueue(max_per_lane=max_per_lane)
        self.frames_done = 0

    # ---- thread loop -------------------------------------------------------
    def run(self):
        last_sweep = time.monotonic()
        while not self.stop_event.is_set():
            # 1) block briefly for one frame, then DRAIN the rest of the input so
            #    the fair queue can interleave a burst across cameras.
            frame = self.in_q.get(timeout=0.1)
            if frame is not None:
                self.fair.push(frame.cam, frame)
                while True:
                    f = self.in_q.get_nowait()
                    if f is None:
                        break
                    self.fair.push(f.cam, f)

            # 2) process the fair buffer round-robin (anti-starvation order).
            while not self.stop_event.is_set():
                f = self.fair.pop()
                if f is None:
                    break
                self._resolve_frame(f)
                rq = self.render_queues.get(f.cam)
                if rq is not None:
                    rq.put(f)
                self.frames_done += 1

            # 3) periodic eviction on the wall clock (TTLs are in seconds).
            now = time.monotonic()
            if now - last_sweep >= self.sweep_interval:
                self.engine.sweep(time.time())
                last_sweep = now

    def _resolve_frame(self, frame):
        """Stamp reid_id / global_id on every detection in the frame, and (in the
        offline-reconcile flow) persist fresh observations to the store."""
        fresh = frame.meta.get("fresh_track_ids", set())
        store_vecs, store_payloads = [], []
        for det in (frame.detections or []):
            if det.track_id is None:
                det.reid_id = None
                det.global_id = None
                continue
            has_fresh_emb = (det.track_id in fresh) and (det.embedding is not None)
            reid = self.engine.assign(frame.cam, det.track_id, det.embedding,
                                      det.crop_quality, frame.ts, has_fresh_emb)
            det.reid_id = reid
            det.global_id = reid if (reid is not None and reid > 0) else None

            # Persist a SAMPLED fresh observation for the offline reconcile.
            if self._store is not None and not self._store_failed and has_fresh_emb:
                key = (frame.cam, det.track_id)
                self._sample_count[key] += 1
                if self._sample_count[key] % self._sample_stride == 0:
                    store_vecs.append(det.embedding)
                    store_payloads.append(self._observation_payload(frame, det))

        if store_vecs:
            try:
                self._store.add_many(store_vecs, store_payloads)
                self.stored += len(store_vecs)
            except Exception as e:
                # Degrade, never crash (v5 §14): losing persistence only weakens
                # the final reconcile; the live pipeline keeps running.
                self._store_failed = True
                print(f"[identity] store persistence DISABLED after error: {e}")

    def _observation_payload(self, frame, det):
        """Metadata stored alongside the embedding. `camera`/`track_id`/`frame`/
        `run_id` are what reconcile.py groups + spans + filters by; bbox /
        confidence / ts / crop_quality are extra audit metadata, kept when
        available. No reid_id/global_id -- the offline reconcile assigns identity
        from scratch (it does not trust live decisions)."""
        payload = {
            "camera": frame.cam,
            "track_id": int(det.track_id),
            "frame": int(frame.frame_index),
            "run_id": self._run_id,
            "ts": float(frame.ts),
        }
        # Extra metadata, only when the detection actually carries it.
        bbox = [getattr(det, a, None) for a in ("x1", "y1", "x2", "y2")]
        if all(v is not None for v in bbox):
            payload["bbox"] = [float(v) for v in bbox]
        conf = getattr(det, "confidence", None)
        if conf is not None:
            payload["confidence"] = float(conf)
        # crop_quality is the metrics dict from reid/service.py (accepted/blur/
        # brightness/...); store it as-is (JSON-serializable) for audit.
        cq = getattr(det, "crop_quality", None)
        if isinstance(cq, dict):
            payload["crop_quality"] = cq
        elif cq is not None:
            payload["crop_quality"] = float(cq)
        return payload

    # ---- metrics surface (used by later stages / shutdown report) ----------
    @property
    def minted(self):
        return self.engine.minted

    def stats(self):
        e = self.engine
        return {
            "minted": e.minted,
            "reacquired": e.reacquired,                 # same-camera cold reactivations
            "linked": e.linked,                         # cross-camera links
            "active_identities": len(e.store.gids()),
            "fair_dropped": self.fair.dropped,
            "frames_done": self.frames_done,
            "stored": self.stored,                      # observations persisted for reconcile
            # cross-camera diagnostics
            "xcam_attempts": e.xcam_attempts,
            "xcam_rej_threshold": e.xcam_rej_threshold,
            "xcam_rej_margin": e.xcam_rej_margin,
            "xcam_rej_reciprocal": e.xcam_rej_reciprocal,
            "xcam_rej_topology": e.xcam_rej_topology,
            "xcam_max_subthreshold": e.xcam_max_subthreshold,
            "recam_attempts": e.recam_attempts,
            "recam_rej_below": e.recam_rej_below,
            "recam_max_rej": e.recam_max_rej,
            "coactive_vetoes": e.coactive_vetoes,
            "topology_pruned": e.topology_pruned,
            "recam_hist": list(e.recam_hist),
            "xcam_hist": list(e.xcam_hist),
        }
