# Handoff — Multi-Camera Person ReID: v5 Real-Time RTSP Pipeline

**Repo:** `github.com/seif744/PersonReID` (local dir name: `Inference`)
**Branch:** `research` (== `origin/research`; merged to `main` via PR #13, so both carry the work)
**Targets:** A100 GPU server (real-time); CPU dev box (functional only).

---

## 0. TL;DR — current state

- **v5 real-time streaming pipeline** (`src/live/`): N RTSP cams → per-camera
  detect/track + ReID → online identity → `output_<cam>.mp4`. Validated on the
  A100 GPU server. The **raw stream is never saved.**
- **Correct-answer-at-the-end (current workflow):** the online engine's live ids
  are provisional; on **Ctrl-C** the live path runs the **offline reconcile**
  (the same `reconcile_tracklets` the file path uses) over persisted
  observations + the captured processed frames, and re-renders
  `output_<cam>.mp4` with correct cross-camera ids. This deliberately relaxes the
  original "no offline reconcile" invariant — the supervisor approved trading
  instant-on-Ctrl-C output for correct ids, without leaving OSNet-AIN. See
  `live.reconcile` in config.yaml and ARCHITECTURE.md §8.
- **Done:** ReID model swap (OSNet-AIN); v5 Stage 0/1 (scaffold + threaded
  skeleton), Stage 2 (per-camera parallel inference), Stage 3 (identity engine),
  and the offline-reconcile-on-stop flow. Committed + pushed on `research`.
- **CPU is only for logic/plumbing tests** — never judge reid quality on CPU
  (choppy, drops frames). The GPU server is the target.

---

## 1. Goal & hard invariants (from the v5 handoff)

- N RTSP streams (test with 3; a 4th added later — must be **N-generic**, never hardcode a count).
- Real-time → annotated `output_<cam>.mp4` per camera. Device-portable (GPU when present, else CPU).
- **Invariants (non-negotiable):** raw stream **never saved** (only the *processed* frames are held transiently, then deleted after the final render); false-merge-conservative (mint when unsure); real-time-first (frame-drop = overload protection); the existing **file-batch path must not break** (behavioural-equivalence regression gate).
- **Amended:** the original "no offline reconcile on the live path" invariant was **lifted** (supervisor decision): correct ids at the end matter more than instant output, so the live path now runs the offline reconcile on Ctrl-C (§4). Still no raw-stream recording, still OSNet-AIN.

## 2. Execution rules (FIXED — highest authority)

1. **Priority order** (never trade a higher for a lower): (1) preserve file-batch path → (2) correctness → (3) identity quality → (4) race-free shutdown/valid MP4s → (5) bounded latency → (6) throughput → (7) GPU optimization → (8) code cleanliness.
2. **No architectural drift:** stage boundaries + queue/drop policies are fixed. If a problem appears, STOP + document + ask — don't collapse stages or merge queues.
3. **ByteTrack stop-rule:** batch the *detector + ReID*, keep **one tracker per camera**. If decoupling needs ultralytics internals or > ~1 stage of effort, fall back to per-cam `.track()` + batched embed. Never rewrite ByteTrack.
4. **Per-stage validation gate:** one stage at a time; after each, run its checks, fix failures, get explicit approval before the next.
5. **Topology is fail-open:** cross-camera veto only rejects pairs it has data for; unknown cameras → appearance-only (never blocked). Adjacency + transit-time data **not yet provided**.

---

## 3. The two run modes (IMPORTANT — do not confuse)

| Mode | What it does | Use |
|---|---|---|
| **`--mode live`** | **v5 real-time** pipeline (`src/live/`). Streams (or files) → real-time processing, **raw feed never recorded**. Online engine gives provisional on-screen ids; **Ctrl-C runs the offline reconcile** and re-renders `output_<cam>.mp4` with correct cross-camera ids. | **The target.** RTSP. |
| `auto` (default) | **Live** pipeline if any source is a stream URL; **file-batch** if all sources are files. | Everyday use. |
| `--mode batch` | **File-batch** flow (`main.py`): per-file detect/track/embed → offline reconcile → re-render. Stream URLs are rejected. | File inputs / the behavioural-equivalence regression gate. |

Both paths now settle identity with the **same** `reconcile_tracklets` (non-causal, whole-gallery). The record-then-batch RTSP stopgap was removed — the live pipeline handles streams directly and reconciles on stop.

---

## 4. What's been done (all committed on `research`)

- **ReID model swap (validated):** `osnet_x1_0_msmt17` → **`osnet_ain_x1_0`** (multi-source domain-generalization checkpoint), 512-d, same preprocessing. `src/reid/extractor.py` (+ `torch.load(weights_only=False)` and `state_dict`/`module.` unwrap for the checkpoint format). Weight committed at `src/reid/weights/osnet_ain_x1_0.pth`.
- **Threshold tuning — tried & REVERTED:** raising identity/verifier thresholds + tightening occlusion hurt accuracy (~97%→~89%); reverted to `identity.threshold 0.63`, `min_score_gap 0.03`, verifier `0.55/0.05`, `max_occlusion_ratio 0.5`. Documented in `config.yaml` comments.
- **v5 Stage 0 (done, approved):** `src/live/` scaffold — `capabilities.py` (device/NVDEC detection, no hardcoded `.cuda()`), `frame.py` (Frame carrier; `ts` stamped once, never regenerated), `decode_backend.py` (`DecodeBackend` ABC + `CPUDecodeBackend` + `NVDECBackend` stub). `live:` config block. `run.mode` routing + `--mode` CLI flag in `main.py`. File-batch regression **passed/approved**.
- **v5 Stage 1 (done, backbone):** full threaded skeleton — `queues.py` (NewestSlot, DropOldestQueue), `capture.py` (per-cam decode + reconnect), `scheduler.py` (freshness batch), `inference.py` (per-cam detector + embedder, bs=1), `identity_stage.py` (evidence-gate + minimal match-or-mint), `render.py` (draw), `writer.py` (x264/mp4v, wall-clock pacing, offline overlay), `pipeline.py` (orchestrator, warm-up, race-free shutdown). Validated locally (file-as-stream → valid annotated MP4, clean finalize). **Real-time, no recording — matches the handoff.**
- **Ops:** `deploy.sh` (rsync alt to git), README §2b server steps (may have been reverted — re-add if missing), requirements torch note.

### src/live/ file map
`capabilities.py` device/NVDEC probe · `frame.py` Frame · `decode_backend.py` decode ABC (CPU impl; NVDEC stub) · `queues.py` NewestSlot + DropOldestQueue · `capture.py` CaptureThread · `scheduler.py` BatchScheduler · `inference.py` InferenceStage · `identity_stage.py` IdentityStage · `render.py` RenderStage · `writer.py` WriterStage · `pipeline.py` LivePipeline.

---

## 5. What's DONE / NEXT

**Done and validated on the GPU server:**
- **Stage 2 — GPU throughput.** Fair scheduler (fairness + starvation bound) + per-camera parallel inference, one ByteTrack per camera (stop-rule obeyed).
- **Stage 3 — full identity engine.** In-memory active-set (prototypes/medoids) + cold reactivation + cross-camera reciprocal-best + anti-starvation priority queue + pluggable fail-open topology veto. Gives the provisional on-screen ids.
- **Offline reconcile on the live path.** `IdentityStage` persists fresh observations to Qdrant; `RenderStage` captures processed frames + geometry; on stop `pipeline.py::_finalize_offline()` runs `reconcile_tracklets` → `render_final_videos` → `print_run_summary` (reused unchanged). Logic-tested (`tests/live/test_stage6_offline_reconcile.py`) and confirmed working on the A100.

**Remaining (optional / future):**
- **Stage 4 — GPU decode.** NVDEC backend. `cv2` here has **no `cudacodec`**; GPU-resident decode needs a server decode lib (OpenCV-CUDA build / PyNvVideoCodec / decord). Until then: CPU decode → 1 upload → GPU inference → 1 download.
- **Stage 5 — ops hardening.** CUDA OOM/context handling, disk/codec degradation (metrics + warm-up ordering already in).
- **Stage 6 — acceptance + server scripts.** `run_gpu_benchmark.py` (N-cam P95 latency, drop %), `run_24h_memory.py`; explicit multi-N test.

**Validation cadence:** logic-test **locally** (deterministic, synthetic — no GPU/video); ship to GPU (git pull) for **quality/throughput** checks — never judge reid quality on CPU.

---

## 6. Environment, deploy, run

- **Local dev:** CPU-only WSL2 (no NVIDIA GPU; `torch.cuda.is_available()==False`). `torch 2.12.1+cu130` is a CUDA build that falls back to CPU. GUI needs `libsm6 libice6 libxext6 libxrender1 libgl1` (installed) for OpenCV windows.
- **Server:** A100, reached via **SSH**. Deploy = **`git pull`** (not FileZilla). One-time: `git clone … && git checkout research && python3 -m venv .venv && pip install -r requirements.txt`. GPU auto-used (`device: auto`). `deploy.sh` (rsync) is an alt (set `DEPLOY_TARGET` in gitignored `.deploy.env`).
- **Run live (the target):**
  ```bash
  python main.py --mode live --videos "rtsp://USER:PASS@HOST:554/path" [more URLs...]
  ```
  Headless; writes `output_<cam>.mp4`; Ctrl-C to stop. **No Qdrant needed** for live (identity in-memory). N cameras = more `--videos` URLs.
- **RTSP creds:** inline in the URL, URL-encode specials (`@`→`%40`). Never commit; rotate if leaked.

## 7. Gotchas (don't re-learn the hard way)

- **Live never records the raw feed.** It reads RTSP live and, for the final re-render, holds only the *processed* frames in a transient `._live_src_<cam>.mp4` (deleted after render). The old record-then-batch RTSP stopgap has been removed.
- **Never judge quality on CPU** — it drops most frames (drop-stale) and starves identity. GPU only.
- **Live on-screen ids are provisional; correct ids come from the Ctrl-C reconcile.** Don't expect the real-time overlay to be the final answer.
- **`reconcile_tracklets` is shared by both paths** but the file-batch entry points are the regression gate: **do not modify** `process_video` / `reconcile_tracklets` / `render_final_videos` / `IdentityService` — the live path *reuses* the latter three unchanged.
- **Merge thresholds live in `identity.reconcile.*`** (shared), not `live.identity.*` (provisional on-screen only). Recalibrate `identity.reconcile.*` on a checkpoint/domain change.
- **Topology is fail-open** and currently OFF (measured transit times pruned true matches on these overlapping cameras) — appearance-only until cameras are separated.

## 8. Pointers
- Pipeline: `src/live/*.py` (orchestrator `pipeline.py`; offline finalize `pipeline.py::_finalize_offline`). Routing + modes: `main.py` (`run.mode`, `--mode`). Offline reconcile reused by both paths: `src/identity/reconcile.py` + `render_final_videos`/`print_run_summary` in `main.py`. Config: `config.yaml` (`live:` block incl. `live.reconcile` + `identity.reconcile`/`reid`). Reused: `src/detector.py`, `src/reid/extractor.py`, `src/reid/service.py`, `src/database/store.py`, `src/drawing.py`, `src/video_source.py`.
- Full staged plan (local, not in repo): `~/.claude/plans/functional-kindling-church.md`.
