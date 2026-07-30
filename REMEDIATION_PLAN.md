# Pipeline Remediation Plan

**Status:** agreed, not yet implemented
**Created:** 2026-07-30
**Scope:** detection, tracking, embedding, reconciliation, and the final rendered output

This document is the reference plan for fixing identity instability in the live RTSP →
reconciled-MP4 product. It records **what was measured**, **what was only read**, what we
deliberately decided *not* to change, and the design decisions settled while writing it.

Read Part A before proposing any threshold change. Read Part H before trusting any number.

---

## Contents

1. [Product scope](#1-product-scope)
2. [Camera inventory](#2-camera-inventory)
3. [Evidence conventions](#3-evidence-conventions)
4. [Part A — Deliberately not changing](#part-a--deliberately-not-changing)
5. [Part B — Phases](#part-b--phases)
6. [Part C — Out of scope / deferred](#part-c--out-of-scope--deferred)
7. [Part D — Verified clean](#part-d--verified-clean-do-not-re-investigate)
8. [Part E — Runs required](#part-e--runs-required)
9. [Part F — Acceptance criteria](#part-f--acceptance-criteria)
10. [Part G — Design decisions log](#part-g--design-decisions-log)
11. [Part H — Measurement reference](#part-h--measurement-reference)
12. [Part I — Historical context](#part-i--historical-context)

---

## 1. Product scope

Live RTSP in, `Ctrl-C`, out come **N reconciled MP4s — one per camera** (`output_<cam>.mp4`).

Success criteria, in priority order:

1. Identities are correct — no flicker, no swapping between people
2. Consistent **within** a camera and **across** cameras
3. Finalization (reconcile + re-render) is not slow

Non-goals: identities persistent across runs (ids are run-scoped); real-time on-screen ids;
batch mode on recorded files. `--mode live --videos foo.mp4` must keep working.

### 1.1 How the per-camera outputs form one set

`build_gid_map` is called **once** across the whole reconciled gallery, returning
`{(camera, track_id): gid}`. Each camera's render looks up its own `(name, track_id)`, so a
person walking cam_213 → cam_219 carries the **same id and colour in both files**. That is
the product claim. Several plan items exist solely to keep those four videos comparable to
each other — see #35, #44, #45.

---

## 2. Camera inventory

| Camera | Resolution | fps | Codec | Notes |
|---|---|---|---|---|
| cam_213 | 1920×1080 | 25 | H.265 | chokepoint |
| cam_224 | 2560×1440 | **15** | H.265 | co-visible with cam_219 |
| cam_206 | 1920×1080 | 25 | H.265 | |
| cam_219 | 2560×1440 | 25 | H.265 | co-visible with cam_224 |

**Co-visibility: only 224↔219.** No other pair can see one person at once, even briefly.
This is a hard physical constraint and Phase 7 exploits it.

Camera names derive from the IP's last octet, so replacing hardware changes names and the
config must follow.

---

## 3. Evidence conventions

Every finding below is tagged:

- **measured** — reproduced by a script that asserts, or quantified with numbers
- **verified** — confirmed by reading the exact code path (and quoted line)
- **read** — found by reading; mechanism is clear but impact is unquantified

Where a measurement has a small sample, the sample size is stated. Do not promote a
**read** finding to a fix without measuring first — that mistake was made four times during
this project's tuning history (Part I).

---

## Part A — Deliberately not changing

These were measured and rejected. Do not revisit without new evidence.

| Change | Why not | Evidence |
|---|---|---|
| NMS `iou: 0.60` | 0.60→0.45 removed 2 boxes of 190 on one clip, **0 of 201** on another. At the old 0.70 default there were 4 high-overlap pairs, so **0.60 already fixed duplicate boxes** | measured |
| `imgsz` 640 → 1280 | Same 5 track ids in every configuration, 2.4× the compute | measured |
| `conf` 0.4 → 0.1 + post-filter | Byte-identical results. ByteTrack's second-association stage is inert (its pool is 0.1–0.25, below the 0.4 pre-filter) but activating it changed nothing | measured |
| `same_camera_threshold` → 0.85 **as a decision** | Evidence was n=6 fragment pairs, one camera, one clip. Stays **0.90** until the Phase 9 sweep | measured, weak sample |
| `cross_camera_threshold` | No cross-camera data exists yet. Untouched until Phase 9 | — |
| Deepening `max_inference_queue` | 6 MB/frame; observed peak depth 900 ≈ 5.6 GB RAM | measured |
| Frame-drop → fragmentation fixes, unvalidated | Dropping 2 of 3 frames produced **no extra fragmentation** on available clips. The effect is real on production footage (35 tracklets → 7 identities in a prior run) but cannot be validated against footage that does not reproduce it | measured |

---

## Part B — Phases

Phases 1–3 change no output and need no new footage. Phases 4–8 each change behaviour and are
measured on the replay harness. Phase 9 is calibration. Phases 10–12 are performance and ops.

### Phase 0 — Baseline run (operator)

Qdrant up. Set `live.reconcile.keep_frames: true` and `live.metrics.log_interval_sec: 10`.
Four cameras, 2–3 minutes, people crossing between views, **one** `Ctrl-C`:

```
python main.py --mode live --videos rtsp://...213/1/1 rtsp://...224/ch01/0 \
    rtsp://...206/1/1 rtsp://...219/ch01/0 2>&1 | tee run1.log
```

Keep `run1.log` and the four `._live_src_cam_*.mp4`. Then `grep -c hevc run1.log`.

Yields: per-camera drop rate, fragmentation ratio, first cross-camera data, replayable
footage, and the H.265 corruption answer.

> **Check free space first** — see #65. Four clips can reach ~21 GB/hour combined, and
> `keep_frames: true` means they are not deleted.

### Phase 1 — State model + instrumentation (no output change)

| # | Item | Evidence |
|---|---|---|
| 1 | State machine first-class: `Tracklet → Candidate → merged Candidate → Resolved \| Expired unresolved` | spec §2 |
| 2 | Provisional handles (`U-07`) in a separate namespace, **internal-only pre-reconciliation**, excluded from every identity tally by construction | spec §2.3 |
| 3 | Reconcile has **zero rejection diagnostics** — 4 `log()` calls, all successes | verified |
| 4 | All gates evaluated independently, **no short-circuit**; `gate_detail` on every record with the applied threshold value | spec §5.1 |
| 5 | Gate order: ranking-independent hard constraints → `ABSOLUTE_THRESHOLD` → ranking → `TOP2_MARGIN` → `RECIPROCAL_BEST` → merge | decision D3 |
| 6 | `TOP2_MARGIN` added, **logged only** (`threshold: null`); both `margin_eligible` and `margin_all_scored`; single `basis` config key | decision D4 |
| 7 | Annotated candidate vector: `excluded_by`, `would_fail_reciprocity`, `pair_similarity_to_best`, cameras, cluster size, observations, `candidates_truncated` | decision D5 |
| 8 | Summary block + test asserting it matches recomputation from the vector | decision D5 |
| 9 | `runner_up_differs` → run-level disagreement rate (bands: <1% drop the distinction, 10–20% meaningful, ~50% hard constraints reshaping the space) | decision D6 |
| 10 | Fragmentation metrics: tracklets per identity (mean/max/distribution); observations per tracklet **per camera** | new |
| 11 | Suppressed-tracklet count and share of total observations | verified silent today |
| 12 | Cross-camera metrics: identities spanning ≥2 cameras; per camera-pair candidates vs merged vs each rejection reason | new |
| 13 | Label-free correctness: two non-overlapping boxes in one frame sharing a positive id; id changes along one `(camera, track_id)` (must be 0) | new |
| 14 | Timing: reconcile + re-render wall time **per camera** | product constraint |
| 15 | `ts` is **receive time** (stamped after `cap.read()` returns), not capture time; no PTS ever read. Quantify per-camera jitter vs nominal 40/66.7 ms; probe `CAP_PROP_POS_MSEC` | verified |
| 16 | Measure 224↔219 offset **as a bound on apparent disagreement**, never as a timestamp-error estimate — see decision D2 | decision D2 |
| 17 | Per non-co-visible pair, negative-gap distribution for high-confidence same-person transitions (prototype cosine ≥ 0.85). Report sample sizes | decision D2 |
| 18 | Fold the verification scripts used to produce Part H into `tests/` as a calibration harness, so the numbers regenerate per clip | housekeeping |

**Gate:** existing 8 test files pass; outputs byte-identical.

### Phase 2 — Store transport + query hygiene

| # | Item | Evidence |
|---|---|---|
| 19 | `store.prefer_grpc` (default `false`) + `store.grpc_port: 6334`. `docker-compose.yml` **already maps 6334** — no infra change | verified |
| 20 | `_gather_tracklets` scrolls **unfiltered** with `with_vectors=True`, discarding other runs in Python — cost grows with every run forever. Add server-side `scroll_filter` on `run_id` here, in `build_gid_map`, and in `print_run_summary` | verified; `scroll_filter` exists, unused |
| 21 | Log active transport at startup; keep embedded `path=` mode working (no gRPC in-process) | required for tests |
| 22 | Store never validates an existing collection's dim/metric on startup | read |

**Gate:** bit-identical assignments over REST and gRPC on the same clip; finalization wall
time reported for both.

### Phase 3 — Offline replayability

| # | Item | Evidence |
|---|---|---|
| 23 | Reconcile runs end-to-end from a persisted score log with **no model in memory**, reproducing assignments bit-for-bit at identical thresholds | spec §6.2 |
| 24 | Replay harness: frozen clips fed with `NewestSlot` and the 2-deep queue decimation bypassed, so runs are deterministic | prerequisite |

This turns Phase 9's sweeps into an interactive loop instead of a re-inference job.

### Phase 4 — Determinate reconcile fixes

| # | Item | Evidence |
|---|---|---|
| 25 | Reconcile stamps **no identity at all** below 2 tracklets (`if len(keys) < 2: return {}`) — the whole video renders as bare `ID <n>` | **measured**: 1 tracklet, 5 observations, zero ids written |
| 26 | Reconcile **never reads `ts`**; spans use per-camera frame indices, which are not comparable across 15/25 fps. Switch to `ts` with frame-index fallback | verified by grep |
| 27 | Phase 2 selects its threshold from a **stale `roots` snapshot** — a camera-sharing pair can merge at 0.63 instead of 0.90 | verified: `union` updates `members`, not `roots` |

`next_gid` cross-run collision **dropped from scope** — harmless with run-scoped ids, since
every consumer filters by `run_id`.

**Gate:** new regression tests; false-merge counter (#13) does not increase.

### Phase 5 — Input integrity

| # | Item | Evidence |
|---|---|---|
| 28 | **No RTSP transport or timeout options anywhere.** Set `OPENCV_FFMPEG_CAPTURE_OPTIONS` with `rtsp_transport;tcp` and a socket timeout | **measured**: 294/682 and 207/1573 broken H.265 references in project recordings; grep confirms none set |
| 29 | Without a socket timeout `cap.read()` can block indefinitely; the capture thread never re-checks `stop_event`, so `Ctrl-C` cannot stop that camera | read |
| 30 | Move credentials out of command-line URLs (visible in `ps` and shell history) into `.env` | observed |

**Rationale:** H.265 reference loss produces smeared frames that feed both the detector and
the ReID crop, so a corrupted crop poisons the tracklet prototype. Packet loss is random,
which no threshold explains — this is the leading hypothesis for random identity behaviour.

**Gate:** `grep -c hevc run2.log` vs run1; same-person cosine spread tightens or holds.

### Phase 6 — Renderer

| # | Item | Evidence |
|---|---|---|
| 31 | Renderer becomes a **pure function of `(state, frame)`** — no state, no promotion. Both purity tests: frame *N* direct vs sequential `0..N` byte-identical; forward-then-backward label stability | spec §3 |
| 32 | Unresolved handles: distinct colour, dashed box, **no ID number**, never in a tally | spec §2.3 |
| 33 | Suppressed tracklets currently render as bare `ID <n>`, reading as the identity vanishing — replaced by #32's treatment | verified |
| 34 | **REID confidences.** `render_final_videos` builds `SimpleNamespace` without `reid_score`, so the final MP4 shows no confidence at all. Add per-tracklet **fit** (prototype vs final cluster prototype) and **margin** (gap to nearest other cluster) → `ID 7 (0.94 / +0.21)`. Label as reconcile-scale — **not** comparable to live-engine scores | verified |
| 35 | Palette has only **8 colours** (`id % 8`). Across four videos, where the check is "did this person keep their colour", collisions read as false merges that did not happen | verified |
| 36 | Labels clipped for boxes with `y1 < ~30` — people entering at the top have no visible id | read |
| 37 | `as-known-at-time` playback mode reading the same state log | spec §4 |

### Phase 7 — Cross-camera simultaneity veto

| # | Item | Evidence |
|---|---|---|
| 38 | `conflict()` skips cross-camera pairs entirely, so tracklets overlapping in time in non-co-visible cameras merge on appearance alone. This is the "id switches to another person" symptom | verified: `if a[0] != b[0]: continue` |

```yaml
identity:
  reconcile:
    covisibility:
      enabled: false              # fail-open default
      default_tolerance_sec: 1.0
      pairs:
        - [cam_224, cam_219, covisible]   # never vetoed
        - [cam_213, cam_206, 1.0]
        - [cam_213, cam_224, 1.0]
        - [cam_213, cam_219, 1.0]
        - [cam_206, cam_224, 1.0]
        - [cam_206, cam_219, 1.0]
```

Semantics: `covisible` = never vetoed. A number = veto when temporal overlap **exceeds** that
many seconds. **A pair absent from the list is unconstrained** — silence loses protection
rather than splitting a real person. Startup enumerates every camera pair present in the run
and warns on any absent, so changing camera arrangement is a config edit with a visible
checklist, never a code change.

Implemented inside `conflict()` so it inherits transitive cluster safety. A single-observation
tracklet has a zero-length span and can never be vetoed — consistent fail-open, surfaced in
metrics so it is not mistaken for the veto working.

> This is a **simultaneity** veto, not the minimum-transit-time rule that was reverted for
> pruning true matches (`topology_pruned=508`). It never constrains transition *speed*, so it
> cannot repeat that failure.

**Gate:** false-merge counter drops; identity count and fragmentation unchanged.

### Phase 8 — Feature tap

| # | Item | Evidence |
|---|---|---|
| 39 | `fc = Sequential(Linear, BatchNorm1d, ReLU)` and eval-mode `forward` returns `self.fc(v)` — the shipped embedding is **post-ReLU**, confined to the non-negative orthant. Add post-BN behind a config flag; record the tap in the run config; refuse to compare score logs across taps | **measured** |

Measured effect (bank scoring, 14 proven-distinct pairs): separation margin **+0.047 → +0.071**,
stranger ceiling **0.828 → 0.772**. But at matched false-accept rates recall is identical
(89% / 0% either way). **Buys calibration robustness, not accuracy** on available footage.

Untested hypothesis: the benefit may be larger cross-camera, where post-ReLU compression
pushes same-person scores toward the stranger band.

> **Changing the tap voids every threshold in Part H.**

### Phase 9 — Calibration sweeps (needs Phase 0 + 3)

| # | Item |
|---|---|
| 40 | `same_camera_threshold` — sweep 0.75–0.95 |
| 41 | `cross_camera_threshold` — currently uncalibrated. With impossible competitors removed by Phase 7, a real match may win at a *higher* bar. A prediction to test, not a change to make |
| 42 | `min_tracklet_observations` — interacts with #33 |
| 43 | `TOP2_MARGIN` — four questions: does it reject anything reciprocal-best doesn't; disagreement rate vs the 10% bar; raw vs normalised margin; which `basis` |
| 44 | Consensus vs `max(proto, exemplar)` scoring — measured margin +0.070 vs +0.047; modest |

**A high `TOP2_MARGIN` failure rate is a diagnostic, not evidence of under-merging.** The
discriminator is `pair_similarity_to_best`:

| `pair_similarity_to_best` | Cause | Remedy |
|---|---|---|
| ~0.95 between tied candidates | one person fragmented into two clusters | lower `same_camera_threshold` |
| ~0.55 | genuinely ambiguous embeddings (uniforms, low light, bad crops) | feature / crop-quality work, not merging |
| many candidates above the floor | over-permissive candidate generation | raise the floor |

Deliverable: **curves, not values.**

### Phase 10 — Per-camera timing

| # | Item | Evidence |
|---|---|---|
| 45 | Single global `output.fps_default: 20` applied to all four cameras ([pipeline.py:203](src/live/pipeline.py#L203), [:430](src/live/pipeline.py#L430)) → four videos with four different wrong time scales, which **blocks visual cross-camera verification** | verified |
| 46 | Output MP4 timeline **5.5× too fast** (157 frames @20fps for 42.9 s of content). `WriterStage` has correct wall-clock pacing but is never constructed in reconcile mode | **measured** |
| 47 | `reid.interval` counts processed frames → cam_224 at 15 fps accumulates ~40% fewer observations per second, giving the weakest prototypes to the hardest camera pair. Convert to seconds | arithmetic |
| 48 | `max_frame_staleness_ms: 100` and `track_buffer: 30` are absolute → 2.5 vs 1.5 frame periods across cameras | read |
| 49 | `detector.per_camera` is `{}` — the intended lever for heterogeneous cameras, unused | read |

### Phase 11 — Throughput (conditional on run1 showing fragmentation)

| # | Item | Evidence |
|---|---|---|
| 50 | `_embed_lock` wraps all of `TrackEmbedder.process` — cropping, a float64 Laplacian, an O(N²) occlusion loop, all preprocessing — not just the forward pass, contradicting its own docstring | verified |
| 51 | Drops dominated by `max_inference_queue: 2` (629 of 637 batches on a file run; `slot_drop` was 1) | **measured** |
| 52 | `warmup_embeddings: 3` fires on consecutive frames → ~1 effective view for short tracklets, exactly those needing to clear 0.90. Spread it | read |
| 53 | 251 Mpx/s of CPU-only H.265 decode (cam_219 alone 92.2); NVDEC is a stub (`NVDEC_IMPLEMENTED = False`) | arithmetic |
| 54 | ByteTrack's Kalman filter advances one step per `update` regardless of elapsed time — no `dt`. Under load-dependent dropping the predicted box is wrong by however long the gap really was | verified |

### Phase 12 — Robustness

| # | Item | Evidence |
|---|---|---|
| 55 | No exception guard in any worker `run()`; no watchdog. A dead `InferenceStage` runs forever writing `CAMERA OFFLINE` frames | verified by grep |
| 56 | ReID batch unbounded in number of people; no chunking → OOM risk on a weaker GPU | verified |
| 57 | Camera death permanent after ~15 s of retries; all-dead ends the run, so a 20 s switch reboot kills a session | read |
| 58 | Clips deleted in a `finally` even when the render failed, destroying the only copy of the footage | read |
| 59 | Qdrant down → `_build_store` returns `None` → reconcile **silently disabled** → live-annotated output instead. Make it fail loudly | verified |
| 60 | `from main import` inside `_finalize_offline` → fails outside the project root, returning without rendering | observed |
| 61 | Threads started outside `try/finally`; identity never drains on shutdown; render's drain races identity | read |
| 62 | Clip writer latches frame size from the first frame → a mid-run resolution change desyncs annotations from the clip permanently | read |
| 63 | `output.codec: h264` ignored on the product path (mp4v hardcoded); metrics always report `written=0` in reconcile mode | verified in a real run |
| 64 | A camera with no annotations produces **no video and no message** (`if not annos: return`). With four cameras this is easy to miss. Log loudly and record per-camera outcome in the summary | verified |
| 65 | Clip disk and annotation RAM are unbounded and scale with camera count. Measured **51.0 KB/frame** at 1920×1080 mp4v → see table below. No cap, no rotation, no free-space check | **measured** |
| 66 | `_g("inference","pose_ensemble", True)` defaults **True** — deleting one config line enables the duplicate-box generator on the live path. Change the default to `False` | verified |
| 67 | `source.videos` points at non-existent `placeholder_video/`, so bare `python main.py` fails | observed |

Clip growth, assuming no frames dropped:

| Camera | Rate | Per hour |
|---|---|---|
| cam_213 1920×1080@25 | ~1.2 MB/s | 4.4 GB |
| cam_224 2560×1440@15 | ~1.3 MB/s | 4.7 GB |
| cam_206 1920×1080@25 | ~1.2 MB/s | 4.4 GB |
| cam_219 2560×1440@25 | ~2.2 MB/s | 7.8 GB |
| **total** | **~5.9 MB/s** | **~21 GB/hour** |

---

## Part C — Out of scope / deferred

### Live `IdentityEngine` — off the product path

**Verified end to end:** with `live.reconcile.enabled: true` (the default and current product
mode) the live engine's ids are **computed and discarded**. `_observation_payload` stores no
`reid_id`, `RenderStage._capture` records only `x1,y1,x2,y2,track_id,confidence`, the final
MP4's ids come solely from `build_gid_map` reading what reconcile stamped, and the live
pipeline is headless. So `live.identity.*` thresholds have **no effect on the deliverable**.

Deferred defects (all **measured** by reproduction): bank poisoning driving a wrong person to
cosine 1.000 via unguarded `_reinforce`; a single bad exemplar reaching 1.000 because
`score()` takes `max(prototype, best_exemplar)`; the two-lane leak letting a stranger take
another person's id at cosine 0.633 while incrementing *both* `recam_rej_below` and `linked`;
co-active window expiry letting a look-alike inherit a gid; cross-camera co-presence not
vetoed (`_gid_coactive` skips other cameras); LRU eviction dropping a gid an active track
still holds; `O(gids × tracks)` matching with per-call `np.stack`.

**Recommendation: disable the engine in reconcile mode** to reclaim the identity thread.

### Batch path — unused

Pose-ensemble duplicate boxes (**measured**: 0 overlapping pairs without it, 2 with it, fires
on 5/40 frames, creates synthetic `track_id + 100000·k` ids whose "primary" flips between
people); `_same_camera_overlap` never returning `True` (**measured**); `_maybe_reassign_track`
dead because `_commit` adds the embedding to the bank first, forcing `current_score` to
exactly 1.0; the verifier's unbounded `observation_count` drift (accept bar 0.733 at obs=0 →
0.543 at 300 → 0.466 at 3000, against a ~0.57 stranger ceiling) and prob-gap saturation (at
obs=300, cosine 0.90 vs 0.85 gives a 0.0019 gap, below `min_prob_gap: 0.05`, so a clean winner
is rejected and a duplicate minted); unfiltered `store.search`; asymmetric
`_reciprocal_best_ok`; ~300 lines of unreachable `_assign_online`.

### Low value

`crop_saver` frame-index throttle bug and unbounded dict (disabled); `http://` static files
misclassified as streams so EOF triggers replay; empty `models/fastreid`.

---

## Part D — Verified clean, do not re-investigate

**Entire embedding path.** Checkpoint loads 550/552 keys with only the discarded classifier
head unloaded — no layer left at random init. `model.training is False`. Output `(N, 512)`,
unit norms. Determinism: same batch twice, max abs diff `0.00e+00`. Batch order preserved
(`argmax(single_i · batch_j) = [0 1 2 3 4]`). Batch invariance `6.26e-07`. BGR→RGB present and
load-bearing (cosine 0.903 if swapped). Resize is `(W=128, H=256)`, not transposed. ImageNet
normalisation applied after the channel permute. Empty crop raises `ValueError` rather than
injecting a vector. Cache row aliasing is memory retention only — no cross-track contamination.

**Qdrant.** `Distance.COSINE`; `score` **is** similarity, higher = closer; round-trip exact
(numpy 0.5846 → qdrant 0.5846).

**Reconcile `_prototype`** math correct, norm 1.0. `_prototype([])` raises but is unreachable
(line 124 filters empty vector lists).

**`InterruptGuard`** soundly engineered — the `pthread_sigmask` + `sigwait` approach with a
Python handler as backup correctly solves the blocked-signal problem. No defect found.

**Clip↔annotation alignment invariant** held in both end-to-end runs (annotations == clip
frames == output frames).

**Per-camera tracker isolation** correct: one `PersonDetector` (hence one `YOLO`, hence one
ByteTrack) per camera, one worker per camera, no cross-contamination possible.

**`_bin6`** correct — `int(0.7 * 10) == 7` in CPython; an earlier off-by-one claim was wrong.

**ByteTrack cannot invent boxes** — `_format_output` returns one row per activated track and
association is one-to-one. Across 120 frames the tracker returned more rows than detections in
**0** frames.

---

## Part E — Runs required

| Run | Purpose |
|---|---|
| **run1** — baseline, Phase 0 | per-camera drop rate, fragmentation ratio, H.265 answer, first cross-camera data, frozen clips. Unblocks Phases 9–11 |
| **run2** — after Phase 5, TCP transport | compare `grep -c hevc` and prototype tightness |
| **Single-person route walk** through all four cameras with rough timings | cleanest data for per-pair tolerances (#17) and `cross_camera_threshold` |
| **Crowded run**, as many people as possible | every measurement in Part H tops out at 6 people; the production regime is untested |

---

## Part F — Acceptance criteria

1. Renderer purity — frame *N* direct equals frame *N* from a sequential `0..N` pass,
   byte-identical; forward-then-backward label stability.
2. No provisional handle appears in any identity count or unique-person tally.
3. A tracklet failing *n* gates has *n* entries in `gates_failed`, proven by a fixture
   constructed to fail two gates simultaneously.
4. `gate_detail` populated for every gate on every record, passing and failing alike.
5. Reconciliation runs from a persisted score log with **no model loaded** and reproduces the
   original assignments bit-for-bit at identical thresholds.
6. Candidate↔Candidate merge exercised by a fixture where two sub-threshold tracklets from
   different cameras resolve correctly **only** when merged.
7. `as-known-at-time` mode and default mode both derive from the same state log.
8. REST and gRPC produce identical assignments on the same clip.
9. The `TOP2_MARGIN` summary block matches recomputation from the annotated candidate vector.
10. The existing 8 test files stay green throughout.

---

## Part G — Design decisions log

**D1 — Co-visibility is configuration, not code.** Changing camera arrangement means editing
the `pairs` list. Startup enumerates every pair present in the run and warns on any absent, so
a forgotten pair is visible rather than silent.

**D2 — The 224↔219 offset is a bound on *apparent disagreement*, not a timestamp-error
estimate.** It conflates: timestamp error proper (network jitter, RTSP buffering, the decode
cost gap between 92.2 and 55.3 Mpx/s), frame-rate quantisation (66.7 ms vs 40 ms grids),
load-dependent frame dropping, detection latency, ByteTrack's two-frame confirmation delay,
the crop-quality gate, `min_tracklet_observations`, and — probably the largest term —
**non-identical fields of view**, since a person can be well inside one view while genuinely
outside the other. Therefore it is a **floor** for the tolerance, never a value to shrink
toward, must not be reported as "our timestamp error is X ms", and must be re-measured after
any throughput change. Because it is partly geometric it does **not** transfer to other pairs,
which is why D7 exists.

**D3 — `RECIPROCAL_BEST` comes *after* `TOP2_MARGIN`.** `best_partner[a]` is single-valued, so
at most one candidate can pass reciprocity with any cluster. Filtering by reciprocity before
ranking leaves ≤1 candidate, no runner-up exists, and the margin gate becomes dead code.
Pre-margin constraints must be **independent of the cluster's own ranking**: temporal
conflicts, `NOT_MERGEABLE_CROSS`, and the absolute floor.

**D4 — Both margins logged, exactly one operational.** `margin_eligible` (runner-up over the
post-hard-constraint set) and `margin_all_scored` (runner-up over everything above the floor)
answer different questions — *selection safety* versus *evidence quality* — and disagree on
real inputs. A single `basis: eligible | all_scored` config key makes gating on both
inexpressible. `threshold: null` in Phase 1 means computed and logged, enforcing nothing.

**D5 — Exclusion reasons live on the candidate vector.** Rather than parallel summary fields,
each candidate carries `excluded_by` (enumerated, ranking-independent) plus a
`would_fail_reciprocity` **annotation** that excludes nothing. Under D3 reciprocity cannot be
an exclusion reason, but the annotation preserves the diagnostic: if near-ties are frequent and
the runner-up almost always carries `would_fail_reciprocity: true`, reciprocal-best is already
resolving them and `TOP2_MARGIN` adds nothing.

Enumerated `excluded_by` values: `TEMPORAL_CONFLICT_SAME_CAMERA`,
`TEMPORAL_CONFLICT_CROSS_CAMERA`, `NOT_MERGEABLE_CROSS`, `BELOW_ABSOLUTE_THRESHOLD`, `null`.

**D6 — Margin disagreement is itself a metric.** `runner_up_differs` aggregated to a rate,
read against <1% (drop the distinction) / 10–20% (meaningful design choice) / ~50% (hard
constraints fundamentally reshaping the candidate space). In this system the only new
pre-margin constraint is the simultaneity veto, so the disagreement rate doubles as a direct
measure of that veto's impact on candidate generation.

**D7 — Simultaneity tolerances are per-camera-pair from the outset**, even though all default
to 1.0 s. Small cost now; the clean path if pairs prove to have materially different timing
characteristics. Overlap is symmetric so pairs are undirected. Whether tolerance should scale
with the pair's coarser frame period is left to the Phase 1 metrics.

**D8 — Ids are run-scoped.** No cross-run gallery. The spec's gallery states collapse to
cluster membership; `next_gid` collisions become harmless; no gallery-hygiene workstream. The
store still accumulates across runs, which is why #20 (query hygiene) matters.

**D9 — Thresholds are hypotheses until swept.** `same_camera_threshold` stays at 0.90.
Deliverables from Phase 9 are curves, not values.

---

## Part H — Measurement reference

All numbers below were produced on `register_file.avi` (2560×1440, **0 H.265 decode errors**),
which matches production camera resolution. **Sample: 6 people, one camera, one clip.** The
"different person" set uses only tracks that **co-occur in the same frame**, since a person
cannot be two simultaneous detections — earlier figures that included non-co-occurring track
pairs were contaminated by fragmentation and are not reproduced here.

### H.1 Raw crop-to-crop cosine

| Tap | same mean | same p5 | other mean | other p95 | other MAX | margin |
|---|---|---|---|---|---|---|
| post-ReLU (ships) | 0.885 | 0.710 | 0.552 | 0.640 | 0.819 | +0.070 |
| post-BN | 0.848 | 0.601 | 0.373 | 0.509 | 0.770 | +0.093 |

### H.2 Bank scoring — `max(prototype, best_exemplar)`

This is what decides same-camera reacquisition and cross-camera links.

| Tap | same p5 | other p95 | other MAX | margin |
|---|---|---|---|---|
| post-ReLU (ships) | 0.768 | 0.722 | **0.828** | +0.047 |
| post-BN | 0.696 | 0.625 | 0.772 | +0.071 |

Operating points, post-ReLU: `0.60 → 100% correct / 33.4% wrong`;
`0.70 → 99.5% / 5.3%`; `0.80 → 93.1% / 1.4%`; `0.85 → 88.7% / 0.0%`; `0.90 → 80.3% / 0.0%`.

**Different people reach 0.828 on production-resolution footage.** At matched false-accept
rates the two taps give identical recall (89% / 0%).

### H.3 Scoring function comparison

| Scoring | margin | other MAX |
|---|---|---|
| `max(proto, exemplar)` (ships) | +0.047 | 0.828 |
| consensus (mean of top half) | **+0.070** | 0.793 |
| prototype only | +0.046 | 0.828 |

### H.4 Reconcile — prototype vs prototype (what Phase 1 of reconcile compares)

Same-person fragments simulated as one track split into two time-disjoint halves.

```
SAME-person fragments : mean 0.951   min 0.851   (n=6)
DIFFERENT people      : mean 0.613   p95 0.656   MAX 0.661   (n=12 pairs)

clean separation window: (0.661, 0.851)
configured same_camera_threshold = 0.90  ->  OUTSIDE that window

thr 0.85: genuine fragments merge 100.0%
thr 0.90: genuine fragments merge  83.3%   <- 1 of 6 fails (0.851)
```

### H.5 Feature space

| Tap | min | exact-zero dims | negative dims |
|---|---|---|---|
| post-ReLU (ships) | +0.0000 | 21.0% | 0% |
| post-BN | −0.0778 | 0% | 21.0% |

Post-ReLU is confined to the non-negative orthant, so cosine of any two vectors is ≥ 0.

### H.6 Detection / tracking (50 consecutive frames, 2560×1440)

| Config | distinct ids | dets | mean track len | ms/frame (CPU) |
|---|---|---|---|---|
| imgsz 640, conf .40, no drop | 5 | 183 | 36.6 | 74 |
| imgsz 640, conf .10 + post .40, no drop | 5 | 183 | 36.6 | 52 |
| imgsz 1280, conf .40, no drop | 5 | 191 | 38.2 | 175 |
| imgsz 640, conf .40, 2-of-3 dropped | 5 | 62 | 12.4 | 55 |
| imgsz 1280, conf .10 + post .40, 2-of-3 dropped | 5 | 64 | 12.8 | 145 |

### H.7 NMS

| iou | boxes (clip A) | pairs IoU≥0.5 | boxes (clip B) | pairs IoU≥0.5 |
|---|---|---|---|---|
| 0.70 (old default) | 193 | 4 | 205 | 4 |
| **0.60 (current)** | **190** | **1** | **201** | **0** |
| 0.45 | 188 | 0 | 201 | 0 |

### H.8 Quality gate on real footage

Rejected **1 crop of 214** (reason `occluded`). `bbox_quality_scalar` on accepted crops: mean
0.409, min 0.194, max 0.699 — never near 1.0, because its brightness term penalises CCTV
footage uniformly, so the aggregate weight carries little identity-relevant signal.

The occlusion gate is **inverted**: it divides intersection by the detection's *own* area, so
a person fully contained inside a nearer person's box scores exactly 1.000 and is rejected
every frame, while the occluder — whose crop genuinely holds two bodies — scores 0.246 and
passes. Mechanism verified; measured impact on this footage is only 0.5%, so it needs crowded
footage to size.

### H.9 Store transport

One 512-d vector: **2048 B binary (gRPC) vs 11265 B JSON (REST) — 5.50×.** At 50k
observations, 102 MB vs 563 MB, plus 0.114 ms/vector of `json.loads` (5.7 s).

### H.10 End-to-end product path

| Config | captured | rendered | discarded | tracklets | identities | aligned |
|---|---|---|---|---|---|---|
| `max_inference_queue: 2` (ships) | 1573 | 10 | 99.4% | 3 | 3 | yes |
| deep queue | 1060 | 157 | 85.2% | 2 | 2 | yes |

Drops occur at the **inference queue** (`infer_q` 629 dropped, peak 900), not the capture slot
(`slot_drop` 1). Both runs used a file read at ~180 fps, so these rates are **not**
representative of A100-on-RTSP — they locate the bottleneck, nothing more.

Output timeline: 157 frames at 20 fps = 7.85 s for 42.9 s of content = **5.5× too fast**.

---

## Part I — Historical context

Git history, parsed from every committed `config.yaml`:

- The `live.reconcile` key **did not exist** before commit `7dcf93d775` (2026-07-28), and
  `LivePipeline` defaults it to **`False`** when absent. So **every live run before that date
  produced live-annotated output, where the engine's ids *were* the product.**
- Since 2026-07-28 it has been `true` in every commit; never explicitly `false`.

Consequences for the tuning record:

- The 2026-07-24 threshold work (`same_camera_threshold` 0.90→0.85, `cross_camera_threshold`
  0.63→0.60, the "A100 calibration" commits) was calibrated for a mode that is **no longer
  shipped**.
- The later 0.85→0.70 change (2026-07-29) landed **after** reconcile was enabled, so it
  **could not have affected the deliverable at all**.
- The `recam_rej_below` counter that motivated those reductions is **corrupted by the two-lane
  leak** — the same resolve increments both `recam_rej_below` and `linked`, so it counted
  resolves that had already succeeded.

Four prior "tighten the knobs" attempts were reverted for hurting accuracy. The lesson encoded
throughout this plan: **instrument first, measure on a frozen clip, and treat every threshold
as a hypothesis.** During the audit that produced this document, four separate hypotheses
derived from reading code were falsified by measurement — including two about duplicate boxes
and one about frame dropping causing fragmentation. Assume the same failure rate for anything
tagged **read** rather than **measured**.
