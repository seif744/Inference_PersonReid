# Pipeline Remediation Plan

**Status:** Phase 1 largely landed. One production run captured and analysed.
**Created:** 2026-07-30 · **Last updated:** 2026-07-30 (after run `20260730_093723`)
**Scope:** detection, tracking, embedding, reconciliation, and the final rendered output

This document is the reference plan for fixing identity instability in the live RTSP →
reconciled-MP4 product. It records **what was measured**, **what was only read**, what we
deliberately decided *not* to change, and the design decisions settled while writing it.

Read Part A before proposing any threshold change. Read Part H before trusting any number.

---

## 0. START HERE — current state and next action

### 0.1 The next thing to do

**Capture run2 and diff it against run1 (`20260730_093723`).** Two changes are staged
in the working tree and neither has ever run on real footage:

| Staged change | What it should move |
|---|---|
| **#40** — `same_camera_threshold` is now per-camera; cam_213 and cam_224 at **0.80**, cam_206 and cam_219 left at 0.90 | orphans (33/89 today) and eligible-set size in cam_213 / cam_224 |
| **detector `yolo11n.pt` → `yolo11m.pt`** (operator's call, 2026-07-30) | detection recall, and tracklet COUNT per camera |

Then diff sections 1, 2 and 5 of the analyser against [J.10](#j10-full-analyser-output-run-20260730_093723):

```bash
python tests/calibration/analyze_decision_log.py logs/reconcile_decisions_<run_id>.jsonl
```

> **These two changes confound each other**, because they attack the same symptom from
> opposite ends: yolo11m creates *fewer* fragments (measured: it holds one 150-frame
> track where yolo11n splits the same person into 64 + 37 frames — see [H.11](#h11-detector-capacity-yolo11n-vs-yolo11m)),
> while #40 makes the fragments that remain *mergeable*. Both reduce the orphan count, so
> one combined run cannot attribute it.
>
> They do leave **different fingerprints**, which is how to read a combined run:
> yolo11m moves **tracklets per camera** (J.9's 61 / 11 / 7 / 18); #40 moves
> **eligible-per-subject and orphans** (J.10 sections 2 and 5) at an unchanged tracklet
> count. If you want clean attribution instead, flip `detector.model` back to
> `yolo11n.pt` for run2 (one line) and take yolo11m in run3.

After the run, in order: **#45a** (reconcile compares prototype *means*, which blurs
front/back into a vector matching neither view — the structural cause behind cam_213;
#40 treats only the symptom), then **#44** (`RECIPROCAL_BEST` rejects 31% of decisions
at up to 0.905, but fewer orphans will change that picture).

**Method, non-negotiable:** change ONE thing, re-run, and diff the analyser output.
Four earlier tuning attempts were reverted without learning anything — see Part I.

### 0.2 What has landed

| Commit | What |
|---|---|
| `182c677` | Phase 1: `src/identity/decision_log.py`, reconcile instrumented additively, config keys, `LivePipeline` wiring, `tests/calibration/` (8 scripts) |
| `e05c476` | **J.5 fix:** finalization survives a failed `print`. Two production runs had been lost to it |
| `286e06a` + `c977f57` | `tests/calibration/analyze_decision_log.py` and a fix to its inverted band report |
| `5598d31` | Part J.6: the per-camera finding from the first real run |
| *working tree* | **#40**: `resolve_same_camera_thresholds` + `strictest_same_camera_bar` in `reconcile.py`, `identity.reconcile.per_camera` in config, both call sites wired, `tests/live/test_per_camera_same_camera_bar.py` (36 checks) |
| *working tree* | `detector.model: yolo11m.pt`, `tests/calibration/compare_detector_models.py`, `_common.DETECT_WEIGHTS` now read from config instead of hardcoded |

Test state: **11 test files pass** (`python tests/run_all.py`), including
`test_phase1_decision_log.py` (28 checks), `test_shutdown_reaches_reconcile.py` (14)
and `test_per_camera_same_camera_bar.py` (36).

### 0.3 Verify state on a fresh machine

```bash
python tests/run_all.py                                    # expect 11 files pass
python tests/calibration/verify_embedding_contract.py      # expect PASS
python tests/calibration/characterize_known_defects.py     # expect 9/9 still PRESENT
python tests/calibration/analyze_decision_log.py <log>     # re-derive J.6
python tests/calibration/compare_detector_models.py        # re-derive H.11
```

Artefacts from run `20260730_093723` live on the A6000 at
`~/seifer_work/Inference_PersonReid`: `run1.log`,
`logs/reconcile_decisions_20260730_093723.jsonl` (286 decisions), and the four
`._live_src_cam_*.mp4` frozen clips. **Those clips are the replay corpus** — they
reproduce every symptom and nothing local does.

### 0.4 Still unanswered

- Does raising `imgsz` or lowering `conf` recover the people cam_206 misses at the start
  of a clip? Measured as no-ops on other footage **with yolo11n**; untested on yolo11m,
  and **untested on cam_206's own clip**, which is a crowded room with a table and is a
  different problem.
- `cross_camera_threshold` remains uncalibrated. Do not touch it before run2.
- **Does yolo11m keep up on four live streams?** It buys real fragmentation reduction on
  a file (H.11) at 2.0× the CPU cost. Whether that cost shows up as dropped frames — the
  one thing that would make it a net loss — is only visible in run2's per-camera
  `dropped%` and `infer_q dropped`. J.1 declared Phase 11 dead *at yolo11n's cost*.
- Whether cam_219 also needs a lowered same-camera bar. It orphans 4 of 6 subjects, but
  has only 7 tracklets at mean 264 observations, so there is little fragmentation to
  repair. Left at 0.90 deliberately, to keep run2's comparison readable.

### 0.5 Reading order for a fresh session

Section 0 (this) → **Part A** (what not to retry, with the evidence) → **Part J** (field
data from the real run) → **Part B Phase 9** (the calibration items) → Part G (design
decisions and why) → Part H (measurements, with their caveats).

Parts C and D exist to stop rediscovery: C is deferred/out-of-scope defects, D is what
was verified clean and should not be re-audited.

---

## Contents

0. [START HERE — current state and next action](#0-start-here--current-state-and-next-action)
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
12. [Part J — Field results, 2026-07-30](#part-j--field-results-2026-07-30-a6000-4-cameras-live-rtsp)
13. [Part I — Historical context](#part-i--historical-context)

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
| Frame-drop → fragmentation fixes | **CONFIRMED DEAD on production hardware (J.1):** real drop rate is 0.5–6.5%, not the 85–99% measured on CPU-with-file. Phase 11 solves a problem that does not exist at this crowd size | measured, field |
| `TOP2_MARGIN` as a **gate** | Accepted and rejected margin distributions are near-identical on 163 real decisions (median 0.0224 vs 0.0186, p5 0.0017 vs 0.0016). A gate would be near-random. Compute and log it; never enforce it. See J.6 | measured, field |
| A **global** `same_camera_threshold` of any value | The per-subject boundaries overlap across cameras (J.6): p95 of "top different" = 0.816 exceeds p5 of "worst same" = 0.719. No single number works. **Per-camera bars landed 2026-07-30 (#40)** — so do not "fix" this by picking a better global number; tune the per-camera entries | measured, field |
| `yolo11n` → `yolo11m` as a **no-op** | Superseded 2026-07-30. Part A previously implied detector changes buy nothing, on the strength of `imgsz`/`conf` results. Capacity is a different lever and it **does** move fragmentation: yolo11m holds one 150-frame track where yolo11n splits the same person into 64 + 37 (H.11). Shipped as a measurement; the throughput half is still unmeasured | measured |

---

## Part B — Phases

Phases 1–3 change no output and need no new footage. Phases 4–8 each change behaviour and are
measured on the replay harness. Phase 9 is calibration. Phases 10–12 are performance and ops.

### Phase 0 — Baseline run (operator)

Qdrant up. Set `live.reconcile.keep_frames: true` and `live.metrics.log_interval_sec: 10`.
Four cameras, 2–3 minutes, people crossing between views, **one** interrupt:

```bash
# NOT `| tee`. See the warning below -- that pipeline cost two complete runs.
python main.py --mode live --videos rtsp://...213/1/1 rtsp://...224/ch01/0 \
    rtsp://...206/1/1 rtsp://...219/ch01/0 > run1.log 2>&1 &
echo $! > run.pid
# ...let it run, then ONCE:
kill -INT $(cat run.pid)
# watch it finish WITHOUT holding the terminal that owns the process:
tail -f run1.log        # Ctrl-C here is safe; it only stops tail
```

Keep `run1.log` and the four `._live_src_cam_*.mp4`. Then `grep -c hevc run1.log`.

> **Never launch this with `| tee`.** Ctrl-C reaches every process in the foreground
> group, `tee` dies first, and every subsequent `print` in the pipeline raises
> `BrokenPipeError` — which used to abandon reconcile entirely (J.5). That fix has
> landed (`e05c476`), so a broken stdout no longer costs the ids, but redirect-and-signal
> is still the correct habit: it keeps finalization off the terminal's process group, so
> a dropped SSH session or a closed window cannot interrupt it either. **Two production
> runs were lost to `| tee`, and a third to running `tail -f` and `kill` in the same
> command block** — the `tail` held the foreground and the interrupt went to the wrong
> process. Send the signal by pid, then follow the log separately.

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
| 18 | ~~Fold the verification scripts used to produce Part H into `tests/` as a calibration harness~~ — **DONE 2026-07-30**, see [`tests/calibration/`](tests/calibration/). Building it immediately caught two methodological errors in Part H (see the note there) | housekeeping |

**Gate:** existing 8 test files pass; outputs byte-identical.

#### Phase 1 progress — 2026-07-30

**Landed.** `src/identity/decision_log.py` (new: gates, annotated candidate vector,
both margin variants, aggregates, JSONL writer, summary-consistency verifier) and
`reconcile.py` instrumented additively. Config under `identity.reconcile`:
`decision_log`, `top2_margin.threshold` (null = inert), `top2_margin.basis`. Wired
into `LivePipeline._decision_log_kwargs`, fail-soft so a bad log config never costs
the run's ids. Covers items **2–12, 18**.

`tests/live/test_phase1_decision_log.py` — **28/28 checks**, auto-discovered by
`run_all.py` (now 9 files, all passing). The load-bearing check is that reconcile
returns an **identical remap with and without a log attached**, across 7 scenarios
including time-overlap, suppression, and the single-tracklet defect. Also covers
acceptance criteria 3, 4, 6 and 9.

**Two things building it caught:**

- `MIN_OBSERVATIONS` could never fail, because suppression uses the same threshold, so
  every surviving tracklet passes by construction. Suppressed tracklets now get a
  decision record too — which is the only place that gate can fail, and it makes
  "how much am I losing to `min_tracklet_observations`" visible for the first time.
- `runner_up_differs` was degenerate. With no selectable candidate there is no margin
  decision, but it still counted as a disagreement — a single-camera run reported
  **100%**. Now `None` in that case, with `no_selectable_candidate` reported
  separately, so the band interpretation stays meaningful.

**Also landed:** `tests/calibration/analyze_decision_log.py` — reads a decision JSONL
and answers the Phase 9 questions (where the bar cuts, orphan count, near-tie
classification, reciprocity breakdown, per-camera eligible-set size, cross-camera
merges vs the stranger ceiling). It produced J.6 on its first real run.

And `tests/live/test_shutdown_reaches_reconcile.py` — 14 checks pinning the J.5 fix.

**Five self-caught errors so far**, all found by building the instrumentation rather
than by reasoning: the two Part H methodology bugs, `MIN_OBSERVATIONS` being
unfailable, `runner_up_differs` counting non-decisions, the analyser printing an
inverted "clean band", and — while landing #40 — the analyser's section 1 judging
**every** configured threshold against **every** subject, which with per-camera bars
blames 0.80 for subjects in a camera that never used it. Now filtered to the subjects
that actually faced each bar, and it prints which cameras used it. **That correction is
a no-op on a single-threshold log**, so run1's J.10 numbers stay directly comparable to
run2's. Assume more remain; prefer the *direction* of a comparison over its magnitude.

**Still open in Phase 1:** item 1 is partial (terminal states recorded; the
`Candidate` / `merged Candidate` distinction is implicit in `phase` + `merged_from`),
and items **13** (label-free correctness counters — these need the rendered output and
gid map, not reconcile), **14** (per-camera timing), and **15–17** (timestamp
diagnostics, which live in capture/pipeline) are not started.

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

Measured effect (bank scoring, 14 proven-distinct pairs — see the corrected H.2/H.3):
post-BN improves the separation margin in **every** scoring mode at **both** sample sizes,
e.g. `+0.055 → +0.086` at 48 frames and `+0.108 → +0.157` at 90, and lowers the
different-person ceiling (`0.845 → 0.782`, `0.892 → 0.858`). The *direction* is the stable
result; the magnitudes are not. At matched false-accept rates recall was comparable, so this
**buys calibration robustness rather than accuracy** on available footage.

Untested hypothesis: the benefit may be larger cross-camera, where post-ReLU compression
pushes same-person scores toward the stranger band.

> **Changing the tap voids every threshold in Part H.**

### Phase 9 — Calibration sweeps (needs Phase 0 + 3)

| # | Item |
|---|---|
| 40 | ~~**`same_camera_threshold` must become PER-CAMERA**~~ — **LANDED 2026-07-30, unmeasured.** Field data (J.6) showed 0.90 yields 9.0 eligible partners per subject in cam_206 but **0.0** in cam_213 and 0.5 in cam_224, with overlapping per-subject boundaries so no global value works. Shipped: `identity.reconcile.per_camera` (cam_213 and cam_224 at **0.80**), resolved by `reconcile.resolve_same_camera_thresholds` — the single merge point both call sites use, so live and file-batch cannot drift, exactly as `resolve_detector_cfg` does for the detector. `pair_threshold()` resolves a CLUSTER pair's bar with `strictest_same_camera_bar`: **max** over the cameras the two clusters share, since merging asserts the same-person claim for every one of them and the loosest camera must not launder a merge past the tightest. Malformed config is skipped loudly (an unnoticed `80` for `0.80` would silently disable same-camera merging); a configured camera absent from the run is named in the log (D1). 36 checks in `tests/live/test_per_camera_same_camera_bar.py`, including that **no overrides reproduces the old behaviour exactly**. **Still a hypothesis from one run — it has not been measured on footage** (D9) |
| 41 | `cross_camera_threshold` — still uncalibrated. Do this only AFTER #40, since fewer orphans means fewer spurious competitors and the picture will change |
| 42 | `min_tracklet_observations` — LOW priority: only 8/97 tracklets and 0.3% of observations suppressed (J.6) |
| 43 | ~~`TOP2_MARGIN` sweep~~ — **ANSWERED, see Part A.** Accepted and rejected margin distributions are near-identical (median 0.0224 vs 0.0186), so it carries no information. Keep `threshold: null` |
| 44 | `RECIPROCAL_BEST` — rejects 31% of decisions at up to 0.905 (J.6). Re-examine only after #40 |
| 45 | Consensus vs `max(proto, exemplar)` scoring — consensus gave the **lowest different-person ceiling** of the three modes at both sample sizes (corrected H.2/H.3). Also the candidate fix for the front/back blurring in #34a |
| 45a | **Reconcile compares prototype MEANS, which blurs front/back into a vector matching neither view.** The live engine's `_bank_score` documents avoiding exactly this with `max(prototype, best_exemplar)`; reconcile never got that fix. This is the structural cause behind cam_213's front/back split — #40 treats the symptom, this treats the cause |

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

### Phase 11 — Throughput (dropped to lowest priority by J.1; **re-open if yolo11m drops frames**)

> J.1 measured real drop rates of 0.5–6.5% and declared this phase largely dead — but that
> was at `yolo11n`'s cost. The detector is now `yolo11m` (~10× the FLOPs, 2.0× measured on
> CPU: H.11). If run2's per-camera `dropped%` climbs, the items below stop being dead code
> and #50 / #51 become the cheapest way to pay for the bigger model.


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
| ~~**run1** — baseline, Phase 0~~ | **DONE**, `20260730_093723`. Per-camera drop rate, fragmentation ratio, H.265 answer, first cross-camera data, frozen clips. Analysis in J.6–J.12 |
| **run2** — #40 + yolo11m, **the next run** | Did the orphan count fall and did the operator's four named symptoms (J.8) go away? Diff analyser sections 1/2/5 against J.10, and per-camera tracklet counts against J.9. **Also the throughput verdict on yolo11m** — per-camera `dropped%` vs J.9's 0.4–2.9% |
| **run3** — after Phase 5, TCP transport | compare `grep -c hevc` and prototype tightness |
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

**Reproduce any of this with the harness in [`tests/calibration/`](tests/calibration/)** —
see its [README](tests/calibration/README.md) for which script produces which subsection.
The scripts are the source of truth; this section is a snapshot.

All numbers below were produced on `register_file.avi` (2560×1440, **0 H.265 decode errors**),
which matches production camera resolution. **Sample: 6 people, one camera, one clip, 14
proven-distinct pairs.** The "different person" set uses only tracks that **co-occur in the
same frame**, since a person cannot be two simultaneous detections — earlier figures that
included non-co-occurring track pairs were contaminated by fragmentation and are not
reproduced here.

> **Two methodological errors were made and corrected while producing this section.** Both
> inflated the results in the optimistic direction, and both are now enforced in code by the
> harness. (1) Different-person pairs must co-occur in a frame — otherwise fragments of one
> person enter the different-person set. (2) Bank queries must be held out of the bank —
> otherwise the max-exemplar term matches a query against itself and returns 1.000. Assume
> further methodological error is possible; prefer the *direction* of a comparison over its
> magnitude.

### H.1 Raw crop-to-crop cosine (48 frames @ stride 6)

| Tap | same mean | same p5 | other mean | other p95 | other MAX | margin |
|---|---|---|---|---|---|---|
| post-ReLU (ships) | 0.885 | 0.710 | 0.552 | 0.640 | 0.819 | +0.070 |
| post-BN | 0.848 | 0.601 | 0.373 | 0.509 | 0.770 | +0.093 |

Sample-size sensitive — at 90 frames the post-ReLU ceiling rises to 0.936. See the note
under H.2/H.3.

### H.2 / H.3 Bank scoring and alternatives — **CORRECTED 2026-07-30**

> **The first version of these two tables was wrong.** Bank queries were not held out of
> the bank, so any query also present in the bank matched *itself* via the max-exemplar
> term and returned 1.000. That inflated the same-person distribution by whatever fraction
> of queries sat in the bank, which made the margin depend on frame count rather than on
> the model. `tests/calibration/measure_score_separation.py::_holdout` now splits each
> track into a bank half and a disjoint query half. The live engine documents this exact
> trap in `_reinforce`. Superseded figures were: bank post-ReLU margin +0.047 / ceiling
> 0.828; consensus +0.070 / 0.793.

Corrected, `register_file.avi`, two sample sizes to show the instability:

| Mode | Tap | margin @48f | ceiling @48f | margin @90f | ceiling @90f |
|---|---|---|---|---|---|
| raw crop-to-crop | post-ReLU | +0.072 | 0.819 | +0.075 | 0.936 |
| raw crop-to-crop | post-BN | +0.097 | 0.765 | +0.114 | 0.914 |
| **bank** `max(proto,exemplar)` (ships) | post-ReLU | +0.055 | 0.845 | +0.108 | 0.892 |
| **bank** `max(proto,exemplar)` | post-BN | +0.086 | 0.782 | +0.157 | 0.858 |
| consensus (mean of top half) | post-ReLU | +0.060 | **0.798** | +0.080 | **0.835** |
| consensus (mean of top half) | post-BN | +0.071 | **0.738** | +0.112 | **0.770** |
| prototype only | post-ReLU | +0.059 | 0.845 | +0.104 | 0.869 |
| prototype only | post-BN | +0.071 | 0.782 | +0.151 | 0.806 |

**Stable conclusions** — hold at both sample sizes, safe to act on:

1. post-BN beats post-ReLU on margin in every mode, and lowers the ceiling
2. consensus gives the **lowest different-person ceiling** of the three scoring modes
3. the different-person ceiling (0.78–0.94) sits **far above**
   `live.identity.same_camera_threshold: 0.70` — that threshold is inside the range where
   strangers score. The correction made this finding *stronger*, not weaker.

**Unstable — do not set a threshold from one run.** `other MAX` is an extreme-value
statistic and grows with sample size (raw post-ReLU: 0.819 → 0.936 between the two runs).
The margin itself moved +0.055 → +0.108. Prefer p95 over MAX, and get actual values from
the Phase 9 sweep on frozen multi-camera footage.

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

### H.11 Detector capacity: yolo11n vs yolo11m

`tests/calibration/compare_detector_models.py`, `register_file.avi` (2560×1440),
150 consecutive frames, shipped `conf 0.40 / iou 0.60 / imgsz 640`, fresh ByteTrack
per model. **4 people in view.**

| Model | dets | mean/frame | track ids | track lengths | ms/frame (CPU) |
|---|---|---|---|---|---|
| yolo11n (was) | 552 | 3.68 | **6** | 150, 150, 150, **64, 37**, 1 | 215 |
| yolo11m (ships) | 600 | 4.00 | **4** | 150, 150, 150, 150 | 435 |

yolo11m found more boxes on 49 of 150 frames and fewer on 1. The result that matters is
the **id count**, not the box count: yolo11n's ids 4 (frames 0–55) and 6 (frames 80–149)
**never co-occur in any frame** and are separated by a 24-frame gap — one person lost and
re-minted. yolo11m covers that person with a single continuous track. yolo11n also emits a
1-frame phantom id.

So the two changes staged for run2 meet in the middle: **yolo11m creates fewer fragments;
#40 merges the fragments that remain.** Same-camera fragmentation is the mechanism behind
both the operator's cam_213 front/back split and their own diagnosis, *"this seems like a
track id change so reid change nonsense"*.

**What this does NOT establish.** One clip, one camera, 4 people, no frame dropping, and
CPU timings — the 2.0× CPU ratio is not the GPU ratio, and a file run never drops frames.
The live path does: a slower detector means more dropping, which fragments tracks, cutting
against the gain measured here. That half is only answerable from run2's per-camera
`dropped%`. Nor is this clip cam_206, whose missed people are in a crowded room with a
table — its own frozen replay clip is the only thing that reproduces that.

> Precedent worth respecting: Part A rejected `imgsz` 640→1280 as a no-op on this same
> clip with yolo11n. Capacity moved what resolution did not, so `imgsz` deserves
> re-measuring on yolo11m before it is treated as settled.

---

## Part J — Field results, 2026-07-30 (A6000, 4 cameras, live RTSP)

First real production data. **Three of my hypotheses were wrong and one new P0 defect
surfaced.** Source: `run_id 20260730_082045`, 4 cameras, 40 s of metrics, `cuda:0`,
`nvdec_usable: False`, `decode_backend: cpu`.

### J.1 Frame dropping is a NON-PROBLEM — Phase 11 largely dead

| Camera | nominal | read | rendered | dropped |
|---|---|---|---|---|
| cam_213 | 25 fps | 24.2 | 905/968 | **6.5%** |
| cam_224 | 15 fps | 14.2 | 551/567 | **2.8%** |
| cam_206 | 25 fps | 21.8 | 868/872 | **0.5%** |
| cam_219 | 25 fps | 22.7 | 877/908 | **3.4%** |

`infer_q` dropped **10 frames in 40 s**. `slot_drop` ≤ 4. `stale_skipped=0`. Inference
completed 3271 of 3315 frames read, at **44% scheduler utilisation**.

My CPU-on-file measurement said 85–99% (H.10). I flagged it unrepresentative; it is
unrepresentative by a factor of ~20. **Narrowing the ReID lock (#50), the queue depth
(#51), `imgsz` and `conf` all solve a problem that does not exist** at this crowd size.
Phase 11 drops to lowest priority. Caveat retained: the ReID batch scales with people
in frame, so this could change in a genuinely crowded scene.

### J.2 Cross-camera linking is 0-for-19, and the threshold is NOT why

```
x-camera: attempts=19 linked=0 rejected[thresh=0 margin=9 recip=10 topology=0]
histogram:  <.5:0  .5-.6:0  .6-.7:9  .7-.8:10  .8-.9:0  .9+:0
```

**`thresh=0`** — not one attempt was below `cross_camera_threshold: 0.60`. All 19 scored
0.60–0.80 and every one died on the **runner-up margin** (9, at `accept_margin: 0.03`) or
**reciprocal-best** (10).

This inverts the assumption in the config history, which lowered `cross_camera_threshold`
0.63 → 0.60 to let real matches through. The threshold was never the obstacle. Nine
near-ties plus ten reciprocity failures means several candidates are bunched in the same
0.6–0.8 band — exactly the ambiguity `TOP2_MARGIN` was added to measure, and strong
support for prioritising Phase 9's margin questions over its threshold sweeps.

### J.3 Same-camera 0.70 cuts through the middle of the distribution

```
same-cam reacquire: attempts=17 ok=10 rejected_below_thr=7 max_rejected=0.690
histogram:  .5-.6:2  .6-.7:5  .7-.8:7  .8-.9:1  .9+:2
```

Max rejected 0.690, a hair under the bar — the same pattern the config comments record at
0.85 and 0.90. `linked=0` this run, so the two-lane leak did not fire and these 7
rejections are genuine mints rather than double-counted.

### J.4 H.265 corruption does NOT occur on the live feed

**Zero** decode errors across 4 live streams. The 294/682 and 207/1573 broken references
in `test_file.avi` / `test_v2.avi` are an artefact of however those were recorded, not
something happening on RTSP. **Issue #28 downgrades** from "leading hypothesis for random
identity behaviour" to "worth setting `rtsp_transport=tcp` anyway"; #29's blocking-read
problem stands on its own merits.

### J.5 NEW P0 — a failed `print` abandoned the run's ids. Twice.

Two consecutive runs produced clips and metrics, then **nothing**: no final summary, no
reconcile, no decision log, no output video. Both launched as `python main.py ... | tee
run1.log`, which puts python and tee in one foreground process group. Ctrl-C reaches
both; tee dies first; every subsequent `print` raises `BrokenPipeError`. And
`_report(final=True)` runs **before** `_finalize_offline()`, so a cosmetic print failure
skipped the reconcile — with the traceback going into the same dead pipe, so nothing was
visible. A dropped SSH session or closed terminal breaks stdout identically.

`InterruptGuard` does not help: it protects against signals and explicitly does not
swallow exceptions.

**Fixed 2026-07-30** — `_report(final=True)` is now guarded, and stdout/stderr are wrapped
in `_QuietOnBrokenPipe` for the finalization phase so no print can raise. Pinned by
`tests/live/test_shutdown_reaches_reconcile.py` (14 checks), including the
report-raises, dead-stdout, and both-at-once cases plus the ordering invariant.

**Diagnostic trap worth remembering:** `output_cam_*.mp4` existing does **not** mean the
last run succeeded. Those files are only overwritten by a completed render, so stale
outputs from an earlier run look like success. Check their mtime against the `run_id`.

### J.6 Decision-log analysis — the per-camera finding that reframes Phase 9

Full analysis of `logs/reconcile_decisions_20260730_093723.jsonl` (286 decisions) via
`tests/calibration/analyze_decision_log.py`. **Operator ground truth, max concurrent
per camera:** cam_206 ≈ 6, cam_224 ≈ 5, cam_219 ≈ 4, cam_213 ≈ 3, with people
continuously entering and leaving, so total distinct people over the 4.5-minute run
is higher than any single maximum — plausibly 8–15.

Against that, **17 identities is roughly plausible in aggregate**, not obviously
broken. But the specific mis-assignments the operator observed are real regardless of
the count, and the analysis explains them exactly.

**`same_camera_threshold` is a PER-CAMERA problem. A global value cannot work.**

| Camera | subjects | eligible/subject @0.90 | zero eligible |
|---|---|---|---|
| cam_206 | 55 | **9.0** (max 21) | 6 |
| cam_213 | 11 | **0.0** (max 0) | **11 / 11** |
| cam_219 | 6 | 0.3 | 4 / 6 |
| cam_224 | 17 | 0.5 (max 2) | **12 / 17** |

At 0.90, **cam_213 achieves zero same-camera merges out of eleven subjects**, and
cam_224 manages it for only 5 of 17 — while in cam_206 every fragment sees nine
partners above the bar. The same value is harmless in one camera and total in another.
The analyser confirms no global bar exists: p95 of the "top different" score (0.816)
**exceeds** p5 of the "worst same" score (0.719), so the per-subject boundaries
overlap. Median boundary sits between 0.721 and 0.856.

Supporting: **50.6%** of subjects lost a candidate above 0.85, 34.5% above 0.88,
highest rejected 0.899. **Orphans — subjects with no eligible same-camera partner —
are 33/89 (37%)**, ten with a best candidate above 0.85.

This maps directly onto the observed symptoms: cam_213's "correct by his front, reid 7
by his back" is a camera with *no* same-camera merging, so front and back fragments
can never join and each is absorbed cross-camera separately. cam_224's "reid 3 becomes
reid 7 at a bad angle" is the same, with 12 of 17 orphaned. cam_206's 26-tracklet
identity is fragmentation that *self-heals* because its scores clear 0.90.

**`TOP2_MARGIN` must NOT gate — it carries no information.**

```
margins on accepted merges: median=0.0224  p5=0.0017  p95=0.1181
margins on rejected merges: median=0.0186  p5=0.0016  p95=0.0779
```

Near-identical distributions, so a gate on it would be close to random. Note this
supersedes an earlier read of mine: from two records I claimed fragmentation dominates
near-ties; at scale it is 55 fragmentation vs **65 genuine ambiguity**, roughly
balanced. The conclusion holds for a better reason than the one I first gave.
Disagreement between the two margin definitions: **63.9%**.

**`RECIPROCAL_BEST` rejects 31% of decisions**, all cross-camera, at mean 0.720 and
**max 0.905**. Rejecting a 0.905 cross-camera match is very likely rejecting a real
person. Concentrated in cam_206 (40) and cam_224 (35).

**Cross-camera merges:** 52 accepted, min 0.639, **4 below the 0.66 stranger ceiling**
and 10 below 0.70. Those four are the false-merge candidates worth eyeballing.

**Suppression is cheap here:** 8 of 97 tracklets, 0.3% of observations.

### J.7 Confirmed as expected

`written=0` on every camera (#63, cosmetic). Software H.265 decode for all four streams
and still keeping pace (#53). The store held **4157 points before this run**, and
`_gather_tracklets` scrolls the whole collection filtering by `run_id` in Python — issue
#20 now has a real number attached, and it grows every run.

### J.8 Operator observations from watching the output videos

Ground truth on the *symptoms*, which no metric replaces. Recorded verbatim in substance
because these are what "working" has to mean.

| Camera | Observation | Mechanism |
|---|---|---|
| cam_224 | "definitely sped up" | 14.8 fps tagged at 20 → plays 1.35× fast (#45/#46) |
| cam_224 | reid 3 → **reid 7** when the person moves at a bad angle | orphan fragment absorbed into the wrong cross-camera cluster; 12 of 17 subjects orphaned here (J.6) |
| cam_219 | "a little slowed"; reid 2 → **reid 4** when he moves to the other side of the table | 24.2 fps tagged 20 → 1.21× slow; the id change is a track-id change that reconcile failed to re-merge |
| cam_213 | reid 2 correct **by his front**, becomes **reid 7 by his back** | cam_213 achieves ZERO same-camera merges at 0.90, so front and back fragments can never join. Compounded by reconcile comparing prototype *means* (#45a) |
| cam_206 | 5 people present but only 1 detected at first; "randomly starts detecting the rest" | **detection recall**, not identity. Nothing downstream can invent a box YOLO never produced. Also explains cam_206's 61 tracklets |

**Responses now staged for run2:** the cam_213 / cam_224 rows are what #40 addresses; the
cam_206 recall row is why `detector.model` moved to `yolo11m` (H.11). The cam_224 and
cam_219 playback-speed rows are **not** addressed — they are #45/#46 and still open, so
expect the same "sped up / slowed" complaint from run2's videos.

Operator's own diagnosis, which was correct: *"this seems like a track id change so reid
change nonsense which should not always be the case."* Final-video ids come from
`gid_map[(camera, track_id)]`, so a new track id is looked up independently — reconcile's
entire job is to re-merge those, and it is failing.

### J.9 Raw metrics, run `20260730_093723` (final, t=+130s)

The run the analysis is based on. ~4.5 min, 4 cameras, `cuda:0`, `nvdec_usable: False`.

| Camera | read | rendered | dropped | slot_drop |
|---|---|---|---|---|
| cam_213 | 3227 (24.8 fps) | 3133 | **2.9%** | 30 |
| cam_224 | 1923 (14.8 fps) | 1909 | **0.7%** | 0 |
| cam_206 | 3135 (24.1 fps) | 3123 | **0.4%** | 7 |
| cam_219 | 3163 (24.3 fps) | 3126 | **1.2%** | 10 |

Even lower than the 082045 run in J.1. `scheduler: batches=3119 frames=11398
avg_batch=3.65 util=46% stale_skipped=0`. `infer_q=7 dropped / peak 2`,
`identity_in=0 / peak 4`, `identity_fair_drop=0`, `inference_done=11370`.

**Live engine counters** (off the product path, but they characterise the score
distributions):

```
identity: minted=16 reacquired=18 linked=2 active=16 stored=1500
x-camera:   attempts=15 linked=2 rejected[thresh=4 margin=5 recip=4] max_subthreshold=0.599
same-cam:   attempts=23 ok=18 rejected_below_thr=5 max_rejected=0.699
coactive_vetoes=49   topology_pruned=0
histograms (<.5 .5-.6 .6-.7 .7-.8 .8-.9 .9+)
  same-cam reacquire:  0  1  4  4  2 12      <- mass well above 0.70; 0.70 works here
  cross-camera:        0  4  7  4  0  0      <- nothing above 0.80
```

Note `max_subthreshold_score=0.599` — four cross-camera matches missed the 0.60 bar by a
thousandth. And unlike the 082045 run (0 links from 19 attempts), this one linked 2 of 15.

**Per-camera tracklets and observations** (from the reconcile diagnostics):

| Camera | tracklets | total obs | min | max | mean |
|---|---|---|---|---|---|
| cam_206 | **61** | 1065 | 1 | 203 | 17.5 |
| cam_213 | 11 | **85** | 3 | **13** | 7.7 |
| cam_219 | **7** | 1848 | 1 | 722 | 264.0 |
| cam_224 | 18 | 1211 | 2 | 355 | 67.3 |

**cam_213 is starved**: it read the *most* frames of any camera yet produced the *fewest*
observations (85, longest track 13 ≈ 4 s at `reid.interval: 10`). Either almost nobody was
in view, or the crop-quality gate rejects nearly everything there. Untested.

**Store:** 4209 observations for this run → 17 identities, 11 cross-camera.

### J.10 Full analyser output, run `20260730_093723`

286 decisions (8 suppressed), 97 outcomes; phase 1 = 89, phase 2 = 189.

```
gate failures : MIN_OBSERVATIONS 8, ABSOLUTE_THRESHOLD 92, RECIPROCAL_BEST 86
exclusions    : BELOW_ABSOLUTE_THRESHOLD 5576, TEMPORAL_CONFLICT_SAME_CAMERA 1008,
                NOT_MERGEABLE_CROSS 720

1. WHERE THE BAR CUTS (same-camera, n=87 subjects)
   highest score REJECTED per subject: mean=0.821 p50=0.853 p95=0.898 max=0.899
     lost a candidate above 0.75: 64/87 (73.6%)
     lost a candidate above 0.80: 55/87 (63.2%)
     lost a candidate above 0.85: 44/87 (50.6%)
     lost a candidate above 0.88: 30/87 (34.5%)
   natural boundary from 77 bimodal subjects:
     gap LOWER edge (top 'different'): mean=0.721 p95=0.816 max=0.841
     gap UPPER edge (worst 'same')   : mean=0.856 p5=0.719  min=0.609
   => boundaries OVERLAP: no global bar works (p95 lower 0.816 > p5 upper 0.719)

2. ORPHANS: 33/89 (37.1%) with zero eligible same-camera partners
   their best rejected candidate: mean=0.795 p95=0.903 max=0.908
   10/33 had a best candidate above 0.85
   by camera: cam_206 6, cam_213 11, cam_219 4, cam_224 12

3. NEAR-TIES (margin <= 0.05): 120 total
   fragmentation (runner similarity >0.85): 55 (45.8%)
   genuine ambiguity (<=0.85)            : 65 (54.2%)
   margins accepted: n=91 p5=0.0017 median=0.0224 p95=0.1181
   margins rejected: n=72 p5=0.0016 median=0.0186 p95=0.0779   <- indistinguishable
   margin-definition disagreement: 124/194 (63.9%) [+84 no selectable candidate]

4. RECIPROCAL_BEST: 86/278 (30.9%), ALL cross_camera
   scores rejected: mean=0.720 p95=0.862 max=0.905
   by camera: cam_206 40, cam_224 35, cam_213 22, cam_219 11

5. ELIGIBLE-SET SIZE @0.90
   cam_206: 55 subjects  mean=9.0  max=21  zero=6
   cam_213: 11 subjects  mean=0.0  max=0   zero=11
   cam_219:  6 subjects  mean=0.3  max=1   zero=4
   cam_224: 17 subjects  mean=0.5  max=2   zero=12

6. CROSS-CAMERA ACCEPTED: n=52 min=0.639 mean=0.801 max=0.937
   below 0.70: 10/52    below 0.66 (stranger ceiling): 4/52
   observed accepted scores: 0.937 ... 0.733 0.721 0.704 0.692 0.690 0.688 0.655 0.639

7. SUPPRESSION: 8/97 tracklets, 0.3% of observations
   by camera: cam_206 6, cam_219 1, cam_224 1

8. OUTCOME: 17 identities, tracklets/identity mean=5.24 max=26
   distribution: [26, 10, 8, 7, 5, 5, 5, 5, 4, 3, 3, 2, 2, 1, 1, 1, 1]
   spanning >1 camera: 11/17
```

### J.11 Worked decision-log examples

Four real records, kept because they show the distributions concretely and are the
clearest single argument for #40.

**`U-0000`** (cam_206, 18 obs, frames 1–155). 54 candidates scored, **3 eligible**,
margin 0.0689. Sorted scores:

```
0.977  0.908  0.906  |  0.897  0.876  0.875  |  0.662  0.633  0.631  0.621 ... 0.495
       ELIGIBLE      |   REJECTED at 0.90    |    genuinely different people
```

A clean gap between **0.875 and 0.662**, with the bar cutting *inside* the same-person
cluster and discarding three genuine merges. A bar in ~(0.68, 0.87) would be perfect here.

**`U-0001`** (cam_206, 10 obs). **16 eligible candidates**, all above 0.90:
`0.9718 0.9694 0.9623 0.9571 0.9457 0.9451 0.9447 0.9367 0.9319 0.9289 0.9213 0.9207
0.9199 0.9185 0.9120 0.9056`. Mutual `pair_similarity_to_best` 0.92–0.97, so not a chain.
**One person, seventeen tracklets, one camera.** Top-2 margin **0.0024** — a near-tie
between two fragments of the same person, which is why a margin gate would be harmful.

**`U-0002`** (cam_206, 23 obs). 4 eligible, margin 0.0069. `U-0000` scored **0.8757**
against it and was rejected — yet both merge with `U-0017`, so they end up in the same
cluster transitively anyway. The 0.90 bar blocks the direct edge but not the outcome.

**`U-0003`** (cam_206, **203 obs**, frames 371–2379). Only 1 eligible candidate;
`TEMPORAL_CONFLICT_SAME_CAMERA: 8` — eight other tracklets overlapped it in time, so the
temporal veto is working. `margin_eligible: null`, `margin_all_scored: 0.3405`.

### J.12 Identity composition, run `20260730_093723`

The 11 cross-camera identities, as reconcile assigned them. Kept so a re-run can be
diffed against it rather than re-derived.

```
GID  1: cam_206(5,8,13,79,81,93) + cam_224(185)
GID  2: cam_206(7,19,21,30,49,54,58,66,76,83,89,96,97,131,137,138,139,140,157,162,165,210)
        + cam_213(26) + cam_219(6) + cam_224(3,43)          <- 26 tracklets, 22 in cam_206
GID  3: cam_206(80) + cam_213(120) + cam_219(143,173) + cam_224(126)
GID  4: cam_206(82) + cam_213(38) + cam_219(48) + cam_224(2,52)
GID  5: cam_206(92) + cam_213(86) + cam_224(12)
GID  7: cam_206(99,104,107,112,114,127) + cam_213(33) + cam_224(159)
GID  8: cam_206(108) + cam_224(132)
GID 11: cam_206(176,192) + cam_213(180,190) + cam_224(183)
GID 12: cam_206(195) + cam_213(116,193) + cam_224(121,189)
GID 15: cam_213(115) + cam_219(118) + cam_224(117)
GID 16: cam_219(4) + cam_224(1,20,202)
```

GID 2's 22 cam_206 members are all **mutually time-disjoint** (reconcile refuses to merge
time-overlapping same-camera tracklets) and each cleared the **0.90** bar, so it is almost
certainly one person entering and leaving that view 22 times — not a false merge.

The operator-reported failures map onto this: cam_213's front/back split is GID 2's
`cam_213(26)` versus GID 7's `cam_213(33)`; cam_224's bad-angle switch is GID 3's
`cam_224(126)` versus GID 7's `cam_224(159)`; cam_219's across-the-table switch is GID 2's
`cam_219(6)` versus GID 4's `cam_219(48)`.

---

## Part K — Library internals verified by reading source

Facts established by reading the installed packages, not from documentation or memory.
Recorded so the next session does not re-derive them. Versions: `ultralytics 8.4.89`,
`torch 2.12.1`, `qdrant-client 1.18.0`, `opencv 5.0.0`.

### K.1 ByteTrack (`ultralytics/trackers/`)

`cfg/trackers/bytetrack.yaml` ships: `track_high_thresh: 0.25`, `track_low_thresh: 0.1`,
`new_track_thresh: 0.25`, `track_buffer: 30`, `match_thresh: 0.8`, `fuse_score: True`.

- `byte_tracker.py:312-315` splits detections by those two thresholds. Since the detector
  pre-filters at `conf=0.4`, the **second-association pool (0.1–0.25) is always empty**, so
  ByteTrack's occlusion-recovery stage never runs. Mechanism real; measured as a no-op on
  available footage (Part A).
- `_format_output` (`:486-488`) returns one row per track that is in `tracked_stracks`
  **and** `is_activated`, and association is a one-to-one linear assignment. **The tracker
  cannot invent a box** — confirmed empirically across 120 frames, 0 with more rows than
  detections. Duplicate boxes therefore require duplicate *detections*.
- `activate()` (`:108-109`) sets `is_activated` immediately only when `frame_id == 1`.
  Otherwise a new track needs a **second consecutive match** before it appears in output,
  so a one-frame duplicate detection is filtered but a persistent one is not.
- `multi_predict` advances the Kalman filter **one step per `update` call with no `dt`**, so
  under irregular frame gaps the predicted box is wrong by however much real time passed.
- `on_predict_start`/`register_tracker` attach the tracker to the *predictor*, so tracker
  state is per-`YOLO`-object. One `PersonDetector` per camera therefore isolates state
  correctly (verified).

### K.2 Ultralytics predict defaults (`cfg/default.yaml`)

`imgsz: 640`, `iou: 0.7`, `max_det: 300`, `agnostic_nms: False`, `device:` unset (auto).
The pipeline never passes `imgsz` or `device`, so 640 and auto-select apply.

### K.3 torchreid OSNet feature tap

`model.fc == Sequential(Linear(512,512), BatchNorm1d(512), ReLU())`, and eval-mode
`forward()` returns `self.fc(v)` — i.e. **post-ReLU**. Measured consequence: 21.0% of dims
exactly zero, 0% negative, so cosine between any two embeddings is ≥ 0 and the usable
range is compressed upward. Post-BN is Phase 8; changing the tap voids every threshold.

### K.4 Qdrant client

`QdrantClient` accepts `prefer_grpc` (default `False`), `grpc_port` (default 6334) and
`grpc_options`. `scroll()` accepts `scroll_filter` — **currently unused everywhere**, which
is why `_gather_tracklets` pulls the entire collection and filters in Python (#20).
`docker-compose.yml` **already maps 6334**, so gRPC needs no infrastructure change.
Embedded `path=` mode is in-process and has no gRPC.

### K.5 Verifier weights (batch path, `identity/verifier.py`)

`w0=-13.5`, `w_cosine=11.0`, `w_rerank=7.0`, `w_observation_count=0.6`,
`w_time_since_last=-0.01`, `w_bbox_quality=1.0`. Because the observation-count term is
`0.6 · log1p(count)`, unbounded and never decayed, the minimum accepted cosine falls:

| observations | 0 | 10 | 30 | 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|---|---|---|
| min accepted cosine | 0.733 | 0.653 | 0.619 | **0.580** | **0.543** | **0.503** | **0.466** |

Different people top out around 0.57, so past ~100 accumulated observations the busiest
identity starts accepting strangers. Separately, `min_prob_gap: 0.05` saturates: at 300
observations, cosine 0.90 vs 0.85 gives a probability gap of **0.0019**, so a clean winner
is rejected and a duplicate minted. Batch path only, hence Part C — but do not reuse these
weights anywhere without re-deriving them.

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
