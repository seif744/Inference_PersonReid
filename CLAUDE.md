# CLAUDE.md — rules that outrank your instincts on this repo

Read this before proposing anything. Every rule below is here because it was
learned the expensive way, and each one contradicts what a reasonable engineer
would otherwise do. Full detail lives in `ARCHITECTURE.md`, `REMEDIATION_PLAN.md`
Part A, and `ADR-003D`.

This file is **tracked in git**, so it reaches the A6000 and every fresh session.
The ADR set is not (see §7).

---

## 1. Geometry: recorded live, consumed offline. Never recomputed.

> **The live run records geometry. Offline reconcile only consumes it.**

Each observation's floor position is computed **once, at capture**, by
`src/geometry/recorder.py`, and stored in that observation's Qdrant payload under
`floor`. `src/identity/reconcile.py` **reads** it. Reconcile must never:

- load a calibration record,
- apply a homography,
- derive a position from a bounding box.

Mechanically enforced: reconcile may import `geometry.reachability` (pure
arithmetic) and nothing else under `geometry/`.
`tests/live/test_geometry_not_recomputed.py` asserts it via the AST.

**Why, since reconcile already has the `bbox` and could just do it:**

1. The live RTSP feed is never recorded. A position not written during the run is
   gone — you cannot go back for it.
2. Re-fitting a calibration is cheap (`tools/fit_floor_frame.py` needs no camera
   time), therefore likely. If reconcile derived positions, re-reconciling a
   finished run would silently return **different identities**, with nothing in
   either output saying why. Reconcile is the authority on final ids; an authority
   that changes its mind for invisible reasons is not one.

One run, one geometry, decided once.

## 2. No metre-based threshold without a trusted metric reference.

The floor frame is fitted from camera imagery alone, which fixes the plane only up
to an **arbitrary scale**. Units are **floor units**, not metres.

> Metric geometry cannot be established from monocular cameras alone. A single
> trusted metric reference must be provided before any metre-based threshold
> (0.5 m, 3.0 m, walking speed in m/s) is considered valid.

The requirement is *one trustworthy source of metric scale* — **not** a tape
measure, and **not** a survey. In order of preference:

1. verified floor plans or CAD drawings;
2. known architectural dimensions (corridor width, door width, column spacing,
   floor-tile pitch), verified on site;
3. one or more independently measured reference distances (laser rangefinder, tape,
   measuring wheel) sufficient to establish **and validate** scale.

**As of 2026-08-03 none of these exist for this deployment**, so nothing claims
metres. `CalibrationRecord.is_metric` is False and the metre-facing API raises
`MetricScaleUnavailable`. This costs nothing: the within-group reachability check
compares a floor-unit distance against a floor-unit speed ceiling measured on the
same footage, so the unknown scale **cancels exactly**.

Metres are needed for exactly one thing: relating **separate floor groups**
(`group_distances`), because an operator-supplied real-world distance arrives in
metres. That is why `cam_206` and `cam_213` have no geometry.

## 3. Threshold tuning is not the lever. Measure first.

**Five** tuning changes have been reverted for *hurting* accuracy:
`identity.threshold` → 0.68, verifier `accept_threshold` → 0.62/0.08,
`max_occlusion_ratio` → 0.35, and the `live.topology` min-transit veto.

Settled by two live runs plus an offline sweep: cam_224 at 0.80 **fused several
people into one reid**; at 0.90 **one person shattered into many**. Both directions
wrong means the number is not the variable.

So: do not propose a threshold change as a fix. Fix the scoring, or add a physical
guard, and re-anchor bars only afterwards.

## 4. A cluster count cannot tell you whether a cluster is one person or three.

Every number quoted for the 2026-07-30 threshold change was **equally consistent
with the good and the bad outcome**, and both settings chosen that way made the
videos worse. Rank candidates with the sweep, then **render and watch**:

```bash
python tests/calibration/rerender_from_clips.py <run_id> --geometry
```

Ground truth is the operator watching video. Labels belong in
`calibration/link_labels.jsonl` via `review_links.py --label`, never in a
conversation — there are only 11 so far.

## 5. False merges are worse than false splits — except for vetoes.

A merge fuses two people's histories and corrupts both. A split just gives one
person a spare id, which reconciliation can still fix. When uncertain, mint.

**A veto inverts this.** Refusing a merge is unrecoverable — reconcile cannot
un-split a person — which is exactly how the topology veto destroyed a run
(`linked` collapsed 5 → 1, `topology_pruned=508`). So every geometric error budget
biases toward *permitting* the merge:

- distance is **shrunk** by both positions' error radii,
- elapsed time is **grown** by the clock-differential budget,
- the speed ceiling is **raised** by a safety factor.

A sloppier calibration therefore produces **fewer** vetoes, never wrong ones. When
in doubt, raise `geometry.reconcile.safety_factor`; never lower it.

## 6. Measurement rules that have already been got wrong once.

1. **A "different person" pair requires co-occurrence in the same frame, within one
   camera.** One body cannot be two simultaneous detections. **Across cameras this
   proves nothing here** — `cam_219` and `cam_224` share a room.
2. **Bank queries must be held out of the bank**, or a query matches itself at
   ~1.0 and inflates the same-person side.
3. **`other MAX` grows with sample size.** Prefer p95. On one clip, 48 vs 90 frames
   moved raw MAX from 0.819 to 0.936.

## 7. Facts about the repo that will otherwise waste your time.

- **The ReID model is FastReID SBS ResNet-101-IBN (MSMT17), 2048-d at 384×128.**
  **R101, not R50** — `fastreid_sbs_R50_ibn` is registered as a selectable option
  and is not used. Two things read as if OSNet were still running and are **not**
  the model in force: `backends.DEFAULT_BACKEND = "osnet_ain_x1_0"` (only the
  fallback when `reid.model` is absent) and `OSNetBackend.__init__`'s
  `arch="osnet_ain_x1_0"` (which OSNet *that class* builds; it cannot select
  FastReID at all). What runs is `config.yaml` → `reid.model` → the `BACKENDS`
  registry, and the run banner prints it. A mismatched checkpoint **raises**.
  OSNet-AIN weights stay in the repo because **every threshold in `config.yaml`
  was derived in its 512-d space** and none has been re-anchored.
- **The dev box is CPU-only WSL2, no CUDA.** FastReID R101-IBN is ~0.76 s/crop
  there. **Any conclusion about throughput, identity quality, or thresholds drawn
  on the dev box is wrong.** Use it for correctness only. The A6000 does all model
  runs, captures and calibration.
- **The ADR set and `NVIDIA_DeepStream_Adoption_Plan.md` are gitignored**, so they
  are **not on the A6000** and not in a fresh clone. Anything that must survive
  belongs here, in `ARCHITECTURE.md`, or in `config.yaml` comments.
- **`python main.py --reset` is not a wipe** — it clears the store *and then runs
  the whole pipeline*.
- **`output_cam_*.mp4` existing does not mean the last run succeeded.** They are
  only overwritten by a completed render, so stale files look like success. Check
  mtime against the `run_id`.
- **Placeholders get pasted literally.** `<run_id>` has been run verbatim; `<` is a
  shell redirect. Write commands that derive their values.
- **Recorded files need MEDIA time, not `frame.ts`.** `ts` is stamped at frame read,
  so on a file it tracks decode speed (125+ fps), not events — two files read in
  parallel get timestamps that differ by thread scheduling. `Frame.event_ts()`
  returns media time (`offset + frame_index / fps`) for files and wall-clock for
  streams; the stored payload's `ts` and all of geometry use it, while scheduler
  freshness, writer pacing and the live engine's TTLs deliberately stay on `ts`.
  Never conflate them — `tests/live/test_media_time.py` pins both halves.
- **File-batch mode records no geometry** — `IdentityService._commit` stores neither
  `bbox` nor `ts`. To run geometry over recorded footage use `--mode live` on the
  files.
- **Never re-add `tests/live/_synth.py` to `.gitignore`.** It was untracked while
  its importers were tracked, and `python tests/run_all.py` failed on every fresh
  clone.

## 8. Before you commit

```bash
python tests/run_all.py                                   # 19 files, CPU, seconds
python tests/calibration/verify_embedding_contract.py     # after ANY src/reid/ change
```

`verify_embedding_contract.py` is the only asserting script, and everything it
covers is a **silent** failure mode: a broken BGR→RGB swap, a transposed resize, a
batch/crop misalignment. No error, no crash, just worse matching.
