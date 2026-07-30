# Calibration harness

Measurement scripts backing [`REMEDIATION_PLAN.md`](../../REMEDIATION_PLAN.md) Part H.
They exist to **produce numbers on real footage**, not to pass. Their whole purpose is
to make threshold and model decisions falsifiable instead of argued.

Run everything from the repo root:

```bash
python tests/calibration/verify_embedding_contract.py
python tests/calibration/measure_score_separation.py     register_file.avi 90 6
python tests/calibration/measure_reconcile_thresholds.py register_file.avi 90 4
python tests/calibration/measure_detection.py            register_file.avi 50
python tests/calibration/measure_pose_ensemble.py        register_file.avi 40 20
python tests/calibration/characterize_known_defects.py
python tests/calibration/audit_product_path.py           test_v2.avi 40 2
```

All are CPU-safe. Nothing writes into the repo — `audit_product_path.py` runs in a temp
directory with a distinct camera name and a local Qdrant, so it cannot overwrite a real
`output_<cam>.mp4` or touch the shared gallery.

## What each script is for

| Script | Asserts? | Produces | Run it when |
|---|---|---|---|
| `verify_embedding_contract.py` | **yes**, exits non-zero | Part D | after any change to `reid/extractor.py` or `reid/service.py` |
| `measure_score_separation.py` | no | H.1, H.2, H.3, H.5 | footage, ReID weights, or feature tap changes |
| `measure_reconcile_thresholds.py` | no | H.4 | before touching any `identity.reconcile.*` threshold |
| `measure_detection.py` | no | H.6, H.7 | before proposing any detector-side change |
| `measure_pose_ensemble.py` | no | — | only if the batch path comes back into use |
| `characterize_known_defects.py` | no, always exits 0 | — | after each phase lands, to see what moved |
| `audit_product_path.py` | no | H.10 | after any pipeline change; the only whole-path check |

`verify_embedding_contract.py` is the one genuine regression test. Everything it covers is
a **silent** failure mode — a broken BGR→RGB swap, a transposed resize, a batch/crop
misalignment — that produces no error and no crash, only worse matching.

## Two methodological rules

**1. A "different person" pair requires co-occurrence in the same frame.** A person cannot be
two simultaneous detections, so co-occurrence proves distinctness. Comparing tracks that never
co-occur silently mixes in fragments of the *same* person — a known defect here — which
inflates every different-person statistic. `_common.proven_distinct_pairs` enforces this and
prints what it excluded. The first version of this measurement got it wrong and reported a
different-person ceiling that was too high.

**2. Bank queries must be held out of the bank.** If a query observation is also in the bank,
the max-exemplar term in `ActiveIdentitySet.score` matches it against itself and returns 1.000,
inflating the same-person distribution by whatever fraction of queries sit in the bank. That
makes the measured margin depend on the frame count rather than the model.
`measure_score_separation._holdout` splits each track into a bank half and a disjoint query
half. The live engine documents this same trap in `_reinforce`. **This also got wrong first
time, and the corrected numbers are lower** — see the note in Part H.

## What these numbers can and cannot decide

**Stable across sample sizes — safe to act on:**

- post-BN beats post-ReLU on separation margin and lowers the different-person ceiling
- consensus scoring lowers the different-person ceiling versus `max(prototype, exemplar)`
- the different-person ceiling sits well above `live.identity.same_camera_threshold` (0.70)

**Not stable — never set a threshold from one run:**

- `other MAX` is an extreme-value statistic and **grows with sample size**. Prefer p95. On the
  same clip, 48 versus 90 frames moved the raw MAX from 0.819 to 0.936.
- the separation margin moved +0.055 → +0.108 between those same two runs
- `KB/frame` for the temp clip is inflated below ~50 frames by container overhead

So: use these scripts to compare **options** (feature tap, scoring mode, NMS threshold) on
identical footage. Actual threshold values come from the Phase 9 sweep on frozen multi-camera
footage, which is also the only way to calibrate `cross_camera_threshold` — every measurement
here is single-camera.

## Footage

Defaults to `register_file.avi`: 2560×1440, matching production resolution, and it decodes with
**zero H.265 reference errors**. The other clips do not:

| Clip | Frames | H.265 reference errors |
|---|---|---|
| `register_file.avi` | 1428 | **0** |
| `test_v2.avi` | 1573 | 207 (13%) |
| `test_file.avi` | 682 | 294 (43%) |

Measuring feature quality on a clip with corrupted frames measures the corruption. Use
`register_file.avi` for anything appearance-related. The corruption itself is issue #28 — no
RTSP transport or timeout option is set anywhere, so packet loss over UDP produces exactly
these errors.

## Known limits of the current sample

Six people, one camera, one clip, 14 proven-distinct pairs. No cross-camera data at all. No
crowded footage. A prior 3-camera RTSP run reconciled 35 tracklets into 7 identities — 5×
fragmentation — and **none of the available clips reproduce that**, so fragmentation fixes
cannot be validated here. See Part E of the plan for the runs that would close these gaps.
