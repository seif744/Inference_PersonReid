# AGENT BRIEF — 2026-08-04. Read this before proposing anything.

Self-contained on purpose. The ADR set is gitignored, so it is **not** in your
clone and not on the A6000 (`CLAUDE.md` §7). Everything you need is restated here.

**This file is TRACKED IN GIT as of 2026-08-04, and that is the point.** It was
hand-pasted into four consecutive sessions because it was a download, not a commit
— the same failure mode `CLAUDE.md` §7 records about the gitignored ADRs:
*"Anything that must survive belongs here, in `ARCHITECTURE.md`, or in
`config.yaml` comments."* It is the only place the 0.574 diagnosis, the
retired-approaches table and the corrected `--reset` rationale are written down.
The load-bearing lines are mirrored into `CLAUDE.md` §3b and §7 so a fresh session
gets them without reading this file at all.

Companion documents, all now tracked: `RECONCILE_PATCHES.md` (the patches),
`CAPTURE_PROTOCOL.md` (the capture), `tools/inspect_tracklet_pairs.py`,
`tests/live/test_same_camera_chain.py`.

---

## 1. State of the world

| Fact | Consequence |
|---|---|
| **CORRECTED 2026-08-04.** An earlier revision of this brief said the Qdrant collection was empty and every measurement was blocked. **That described the DEV BOX.** On the A6000: `collection 'persons': 6017 point(s), dim=2048, metric=Cosine`, with `run_id=20260804_064551` holding **2238 observations**. | **Nothing is blocked.** The measurement tools run today, on the A6000, against that run. That is where the 0.574 table came from. This was a reasoning failure, not a measurement failure: the table's existence was itself proof the run survived, and it went unchecked. |
| Clips + sidecars survive for that run (`._live_src_cam_{213,219,224}.annotations.json`) | `rerender_from_clips.py` works on it. Earlier runs lacked sidecars. Clip + sidecar + stored embeddings is a **complete record** — any setting, any backbone, replayable with zero camera time. |
| **`HEAD` shipped `scoring: consensus` and `threshold: 0.45`** (commit `8bede5e7b2`). Reverted to `prototype`/`0.63` in `279049a34d`. | Landing the revert was a **prerequisite**, not a config change to be avoided. See §4 rule 2, which an earlier revision of this brief got backwards. |
| The dev box `persons` collection genuinely has 0 points | Run nothing measurement-shaped there. `CLAUDE.md` §7: any conclusion about throughput, identity quality or thresholds drawn on the dev box is wrong. |
| `geometry.enabled: false`, `geometry.reconcile.enabled: false` | No positions recorded, veto inert. Geometry is **not** in scope. |
| `identity.reconcile.covisibility.enabled: **true**` | A hard cross-camera veto **is** live. Different switch from geometry. Do not conflate them. |
| `same_camera_reciprocal_best: **true**`, `same_camera_rounds: false` | The cam_224 greedy-fuse path is closed; the cam_206 stranding path is open. |
| `scoring: prototype`, reverted from `consensus` on 2026-08-04 | Sixth reverted tuning change. See §4. |
| cam_206 missing from every run since `20260730_093723` | Must be back in the next capture. It is also the camera with the unmeasured detection-recall complaint under `yolo11m`. |

---

## 2. The diagnosis — established, with numbers

The reported symptom is a **false split**: one person carries one reid in cam_219
and a different reid in cam_224 + cam_213.

From `explain_merge_failure.py 20260804_064551 1 2`, recorded in `config.yaml`:

```
camera    fragment pair     prototype   max_exemplar   consensus   bar
cam_213   0031 vs 0035      0.630 fail   0.663 fail    0.567 fail  0.80
cam_219   0020 vs 0008      0.574 fail   0.600 fail    0.468 fail  0.90
cam_224   0001 vs 0030      0.907 PASS   0.907 PASS    0.582 fail  0.80
```

The chain:

1. In **cam_219**, the person's front and back fragments score **0.574** against
   cam_219's **0.90** same-camera bar. Phase 1 does not merge them.
2. Each fragment is then absorbed cross-camera into a **different** cluster. This
   is the mechanism the `per_camera` comment already documents: a camera that
   cannot merge its own fragments feeds them into separate identities.
3. Both resulting clusters now contain cam_219 members, so `pair_threshold` sees a
   **shared camera** and returns `strictest_same_camera_bar` =
   `max(0.80, 0.90, 0.80)` = **0.90**.
4. The merge that would repair the split must clear **0.90**. It scores nowhere
   near it.

**The gate is a same-camera appearance bar, not a cross-camera one.**

---

## 3. What this retires — do not spend time here

Ruled out **for this symptom**. Each may still be valid for something else; none is
the fix.

| Ruled out | Why |
|---|---|
| Tuning `identity.reconcile.threshold` (the cross bar) | **Never consulted** for this pair. Clusters share cam_219, so the same-camera bar applies. The 0.45→0.63 revert was orthogonal to this bug. |
| Per-camera-mean feature centering (`RECONCILE_PATCHES.md` C2) | Removes a *between-camera* offset. `0020 vs 0008` is a **within-camera** pair — both fragments get the same vector subtracted, so the comparison is unchanged. Cannot be the fix. |
| `same_camera_rounds: true` | Under **complete linkage** round 2's member-pair tests are a subset of round 1's, so rounds can never admit an edge round 1 rejected on score, and 0.574 fails on score. **Qualified 2026-08-04 — see §5 Task 4:** that argument holds only at `member_quorum = 1.0`. Below it, rounds becomes load-bearing. |
| `scoring: consensus` | Measured worse on all three pairs; cam_224 crossed PASS→FAIL. Reverted. |
| `scoring: max_exemplar` | Weakly best of the three, but moves the blocking pair only 0.574 → 0.600. Not sufficient. |
| Anything geometric | A veto can **only refuse** a merge. It is structurally incapable of creating the link that is missing, and can only make a split worse. |
| Lowering cam_219 to 0.80 | **CONTRAINDICATED, not merely insufficient.** Corrected 2026-08-04 from measurement: the sweep on `20260804_064551` showed `cam_219(8)` captured by `cam_219(38)` instead of the operator's person. (`multi` also fell 2 → 1, but that is a cluster count and cannot carry the claim on its own — `CLAUDE.md` §4. The specific mis-capture is the evidence.) It also would not have merged 0.574 anyway. |

**What remains, honestly:** reaching 0.574 needs a bar near 0.55, which is close to
the measured stranger ceiling (p95 0.427, MAX 0.434, **n=6**). That is the
distribution overlap `ARCHITECTURE.md` §6 describes, now measured in FastReID
space. No single number resolves it.

### The measurement that is misleading everyone

`config.yaml`'s same-camera window — *"same-person fragments min=0.810, p5=0.821 …
any bar in (0.434, 0.810) is PERFECT"* — was measured by splitting **one
continuous track into two disjoint halves** on `register_file.avi`. Two halves of
one track share pose, lighting and viewpoint. That is the **easy** case.

The real same-camera merge case is **re-appearance**: someone leaves and returns
facing the other way. The cam_219 pair at **0.574** *is* that case, on real
footage. So the same-person lower edge is ~0.57, not 0.810, and the usable window
is far narrower than the comment claims. Getting that distribution measured
properly is the point of the capture.

---

## 4. Rules that outrank your instincts on this work

Additional to `CLAUDE.md`, which still applies in full.

1. **Propose no threshold change.** Six have been reverted for hurting accuracy.
   `CLAUDE.md` §3.
2. **The `prototype`/`0.63` revert is landed (`279049a34d`); now freeze the
   config.** An earlier revision said "change nothing before the capture." That was
   **backwards**: `HEAD` shipped `consensus`/`0.45`, the pair measured to make the
   defect worse (cam_224's fragment pair crossing 0.907 PASS → 0.582 FAIL).
   `config.yaml`'s own `scoring:` comment records that the operator's front/back
   symptom is what *motivated* the consensus change — so the symptom was observed
   under **prototype**, making `prototype`/`0.63` the configuration worth measuring
   against. This specifically does **not** license lowering cam_219 to 0.80 (§3).
3. **`config.yaml`'s comment blocks are the corpus.** They hold measurements that
   exist nowhere else, including several whose runs are gone. Do not condense,
   reformat or delete them, even where they contradict each other — the
   contradictions are documented history (see the `threshold:` block, which argues
   with itself on purpose and says so).
4. **A cluster count proves nothing.** `CLAUDE.md` §4.
   `sweep_reconcile_thresholds.py` reports cluster *shape*; that is exactly what
   made `consensus` look good before it was measured on fragment pairs. Judge a
   scoring mode on `explain_merge_failure.py`'s fragment-pair table, never on
   identity counts.
5. **Do not build a third decision logger.** `identity/decision_log.py`,
   `logs/verification_decisions.jsonl`, `explain_merge_failure.py` and
   `analyze_decision_log.py` already exist.
6. **"Geometry is off" ≠ "the physical vetoes are off."** `covisibility` is on.
7. **Never paste a placeholder.** `<run_id>` has been run verbatim; `<` is a shell
   redirect. Write commands that derive their values.
8. **`main.py --reset` is not a wipe** — it clears the store *and then runs the
   whole pipeline*, so it looks like an ordinary run. **Never use it.** Corrected
   2026-08-04: an earlier revision justified this with "the store is already
   empty," which was the dev box. On the A6000 `--reset` would destroy the **only
   run in the project with usable ground truth** (`20260804_064551`, 2238
   observations, clips and sidecars intact). The project has already lost its whole
   corpus once this way — `ONBOARDING.md` §7 attributes the 2026-08-03 wipe to
   exactly this command.

---

## 5. Tasks, in order, with acceptance criteria

### Task 0 — Prerequisites

**0a. Land the `prototype`/`0.63` revert. DONE — `279049a34d`.**

Resolved: **nothing was unpushed.** `origin/research..HEAD` was empty;
`git status --short` showed only `M config.yaml`. The geometry subsystem,
`preflight.py`, `backfill_geometry.py`, media time and the
`measure_score_separation` FastReID fix (`56ca40f61e`) were all already in
`origin/research`. `HEAD`'s `consensus`/`0.45` came from `8bede5e7b2`. The revert
was made on the **dev box** (`/home/seif/Projects/Inference`), so the route is
`git push` here → `git pull` there. Not `deploy.sh`.

**Still open — check the A6000's tree BEFORE pulling:**

```bash
git -C ~/seifer_work/Inference_PersonReid status --short
git -C ~/seifer_work/Inference_PersonReid log --oneline -3
git -C ~/seifer_work/Inference_PersonReid diff --stat origin/research
```

Why: `deploy.sh` rsyncs code, so the server can hold a `config.yaml` that arrived
by rsync while its git state sits elsewhere. If the tree is dirty or behind, a pull
either conflicts or clobbers local edits. Report the state; do not pull blind.

*One inference not to rely on:* it was argued that because `56ca40f61e` is an
ancestor of `8bede5e7b2`, a clean pull carrying `consensus/0.45` must also carry
the `measure_score_separation` fix — yet the script was seen crashing on pre-fix
code, implying a dirty server tree. `56ca40f61e` is dated **2026-08-04 10:36:24
+0530** and run `20260804_064551` is timestamped 06:45, so the crash may simply
predate the fix. **Suggestive, not established.** The state check above settles it.

The operator decides whether to push.

**0b. Place the four companion files. DONE — `d3e695641e`.** All four arrived as
browser downloads in the repo root, so two were misplaced and each carried a
`:Zone.Identifier` sidecar (WSL surfacing the NTFS alternate data stream).

`test_same_camera_chain.py` **broke `tests/run_all.py` as delivered**: it resolved
`src/` with a cwd-relative `sys.path.insert(0, "src")`, which passes standalone
from the repo root and raises `ModuleNotFoundError` under `run_all.py`, since that
deliberately runs each test with `cwd = the test's own directory`. `discover()`
globs `tests/**/test_*.py`, so the file joined the suite and took it from PASS to
FAIL. Fixed to resolve from `__file__`. This is the §7 breakage class exactly, and
it would have hit every fresh clone.

### Task 1 — ANSWERED, no action

`config.yaml`'s `decision_log` and `top2_margin.{threshold,basis}` are **not**
returned by `resolve_reconcile_kwargs` — 12 keys, none of those three — but they
are **not silently dropped**. `pipeline.py::_decision_log_kwargs()` reads all
three and splats them in at the `reconcile_tracklets` call. The split is
deliberate: the resolver returns merge *policy*, while a `DecisionLog` is a live
object needing a `run_id` and a close, which cannot be a plain config value.

**Residual finding, worth recording:** `main.py` passes **no** decision log at all,
so **file-batch reconcile runs unlogged** while the live path is logged. That is
the file/live asymmetry `REMEDIATION_PLAN.md` Part M exists to track. Report it
there; do not fix it as a side quest.

### Task 2 — ANSWERED, no action

A6000: `collection 'persons': 6017 point(s), dim=2048, metric=Cosine`, with
`run_id=20260804_064551` holding 2238 observations, plus surviving clips and
sidecars for cam_213 / cam_219 / cam_224. Dev box: 0 points. Correct width, no
stale 512-d collection. **Nothing is blocked.**

### Task 3 — DONE — `d7bcbddd3a`

All 13 `OLD` anchors matched exactly once. `tests/run_all.py` 19/19, matching the
pre-patch baseline. **A9 broke nothing** — no test relied on the `1` default, so
nothing was pinning that drift. A7 and A8 verified by observable output rather than
by the suite (malformed covisibility entries now skipped visibly instead of
producing a two-character camera pair or an uncaught `KeyError`; `cap=` and
`safety=`/`clock=` now appear in the settings line).

**A2 and A3 were applied** under an earlier revision of `RECONCILE_PATCHES.md` that
said "apply anyway"; the current revision says skip them for the smaller diff.
Leave them. They are correct code, merely inert while geometry is off. Reverting is
churn.

### Task 4 — DONE — `f3c50842ee`. **PREDICTION REFUTED.**

First run, before C1 (`member_quorum` unavailable, complete linkage fixed at 1.0):
the prediction held — no setting closed the chain while keeping the stranger out.
Ran clean, with no import or signature mismatch.

**After applying C1, two settings achieve both:**

```
scoring       recip rounds quorum  ids chain stranger
prototype     True  True   1.00     3  NO    SEPARATE
prototype     True  True   0.60     2  NO    FUSED
max_exemplar  True  True   0.60     2  YES   SEPARATE   <== both
consensus     True  True   0.60     2  YES   SEPARATE   <== both
```

Two corrections follow, and both matter more than the win itself:

1. **`same_camera_rounds` IS load-bearing — but only below quorum 1.0.** The
   first grid showed every `rounds=True` row identical to `rounds=False`, and the
   subset argument explains why: under complete linkage, round 2 re-tests exactly
   round 1's pairs. That argument **stops holding at quorum < 1.0**, because a
   2-of-3 quorum can admit a cluster edge no 1-of-1 pairwise test would.
   `quorum=0.6` with `rounds=False` still gives 3 ids / chain NO — it takes
   **both** flags. So "rounds cannot fix the cam_206 stranding" is true only at
   complete linkage.
2. **The scoring column is not wholly an artifact of `SPREAD_DEG=0`.** The
   docstring's caveat is right for *singleton* comparisons — zero intra-tracklet
   variance makes all three modes agree there — but it does not extend to
   **multi-member clusters**, whose members are distinct vectors however tight each
   tracklet is. The winning rows are exactly the multi-member ones, and `prototype`
   diverges from `max_exemplar`/`consensus` there (FUSED vs SEPARATE). Real, not
   fixture noise. Still 4 synthetic tracklets, and says nothing yet about cam_206.

### C1 — RESOLVED: apply the code, do not flip the switch

The fixture's closing line said "apply C1 and re-run"; §4 and Task 3 said "do not
apply Part C." **The second was over-broad.** It was written against C2 (camera
centering) and C3 (consensus variant), both of which change scores.

`same_camera_member_quorum` defaults to **1.0**, which is complete linkage —
bit-identical to today. Applying that code changes **no decision**.

- **Sanctioned, and DONE:** apply C1, re-run the fixture, record the answer.
- **Not sanctioned, and NOT done:** setting the quorum below 1.0 in `config.yaml`.
  That voids the bars, needs a sweep and a re-render, and targets the **cam_206**
  stranding, not the operator's reported split. The key is deliberately absent from
  `config.yaml`; the resolver's default supplies 1.0.

### Task 5 — Measure on `20260804_064551`, on the A6000

Runs today against the surviving run and its clips. No room time.

```bash
RUN=20260804_064551

# 1. Re-render under prototype and WATCH IT. The only way to learn which reids the
#    operator carries -- everything below needs those two numbers. Every one of
#    these tools prints the settings in force on its first line. If that line says
#    consensus, STOP: the pull has not landed and you are watching the wrong
#    configuration.
python tests/calibration/rerender_from_clips.py "$RUN"

# 2. The operator's own split, under the reverted scoring. Substitute the two reids
#    read off the video above.
python tests/calibration/explain_merge_failure.py "$RUN" <reid_a> <reid_b>

# 3. Camera bias. Section 2 is the measurement this project has never taken:
#    camera-mean cosines, the across-camera variance profile, and the LABEL-FREE
#    same-camera neighbour rate.
python tools/inspect_tracklet_pairs.py --run "$RUN"

# 4. The 1961-exclusion question (section 6).
python tests/calibration/analyze_decision_log.py "logs/reconcile_decisions_$RUN.jsonl"
```

**Accept when:** you can state (a) which reids the operator carries and whether the
split reproduces under prototype, (b) the same-camera neighbour rate per camera
against each camera's share of tracklets, and (c) how many candidates
`TEMPORAL_CONFLICT_CROSS_CAMERA` excluded and between which camera pairs.

Note section 3 of `inspect_tracklet_pairs.py` scores **singleton** cross-camera
pairs, so it cannot show the gate that blocked the split — once both clusters share
cam_219, `strictest_same_camera_bar` applies instead. Only
`explain_merge_failure.py` shows that. The script prints this caveat itself.

### Task 6 — Mine re-appearance pairs from the surviving clips

The clips carry every box **and its `track_id`**, so time-disjoint same-camera
fragments of one person can be identified by watching and recorded with provenance:

```bash
python tests/calibration/review_links.py "$RUN" --label
```

This is the 0.574 distribution — the same-camera *re-appearance* case, as opposed
to the split-one-track-in-half case the published 0.810 window came from. Every
label is permanent project capital; there are currently **11** in the whole
project's history and their runs are gone.

Labels belong in `calibration/link_labels.jsonl`, never in a conversation.

### Task 7 — The capture (not the critical path)

See `CAPTURE_PROTOCOL.md`. It supplies four things the existing run genuinely
cannot: **cam_206**, a **written headcount**, a **known route**, and deliberate
re-appearance repeats. New information, not unblocking. Do it after Tasks 5 and 6,
so the capture is designed around what those show.

Agent's part is pre-flight only: `.env`, `source.env_urls`, `preflight.py`, and
confirming cam_206 produces detections at all. No threshold edits.

---

## 6. Covisibility — a background question, NOT a suspect

**Corrected 2026-08-04.** An earlier revision titled this "the open suspect nobody
has measured." That overstated it. `explain_merge_failure.py` on `20260804_064551`
named the **appearance bar** as the gate that blocked the operator's split.
Covisibility did not cause the reported symptom. Do not open a workstream here.

A distinction that keeps getting collapsed: **covisibility is not geometry.**
`identity.reconcile.covisibility` is a temporal simultaneity veto using only `ts`
and a configured camera-pair table — no positions, no homography, no floor frame,
no calibration. `src/geometry/` is a *distance* veto and is entirely disabled. Two
switches, two config blocks, two code paths. "Geometry is off" says nothing about
covisibility, which is **on**.

What remains genuinely unexamined, worth exactly one line of output:

- `reconcile.py`'s comment records **1961** `TEMPORAL_CONFLICT_CROSS_CAMERA`
  candidate exclusions on run `20260731_060425` — a different, older, 4-camera run.
  Nobody has looked at what they were.
- The `live.topology` veto was disabled because *"the view-to-view handoff is
  FASTER than the walking distance suggests (adjacent/overlapping fields of
  view)."* If handoff is effectively instant, a person genuinely **can** appear in
  two cameras inside one second — and `ts` is receive time, carrying network
  jitter, decode-cost differences and frame-rate quantisation (ADR-003A §3.1 puts
  the differential as high as 500 ms). So the evidence that killed the transit veto
  argues the 1.0s tolerances are tight, on a veto that is hard and
  **unrecoverable**: reconcile cannot un-split a person whose merge it refused.

`analyze_decision_log.py` in Task 5 already prints this count. Read it, record it,
move on. Change the tolerance only if the count is large *and* concentrated on a
camera pair where fast handoff is plausible — and then alone, with a re-render.

---

## 7. What is still genuinely open

- Whether the same-camera bar can be lowered enough to catch re-appearance
  (~0.55) without fusing strangers. Needs the capture's labelled pairs. The
  current stranger sample is **n=6**.
- What the 1961 covisibility exclusions were (§6).
- Whether `max_exemplar` helps anything, judged on fragment pairs and on video.
  Task 4 gives it a synthetic point in its favour at `quorum < 1.0`; that is not
  evidence about footage.
- Whether cam_219 is optically different from cam_224/cam_213 — never checked, and
  it is a five-minute side-by-side frame comparison.
- Whether the split appears in reconciled ids only, or also in live provisional
  ids. Different bugs, similar-looking outputs. **Check `output_cam_*.mp4` mtime
  against the `run_id`** — stale files look like success.
