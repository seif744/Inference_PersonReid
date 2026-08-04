"""
reconcile.py  --  STAGE 7b:  OFFLINE identity reconciliation.

============================ WHY THIS EXISTS ==============================
The live IdentityService (service.py) decides a track's global_id ONCE, on the
track's first observation, and never revisits it (stickiness prevents id flicker
frame-to-frame). That is correct for a single stream, but it has a blind spot the
moment two cameras run at once:

    A person who enters BOTH cameras at the same instant is embedded and decided
    in each camera before the other camera has committed anything to the gallery.
    Both cameras therefore MINT -- and, because the decision is sticky, they can
    never later discover they are the same person. The match window never existed.

That is exactly the failure this pass repairs. AFTER all camera workers finish
and the gallery is fully populated, we look back over every global_id, build an
appearance prototype for each, and MERGE ids that are the same person. This is a
batch, whole-gallery view the live path structurally cannot have.

------------------------------- SAFETY --------------------------------------
Merging is the most destructive identity error (it fuses two people's histories),
so this pass is conservative, matching service.py's "prefer a split" bias:

  * APPEARANCE GATE: two ids merge only if their prototype cosine >= threshold
    (default: the same threshold the live service uses). The SAME-CAMERA bar is
    PER-CAMERA (`identity.reconcile.per_camera`), because one number is harmless
    in one view and total in another: at a global 0.90 cam_213 achieved ZERO
    same-camera merges across all 11 of its subjects while cam_206 saw 9.0
    eligible partners per subject. A camera with no same-camera merging cannot
    join one person's front-view and back-view fragments, so each is absorbed
    cross-camera into a DIFFERENT cluster -- the "reid 2 becomes reid 7" symptom.
    See REMEDIATION_PLAN.md J.6 and item #40.
  * PHYSICAL EXCLUSION: two ids are NEVER merged if they are co-present in the
    SAME camera (their tracks overlap in time). One body cannot be two
    simultaneous tracks in one view, so such a pair is provably two people --
    regardless of how alike they look (twins, uniforms). This also correctly
    still ALLOWS merging disjoint fragments of one person in a single camera
    (a dropped-then-reacquired track), which fixes over-counting too.
  * TRANSITIVE-SAFE: clusters grow by union only when NO member of one side
    conflicts with ANY member of the other, so a chain of merges can never
    sneak two co-present ids into the same identity.
  * RECIPROCAL BEST MATCH (optional, on by default): a pair merges only if each
    id is the OTHER's single best above-threshold partner. On out-of-domain
    footage the appearance model compresses scores -- many different people land
    in the same 0.8-0.9 band as the true match -- and a lone threshold then
    over-merges. Requiring mutual nearest-neighbour keeps the one real pair and
    rejects the look-alike crowd. Turn off only with a model whose same/different
    scores are cleanly separated (see measure_reconcile_thresholds.py).
  * GEOMETRIC REACHABILITY: two tracklets never merge if doing so would require a
    person to move faster than anyone on this floor has been observed to move.
    Covers both "two places at the same instant" and "could not have got there in
    time" with one formula (distance / elapsed).

---------------------- GEOMETRY IS CONSUMED, NEVER COMPUTED -------------------
This module reads the floor positions the LIVE RUN recorded into each observation's
payload. It does NOT load a calibration, apply a homography, or derive a position
from a bounding box -- the only import it takes from `geometry/` is the arithmetic
in `geometry.reachability`, and `tests/live/test_geometry_not_recomputed.py`
asserts that stays true.

WHY THAT MATTERS ENOUGH TO BE A RULE. The live RTSP feed is never recorded, so a
position that was not written during the run is gone for good. And a reconcile that
re-derived positions would silently produce DIFFERENT geometry from the run it is
reconciling the moment anyone re-fits a calibration -- so two reconciles of one run
could disagree, with nothing in either output to say why. One run, one geometry,
decided once, at capture.

Deterministic: the surviving id of a merged cluster is its smallest global_id,
and vectors/point-ids are never touched -- only the global_id payload is
re-stamped. Safe to re-run (idempotent): a reconciled gallery has no pairs left
to merge.
============================================================================
"""

import bisect
from collections import defaultdict

import numpy as np

from identity import decision_log as dlog
from identity.decision_log import Candidate, DecisionRecord, GateResult
# The ONLY geometry import allowed here: pure arithmetic over positions the live
# run already recorded. `geometry.floor` and `geometry.calibration` -- the modules
# that could turn a box into a position -- must never appear in this file.
from geometry.reachability import (IMPOSSIBLE, RecordedPosition, SpeedEnvelope,
                                   observed_speed_ceiling)


# Keys `identity.reconcile.per_camera[<cam>]` is allowed to override. Anything else
# in that block is IGNORED and warned about rather than silently doing nothing --
# `threshold` is a property of a camera PAIR, not a camera, and the remaining
# reconcile keys are whole-run settings. Extend this tuple (and the lookup that
# reads it) when a second key genuinely becomes per-camera.
PER_CAMERA_KEYS = ("same_camera_threshold",)


def resolve_same_camera_thresholds(recon_cfg, log=print):
    """`identity.reconcile.per_camera` -> {camera: same_camera_threshold}.

    The one place this merge happens, so the live path and the file-batch path can
    never drift apart on it -- the same argument as detector.resolve_detector_cfg.
    Both call sites pass the result to reconcile_tracklets as
    `same_camera_thresholds`; a camera absent from the result keeps the global
    `same_camera_threshold`.

    FAIL-SOFT AND LOUD: every malformed entry is skipped with a message and the
    global value stays in force. A bad threshold config must never be able to cost
    a run its identities, and must never be able to change merging silently -- an
    unnoticed `80` where `0.80` was meant would disable same-camera merging for
    that camera completely, which is exactly the failure this item exists to fix.
    """
    per_camera = (recon_cfg or {}).get("per_camera") or {}
    out = {}
    if not isinstance(per_camera, dict):
        log(f"  [reconcile] identity.reconcile.per_camera must be a mapping of "
            f"camera -> overrides, got {type(per_camera).__name__} -- ignored.")
        return out
    for cam, overrides in per_camera.items():
        if not isinstance(overrides, dict):
            log(f"  [reconcile] per_camera[{cam!r}] must be a mapping of "
                f"key -> value, got {type(overrides).__name__} -- ignored.")
            continue
        unknown = [k for k in overrides if k not in PER_CAMERA_KEYS]
        if unknown:
            log(f"  [reconcile] per_camera[{cam!r}]: {unknown} is NOT honoured "
                f"per-camera (only {list(PER_CAMERA_KEYS)}) -- ignored, the global "
                f"value applies.")
        if "same_camera_threshold" not in overrides:
            continue
        raw = overrides["same_camera_threshold"]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            log(f"  [reconcile] per_camera[{cam!r}].same_camera_threshold={raw!r} "
                f"is not a number -- ignored, the global value applies.")
            continue
        # A same-camera bar outside [0, 1] is a typo, not a choice -- an unnoticed
        # `80` where `0.80` was meant disables same-camera merging for that camera
        # entirely, which is the failure this whole resolver exists to prevent.
        # NOTE the old justification here -- "the shipped embedding is post-ReLU" --
        # is NO LONGER TRUE. FastReID's post-bnneck features are ~60.5% negative
        # dimensions, so a cosine can legitimately be negative. The clamp stays as a
        # typo guard; nothing may assume non-negativity from it.
        if not 0.0 <= value <= 1.0:
            log(f"  [reconcile] per_camera[{cam!r}].same_camera_threshold={value} "
                f"is outside [0, 1] (it is a cosine) -- ignored, the global value "
                f"applies.")
            continue
        out[str(cam)] = value
    return out


COVISIBLE = "covisible"


def resolve_covisibility(recon_cfg, log=print):
    """`identity.reconcile.covisibility` -> (enabled, {frozenset(pair): tolerance}).

    A tolerance of None means the pair is CO-VISIBLE: both cameras can genuinely
    see one person at the same moment, so simultaneous tracklets prove nothing and
    are never vetoed. A number means "veto when the two tracklets overlap in time
    by MORE than this many seconds".

    FAIL-OPEN, deliberately (decision D1): a pair that is absent from the config is
    unconstrained. Silence loses protection rather than splitting a real person in
    half, because a wrong veto is worse than a missing one -- it manufactures a
    second identity for someone who only has one.
    """
    cfg = (recon_cfg or {}).get("covisibility") or {}
    enabled = bool(cfg.get("enabled", False))
    default_tol = cfg.get("default_tolerance_sec", 1.0)
    pairs = {}
    for entry in (cfg.get("pairs") or []):
        # EXPLICIT TYPE CHECK. A bare string indexes cleanly ("abc" -> 'a','b','c')
        # so a YAML typo silently became a camera pair named after two characters;
        # a dict raised KeyError, which was not caught, so it killed the run.
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            log(f"  [reconcile] covisibility: skipping malformed pair {entry!r} "
                f"(expected a 3-item list [cam_a, cam_b, seconds|'covisible']).")
            continue
        a, b, spec = entry[0], entry[1], entry[2]
        key = frozenset((str(a), str(b)))
        if isinstance(spec, str) and spec.strip().lower() == COVISIBLE:
            pairs[key] = None
            continue
        try:
            pairs[key] = float(spec)
        except (TypeError, ValueError):
            log(f"  [reconcile] covisibility: {a}<->{b} tolerance {spec!r} is "
                f"neither a number nor {COVISIBLE!r} -- using the default "
                f"{default_tol}s.")
            pairs[key] = float(default_tol)
    return enabled, pairs


def resolve_geometry_policy(cfg, log=print):
    """Whole config -> the `geometry` POLICY dict reconcile_tracklets expects.

    POLICY ONLY -- three numbers that say how hard to bite. No calibration, no
    homography, no positions: those were recorded by the live run and travel in the
    observation payloads (geometry/__init__.py invariant 1). That separation is why
    this resolver never touches `geometry.calibration_path`.

    `geometry.reconcile.enabled` gates the veto independently of
    `geometry.enabled`, which gates RECORDING. The asymmetry is deliberate: recording
    is additive and safe to leave on, while the veto changes identities and should be
    switchable without losing a run's geometry.
    """
    gcfg = (cfg or {}).get("geometry") or {}
    rec = gcfg.get("reconcile") or {}
    enabled = bool(rec.get("enabled", False))
    return {
        "enabled": enabled,
        # Cross-camera timestamp uncertainty. `ts` is decode time, not sensor time,
        # so two cameras' stamps for one instant differ by their buffering. Added to
        # every elapsed time, which can only make the veto more permissive.
        "clock_error_sec": float(rec.get("clock_error_sec", 0.5)),
        # Multiplies the measured speed ceiling before it becomes a veto line.
        "safety_factor": float(rec.get("safety_factor", 1.5)),
    }


def resolve_reconcile_kwargs(cfg, log=print):
    """Whole config -> the kwargs `reconcile_tracklets` should be called with.

    THE ONE PLACE reconcile's settings are read, for the same reason
    resolve_same_camera_thresholds and resolve_covisibility exist -- except that
    those two only covered the two newest keys, and the FOUR call sites went on
    reading the rest independently. They had already drifted:

      * main.py defaulted `min_tracklet_observations` to 1, the live pipeline to 3
      * main.py fell back to identity.threshold=0.85, the live pipeline to 0.63
      * tests/calibration/sweep_reconcile_thresholds.py and rerender_from_clips.py
        passed NEITHER `covisibility` NOR `same_camera_reciprocal_best`, so both
        defaulted OFF while production ran with both ON

    That last one is the expensive one: the offline sweep and the offline
    re-render are the ONLY cheap feedback loop this project has, and they were
    measuring a different clustering algorithm than the one that ships. Every
    threshold conclusion drawn from them was drawn about a run that never
    happened. See REMEDIATION_PLAN.md Part M.

    Returns the kwargs only; `store`, `run_id`, `quality_out`, `decision_log` and
    `log` stay with the caller because they are per-run, not configuration.
    """
    cfg = cfg or {}
    recon = (cfg.get("identity") or {}).get("reconcile") or {}
    threshold = recon.get("threshold")
    if threshold is None:
        threshold = (cfg.get("identity") or {}).get("threshold", 0.63)
    return {
        "threshold": float(threshold),
        "same_camera_threshold": float(recon.get("same_camera_threshold", 0.90)),
        "same_camera_thresholds": resolve_same_camera_thresholds(recon, log=log),
        "require_reciprocal_best": bool(recon.get("require_reciprocal_best", True)),
        "min_tracklet_observations": int(recon.get("min_tracklet_observations", 3)),
        "scoring": recon.get("scoring", PROTOTYPE),
        "consensus_top_frac": float(recon.get("consensus_top_frac", 0.25)),
        "max_observations_per_side": int(recon.get("max_observations_per_side", 64)),
        "covisibility": resolve_covisibility(recon, log=log),
        "same_camera_reciprocal_best": bool(
            recon.get("same_camera_reciprocal_best", False)),
        "same_camera_rounds": bool(recon.get("same_camera_rounds", False)),
        "geometry": resolve_geometry_policy(cfg, log=log),
    }


def describe_reconcile_kwargs(kw):
    """One-line summary of the settings actually in force.

    Printed by every offline tool so a sweep's output can never again be read as
    if it described production. The three switches that silently differed are all
    here by name.
    """
    covis_on, covis_pairs = kw.get("covisibility") or (False, {})
    per_cam = kw.get("same_camera_thresholds") or {}
    return (
        f"cross={kw['threshold']:.2f} "
        f"same={kw['same_camera_threshold']:.2f}"
        + (" (" + ", ".join(f"{c}={v:.2f}" for c, v in sorted(per_cam.items())) + ")"
           if per_cam else "")
        + f" scoring={kw['scoring']}"
        + (f"/top_frac={kw['consensus_top_frac']}" if kw["scoring"] == CONSENSUS else "")
        + f" min_obs={kw['min_tracklet_observations']}"
        f" reciprocal={'on' if kw['require_reciprocal_best'] else 'OFF'}"
        f" same_reciprocal={'on' if kw['same_camera_reciprocal_best'] else 'OFF'}"
        f" same_rounds={'on' if kw.get('same_camera_rounds') else 'OFF'}"
        f" covisibility={'on/' + str(len(covis_pairs)) + ' pairs' if covis_on else 'OFF'}"
        + f" cap={kw.get('max_observations_per_side')}"
        + (f" geometry=on/safety="
           f"{(kw.get('geometry') or {}).get('safety_factor')}"
           f"/clock={(kw.get('geometry') or {}).get('clock_error_sec')}"
           if (kw.get('geometry') or {}).get('enabled') else " geometry=OFF"))


def temporal_overlap_sec(span_a, span_b):
    """Seconds two wall-clock spans overlap. NEGATIVE when they do not (the size of
    the gap), None if either span is missing -- callers compare `> tol`, so a
    negative value fails every tolerance, which is the intended fail-open. A
    single-observation tracklet has a zero-length span, so it can never register an
    overlap."""
    if not span_a or not span_b:
        return None
    return min(span_a[1], span_b[1]) - max(span_a[0], span_b[0])


def strictest_same_camera_bar(cameras, per_camera, default):
    """The same-camera bar that applies to a claim about `cameras`.

    One camera -> its own bar. SEVERAL cameras (two clusters that share more than
    one camera) -> the STRICTEST of them, because merging asserts the same-person
    claim separately for every shared camera and all of them have to hold. Taking
    the minimum instead would let a camera calibrated loose (cam_213 at 0.80)
    launder a merge past a camera calibrated tight (cam_206 at 0.90) -- a false
    merge in the tight camera, and merging two people is the most destructive
    identity error, so this stays on reconcile's "prefer a split" side.

    An empty `cameras` cannot make a same-camera claim; callers must not rely on
    the return value in that case, and it degrades to the global default.
    """
    return max((per_camera.get(c, default) for c in cameras), default=default)


# ---------------------------------------------------------------- scoring modes
#
# WHY MORE THAN ONE EXISTS (plan #45/#45a). Reconcile has always compared
# prototype MEANS, and a mean is the wrong summary for a person who changes
# appearance mode. Someone seen front-on and then from behind has two clusters of
# observations, and their mean sits between them, matching NEITHER view. So that
# person's own two fragments can score LOW, while two different people in similar
# clothing -- whose means both sit in the same "average person" region -- score
# HIGH. The comparison is biased in exactly the wrong direction, which is why no
# single threshold both merges one person's fragments and rejects strangers: J.6
# measured the per-subject boundaries OVERLAPPING (p95 of the top "different"
# score 0.816 above p5 of the worst "same" score 0.719).
#
# The live engine already avoids this and says so in ActiveIdentitySet.score:
# "medoid-style ... more robust than a mean alone when a person flips front/back
# or pose-shifts". Reconcile never got that fix.
#
# CHANGING THE MODE VOIDS EVERY THRESHOLD, exactly as changing the feature tap
# does (Part H). Each mode has its own scale; re-derive the bars with
# tests/calibration/sweep_reconcile_thresholds.py --scoring <mode> before reading
# anything into a number.
PROTOTYPE = "prototype"
MAX_EXEMPLAR = "max_exemplar"
CONSENSUS = "consensus"
SCORING_MODES = (PROTOTYPE, MAX_EXEMPLAR, CONSENSUS)


def _unit_rows(vectors):
    """(N, D) of L2-normalized rows."""
    mat = np.stack(vectors).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


def _subsample_rows(mat, cap):
    """At most `cap` rows, evenly spaced -- DETERMINISTIC, so a replay reproduces
    the same score. A long tracklet can hold hundreds of observations and the
    cross-observation product is O(n*m); evenly spaced keeps the pose/lighting
    variety a random sample would keep, without the nondeterminism."""
    if cap <= 0 or mat.shape[0] <= cap:
        return mat
    idx = np.linspace(0, mat.shape[0] - 1, cap).round().astype(int)
    return mat[idx]


def score_observation_sets(rows_a, rows_b, proto_a, proto_b,
                           mode=PROTOTYPE, top_frac=0.25, cap=64):
    """Similarity between two sets of observations, under one scoring mode.

    prototype     -- cosine of the two means. What reconcile has always used.
    max_exemplar  -- max(prototype cosine, best single observation pair). Answers
                     "is there ANY view of these two that matches", which is what
                     recognises a person's front against their own back. Its known
                     risk is the mirror image: one bad crop can match a stranger
                     at ~1.0, a defect already recorded for the live bank.
    consensus     -- mean of the top `top_frac` of all observation pairs. A middle
                     course: it needs MANY matching view pairs rather than one, so
                     a single bad crop cannot carry a merge. Part H measured it
                     with the LOWEST different-person ceiling of the three modes at
                     both sample sizes.

                     `top_frac` MATTERS MORE THAN IT LOOKS. Part H's "mean of the
                     top half" was measured comparing whole tracks, where most
                     view pairs match. Between a mostly-front fragment and a
                     mostly-back one only a minority of pairs match at all -- in
                     the test fixture, 32 of 100 -- so averaging the top HALF
                     necessarily averages in mismatched pairs and buries the
                     signal (0.727 where the matching pairs all sit at 1.000).
                     Hence the 0.25 default. It is a floor on "how much of the two
                     tracklets has to agree", so it trades directly against
                     robustness: too low and it degenerates toward max_exemplar
                     and its single-bad-crop failure.
    """
    proto = float(proto_a @ proto_b)
    if mode == PROTOTYPE:
        return proto
    sims = _subsample_rows(rows_a, cap) @ _subsample_rows(rows_b, cap).T
    if mode == MAX_EXEMPLAR:
        return max(proto, float(sims.max()))
    if mode == CONSENSUS:
        flat = np.sort(sims.ravel())[::-1]
        k = max(1, int(round(flat.size * float(top_frac))))
        return float(flat[:k].mean())
    raise ValueError(f"unknown reconcile scoring mode {mode!r}; "
                     f"expected one of {list(SCORING_MODES)}")


def _prototype(vectors):
    """Mean of L2-normalized vectors, renormalized -- the id's appearance center."""
    mat = np.stack(vectors).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-12, None)
    proto = mat.mean(axis=0)
    n = np.linalg.norm(proto)
    if n <= 0:
        return None
    return proto / n


def _spans_disjoint(span_a, span_b):
    """True when two same-camera tracklets do not overlap in time."""
    return span_a[1] < span_b[0] or span_b[1] < span_a[0]


def _gather_tracklets(store, run_id):
    """
    Read gallery observations grouped by camera-local tracklet.

    Returns tracklet -> {
        vectors: [embedding],
        points: [point ids],
        span: (min_frame, max_frame),
        span_ts: (min_ts, max_ts) or None,
        gids: {original global ids},
        floor: [(ts, x, y, error, group)] sorted by ts -- RECORDED positions only,
    }.
    """
    data = defaultdict(lambda: {
        "vectors": [],
        "points": [],
        "frames": [],
        "times": [],
        "gids": set(),
        "floor": [],
    })

    # #20: filter run_id SERVER-SIDE. This used to scroll the ENTIRE collection
    # with_vectors=True and discard other runs in Python -- so every run paid to
    # download every previous run's 512-d vectors. The store held 4157 points
    # before one run and 15892 a few runs later; that cost grows forever and lands
    # squarely in finalization, where the operator is waiting. Falls back to the
    # old client-side filter if the server rejects the filter (embedded mode, or a
    # client version without it), so this can never cost a run its identities.
    scroll_filter = None
    if run_id is not None:
        try:
            from qdrant_client import models as _qmodels
            scroll_filter = _qmodels.Filter(must=[_qmodels.FieldCondition(
                key="run_id", match=_qmodels.MatchValue(value=run_id))])
        except Exception:                                       # noqa: BLE001
            scroll_filter = None

    offset = None
    while True:
        try:
            pts, offset = store.client.scroll(
                store.collection, limit=1000, offset=offset,
                scroll_filter=scroll_filter,
                with_payload=True, with_vectors=True)
        except Exception:                                       # noqa: BLE001
            scroll_filter = None
            pts, offset = store.client.scroll(
                store.collection, limit=1000, offset=offset,
                with_payload=True, with_vectors=True)
        for p in pts:
            pl = p.payload or {}
            if run_id is not None and pl.get("run_id") != run_id:
                continue
            camera = pl.get("camera")
            track_id = pl.get("track_id")
            frame = pl.get("frame")
            if camera is None or track_id is None or frame is None:
                continue

            vector = p.vector
            if isinstance(vector, dict):
                vector = next(iter(vector.values()), None)
            if vector is None:
                continue

            key = (camera, int(track_id))
            data[key]["vectors"].append(np.asarray(vector, dtype=np.float32))
            data[key]["points"].append(p.id)
            data[key]["frames"].append(int(frame))
            # WALL-CLOCK span (#26). Frame indices are per-camera and NOT
            # comparable across cameras running at different rates -- 15 fps and
            # 25 fps put frame 1000 forty seconds apart. Any cross-camera timing
            # rule therefore needs `ts`, which the store has always carried and
            # reconcile has never read. Missing/garbage ts leaves `times` short and
            # the tracklet keeps frame-index spans only.
            ts = pl.get("ts")
            if ts is not None:
                try:
                    data[key]["times"].append(float(ts))
                except (TypeError, ValueError):
                    pass

            # The floor position the LIVE RUN recorded for this observation. We
            # READ it; we never derive it. There is deliberately no bbox ->
            # position code path anywhere in this module -- see
            # geometry/__init__.py invariant 1 and
            # tests/live/test_geometry_not_recomputed.py.
            #
            # Observations stored before geometry existed, or by a run with it
            # disabled, simply have no `floor` key. That leaves `floor` short and
            # every geometric rule then fails open, which is the same contract
            # `span_ts` already uses for a missing wall clock.
            pos = RecordedPosition.from_payload(pl)
            if pos is not None and pos.ts is not None:
                data[key]["floor"].append(
                    (pos.ts, pos.x, pos.y, pos.error, pos.group))

            # Explicit, not a dict default: a payload carrying reid_id=null would
            # take the None and never fall through to global_id.
            gid = pl.get("reid_id")
            if gid is None:
                gid = pl.get("global_id")
            if gid is not None:
                data[key]["gids"].add(int(gid))

        if offset is None:
            break

    out = {}
    for key, info in data.items():
        if not info["vectors"]:
            continue
        frames = info["frames"]
        times = info["times"]
        # OBSERVATION ORDER IS CAPTURE ORDER, not Qdrant scroll order. `scroll`
        # returns points in point-id order, so `vectors` / `points` arrived in an
        # order with no temporal meaning -- which made _subsample_rows' "evenly
        # spaced" sampling evenly spaced in ID space, and made its determinism
        # claim rest on the store's internal ordering rather than on anything this
        # module controls. Sorted by FRAME INDEX (always present; `ts` can be
        # missing), so `vectors`, `points` and `frames` are INDEX-ALIGNED.
        # `times` is deliberately NOT in that alignment -- it is only appended when
        # a usable ts exists, so it can be shorter -- and stays its own sorted list.
        order = sorted(range(len(frames)), key=lambda i: frames[i])
        vectors = [info["vectors"][i] for i in order]
        points = [info["points"][i] for i in order]
        frames = [frames[i] for i in order]
        out[key] = {
            "vectors": vectors,
            "points": points,
            "span": (frames[0], frames[-1]),
            # None when no usable ts was stored, which is the signal every
            # ts-dependent rule uses to fail open rather than guess.
            "span_ts": ((min(times), max(times)) if times else None),
            # The SAMPLED instants themselves, not just their envelope. `span` is
            # min/max over observations taken every reid.interval_sec, so a tracklet
            # with a hole in the middle claims the whole envelope -- and the
            # same-camera veto then reads a gap as occupancy. Anything that wants to
            # ask "were these two ever really co-present" needs the instants (Part
            # M.2); explain_merge_failure.py does exactly that.
            "frames": frames,          # already sorted above, with vectors/points
            "times": sorted(times),
            "gids": info["gids"],
            # Sorted by ts so the reachability check can pair observations by
            # nearest time with a binary search instead of a scan.
            "floor": sorted(info["floor"]),
        }
    return out


def _cluster_prototype(members, protos):
    return _prototype([protos[m] for m in members])


# ------------------------------------------------------ geometric reachability

# How many observation pairings a verdict needs. Below this the check reports no
# opinion. MV3DT uses minCommonFrames4MatchScore = 2 for the same reason: with one
# shared instant a geometric "verdict" is an accident of sampling.
MIN_GEOMETRY_PAIRINGS = 3

# How many of the temporally CLOSEST pairings the verdict is taken over. Bounded
# because the median has to stay anchored to the moment that actually constrains
# the merge: for time-overlapping tracklets that is the overlap, and for disjoint
# ones it is the handover boundary. Include every pairing instead and a long
# tracklet's far-apart-in-time pairs (which imply no speed at all) would drown the
# boundary and the check would never fire.
MAX_GEOMETRY_PAIRINGS = 9


def build_speed_envelope(tracklets, clock_error_sec, safety_factor, log=print):
    """Measure this run's own speed ceiling and build one envelope per floor group.

    The ceiling comes from the run being reconciled, not from a config value and not
    from a calibration file. Two reasons, both learned here the hard way:

      * NO GUESSED CONSTANTS. Consecutive observations of ONE tracklet are provably
        one person walking, so their speed distribution is this floor's real answer.
        Every numeric constant this project invented has been reverted for hurting
        accuracy -- four thresholds and the topology veto -- and a number read off
        the data cannot repeat that.
      * NO UNITS PROBLEM. The floor frame's scale is unknown (no metric reference),
        but the distances and the ceiling come out of the same frame, so the unknown
        scale cancels exactly. Nothing here claims metres.

    Returns {group: SpeedEnvelope}. A group with too little data is absent, which
    makes the check unavailable for it -- fail-open, by construction.
    """
    per_group = defaultdict(dict)
    for key, info in tracklets.items():
        rows = info.get("floor") or []
        for ts, x, y, _err, group in rows:
            per_group[group].setdefault(key, []).append((ts, x, y))

    envelopes = {}
    for group, samples in sorted(per_group.items()):
        stats = observed_speed_ceiling(samples)
        if stats is None:
            log(f"  geometry: group {group} has too few usable observation pairs to "
                f"measure a speed ceiling -> reachability check UNAVAILABLE here "
                f"(fails open, nothing is vetoed).")
            continue
        env = SpeedEnvelope(stats["ceiling_units_per_sec"],
                            clock_error_sec=clock_error_sec,
                            safety_factor=safety_factor, group=group)
        envelopes[group] = env
        log(f"  geometry: group {group} -- {env.describe()} "
            f"[measured over {stats['n_pairs']} pairs from {stats['n_tracks']} "
            f"tracklet(s); median {stats['median']:.3f}, p99 {stats['p99']:.3f}]")
    return envelopes


def closest_pairings(rows_a, rows_b, limit=MAX_GEOMETRY_PAIRINGS):
    """The `limit` temporally closest (obs_a, obs_b) pairings between two tracklets.

    `rows_a` / `rows_b` are the `floor` lists -- (ts, x, y, error, group), sorted by
    ts. For each observation on each side we take its nearest-in-time counterpart on
    the other, then keep the pairings with the smallest |dt|.

    That single rule handles both questions with no special-casing:

      overlapping in time -> the closest pairings are near-simultaneous, so the
                             verdict is about standing in two places at once;
      disjoint in time    -> the closest pairings cluster at the handover boundary,
                             so the verdict is about whether the walk was possible.
    """
    if not rows_a or not rows_b:
        return []
    times_b = [r[0] for r in rows_b]
    times_a = [r[0] for r in rows_a]
    seen = set()
    out = []

    def nearest(rows, times, ts):
        i = bisect.bisect_left(times, ts)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(rows):
                d = abs(rows[j][0] - ts)
                if best is None or d < best[0]:
                    best = (d, j)
        return best

    for ia, ra in enumerate(rows_a):
        got = nearest(rows_b, times_b, ra[0])
        if got is not None and (ia, got[1]) not in seen:
            seen.add((ia, got[1]))
            out.append((got[0], ra, rows_b[got[1]]))
    for ib, rb in enumerate(rows_b):
        got = nearest(rows_a, times_a, rb[0])
        if got is not None and (got[1], ib) not in seen:
            seen.add((got[1], ib))
            out.append((got[0], rows_a[got[1]], rb))

    out.sort(key=lambda p: p[0])
    return out[:limit]


def reachability_verdict(rows_a, rows_b, envelopes):
    """-> (is_impossible, detail dict). The geometric veto for one tracklet pair.

    THE VERDICT IS THE MEDIAN, NOT THE WORST CASE, and that choice is the whole
    difference between a usable guard and the topology veto that had to be disabled.
    One bad foot point -- a clipped box, a person behind a desk, a momentary
    detector wobble -- produces one impossible pairing. Vetoing on the worst case
    would let that single artefact split a person permanently, and reconcile has no
    way to undo a refusal. A median demands that the pair be impossible
    CONSISTENTLY, across independent instants.

    Fails open on every uncertainty: no positions, mixed floor groups, an unmeasured
    group, or fewer than MIN_GEOMETRY_PAIRINGS pairings all return "not impossible".
    """
    if not rows_a or not rows_b or not envelopes:
        return False, None

    groups = {r[4] for r in rows_a} | {r[4] for r in rows_b}
    if len(groups) != 1:
        # Either a camera pair spanning two floor frames (no comparable
        # coordinates), or a tracklet whose calibration changed mid-run. Neither is
        # a distance.
        return False, None
    group = next(iter(groups))
    env = envelopes.get(group)
    if env is None:
        return False, None

    pairings = closest_pairings(rows_a, rows_b)
    if len(pairings) < MIN_GEOMETRY_PAIRINGS:
        return False, None

    speeds, verdicts = [], []
    for _dt, ra, rb in pairings:
        a = RecordedPosition(ra[1], ra[2], ra[3], group, ra[0])
        b = RecordedPosition(rb[1], rb[2], rb[3], group, rb[0])
        reason = env.check(a, b)
        if reason.required_speed is None:
            continue
        speeds.append(reason.required_speed)
        verdicts.append(reason)

    if len(speeds) < MIN_GEOMETRY_PAIRINGS:
        return False, None

    speeds.sort()
    median = speeds[len(speeds) // 2] if len(speeds) % 2 else (
        0.5 * (speeds[len(speeds) // 2 - 1] + speeds[len(speeds) // 2]))
    impossible = median > env.limit
    detail = {
        "group": group,
        "pairings": len(speeds),
        "median_required_speed": round(float(median), 4),
        "min_required_speed": round(float(speeds[0]), 4),
        "max_required_speed": round(float(speeds[-1]), 4),
        "limit": round(float(env.limit), 4),
        "n_impossible": sum(1 for r in verdicts if r.verdict == IMPOSSIBLE),
        "verdict": IMPOSSIBLE if impossible else "plausible",
    }
    return impossible, detail


def reconcile_tracklets(store, threshold, run_id=None,
                        same_camera_threshold=0.90,
                        same_camera_thresholds=None,
                        require_reciprocal_best=True,
                        min_tracklet_observations=3, log=print,
                        decision_log=None, top2_margin_threshold=None,
                        top2_margin_basis="eligible",
                        scoring=PROTOTYPE, consensus_top_frac=0.25,
                        max_observations_per_side=64,
                        covisibility=None,
                        same_camera_reciprocal_best=False,
                        same_camera_rounds=False,
                        geometry=None,
                        quality_out=None):
    """
    Rebuild global ids from camera-local tracklets.

    It does not trust the existing global_id buckets, because a bad live
    assignment can already contain several people. The tracklet -- one camera's
    view of one track -- is the unit of evidence.

    same_camera_threshold : the GLOBAL same-camera bar, used for any camera with
        no override.
    same_camera_thresholds : optional {camera: bar} overrides, normally built by
        resolve_same_camera_thresholds() from `identity.reconcile.per_camera`.
        None/{} reproduces the old single-global behaviour exactly.

        Why per-camera: the per-subject same/different boundaries OVERLAP across
        cameras, so no global value works -- p95 of the "top different" score
        (0.816) exceeds p5 of the "worst same" score (0.719) on real field data.
        At 0.90, cam_213 got zero eligible same-camera partners across 11 subjects
        and cam_224 managed it for 5 of 17, while cam_206 saw 9.0 per subject.
        See REMEDIATION_PLAN.md J.6.

    min_tracklet_observations : a real identity needs sustained observation. A
        tracklet with fewer stored observations than this is treated as detector
        noise / an unusable fragment: it is marked UNIDENTIFIED (its global_id is
        cleared) rather than becoming its own person, so it does not inflate the
        head-count. Its vectors stay in the gallery. 1 = keep everything.

    decision_log : optional identity.decision_log.DecisionLog. When supplied,
        EVERY merge decision -- accepted and rejected alike -- is recorded with
        all gates evaluated independently (no short-circuiting) and the full
        annotated candidate vector. Purely additive: the merge decisions below are
        computed exactly as before and the log only observes them, which
        tests/live/test_phase1_decision_log.py asserts by comparing the returned
        remap with and without a log attached.

    top2_margin_threshold : None (the default) means TOP2_MARGIN is COMPUTED AND
        LOGGED BUT ENFORCES NOTHING. Reconcile has never had a runner-up margin
        rule -- reciprocal-best fills that role -- so this ships inert until the
        logged distribution says whether it rejects anything reciprocal-best does
        not. See REMEDIATION_PLAN.md Phase 9.
    top2_margin_basis : "eligible" | "all_scored" -- which margin variant would
        gate if a threshold were set. Exactly one can ever gate, by construction.

    geometry : optional dict of POLICY for the reachability veto -- never geometry
        data. Keys: `enabled` (default False), `clock_error_sec`, `safety_factor`.
        The positions themselves come from the observation payloads the live run
        wrote, and the speed ceiling is measured from those same positions, so
        nothing here needs a calibration file. `enabled: false` reproduces the
        pre-geometry behaviour exactly.
    """
    dl = decision_log
    tracklets = _gather_tracklets(store, run_id)
    all_keys = sorted(tracklets)

    # Stable, deterministic handles so a replay reproduces them exactly. Provisional
    # handles live in their own namespace (`U-`) and never reach a tally or a frame.
    handles = {k: f"U-{i:04d}" for i, k in enumerate(all_keys)}

    # ---- per-camera same-camera bars -------------------------------------
    per_cam_bar = {}
    for cam, raw in (same_camera_thresholds or {}).items():
        try:
            per_cam_bar[cam] = float(raw)
        except (TypeError, ValueError):
            log(f"  tracklet reconcile: ignoring non-numeric same-camera bar "
                f"{raw!r} for camera {cam!r}; using the global "
                f"{same_camera_threshold}")

    def cam_bar(*cameras):
        """The same-camera bar for one camera, or the strictest over several."""
        return strictest_same_camera_bar(cameras, per_cam_bar,
                                        same_camera_threshold)

    # A configured camera that is not in this run is almost always a typo, and a
    # typo here is invisible in the output -- the camera just keeps the global bar
    # and stays fragmented. Same reasoning as decision D1: a config that stops
    # matching the hardware must be VISIBLE, never silent. Stream cameras are
    # auto-named cam_<last-IP-octet>, so replacing a switch/camera renames them.
    cameras_present = {k[0] for k in all_keys}
    absent = sorted(str(c) for c in per_cam_bar if c not in cameras_present)
    if absent:
        log(f"  tracklet reconcile: per_camera same-camera bar configured for "
            f"{absent}, which produced no tracklets in this run -- check the "
            f"name(s) against {sorted(str(c) for c in cameras_present)}")
    if per_cam_bar:
        log("  tracklet reconcile: same-camera bars "
            + ", ".join(f"{c}={cam_bar(c):.2f}"
                        for c in sorted(cameras_present, key=str))
            + f"  (global {same_camera_threshold:.2f}, cross-camera {threshold:.2f})")

    # ---- cross-camera simultaneity veto (#38, Phase 7) --------------------
    covis_enabled, covis_pairs = (covisibility if covisibility is not None
                                  else (False, {}))
    if covis_enabled:
        listed = {c for pair in covis_pairs for c in pair}
        present = sorted(cameras_present, key=str)
        missing = [f"{a}<->{b}" for i, a in enumerate(present)
                   for b in present[i + 1:]
                   if frozenset((a, b)) not in covis_pairs]
        with_ts = sum(1 for k in all_keys if tracklets[k]["span_ts"])
        log(f"  tracklet reconcile: cross-camera simultaneity veto ON over "
            f"{len(covis_pairs)} configured pair(s); {with_ts}/{len(all_keys)} "
            f"tracklet(s) have a wall-clock span")
        if missing:
            # D1: a pair nobody configured is UNCONSTRAINED. That is a deliberate
            # fail-open, but it must be visible -- silence here means a camera pair
            # that can never veto, which looks exactly like the veto not working.
            log(f"  tracklet reconcile: NO covisibility entry for {missing} -- "
                f"those pairs are unconstrained (add them to "
                f"identity.reconcile.covisibility.pairs)")
        unknown = sorted(str(c) for c in listed if c not in cameras_present)
        if unknown:
            log(f"  tracklet reconcile: covisibility names {unknown}, absent from "
                f"this run")

    # ---- geometric reachability veto --------------------------------------
    # Consumes the floor positions the LIVE RUN recorded; computes none of them.
    gcfg = geometry or {}
    geo_envelopes = {}
    positioned = sum(1 for k in all_keys if tracklets[k].get("floor"))
    if gcfg.get("enabled"):
        if positioned == 0:
            # Enabled but no recorded positions: almost always a run captured with
            # geometry off, or one whose calibration did not match the frames. Said
            # loudly because the alternative -- an enabled guard that silently never
            # fires -- is indistinguishable from a working one that finds nothing.
            log("  geometry: reachability veto ENABLED but NOT ONE observation in "
                "this run carries a recorded floor position -> the veto can never "
                "fire. Either the run was captured with geometry.enabled: false, or "
                "the calibration's image_size did not match its frames. Reconcile "
                "continues on appearance alone.")
        else:
            geo_envelopes = build_speed_envelope(
                tracklets,
                clock_error_sec=float(gcfg.get("clock_error_sec", 0.5)),
                safety_factor=float(gcfg.get("safety_factor", 1.5)),
                log=log)
            log(f"  geometry: reachability veto ON -- {positioned}/{len(all_keys)} "
                f"tracklet(s) carry recorded positions, "
                f"{len(geo_envelopes)} floor group(s) measured. Tracklets without a "
                f"position, and pairs across floor groups, are never vetoed.")
    elif positioned:
        log(f"  geometry: {positioned}/{len(all_keys)} tracklet(s) carry recorded "
            f"positions, but the reachability veto is OFF "
            f"(geometry.reconcile.enabled is false) -- nothing is vetoed.")

    geo_details = {}          # (key_a, key_b) -> detail dict, for the log/summary
    geo_unavailable = set()   # pairs geometry could NOT judge -- cached, see below
    geo_vetoes = 0

    def geometry_blocks(a, b):
        """True when merging these two tracklets would be physically impossible."""
        nonlocal geo_vetoes
        if not geo_envelopes:
            return False
        pair = (a, b) if a <= b else (b, a)
        if pair in geo_details:
            return geo_details[pair].get("verdict") == IMPOSSIBLE
        # FAIL-OPEN VERDICTS ARE CACHED TOO. reachability_verdict returns
        # detail=None for every fail-open path -- no recorded positions, mixed floor
        # groups, an unmeasured group, too few pairings -- and those are the COMMON
        # cases today. Caching only judged pairs meant recomputing each of those on
        # every call. Separate set so geo_details stays "pairs geometry judged",
        # which is what the run summary reports.
        if pair in geo_unavailable:
            return False
        impossible, detail = reachability_verdict(
            tracklets[a].get("floor"), tracklets[b].get("floor"), geo_envelopes)
        if detail is None:
            geo_unavailable.add(pair)
        else:
            geo_details[pair] = detail
        if impossible:
            geo_vetoes += 1
        return impossible

    def _obs(k):
        return len(tracklets[k]["vectors"])

    def _record_outcome(k, state, assigned_id=None, merged_from=()):
        if dl is None:
            return
        dl.set_outcome(k, state=state, assigned_id=assigned_id,
                       handle=handles[k], merged_from=merged_from,
                       observations=_obs(k), cameras=[k[0]],
                       frame_range=list(tracklets[k]["span"]))

    # Suppress spurious tracklets before clustering. One- or two-frame tracks are
    # almost always a missed/duplicate detection, too short to embed reliably;
    # letting each become an identity is what inflates the person count.
    suppressed = [k for k in all_keys
                  if len(tracklets[k]["vectors"]) < min_tracklet_observations]
    for k in suppressed:
        store.clear_global_id(tracklets[k]["points"])
        log(f"  tracklet reconcile: suppressed {k} "
            f"({len(tracklets[k]['vectors'])} obs < {min_tracklet_observations})")
        _record_outcome(k, dlog.SUPPRESSED)

    keys = [k for k in all_keys if k not in set(suppressed)]
    # #25 FIXED. This used to `return {}` whenever fewer than two tracklets
    # survived, which stamped NO identity on the survivor -- so a sparse camera
    # rendered its whole video as a bare "ID <track_id>" with no reid at all.
    # There is nothing to MERGE with one tracklet, but there is still an identity
    # to ASSIGN, and the code below already handles a single-member cluster
    # correctly: no pair clears any bar, one root survives, and it gets a gid.
    if not keys:
        log("  tracklet reconcile: no tracklets survived suppression -> "
            "nothing to reconcile")
        return {}
    if len(keys) == 1:
        log(f"  tracklet reconcile: only 1 tracklet survived -> assigning it an "
            f"identity (nothing to merge with)")

    protos = {k: _prototype(tracklets[k]["vectors"]) for k in keys}
    # A zero-norm prototype cannot be scored, so the tracklet cannot be clustered.
    # It used to vanish from `keys` SILENTLY -- never entering `remap`, its points
    # never cleared, keeping whatever provisional gid the live run gave it, with
    # nothing in the output saying so.
    for k in [k for k in keys if protos[k] is None]:
        store.clear_global_id(tracklets[k]["points"])
        log(f"  tracklet reconcile: DEGENERATE prototype for {k} "
            f"({len(tracklets[k]['vectors'])} obs, unnormalizable) -> id cleared; "
            f"it cannot be clustered")
        _record_outcome(k, dlog.SUPPRESSED)
    keys = [k for k in keys if protos[k] is not None]

    # ---- pair scoring (plan #45a) -----------------------------------------
    if scoring not in SCORING_MODES:
        log(f"  tracklet reconcile: unknown scoring {scoring!r}; falling back to "
            f"{PROTOTYPE!r} (expected one of {list(SCORING_MODES)})")
        scoring = PROTOTYPE
    if scoring != PROTOTYPE:
        log(f"  tracklet reconcile: scoring={scoring} "
            f"(top_frac={consensus_top_frac}, cap={max_observations_per_side}) "
            f"-- NOTE thresholds are mode-specific; re-derive them with the sweep")
    rows = {k: _unit_rows(tracklets[k]["vectors"]) for k in keys}
    _rows_cache = {}

    def cluster_rows(members):
        """Observation rows for a whole cluster, capped and cached per round."""
        key = frozenset(members)
        cached = _rows_cache.get(key)
        if cached is None:
            mats = [rows[m] for m in sorted(members)]
            cached = _subsample_rows(
                mats[0] if len(mats) == 1 else np.concatenate(mats, axis=0),
                max_observations_per_side)
            _rows_cache[key] = cached
        return cached

    def pair_score(members_a, members_b, proto_a, proto_b):
        if scoring == PROTOTYPE:            # fast path: identical to before
            return float(proto_a @ proto_b)
        return score_observation_sets(
            cluster_rows(members_a), cluster_rows(members_b), proto_a, proto_b,
            mode=scoring, top_frac=consensus_top_frac,
            cap=max_observations_per_side)

    parent = {k: k for k in keys}
    members = {k: {k} for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def conflict_reason(set_a, set_b):
        """None when the two clusters may merge, else the blocking gate name.

        Returns a REASON rather than a bool so the decision log can attribute the
        exclusion.

        THREE physical impossibilities are checked, and all are HARD -- no appearance
        score can override them, because they are statements about where one body
        can be, not about how alike two crops look:

        SAME CAMERA: two tracklets overlapping in time in ONE view are provably two
        people. One body cannot be two simultaneous detections.

        CROSS CAMERA (#38, Phase 7): two tracklets overlapping in time in cameras
        that CANNOT both see one person are equally provably two people. Until now
        this check did not exist -- `conflict()` skipped every cross-camera pair
        outright -- so a person in cam_213 and a different person in cam_219 at the
        same moment merged on appearance alone. That is the "one number on two
        people" the operator kept reporting, and no threshold addresses it: the
        pair's cosine can be perfectly high and the merge still be impossible.

        GEOMETRIC REACHABILITY: merging two tracklets whose recorded FLOOR POSITIONS
        would require a person to move faster than anyone on this floor has been
        observed to move is equally impossible. Unlike the two above, this one is a
        statement about DISTANCE rather than mere co-presence, so it also catches the
        cases those miss: two people standing in the same camera pair at slightly
        different times, and two people in one camera whose tracks do not overlap.
        Positions come from the live run's payload; nothing is recomputed here.

        Note this is where the disabled `live.topology` min-transit veto's job now
        lives, done properly. That veto guessed 2-3 second minimums between cameras
        that are actually adjacent, and pruned the true cross-camera match. Measured
        positions cannot make that mistake: overlapping cameras yield a distance near
        zero, so the check stays silent exactly where the guess over-fired.

        Fail-open everywhere it is unsure: disabled by config, an unlisted camera
        pair, a co-visible pair, a missing wall-clock span, an observation with no
        recorded position, a pair spanning two floor groups, or a floor group whose
        speed ceiling could not be measured all mean "no veto". A wrong veto is worse
        than a missing one -- it manufactures a second identity for someone who only
        has one, and nothing downstream can undo that.
        """
        for a in set_a:
            for b in set_b:
                # CERTAIN AND CHEAP RULE FIRST. Two comparisons, and the claim is
                # provable. Geometry costs a binary search plus up to
                # MAX_GEOMETRY_PAIRINGS envelope evaluations and, as of 2026-08-04,
                # has never run on real data. Evaluating it first attributed every
                # pair that violated BOTH rules to geometry, pointing the next
                # investigation at the least-validated gate.
                if a[0] == b[0] and not _spans_disjoint(tracklets[a]["span"],
                                                        tracklets[b]["span"]):
                    return dlog.TEMPORAL_CONFLICT_SAME_CAMERA
                if geometry_blocks(a, b):
                    return dlog.GEOMETRIC_UNREACHABLE
                if a[0] == b[0]:
                    continue
                if not covis_enabled:
                    continue
                pair = frozenset((a[0], b[0]))
                if pair not in covis_pairs:
                    continue                      # unlisted -> unconstrained (D1)
                tol = covis_pairs[pair]
                if tol is None:
                    continue                      # co-visible -> never vetoed
                overlap = temporal_overlap_sec(tracklets[a]["span_ts"],
                                               tracklets[b]["span_ts"])
                if overlap is None:
                    continue                      # no wall clock -> cannot judge
                if overlap > tol:
                    return dlog.TEMPORAL_CONFLICT_CROSS_CAMERA
        return None

    def conflict(set_a, set_b):
        return conflict_reason(set_a, set_b) is not None

    # ---- decision-log helpers (no effect on any decision) ----------------
    def _na_gate():
        """A gate that does not apply to this phase. Recorded anyway so every
        record carries every gate and analysis never has to special-case."""
        return GateResult(value=None, threshold=None, passed=True,
                          extra={"applies": False})

    def _cluster_meta(members_set):
        cams = sorted({m[0] for m in members_set})
        obs = sum(_obs(m) for m in members_set)
        return cams, obs

    def _emit(subject_handle, phase, round_index, subject_members, cands,
              gates, accepted_partner=None, context="same_camera"):
        """Build + register one DecisionRecord. Returns None when logging is off."""
        if dl is None:
            return None
        cams, obs = _cluster_meta(subject_members)
        spans = [tracklets[m]["span"] for m in subject_members]
        rec = DecisionRecord(
            handle=subject_handle,
            state=(dlog.RESOLVED if accepted_partner else dlog.EXPIRED_UNRESOLVED),
            phase=phase, round_index=round_index,
            accepted_partner=accepted_partner,
            observations=obs, cameras=cams,
            frame_range=[min(s[0] for s in spans), max(s[1] for s in spans)],
            context=context,
            gates=gates, candidates=cands,
            scored_count=len(cands),
            eligible_count=sum(1 for c in cands if c.eligible),
        )
        return dl.add(rec)

    # A SUPPRESSED tracklet is a decision too, and the only place MIN_OBSERVATIONS
    # can actually fail: every tracklet that survives suppression passes that gate
    # by construction, since suppression uses the same threshold. Recording them
    # here is what makes the gate-failure histogram answer "how much am I losing to
    # min_tracklet_observations", which was previously invisible.
    if dl is not None:
        for k in suppressed:
            _emit(handles[k], "same_camera", 0, {k}, [], {
                dlog.MIN_OBSERVATIONS: GateResult(
                    value=_obs(k), threshold=min_tracklet_observations,
                    passed=False),
                dlog.ABSOLUTE_THRESHOLD: GateResult(
                    value=None, threshold=cam_bar(k[0]), passed=False,
                    extra={"note": "suppressed before any candidate was scored"}),
                dlog.TOP2_MARGIN: dlog.compute_top2_margin(
                    [], cam_bar(k[0]), basis=top2_margin_basis,
                    threshold=top2_margin_threshold)[0],
                dlog.RECIPROCAL_BEST: _na_gate(),
                dlog.TEMPORAL_CONFLICT_SAME_CAMERA: _na_gate(),
                dlog.TEMPORAL_CONFLICT_CROSS_CAMERA: _na_gate(),
                dlog.GEOMETRIC_UNREACHABLE: _na_gate(),
            }, accepted_partner=None, context="suppressed")
            dl.records[-1].state = dlog.SUPPRESSED

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        ma, mb = members[ra], members[rb]
        if conflict(ma, mb):
            return False
        winner, loser = (ra, rb) if ra < rb else (rb, ra)
        parent[loser] = winner
        members[winner] = ma | mb
        members.pop(loser, None)
        return True

    def current_roots():
        roots = defaultdict(set)
        for k in keys:
            roots[find(k)].add(k)
        return roots

    # Phase 1: repair same-camera track fragmentation with a higher threshold.
    # A pair is a merge candidate iff it shares a camera, its spans are disjoint,
    # and it clears THAT CAMERA's bar (both tracklets are in the same camera here,
    # so the bar is unambiguous -- a[0] == b[0]).
    #
    # RECIPROCAL BEST in Phase 1. Phase 2 has always required mutual
    # nearest-neighbour; Phase 1 never did, so it merged EVERY above-bar pair in
    # score order. That is how one weak edge drags a stranger into a cluster: in
    # cam_224, tracklet 58's best partner was 16 at 0.937, yet 58+112 (0.823) and
    # 58+120 (0.872) merged anyway and fused two people into one reid. Mutual-best
    # refuses both WITHOUT moving any threshold, which is why it is worth more than
    # another bar sweep -- it removes the trade instead of repositioning it.
    #
    # ---- WHY THIS NOW RUNS IN ROUNDS (Part M.9.11) -------------------------
    # The comment here used to say: "a person fragmented into three or more pieces
    # has only ONE mutual-best pair per round, so the rest wait for Phase 2's rounds
    # to consolidate them -- chains still close, just later." THAT WAS FALSE, and it
    # cost a real identity. Phase 2 cannot close a same-camera chain at all:
    # `mergeable_cross` requires at least one CROSS-camera member pair, so a pair of
    # clusters that both sit in one camera is excluded outright
    # (dlog.NOT_MERGEABLE_CROSS). The promised later consolidation never happens.
    #
    # Measured on run 20260731_060425, cam_206, one person in three fragments:
    #     206/26 <-> 206/43  = 0.916   merged (each other's best)
    #     206/12 <-> 206/43  = 0.906   ABOVE cam_206's 0.90 bar, REFUSED
    # 12's best partner is 43, but 43's best is 26, so mutual-best rejected an edge
    # that was above the bar by 0.006. 12 was left orphaned, then captured
    # cross-camera at 0.771 into a DIFFERENT person's identity -- which the operator
    # saw as "it misidentified someone in cam_206 as reid 1, then a few seconds
    # later they get reid 6". Lowering cam_206's bar cannot fix it (the edge already
    # clears the bar) and measurably makes cam_206 worse (25 -> 20 identities).
    #
    # So Phase 1 now iterates like Phase 2: re-score from CURRENT clusters each
    # round, re-derive mutual-best, merge, repeat until a round merges nothing.
    # Round 1 is identical to the old single pass by construction -- every cluster
    # is a single tracklet then -- so `same_camera_rounds=False` reproduces the old
    # behaviour exactly, and the flag defaults off.
    #
    # ---- WHY EVERY MEMBER PAIR MUST CLEAR THE BAR, not just the cluster mean ----
    # Iterating alone WOULD have relaxed the guard, and the regression test caught it:
    # once a round has only two clusters left, mutual-best is VACUOUS (the single
    # remaining pair is trivially each other's best), so a borderline stranger walks
    # in on a cluster-mean score. In the fixture, a stranger at 0.910 against one
    # fragment and 0.890 against the other scored 0.905 against their mean and merged
    # -- precisely the weak-edge capture mutual-best was added to prevent.
    #
    # So a same-camera merge additionally requires EVERY cross-member pair to clear
    # the bar. Merging asserts "all of these fragments are one person", so every
    # fragment has to agree; a mean lets a majority outvote a member that disagrees.
    # Same insight as M.3: judge the claim on the fragments, not on a mean of them.
    #
    #   chain    C vs A 0.906, C vs B 0.974  -> both clear 0.90, merges
    #   stranger S vs A 0.910, S vs B 0.890  -> B disagrees, refused
    #
    # For singleton clusters this is exactly the old single-pair test, so round 1 --
    # and therefore `same_camera_rounds=False` -- is unchanged by it.
    def all_member_pairs_clear(ma, mb, bar):
        for x in ma:
            for y in mb:
                if pair_score({x}, {y}, protos[x], protos[y]) < bar:
                    return False
        return True

    accepted_same = set()
    rejected_not_mutual = 0
    same_round = 0
    while True:
        # Phase 1 runs before Phase 2, so every cluster is still single-camera.
        by_camera = defaultdict(list)
        for root, ms in current_roots().items():
            cams = {m[0] for m in ms}
            if len(cams) == 1:
                by_camera[next(iter(cams))].append(root)

        same_pairs = []
        for cam, roots_here in by_camera.items():
            roots_here.sort()
            for i, a in enumerate(roots_here):
                for b in roots_here[i + 1:]:
                    ma, mb = members[a], members[b]
                    # conflict() subsumes the old _spans_disjoint check and stays
                    # correct once a side holds more than one tracklet.
                    if conflict(ma, mb):
                        continue
                    proto_a = _cluster_prototype(ma, protos)
                    proto_b = _cluster_prototype(mb, protos)
                    if proto_a is None or proto_b is None:
                        continue
                    s = pair_score(ma, mb, proto_a, proto_b)
                    if s < cam_bar(cam):
                        continue
                    if not all_member_pairs_clear(ma, mb, cam_bar(cam)):
                        continue
                    same_pairs.append((s, a, b))

        same_best = {}
        if same_camera_reciprocal_best:
            for s, a, b in same_pairs:
                if s > same_best.get(a, (-1.0, None))[0]:
                    same_best[a] = (s, b)
                if s > same_best.get(b, (-1.0, None))[0]:
                    same_best[b] = (s, a)

        def same_pair_mutual(a, b, _best=same_best):
            if not same_camera_reciprocal_best:
                return True
            return (_best.get(a, (None, None))[1] == b
                    and _best.get(b, (None, None))[1] == a)

        merged_here = 0
        for s, a, b in sorted(same_pairs, reverse=True):
            if not same_pair_mutual(a, b):
                rejected_not_mutual += 1
                continue
            if union(a, b):
                log(f"  tracklet reconcile: same-camera merge {a} + {b} "
                    f"(cosine {s:.3f})"
                    + (f" [round {same_round + 1}]" if same_round else ""))
                accepted_same.add((a, b))
                merged_here += 1
        same_round += 1
        if not merged_here or not same_camera_rounds:
            break

    if same_camera_reciprocal_best:
        log(f"  tracklet reconcile: same-camera reciprocal-best ON -- "
            f"{len(accepted_same)} merged over {same_round} round(s), "
            f"{rejected_not_mutual} above-bar pair(s) refused for not being each "
            f"other's best")

    # Phase 1 cluster membership, captured BEFORE Phase 2 touches it. The
    # instrumentation below iterates TRACKLET keys, while `accepted_same` holds
    # ROOT pairs as they were at merge time -- identical on round 1, divergent once
    # same_camera_rounds is on. Log-only.
    same_camera_root = {k: find(k) for k in keys}

    # ---- Phase 1 instrumentation, subject-centric so the margin is meaningful.
    # Runs AFTER the merges above and reads only `protos` / `tracklets`, so it
    # cannot influence anything. Every same-camera peer is scored (including
    # time-overlapping ones, which the decision loop skips before scoring) so the
    # candidate vector can carry an exclusion reason for each.
    if dl is not None:
        for a in keys:
            peers = [b for b in keys if b != a and b[0] == a[0]]
            if not peers:
                continue
            bar_a = cam_bar(a[0])          # peers are same-camera, so one bar
            scored = []
            for b in peers:
                score = pair_score({a}, {b}, protos[a], protos[b])
                # Use the reason conflict_reason ACTUALLY gave. It used to be
                # hardcoded to the same-camera overlap gate, which was true while
                # that was the only rule reachable here; a geometric veto attributed
                # to time overlap would send the next investigation to the wrong
                # rule, which is exactly the failure Part M was written about.
                blocked = conflict_reason({a}, {b})
                if blocked is not None:
                    reason = blocked
                elif score < bar_a:
                    reason = dlog.BELOW_ABSOLUTE_THRESHOLD
                else:
                    reason = None
                scored.append((b, score, reason))
            best_b = max((t for t in scored if t[2] is None),
                         key=lambda t: t[1], default=None)
            cands = []
            for b, score, reason in scored:
                cands.append(Candidate(
                    handle=handles[b], score=round(score, 6), excluded_by=reason,
                    would_fail_reciprocity=None,      # Phase 1 has no reciprocity rule
                    cameras=[b[0]], cluster_size=1, observations=_obs(b),
                    pair_similarity_to_best=(None if best_b is None else
                                             round(pair_score({b}, {best_b[0]},
                                                              protos[b],
                                                              protos[best_b[0]]), 6)),
                ))
            margin_gate, best, _, _ = dlog.compute_top2_margin(
                cands, bar_a, basis=top2_margin_basis,
                threshold=top2_margin_threshold)
            partner = None
            if best is not None:
                bkey = next(b for b in peers if handles[b] == best.handle)
                partner = (best.handle
                           if same_camera_root[a] == same_camera_root[bkey]
                           else None)
            _emit(handles[a], "same_camera", 0, {a}, cands, {
                dlog.MIN_OBSERVATIONS: GateResult(
                    value=_obs(a), threshold=min_tracklet_observations,
                    passed=_obs(a) >= min_tracklet_observations),
                dlog.ABSOLUTE_THRESHOLD: GateResult(
                    value=(None if best is None else best.score),
                    threshold=bar_a,
                    passed=best is not None),
                dlog.TOP2_MARGIN: margin_gate,
                dlog.RECIPROCAL_BEST: _na_gate(),
                dlog.TEMPORAL_CONFLICT_SAME_CAMERA: GateResult(
                    value=sum(1 for _, _, r in scored
                              if r == dlog.TEMPORAL_CONFLICT_SAME_CAMERA),
                    threshold=0, passed=True,
                    extra={"note": "count of peers excluded for time overlap"}),
                dlog.TEMPORAL_CONFLICT_CROSS_CAMERA: _na_gate(),
                dlog.GEOMETRIC_UNREACHABLE: GateResult(
                    value=sum(1 for _, _, r in scored
                              if r == dlog.GEOMETRIC_UNREACHABLE),
                    threshold=0, passed=True,
                    extra={"note": "count of peers excluded as physically "
                                   "unreachable from the recorded floor positions"}),
            }, accepted_partner=partner, context="same_camera")

    # Phase 2 runs in ROUNDS, re-scoring from scratch each round, so a chain of
    # 3+ mutually-similar fragments (e.g. A's best match is B, but B's own best
    # match is C, not A) can fully consolidate instead of stalling after one
    # pass. Each round still requires reciprocal-best (when enabled) using the
    # CURRENT cluster prototypes -- merging never gets easier than the safety
    # rule allows, it just gets to re-check after each merge updates the
    # clusters. A round that merges nothing ends the loop; every merge strictly
    # reduces the number of roots by one, so this always terminates.
    round_index = 0
    while True:
        roots = current_roots()
        root_protos = {r: _cluster_prototype(ms, protos) for r, ms in roots.items()}
        root_keys = sorted(r for r, p in root_protos.items() if p is not None)

        def mergeable_cross(a, b):
            if conflict(roots[a], roots[b]):
                return False
            return any(x[0] != y[0] for x in roots[a] for y in roots[b])

        def live_members(x):
            """The cluster x belongs to RIGHT NOW (post any merge already made this
            round), as opposed to `roots[x]`, the snapshot taken when the round
            started."""
            return members.get(find(x), roots.get(x, {x}))

        def shared_cameras(a, b):
            """Cameras both clusters contain in the ROUND SNAPSHOT.

            Deliberately snapshot-based. This feeds scoring, reciprocal-best and the
            decision log, all of which describe the round as it was SCORED -- mixing
            a post-merge camera set into a pre-merge score would log a bar that never
            applied to that number. The live view is `live_shared_cameras` and is
            used only where a merge is actually about to happen.
            """
            return {k[0] for k in roots[a]} & {k[0] for k in roots[b]}

        def live_shared_cameras(a, b):
            return ({k[0] for k in live_members(a)}
                    & {k[0] for k in live_members(b)})

        def pair_threshold(a, b):
            """The bar THIS cluster pair must clear.

            Phase 1 deliberately repairs same-camera fragmentation at the strict
            same-camera bar, because "these two tracks in ONE camera are the same
            person" is a much easier claim to get wrong than a genuine
            cross-camera link (same lighting, same pose distribution, so unrelated
            people score higher). Phase 2 used to apply the LOW cross-camera
            `threshold` to every mergeable pair, which quietly threw that bar away:
            as soon as a cluster had absorbed a second camera, `mergeable_cross`
            was satisfied by that pre-existing member, and a same-camera fragment
            could then join at the cross-camera bar. Two strangers seen in one
            camera at different times merged at ~0.69.

            So: if the two clusters SHARE a camera, merging them asserts that that
            camera's fragments are one person -- exactly Phase 1's claim -- and it
            must clear the same-camera bar. Only genuinely camera-disjoint clusters
            get the lower cross-camera bar.

            With PER-CAMERA bars a cluster pair can share several cameras, each with
            its own bar; strictest_same_camera_bar() resolves that (and says why).

            NOTE the pre-existing defect this inherits unchanged (plan item #27):
            `roots` is the snapshot taken at the start of the round, and `union`
            updates `members`, not `roots`, so a pair merged earlier in THIS round
            is judged on its stale camera set. Per-camera bars do not widen that
            hole -- the worst case is still "a bar lower than the one that should
            apply" -- but fixing #27 becomes more valuable now that the applicable
            bar varies by camera.
            """
            shared = shared_cameras(a, b)
            if not shared:
                return threshold
            return cam_bar(*shared)

        def pair_bar_now(a, b):
            """The bar for this pair given the CURRENT clusters -- the #27 fix.

            `roots` is the snapshot taken at the start of the round, and `union`
            updates `members`, not `roots`. So a cluster that absorbed a second
            camera earlier in THIS round was still judged on its stale camera set,
            and a pair that had come to share a camera was offered the LOW
            cross-camera bar instead of the same-camera one: two strangers seen in
            one camera at different times could merge at 0.63.

            Identical to pair_threshold until the round's first merge, which is why
            scoring and the decision log can keep using the snapshot.
            """
            shared = live_shared_cameras(a, b)
            if not shared:
                return threshold
            return cam_bar(*shared)

        root_scores = {}
        for i, a in enumerate(root_keys):
            for b in root_keys[i + 1:]:
                if not mergeable_cross(a, b):
                    continue
                root_scores[(a, b)] = pair_score(roots[a], roots[b],
                                                root_protos[a], root_protos[b])

        def root_score(a, b):
            return root_scores[(a, b)] if a < b else root_scores[(b, a)]

        # Reciprocal-best uses the SAME per-pair bar, so a cluster never picks a
        # partner it would then be refused, which would block a legitimate merge.
        best_partner = {}
        if require_reciprocal_best:
            for a in root_keys:
                candidates = [
                    (root_score(a, b), b) for b in root_keys
                    if b != a
                    and ((a, b) in root_scores or (b, a) in root_scores)
                    and root_score(a, b) >= pair_threshold(a, b)
                ]
                if candidates:
                    best_partner[a] = max(candidates)[1]

        cross_pairs = []
        for (a, b), s in root_scores.items():
            if s < pair_threshold(a, b):
                continue
            if require_reciprocal_best and not (
                    best_partner.get(a) == b and best_partner.get(b) == a):
                continue
            cross_pairs.append((s, a, b))

        merged_this_round = False
        accepted_pairs = set()
        for s, a, b in sorted(cross_pairs, reverse=True):
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            # Re-read the bar from CURRENT membership (#27): an earlier merge in
            # this same round may have given one side a camera it did not have when
            # the round's scores were taken, which RAISES the bar this pair must
            # clear. Re-checking here is what stops a same-camera claim sneaking
            # through at the cross-camera threshold.
            bar = pair_bar_now(a, b)
            if s < bar:
                log(f"  tracklet reconcile: SKIP {a} + {b} (cosine {s:.3f} < "
                    f"{bar:.2f}) -- an earlier merge this round made this a "
                    f"same-camera claim (#27)")
                continue
            # Name the lane by the CLAIM being made, not by comparing the bar to a
            # number: with per-camera bars one camera's same-camera bar can equal
            # the cross-camera threshold, and then a bar comparison mislabels the
            # lane. (The old message called every Phase 2 merge "cross-camera" even
            # when both printed root keys were the same camera, which made the log
            # actively misleading -- this keeps that fix while staying correct
            # under per-camera bars.)
            lane = "same-camera" if live_shared_cameras(a, b) else "cross-camera"
            if union(ra, rb):
                log(f"  tracklet reconcile: {lane} cluster merge {a} + {b} "
                    f"(cosine {s:.3f} >= {bar:.2f})")
                merged_this_round = True
                accepted_pairs.add((a, b))
                accepted_pairs.add((b, a))

        # ---- Phase 2 instrumentation. Runs after the round's merges and reads
        # only this round's scores, so it cannot alter the outcome. Unlike the
        # decision path it scores EVERY root pair, so a candidate excluded by a
        # hard constraint carries a reason instead of silently vanishing.
        if dl is not None:
            cluster_handle = {r: f"C-{i:04d}" for i, r in enumerate(root_keys)}
            all_scores = {}
            for i, a in enumerate(root_keys):
                for b in root_keys[i + 1:]:
                    all_scores[(a, b)] = pair_score(roots[a], roots[b],
                                                    root_protos[a], root_protos[b])

            def _score(a, b):
                return all_scores[(a, b)] if (a, b) in all_scores else all_scores[(b, a)]

            for a in root_keys:
                peers = [b for b in root_keys if b != a]
                if not peers:
                    continue
                scored = []
                for b in peers:
                    sc, bar_b = _score(a, b), pair_threshold(a, b)
                    reason = conflict_reason(roots[a], roots[b])
                    if reason is None and not any(x[0] != y[0]
                                                  for x in roots[a] for y in roots[b]):
                        reason = dlog.NOT_MERGEABLE_CROSS
                    if reason is None and sc < bar_b:
                        reason = dlog.BELOW_ABSOLUTE_THRESHOLD
                    scored.append((b, sc, bar_b, reason))
                best_t = max((t for t in scored if t[3] is None),
                             key=lambda t: t[1], default=None)
                cands = []
                for b, sc, bar_b, reason in scored:
                    cams_b, obs_b = _cluster_meta(roots[b])
                    cands.append(Candidate(
                        handle=cluster_handle[b], score=round(sc, 6),
                        excluded_by=reason,
                        # ANNOTATION ONLY -- reciprocity never excludes a candidate
                        # from the eligible set (see decision_log's module docstring).
                        would_fail_reciprocity=(None if not require_reciprocal_best
                                                else best_partner.get(b) != a),
                        cameras=cams_b, cluster_size=len(roots[b]), observations=obs_b,
                        pair_similarity_to_best=(None if best_t is None else
                                                 round(_score(b, best_t[0]), 6)
                                                 if b != best_t[0] else 1.0),
                    ))
                # The floor differs per candidate here, so pass the SUBJECT's best
                # applicable bar for the recorded threshold.
                subj_bar = best_t[2] if best_t else threshold
                margin_gate, best, _, _ = dlog.compute_top2_margin(
                    cands, subj_bar, basis=top2_margin_basis,
                    threshold=top2_margin_threshold)
                partner = None
                recip_passed = True
                if best is not None:
                    bkey = next(b for b in peers if cluster_handle[b] == best.handle)
                    partner = best.handle if (a, bkey) in accepted_pairs else None
                    if require_reciprocal_best:
                        recip_passed = (best_partner.get(a) == bkey
                                        and best_partner.get(bkey) == a)
                cams_a, obs_a = _cluster_meta(roots[a])
                # Which lane this subject's best candidate sits in -- derived from
                # the shared-camera claim, not from a bar comparison, for the same
                # reason as `lane` above.
                ctx = ("same_camera"
                       if (best_t is not None and shared_cameras(a, best_t[0]))
                       else "cross_camera")
                _emit(cluster_handle[a], "cross_camera", round_index, roots[a],
                      cands, {
                    dlog.MIN_OBSERVATIONS: GateResult(
                        value=obs_a, threshold=min_tracklet_observations,
                        passed=obs_a >= min_tracklet_observations),
                    dlog.ABSOLUTE_THRESHOLD: GateResult(
                        value=(None if best is None else best.score),
                        threshold=subj_bar, passed=best is not None),
                    dlog.TOP2_MARGIN: margin_gate,
                    dlog.RECIPROCAL_BEST: GateResult(
                        value=(None if best is None else best.handle),
                        threshold="mutual-best" if require_reciprocal_best else None,
                        passed=recip_passed,
                        extra={"applies": bool(require_reciprocal_best)}),
                    dlog.TEMPORAL_CONFLICT_SAME_CAMERA: GateResult(
                        value=sum(1 for _, _, _, r in scored
                                  if r == dlog.TEMPORAL_CONFLICT_SAME_CAMERA),
                        threshold=0, passed=True,
                        extra={"note": "count of peers excluded for time overlap"}),
                    # Was _na_gate() -- "does not apply" -- while candidates were
                    # being excluded by exactly this veto (1961 exclusions in run
                    # 20260731_060425). The gate-failure histogram therefore could
                    # never show the guard that dominates the candidate space, which
                    # is the opposite of why the log exists. Counted like its
                    # same-camera twin now. Part M.7 item M-f.
                    dlog.TEMPORAL_CONFLICT_CROSS_CAMERA: GateResult(
                        value=sum(1 for _, _, _, r in scored
                                  if r == dlog.TEMPORAL_CONFLICT_CROSS_CAMERA),
                        threshold=0, passed=True,
                        extra={"note": "count of peers excluded for cross-camera "
                                       "simultaneity",
                               "applies": bool(covis_enabled)}),
                    dlog.GEOMETRIC_UNREACHABLE: GateResult(
                        value=sum(1 for _, _, _, r in scored
                                  if r == dlog.GEOMETRIC_UNREACHABLE),
                        threshold=0, passed=True,
                        extra={"note": "count of peers excluded as physically "
                                       "unreachable from the recorded floor "
                                       "positions",
                               "applies": bool(geo_envelopes)}),
                }, accepted_partner=partner, context=ctx)

        round_index += 1
        if not merged_this_round:
            break

    final_roots = current_roots()
    all_gids = sorted(
        gid for info in tracklets.values() for gid in info["gids"])
    next_gid = (max(all_gids) + 1) if all_gids else 1
    used_survivors = set()
    remap = {}
    for root, cluster in sorted(final_roots.items()):
        existing = sorted(
            gid for k in cluster for gid in tracklets[k]["gids"])
        survivor = None
        for gid in existing:
            if gid not in used_survivors:
                survivor = gid
                break
        if survivor is None:
            survivor = next_gid
            next_gid += 1
        used_survivors.add(survivor)
        for k in sorted(cluster):
            point_ids = tracklets[k]["points"]
            store.set_global_id(point_ids, survivor)
            remap[k] = survivor
            # merged_from records the OTHER tracklets this identity absorbed, so a
            # merge decision can be audited independently of the id it produced.
            _record_outcome(k, dlog.RESOLVED, assigned_id=survivor,
                            merged_from=[handles[m] for m in sorted(cluster)
                                         if m != k])

    # #34: per-tracklet FIT and MARGIN, for the label in the final video.
    #   fit    = this tracklet's prototype vs the prototype of the cluster it
    #            ended up in ("does this piece look like the person it was filed as")
    #   margin = fit minus its best score against any OTHER cluster ("how close was
    #            the call"). Negative would mean it resembles another cluster MORE,
    #            which is worth seeing on screen.
    # Computed after clustering so it describes the decision that was actually made.
    if quality_out is not None:
        final_protos = {}
        for root, cluster in final_roots.items():
            proto = _cluster_prototype(cluster, protos)
            if proto is not None:
                final_protos[root] = proto
        for root, cluster in final_roots.items():
            own = final_protos.get(root)
            if own is None:
                continue
            for k in cluster:
                fit = float(protos[k] @ own)
                others = [float(protos[k] @ p)
                          for r, p in final_protos.items() if r != root]
                quality_out[k] = {
                    "fit": round(fit, 4),
                    "margin": (round(fit - max(others), 4) if others else None),
                }

    log(f"  tracklet reconcile: {len(keys)} tracklets -> "
        f"{len(set(remap.values()))} identities.")
    if geo_envelopes:
        # A veto count of zero is a RESULT, not a silence: on these overlapping
        # cameras it means no proposed merge was physically impossible, which is
        # exactly what the disabled min-transit veto got wrong. Print the pair that
        # came closest so the envelope's position can be judged rather than assumed.
        judged = len(geo_details)
        log(f"  geometry: {geo_vetoes} merge(s) vetoed as physically unreachable, "
            f"out of {judged} tracklet pair(s) it could judge.")
        if judged:
            worst = max(geo_details.items(),
                        key=lambda kv: kv[1]["median_required_speed"])
            (ka, kb), d = worst
            log(f"  geometry: closest call {ka[0]}:{ka[1]} <-> {kb[0]}:{kb[1]} -- "
                f"median {d['median_required_speed']:.2f} units/s vs limit "
                f"{d['limit']:.2f} ({d['pairings']} pairings, "
                f"{d['n_impossible']} individually impossible) -> {d['verdict']}")
    if dl is not None:
        dl.print_summary(log=log)
        written = dl.write()
        if written:
            log(f"  tracklet reconcile: decision log -> {written}")
    return remap
