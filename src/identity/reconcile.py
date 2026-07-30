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
    scores are cleanly separated (see the sweep in demo_identity.py).

Deterministic: the surviving id of a merged cluster is its smallest global_id,
and vectors/point-ids are never touched -- only the global_id payload is
re-stamped. Safe to re-run (idempotent): a reconciled gallery has no pairs left
to merge.
============================================================================
"""

from collections import defaultdict

import numpy as np

from identity import decision_log as dlog
from identity.decision_log import Candidate, DecisionRecord, GateResult


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
        # Prototypes are unit vectors and the shipped embedding is post-ReLU, so a
        # cosine bar outside [0, 1] is a typo, not a choice.
        if not 0.0 <= value <= 1.0:
            log(f"  [reconcile] per_camera[{cam!r}].same_camera_threshold={value} "
                f"is outside [0, 1] (it is a cosine) -- ignored, the global value "
                f"applies.")
            continue
        out[str(cam)] = value
    return out


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
        gids: {original global ids},
    }.
    """
    data = defaultdict(lambda: {
        "vectors": [],
        "points": [],
        "frames": [],
        "gids": set(),
    })

    offset = None
    while True:
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
            gid = pl.get("reid_id", pl.get("global_id"))
            if gid is not None:
                data[key]["gids"].add(int(gid))

        if offset is None:
            break

    out = {}
    for key, info in data.items():
        if not info["vectors"]:
            continue
        frames = info["frames"]
        out[key] = {
            "vectors": info["vectors"],
            "points": info["points"],
            "span": (min(frames), max(frames)),
            "gids": info["gids"],
        }
    return out


def _cluster_prototype(members, protos):
    return _prototype([protos[m] for m in members])


def reconcile_tracklets(store, threshold, run_id=None,
                        same_camera_threshold=0.90,
                        same_camera_thresholds=None,
                        require_reciprocal_best=True,
                        min_tracklet_observations=1, log=print,
                        decision_log=None, top2_margin_threshold=None,
                        top2_margin_basis="eligible"):
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
    if len(keys) < 2:
        # KNOWN DEFECT (REMEDIATION_PLAN.md #25): returning here stamps NO identity
        # on the surviving tracklet, so the whole video renders as a bare
        # "ID <track_id>". Left in place for Phase 1 (instrumentation only) but now
        # VISIBLE in the log instead of silent.
        for k in keys:
            _record_outcome(k, dlog.EXPIRED_UNRESOLVED)
        if dl is not None and keys:
            log(f"  tracklet reconcile: only {len(keys)} tracklet(s) survived -> "
                f"returning with NO identities assigned (known defect #25)")
        return {}

    protos = {k: _prototype(tracklets[k]["vectors"]) for k in keys}
    keys = [k for k in keys if protos[k] is not None]

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
        exclusion. Today the only reason is a same-camera time overlap; Phase 7
        adds TEMPORAL_CONFLICT_CROSS_CAMERA here for non-co-visible camera pairs,
        which is why this is shaped to carry more than one.
        """
        for a in set_a:
            for b in set_b:
                if a[0] != b[0]:
                    continue
                if not _spans_disjoint(tracklets[a]["span"], tracklets[b]["span"]):
                    return dlog.TEMPORAL_CONFLICT_SAME_CAMERA
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

    # Phase 1: repair same-camera track fragmentation with a higher threshold.
    # A pair is a merge candidate iff it shares a camera, its spans are disjoint,
    # and it clears THAT CAMERA's bar (both tracklets are in the same camera here,
    # so the bar is unambiguous -- a[0] == b[0]).
    same_pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[0] != b[0]:
                continue
            if not _spans_disjoint(tracklets[a]["span"], tracklets[b]["span"]):
                continue
            s = float(protos[a] @ protos[b])
            if s >= cam_bar(a[0]):
                same_pairs.append((s, a, b))
    accepted_same = set()
    for s, a, b in sorted(same_pairs, reverse=True):
        if union(a, b):
            log(f"  tracklet reconcile: same-camera merge {a} + {b} "
                f"(cosine {s:.3f})")
            accepted_same.add((a, b))

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
                score = float(protos[a] @ protos[b])
                if conflict_reason({a}, {b}) is not None:
                    reason = dlog.TEMPORAL_CONFLICT_SAME_CAMERA
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
                                             round(float(protos[b] @ protos[best_b[0]]), 6)),
                ))
            margin_gate, best, _, _ = dlog.compute_top2_margin(
                cands, bar_a, basis=top2_margin_basis,
                threshold=top2_margin_threshold)
            partner = None
            if best is not None:
                bkey = next(b for b in peers if handles[b] == best.handle)
                partner = (best.handle if ((a, bkey) in accepted_same
                                           or (bkey, a) in accepted_same) else None)
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
            }, accepted_partner=partner, context="same_camera")

    def current_roots():
        roots = defaultdict(set)
        for k in keys:
            roots[find(k)].add(k)
        return roots

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

        def shared_cameras(a, b):
            """Cameras both clusters contain -- non-empty means merging them makes a
            SAME-CAMERA claim about each of those cameras."""
            return {k[0] for k in roots[a]} & {k[0] for k in roots[b]}

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

        root_scores = {}
        for i, a in enumerate(root_keys):
            for b in root_keys[i + 1:]:
                if not mergeable_cross(a, b):
                    continue
                root_scores[(a, b)] = float(root_protos[a] @ root_protos[b])

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
            bar = pair_threshold(a, b)
            # Name the lane by the CLAIM being made, not by comparing the bar to a
            # number: with per-camera bars one camera's same-camera bar can equal
            # the cross-camera threshold, and then a bar comparison mislabels the
            # lane. (The old message called every Phase 2 merge "cross-camera" even
            # when both printed root keys were the same camera, which made the log
            # actively misleading -- this keeps that fix while staying correct
            # under per-camera bars.)
            lane = "same-camera" if shared_cameras(a, b) else "cross-camera"
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
                    all_scores[(a, b)] = float(root_protos[a] @ root_protos[b])

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
                    dlog.TEMPORAL_CONFLICT_CROSS_CAMERA: _na_gate(),
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

    log(f"  tracklet reconcile: {len(keys)} tracklets -> "
        f"{len(set(remap.values()))} identities.")
    if dl is not None:
        dl.print_summary(log=log)
        written = dl.write()
        if written:
            log(f"  tracklet reconcile: decision log -> {written}")
    return remap
