"""
Does co-temporal FLOOR POSITION separate people where appearance cannot?

    python tests/calibration/measure_covisible_geometry.py <run_id>
    python tests/calibration/measure_covisible_geometry.py <run_id> --window 0.25
    python tests/calibration/measure_covisible_geometry.py <run_id> --pairs cam_219:6,cam_224:5

WHAT THIS DECIDES. REMEDIATION_PLAN.md M.9.24 concluded that appearance is
exhausted on this footage: every global bar now trades one operator-confirmed
error for another, and the margins sit inside the noise of the embedding.
ADR-003 proposes geometry as the way out. Before building a subsystem for it,
this script asks the only question that matters, on footage already captured:

    when two tracks are seen AT THE SAME INSTANT by the two co-visible cameras,
    does where they STAND tell same from different -- in the cases where cosine
    does not?

THE CONTROL, AND WHY IT NEEDS NO LABELS. Two tracklets that co-occur in ONE
camera are provably two different people; one body cannot be two simultaneous
detections. So for a tracklet X in cam_a and two mutually co-present tracklets
Y, Z in cam_b, X can be the same person as AT MOST ONE of them. Every such
triangle is a self-validating test: geometry must put X close to one arm and far
from the other. No operator, no ground truth, no threshold chosen in advance.

This is exactly the configuration behind the one operator complaint that
survived M.9.28 -- cam_219:6 against cam_224:5 and cam_224:30, where 224:5 and
224:30 co-occur (5 sampled instants) and appearance scores 0.717 / 0.733 / 0.683
across all three modes, below every bar in the config.

WHAT IT DOES NOT DO. No world frame, no metres. Residuals are cam_b PIXELS,
reported beside the observed width of a person in the same image, so "12 px" is
read as "a fifth of a person" rather than converted through a scale nobody
measured. Metres are needed later, for the 0.5 m / 3.0 m mining bands, for BEV,
and for bringing the non-co-visible cameras into one frame -- not for this.

Foot point is BOX-BOTTOM-CENTRE, the weakest of the three sources in ADR-003A
section 2, because live runs with `live.inference.pose_ensemble: false` have no
ankle keypoints to use. That is a floor on the achievable accuracy, not a
ceiling: if the signal already separates from box bottoms, ankles only widen it.

READ-ONLY. Scrolls the store with `with_vectors=False` and writes nothing.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

from _common import arg, bootstrap, describe, flag, header, project_root

bootstrap()

from database.store import PersonVectorStore                        # noqa: E402

KNOWN_FLAGS = {"--url", "--path", "--calib", "--window", "--copresence-window",
               "--min-samples", "--min-common", "--pairs", "--top", "--no-tau"}

LABELS_PATH = os.path.join(project_root(), "calibration", "link_labels.jsonl")

# MV3DT uses minCommonFrames4MatchScore = 2 for exactly this: below a couple of
# shared instants a geometric "match" is an accident of sampling, not evidence.
DEFAULT_MIN_COMMON = 2


def strict_flags():
    unknown = [t for t in sys.argv[1:]
               if t.startswith("--") and t not in KNOWN_FLAGS]
    if unknown:
        raise SystemExit(f"[calib] unknown flag(s) {unknown}.\n"
                         f"        Known: {', '.join(sorted(KNOWN_FLAGS))}")


def key_str(k):
    return f"{k[0]}:{k[1]}"


# ------------------------------------------------------------------ calibration

def load_calibration(cam_a=None, cam_b=None):
    path = arg("--calib")
    if path is None:
        cdir = os.path.join(project_root(), "calibration")
        found = sorted(f for f in os.listdir(cdir)
                       if f.startswith("homography_") and f.endswith(".json")) \
            if os.path.isdir(cdir) else []
        if not found:
            raise SystemExit(
                "[calib] no calibration/homography_*.json found.\n"
                "        Make one first:\n"
                "          python tests/calibration/click_covisible_points.py --export\n"
                "          python tests/calibration/click_covisible_points.py --click")
        if len(found) > 1:
            print(f"[calib] {len(found)} homographies present; using {found[0]}. "
                  f"Pass --calib to choose.")
        path = os.path.join(cdir, found[0])
    with open(path) as f:
        rec = json.load(f)
    rec["_path"] = path
    return rec


def report_calibration(rec):
    header("THE CALIBRATION IN FORCE")
    print(f"  file        {os.path.relpath(rec['_path'], project_root())}")
    print(f"  version     {rec.get('calib_version')}   ({rec.get('created_at')})")
    print(f"  maps        {rec['cam_a']} -> {rec['cam_b']}   "
          f"[{rec.get('source')}]")
    print(f"  points      {rec.get('n_points')} "
          f"({rec.get('ransac_inliers')} RANSAC inliers)")
    med = rec.get("heldout_px_median")
    if med is None:
        print(f"  held-out    NOT MEASURABLE -- fewer than 5 points. Every number")
        print(f"              below inherits an unknown calibration error.")
    else:
        print(f"  held-out    median {med:.1f} px   p95 "
              f"{rec.get('heldout_px_p95', float('nan')):.1f} px   max "
              f"{rec.get('heldout_px_max', float('nan')):.1f} px  (in "
              f"{rec['cam_b']} pixels)")
    if (rec.get("n_points") or 0) < 8:
        print(f"  !! ADR-003A section 1.1 asks for 8-12 points. Below that the")
        print(f"     error estimate is weak, and below 5 it does not exist.")
    return rec


# ------------------------------------------------------------------ observations

def load_observations(store, run_id, cameras):
    """-> {(camera, track_id): [(ts, foot_xy, bbox), ...] sorted by ts}

    Server-side run_id filter with the same client-side fallback reconcile uses
    (#20): embedded mode and older clients reject the filter, and losing the
    measurement to that would be silly.
    """
    scroll_filter = None
    try:
        from qdrant_client import models as qmodels
        scroll_filter = qmodels.Filter(must=[qmodels.FieldCondition(
            key="run_id", match=qmodels.MatchValue(value=run_id))])
    except Exception:                                           # noqa: BLE001
        scroll_filter = None

    out = defaultdict(list)
    seen_run, no_bbox, no_ts = 0, 0, 0
    offset = None
    while True:
        try:
            pts, offset = store.client.scroll(
                store.collection, limit=1000, offset=offset,
                scroll_filter=scroll_filter, with_payload=True,
                with_vectors=False)
        except Exception:                                       # noqa: BLE001
            scroll_filter = None
            pts, offset = store.client.scroll(
                store.collection, limit=1000, offset=offset,
                with_payload=True, with_vectors=False)
        for p in pts:
            pl = p.payload or {}
            if pl.get("run_id") != run_id:
                continue
            seen_run += 1
            cam = pl.get("camera")
            if cam not in cameras:
                continue
            ts, bbox = pl.get("ts"), pl.get("bbox")
            if ts is None:
                no_ts += 1
                continue
            if not bbox or len(bbox) != 4:
                no_bbox += 1
                continue
            x1, y1, x2, y2 = (float(v) for v in bbox)
            foot = ((x1 + x2) / 2.0, y2)          # box bottom-centre
            out[(cam, int(pl["track_id"]))].append((float(ts), foot,
                                                    (x1, y1, x2, y2)))
        if offset is None:
            break

    for k in out:
        out[k].sort(key=lambda r: r[0])
    return dict(out), seen_run, no_bbox, no_ts


def report_data(obs, run_id, cameras, seen_run, no_bbox, no_ts):
    header(f"THE DATA -- run {run_id}")
    if seen_run == 0:
        raise SystemExit(
            f"[calib] run_id {run_id!r} has no observations in this store.")
    for cam in cameras:
        tracks = {k: v for k, v in obs.items() if k[0] == cam}
        n = sum(len(v) for v in tracks.values())
        print(f"  {cam:<10} {len(tracks):>3} tracklet(s)  {n:>5} observation(s)")
    if no_bbox or no_ts:
        print(f"  skipped: {no_bbox} without bbox, {no_ts} without ts")
    if not obs:
        raise SystemExit(
            f"[calib] run {run_id} carries no bbox/ts on these cameras.\n"
            f"        Only the LIVE path persists them (identity_stage.py's\n"
            f"        _observation_payload); IdentityService._commit does not.\n"
            f"        A file-batch run cannot be measured this way.")


def check_image_space(obs, rec):
    """A homography belongs to the pixel space it was authored in (ADR-003A 1.3).
    Boxes that run past the recorded frame size mean the two disagree -- so say
    so and stop, rather than emit a plausible wrong number."""
    for cam, size in ((rec["cam_a"], rec["image_size_a"]),
                      (rec["cam_b"], rec["image_size_b"])):
        rows = [r for k, v in obs.items() if k[0] == cam for r in v]
        if not rows:
            continue
        max_x = max(r[2][2] for r in rows)
        max_y = max(r[2][3] for r in rows)
        w, h = float(size[0]), float(size[1])
        if max_x > w * 1.02 or max_y > h * 1.02:
            raise SystemExit(
                f"[calib] {cam}: boxes reach ({max_x:.0f}, {max_y:.0f}) but the "
                f"calibration was authored at {w:.0f}x{h:.0f}.\n"
                f"        The homography does not belong to this run's pixel "
                f"space (source.resize_width, or a different camera profile).\n"
                f"        Re-export the frames from THIS run and re-click.")


# ------------------------------------------------------------------ the geometry

def apply_h(H, pts):
    """(N,2) -> (N,2) through the homography; NaN at/behind the horizon."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    out = hom @ np.asarray(H, dtype=np.float64).T
    w = out[:, 2:3]
    w = np.where(np.abs(w) < 1e-9, np.nan, w)
    return out[:, :2] / w


def match_in_time(rows_a, rows_b, window, tau=0.0):
    """Nearest-in-time pairing -> (idx_a, idx_b, dt) for |ts_a + tau - ts_b| <= window.

    Each cam_a observation takes its single closest cam_b partner. Frames are
    dropped by design on the live path (NewestSlot keeps only the freshest), so
    the two streams are sparse and unevenly spaced -- an all-pairs join would
    count one instant many times and inflate n.
    """
    ts_a = np.array([r[0] for r in rows_a]) + tau
    ts_b = np.array([r[0] for r in rows_b])
    if ts_a.size == 0 or ts_b.size == 0:
        return []
    idx = np.searchsorted(ts_b, ts_a)
    out = []
    for i, j in enumerate(idx):
        best, best_dt = None, None
        for cand in (j - 1, j):
            if 0 <= cand < len(ts_b):
                dt = abs(ts_a[i] - ts_b[cand])
                if best_dt is None or dt < best_dt:
                    best, best_dt = cand, dt
        if best is not None and best_dt <= window:
            out.append((i, best, float(ts_a[i] - ts_b[best])))
    return out


def pair_residuals(rows_a, rows_b, H, window, tau=0.0):
    """-> dict of residuals in cam_b pixels for one tracklet pair."""
    matches = match_in_time(rows_a, rows_b, window, tau)
    if not matches:
        return {"n": 0}
    feet_a = np.array([rows_a[i][1] for i, _, _ in matches])
    feet_b = np.array([rows_b[j][1] for _, j, _ in matches])
    widths = np.array([rows_b[j][2][2] - rows_b[j][2][0] for _, j, _ in matches])
    dts = np.array([dt for _, _, dt in matches])

    proj = apply_h(H, feet_a)
    ok = ~np.isnan(proj).any(axis=1)
    horizon = int((~ok).sum())
    if not ok.any():
        return {"n": 0, "horizon": horizon}
    resid = np.linalg.norm(proj[ok] - feet_b[ok], axis=1)
    width = float(np.median(widths[ok])) if widths[ok].size else float("nan")
    return {
        "n": int(ok.sum()),
        "horizon": horizon,
        "resid": resid,
        "median": float(np.median(resid)),
        "p95": float(np.percentile(resid, 95)),
        "widths": float(width),
        "in_widths": float(np.median(resid) / width) if width and width > 0
        else float("nan"),
        "dt_ms": float(np.median(np.abs(dts)) * 1000.0),
    }


# ------------------------------------------------------------------ co-presence

def interp_at(rows, times, max_gap=1.0):
    """Where a tracklet was at arbitrary instants -> (N,2), NaN where unknown.

    Linear interpolation between the two bracketing observations. This exists for
    the tau scan below and NOT for the main measurement, and the difference
    matters: nearest-neighbour pairing makes the residual a STEP function of tau
    (shifting tau by less than half a sampling period changes no pairing at all),
    so the curve is flat across a plateau and its argmin is arbitrary within it.
    A synthetic walker with a known +120 ms offset came back as -60 ms for
    exactly this reason. Interpolating makes the residual vary continuously with
    tau, which is what makes the minimum mean something.

    `max_gap` refuses to interpolate across a long absence -- a person who left
    and came back was not walking in a straight line between the two sightings.
    """
    ts = np.array([r[0] for r in rows], dtype=np.float64)
    xy = np.array([r[1] for r in rows], dtype=np.float64)
    out = np.full((len(times), 2), np.nan)
    if ts.size < 2:
        return out
    idx = np.searchsorted(ts, times)
    for n, t in enumerate(times):
        j = int(idx[n])
        if j <= 0 or j >= ts.size:
            continue                       # outside the tracklet's span
        t0, t1 = ts[j - 1], ts[j]
        if (t1 - t0) > max_gap:
            continue
        w = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        out[n] = xy[j - 1] * (1.0 - w) + xy[j] * w
    return out


def copresent(rows_a, rows_b, window, min_common):
    """Two tracklets in ONE camera that share >= min_common instants are two
    different people. This is the same axiom reconcile's same-camera veto uses,
    measured on the sampled instants rather than the min/max envelope (M.2)."""
    n = len(match_in_time(rows_a, rows_b, window))
    return n >= min_common, n


# ------------------------------------------------------------------ the report

def tau_scan(obs, stats, H, min_samples, max_gap=1.0):
    """Latency differential between the two streams (ADR-003A section 3.2).

    `ts` is decode time, not sensor time, so the streams can sit at a constant
    offset. If they do, the residual of genuine same-person pairs is minimised at
    a non-zero tau. Circular by construction -- it takes each cam_a tracklet's
    LOWEST-residual partner as if it were the right one -- so it is reported as a
    diagnostic and never corrects anything here.

    Uses interpolation, not nearest-neighbour pairing: see interp_at.
    """
    designated = {}
    for (ka, kb), st in stats.items():
        if st["n"] < min_samples:
            continue
        if ka not in designated or st["median"] < designated[ka][1]:
            designated[ka] = (kb, st["median"])
    if not designated:
        return None, None, []
    pairs = [(ka, kb) for ka, (kb, _) in designated.items()]

    taus = np.arange(-0.60, 0.601, 0.01)
    curve = []
    for tau in taus:
        meds = []
        for ka, kb in pairs:
            rows_a, rows_b = obs[ka], obs[kb]
            times = np.array([r[0] for r in rows_a]) + float(tau)
            pos_b = interp_at(rows_b, times, max_gap)
            proj = apply_h(H, [r[1] for r in rows_a])
            ok = ~(np.isnan(proj).any(axis=1) | np.isnan(pos_b).any(axis=1))
            if ok.sum() >= min_samples:
                meds.append(float(np.median(
                    np.linalg.norm(proj[ok] - pos_b[ok], axis=1))))
        curve.append(float(np.median(meds)) if meds else float("nan"))
    curve = np.array(curve)
    if np.all(np.isnan(curve)):
        return None, None, pairs
    return float(taus[int(np.nanargmin(curve))]), (taus, curve), pairs


def load_labels(run_id):
    """Operator labels FOR THIS RUN ONLY.

    Track ids are run-scoped -- cam_219:7 in one run is a different person from
    cam_219:7 in the next -- so a label may only be applied to the run it was
    recorded against.
    """
    out = {}
    if not os.path.exists(LABELS_PATH):
        return out
    with open(LABELS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "verdict" not in rec or rec.get("run_id") != run_id:
                continue
            out[frozenset((rec["a"], rec["b"]))] = rec
    return out


def main():
    strict_flags()
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        raise SystemExit(__doc__.strip().split("\n\n")[1])
    run_id = sys.argv[1]

    window = float(arg("--window", "0.15"))
    copres_window = float(arg("--copresence-window", "0.20"))
    min_samples = int(arg("--min-samples", "5"))
    min_common = int(arg("--min-common", str(DEFAULT_MIN_COMMON)))
    top = int(arg("--top", "20"))
    url = arg("--url", "http://localhost:6333") or None

    rec = report_calibration(load_calibration())
    cam_a, cam_b = rec["cam_a"], rec["cam_b"]
    H = np.array(rec["H_a_to_b"], dtype=np.float64)

    store = PersonVectorStore(path=arg("--path", "qdrant_data"), url=url)
    obs, seen_run, no_bbox, no_ts = load_observations(store, run_id,
                                                      {cam_a, cam_b})
    report_data(obs, run_id, (cam_a, cam_b), seen_run, no_bbox, no_ts)
    check_image_space(obs, rec)

    a_tracks = {k: v for k, v in obs.items() if k[0] == cam_a}
    b_tracks = {k: v for k, v in obs.items() if k[0] == cam_b}
    if not a_tracks or not b_tracks:
        raise SystemExit(f"[calib] need tracklets in BOTH {cam_a} and {cam_b}.")

    # ---- every co-temporal cross-camera tracklet pair --------------------
    stats = {}
    for ka, ra in a_tracks.items():
        for kb, rb in b_tracks.items():
            st = pair_residuals(ra, rb, H, window)
            if st["n"] >= min_samples:
                stats[(ka, kb)] = st

    header(f"CO-TEMPORAL COVERAGE  (|dt| <= {window * 1000:.0f} ms)")
    print(f"  {len(stats)} cross-camera tracklet pair(s) with >= {min_samples} "
          f"co-temporal sample(s)")
    if not stats:
        raise SystemExit(
            f"[calib] no cross-camera pair reaches {min_samples} samples within "
            f"{window * 1000:.0f} ms.\n"
            f"        Try a wider --window. If it stays empty at 400 ms the two "
            f"streams are not co-temporal at all, and Rule 2 (ADR-003 section 3) "
            f"silences geometry for this run.")
    all_dt = np.array([s["dt_ms"] for s in stats.values()])
    print(describe(all_dt, "per-pair median |dt|, ms"))
    horizon = sum(s.get("horizon", 0) for s in stats.values())
    if horizon:
        print(f"  {horizon} sample(s) projected to/past the horizon and were "
              f"dropped (see apply_h)")

    # ---- scale reference -------------------------------------------------
    widths = np.array([s["widths"] for s in stats.values()
                       if np.isfinite(s["widths"])])
    header("SCALE REFERENCE")
    print(f"  A person in {cam_b} is a median of {np.median(widths):.0f} px wide "
          f"across these samples.")
    print(f"  Residuals below are given in px AND in person-widths, so no metric")
    print(f"  scale is assumed. Shoulder width is roughly half a metre, if you")
    print(f"  want a rough physical reading.")

    # ---- the clock -------------------------------------------------------
    if not flag("--no-tau"):
        tau, curve, designated = tau_scan(obs, stats, H, min_samples)
        header("THE CLOCK -- latency differential scan (G0-zero)")
        if tau is None:
            print("  not computable: no pair had enough samples.")
        else:
            print(f"  best-matching partner assumed per {cam_a} tracklet "
                  f"({len(designated)} pair(s)); residual minimised at "
                  f"tau = {tau * 1000:+.0f} ms")
            taus, vals = curve
            at0 = float(vals[int(np.argmin(np.abs(taus)))])
            best = float(np.nanmin(vals))
            print(f"  median residual: {at0:.1f} px at tau=0, {best:.1f} px at "
                  f"the minimum")
            if abs(tau) > 0.15:
                print(f"  !! {abs(tau) * 1000:.0f} ms is large. At walking pace "
                      f"that is ~{abs(tau) * 1.4 * 100:.0f} cm of induced error. "
                      f"Fix capture (RTSP buffering, matched encoder settings) "
                      f"before modelling around it -- ADR-003A section 3.1.")
            print(f"  Circular by construction: it trusts that the lowest-residual "
                  f"partner is the right one. Diagnostic only.")

    # ---- the triangles: the test that needs no labels --------------------
    header("THE TRIANGLES -- proven-different controls")
    print(f"  Two {cam_b} tracklets that co-occur are two different people, so a")
    print(f"  {cam_a} tracklet can match AT MOST ONE of them. Geometry must put it")
    print(f"  near one arm and far from the other.\n")
    triangles = []
    b_keys = sorted(b_tracks)
    for i, kb1 in enumerate(b_keys):
        for kb2 in b_keys[i + 1:]:
            together, n_common = copresent(b_tracks[kb1], b_tracks[kb2],
                                           copres_window, min_common)
            if not together:
                continue
            for ka in sorted(a_tracks):
                s1, s2 = stats.get((ka, kb1)), stats.get((ka, kb2))
                if s1 is None or s2 is None:
                    continue
                near, far = ((kb1, s1), (kb2, s2)) if s1["median"] <= s2["median"] \
                    else ((kb2, s2), (kb1, s1))
                triangles.append((ka, near, far, n_common))
    if not triangles:
        print(f"  none: no two {cam_b} tracklets co-occur while a {cam_a} tracklet")
        print(f"  is co-temporal with both. Widen --window, or this run simply")
        print(f"  never had two people in the room together.")
    else:
        print(f"  {'anchor':<16}{'nearer arm':<16}{'px':>7}{'/w':>6}"
              f"{'  |  ':<5}{'far arm':<16}{'px':>7}{'/w':>6}{'ratio':>8}")
        for ka, (kn, sn), (kf, sf) in [(t[0], t[1], t[2]) for t in triangles]:
            ratio = sf["median"] / sn["median"] if sn["median"] > 0 else float("inf")
            print(f"  {key_str(ka):<16}{key_str(kn):<16}{sn['median']:>7.1f}"
                  f"{sn['in_widths']:>6.2f}{'  |  ':<5}{key_str(kf):<16}"
                  f"{sf['median']:>7.1f}{sf['in_widths']:>6.2f}{ratio:>8.2f}")
        ratios = np.array([(t[2][1]["median"] / t[1][1]["median"])
                           for t in triangles if t[1][1]["median"] > 0])
        near_w = np.array([t[1][1]["in_widths"] for t in triangles])
        far_w = np.array([t[2][1]["in_widths"] for t in triangles])
        print()
        print(describe(ratios, "far/near residual ratio"))
        print(describe(near_w, "nearer arm, person-widths"))
        print(describe(far_w, "far arm, person-widths"))

    # ---- the full table --------------------------------------------------
    header(f"ALL CO-TEMPORAL PAIRS, closest first  (top {top})")
    labels = load_labels(run_id)
    ranked = sorted(stats.items(), key=lambda kv: kv[1]["median"])
    print(f"  {'pair':<34}{'n':>5}{'px':>8}{'/width':>8}{'p95px':>8}"
          f"{'dt ms':>7}  label")
    for (ka, kb), st in ranked[:top]:
        lbl = labels.get(frozenset((key_str(ka), key_str(kb))))
        mark = f"  {lbl['verdict'].upper()}" if lbl else ""
        if lbl and lbl.get("stated"):
            mark += " (stated)"
        print(f"  {key_str(ka) + ' <-> ' + key_str(kb):<34}{st['n']:>5}"
              f"{st['median']:>8.1f}{st['in_widths']:>8.2f}{st['p95']:>8.1f}"
              f"{st['dt_ms']:>7.0f}{mark}")

    # ---- named pairs -----------------------------------------------------
    wanted = arg("--pairs")
    if wanted:
        header("PAIRS YOU ASKED FOR")
        keys = [k.strip() for k in wanted.split(",") if k.strip()]
        parsed = []
        for k in keys:
            cam, _, tid = k.partition(":")
            parsed.append((cam, int(tid)))
        for i, ka in enumerate(parsed):
            for kb in parsed[i + 1:]:
                pa, pb = (ka, kb) if ka[0] == cam_a else (kb, ka)
                st = stats.get((pa, pb))
                if st is None:
                    print(f"  {key_str(ka)} <-> {key_str(kb)}: no co-temporal "
                          f"samples at this window (or not a {cam_a}/{cam_b} pair)")
                    continue
                print(f"  {key_str(pa)} <-> {key_str(pb)}: n={st['n']}  "
                      f"median {st['median']:.1f} px = {st['in_widths']:.2f} "
                      f"person-widths  p95 {st['p95']:.1f} px  "
                      f"|dt| {st['dt_ms']:.0f} ms")

    # ---- verdict ---------------------------------------------------------
    header("VERDICT")
    if not triangles:
        print("  UNDECIDED -- no proven-different control existed in this run.")
        print("  The ranked table above is suggestive but nothing in it is")
        print("  self-validating. Do not build on it.")
    else:
        ratios = np.array([(t[2][1]["median"] / t[1][1]["median"])
                           for t in triangles if t[1][1]["median"] > 0])
        near_w = np.array([t[1][1]["in_widths"] for t in triangles])
        clean = float(np.median(ratios)) if ratios.size else float("nan")
        tight = float(np.median(near_w)) if near_w.size else float("nan")
        print(f"  {len(triangles)} triangle(s). Median far/near ratio {clean:.2f}; "
              f"nearer arm sits at {tight:.2f} person-widths.")
        if clean >= 3.0 and tight <= 0.75:
            print("  SEPARATES. Position tells these apart where cosine did not.")
            print("  ADR-003's matching path has earned its Phase 0. Next: metric")
            print("  world coordinates, so the bands can be stated in metres.")
        elif clean >= 1.8:
            print("  PARTIAL. The right arm is consistently nearer, but not by")
            print("  enough to carry a decision on its own. Improve the foot point")
            print("  (ankles, ADR-003A section 2) and the calibration before")
            print("  concluding either way.")
        else:
            print("  DOES NOT SEPARATE on this run. Either the calibration is too")
            print("  loose, the clock offset is too large (see the tau scan), or")
            print("  the co-visible geometry genuinely does not discriminate here.")
            print("  ADR-003 section 1.1 anticipated this: the SUPERVISION path")
            print("  (003B) is unaffected and still worth shipping.")
    print(f"\n  Sample sizes are small by nature. Treat this as a hypothesis and")
    print(f"  re-run it on a second run before it changes any code.")


if __name__ == "__main__":
    main()
