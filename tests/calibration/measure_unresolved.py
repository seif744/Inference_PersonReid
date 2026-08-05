"""
measure_unresolved.py  --  WHY do visible people render as grey "UNRESOLVED"?

    python tests/calibration/measure_unresolved.py <run_id> [--clips DIR]

THE COMPLAINT THIS EXISTS FOR. The operator watched a re-render and reported that
many plainly visible people carried no identity at all. That is a different failure
from every one this project has been measuring: not a WRONG id, an ABSENT one. And it
is upstream of every threshold, scoring mode, veto and camera-bias question, because a
person who never reaches the store is invisible to all of them -- including to every
calibration script, which reads stored observations.

HOW A BOX ENDS UP GREY. The final video draws boxes from the annotations SIDECAR,
which holds every tracked detection the live pass saw. The id comes from `gid_map`,
built from the STORE. Anything in the first and not the second renders as UNRESOLVED
(main.py: `unresolved=(gid_map.get((name, track_id)) is None)`; drawing.py prints the
word and a neutral grey, deliberately not a palette colour). Three ways to fall in
the gap, and only the third is logged anywhere today:

  1. track_id is None -- ByteTrack had not confirmed the box yet. TrackEmbedder skips
     it (`if det.track_id is None: continue`), so nothing is ever embedded or stored.
  2. every crop was REJECTED by the quality gate -- min_height / min_area /
     min_box_area_ratio / min_aspect / MAX_ASPECT / min_blur / brightness /
     max_occlusion_ratio. reid/service.py records these in `last_quality_rejected`
     and NOTHING READS IT: no counter, no per-camera breakdown, nothing in the run
     summary. A person can be on screen for ten seconds with every crop rejected and
     leave no trace at all.
  3. fewer than `min_tracklet_observations` stored observations -- reconcile clears
     the id (this one does print a line per tracklet).

WHAT THE OUTPUT IS FOR. Cause 1 and 3 are cheap and mostly benign (brief blips).
Cause 2 is the dangerous one, so the second table separates "a track that existed for
three frames" from "a track that was on screen for ten seconds and produced nothing".
The second is a real person the pipeline threw away, and the duration column is how
you tell.

NEEDS NO MODEL. Sidecar JSON plus a Qdrant payload scan. Run it on the box that holds
the clips (the A6000); point --clips at their directory if not the repo root.
"""

import argparse
import collections
import glob
import json
import os

import numpy as np

from _common import bootstrap, header

ROOT = bootstrap()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "persons")


def load_sidecar(path):
    """-> (annotations, frame_ts, run_id). Accepts a bare list or a dict wrapper,
    because the sidecar's shape has changed once already and a hard assumption here
    would make this script fail on exactly the old runs it is most useful for."""
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        annos = raw.get("annotations") or raw.get("frames") or []
        return annos, raw.get("frame_ts") or [], raw.get("run_id")
    return raw, [], None


def store_tracklets(run_id):
    """(camera, track) -> (n_observations, assigned_gid_or_None)."""
    import requests
    obs = collections.Counter()
    gid = {}
    offset = None
    while True:
        body = {"limit": 1000, "with_vector": False,
                "with_payload": ["camera", "track_id", "reid_id", "global_id"],
                "filter": {"must": [{"key": "run_id", "match": {"value": run_id}}]}}
        if offset is not None:
            body["offset"] = offset
        r = requests.post(f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
                          json=body, timeout=300)
        r.raise_for_status()
        res = r.json()["result"]
        for p in res["points"]:
            pl = p["payload"]
            k = (pl.get("camera"), int(pl.get("track_id")))
            obs[k] += 1
            g = pl.get("reid_id", pl.get("global_id"))
            if g is not None:
                gid[k] = int(g)
        offset = res.get("next_page_offset")
        if offset is None:
            break
    return {k: (n, gid.get(k)) for k, n in obs.items()}


NO_TRACK = "no track_id (ByteTrack unconfirmed)"
NOT_STORED = "never reached the store (quality gate / throttle)"
CLEARED = "stored but id CLEARED (< min_tracklet_observations)"
RESOLVED = "resolved"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--clips", default=".")
    ap.add_argument("--long-track-sec", type=float, default=2.0,
                    help="a never-stored track visible longer than this is a person "
                         "the pipeline threw away, not a blip")
    args = ap.parse_args()

    store = store_tracklets(args.run_id)
    if not store:
        print(f"[calib] run {args.run_id}: no stored observations at "
              f"{QDRANT_URL} -- wrong host, or the run persisted nothing.")
        return 1
    sidecars = sorted(glob.glob(os.path.join(args.clips,
                                             "._live_src_*.annotations.json")))
    if not sidecars:
        print(f"[calib] no ._live_src_*.annotations.json under {args.clips!r}. "
              f"Runs captured before the sidecar existed cannot be measured; "
              f"keep_frames must have been true.")
        return 1

    print(f"[calib] run {args.run_id}  qdrant={QDRANT_URL}  clips={args.clips}")
    header("A. EVERY PERSON-BOX IN THE FINAL VIDEO, BY WHETHER IT CARRIES AN ID")
    print(f"  {'camera':<10}{'frames':>7}{'boxes':>8}{'resolved':>10}"
          f"{'no track_id':>13}{'not stored':>12}{'cleared':>9}{'UNRESOLVED':>12}")
    grand = collections.Counter()
    per_cam_tracks = {}
    for sc in sidecars:
        cam = os.path.basename(sc)[len("._live_src_"):-len(".annotations.json")]
        annos, fts, clip_run = load_sidecar(sc)
        # HARD REFUSAL on a cross-run join. Clip filenames carry no run_id --
        # `._live_src_<cam>.mp4` is overwritten by every run while keep_frames
        # preserves it -- so the files on disk belong to the LAST run. Joining an
        # earlier run's store against them compares two disjoint ByteTrack id
        # spaces, which reads as a total identity failure: measured on 064551 vs
        # 120409's clips, cam_219 came out 100% UNRESOLVED and the headline was
        # 84% of all boxes. That number was a join error, not a pipeline failure.
        if clip_run is not None and clip_run != args.run_id:
            print(f"  REFUSED {os.path.basename(sc)}: sidecar is from run "
                  f"{clip_run}, not {args.run_id}. ByteTrack renumbers every run, "
                  f"so joining them would report a meaningless UNRESOLVED rate. "
                  f"Measure run {clip_run} instead, or point --clips at a copy of "
                  f"{args.run_id}'s footage.")
            continue
        if clip_run is None:
            print(f"  {os.path.basename(sc)}: NO run_id in the sidecar -- cannot "
                  f"verify it belongs to {args.run_id}. Check its frame count "
                  f"against the run's stored frame span before trusting this.")
        counts = collections.Counter()
        # per-track: first/last frame index seen, box heights
        seen = collections.defaultdict(lambda: {"first": None, "last": None, "h": []})
        for fi, boxes in enumerate(annos):
            for b in boxes:
                if isinstance(b, dict):
                    tid = b.get("track_id")
                    y1, y2 = b.get("y1"), b.get("y2")
                else:
                    tid, y1, y2 = b[4], b[1], b[3]
                if tid is None:
                    counts[NO_TRACK] += 1
                    continue
                tid = int(tid)
                rec = seen[tid]
                rec["first"] = fi if rec["first"] is None else rec["first"]
                rec["last"] = fi
                if y1 is not None and y2 is not None:
                    rec["h"].append(int(y2) - int(y1))
                entry = store.get((cam, tid))
                if entry is None:
                    counts[NOT_STORED] += 1
                elif entry[1] is None:
                    counts[CLEARED] += 1
                else:
                    counts[RESOLVED] += 1
        boxes = sum(counts.values())
        unres = boxes - counts[RESOLVED]
        grand.update(counts)
        per_cam_tracks[cam] = (seen, fts, len(annos))
        print(f"  {cam:<10}{len(annos):>7}{boxes:>8}{counts[RESOLVED]:>10}"
              f"{counts[NO_TRACK]:>13}{counts[NOT_STORED]:>12}{counts[CLEARED]:>9}"
              f"{(unres / boxes if boxes else 0):>11.0%}")
    tot = sum(grand.values())
    if tot:
        print(f"\n  ALL CAMERAS: {tot} person-boxes drawn, "
              f"{tot - grand[RESOLVED]} ({(tot - grand[RESOLVED]) / tot:.0%}) carry "
              f"NO identity.")
        for cause in (NO_TRACK, NOT_STORED, CLEARED):
            if grand[cause]:
                print(f"    {grand[cause] / tot:5.1%}  {cause}")

    header("B. THE TRACKS THAT NEVER REACHED THE STORE -- blip, or a real person?")
    print("  A never-stored track that was on screen for a long time is a person the")
    print("  QUALITY GATE threw away. reid/service.py records the rejection reason in")
    print("  `last_quality_rejected` and nothing reads it, so the reason is not")
    print("  recoverable after the fact -- only the duration is.\n")
    print(f"  {'camera':<10}{'never-stored tracks':>20}{'of which > ' + str(args.long_track_sec) + 's':>18}"
          f"{'longest (s)':>13}{'median box h':>14}")
    worst = []
    for cam, (seen, fts, nframes) in sorted(per_cam_tracks.items()):
        fps = None
        if fts and len(fts) > 1 and fts[0] is not None and fts[-1] is not None:
            span = fts[-1] - fts[0]
            fps = (len(fts) - 1) / span if span > 0 else None
        never = [(t, r) for t, r in seen.items() if (cam, t) not in store]
        durs = []
        for t, r in never:
            nf = (r["last"] - r["first"] + 1)
            durs.append((nf / fps if fps else float("nan"), t, r))
        durs.sort(reverse=True)
        long_ones = [d for d in durs if d[0] == d[0] and d[0] > args.long_track_sec]
        hs = [int(np.median(r["h"])) for _, _, r in durs if r["h"]]
        print(f"  {cam:<10}{len(never):>20}{len(long_ones):>18}"
              f"{(durs[0][0] if durs else float('nan')):>13.1f}"
              f"{(int(np.median(hs)) if hs else 0):>14}")
        worst.extend((cam, d[0], d[1], d[2]) for d in durs[:5])
    if worst:
        worst.sort(key=lambda x: -(x[1] if x[1] == x[1] else 0))
        print("\n  Longest never-stored tracks (scrub the clip to these frames and look):")
        print(f"    {'camera':<10}{'track':>7}{'sec':>7}{'frames':>16}{'median box h':>14}")
        for cam, sec, t, r in worst[:12]:
            span = "{}-{}".format(r["first"], r["last"])
            med_h = int(np.median(r["h"])) if r["h"] else 0
            print(f"    {cam:<10}{t:>7}{sec:>7.1f}{span:>16}{med_h:>14}")

    header("WHAT TO DO WITH THIS")
    print("  * A high `not stored` fraction with LONG durations => the crop-quality")
    print("    gate is rejecting real people. The gate is in reid/service.py; the")
    print("    candidates for an office are max_aspect (1.20 -- a seated person is")
    print("    near-square and gets rejected outright) and min_box_area_ratio /")
    print("    min_height for anyone far from the camera.")
    print("  * A high `no track_id` fraction => ByteTrack is not confirming boxes;")
    print("    that is a tracker/detector question, not an identity one.")
    print("  * A high `cleared` fraction => min_tracklet_observations. Cheapest fix,")
    print("    and the only one already visible in the reconcile log.")
    print("  * FIX THE INSTRUMENTATION FIRST. Until `last_quality_rejected` is")
    print("    counted per camera per reason, this script can only measure the gap,")
    print("    never attribute it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
