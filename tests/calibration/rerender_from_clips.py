"""
Re-render a FINISHED run's videos at different reconcile settings -- no cameras,
no models, no walking the room again.

    python tests/calibration/rerender_from_clips.py <run_id>
    python tests/calibration/rerender_from_clips.py <run_id> --cross 0.70,0.75
    python tests/calibration/rerender_from_clips.py <run_id> --cross 0.75 \
        --same "cam_213=0.80" --out "cmp_{name}_strict.mp4"

WHY THIS EXISTS. Judging identity is a VISUAL act -- "that is the same guy" is
something an operator sees, not something a metric reports. But until now seeing
a different threshold meant capturing a whole new run: four cameras, people
walking, five minutes, a finalization that must not be interrupted, and a fresh
set of track ids that cannot be compared to the last set. Two runs were lost to
that loop, and every threshold answer cost one.

A run leaves behind everything needed to redo the last two stages:
  * ._live_src_<cam>.mp4             the CLEAN processed frames
  * ._live_src_<cam>.annotations.json  per-frame box geometry, index-aligned
  * the embeddings in Qdrant, under this run_id

So reconcile + render can be replayed at any settings, on the SAME footage and
the SAME track ids, and the results watched side by side. Requires the run to
have been captured with `live.reconcile.keep_frames: true`.

WHAT IT REPLAYS: everything downstream of the store -- every threshold,
reciprocal-best, min_tracklet_observations, and any change to reconcile's
scoring. WHAT IT CANNOT: anything that changes what got recorded -- the detector
model, imgsz, reid.interval, crop quality. Those still need a live run.

READ-ONLY. The gid map is taken from what reconcile RETURNS, never from ids
stamped into the store, so re-rendering cannot rewrite the gallery and two
settings rendered back-to-back cannot contaminate each other.
"""

import glob
import json
import os
import sys

from _common import arg, bootstrap, header, reconcile_settings

# Where the user invoked us, captured BEFORE bootstrap() -- it chdirs to the repo
# root so relative model paths resolve, which would otherwise silently look for
# clips in the wrong directory. In production the clips ARE in the repo root, so
# that would have worked by luck and broken the moment they were not.
INVOKED_FROM = os.getcwd()
ROOT = bootstrap()
sys.path.insert(0, ROOT)             # for main.render_final_videos

from database.store import PersonVectorStore                      # noqa: E402
from identity.reconcile import (SCORING_MODES,                     # noqa: E402
                                describe_reconcile_kwargs,
                                reconcile_tracklets)


class _ReadOnlyStore:
    """Real observations in, no writes out -- see sweep_reconcile_thresholds."""

    def __init__(self, store):
        self.client = store.client
        self.collection = store.collection

    def set_global_id(self, point_ids, global_id):
        pass

    def clear_global_id(self, point_ids):
        pass


def load_clips(pattern="._live_src_*.mp4"):
    """-> [(camera, clip_path, annotations)] for every clip WITH its sidecar."""
    found, missing = [], []
    for clip in sorted(glob.glob(pattern)):
        side = os.path.splitext(clip)[0] + ".annotations.json"
        if not os.path.exists(side):
            missing.append(clip)
            continue
        with open(side) as f:
            blob = json.load(f)
        cam = blob.get("camera") or os.path.basename(clip)[len("._live_src_"):-4]
        found.append((cam, clip, blob))
    for clip in missing:
        print(f"[rerender] {clip} has no .annotations.json sidecar -- skipped. "
              f"(Runs captured before that sidecar existed cannot be re-rendered; "
              f"the box geometry was only ever in memory.)")
    return found


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
        raise SystemExit("usage: rerender_from_clips.py <run_id> "
                         "[--cross 0.70,0.75] [--same cam_213=0.80] "
                         "[--out 'cmp_{name}_{tag}.mp4']")
    run_id = sys.argv[1]

    # Base = exactly what production would run; only the swept axes are overridden.
    # This tool used to pass neither `covisibility` nor
    # `same_camera_reciprocal_best`, so the video it produced was rendered from a
    # clustering that does not ship -- and the video is the ONLY ground truth this
    # project has. See REMEDIATION_PLAN.md Part M.
    base_kw = reconcile_settings(owns=("--cross", "--scoring"),
                                 extra_flags=("--url", "--path", "--out",
                                              "--fps", "--dir"))
    crosses = [float(x) for x in
               (arg("--cross") or f"{base_kw['threshold']:.2f}").split(",")]
    scorings = (arg("--scoring", "") or "").split(",")
    scorings = [m.strip() for m in scorings if m.strip()] or [None]
    for m in scorings:
        if m is not None and m not in SCORING_MODES:
            raise SystemExit(f"[rerender] unknown --scoring {m!r}; "
                             f"expected any of {list(SCORING_MODES)}")
    out_pattern = arg("--out", "rerender_{name}_{tag}.mp4")
    fps = float(arg("--fps", "0") or 0)

    # Work where the clips are, so the re-rendered videos land beside them.
    clip_dir = os.path.abspath(arg("--dir", INVOKED_FROM))
    os.chdir(clip_dir)
    print(f"[rerender] clips from {clip_dir}")

    clips = load_clips()
    if not clips:
        raise SystemExit(
            "[rerender] no ._live_src_*.mp4 + .annotations.json pairs in this "
            "directory.\n"
            "           Set live.reconcile.keep_frames: true and capture a run "
            "first -- without it\n"
            "           the clips are deleted as soon as the final render "
            "finishes.")
    print(f"[rerender] {len(clips)} camera(s): "
          + ", ".join(f"{c} ({b['frames']} frames @ {b['clip_fps']}fps)"
                      for c, _, b in clips))

    store = PersonVectorStore(path=arg("--path", "qdrant_data"),
                              url=arg("--url", "http://localhost:6333") or None)
    ro = _ReadOnlyStore(store)
    print(f"[rerender] base settings: {describe_reconcile_kwargs(base_kw)}")

    from main import render_final_videos                          # noqa: E402

    jobs = [(cam, clip) for cam, clip, _ in clips]
    shared = {"annotations": {cam: blob["annotations"] for cam, _, blob in clips}}

    combos = [(m, c) for m in scorings for c in crosses]
    for mode, cross in combos:
        tag = f"x{cross:.2f}".replace(".", "")
        if mode is not None:
            tag += f"_{mode}"
        kw = dict(base_kw)
        kw["threshold"] = cross
        if mode is not None:
            kw["scoring"] = mode
        header(f"cross={cross:.2f}  {describe_reconcile_kwargs(kw)}")
        lines = []
        remap = reconcile_tracklets(ro, run_id=run_id, log=lines.append, **kw)
        if not remap:
            print("[rerender] reconcile produced NO assignments for this run_id "
                  "-- nothing to draw. Check the run_id against the store.")
            continue
        print(f"  {len(remap)} tracklet(s) -> {len(set(remap.values()))} identities")

        # Each camera keeps its own recorded rate unless overridden, so the
        # re-render does not inherit the single global output fps that makes
        # cam_224 play fast and cam_219 slow (plan #45/#46).
        for cam, clip, blob in clips:
            # measured_fps is the rate the frames were really produced at;
            # clip_fps is only what the container was tagged with (the old global
            # default), which is why an earlier re-render still played cam_224 fast.
            cam_fps = fps or float(blob.get("measured_fps")
                                   or blob.get("clip_fps") or 20.0)
            render_final_videos(
                [(cam, clip)],
                {"source": {"resize_width": 0},
                 "display": {"output_fps": cam_fps}},
                shared, ro, run_id,
                gid_map=remap,
                out_pattern=out_pattern.replace("{tag}", tag))
        written = [out_pattern.replace("{tag}", tag).format(name=c)
                   for c, _, _ in clips]
        print("  wrote: " + ", ".join(written))

    header("WHAT TO DO WITH THESE")
    print("""  Play the same camera's files from two settings side by side and watch one
  person walk. You are looking for two different failures:

    * one person whose number CHANGES  -> under-merging; the bar is too high
    * one number on two people         -> over-merging; the bar is too low

  Colour is the fast read: ids are coloured by number, so a person changing
  colour mid-walk is a split, and two people sharing a colour is a merge. Watch
  the same person across two CAMERAS too -- carrying one id between them is the
  product claim.

  Note the palette only has 8 colours (plan #35), so two distant ids can collide
  by coincidence. Confirm against the printed number before calling it a merge.""")


if __name__ == "__main__":
    main()
