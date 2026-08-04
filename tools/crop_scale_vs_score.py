"""
Is the same-person score drop a CROP-QUALITY problem rather than an appearance one?

Read-only. For one run: per same-camera tracklet pair, the prototype cosine beside
the two tracklets' crop geometry. If low-scoring same-camera pairs are systematically
pairs where one side is small/blurry and the other is not, the defect is the input
gate (reid.quality.min_height = 64 px, upscaled to 384x128 -- a 6x stretch), not the
similarity bar. That distinction matters because pixels-on-target is a physical
quantity that transfers across footage, while a cosine bar demonstrably does not.

    python crop_scale_vs_score.py <run_id> [--url http://localhost:6333]
"""

import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "src")
from database.store import PersonVectorStore                      # noqa: E402
from identity.reconcile import _prototype                         # noqa: E402

if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
    raise SystemExit(__doc__)
RUN = sys.argv[1]
URL = "http://localhost:6333"
for i, a in enumerate(sys.argv):
    if a == "--url" and i + 1 < len(sys.argv):
        URL = sys.argv[i + 1]

store = PersonVectorStore(url=URL, ensure_collection=False, read_only=True)

per = defaultdict(lambda: {"vecs": [], "h": [], "w": [], "blur": [], "ts": []})
offset = None
while True:
    pts, offset = store.client.scroll(store.collection, limit=1000, offset=offset,
                                      with_payload=True, with_vectors=True)
    for p in pts:
        pl = p.payload or {}
        if pl.get("run_id") != RUN:
            continue
        key = (pl.get("camera"), pl.get("track_id"))
        rec = per[key]
        v = p.vector
        if isinstance(v, dict):
            v = next(iter(v.values()), None)
        if v is not None:
            rec["vecs"].append(np.asarray(v, dtype=np.float32))
        bb = pl.get("bbox")
        if bb and len(bb) == 4:
            rec["h"].append(float(bb[3]) - float(bb[1]))
            rec["w"].append(float(bb[2]) - float(bb[0]))
        cq = pl.get("crop_quality")
        if isinstance(cq, dict) and cq.get("blur") is not None:
            rec["blur"].append(float(cq["blur"]))
        if pl.get("ts") is not None:
            rec["ts"].append(float(pl["ts"]))
    if offset is None:
        break

if not per:
    raise SystemExit(f"[crop] run_id {RUN!r} has no observations in this store.")

print("=" * 78)
print(f"1. TRACKLET CROP GEOMETRY -- run {RUN}")
print("=" * 78)
print("  A crop is resized to 384x128 whatever its source size, so 'upscale' is how")
print("  much this tracklet's median crop had to be STRETCHED. reid.quality accepts")
print("  min_height=64 px, i.e. a 6x stretch, on the same footing as a 300 px crop.")
print()
print(f"  {'tracklet':<20}{'obs':>5}{'h med':>8}{'h min':>7}{'h max':>7}"
      f"{'upscale':>9}{'blur med':>10}")
stats = {}
for key in sorted(per, key=lambda k: (str(k[0]), k[1])):
    r = per[key]
    if not r["h"]:
        continue
    h = np.asarray(r["h"])
    blur = np.asarray(r["blur"]) if r["blur"] else np.array([float("nan")])
    stats[key] = {"h_med": float(np.median(h)), "blur_med": float(np.nanmedian(blur)),
                  "n": len(r["vecs"])}
    print(f"  {key[0]}:{key[1]:<12}{len(r['vecs']):>5}{np.median(h):>8.0f}"
          f"{h.min():>7.0f}{h.max():>7.0f}{384.0 / max(np.median(h), 1):>8.1f}x"
          f"{np.nanmedian(blur):>10.1f}")

protos = {k: _prototype(r["vecs"]) for k, r in per.items() if r["vecs"]}
protos = {k: v for k, v in protos.items() if v is not None}

print()
print("=" * 78)
print("2. SAME-CAMERA PAIRS -- score beside the crop-scale mismatch")
print("=" * 78)
print("  'h ratio' is the larger median crop height over the smaller. If the LOW")
print("  scores concentrate at high h ratio, the bar is not the problem: the two")
print("  sides were embedded from very different amounts of real pixel detail.")
print()
print(f"  {'pair':<30}{'cosine':>8}{'h_a':>6}{'h_b':>6}{'h ratio':>9}"
      f"{'blur_a':>8}{'blur_b':>8}{'overlap':>9}")

rows = []
keys = sorted(protos, key=lambda k: (str(k[0]), k[1]))
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        if a[0] != b[0]:
            continue
        if a not in stats or b not in stats:
            continue
        s = float(protos[a] @ protos[b])
        ha, hb = stats[a]["h_med"], stats[b]["h_med"]
        ratio = max(ha, hb) / max(min(ha, hb), 1.0)
        ta, tb = per[a]["ts"], per[b]["ts"]
        if ta and tb:
            ov = min(max(ta), max(tb)) - max(min(ta), min(tb))
        else:
            ov = float("nan")
        rows.append((s, a, b, ha, hb, ratio, stats[a]["blur_med"],
                     stats[b]["blur_med"], ov))

for s, a, b, ha, hb, ratio, ba, bb, ov in sorted(rows):
    pair = f"{a[0]}:{a[1]}+{b[1]}"
    flag = "  CO-PRESENT" if ov == ov and ov > 0 else ""
    print(f"  {pair:<30}{s:>8.3f}{ha:>6.0f}{hb:>6.0f}{ratio:>9.2f}"
          f"{ba:>8.1f}{bb:>8.1f}{ov:>9.1f}{flag}")

print()
print("=" * 78)
print("3. THE TEST")
print("=" * 78)
disjoint = [r for r in rows if not (r[8] == r[8] and r[8] > 0)]
copresent = [r for r in rows if r[8] == r[8] and r[8] > 0]
print(f"  time-DISJOINT same-camera pairs (could be one person): {len(disjoint)}")
print(f"  CO-PRESENT   same-camera pairs (provably two people):  {len(copresent)}")


def _corr(rs):
    if len(rs) < 4:
        return None
    s = np.array([r[0] for r in rs])
    q = np.array([r[5] for r in rs])
    if s.std() == 0 or q.std() == 0:
        return None
    return float(np.corrcoef(s, q)[0, 1])


for name, rs in (("disjoint", disjoint), ("co-present", copresent)):
    c = _corr(rs)
    if c is None:
        print(f"  {name}: too few pairs to correlate")
        continue
    s = np.array([r[0] for r in rs])
    q = np.array([r[5] for r in rs])
    print(f"  {name}: n={len(rs)}  cosine mean={s.mean():.3f}  "
          f"h-ratio mean={q.mean():.2f}  corr(cosine, h_ratio)={c:+.3f}")
    hi = s[q > np.median(q)]
    lo = s[q <= np.median(q)]
    if len(hi) and len(lo):
        print(f"      pairs with MATCHED crop scale  (h ratio <= median): "
              f"mean cosine {lo.mean():.3f}")
        print(f"      pairs with MISMATCHED crop scale (h ratio >  median): "
              f"mean cosine {hi.mean():.3f}")
print()
print("  A strongly NEGATIVE correlation, and matched-scale pairs scoring well above")
print("  mismatched ones, means the defect is the INPUT GATE, not the similarity bar:")
print("  raise reid.quality.min_height / min_area so small crops never enter a")
print("  prototype. That is a pixels-on-target rule, which transfers across footage.")
print("  A correlation near zero means crop scale is NOT the driver and the")
print("  front/back appearance explanation stands.")
