# Multi-Camera Person Re-Identification

Detect and track people across multiple cameras, embed their appearance, and
assign a **reid id** that stays the same for one real person **across cameras**
and **across re-appearances**. Two input modes, one identity result:

- **Live RTSP** (`--mode live`) — reads N streams in real time, never recording
  the raw feed; on stop it runs the offline reconcile and writes corrected
  `output_<cam>.mp4`. Real-time targets a GPU server.
- **Video files** (default / `--mode batch`) — the same detect → track → embed →
  reconcile → re-render flow on recorded footage. Runs on CPU.

The pipeline is fully self-contained — it builds and owns its own Qdrant gallery;
there is no separate registration step or external service. Both modes settle
cross-camera identity with the **same offline reconcile** and produce annotated
videos where one person carries one reid id/colour in every camera.

> **How it works internally:** see **[ARCHITECTURE.md](ARCHITECTURE.md)** for the
> full data flow, every component, the concurrency model, and design rationale.

---

## Quickstart

Start to finish (Linux; GPU auto-detected). Details for each step are in the
numbered sections below.

```bash
# 1. clone
git clone https://github.com/seif744/PersonReID.git && cd PersonReID

# 2. OpenCV system libs (Linux/WSL/Docker only; skip on macOS/Windows) -- see §2
sudo apt-get update && sudo apt-get install -y \
  libgl1 libglib2.0-0 libsm6 libice6 libxext6 libxrender1

# 3. Python 3.10 env + deps (torch pin is CUDA-enabled; falls back to CPU) -- §2
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 4. start the Qdrant gallery (Docker) -- §4
docker compose up -d

# 5. run it -- §6
#    (a) video files (CPU is fine):
python main.py --videos /path/cam_a.mp4 /path/cam_b.mp4
#    (b) OR live RTSP (GPU; Ctrl-C to reconcile + write the final videos):
python main.py --mode live --videos "rtsp://USER:PASS@HOST:554/ch01/0" "rtsp://..."
```

Output: `output_<cam>.mp4` per camera (same person = same reid id/colour
everywhere) + a console run summary. The ReID weight is already in the repo;
`yolo11n.pt` auto-downloads on first run. Sanity-check your install first with
the tests in [§2 "Verify the install"](#2-install-the-python-environment).

---

## How it flows

```mermaid
flowchart TD
    CFG(["config.yaml / CLI"]) --> A1 & B1

    subgraph LIVE[" LIVE — one thread per camera, running at the same time"]
        direction LR
        subgraph CAM_A["Camera A"]
            direction TB
            A1(["Video frame"]) --> A2["Detect + Track\n(YOLO11n + ByteTrack)"]
            A2 --> A3["Embed crop\n(OSNet, 512-d)"]
            A3 --> A4["Re-rank + verify\ncandidates -> assign\na reid id"]
        end
        subgraph CAM_B["Camera B"]
            direction TB
            B1(["Video frame"]) --> B2["Detect + Track\n(YOLO11n + ByteTrack)"]
            B2 --> B3["Embed crop\n(OSNet, 512-d)"]
            B3 --> B4["Re-rank + verify\ncandidates -> assign\na reid id"]
        end
    end

    A4 --> QD
    B4 --> QD

    QD[("Qdrant\nown gallery\n(all cameras write here)")]

    QD --> RECON

    subgraph FINAL["OFFLINE — runs ONCE, after every camera finishes"]
        direction TB
        RECON["Reconcile tracklets\nlink the same person\nacross cameras"]
        RECON --> RENDER["Re-render videos\nusing the FINAL\nreid ids"]
        RECON --> SUMMARY["Print run summary\n(console only)"]
    end

    RENDER --> OUT[(" output_camA.mp4\n output_camB.mp4\nsame person = same REID\nin BOTH videos")]

    style CFG fill:#e8eef7,stroke:#4a6fa5,color:#1a1a1a
    style QD fill:#fdf3d7,stroke:#b8860b,color:#1a1a1a,stroke-width:2px
    style OUT fill:#e3f5e6,stroke:#2e7d32,color:#1a1a1a,stroke-width:2px
    style LIVE fill:#f7f9fc,stroke:#4a6fa5
    style FINAL fill:#f2f8f3,stroke:#2e7d32
    style CAM_A fill:#ffffff,stroke:#9fb3cf
    style CAM_B fill:#ffffff,stroke:#9fb3cf
```

Each camera runs the **top** section live, on its own thread, against ONE
shared Qdrant gallery that this pipeline itself writes to and reads from —
no other service is involved. Only after **every** camera finishes does the
**bottom** section run — once: it links the same person across cameras, then
re-renders the annotated videos with those final IDs, so one person carries
the same `REID n` in every camera's output. `global_id` is still written for
compatibility, but `reid_id` is the label the pipeline presents. Full per-stage detail:
**[ARCHITECTURE.md §2](ARCHITECTURE.md)**.

---

## Table of contents
- [Quickstart](#quickstart)
1. [Prerequisites](#1-prerequisites)
2. [Install the Python environment](#2-install-the-python-environment)
3. [Model weights](#3-model-weights)
4. [Connect to Qdrant](#4-connect-to-qdrant)
5. [Configure your run](#5-configure-your-run)
6. [Run the pipeline](#6-run-the-pipeline)
7. [Understand the output](#7-understand-the-output)
8. [Troubleshooting](#8-troubleshooting)
9. [Project layout](#9-project-layout)
10. [Known limitations & roadmap](#10-known-limitations--roadmap)
11. [Models & credits](#11-models--credits)
12. [License](#12-license)

---

## 1. Prerequisites

- **Python 3.10** (the project is verified on 3.10.12).
- **Docker** + **Docker Compose** — used to run the Qdrant server this
  pipeline writes its own gallery to. Install Docker Desktop (Windows/macOS)
  or Docker Engine (Linux). Verify:
  ```bash
  docker --version
  docker compose version
  ```
- **One or more video files** (e.g. `.avi`, `.mp4`) **or RTSP/HTTP stream URLs**.
- **CPU** is fine for file inputs, development, and the (synthetic) logic tests.
  **Live RTSP in real time needs a GPU** — the live pipeline was developed and
  validated on an **NVIDIA A100** server (it auto-detects the device). CPU drops
  too many frames to keep up with live streams or to judge live reid quality;
  never benchmark the live path on CPU.

---

## 2. Install the Python environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins the exact verified versions (ultralytics, torch,
torchvision, torchreid, qdrant-client, numpy, opencv-python, PyYAML).

> **CPU vs GPU torch:** the pinned `torch==2.12.1` default Linux wheel is
> **CUDA-enabled** — it uses an NVIDIA GPU when present (live pipeline
> `device: auto`) and falls back to CPU otherwise, so the *same* install works on
> the CPU dev box **and** the A100 server (the server just needs a compatible
> NVIDIA driver — check with `nvidia-smi`). For a specific CUDA version, install
> `torch`/`torchvision` from the matching
> [PyTorch wheel index](https://pytorch.org/get-started/locally/) *before*
> `pip install -r requirements.txt`.

### System libraries (Linux / WSL / Docker)

`opencv-python` links a few shared libraries that headless Linux images (and many
Docker bases / WSL) don't ship. If `import cv2` fails with
`ImportError: libGL.so.1: cannot open shared object file` (or a `libgthread` /
Qt `xcb` error), install them once:

```bash
sudo apt-get update && sudo apt-get install -y \
  libgl1 libglib2.0-0 libsm6 libice6 libxext6 libxrender1
```

Not needed on macOS or Windows. (These are also required for the live OpenCV
windows if you ever set `display.show_window: true`; the default is headless.)

### Verify the install

Run the deterministic logic tests — synthetic embeddings, **no GPU/model/video
needed** — to confirm the identity engine + offline-reconcile wiring:

```bash
for t in tests/live/test_*.py; do PYTHONPATH=src:tests/live python "$t"; done
```

Each script prints `OK` on success.

---

## 3. Model weights

Two model files are needed:

| File | How to get it | In a fresh clone? |
|---|---|---|
| `yolo11n.pt` (detector) | **Auto-downloaded** by ultralytics on first run | No (gitignored; fetched automatically) |
| `yolo11n-pose.pt` (pose ensemble) | **Auto-downloaded** by ultralytics — but only on a **file-batch** run (the live path disables the pose ensemble, so live runs never fetch it) | No (gitignored; fetched automatically) |
| `src/reid/weights/osnet_ain_x1_0.pth` (ReID, default) | Committed to the repo | Yes — already present |

The default ReID checkpoint is **OSNet-AIN x1_0** (set by `reid.weights` in
`config.yaml`), the same torchreid OSNet family with Adaptive Instance
Normalization added for better generalization to unseen camera domains. It
replaced a plain `osnet_x1_0` (MSMT17) checkpoint whose embeddings weren't
discriminative enough on this project's out-of-domain CCTV footage
(same-person cross-camera cosine ~0.72 vs. different-person ~0.55 — too
close). On this project's footage, the AIN swap measurably widens that gap
(see [ARCHITECTURE.md §6](ARCHITECTURE.md)). If you ever need to re-fetch it:
it's the multi-source domain-generalization checkpoint (trained on
DukeMTMC-reID + Market1501 + CUHK03, evaluated on MSMT17) from the official
`deep-person-reid` MODEL_ZOO (`osnet_ain_x1_0`) — download via `gdown` from the
Google Drive link on that page and place it at `src/reid/weights/`.

To compare against an earlier checkpoint (e.g. `osnet_x1_0_msmt17` or
`osnet_x1_0` Market1501), fetch it the same way and point `reid.weights` at it —
but OSNet-AIN is the recommended default given the measured separation above.

---

## 4. Connect to Qdrant

The pipeline needs somewhere to store its own gallery of embeddings + global
ids. It both writes to and reads from this store — nothing else does.

### 4a. Start the Qdrant server (recommended)

A `docker-compose.yml` is included. From the project root:

```bash
docker compose up -d          # start Qdrant in the background
docker compose logs -f        # (optional) watch logs; Ctrl-C to stop watching
```

This launches `qdrant/qdrant:latest` and exposes:
- **REST API + dashboard:** http://localhost:6333/dashboard
- **gRPC (optional):** `localhost:6334`

Data persists in `./qdrant_storage/` (gitignored), so it survives restarts.
If you already run a Qdrant server elsewhere, point the pipeline at that endpoint
instead (see 4b).

Verify it's up:
```bash
curl http://localhost:6333/readyz        # -> "all shards are ready"
```

Stop it later with `docker compose down` (your data is kept).

### 4b. Tell the pipeline where Qdrant is

Backend precedence: **`QDRANT_URL` env var → `store.url` in config.yaml →
`store.path`** (a local embedded folder, single-process only).

Out of the box, `config.yaml` already has:
```yaml
store:
  enabled: true
  url: http://localhost:6333
```
so no extra step is needed for local Docker. If you prefer env-based config,
create a `.env` file in the project root:
```
QDRANT_URL=http://localhost:6333
```

For **Qdrant Cloud** instead of local Docker, set both (the API key is a secret —
`.env` is gitignored, never commit it):
```
QDRANT_URL=https://<your-cluster>.cloud.qdrant.io:6333
QDRANT_API_KEY=<your-key>
```

### 4c. Embedded mode (no Docker, single process only)

```yaml
store:
  enabled: true
  url:                 # leave blank
  path: qdrant_data    # local folder, created automatically
```
This locks the folder to one process — fine for a quick local test, not for
multiple concurrent runs.

---

## 5. Configure your run

> **You must provide your own video files.** Sample footage is **not** shipped in
> the repo (videos are gitignored). Either drop your files in and point
> `source.videos` in `config.yaml` at them, or pass them on the command line with
> `--videos` / `--videos-dir` (see section 6). The default `config.yaml` lists two
> example filenames purely as a template — edit them to your paths.

Edit [config.yaml](config.yaml). The key sections:

```yaml
source:
  videos:                        # one entry per camera
    - name: cam_219
      path: your_camera_1.avi
    - name: cam_224
      path: your_camera_2.avi
  max_frames: 0                  # 0 = whole video; N = stop early (quick test)
  resize_width: 0                # 0 = native; e.g. 1280 = faster on CPU

detector:
  model: yolo11n.pt              # auto-downloaded
  confidence_threshold: 0.4

tracker:
  enabled: true
  config: bytetrack.yaml

reid:
  enabled: true
  weights: src/reid/weights/osnet_ain_x1_0.pth
  device: cpu                    # "cuda" if you have a GPU (file-batch path)
  interval: 10                   # re-embed a track at most every N frames

identity:
  enabled: true
  threshold: 0.63                # plain-path cosine acceptance (used only
                                  # when verification.enabled is false) --
                                  # calibrated for the osnet_ain_x1_0 checkpoint
  min_score_gap: 0.03
  rerank:                        # ADR-002 Upgrade 1: camera-aware re-ranking
    enabled: true
    k1: 8
    cross_camera_k1_boost: 4
    lambda_: 0.7
  verification:                  # ADR-002 Upgrade 2: scored verification layer
    enabled: true
    accept_threshold: 0.55
    min_prob_gap: 0.05
    log_path: logs/verification_decisions.jsonl
  reconcile:
    enabled: true
    threshold: 0.63               # cross-camera merge bar -- independent of the
                                   # verifier above; recalibrate both together
    same_camera_threshold: 0.90
    min_tracklet_observations: 3
    require_reciprocal_best: true

display:
  show_window: false             # true = live OpenCV windows; false = headless
  save_annotated: true           # write output_<camera>.mp4

live:                            # real-time streaming pipeline (--mode live)
  run:
    mode: auto                   # live | auto (live for stream URLs) | batch
    device: auto                 # auto picks GPU when present, else CPU
  reconcile:
    enabled: true                # on stop, run the offline reconcile + re-render
    sample_stride: 1             # store every Nth fresh embedding per track
```

See [ARCHITECTURE.md §7](ARCHITECTURE.md) for what every knob does.

---

## 6. Run the pipeline

With the venv active and Qdrant reachable:

```bash
python main.py                       # uses the videos listed in config.yaml
```

### Run on your OWN videos (dynamic input — no config edits)

Point the pipeline at any videos from the command line; this **overrides**
`config.yaml`, so you never have to hardcode paths:

```bash
# one or more explicit files (camera name = the file's base name):
python main.py --videos /path/cam_a.mp4 /path/cam_b.mp4

# every video in a folder (.mp4/.avi/.mov/.mkv/...):
python main.py --videos-dir /path/to/footage

```

Precedence: `--videos` > `--videos-dir` > `config.yaml source.videos`. Missing
files fail fast with a clear message; duplicate names are made unique
automatically.

### Run on LIVE RTSP streams (`--mode live`)

Pass one URL per camera. The raw feed is **never recorded**; each stream is
processed in real time, and **Ctrl-C** triggers the offline reconcile that
settles correct cross-camera ids and writes the final `output_<cam>.mp4`:

```bash
python main.py --mode live \
  --videos \
  "rtsp://USER:PASS@192.168.1.224:554/ch01/0" \
  "rtsp://USER:PASS@192.168.1.219:554/ch01/0"
```

- **N cameras = more URLs.** Camera names auto-derive from the last IP octet
  (`cam_224`, `cam_219`, …). URL-encode credential specials (`@` → `%40`).
- On Ctrl-C the console prints *"Please wait a moment while we render the final
  outputs…"*, reconciles, and re-renders — let it finish. Qdrant must be
  reachable (`store.url`); otherwise it logs and falls back to the live output.
- `run.mode: auto` (the default) routes any stream URL to this same live path,
  so `--mode live` is only needed to force it for a file source.

`--reset` wipes the store's collection first, so a run starts from an empty
gallery. **Destructive** — deletes every stored embedding/identity:

```bash
python main.py --reset --videos-dir /path/to/footage
```

Headless (`display.show_window: false`) runs to completion and prints a summary.
With windows enabled, press `q` in a window to stop early.

---

## 7. Understand the output

A run produces:

| Artifact | Location | What it is |
|---|---|---|
| Annotated videos | `output_<camera>.mp4` | source video with boxes + `REID n  IDk` labels drawn |
| Gallery | Qdrant (`store.url` / `store.path`) | every observation + assigned reid id (plus compatibility `global_id`) — this pipeline's own data |
| Verification decision log | `logs/verification_decisions.jsonl` | one line per accept/reject decision, with its full feature vector — for calibrating or eventually training the verifier |

That's it — no crop images are written. On-disk crop saving is disabled in the
code (`crop_saver.py` writes nothing); the embedder makes its own in-memory
crops, so the ReID path never needs files on disk. `crops.save: true` only builds
an in-memory (discarded) crop helper — it does not produce files.

The console prints a **RUN SUMMARY** at the end, e.g.:
```
Store: 516 observations -> 11 distinct people (reid_ids)
Cross-camera people: 1
  REID 1: cam_219 (track 0004 + 0025) + cam_224 (track 0001)
```
- **distinct people** = number of reid IDs after reconciliation.
- **Cross-camera people** = reid IDs seen in more than one camera (the product's
  core result).

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ImportError: libGL.so.1` (or `libgthread` / Qt `xcb`) on `import cv2` | Missing OpenCV system libs on Linux/WSL/Docker. Install them (see §2 "System libraries"). |
| `Connection refused` | Qdrant isn't running. `docker compose up -d`, then `curl http://localhost:6333/readyz`. |
| Live run uses CPU / is very slow on a GPU box | torch can't see the GPU. Check `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`; fix the NVIDIA driver or install a matching CUDA torch (§2). |
| `Unexpected checkpoint keys dropped` | Wrong/corrupt ReID weights. Re-fetch `osnet_ain_x1_0.pth` to `src/reid/weights/`. |
| Hangs on first run for a while | `yolo11n.pt` is downloading; subsequent runs are fast. |
| Very slow on CPU | Set `source.resize_width: 1280` and/or `source.max_frames` for tests. |
| "database is locked" | Embedded mode (`store.path`) is single-process. Use the shared Docker server (`store.url`), or stop other runs using the same folder. |
| No windows appear | `display.show_window: false` (headless). Set `true` for live windows. |
| Same person shows up as two different reid ids | Expected on out-of-domain footage when cosine similarity is borderline — see limitations below. Check `logs/verification_decisions.jsonl` for that decision's actual scores. |
| Two different people merge into one reid id | The rarer, worse failure. If it happens often, tighten `identity.verification.accept_threshold` / `min_prob_gap` (or `identity.threshold` / `min_score_gap` if verification is disabled). |

---

## 9. Project layout

```
main.py                     entry point + routing + FILE-BATCH orchestration
config.yaml                 all runtime configuration
requirements.txt            pinned dependencies
docker-compose.yml          Qdrant server
ARCHITECTURE.md             deep-dive: data flow, components, design
src/
  video_source.py           frame decoding (files + streams)
  detector.py                YOLO11 + ByteTrack, Detection, crop_person
  crop_saver.py               per-track crop helper (in-memory; disk saving disabled)
  drawing.py                  boxes / HUD overlay
  reid/
    extractor.py              crop -> 512-d L2-normalized embedding (OSNet)
    service.py                 TrackEmbedder: throttle, cache, quality + occlusion gates
    weights/                    ReID model checkpoint (osnet_ain_x1_0.pth)
  database/
    store.py                   Qdrant wrapper (PersonVectorStore) -- this pipeline's own gallery
  identity/
    service.py                  global-ID assignment (candidate -> re-rank -> verify -> decide -> commit)
    reranking.py                 ADR-002 Upgrade 1: camera-aware k-reciprocal + Jaccard re-ranking
    verifier.py                  ADR-002 Upgrade 2: scored verification layer + decision logging
    reconcile.py                 offline cross-camera reconciliation (used by BOTH paths)
    DESIGN.md                    why the layers are separated
  live/                       REAL-TIME streaming pipeline (--mode live)
    pipeline.py                orchestrator: wire stages, run, shutdown + offline reconcile
    capture.py / decode_backend.py / capabilities.py   per-camera capture + device probe
    scheduler.py / inference.py                         freshness batching + detect/track/embed
    identity_stage.py / identity_engine.py / priority.py   online ids + fair queue + store persistence
    topology.py                 fail-open cross-camera physical-impossibility veto
    render.py / writer.py / queues.py / frame.py        draw/encode + load-shedding primitives
tests/live/                  deterministic logic tests (synthetic; no GPU/models needed)
```

Generated at runtime (gitignored): `output_*.mp4`, `logs/`, `qdrant_storage/`,
`qdrant_data/`, `yolo11n.pt`.

---

## 10. Known limitations & roadmap

The pipeline is correct; the embedding model is the ceiling on this domain —
see **[ARCHITECTURE.md §6](ARCHITECTURE.md)** for the measured same-vs-different
score overlap and why no threshold (or hand-tuned heuristic on top of it) can
perfectly resolve it.

ADR-002's P0 items (camera-aware re-ranking, a scored verification layer) are
implemented; the following are deferred and documented but not built:
- Prototype confidence/variance with adaptive per-identity thresholds.
- A camera transition graph rejecting physically-impossible transitions.
- A MetaBIN training pass to reduce the underlying domain gap.
- Training a real classifier from `logs/verification_decisions.jsonl` once
  enough runs accumulate, to replace the current hand-set verifier weights.

---

## 11. Models & credits

This pipeline is assembled from pretrained models and open-source libraries — no
model is trained or fine-tuned here. Each component and its role:

| Model / component | Role in the pipeline | Source / library | Reference |
|---|---|---|---|
| **YOLO11n** (`yolo11n.pt`) | Person **detection** every frame (COCO class 0), the entry point for both paths. | [Ultralytics](https://github.com/ultralytics/ultralytics) (COCO-pretrained) | Ultralytics YOLO11 |
| **ByteTrack** (`bytetrack.yaml`) | Multi-object **tracking** — stable per-camera `track_id`s frame-to-frame; the tracklet is the unit of identity evidence. | Ships with Ultralytics | Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022 |
| **YOLO11n-pose** (`yolo11n-pose.pt`) | Optional **pose ensemble** that splits a tracker box merging two overlapping people into one. File-batch only; off in live (`inference.pose_ensemble: false`) for throughput. | [Ultralytics](https://github.com/ultralytics/ultralytics) | Ultralytics YOLO11-pose |
| **OSNet-AIN x1_0** (`osnet_ain_x1_0.pth`) | **Appearance embedding** — one crop → 512-d L2-normalized vector; the cosine similarity that drives re-ranking, verification, and reconciliation. AIN (Adaptive Instance Normalization) adds cross-domain generalization. Multi-source checkpoint (DukeMTMC-reID + Market1501 + CUHK03, eval MSMT17). | [torchreid / deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) | Zhou et al., *Omni-Scale Feature Learning for Person Re-ID*, ICCV 2019; *Learning Generalisable Omni-Scale Representations…*, IEEE TPAMI 2021 |
| **Qdrant** | **Vector store** (not a model) — this pipeline's own gallery: stores every observation embedding + payload and serves nearest-neighbour search. | [Qdrant](https://github.com/qdrant/qdrant) | — |

**Libraries:** [torchreid](https://github.com/KaiyangZhou/deep-person-reid)
(Zhou & Xiang, *Torchreid: A Library for Deep Learning Person Re-ID in Pytorch*,
2019) provides the OSNet backbone, preprocessing, and feature utilities; PyTorch;
OpenCV; NumPy; PyYAML.

**Dependency licenses:** Ultralytics YOLO11 is **AGPL-3.0**, torchreid is
**MIT**, Qdrant is **Apache-2.0**. Because this project builds on AGPL-3.0 code
(Ultralytics), the project as a whole is distributed under **AGPL-3.0** (see
§12). Swapping to a non-AGPL detector would be a code change (not just
`detector.model`), since `src/detector.py` is built on the `ultralytics` API.

---

## 12. License

This project is licensed under the **GNU Affero General Public License v3.0
(AGPL-3.0)** — full text in [LICENSE](LICENSE). AGPL applies because the pipeline
combines with **Ultralytics YOLO11 (AGPL-3.0)**; a combined/derivative work must
carry the same license.

```
Copyright (C) 2026 Seifer Mathias

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version. This program is distributed WITHOUT ANY WARRANTY; see the GNU
AGPL v3.0 for details.
```

What this means in practice:
- **Open source + public source = compliant**, and **no Ultralytics Enterprise
  license is needed** for this use.
- Anyone who redistributes this (modified or not), **or offers it to users over a
  network** (AGPL §13, the "Remote Network Interaction" clause), must make the
  corresponding **source available under AGPL-3.0**.
- A **closed-source / proprietary** use would instead require an Ultralytics
  Enterprise license (or removing the AGPL dependency).

> Dependencies keep their own licenses (§11); AGPL-3.0 covers this project's own
> code.
