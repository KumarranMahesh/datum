# Datum

**Single-camera scouting infrastructure for football.**

Datum ingests broadcast video, the same 1080p feed your television sees, and produces searchable per-player style embeddings derived from observed on-pitch behaviour. No multi-camera rig. No per-stadium calibration. No proprietary sensor mesh.

If you can stream the match, you can index it.

---

## Status

| Field | Value |
|---|---|
| Stage | Pre-alpha |
| Stable surface area | None. APIs will break without warning until `0.1.0`. |
| Production readiness | Do not. |
| What works today | Ingestion CLI scaffold, frame demux, detector adapter interface, pitch homography prototype. |
| What does not work yet | Player2Vec training, search API, identity continuity through occlusion. |
| Tested on | WSL2 / Ubuntu 22.04, Python 3.11, CUDA 12.x, RTX 40-series. |

This project ships `0.1` before promising `1.0`.

---

## What this is

A modular, hackable pipeline that takes broadcast video and emits, per player per match, a learned style embedding suitable for vector similarity search.

The target is a representation good enough that **"find me three players who play like prime Modrić but in the U-21 Argentine league"** returns a useful answer. Not a perfect one. A useful one.

## What this is not

| | |
|---|---|
| **Not** a score predictor | Datum does not predict match outcomes. |
| **Not** a betting tool | The codebase contains no odds ingestion or market data. |
| **Not** a free knockoff of Opta or StatsBomb | Datum is *complementary* to licensed event data. If you have a feed, Datum will use it and produce better results. |
| **Not** multi-camera | Multi-camera capture is solved by people with budgets this project does not have. Single-camera broadcast is where the leverage is. |
| **Not** a wrapper around a frontier LLM | The substance is computer vision, geometry, and representation learning. |

---

## Why this exists

Professional scouting tools gate this category at price points that exclude roughly 95% of the global football pyramid. Academy directors in Senegal, second-division clubs in Paraguay, university analytics programs: none of them have €50K/year for a tracking license. They do, however, have YouTube and a GPU.

Datum is the proposition that broadcast video plus modern CV is, in 2026, sufficient for a credible scouting pipeline. The claim is not StatsBomb quality. The claim is something materially better than "watching the tape and writing notes."

---

## Architecture

```
                broadcast video (mp4 / hls / rtmp)
                              │
                              ▼
                      ┌───────────────┐
                      │    ingest     │   demux · frame sampling · scene segmentation
                      └───────┬───────┘
                              ▼
                      ┌───────────────┐
                      │      cv       │   detection · tracking · pose · pitch homography
                      └───────┬───────┘
                              ▼
                      ┌───────────────┐
                      │    spatial    │   pixel → pitch coordinates · physical constraints
                      └───────┬───────┘
                              ▼
                      ┌───────────────┐
                      │    events     │   action recognition · possession segmentation
                      └───────┬───────┘
                              ▼
                      ┌───────────────┐
                      │   features    │   per-player rolling statistics in pitch frame
                      └───────┬───────┘
                              ▼
                      ┌───────────────┐
                      │     embed     │   Player2Vec encoder
                      └───────┬───────┘
                              ▼
                      ┌───────────────┐
                      │ index · search│   vector store · similarity · role-fit queries
                      └───────────────┘
```

Each block is a swappable module behind a stable interface. Detector adapters live behind `datum.cv.detect.Detector`. Embedding adapters live behind `datum.embed.Encoder`. Bring your own model. The contract is small and documented.

### Design posture

| Principle | What it means in practice |
|---|---|
| Systems-first, not model-first | Robust data generation and physical constraints carry more weight than parameter counts. A 30M-parameter model on clean, geometrically-correct features will beat a 3B-parameter model on noisy pixels. |
| Determinism by default | Every stage is a deterministic function of `(input artifacts, config)`. Re-runs produce bitwise-identical outputs. |
| Fail loud, fail early | 480p feeds are refused outright rather than processed into silent garbage. Confidence is logged alongside every output. |
| Stage isolation | No stage may read from another stage's internal state. Only declared artifacts. |
| The library is not the app | `src/datum/` is a library. The CLI and API are clients of it. Build your own client if you want to. |

---

## Repository structure

```
datum/
├── src/datum/         the library. Public surface lives here.
│   ├── ingest/        video demux, scene cuts, sampling
│   ├── cv/            detection, tracking, pose, pitch homography
│   │   ├── detect/
│   │   ├── track/
│   │   ├── pose/
│   │   └── pitch/
│   ├── spatial/       pixel→pitch transforms, physical constraints
│   ├── events/        action recognition, possession segmentation
│   ├── features/      per-player feature extraction
│   ├── embed/         Player2Vec training and inference
│   ├── index/         vector store adapters (faiss, qdrant, milvus)
│   ├── search/        similarity / role-fit query layer
│   ├── store/         metadata DB (postgres)
│   ├── api/           FastAPI service
│   ├── cli/           the `datum` command-line
│   ├── eval/          benchmarks, golden-match validation
│   └── utils/
├── configs/           YAML configs for pipelines, models, environments
├── data/              local data lake (gitignored by default)
├── docker/            service images and compose files
├── docs/              architecture, ADRs, contributor guides
├── notebooks/         exploration only. Not part of the library.
├── scripts/           one-shot ops (bootstrap, reindex, download samples)
├── tests/             unit · integration · golden
└── benchmarks/        throughput and accuracy regression suite
```

Anything under `notebooks/` is **not** part of the public surface. Compatibility is not preserved for code imported from a notebook.

---

## Quickstart

Two supported paths: native Windows (recommended for solo development on the kind of i9 / RTX 40-series laptop this project is built on) and WSL2 / Linux (recommended for contributors who already live there or who need Linux-only CV libraries that this project does not yet depend on).

Native Windows is the default. Switch to WSL2 when (and only when) a Linux-only dependency forces it.

### Prerequisites (both paths)

| Component | Requirement |
|---|---|
| Python | 3.11.x. Minor version is pinned; 3.12 is not yet validated. |
| GPU | CUDA 12.x, ≥ 8 GB VRAM for inference, ≥ 16 GB recommended for training. |
| Disk | ≥ 50 GB free on the drive holding `data/`. Broadcast video is large. |
| Docker | Optional. Required only for the index/search services. |
| `uv` | Installed automatically by the bootstrap script. |

### Path A. Native Windows (recommended)

1. Install Python 3.11 from python.org. Do **not** use the Microsoft Store build; its sandboxed file paths break things in subtle ways. `winget install Python.Python.3.11` is the cleanest one-liner.
2. Install `uv`: `winget install --id=astral-sh.uv -e` or `irm https://astral.sh/uv/install.ps1 | iex`.
3. Verify the GPU is visible: open PowerShell, run `nvidia-smi`. Driver 550 or newer and CUDA 12.4 or newer is the comfortable floor.
4. From the repo directory:

   ```powershell
   cd D:\path\to\datum
   .\scripts\bootstrap.ps1
   ```

   That script does the same five things `bootstrap_wsl.sh` does, with the Windows-correct equivalents. Output lands in the same place.

5. If `nvidia-smi` showed a healthy GPU but `torch.cuda.is_available()` is `False` after bootstrap, the CPU wheel was selected. Reinstall against the CUDA index:

   ```powershell
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

A note on paths. On Windows there is no `/mnt/` performance trap; native NTFS is fine. Keep `data/` on the same drive as the repo (typically D:) if disk pressure on C: is a concern.

### Path B. WSL2 / Linux

1. Open WSL2 (Ubuntu 22.04 recommended). Native Linux is identical from here.
2. Clone the repo into the WSL filesystem (e.g. `~/datum`), **not** `/mnt/c/...` or `/mnt/d/...`:

   ```bash
   cd ~
   git clone https://github.com/your-org/datum.git
   cd datum
   ./scripts/bootstrap_wsl.sh
   ```

3. The script installs `uv`, creates `.venv/`, syncs pinned deps, downloads the sample match, and runs the smoke test.

#### The `/mnt/` warning, made explicit

> **Do not put video data under `/mnt/c/` or any Windows-mounted drive when working from WSL2.** Frame decode drops from roughly 240 fps to roughly 9 fps just from crossing the filesystem boundary. The bootstrap script refuses to run from a `/mnt/` path for this reason.

If you must edit code from Windows-side VSCode against a WSL clone, Remote-WSL handles it correctly. But the data lives on the Linux side.

#### Line endings

The repo enforces LF via `.gitattributes`. On Windows-side git, set `core.autocrlf=input` so CRLF does not creep in via your editor.

---

## Running the pipeline

```bash
# end-to-end on the sample match
uv run datum pipeline run \
  --match data/samples/sample_match.mp4 \
  --config configs/pipelines/default.yaml

# step-by-step. Useful when debugging which stage emitted the garbage.
uv run datum ingest    --match <path>
uv run datum cv        --run <run-id>
uv run datum spatial   --run <run-id>
uv run datum events    --run <run-id>
uv run datum features  --run <run-id>
uv run datum embed     --run <run-id>
uv run datum index     --run <run-id>

# query
uv run datum search "players like Modrić, U-21, South American leagues only" --k 10
```

Every stage emits a manifest. If you hit a bug, attaching the manifest to the issue makes it much easier to reproduce on someone else's machine.

---

## Swapping components

The whole point of the layered architecture is that you can replace any block without touching the others. Example, your own detector:

```python
from datum.cv.detect import Detector, register
from datum.cv.detect.types import DetectionBatch, FrameBatch

@register("my-detector")
class MyDetector(Detector):
    def detect(self, frames: FrameBatch) -> DetectionBatch:
        # your model here
        ...
```

Then point the config at it:

```yaml
cv:
  detector:
    name: my-detector
    config:
      confidence_threshold: 0.4
```

The same pattern applies to:

| Layer | Interface | Provided implementations |
|---|---|---|
| Detector | `datum.cv.detect.Detector` | `yolov8`, `rt-detr`, `noop` (debug) |
| Tracker | `datum.cv.track.Tracker` | `bytetrack`, `botsort`, `noop` |
| Pose estimator | `datum.cv.pose.PoseEstimator` | `vitpose`, `noop` |
| Pitch homography | `datum.cv.pitch.PitchSolver` | `keypoint-based`, `feature-based` |
| Embedding encoder | `datum.embed.Encoder` | `player2vec-baseline` (stub) |
| Vector store | `datum.index.VectorStore` | `faiss`, `qdrant`, `milvus`, `inmem` |

See `docs/extensibility.md` for the full contract and the test suite you need to pass.

---

## Known limitations

The quiet parts, said out loud:

| Limitation | Status |
|---|---|
| Broadcast feeds below 720p produce unreliable pitch homography. | Datum detects and refuses them rather than emit silent garbage. |
| Single-camera tracking through heavy occlusion swaps identities. | Mitigated with re-ID embeddings. Not solved. |
| Broadcast cuts to crowd / bench / commentators are unusable. | The ingest stage segments them out. Expect 8 to 20% of total broadcast time to be discarded. |
| Embedding quality is bound by training data diversity. | A model pretrained on European broadcasts will be biased toward European broadcast conventions. Documented in `docs/bias.md`. |
| Lobbed passes and aerial duels frequently lose the ball above the frame. | Inherent to broadcast framing. These intervals are marked low-confidence. |
| GPU memory will spike at scene cuts where the detector re-initialises. | Configure `cv.detector.batch_size` down on smaller cards. |

If you find a failure mode that is not on this list, open an issue with the video timestamp and the run manifest. The list grows as failure modes surface.

---

## Roadmap

| Phase | Target | Status |
|---|---|---|
| `0.1` | Single-camera ingestion + detection + naïve tracking | in progress |
| `0.2` | Pitch homography + spatial transforms hardened | scaffolded |
| `0.3` | First Player2Vec checkpoint trained on the open broadcast corpus | not started |
| `0.4` | Vector search API + CLI feature parity | not started |
| `0.5` | Bias audit harness + public benchmark suite | not started |
| `0.6` | Optional licensed-data adapters (StatsBomb, Skillcorner free tiers) | not started |
| `1.0` | API stability and semver guarantees | far away |

Dates are deliberately absent. Open-source projects ship when they ship.

---

<!-- ## Contributing

Read `docs/contributing.md` first. The short version:

| Rule | Why |
|---|---|
| Bug reports beat feature requests. | A reproducer is worth a thousand vision decks. |
| New detectors, encoders, vector stores: behind existing interfaces. Don't fork the pipeline. | Every adapter must share the same contract. |
| Performance regressions > 5% on the benchmark suite block merge. | Speed is treated as a feature. |
| Please write PR descriptions in your own words. | A short summary you've written yourself helps reviewers follow what changed and why. |
| Add a test that fails without your change, then passes with it. | Standard. |

Code style is enforced in CI: `ruff format`, `ruff check`, `mypy --strict` on `src/datum/`.

--- -->

<!-- ## License

Apache-2.0. See `LICENSE`.

Training data is **not** part of this license. You are responsible for the legality of any video you ingest into your own instance. Licensing questions sit outside the scope of this project, so the right place to take them is the rights-holder or a lawyer, rather than the issue tracker.

--- -->

<!-- ## Citing

If you use Datum in academic work:

```
@software{datum2026,
  title   = {Datum: Open Single-Camera Scouting Infrastructure for Football},
  year    = {2026},
  url     = {https://github.com/your-org/datum}
}
```

This entry will be replaced with a proper publication once the embedding model is validated.

--- -->

This is going to be a long project. That is fine.
