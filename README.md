# Datum

**Single-camera scouting infrastructure for football.**

Datum ingests broadcast video — the same 1080p feed your television sees — and produces searchable per-player style embeddings derived from observed on-pitch behaviour. No multi-camera rig. No per-stadium calibration. No proprietary sensor mesh.

If you can stream the match, you can index it.

This is the open-source baseline for what professional scouting platforms cost six figures a year to use.

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

We ship `0.1` before we promise `1.0`.

---

## What this is

A modular, hackable pipeline that takes broadcast video and emits, per player per match, a learned style embedding suitable for vector similarity search.

The target is a representation good enough that **"find me three players who play like prime Modrić but in the U-21 Argentine league"** returns a useful answer. Not a perfect one. A useful one.

## What this is not

| | |
|---|---|
| **Not** a score predictor | We do not predict match outcomes. |
| **Not** a betting tool | The codebase contains no odds ingestion. Build that elsewhere. |
| **Not** a free knockoff of Opta or StatsBomb | We are *complementary* to licensed event data. If you have a feed, Datum will use it and produce better results. |
| **Not** multi-camera | Multi-camera capture is solved by people with more budget than us. Single-camera broadcast is where the leverage is. |
| **Not** a wrapper around a frontier LLM | This is computer vision, geometry, and representation learning. There is no chatbot in here. |

---

## Why this exists

Professional scouting tools gate this category at price points that exclude roughly 95% of the global football pyramid. Academy directors in Senegal, second-division clubs in Paraguay, university analytics programs — none of them have €50K/year for a tracking license. They do, however, have YouTube and a GPU.

Datum is the proposition that broadcast video plus modern CV is, in 2026, sufficient for a credible scouting pipeline. We do not promise StatsBomb quality. We promise something materially better than "watching the tape and writing notes."

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

Each block is a swappable module behind a stable interface. Detector adapters live behind `datum.cv.detect.Detector`. Embedding adapters live behind `datum.embed.Encoder`. Bring your own model — the contract is small and documented.

### Design posture

| Principle | What it means in practice |
|---|---|
| Systems-first, not model-first | Robust data generation and physical constraints carry more weight than parameter counts. A 30M-parameter model on clean, geometrically-correct features will beat a 3B-parameter model on noisy pixels. |
| Determinism by default | Every stage is a deterministic function of `(input artifacts, config)`. Re-runs produce bitwise-identical outputs. |
| Fail loud, fail early | We refuse 480p feeds rather than emit silent garbage. Confidence is logged alongside every output. |
| Stage isolation | No stage may read from another stage's internal state. Only declared artifacts. |
| The library is not the app | `src/datum/` is a library. The CLI and API are clients of it. Build your own client if you want to. |

---

## Repository structure

```
datum/
├── src/datum/         the library — public surface lives here
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
├── notebooks/         exploration only — not part of the library
├── scripts/           one-shot ops (bootstrap, reindex, download samples)
├── tests/             unit · integration · golden
└── benchmarks/        throughput and accuracy regression suite
```

Anything under `notebooks/` is **not** part of the public surface. We will not preserve compatibility for code you imported from a notebook.

---

## Quickstart (WSL2)

The primary supported environment is WSL2 / Ubuntu 22.04 on Windows 11 with an Nvidia GPU. Native Linux works. macOS works for the non-CV bits. Native Windows is not supported and will not be.

### Prerequisites

| Component | Requirement |
|---|---|
| OS | WSL2 / Ubuntu 22.04 (or native Linux) |
| Python | 3.11.x — we pin minor versions; 3.12 is not yet validated |
| GPU | CUDA 12.x, ≥ 8 GB VRAM for inference, ≥ 16 GB recommended for training |
| Disk | ≥ 50 GB free in the WSL filesystem. Broadcast video is large. |
| Docker | Optional, required only for the index/search services |
| `uv` | Installed automatically by the bootstrap script |

### Setup

```bash
git clone https://github.com/your-org/datum.git
cd datum
./scripts/bootstrap_wsl.sh
```

That script does the following:

| Step | Action |
|---|---|
| 1 | Installs `uv` if missing |
| 2 | Creates `.venv/` via `uv venv` |
| 3 | Installs pinned dependencies from `pyproject.toml` |
| 4 | Downloads a small sample match (1.2 GB) into `data/samples/` |
| 5 | Runs the smoke test against the first 60 seconds |

If it finishes clean you'll see `bootstrap ok` and a path to the smoke-test artifacts. If it does not, read the last 40 lines of the log before opening an issue.

### About WSL2 paths — read this

> **Do not put video data under `/mnt/c/` or any Windows-mounted drive.** Frame decode drops from ~240 fps to ~9 fps just from crossing the filesystem boundary. We have measured it. It is not a typo.

Keep `data/` inside the WSL filesystem (`~/datum/data` or similar). The bootstrap script enforces this and will refuse to run from a `/mnt/` path. If you must edit code from Windows-side VSCode, that's fine — Remote-WSL handles it correctly. But the data lives on the Linux side. Always.

A second WSL2 nuisance: line endings. The repo enforces LF via `.gitattributes`. If you committed CRLF, CI will reject the PR. Fix your editor, not ours.

---

## Running the pipeline

```bash
# end-to-end on the sample match
uv run datum pipeline run \
  --match data/samples/sample_match.mp4 \
  --config configs/pipelines/default.yaml

# step-by-step — useful when debugging which stage emitted the garbage
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

Every stage emits a manifest. If you find a bug, attach the manifest in the issue. Without it, we cannot reproduce, and we will close the issue.

---

## Swapping components

The whole point of the layered architecture is that you can replace any block without touching the others. Example — your own detector:

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

We say the quiet parts out loud.

| Limitation | Status |
|---|---|
| Broadcast feeds below 720p produce unreliable pitch homography. | We detect and refuse them rather than emit silent garbage. |
| Single-camera tracking through heavy occlusion swaps identities. | Mitigated with re-ID embeddings. Not solved. |
| Broadcast cuts to crowd / bench / commentators are unusable. | The ingest stage segments them out. Expect 8–20% of total broadcast time to be discarded. |
| Embedding quality is bound by training data diversity. | A model pretrained on European broadcasts will be biased toward European broadcast conventions. Documented in `docs/bias.md`. |
| Lobbed passes and aerial duels frequently lose the ball above the frame. | Inherent to broadcast framing. We mark these intervals as low-confidence. |
| GPU memory will spike at scene cuts where the detector re-initialises. | Configure `cv.detector.batch_size` down on smaller cards. |

If you find a failure mode that's not on this list, open an issue with the video timestamp and the run manifest. We add to the list as we find them.

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

## Contributing

Read `docs/contributing.md` first. The short version:

| Rule | Why |
|---|---|
| Bug reports beat feature requests. | A reproducer is worth a thousand vision decks. |
| New detectors, encoders, vector stores: behind existing interfaces. Don't fork the pipeline. | We need every adapter to share the same contract. |
| Performance regressions > 5% on the benchmark suite block merge. | We treat speed as a feature. |
| We do not accept generated PR descriptions. | If you can't summarise your own change, neither can we. |
| Add a test that fails without your change, then passes with it. | Standard. |

Code style is enforced in CI: `ruff format`, `ruff check`, `mypy --strict` on `src/datum/`.

---

## License

Apache-2.0. See `LICENSE`.

Training data is **not** part of this license. You are responsible for the legality of the video you ingest into your own instance. Don't email us about it. Don't open issues about it. We are an engineering project, not a clearinghouse.

---

## Citing

If you use Datum in academic work:

```
@software{datum2026,
  title   = {Datum: Open Single-Camera Scouting Infrastructure for Football},
  year    = {2026},
  url     = {https://github.com/your-org/datum}
}
```

We will replace this entry with a proper publication once the embedding model is validated.

---

This is going to be a long project. We are okay with that.
