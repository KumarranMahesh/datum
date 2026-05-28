# Contributing to Datum

The README has the short version. This is the long one.

---

## Before you open anything

| | |
|---|---|
| Found a bug? | Open an issue with a video timestamp, a run manifest, and the smallest reproducer you can manage. |
| Want a feature? | Open a discussion first. We close feature PRs that arrive without one. |
| New adapter (detector, tracker, encoder, vector store)? | Skip the discussion. Open a PR against the existing interface. |
| Refactor? | Open an issue describing what's wrong with the current code. We don't merge taste-based refactors. |

---

## The contract for new adapters

Every swappable layer has an `ABC` somewhere under `src/datum/`. Your PR must:

1. Implement the interface fully. No `NotImplementedError` stubs in the merged code.
2. Register via the layer's `@register("your-name")` decorator.
3. Add a unit test under `tests/unit/<layer>/test_<your_name>.py` that runs without GPU.
4. Add an integration test under `tests/integration/` if your adapter changes wire-level behaviour.
5. Document the adapter's expected inputs, outputs, and known failure modes in `docs/adapters/<layer>/<your-name>.md`.

If your adapter needs a heavyweight optional dependency (e.g. `faiss-gpu`, `onnxruntime-gpu`), add it as an extra in `pyproject.toml`. Do not add it to the base `dependencies`.

---

## Style and CI

| Tool | What it checks | Run locally |
|---|---|---|
| `ruff format` | Formatting | `make fmt` |
| `ruff check` | Lint, imports, common bugs | `make lint` |
| `mypy --strict` | Types on `src/datum/` | `make typecheck` |
| `pytest` | Unit + integration | `make test` |
| Benchmark suite | Throughput regression | `make bench` |

CI runs every one. A red CI run will not be merged. Don't ask.

---

## Performance

We treat speed as a feature. The benchmark suite tracks ingestion fps, end-to-end pipeline wall-clock, and per-stage GPU memory.

| Regression class | Policy |
|---|---|
| < 2% on any tracked metric | Acceptable. Mention it in the PR. |
| 2–5% | Requires justification. Reviewer's call. |
| > 5% | Blocks merge. Either fix it or split the PR. |

The exception is correctness fixes that happen to cost performance. Those are negotiated case by case.

---

## Commits and PRs

| | |
|---|---|
| Commit message format | `<area>: <imperative summary>` — e.g. `cv.detect: handle 1080i fields correctly` |
| PR title | Same convention. The title is what lands on `main` if we squash. |
| PR description | Write it yourself. If we detect generated text, we will ask you to rewrite. |
| Linked issue | Required for anything other than typo / docs PRs. |

Squash on merge. We don't keep noisy histories.

---

## What we will not accept

| | |
|---|---|
| Generated PR descriptions or commit messages. | If you can't summarise your change, neither can we. |
| Detectors / encoders behind paid APIs without an open-source fallback. | OSS-first. Paid adapters live in separate repos. |
| New top-level directories without an ADR. | The architecture is intentional. Argue for additions. |
| `import *` anywhere in `src/datum/`. | Just no. |
| Notebook-driven library changes. | Notebooks are for exploration. Promote code into the library properly. |
| Dependencies added without a justification in the PR. | Every dep is a maintenance cost. |

---

## Architecture decisions

Material changes go through an ADR under `docs/adr/`. The format is short — one page is plenty. Existing ADRs are the template.

---

## Reporting security issues

Email the maintainers directly (see `MAINTAINERS.md` when it lands). Do not open a public issue for a security finding. We will credit responsible disclosure in the changelog.
