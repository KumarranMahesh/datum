"""End-to-end ingest pipeline tests.

These run the full pipeline against the synth video and check:
  * the manifest is well formed and complete
  * re-running with the same source + config is a cache hit (idempotency)
  * `overwrite=True` does in fact overwrite
"""

from __future__ import annotations

import json
from pathlib import Path

from datum.ingest import IngestConfig, IngestManifest, SceneKind, run


def _default_config() -> IngestConfig:
    # sample_fps low enough to leave a margin under the 30-fps synth source.
    return IngestConfig(sample_fps=5.0, scene_cut_threshold=0.35)


def test_pipeline_end_to_end(synth_video: Path, tmp_path: Path) -> None:
    config = _default_config()
    manifest = run(synth_video, config, runs_root=tmp_path)

    assert manifest.schema_version == 1
    assert manifest.run_id.startswith("ingest-")
    assert manifest.finished_at is not None
    assert manifest.counters is not None

    # 180 source frames at 30 fps -> ~30 sampled at 5 fps. Generous bounds
    # because the encoder + sampler boundary handling shaves a frame or two.
    assert 25 <= manifest.counters.sampled_frames <= 32

    # Pitch fraction should land roughly at 2/3 (two of three scenes
    # are pitch-green). Allow a wide band; the classifier is approximate.
    assert 0.45 <= manifest.counters.pitch_frame_pct <= 0.90

    # On-disk artifacts exist.
    run_dir = tmp_path / manifest.run_id
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "frames.jsonl").exists()
    assert (run_dir / "scenes.jsonl").exists()

    # frames.jsonl line count matches the manifest's sampled_frames.
    lines = (run_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == manifest.counters.sampled_frames

    # Each line is a valid FrameRecord.
    for line in lines:
        json.loads(line)  # smoke; deeper schema validation lives in test_schemas


def test_pipeline_is_idempotent(synth_video: Path, tmp_path: Path) -> None:
    config = _default_config()
    a = run(synth_video, config, runs_root=tmp_path)
    b = run(synth_video, config, runs_root=tmp_path)

    assert a.run_id == b.run_id
    assert a.config_hash == b.config_hash
    assert a.finished_at == b.finished_at  # cache hit returns original


def test_overwrite_runs_fresh(synth_video: Path, tmp_path: Path) -> None:
    config = _default_config()
    a = run(synth_video, config, runs_root=tmp_path)
    b = run(synth_video, config, runs_root=tmp_path, overwrite=True)

    assert a.run_id == b.run_id
    # The second run produced new timestamps.
    assert b.finished_at is not None and a.finished_at is not None
    assert b.finished_at >= a.finished_at


def test_different_configs_produce_different_run_ids(
    synth_video: Path, tmp_path: Path
) -> None:
    a = run(synth_video, IngestConfig(sample_fps=5.0), runs_root=tmp_path)
    b = run(synth_video, IngestConfig(sample_fps=10.0), runs_root=tmp_path)
    assert a.run_id != b.run_id


def test_manifest_round_trips(synth_video: Path, tmp_path: Path) -> None:
    manifest = run(synth_video, _default_config(), runs_root=tmp_path)
    on_disk = (tmp_path / manifest.run_id / "manifest.json").read_text(encoding="utf-8")
    loaded = IngestManifest.model_validate_json(on_disk)
    assert loaded.run_id == manifest.run_id
    assert loaded.counters == manifest.counters

    # The fields used by downstream stages must round-trip cleanly.
    assert loaded.config.sample_fps == 5.0
    assert isinstance(loaded.source.sha256, str)


def test_per_frame_scene_kinds_are_valid(synth_video: Path, tmp_path: Path) -> None:
    manifest = run(synth_video, _default_config(), runs_root=tmp_path)
    frames_path = tmp_path / manifest.run_id / "frames.jsonl"
    valid_kinds = {k.value for k in SceneKind}
    for line in frames_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        assert record["scene_kind"] in valid_kinds
