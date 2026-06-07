from __future__ import annotations

from pathlib import Path

import pytest

from cymatesserae import cli
from cymatesserae.config import GraphicLayerConfig, RenderConfig
from cymatesserae.project_io import save_project_file


def test_cli_main_builds_render_config_and_calls_renderer(monkeypatch, tmp_path: Path, generated_wav: Path) -> None:
    output_path = tmp_path / "render_out"
    captured: dict[str, object] = {}

    def fake_render_project(config) -> None:
        captured["config"] = config

    monkeypatch.setattr("cymatesserae.renderer.render_project", fake_render_project)
    monkeypatch.setattr(
        "sys.argv",
        [
            "cymatesserae",
            str(generated_wav),
            "--output",
            str(output_path),
            "--width",
            "321",
            "--height",
            "179",
            "--fps",
            "12",
            "--points",
            "24",
            "--duration",
            "0.5",
            "--cymatic",
            "--style",
            "glass",
            "--style-b",
            "monolith",
            "--graphic-cycle",
            "circles,lines",
            "--beats-per-switch",
            "2",
        ],
    )

    result = cli.main()
    config = captured["config"]

    assert result == 0
    assert config.audio_path == generated_wav
    assert config.output_path == output_path.with_suffix(".mp4")
    assert config.width == 322
    assert config.height == 180
    assert config.fps == 12
    assert config.point_count == 24
    assert config.duration_limit == 0.5
    assert config.cymatic_mode is True
    assert config.style_a == "glass"
    assert config.style_b == "monolith"
    assert config.graphic_cycle == ("circles", "lines")
    assert config.beats_per_switch == 2


def test_cli_main_loads_project_file(monkeypatch, tmp_path: Path, generated_wav: Path) -> None:
    output_path = tmp_path / "project_out.mp4"
    project_path = tmp_path / "project.json"
    captured: dict[str, object] = {}

    def fake_render_project(config) -> None:
        captured["config"] = config

    save_project_file(
        project_path,
        RenderConfig(
            audio_path=generated_wav,
            output_path=output_path,
            width=400,
            height=222,
            fps=15,
            point_count=50,
            cymatic_mode=True,
            graphic_layers=(GraphicLayerConfig(name="Layer 1", family="voronoi", enabled=True),),
        ),
    )

    monkeypatch.setattr("cymatesserae.renderer.render_project", fake_render_project)
    monkeypatch.setattr("sys.argv", ["cymatesserae", "--project", str(project_path)])

    result = cli.main()
    config = captured["config"]

    assert result == 0
    assert config.audio_path == generated_wav.resolve()
    assert config.output_path == output_path.resolve()
    assert config.width == 400
    assert config.height == 222
    assert config.fps == 15
    assert config.point_count == 50
    assert config.cymatic_mode is True
    assert len(config.graphic_layers) == 1


def test_cli_main_project_values_can_be_overridden(monkeypatch, tmp_path: Path, generated_wav: Path) -> None:
    project_path = tmp_path / "project.json"
    captured: dict[str, object] = {}

    def fake_render_project(config) -> None:
        captured["config"] = config

    save_project_file(
        project_path,
        RenderConfig(
            audio_path=generated_wav,
            output_path=tmp_path / "base.mp4",
            width=400,
            height=222,
            fps=15,
            point_count=50,
            cymatic_mode=False,
        ),
    )

    monkeypatch.setattr("cymatesserae.renderer.render_project", fake_render_project)
    monkeypatch.setattr(
        "sys.argv",
        [
            "cymatesserae",
            "--project",
            str(project_path),
            "--width",
            "321",
            "--fps",
            "12",
            "--cymatic",
            "--output",
            str(tmp_path / "override"),
        ],
    )

    cli.main()
    config = captured["config"]

    assert config.width == 322
    assert config.height == 222
    assert config.fps == 12
    assert config.cymatic_mode is True
    assert config.output_path == (tmp_path / "override.mp4")


def test_cli_main_applies_preset_without_replacing_audio_or_output(monkeypatch, tmp_path: Path, generated_wav: Path) -> None:
    from cymatesserae.project_io import save_preset_file

    preset_path = tmp_path / "preset.json"
    captured: dict[str, object] = {}

    def fake_render_project(config) -> None:
        captured["config"] = config

    save_preset_file(
        preset_path,
        RenderConfig(
            audio_path=tmp_path / "ignored.wav",
            output_path=tmp_path / "ignored.mp4",
            point_count=88,
            cymatic_mode=True,
            style_a="glass",
            style_b="monolith",
        ),
    )

    monkeypatch.setattr("cymatesserae.renderer.render_project", fake_render_project)
    monkeypatch.setattr(
        "sys.argv",
        [
            "cymatesserae",
            str(generated_wav),
            "--output",
            str(tmp_path / "render.mp4"),
            "--preset",
            str(preset_path),
        ],
    )

    cli.main()
    config = captured["config"]

    assert config.audio_path == generated_wav
    assert config.output_path == (tmp_path / "render.mp4")
    assert config.point_count == 88
    assert config.cymatic_mode is True
    assert config.style_a == "glass"
    assert config.style_b == "monolith"


def test_cli_main_project_load_errors_are_friendly_for_missing_file(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr("sys.argv", ["cymatesserae", "--project", str(missing)])

    with pytest.raises(SystemExit, match="Could not load project file: file not found"):
        cli.main()


def test_cli_main_project_load_errors_are_friendly_for_bad_json(monkeypatch, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{bad json", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["cymatesserae", "--project", str(bad)])

    with pytest.raises(SystemExit, match="Could not load project file: Project file is not valid JSON."):
        cli.main()
