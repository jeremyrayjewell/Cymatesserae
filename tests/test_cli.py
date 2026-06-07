from __future__ import annotations

from pathlib import Path

from cymatesserae import cli


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
