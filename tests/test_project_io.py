from __future__ import annotations

import json
from pathlib import Path

import pytest

from cymatesserae.config import GraphicLayerConfig, RenderConfig
from cymatesserae.project_io import load_project_file, save_project_file


def test_project_file_round_trip(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    output_path = tmp_path / "output.mp4"
    custom_path = tmp_path / "custom.bmp"
    audio_path.write_bytes(b"audio")
    custom_path.write_bytes(b"bmp")

    config = RenderConfig(
        audio_path=audio_path,
        output_path=output_path,
        width=640,
        height=360,
        fps=24,
        point_count=120,
        cymatic_mode=True,
        plate_mode=(3, 5),
        duration_limit=2.5,
        seed=99,
        style_a="glass",
        style_b="monolith",
        morph_rate=0.25,
        layer_count=4,
        overlap=0.4,
        pattern_layout="grid",
        grid_strength=0.9,
        geometry_rigidity=0.8,
        layer_rigidity=0.6,
        tile_overlap=0.3,
        grid_columns=8,
        grid_rows=4,
        grid_pattern="hex",
        cell_alternation="both",
        reorg_mode="shockwave",
        graphic_cycle=("circles", "custom"),
        beats_per_switch=2,
        stack_interaction="pulse",
        chroma_key_color=(0, 255, 0),
        graphic_layers=(
            GraphicLayerConfig(
                name="Layer A",
                family="custom",
                enabled=True,
                opacity=0.8,
                transparent_colors=((255, 0, 255),),
                graphic_cycle=("custom", "lines"),
                beats_per_switch=3,
                response_gain=1.2,
                style_a="ceramic",
                style_b="lava",
                morph_rate=0.3,
                layer_count=2,
                overlap=0.5,
                pattern_layout="grid",
                grid_strength=0.7,
                geometry_rigidity=0.65,
                layer_rigidity=0.45,
                tile_overlap=0.22,
                grid_columns=6,
                grid_rows=3,
                grid_pattern="brick",
                cell_alternation="color",
                reorg_mode="split",
                custom_element_paths=(custom_path,),
            ),
        ),
    )

    project_path = tmp_path / "project.json"
    save_project_file(project_path, config)
    restored = load_project_file(project_path)

    assert restored.audio_path == audio_path.resolve()
    assert restored.output_path == output_path.resolve()
    assert restored.width == 640
    assert restored.height == 360
    assert restored.fps == 24
    assert restored.point_count == 120
    assert restored.cymatic_mode is True
    assert restored.plate_mode == (3, 5)
    assert restored.duration_limit == 2.5
    assert restored.graphic_cycle == ("circles", "custom")
    assert restored.chroma_key_color == (0, 255, 0)
    assert len(restored.graphic_layers) == 1
    assert restored.graphic_layers[0].name == "Layer A"
    assert restored.graphic_layers[0].custom_element_paths == (custom_path.resolve(),)


def test_project_file_uses_relative_paths_when_practical(tmp_path: Path) -> None:
    project_dir = tmp_path / "projects"
    asset_dir = tmp_path / "assets"
    project_dir.mkdir()
    asset_dir.mkdir()
    audio_path = asset_dir / "audio.wav"
    output_path = asset_dir / "output.mp4"
    audio_path.write_bytes(b"audio")

    config = RenderConfig(audio_path=audio_path, output_path=output_path)
    project_path = project_dir / "project.json"
    save_project_file(project_path, config)

    raw = json.loads(project_path.read_text(encoding="utf-8"))
    assert raw["render_config"]["audio_path"] == "../assets/audio.wav"
    assert raw["render_config"]["output_path"] == "../assets/output.mp4"


def test_load_project_file_rejects_non_object_json(tmp_path: Path) -> None:
    project_path = tmp_path / "project.json"
    project_path.write_text('["not", "an", "object"]', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_project_file(project_path)


def test_load_project_file_rejects_malformed_json(tmp_path: Path) -> None:
    project_path = tmp_path / "project.json"
    project_path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        load_project_file(project_path)


def test_load_project_file_rejects_unsupported_format(tmp_path: Path) -> None:
    project_path = tmp_path / "project.json"
    project_path.write_text(json.dumps({"format": "other", "version": 1, "render_config": {"audio_path": "a.wav"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported project format"):
        load_project_file(project_path)


def test_load_project_file_rejects_unsupported_version(tmp_path: Path) -> None:
    project_path = tmp_path / "project.json"
    project_path.write_text(
        json.dumps({"format": "cymatesserae-project", "version": 99, "render_config": {"audio_path": "a.wav"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported project version"):
        load_project_file(project_path)


def test_load_project_file_rejects_wrong_render_config_shape(tmp_path: Path) -> None:
    project_path = tmp_path / "project.json"
    project_path.write_text(
        json.dumps({"format": "cymatesserae-project", "version": 1, "render_config": ["wrong"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid render_config"):
        load_project_file(project_path)
