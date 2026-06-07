from __future__ import annotations

import json
from pathlib import Path

from cymatesserae.config import GraphicLayerConfig, RenderConfig
from cymatesserae.project_io import load_preset_file, save_preset_file


def test_preset_file_round_trip(tmp_path: Path) -> None:
    custom_path = tmp_path / "custom.bmp"
    custom_path.write_bytes(b"bmp")
    config = RenderConfig(
        audio_path=tmp_path / "audio.wav",
        output_path=tmp_path / "output.mp4",
        point_count=96,
        cymatic_mode=True,
        plate_mode=(5, 7),
        seed=11,
        style_a="glass",
        style_b="monolith",
        morph_rate=0.22,
        layer_count=4,
        overlap=0.4,
        pattern_layout="grid",
        grid_strength=0.88,
        geometry_rigidity=0.73,
        layer_rigidity=0.51,
        tile_overlap=0.28,
        grid_columns=9,
        grid_rows=5,
        grid_pattern="hex",
        cell_alternation="both",
        reorg_mode="shockwave",
        graphic_cycle=("circles", "custom"),
        beats_per_switch=3,
        stack_interaction="pulse",
        chroma_key_color=(0, 255, 0),
        graphic_layers=(
            GraphicLayerConfig(
                name="Layer A",
                family="custom",
                enabled=True,
                opacity=0.8,
                graphic_cycle=("custom", "lines"),
                style_a="ceramic",
                style_b="lava",
                custom_element_paths=(custom_path,),
            ),
        ),
    )
    preset_path = tmp_path / "preset.json"
    save_preset_file(preset_path, config)

    restored = load_preset_file(
        preset_path,
        RenderConfig(audio_path=tmp_path / "keep.wav", output_path=tmp_path / "keep.mp4"),
    )

    assert restored.audio_path == (tmp_path / "keep.wav")
    assert restored.output_path == (tmp_path / "keep.mp4")
    assert restored.point_count == 96
    assert restored.cymatic_mode is True
    assert restored.plate_mode == (5, 7)
    assert restored.style_a == "glass"
    assert restored.style_b == "monolith"
    assert restored.graphic_cycle == ("circles", "custom")
    assert restored.graphic_layers[0].custom_element_paths == (custom_path.resolve(),)


def test_preset_file_uses_relative_paths_when_practical(tmp_path: Path) -> None:
    preset_dir = tmp_path / "presets"
    asset_dir = tmp_path / "assets"
    preset_dir.mkdir()
    asset_dir.mkdir()
    custom_path = asset_dir / "custom.bmp"
    custom_path.write_bytes(b"bmp")

    config = RenderConfig(
        audio_path=tmp_path / "ignored.wav",
        output_path=tmp_path / "ignored.mp4",
        graphic_layers=(GraphicLayerConfig(name="Layer 1", family="custom", custom_element_paths=(custom_path,)),),
    )
    preset_path = preset_dir / "preset.json"
    save_preset_file(preset_path, config)
    raw = json.loads(preset_path.read_text(encoding="utf-8"))

    assert raw["preset"]["graphic_layers"][0]["custom_element_paths"] == ["../assets/custom.bmp"]


def test_load_preset_file_preserves_audio_and_output_paths(tmp_path: Path) -> None:
    custom_path = tmp_path / "custom.bmp"
    custom_path.write_bytes(b"bmp")
    preset_source = RenderConfig(
        audio_path=tmp_path / "ignored.wav",
        output_path=tmp_path / "ignored.mp4",
        style_a="lava",
        graphic_layers=(GraphicLayerConfig(name="Layer 1", family="custom", custom_element_paths=(custom_path,)),),
    )
    preset_path = tmp_path / "preset.json"
    save_preset_file(preset_path, preset_source)

    base = RenderConfig(
        audio_path=tmp_path / "keep.wav",
        output_path=tmp_path / "keep.mp4",
        style_a="ceramic",
    )
    applied = load_preset_file(preset_path, base)

    assert applied.audio_path == base.audio_path
    assert applied.output_path == base.output_path
    assert applied.style_a == "lava"
    assert applied.graphic_layers[0].custom_element_paths == (custom_path.resolve(),)
