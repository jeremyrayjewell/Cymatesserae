from __future__ import annotations

from pathlib import Path

from cymatesserae.config import GraphicLayerConfig, RenderConfig


def test_render_config_defaults_are_stable() -> None:
    config = RenderConfig(audio_path=Path("input.wav"), output_path=Path("output.mp4"))

    assert config.width == 1280
    assert config.height == 720
    assert config.fps == 30
    assert config.point_count == 180
    assert config.plate_mode == (4, 6)
    assert config.graphic_cycle == ("voronoi", "circles", "scribbles", "lines", "geometrics")
    assert config.graphic_layers == ()
    assert config.custom_element_paths == ()


def test_graphic_layer_config_defaults_are_stable() -> None:
    layer = GraphicLayerConfig(name="layer1", family="voronoi")

    assert layer.enabled is True
    assert layer.opacity == 1.0
    assert layer.beats_per_switch == 4
    assert layer.response_gain == 1.0
    assert layer.pattern_layout == "flow"
    assert layer.grid_pattern == "rect"
    assert layer.custom_element_paths == ()
