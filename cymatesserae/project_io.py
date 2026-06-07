from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

from .config import GraphicLayerConfig, RenderConfig
from .shared import normalize_output_path, parse_chroma_key_color


PROJECT_FORMAT = "cymatesserae-project"
PROJECT_VERSION = 1


def _serialize_path(path: Path, base_dir: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), base_dir.resolve())).as_posix()
    except ValueError:
        return str(path.resolve())


def _deserialize_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _serialize_color(color: tuple[int, int, int]) -> str:
    return "".join(f"{channel:02x}" for channel in color)


def _deserialize_color(value: Any) -> tuple[int, int, int]:
    if isinstance(value, str):
        return parse_chroma_key_color(value)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(channel) for channel in value)
    raise ValueError(f"Unsupported color value: {value!r}")


def graphic_layer_to_dict(layer: GraphicLayerConfig, base_dir: Path) -> dict[str, Any]:
    return {
        "name": layer.name,
        "family": layer.family,
        "enabled": layer.enabled,
        "opacity": layer.opacity,
        "transparent_colors": [_serialize_color(color) for color in layer.transparent_colors],
        "graphic_cycle": list(layer.graphic_cycle),
        "beats_per_switch": layer.beats_per_switch,
        "response_gain": layer.response_gain,
        "style_a": layer.style_a,
        "style_b": layer.style_b,
        "morph_rate": layer.morph_rate,
        "layer_count": layer.layer_count,
        "overlap": layer.overlap,
        "pattern_layout": layer.pattern_layout,
        "grid_strength": layer.grid_strength,
        "geometry_rigidity": layer.geometry_rigidity,
        "layer_rigidity": layer.layer_rigidity,
        "tile_overlap": layer.tile_overlap,
        "grid_columns": layer.grid_columns,
        "grid_rows": layer.grid_rows,
        "grid_pattern": layer.grid_pattern,
        "cell_alternation": layer.cell_alternation,
        "reorg_mode": layer.reorg_mode,
        "custom_element_paths": [_serialize_path(path, base_dir) for path in layer.custom_element_paths],
    }


def graphic_layer_from_dict(data: dict[str, Any], base_dir: Path) -> GraphicLayerConfig:
    transparent_colors_raw = data.get("transparent_colors", [])
    transparent_colors = tuple(_deserialize_color(color) for color in transparent_colors_raw)
    custom_paths = tuple(_deserialize_path(path, base_dir) for path in data.get("custom_element_paths", []))
    return GraphicLayerConfig(
        name=str(data.get("name", "layer")),
        family=str(data.get("family", "voronoi")),
        enabled=bool(data.get("enabled", True)),
        opacity=float(data.get("opacity", 1.0)),
        transparent_colors=transparent_colors,
        graphic_cycle=tuple(str(value) for value in data.get("graphic_cycle", [])),
        beats_per_switch=int(data.get("beats_per_switch", 4)),
        response_gain=float(data.get("response_gain", 1.0)),
        style_a=str(data.get("style_a", "ceramic")),
        style_b=str(data.get("style_b", "neon")),
        morph_rate=float(data.get("morph_rate", 0.18)),
        layer_count=int(data.get("layer_count", 3)),
        overlap=float(data.get("overlap", 0.35)),
        pattern_layout=str(data.get("pattern_layout", "flow")),
        grid_strength=float(data.get("grid_strength", 0.82)),
        geometry_rigidity=float(data.get("geometry_rigidity", 0.75)),
        layer_rigidity=float(data.get("layer_rigidity", 0.55)),
        tile_overlap=float(data.get("tile_overlap", 0.25)),
        grid_columns=int(data.get("grid_columns", 0)),
        grid_rows=int(data.get("grid_rows", 0)),
        grid_pattern=str(data.get("grid_pattern", "rect")),
        cell_alternation=str(data.get("cell_alternation", "none")),
        reorg_mode=str(data.get("reorg_mode", "burst")),
        custom_element_paths=custom_paths,
    )


def render_config_to_project_dict(config: RenderConfig, base_dir: Path) -> dict[str, Any]:
    return {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "render_config": {
            "audio_path": _serialize_path(config.audio_path, base_dir),
            "output_path": _serialize_path(config.output_path, base_dir),
            "width": config.width,
            "height": config.height,
            "fps": config.fps,
            "point_count": config.point_count,
            "preview": config.preview,
            "preview_only": config.preview_only,
            "cymatic_mode": config.cymatic_mode,
            "plate_mode": list(config.plate_mode),
            "hop_length": config.hop_length,
            "n_fft": config.n_fft,
            "duration_limit": config.duration_limit,
            "seed": config.seed,
            "style_a": config.style_a,
            "style_b": config.style_b,
            "morph_rate": config.morph_rate,
            "layer_count": config.layer_count,
            "overlap": config.overlap,
            "pattern_layout": config.pattern_layout,
            "grid_strength": config.grid_strength,
            "geometry_rigidity": config.geometry_rigidity,
            "layer_rigidity": config.layer_rigidity,
            "tile_overlap": config.tile_overlap,
            "grid_columns": config.grid_columns,
            "grid_rows": config.grid_rows,
            "grid_pattern": config.grid_pattern,
            "cell_alternation": config.cell_alternation,
            "reorg_mode": config.reorg_mode,
            "graphic_cycle": list(config.graphic_cycle),
            "beats_per_switch": config.beats_per_switch,
            "stack_interaction": config.stack_interaction,
            "chroma_key_color": _serialize_color(config.chroma_key_color) if config.chroma_key_color is not None else None,
            "transparent_colors": [_serialize_color(color) for color in config.transparent_colors],
            "custom_element_paths": [_serialize_path(path, base_dir) for path in config.custom_element_paths],
            "graphic_layers": [graphic_layer_to_dict(layer, base_dir) for layer in config.graphic_layers],
        },
    }


def render_config_from_project_dict(data: dict[str, Any], base_dir: Path) -> RenderConfig:
    render = data.get("render_config", data)
    if not isinstance(render, dict):
        raise ValueError("Project file has an invalid render_config section.")
    audio_path_value = render.get("audio_path")
    if not audio_path_value:
        raise ValueError("Project file is missing audio_path.")
    output_path_value = render.get("output_path", "cymatesserae_output.mp4")

    graphic_layers_raw = render.get("graphic_layers", ())
    transparent_colors = tuple(_deserialize_color(color) for color in render.get("transparent_colors", ()))
    custom_element_paths = tuple(_deserialize_path(path, base_dir) for path in render.get("custom_element_paths", ()))
    chroma_key_raw = render.get("chroma_key_color")

    return RenderConfig(
        audio_path=_deserialize_path(audio_path_value, base_dir),
        output_path=normalize_output_path(_deserialize_path(output_path_value, base_dir)),
        width=int(render.get("width", 1280)),
        height=int(render.get("height", 720)),
        fps=int(render.get("fps", 30)),
        point_count=int(render.get("point_count", 180)),
        preview=bool(render.get("preview", False)),
        preview_only=bool(render.get("preview_only", False)),
        cymatic_mode=bool(render.get("cymatic_mode", False)),
        plate_mode=tuple(int(value) for value in render.get("plate_mode", (4, 6))),
        hop_length=int(render.get("hop_length", 512)),
        n_fft=int(render.get("n_fft", 2048)),
        duration_limit=render.get("duration_limit"),
        seed=int(render.get("seed", 7)),
        style_a=str(render.get("style_a", "ceramic")),
        style_b=str(render.get("style_b", "neon")),
        morph_rate=float(render.get("morph_rate", 0.18)),
        layer_count=int(render.get("layer_count", 3)),
        overlap=float(render.get("overlap", 0.35)),
        pattern_layout=str(render.get("pattern_layout", "flow")),
        grid_strength=float(render.get("grid_strength", 0.82)),
        geometry_rigidity=float(render.get("geometry_rigidity", 0.75)),
        layer_rigidity=float(render.get("layer_rigidity", 0.55)),
        tile_overlap=float(render.get("tile_overlap", 0.25)),
        grid_columns=int(render.get("grid_columns", 0)),
        grid_rows=int(render.get("grid_rows", 0)),
        grid_pattern=str(render.get("grid_pattern", "rect")),
        cell_alternation=str(render.get("cell_alternation", "none")),
        reorg_mode=str(render.get("reorg_mode", "burst")),
        graphic_cycle=tuple(str(value) for value in render.get("graphic_cycle", ("voronoi", "circles", "scribbles", "lines", "geometrics"))),
        beats_per_switch=int(render.get("beats_per_switch", 4)),
        stack_interaction=str(render.get("stack_interaction", "none")),
        chroma_key_color=_deserialize_color(chroma_key_raw) if chroma_key_raw is not None else None,
        transparent_colors=transparent_colors,
        custom_element_paths=custom_element_paths,
        graphic_layers=tuple(graphic_layer_from_dict(layer, base_dir) for layer in graphic_layers_raw),
    )


def save_project_file(path: Path, config: RenderConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = render_config_to_project_dict(config, path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_project_file(path: Path) -> RenderConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Project file is not valid JSON.") from exc

    if not isinstance(raw, dict):
        raise ValueError("Project file must contain a JSON object.")

    project_format = raw.get("format")
    if project_format is not None and project_format != PROJECT_FORMAT:
        raise ValueError(f"Unsupported project format: {project_format!r}.")

    version = raw.get("version")
    if version is not None and version != PROJECT_VERSION:
        raise ValueError(f"Unsupported project version: {version!r}.")

    try:
        return render_config_from_project_dict(raw, path.parent)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Project file"):
            raise
        raise ValueError(f"Project file has invalid settings: {exc}") from exc
