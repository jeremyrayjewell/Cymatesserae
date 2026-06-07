from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pygame

if TYPE_CHECKING:
    from .styles import StylePreset


@dataclass(slots=True)
class RenderConfig:
    audio_path: Path
    output_path: Path
    width: int = 1280
    height: int = 720
    fps: int = 30
    point_count: int = 180
    preview: bool = False
    preview_only: bool = False
    cymatic_mode: bool = False
    plate_mode: tuple[int, int] = (4, 6)
    hop_length: int = 512
    n_fft: int = 2048
    duration_limit: float | None = None
    seed: int = 7
    style_a: str = "ceramic"
    style_b: str = "neon"
    morph_rate: float = 0.18
    layer_count: int = 3
    overlap: float = 0.35
    pattern_layout: str = "flow"
    grid_strength: float = 0.82
    geometry_rigidity: float = 0.75
    layer_rigidity: float = 0.55
    tile_overlap: float = 0.25
    grid_columns: int = 0
    grid_rows: int = 0
    grid_pattern: str = "rect"
    cell_alternation: str = "none"
    reorg_mode: str = "burst"
    graphic_cycle: tuple[str, ...] = ("voronoi", "circles", "scribbles", "lines", "geometrics")
    beats_per_switch: int = 4
    stack_interaction: str = "none"
    chroma_key_color: tuple[int, int, int] | None = None
    transparent_colors: tuple[tuple[int, int, int], ...] = ()
    custom_element_paths: tuple[Path, ...] = ()
    graphic_layers: tuple["GraphicLayerConfig", ...] = ()


@dataclass(slots=True)
class GraphicLayerConfig:
    name: str
    family: str
    enabled: bool = True
    opacity: float = 1.0
    transparent_colors: tuple[tuple[int, int, int], ...] = ()
    graphic_cycle: tuple[str, ...] = ()
    beats_per_switch: int = 4
    response_gain: float = 1.0
    style_a: str = "ceramic"
    style_b: str = "neon"
    morph_rate: float = 0.18
    layer_count: int = 3
    overlap: float = 0.35
    pattern_layout: str = "flow"
    grid_strength: float = 0.82
    geometry_rigidity: float = 0.75
    layer_rigidity: float = 0.55
    tile_overlap: float = 0.25
    grid_columns: int = 0
    grid_rows: int = 0
    grid_pattern: str = "rect"
    cell_alternation: str = "none"
    reorg_mode: str = "burst"
    custom_element_paths: tuple[Path, ...] = ()


@dataclass(slots=True)
class LayerRuntimeState:
    layer: GraphicLayerConfig
    points: np.ndarray
    base_points: np.ndarray
    velocity: np.ndarray
    rng: np.random.Generator
    previous_onset: float
    style_a: StylePreset
    style_b: StylePreset
    custom_elements: list[pygame.Surface]
    inverted_custom_elements: list[pygame.Surface]
    custom_scale_cache: dict[tuple[int, int, int, bool], pygame.Surface]
    custom_transform_cache: dict[tuple[int, int, int, int, bool, bool], pygame.Surface]
    voronoi_cache: dict[int, tuple[int, list[list[int]], np.ndarray]]
    channel_surface: pygame.Surface
    is_grid_layout: bool
    grid_cols: int
    grid_rows: int
    grid_cell_span: tuple[float, float] | None
