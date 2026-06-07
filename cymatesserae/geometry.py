from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pygame
from scipy.spatial import Voronoi

from .audio_analysis import AudioSnapshot
from .config import GraphicLayerConfig, LayerRuntimeState, RenderConfig
from .styles import ActiveStyle, resolve_style


GRAPHIC_TYPES = ("voronoi", "circles", "scribbles", "lines", "geometrics", "custom")
CUSTOM_ELEMENT_KEY = (255, 0, 255)


def voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None) -> tuple[list[list[int]], np.ndarray]:
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
    new_regions: list[list[int]] = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2.0

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices, strict=False):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if all(v >= 0 for v in region):
            new_regions.append(region)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in region if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region], dtype=np.float64)
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region, strict=False))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices, dtype=np.float64)


def clip_polygon(points: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    return clipped


def resolve_graphic_cycle(cycle: tuple[str, ...]) -> tuple[str, ...]:
    resolved = tuple(item for item in cycle if item in GRAPHIC_TYPES)
    return resolved if resolved else ("voronoi",)


def resolve_track_families(layer: GraphicLayerConfig, beat_index: int) -> tuple[str, ...]:
    graphic_cycle = resolve_graphic_cycle(layer.graphic_cycle if layer.graphic_cycle else (layer.family,))
    if len(graphic_cycle) > 1:
        beats_per_switch = max(1, layer.beats_per_switch)
        active = graphic_cycle[(max(beat_index, 0) // beats_per_switch) % len(graphic_cycle)]
        return (active,)
    return (graphic_cycle[0],)


def build_seed_points(count: int, width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    cols = max(1, int(np.sqrt(count * width / max(height, 1))))
    rows = int(np.ceil(count / max(cols, 1)))
    xs = np.linspace(0.08, 0.92, cols, dtype=np.float32)
    ys = np.linspace(0.08, 0.92, rows, dtype=np.float32)
    grid = np.array(np.meshgrid(xs, ys), dtype=np.float32).reshape(2, -1).T[:count]
    jitter = rng.normal(0.0, 0.02, size=grid.shape).astype(np.float32)
    pts = np.clip(grid + jitter, 0.04, 0.96)
    pts[:, 0] *= width
    pts[:, 1] *= height
    return pts


def resolve_grid_dimensions(count: int, width: int, height: int, grid_columns: int = 0, grid_rows: int = 0) -> tuple[int, int, int]:
    cols = max(0, int(grid_columns))
    rows = max(0, int(grid_rows))
    explicit_grid = cols > 0 and rows > 0
    if cols == 0 and rows == 0:
        cols = max(1, int(np.sqrt(count * width / max(height, 1))))
        rows = int(np.ceil(count / max(cols, 1)))
    elif cols == 0:
        cols = int(np.ceil(count / max(rows, 1)))
    elif rows == 0:
        rows = int(np.ceil(count / max(cols, 1)))

    cols = max(1, cols)
    rows = max(1, rows)
    if explicit_grid:
        count = cols * rows
    else:
        count = min(count, cols * rows)
    return cols, rows, count


def build_grid_points(
    count: int,
    width: int,
    height: int,
    overlap: float,
    grid_columns: int = 0,
    grid_rows: int = 0,
    grid_pattern: str = "rect",
) -> np.ndarray:
    cols, rows, count = resolve_grid_dimensions(count, width, height, grid_columns, grid_rows)
    edge_x = 0.0 if cols <= 1 else 8.0 / max(width, 1)
    edge_y = 0.0 if rows <= 1 else 8.0 / max(height, 1)
    xs = np.linspace(edge_x, 1.0 - edge_x, cols, dtype=np.float32)
    ys = np.linspace(edge_y, 1.0 - edge_y, rows, dtype=np.float32)
    grid = np.array(np.meshgrid(xs, ys), dtype=np.float32).reshape(2, -1).T[:count]
    step_x = 0.0 if cols <= 1 else float(xs[1] - xs[0])

    if grid_pattern == "brick" and rows > 1 and cols > 1:
        row_indices = np.repeat(np.arange(rows, dtype=np.int32), cols)[:count]
        grid[:, 0] += (row_indices % 2) * (step_x * 0.5)
        grid[:, 0] = np.clip(grid[:, 0], edge_x, 1.0 - edge_x)
    elif grid_pattern == "hex" and rows > 1 and cols > 1:
        row_indices = np.repeat(np.arange(rows, dtype=np.int32), cols)[:count]
        grid[:, 0] += (row_indices % 2) * (step_x * 0.5)
        grid[:, 0] = np.clip(grid[:, 0], edge_x, 1.0 - edge_x)
        center = np.array([0.5, 0.5], dtype=np.float32)
        grid = center + (grid - center) * np.array([1.0, 0.92], dtype=np.float32)
    elif grid_pattern == "diamond":
        center = np.array([0.5, 0.5], dtype=np.float32)
        shifted = grid - center
        diamond = np.empty_like(shifted)
        diamond[:, 0] = shifted[:, 0] + shifted[:, 1] * 0.45
        diamond[:, 1] = shifted[:, 1] + shifted[:, 0] * 0.45
        mins = diamond.min(axis=0)
        maxs = diamond.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-6)
        diamond[:, 0] = edge_x + (diamond[:, 0] - mins[0]) * ((1.0 - edge_x * 2.0) / spans[0])
        diamond[:, 1] = edge_y + (diamond[:, 1] - mins[1]) * ((1.0 - edge_y * 2.0) / spans[1])
        grid = diamond
    grid[:, 0] *= width
    grid[:, 1] *= height
    return grid.astype(np.float32)


def cymatic_force(points: np.ndarray, width: int, height: int, mode: tuple[int, int], strength: float) -> np.ndarray:
    m, n = mode
    unit = np.empty_like(points, dtype=np.float32)
    unit[:, 0] = points[:, 0] / width
    unit[:, 1] = points[:, 1] / height
    nodal_x = np.round(unit[:, 0] * m) / max(m, 1)
    nodal_y = np.round(unit[:, 1] * n) / max(n, 1)
    target = np.empty_like(points, dtype=np.float32)
    target[:, 0] = nodal_x * width
    target[:, 1] = nodal_y * height
    return (target - points) * strength


def prepare_layers(
    points: np.ndarray,
    style: ActiveStyle,
    snapshot: AudioSnapshot,
    layer_count: int,
    overlap: float,
    time_phase: float,
    pattern_layout: str = "flow",
    layer_rigidity: float = 0.0,
) -> list[np.ndarray]:
    layers: list[np.ndarray] = []
    center = points.mean(axis=0, dtype=np.float32)
    delta = points - center
    count = max(1, layer_count)
    rigid_mix = np.clip(layer_rigidity, 0.0, 1.0) if pattern_layout == "grid" else 0.0
    for idx in range(count):
        layer_t = 0.0 if count == 1 else idx / (count - 1)
        offset_angle = time_phase * 0.045 + layer_t * math.tau * 0.6 + snapshot.brightness * 1.6
        offset_vec = np.array([math.cos(offset_angle), math.sin(offset_angle)], dtype=np.float32)
        offset_mag = style.layer_spread * overlap * (0.4 + layer_t) * (0.8 + snapshot.percussive_flux * 0.9)
        offset_mag *= 1.0 - rigid_mix * 0.92
        pulse = 1.0 + (layer_t - 0.5) * style.pulse_gain * snapshot.bass * 0.22 * (1.0 - rigid_mix * 0.9)
        shear = np.empty_like(points, dtype=np.float32)
        shear_gain_x = 0.08 * overlap * (idx + 1) * (1.0 - rigid_mix * 0.95)
        shear_gain_y = 0.06 * overlap * (idx + 1) * (1.0 - rigid_mix * 0.95)
        shear[:, 0] = delta[:, 0] + delta[:, 1] * shear_gain_x
        shear[:, 1] = delta[:, 1] - delta[:, 0] * shear_gain_y
        layer_points = center + shear * pulse
        layer_points += offset_vec * offset_mag * style.offset_gain
        if rigid_mix > 0.0:
            layer_points = points * rigid_mix + layer_points * (1.0 - rigid_mix)
        layers.append(layer_points.astype(np.float32))
    return layers


def estimate_grid_cell_span(points: np.ndarray, width: int, height: int) -> tuple[float, float]:
    if len(points) <= 1:
        return float(width), float(height)

    def _axis_span(values: np.ndarray, fallback: float) -> float:
        rounded = np.unique(np.round(values, 3))
        if len(rounded) <= 1:
            return fallback
        diffs = np.diff(np.sort(rounded))
        positive = diffs[diffs > 1e-3]
        if len(positive) == 0:
            return fallback
        return float(np.min(positive))

    span_x = _axis_span(points[:, 0], float(width))
    span_y = _axis_span(points[:, 1], float(height))
    return span_x, span_y


def grid_cell_span_from_dimensions(width: int, height: int, cols: int, rows: int) -> tuple[float, float]:
    if cols <= 1:
        span_x = float(width - 16)
    else:
        span_x = float((width - 16) / max(cols - 1, 1))
    if rows <= 1:
        span_y = float(height - 16)
    else:
        span_y = float((height - 16) / max(rows - 1, 1))
    return max(span_x, 8.0), max(span_y, 8.0)


def load_custom_elements(custom_element_paths: tuple[Path, ...]) -> list[pygame.Surface]:
    sprites: list[pygame.Surface] = []
    for custom_element_path in custom_element_paths:
        if not custom_element_path.exists():
            raise FileNotFoundError(f"Custom element file not found: {custom_element_path}")
        sprite = pygame.image.load(custom_element_path.as_posix()).convert()
        sprite.set_colorkey(CUSTOM_ELEMENT_KEY)
        sprites.append(sprite)
    return sprites


def build_legacy_graphic_layers(config: RenderConfig) -> tuple[GraphicLayerConfig, ...]:
    layers: list[GraphicLayerConfig] = []
    for family in resolve_graphic_cycle(config.graphic_cycle):
        custom_paths = config.custom_element_paths if family == "custom" else ()
        layers.append(
            GraphicLayerConfig(
                name=family,
                family=family,
                enabled=True,
                opacity=1.0,
                transparent_colors=(),
                graphic_cycle=(family,),
                beats_per_switch=config.beats_per_switch,
                response_gain=1.0,
                style_a=config.style_a,
                style_b=config.style_b,
                morph_rate=config.morph_rate,
                layer_count=config.layer_count,
                overlap=config.overlap,
                pattern_layout=config.pattern_layout,
                grid_strength=config.grid_strength,
                geometry_rigidity=config.geometry_rigidity,
                layer_rigidity=config.layer_rigidity,
                tile_overlap=config.tile_overlap,
                grid_columns=config.grid_columns,
                grid_rows=config.grid_rows,
                grid_pattern=config.grid_pattern,
                cell_alternation=config.cell_alternation,
                reorg_mode=config.reorg_mode,
                custom_element_paths=custom_paths,
            )
        )
    return tuple(layers)


def effective_graphic_layers(config: RenderConfig) -> tuple[GraphicLayerConfig, ...]:
    if config.graphic_layers:
        enabled = tuple(layer for layer in config.graphic_layers if layer.enabled and layer.family in GRAPHIC_TYPES)
        if enabled:
            return enabled
    return build_legacy_graphic_layers(config)


def build_layer_runtime(layer: GraphicLayerConfig, config: RenderConfig, seed: int) -> LayerRuntimeState:
    rng = np.random.default_rng(seed)
    is_grid_layout = layer.pattern_layout == "grid"
    if is_grid_layout:
        grid_cols, grid_rows, _ = resolve_grid_dimensions(
            config.point_count,
            config.width,
            config.height,
            layer.grid_columns,
            layer.grid_rows,
        )
        points = build_grid_points(
            config.point_count,
            config.width,
            config.height,
            layer.tile_overlap,
            layer.grid_columns,
            layer.grid_rows,
            layer.grid_pattern,
        )
        grid_cell_span = grid_cell_span_from_dimensions(config.width, config.height, grid_cols, grid_rows)
    else:
        grid_cols, grid_rows = 0, 0
        points = build_seed_points(config.point_count, config.width, config.height, rng)
        grid_cell_span = None
    base_points = points.copy()
    velocity = np.zeros_like(points, dtype=np.float32)
    custom_elements = load_custom_elements(layer.custom_element_paths)
    return LayerRuntimeState(
        layer=layer,
        points=points,
        base_points=base_points,
        velocity=velocity,
        rng=rng,
        previous_onset=0.0,
        style_a=resolve_style(layer.style_a),
        style_b=resolve_style(layer.style_b),
        custom_elements=custom_elements,
        inverted_custom_elements=[invert_custom_surface(sprite) for sprite in custom_elements],
        custom_scale_cache={},
        custom_transform_cache={},
        voronoi_cache={},
        channel_surface=pygame.Surface((config.width, config.height), pygame.SRCALPHA, 32),
        is_grid_layout=is_grid_layout,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        grid_cell_span=grid_cell_span,
    )


def invert_custom_surface(surface: pygame.Surface) -> pygame.Surface:
    inverted = surface.copy()
    rgb = pygame.surfarray.array3d(inverted)
    key = np.asarray(CUSTOM_ELEMENT_KEY, dtype=np.uint8)
    mask = np.any(rgb != key[None, None, :], axis=2)
    rgb[mask] = 255 - rgb[mask]
    pygame.surfarray.blit_array(inverted, rgb)
    inverted.set_colorkey(CUSTOM_ELEMENT_KEY)
    return inverted


CUSTOM_SCALE_CACHE_LIMIT = 96
CUSTOM_TRANSFORM_CACHE_LIMIT = 192
CUSTOM_MAX_SURFACE_DIMENSION_MULTIPLIER = 1.25
CUSTOM_MAX_SURFACE_AREA_MULTIPLIER = 1.25
CUSTOM_CACHEABLE_SURFACE_AREA_MULTIPLIER = 0.35


def clamp_custom_surface_size(width: int, height: int, render_width: int, render_height: int) -> tuple[int, int]:
    width = max(8, int(width))
    height = max(8, int(height))
    max_width = max(8, int(render_width * CUSTOM_MAX_SURFACE_DIMENSION_MULTIPLIER))
    max_height = max(8, int(render_height * CUSTOM_MAX_SURFACE_DIMENSION_MULTIPLIER))
    scale = min(max_width / width, max_height / height, 1.0)
    max_area = max(64, int(render_width * render_height * CUSTOM_MAX_SURFACE_AREA_MULTIPLIER))
    scaled_area = width * height * (scale ** 2)
    if scaled_area > max_area:
        scale = min(scale, (max_area / max(width * height, 1)) ** 0.5)
    return max(8, int(width * scale)), max(8, int(height * scale))


def _should_cache_custom_surface(width: int, height: int, render_width: int, render_height: int) -> bool:
    max_cacheable_area = max(64, int(render_width * render_height * CUSTOM_CACHEABLE_SURFACE_AREA_MULTIPLIER))
    return width * height <= max_cacheable_area


def get_scaled_custom_surface(
    sprite_index: int,
    width: int,
    height: int,
    inverted: bool,
    custom_elements: list[pygame.Surface],
    inverted_custom_elements: list[pygame.Surface],
    scale_cache: dict[tuple[int, int, int, bool], pygame.Surface],
) -> pygame.Surface:
    key = (sprite_index, width, height, inverted)
    cached = scale_cache.get(key)
    if cached is not None:
        return cached
    source_list = inverted_custom_elements if inverted else custom_elements
    source = source_list[sprite_index]
    if source.get_width() == width and source.get_height() == height:
        scaled = source
    else:
        try:
            scaled = pygame.transform.scale(source, (width, height))
        except pygame.error:
            scaled = source
    if len(scale_cache) >= CUSTOM_SCALE_CACHE_LIMIT:
        scale_cache.clear()
    scale_cache[key] = scaled
    return scaled


def get_transformed_custom_surface(
    sprite_index: int,
    width: int,
    height: int,
    render_width: int,
    render_height: int,
    angle: float,
    inverted: bool,
    flip_xy: bool,
    custom_elements: list[pygame.Surface],
    inverted_custom_elements: list[pygame.Surface],
    scale_cache: dict[tuple[int, int, int, bool], pygame.Surface],
    transform_cache: dict[tuple[int, int, int, int, bool, bool], pygame.Surface],
    fast_preview: bool,
) -> pygame.Surface:
    width, height = clamp_custom_surface_size(width, height, render_width, render_height)
    scaled = get_scaled_custom_surface(
        sprite_index,
        width,
        height,
        inverted,
        custom_elements,
        inverted_custom_elements,
        scale_cache,
    )
    transformed = pygame.transform.flip(scaled, True, True) if flip_xy else scaled
    if not fast_preview:
        try:
            return pygame.transform.rotate(transformed, angle)
        except pygame.error:
            return transformed

    angle_bucket = int(round(angle / 12.0)) * 12
    key = (sprite_index, width, height, angle_bucket, inverted, flip_xy)
    cached = transform_cache.get(key)
    if cached is not None:
        return cached
    try:
        rotated = pygame.transform.rotate(transformed, float(angle_bucket))
    except pygame.error:
        return transformed
    if not _should_cache_custom_surface(rotated.get_width(), rotated.get_height(), render_width, render_height):
        return rotated
    if len(transform_cache) >= CUSTOM_TRANSFORM_CACHE_LIMIT:
        transform_cache.clear()
    transform_cache[key] = rotated
    return rotated


def rebuild_bases(config: RenderConfig, reorg_mode: str, rng: np.random.Generator, points: np.ndarray, base_points: np.ndarray) -> np.ndarray:
    if reorg_mode == "split":
        direction = np.sign(points[:, 0] - config.width * 0.5).astype(np.float32)
        direction[direction == 0.0] = 1.0
        rebuilt = base_points.copy()
        rebuilt[:, 0] += direction * config.width * 0.08
        rebuilt[:, 1] += rng.normal(0.0, config.height * 0.03, size=len(rebuilt)).astype(np.float32)
        return np.clip(rebuilt, [8.0, 8.0], [config.width - 8.0, config.height - 8.0])
    return build_seed_points(config.point_count, config.width, config.height, rng)


def reorg_impulse(
    config: RenderConfig,
    reorg_mode: str,
    rng: np.random.Generator,
    points: np.ndarray,
    snapshot: AudioSnapshot,
) -> np.ndarray:
    impulse = rng.normal(0.0, 1.0, size=points.shape).astype(np.float32)
    strength = 18.0 + 56.0 * snapshot.percussive_flux
    if reorg_mode == "swirl":
        center = np.array([config.width * 0.5, config.height * 0.5], dtype=np.float32)
        delta = points - center
        impulse[:, 0] = -delta[:, 1]
        impulse[:, 1] = delta[:, 0]
        impulse /= np.maximum(np.linalg.norm(impulse, axis=1, keepdims=True), 1.0)
        strength *= 1.35
    elif reorg_mode == "split":
        impulse[:, 0] = np.sign(points[:, 0] - config.width * 0.5)
        impulse[:, 1] = rng.normal(0.0, 0.25, size=len(points)).astype(np.float32)
        strength *= 1.20
    elif reorg_mode == "shockwave":
        center = np.array([config.width * 0.5, config.height * 0.5], dtype=np.float32)
        impulse = points - center
        impulse /= np.maximum(np.linalg.norm(impulse, axis=1, keepdims=True), 1.0)
        strength *= 1.45
    return impulse * strength
