from __future__ import annotations

import math

import numpy as np
import pygame
from scipy.spatial import Voronoi

from ..audio_analysis import AudioSnapshot
from ..geometry import (
    clip_polygon,
    estimate_grid_cell_span,
    get_transformed_custom_surface,
    voronoi_finite_polygons_2d,
)
from ..styles import ActiveStyle, build_palette, sanitize_rgb_array, sanitize_rgb_triplet


def apply_channel_composite(surface: pygame.Surface, opacity: float, transparent_colors: tuple[tuple[int, int, int], ...]) -> pygame.Surface:
    result = surface
    if transparent_colors:
        rgb = pygame.surfarray.pixels3d(result)
        alpha = pygame.surfarray.pixels_alpha(result)
        for transparent_color in transparent_colors:
            key = np.asarray(transparent_color, dtype=np.uint8)
            matches = np.all(rgb == key[None, None, :], axis=2)
            if np.any(matches):
                alpha[matches] = 0
        del alpha
        del rgb
    if opacity < 0.999:
        result.set_alpha(max(0, min(255, int(round(opacity * 255.0)))))
    return result


def _circle_radius(style: ActiveStyle, snapshot: AudioSnapshot, beat_pulse: float, layer_weight: float, width: int) -> int:
    radius = width * (0.008 + 0.018 * snapshot.bass + 0.012 * beat_pulse)
    radius *= 0.85 + layer_weight * style.pulse_gain
    return max(2, int(radius))


def _draw_circles(
    surface: pygame.Surface,
    layers: list[np.ndarray],
    style: ActiveStyle,
    snapshot: AudioSnapshot,
    style_mix: float,
    beat_pulse: float,
    chroma_key_color: tuple[int, int, int] | None,
    cell_alternation: str = "none",
) -> None:
    count = max(1, len(layers))
    border_color = sanitize_rgb_triplet(style.border_color, chroma_key_color).tolist()
    for layer_idx, layer_points in enumerate(layers):
        layer_weight = 0.0 if count == 1 else layer_idx / (count - 1)
        phases = (
            np.arange(len(layer_points), dtype=np.float32) / max(len(layer_points), 1)
            + style_mix * 0.45
            + snapshot.brightness * 0.9
            + layer_idx * 0.11
        )
        colors = sanitize_rgb_array(build_palette(style, phases), chroma_key_color)
        radius = _circle_radius(style, snapshot, beat_pulse, layer_weight, surface.get_width())
        for idx, point in enumerate(layer_points):
            parity = idx % 2
            center = (int(point[0]), int(point[1]))
            fill = tuple(int(v) for v in colors[idx])
            if parity == 1 and "color" in cell_alternation:
                fill = tuple(255 - c for c in fill)
            pygame.draw.circle(surface, fill, center, radius, 0)
            ring = radius + max(1, int(beat_pulse * 8))
            pygame.draw.circle(surface, border_color, center, ring, 1)


def _draw_scribbles(
    surface: pygame.Surface,
    layers: list[np.ndarray],
    style: ActiveStyle,
    snapshot: AudioSnapshot,
    beat_pulse: float,
    time_phase: float,
    chroma_key_color: tuple[int, int, int] | None,
    cell_alternation: str = "none",
) -> None:
    for layer_idx, layer_points in enumerate(layers):
        if len(layer_points) < 4:
            continue
        order = np.argsort(np.arctan2(layer_points[:, 1] - surface.get_height() * 0.5, layer_points[:, 0] - surface.get_width() * 0.5))
        ordered = layer_points[order]
        stride = max(3, len(ordered) // 18)
        jitter_amp = 8.0 + 22.0 * snapshot.percussive_flux + 14.0 * beat_pulse
        for offset in range(min(stride, 6)):
            strand = ordered[offset::stride]
            if len(strand) < 3:
                continue
            parity = offset % 2
            jitter = np.empty_like(strand)
            phase = time_phase * 2.0 + offset * 0.7 + layer_idx * 0.5
            jitter[:, 0] = np.sin(np.arange(len(strand), dtype=np.float32) * 0.9 + phase) * jitter_amp
            jitter[:, 1] = np.cos(np.arange(len(strand), dtype=np.float32) * 0.8 - phase) * jitter_amp
            squiggle = np.clip(strand + jitter, [0, 0], [surface.get_width() - 1, surface.get_height() - 1]).astype(np.int32)
            color_phase = np.linspace(0.0, 1.0, len(squiggle), dtype=np.float32)
            color = sanitize_rgb_triplet(
                build_palette(style, color_phase + snapshot.harmonic_ratio + offset * 0.08)[0],
                chroma_key_color,
            )
            if parity == 1 and "color" in cell_alternation:
                color = tuple(255 - c for c in color)
            pygame.draw.lines(surface, color.tolist(), False, squiggle.tolist(), 2 + (offset % 2))


def _draw_lines(
    surface: pygame.Surface,
    layers: list[np.ndarray],
    style: ActiveStyle,
    snapshot: AudioSnapshot,
    beat_pulse: float,
    time_phase: float,
    chroma_key_color: tuple[int, int, int] | None,
    cell_alternation: str = "none",
) -> None:
    direction_angle = time_phase * 0.9 + snapshot.brightness * math.tau + snapshot.contrast * 1.4
    base_direction = np.array([math.cos(direction_angle), math.sin(direction_angle)], dtype=np.float32)
    for layer_idx, layer_points in enumerate(layers):
        phases = np.linspace(0.0, 1.0, len(layer_points), dtype=np.float32) + layer_idx * 0.16 + snapshot.bass * 0.25
        colors = sanitize_rgb_array(build_palette(style, phases), chroma_key_color)
        length = 24.0 + surface.get_width() * 0.04 * (0.4 + snapshot.contrast + beat_pulse)
        wobble = 0.8 + snapshot.percussive_flux * 1.4
        for idx, point in enumerate(layer_points):
            parity = idx % 2
            tangent = np.array(
                [
                    math.cos(direction_angle + idx * 0.14 * wobble),
                    math.sin(direction_angle - idx * 0.11 * wobble),
                ],
                dtype=np.float32,
            )
            direction = (base_direction * 0.35 + tangent * 0.65)
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            start = np.clip(point - direction * length * 0.5, [0, 0], [surface.get_width() - 1, surface.get_height() - 1]).astype(np.int32)
            end = np.clip(point + direction * length * 0.5, [0, 0], [surface.get_width() - 1, surface.get_height() - 1]).astype(np.int32)
            line_color = colors[idx].tolist()
            if parity == 1 and "color" in cell_alternation:
                line_color = [255 - c for c in line_color]
            pygame.draw.line(surface, line_color, start.tolist(), end.tolist(), 1 + layer_idx)


def _draw_geometrics(
    surface: pygame.Surface,
    layers: list[np.ndarray],
    style: ActiveStyle,
    snapshot: AudioSnapshot,
    style_mix: float,
    beat_pulse: float,
    time_phase: float,
    chroma_key_color: tuple[int, int, int] | None,
    cell_alternation: str = "none",
) -> None:
    center = np.array([surface.get_width() * 0.5, surface.get_height() * 0.5], dtype=np.float32)
    border_color = sanitize_rgb_triplet(style.border_color, chroma_key_color).tolist()
    for layer_idx, layer_points in enumerate(layers):
        phases = np.linspace(0.0, 1.0, len(layer_points), dtype=np.float32) + style_mix * 0.4 + layer_idx * 0.12
        colors = sanitize_rgb_array(build_palette(style, phases + snapshot.brightness * 0.2), chroma_key_color)
        for idx, point in enumerate(layer_points):
            parity = idx % 2
            delta = point - center
            angle = math.atan2(float(delta[1]), float(delta[0])) + time_phase * 0.3
            if parity == 1 and "orientation" in cell_alternation:
                angle += math.pi
            size = 10.0 + 28.0 * snapshot.bass + 24.0 * beat_pulse + (idx % 7)
            tri = np.array(
                [
                    point + np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * size,
                    point + np.array([math.cos(angle + 2.2), math.sin(angle + 2.2)], dtype=np.float32) * size * 0.75,
                    point + np.array([math.cos(angle - 2.2), math.sin(angle - 2.2)], dtype=np.float32) * size * 0.75,
                ],
                dtype=np.float32,
            )
            tri = np.clip(tri, [0, 0], [surface.get_width() - 1, surface.get_height() - 1]).astype(np.int32)
            fill_color = colors[idx].tolist()
            if parity == 1 and "color" in cell_alternation:
                fill_color = [255 - c for c in fill_color]
            pygame.draw.polygon(surface, fill_color, tri.tolist(), 0)
            pygame.draw.polygon(surface, border_color, tri.tolist(), 1)


def _draw_custom(
    surface: pygame.Surface,
    layers: list[np.ndarray],
    snapshot: AudioSnapshot,
    beat_pulse: float,
    time_phase: float,
    beat_index: int,
    beats_per_switch: int,
    custom_elements: list[pygame.Surface],
    pattern_layout: str,
    geometry_rigidity: float,
    tile_overlap: float,
    grid_cell_span: tuple[float, float] | None,
    grid_columns: int,
    cell_alternation: str,
    inverted_custom_elements: list[pygame.Surface],
    scale_cache: dict[tuple[int, int, int, bool], pygame.Surface],
    transform_cache: dict[tuple[int, int, int, int, bool, bool], pygame.Surface],
    fast_preview: bool,
) -> None:
    if not custom_elements:
        return
    count = max(1, len(layers))
    rigid_mix = np.clip(geometry_rigidity, 0.0, 1.0) if pattern_layout == "grid" else 0.0
    sprite_cycle_offset = 0 if len(custom_elements) <= 1 else (max(beat_index, 0) // max(beats_per_switch, 1)) % len(custom_elements)
    for layer_idx, layer_points in enumerate(layers):
        layer_weight = 0.0 if count == 1 else layer_idx / (count - 1)
        step = 1 if pattern_layout == "grid" else max(1, len(layer_points) // max(14, int(36 - tile_overlap * 20)))
        if pattern_layout == "grid" and grid_cell_span is not None:
            cell_span_x, cell_span_y = grid_cell_span
        else:
            cell_span_x, cell_span_y = estimate_grid_cell_span(layer_points, surface.get_width(), surface.get_height())
        fit_ratio = 0.92 + tile_overlap * 0.35
        target_w = max(8.0, cell_span_x * fit_ratio)
        target_h = max(8.0, cell_span_y * fit_ratio)
        for idx, point in enumerate(layer_points[::step]):
            point_index = idx * step
            sprite_index = (sprite_cycle_offset + layer_idx) % len(custom_elements)
            sprite = custom_elements[sprite_index]
            parity = 0
            if pattern_layout == "grid" and grid_columns > 0:
                row_idx = point_index // grid_columns
                col_idx = point_index % grid_columns
                parity = (row_idx + col_idx) % 2
            elif pattern_layout == "flow":
                parity = point_index % 2
            if pattern_layout == "grid":
                sprite_w = max(float(sprite.get_width()), 1.0)
                sprite_h = max(float(sprite.get_height()), 1.0)
                fit_scale = min(target_w / sprite_w, target_h / sprite_h)
                audio_breath = 1.0 + snapshot.bass * 0.10 + beat_pulse * 0.08
                scale = fit_scale * (0.96 + (audio_breath - 1.0) * (1.0 - rigid_mix * 0.35))
                width = sprite_w * scale
                height = sprite_h * scale
            else:
                scale = 0.65 + snapshot.bass * 1.0 + beat_pulse * 0.7 + layer_weight * 0.38 + tile_overlap * 0.30
                width = sprite.get_width() * scale
                height = sprite.get_height() * scale
            width = max(8, int(width))
            height = max(8, int(height))
            invert_color = parity == 1 and cell_alternation in {"color", "both"}
            flip_xy = parity == 1 and cell_alternation in {"orientation", "both"}
            angle = (time_phase * 32.0 + idx * 11.0 + layer_idx * 16.0) * (0.35 + snapshot.brightness * 0.5)
            if pattern_layout == "grid":
                angle *= 1.0 - rigid_mix
            rotated = get_transformed_custom_surface(
                sprite_index,
                width,
                height,
                angle,
                invert_color,
                flip_xy,
                custom_elements,
                inverted_custom_elements,
                scale_cache,
                transform_cache,
                fast_preview,
            )
            rect = rotated.get_rect(center=(int(point[0]), int(point[1])))
            surface.blit(rotated, rect)


def _draw_voronoi_layer(
    surface: pygame.Surface,
    points: np.ndarray,
    style: ActiveStyle,
    snapshot: AudioSnapshot,
    layer_idx: int,
    layer_count: int,
    overlap: float,
    style_mix: float,
    chroma_key_color: tuple[int, int, int] | None,
    cell_alternation: str = "none",
    voronoi_cache: dict[int, tuple[int, list[list[int]], np.ndarray]] | None = None,
    frame_idx: int = 0,
    fast_preview: bool = False,
) -> None:
    cached = voronoi_cache.get(layer_idx) if voronoi_cache is not None else None
    if fast_preview and cached is not None and frame_idx - cached[0] < 2:
        regions, vertices = cached[1], cached[2]
    else:
        ghost = np.array(
            [
                [-surface.get_width(), -surface.get_height()],
                [surface.get_width() * 2.0, -surface.get_height()],
                [-surface.get_width(), surface.get_height() * 2.0],
                [surface.get_width() * 2.0, surface.get_height() * 2.0],
            ],
            dtype=np.float64,
        )
        vor = Voronoi(np.vstack([points.astype(np.float64), ghost]))
        regions, vertices = voronoi_finite_polygons_2d(vor, radius=max(surface.get_size()) * 4.0)
        if voronoi_cache is not None:
            voronoi_cache[layer_idx] = (frame_idx, regions, vertices)

    phases = (
        np.arange(len(points), dtype=np.float32) / max(len(points), 1)
        + snapshot.brightness * 0.7
        + snapshot.contrast * 0.25
        + style_mix * 0.35
        + layer_idx * 0.13
    )
    colors = sanitize_rgb_array(build_palette(style, phases + snapshot.harmonic_ratio * 0.38), chroma_key_color)
    layer_weight = 0.0 if layer_count <= 1 else layer_idx / (layer_count - 1)
    scale = style.scale_bias + snapshot.bass * style.scale_response - layer_weight * overlap * 0.12
    echo_shift = sanitize_rgb_triplet(style.glow_color, chroma_key_color).astype(np.float32) * (0.14 + snapshot.percussive_flux * 0.12)
    fill_mix = 0.78 + 0.22 * layer_weight
    border_color = sanitize_rgb_triplet(style.border_color, chroma_key_color).tolist()
    glow_color = sanitize_rgb_triplet(style.glow_color, chroma_key_color).tolist()

    for idx, region in enumerate(regions[: len(points)]):
        polygon = clip_polygon(vertices[region], surface.get_width(), surface.get_height())
        if len(polygon) < 3:
            continue
        parity = idx % 2
        centroid = polygon.mean(axis=0, keepdims=True)
        polygon = centroid + (polygon - centroid) * scale
        if parity == 1 and "orientation" in cell_alternation:
            polygon = centroid - (polygon - centroid)
        poly_int = polygon.astype(np.int32)

        fill = np.clip(colors[idx] * fill_mix + echo_shift * (1.0 - fill_mix), 0, 255).astype(np.uint8)
        if parity == 1 and "color" in cell_alternation:
            fill = 255 - fill
        fill = sanitize_rgb_triplet(fill, chroma_key_color)
        if layer_idx > 0:
            echo_offset = np.array([layer_idx * overlap * 1.8, -layer_idx * overlap * 1.4], dtype=np.float64)
            shadow_poly = np.clip(polygon + echo_offset, [0, 0], [surface.get_width() - 1, surface.get_height() - 1]).astype(np.int32)
            pygame.draw.polygon(surface, glow_color, shadow_poly, 0)

        pygame.draw.polygon(surface, fill.tolist(), poly_int, 0)
        pygame.draw.polygon(surface, border_color, poly_int, style.outline_width)


def draw_graphic_family(
    surface: pygame.Surface,
    family: str,
    layers: list[np.ndarray],
    style: ActiveStyle,
    snapshot: AudioSnapshot,
    style_mix: float,
    beat_pulse: float,
    overlap: float,
    time_phase: float,
    beat_index: int,
    beats_per_switch: int,
    chroma_key_color: tuple[int, int, int] | None,
    custom_elements: list[pygame.Surface],
    pattern_layout: str,
    geometry_rigidity: float,
    tile_overlap: float,
    grid_cell_span: tuple[float, float] | None,
    grid_columns: int,
    cell_alternation: str,
    inverted_custom_elements: list[pygame.Surface],
    scale_cache: dict[tuple[int, int, int, bool], pygame.Surface],
    transform_cache: dict[tuple[int, int, int, int, bool, bool], pygame.Surface],
    voronoi_cache: dict[int, tuple[int, list[list[int]], np.ndarray]],
    frame_idx: int,
    fast_preview: bool,
) -> None:
    if family == "circles":
        _draw_circles(surface, layers, style, snapshot, style_mix, beat_pulse, chroma_key_color, cell_alternation)
        return
    if family == "scribbles":
        _draw_scribbles(surface, layers, style, snapshot, beat_pulse, time_phase, chroma_key_color, cell_alternation)
        return
    if family == "lines":
        _draw_lines(surface, layers, style, snapshot, beat_pulse, time_phase, chroma_key_color, cell_alternation)
        return
    if family == "geometrics":
        _draw_geometrics(surface, layers, style, snapshot, style_mix, beat_pulse, time_phase, chroma_key_color, cell_alternation)
        return
    if family == "custom":
        _draw_custom(
            surface,
            layers,
            snapshot,
            beat_pulse,
            time_phase,
            beat_index,
            beats_per_switch,
            custom_elements,
            pattern_layout,
            geometry_rigidity,
            tile_overlap,
            grid_cell_span,
            grid_columns,
            cell_alternation,
            inverted_custom_elements,
            scale_cache,
            transform_cache,
            fast_preview,
        )
        return

    for layer_idx, layer_points in enumerate(layers):
        _draw_voronoi_layer(
            surface,
            layer_points,
            style,
            snapshot,
            layer_idx,
            len(layers),
            overlap,
            style_mix,
            chroma_key_color,
            cell_alternation,
            voronoi_cache,
            frame_idx,
            fast_preview,
        )
