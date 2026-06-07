from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pygame

from .audio_analysis import AudioSnapshot


@dataclass(slots=True)
class StylePreset:
    name: str
    palette_a: np.ndarray
    palette_b: np.ndarray
    palette_c: np.ndarray
    palette_d: np.ndarray
    background_lo: np.ndarray
    background_hi: np.ndarray
    border_color: np.ndarray
    glow_color: np.ndarray
    scale_bias: float
    scale_response: float
    swirl_bias: float
    outline_width: int
    layer_spread: float
    offset_gain: float
    pulse_gain: float


@dataclass(slots=True)
class ActiveStyle:
    palette_a: np.ndarray
    palette_b: np.ndarray
    palette_c: np.ndarray
    palette_d: np.ndarray
    background_lo: np.ndarray
    background_hi: np.ndarray
    border_color: np.ndarray
    glow_color: np.ndarray
    scale_bias: float
    scale_response: float
    swirl_bias: float
    outline_width: int
    layer_spread: float
    offset_gain: float
    pulse_gain: float


STYLE_PRESETS: dict[str, StylePreset] = {
    "ceramic": StylePreset(
        name="ceramic",
        palette_a=np.array([0.57, 0.54, 0.50], dtype=np.float32),
        palette_b=np.array([0.22, 0.18, 0.16], dtype=np.float32),
        palette_c=np.array([0.92, 0.73, 0.58], dtype=np.float32),
        palette_d=np.array([0.09, 0.14, 0.22], dtype=np.float32),
        background_lo=np.array([18.0, 22.0, 28.0], dtype=np.float32),
        background_hi=np.array([178.0, 144.0, 118.0], dtype=np.float32),
        border_color=np.array([238.0, 232.0, 220.0], dtype=np.float32),
        glow_color=np.array([196.0, 162.0, 126.0], dtype=np.float32),
        scale_bias=0.76,
        scale_response=0.54,
        swirl_bias=0.10,
        outline_width=2,
        layer_spread=18.0,
        offset_gain=0.7,
        pulse_gain=0.5,
    ),
    "neon": StylePreset(
        name="neon",
        palette_a=np.array([0.45, 0.48, 0.54], dtype=np.float32),
        palette_b=np.array([0.55, 0.48, 0.42], dtype=np.float32),
        palette_c=np.array([1.10, 1.28, 1.40], dtype=np.float32),
        palette_d=np.array([0.02, 0.33, 0.58], dtype=np.float32),
        background_lo=np.array([6.0, 10.0, 18.0], dtype=np.float32),
        background_hi=np.array([46.0, 74.0, 128.0], dtype=np.float32),
        border_color=np.array([244.0, 246.0, 255.0], dtype=np.float32),
        glow_color=np.array([94.0, 220.0, 255.0], dtype=np.float32),
        scale_bias=0.70,
        scale_response=0.72,
        swirl_bias=0.22,
        outline_width=1,
        layer_spread=28.0,
        offset_gain=1.2,
        pulse_gain=0.9,
    ),
    "lava": StylePreset(
        name="lava",
        palette_a=np.array([0.62, 0.30, 0.16], dtype=np.float32),
        palette_b=np.array([0.34, 0.24, 0.14], dtype=np.float32),
        palette_c=np.array([0.98, 0.90, 0.68], dtype=np.float32),
        palette_d=np.array([0.00, 0.10, 0.22], dtype=np.float32),
        background_lo=np.array([16.0, 8.0, 5.0], dtype=np.float32),
        background_hi=np.array([204.0, 92.0, 36.0], dtype=np.float32),
        border_color=np.array([255.0, 230.0, 192.0], dtype=np.float32),
        glow_color=np.array([255.0, 120.0, 24.0], dtype=np.float32),
        scale_bias=0.82,
        scale_response=0.82,
        swirl_bias=0.14,
        outline_width=2,
        layer_spread=24.0,
        offset_gain=1.0,
        pulse_gain=1.1,
    ),
    "glass": StylePreset(
        name="glass",
        palette_a=np.array([0.68, 0.76, 0.82], dtype=np.float32),
        palette_b=np.array([0.20, 0.24, 0.26], dtype=np.float32),
        palette_c=np.array([0.84, 1.06, 1.12], dtype=np.float32),
        palette_d=np.array([0.10, 0.26, 0.42], dtype=np.float32),
        background_lo=np.array([12.0, 24.0, 34.0], dtype=np.float32),
        background_hi=np.array([122.0, 176.0, 204.0], dtype=np.float32),
        border_color=np.array([245.0, 252.0, 255.0], dtype=np.float32),
        glow_color=np.array([126.0, 198.0, 222.0], dtype=np.float32),
        scale_bias=0.74,
        scale_response=0.50,
        swirl_bias=0.18,
        outline_width=1,
        layer_spread=14.0,
        offset_gain=1.1,
        pulse_gain=0.45,
    ),
    "monolith": StylePreset(
        name="monolith",
        palette_a=np.array([0.42, 0.42, 0.42], dtype=np.float32),
        palette_b=np.array([0.24, 0.24, 0.24], dtype=np.float32),
        palette_c=np.array([1.00, 1.00, 1.00], dtype=np.float32),
        palette_d=np.array([0.02, 0.03, 0.05], dtype=np.float32),
        background_lo=np.array([10.0, 10.0, 12.0], dtype=np.float32),
        background_hi=np.array([164.0, 168.0, 176.0], dtype=np.float32),
        border_color=np.array([236.0, 236.0, 236.0], dtype=np.float32),
        glow_color=np.array([166.0, 166.0, 170.0], dtype=np.float32),
        scale_bias=0.78,
        scale_response=0.42,
        swirl_bias=0.08,
        outline_width=2,
        layer_spread=12.0,
        offset_gain=0.55,
        pulse_gain=0.35,
    ),
}


def _mix_arrays(a: np.ndarray, b: np.ndarray, mix: float) -> np.ndarray:
    return a * (1.0 - mix) + b * mix


def resolve_style(name: str) -> StylePreset:
    return STYLE_PRESETS.get(name, STYLE_PRESETS["ceramic"])


def compute_style_mix(frame_idx: int, total_frames: int, snapshot: AudioSnapshot, morph_rate: float) -> float:
    progress = frame_idx / max(total_frames - 1, 1)
    sweep = 0.5 + 0.5 * math.sin((progress * (1.0 + morph_rate * 3.0) + snapshot.brightness * 0.4) * math.tau)
    audio_bias = 0.35 * snapshot.contrast + 0.25 * snapshot.harmonic_ratio + 0.20 * snapshot.percussive_flux
    return float(np.clip(0.55 * sweep + audio_bias, 0.0, 1.0))


def activate_style(style_a: StylePreset, style_b: StylePreset, mix: float) -> ActiveStyle:
    return ActiveStyle(
        palette_a=_mix_arrays(style_a.palette_a, style_b.palette_a, mix),
        palette_b=_mix_arrays(style_a.palette_b, style_b.palette_b, mix),
        palette_c=_mix_arrays(style_a.palette_c, style_b.palette_c, mix),
        palette_d=_mix_arrays(style_a.palette_d, style_b.palette_d, mix),
        background_lo=_mix_arrays(style_a.background_lo, style_b.background_lo, mix),
        background_hi=_mix_arrays(style_a.background_hi, style_b.background_hi, mix),
        border_color=_mix_arrays(style_a.border_color, style_b.border_color, mix),
        glow_color=_mix_arrays(style_a.glow_color, style_b.glow_color, mix),
        scale_bias=float((1.0 - mix) * style_a.scale_bias + mix * style_b.scale_bias),
        scale_response=float((1.0 - mix) * style_a.scale_response + mix * style_b.scale_response),
        swirl_bias=float((1.0 - mix) * style_a.swirl_bias + mix * style_b.swirl_bias),
        outline_width=max(1, int(round((1.0 - mix) * style_a.outline_width + mix * style_b.outline_width))),
        layer_spread=float((1.0 - mix) * style_a.layer_spread + mix * style_b.layer_spread),
        offset_gain=float((1.0 - mix) * style_a.offset_gain + mix * style_b.offset_gain),
        pulse_gain=float((1.0 - mix) * style_a.pulse_gain + mix * style_b.pulse_gain),
    )


def build_palette(style: ActiveStyle, phase: np.ndarray) -> np.ndarray:
    rgb = style.palette_a + style.palette_b * np.cos(math.tau * (style.palette_c * phase[:, None] + style.palette_d))
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def sanitize_rgb_triplet(color: np.ndarray | list[int] | tuple[int, int, int], chroma_key_color: tuple[int, int, int] | None) -> np.ndarray:
    arr = np.asarray(color, dtype=np.uint8).copy()
    if chroma_key_color is None:
        return arr
    key = np.asarray(chroma_key_color, dtype=np.uint8)
    if np.array_equal(arr, key):
        arr[2] = np.uint8((int(arr[2]) + 1) % 256)
        if np.array_equal(arr, key):
            arr[1] = np.uint8((int(arr[1]) + 1) % 256)
    return arr


def sanitize_rgb_array(colors: np.ndarray, chroma_key_color: tuple[int, int, int] | None) -> np.ndarray:
    arr = np.asarray(colors, dtype=np.uint8).copy()
    if chroma_key_color is None or arr.size == 0:
        return arr
    key = np.asarray(chroma_key_color, dtype=np.uint8)
    matches = np.all(arr == key, axis=1)
    if np.any(matches):
        arr[matches, 2] = (arr[matches, 2].astype(np.uint16) + 1) % 256
        second_matches = np.all(arr == key, axis=1)
        if np.any(second_matches):
            arr[second_matches, 1] = (arr[second_matches, 1].astype(np.uint16) + 1) % 256
        arr = arr.astype(np.uint8)
    return arr


def frame_background(
    surface: pygame.Surface,
    style: ActiveStyle,
    snapshot: AudioSnapshot,
    style_mix: float,
    time_phase: float,
    chroma_key_color: tuple[int, int, int] | None,
) -> None:
    if chroma_key_color is not None:
        surface.fill(chroma_key_color)
        return
    width, height = surface.get_size()
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    wave_a = 0.5 + 0.5 * np.sin((x * (3.4 + style_mix * 1.6) + y * 2.2 + time_phase * 0.14) * math.pi)
    wave_b = 0.5 + 0.5 * np.cos((x * 1.8 - y * (4.2 + snapshot.contrast * 1.8) + time_phase * 0.09) * math.pi)
    gradient = np.clip(0.45 * wave_a + 0.55 * wave_b + snapshot.brightness * 0.24, 0.0, 1.0)
    base = style.background_lo + (style.background_hi - style.background_lo) * gradient[..., None]
    shimmer = style.glow_color * (0.08 + 0.10 * snapshot.harmonic_ratio)
    rgb = np.clip(base + shimmer * wave_a[..., None], 0, 255).astype(np.uint8)
    pygame.surfarray.blit_array(surface, np.transpose(rgb, (1, 0, 2)))
