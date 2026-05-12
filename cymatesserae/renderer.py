from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter
from scipy.spatial import Voronoi


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


@dataclass(slots=True)
class AudioSnapshot:
    bass: float
    brightness: float
    contrast: float
    harmonic_ratio: float
    percussive_flux: float
    onset: float
    rms: float


class AudioFeatureTimeline:
    def __init__(self, y: np.ndarray, sr: int, features: dict[str, np.ndarray], bpm: float, beat_times: np.ndarray) -> None:
        self.y = y
        self.sr = sr
        self.features = features
        self.duration = len(y) / sr
        self.bpm = bpm
        self.beat_times = beat_times.astype(np.float32)

    def sample(self, key: str, frame_idx: int, total_frames: int) -> float:
        values = self.features[key]
        if len(values) == 1:
            return float(values[0])
        pos = frame_idx * (len(values) - 1) / max(total_frames - 1, 1)
        lo = int(np.floor(pos))
        hi = min(lo + 1, len(values) - 1)
        frac = pos - lo
        return float(values[lo] * (1.0 - frac) + values[hi] * frac)

    def beat_index_at_time(self, time_s: float) -> int:
        if len(self.beat_times) == 0:
            return 0
        return int(np.searchsorted(self.beat_times, time_s, side="right") - 1)

    def beat_pulse_at_time(self, time_s: float) -> float:
        if len(self.beat_times) == 0:
            return 0.0
        idx = int(np.clip(np.searchsorted(self.beat_times, time_s), 0, len(self.beat_times) - 1))
        candidates = [self.beat_times[idx]]
        if idx > 0:
            candidates.append(self.beat_times[idx - 1])
        distance = min(abs(time_s - beat_time) for beat_time in candidates)
        beat_period = 60.0 / max(self.bpm, 1.0)
        width = max(beat_period * 0.18, 0.04)
        return float(np.clip(1.0 - distance / width, 0.0, 1.0))


def _emit_status(message: str) -> None:
    print(f"[cymatesserae] {message}", flush=True)


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

GRAPHIC_TYPES = ("voronoi", "circles", "scribbles", "lines", "geometrics", "custom")
CUSTOM_ELEMENT_KEY = (255, 0, 255)


def _get_librosa():
    try:
        import librosa
    except Exception:
        return None
    return librosa


def _normalize(values: np.ndarray, power: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    floor = float(values.min())
    ceiling = float(values.max())
    span = ceiling - floor
    if span <= 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    norm = (values - floor) / span
    if power != 1.0:
        norm = np.power(norm, power)
    return norm.astype(np.float32)


def analyze_audio(audio_path: Path, hop_length: int, n_fft: int, duration_limit: float | None) -> AudioFeatureTimeline:
    y, sr = load_audio(audio_path, duration_limit)
    magnitude, freqs = compute_stft(y, sr, n_fft, hop_length)
    harmonic, percussive = compute_hpss(magnitude)

    contrast = spectral_contrast(magnitude, freqs)
    centroid = spectral_centroid(magnitude, freqs)
    rms = frame_rms(magnitude)
    harmonic_energy = harmonic.mean(axis=0)
    percussive_energy = percussive.mean(axis=0)

    bass_mask = freqs <= 180.0
    bass_energy = magnitude[bass_mask].mean(axis=0) if bass_mask.any() else rms.copy()
    brightness = centroid / max(sr / 2.0, 1.0)

    harmonic_ratio = harmonic_energy / np.maximum(harmonic_energy + percussive_energy, 1e-6)
    percussive_flux = np.diff(percussive_energy, prepend=percussive_energy[0])
    onset_env = onset_strength(percussive)
    bpm, beat_times = detect_bpm_and_beats(onset_env, sr, hop_length, len(y) / max(sr, 1))

    features = {
        "bass": _normalize(bass_energy, power=0.8),
        "brightness": _normalize(brightness),
        "contrast": _normalize(contrast.mean(axis=0)),
        "harmonic_ratio": _normalize(harmonic_ratio),
        "percussive": _normalize(percussive_energy),
        "percussive_flux": _normalize(np.maximum(percussive_flux, 0.0), power=0.7),
        "onset": _normalize(onset_env, power=0.8),
        "rms": _normalize(rms),
    }
    return AudioFeatureTimeline(y=y, sr=sr, features=features, bpm=bpm, beat_times=beat_times)


def load_audio(audio_path: Path, duration_limit: float | None) -> tuple[np.ndarray, int]:
    librosa = _get_librosa()
    if sys.version_info < (3, 13) and librosa is not None:
        y, sr = librosa.load(audio_path.as_posix(), sr=None, mono=True, duration=duration_limit)
        return y.astype(np.float32), int(sr)

    y, sr = sf.read(audio_path.as_posix(), always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1, dtype=np.float32)
    if duration_limit is not None:
        y = y[: int(sr * duration_limit)]
    return y, int(sr)


def compute_stft(y: np.ndarray, sr: int, n_fft: int, hop_length: int) -> tuple[np.ndarray, np.ndarray]:
    librosa = _get_librosa()
    if sys.version_info < (3, 13) and librosa is not None:
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft).astype(np.float32)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32)
        return magnitude, freqs

    freqs, _, stft = signal.stft(
        y,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    return np.abs(stft).astype(np.float32), freqs.astype(np.float32)


def compute_hpss(magnitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    librosa = _get_librosa()
    if sys.version_info < (3, 13) and librosa is not None:
        return librosa.decompose.hpss(magnitude)

    harm_med = median_filter(magnitude, size=(1, 17), mode="nearest")
    perc_med = median_filter(magnitude, size=(17, 1), mode="nearest")
    mask_h = harm_med >= perc_med
    harmonic = np.where(mask_h, magnitude, 0.35 * magnitude)
    percussive = np.where(mask_h, 0.35 * magnitude, magnitude)
    return harmonic.astype(np.float32), percussive.astype(np.float32)


def spectral_centroid(magnitude: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    weighted = (magnitude * freqs[:, None]).sum(axis=0)
    total = np.maximum(magnitude.sum(axis=0), 1e-6)
    return (weighted / total).astype(np.float32)


def frame_rms(magnitude: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(magnitude), axis=0)).astype(np.float32)


def spectral_contrast(magnitude: np.ndarray, freqs: np.ndarray, bands: int = 6) -> np.ndarray:
    nyquist = max(freqs[-1], 1.0)
    edges = np.geomspace(50.0, nyquist, num=bands + 2, dtype=np.float32)
    contrasts: list[np.ndarray] = []
    for low, high in zip(edges[:-1], edges[1:], strict=False):
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            continue
        band = magnitude[mask]
        peaks = np.percentile(band, 90, axis=0)
        valleys = np.percentile(band, 10, axis=0)
        contrasts.append(np.log1p(peaks) - np.log1p(valleys))
    if not contrasts:
        return np.zeros((1, magnitude.shape[1]), dtype=np.float32)
    return np.vstack(contrasts).astype(np.float32)


def onset_strength(percussive: np.ndarray) -> np.ndarray:
    log_spec = np.log1p(percussive)
    diff = np.diff(log_spec, axis=1, prepend=log_spec[:, :1])
    return np.maximum(diff, 0.0).mean(axis=0).astype(np.float32)


def detect_bpm_and_beats(onset_env: np.ndarray, sr: int, hop_length: int, duration: float) -> tuple[float, np.ndarray]:
    librosa = _get_librosa()
    if sys.version_info < (3, 13) and librosa is not None:
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        if len(beat_times) > 1:
            return float(tempo), beat_times.astype(np.float32)

    envelope = _normalize(onset_env, power=0.9)
    frame_rate = sr / max(hop_length, 1)
    min_distance = max(1, int(frame_rate * 60.0 / 180.0))
    height = max(0.18, float(envelope.mean() + envelope.std() * 0.35))
    peaks, _ = signal.find_peaks(envelope, distance=min_distance, height=height)

    if len(peaks) < 2:
        bpm = 120.0
        beat_period = 60.0 / bpm
        beat_times = np.arange(0.0, duration + beat_period, beat_period, dtype=np.float32)
        return bpm, beat_times

    beat_times = peaks.astype(np.float32) * (hop_length / sr)
    intervals = np.diff(beat_times)
    valid = intervals[(intervals > 0.25) & (intervals < 1.5)]
    if len(valid) == 0:
        bpm = 120.0
    else:
        bpm = float(np.clip(60.0 / np.median(valid), 60.0, 180.0))
    return bpm, beat_times


def _voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None) -> tuple[list[list[int]], np.ndarray]:
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


def _clip_polygon(points: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    return clipped


def _mix_arrays(a: np.ndarray, b: np.ndarray, mix: float) -> np.ndarray:
    return a * (1.0 - mix) + b * mix


def _resolve_style(name: str) -> StylePreset:
    return STYLE_PRESETS.get(name, STYLE_PRESETS["ceramic"])


def _resolve_graphic_cycle(cycle: tuple[str, ...]) -> tuple[str, ...]:
    resolved = tuple(item for item in cycle if item in GRAPHIC_TYPES)
    return resolved if resolved else ("voronoi",)


def _resolve_track_families(layer: GraphicLayerConfig, beat_index: int) -> tuple[str, ...]:
    graphic_cycle = _resolve_graphic_cycle(layer.graphic_cycle if layer.graphic_cycle else (layer.family,))
    if len(graphic_cycle) > 1:
        beats_per_switch = max(1, layer.beats_per_switch)
        active = graphic_cycle[(max(beat_index, 0) // beats_per_switch) % len(graphic_cycle)]
        return (active,)
    return (graphic_cycle[0],)


def _resolve_channel_stack_plan(
    runtimes: list[LayerRuntimeState],
    beat_index: int,
    time_phase: float,
    stack_interaction: str,
    snapshot: AudioSnapshot,
    beat_pulse: float,
) -> list[tuple[LayerRuntimeState, float]]:
    if not runtimes:
        return []
    mode = stack_interaction if stack_interaction in {"none", "crossfade", "shuffle", "pulse", "duck", "spotlight"} else "none"
    order = list(runtimes)
    if mode == "shuffle" and len(order) > 1:
        offset = max(beat_index, 0) % len(order)
        order = order[offset:] + order[:offset]
    if mode == "none" or len(order) == 1:
        return [(runtime, 1.0) for runtime in order]

    planned: list[tuple[LayerRuntimeState, float]] = []
    if mode == "crossfade":
        for idx, runtime in enumerate(order):
            phase = time_phase * 0.35 + idx * 0.22
            gain = 0.60 + 0.40 * (0.5 + 0.5 * math.sin(math.tau * phase))
            planned.append((runtime, float(np.clip(gain, 0.25, 1.0))))
        return planned

    if mode == "pulse":
        bass_drive = 0.35 + snapshot.bass * 0.45 + beat_pulse * 0.20
        for idx, runtime in enumerate(order):
            phase = time_phase * 0.45 + idx * 0.17
            wave = 0.5 + 0.5 * math.sin(math.tau * phase)
            gain = 0.45 + bass_drive * (0.45 + 0.55 * wave)
            planned.append((runtime, float(np.clip(gain, 0.25, 1.0))))
        return planned

    if mode == "duck":
        duck_amount = float(np.clip(0.25 + snapshot.percussive_flux * 0.55 + beat_pulse * 0.25, 0.0, 0.8))
        top_bias_start = max(0, len(order) - 1)
        for idx, runtime in enumerate(order):
            top_bias = 0.0 if top_bias_start == 0 else idx / top_bias_start
            gain = (1.0 - duck_amount) + top_bias * duck_amount
            planned.append((runtime, float(np.clip(gain, 0.20, 1.0))))
        return planned

    if mode == "spotlight":
        focus = max(beat_index, 0) % len(order)
        for idx, runtime in enumerate(order):
            circular_distance = min((idx - focus) % len(order), (focus - idx) % len(order))
            gain = 1.0 - circular_distance * (0.32 - snapshot.harmonic_ratio * 0.10 - beat_pulse * 0.08)
            if idx == focus:
                gain += 0.16 + snapshot.brightness * 0.10
            planned.append((runtime, float(np.clip(gain, 0.22, 1.0))))
        return planned

    return [(runtime, 1.0) for runtime in order]


def _apply_channel_composite(surface: pygame.Surface, opacity: float, transparent_colors: tuple[tuple[int, int, int], ...]) -> pygame.Surface:
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


def _compute_style_mix(frame_idx: int, total_frames: int, snapshot: AudioSnapshot, morph_rate: float) -> float:
    progress = frame_idx / max(total_frames - 1, 1)
    sweep = 0.5 + 0.5 * math.sin((progress * (1.0 + morph_rate * 3.0) + snapshot.brightness * 0.4) * math.tau)
    audio_bias = 0.35 * snapshot.contrast + 0.25 * snapshot.harmonic_ratio + 0.20 * snapshot.percussive_flux
    return float(np.clip(0.55 * sweep + audio_bias, 0.0, 1.0))


def _activate_style(style_a: StylePreset, style_b: StylePreset, mix: float) -> ActiveStyle:
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


def _build_palette(style: ActiveStyle, phase: np.ndarray) -> np.ndarray:
    rgb = style.palette_a + style.palette_b * np.cos(math.tau * (style.palette_c * phase[:, None] + style.palette_d))
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _sanitize_rgb_triplet(color: np.ndarray | list[int] | tuple[int, int, int], chroma_key_color: tuple[int, int, int] | None) -> np.ndarray:
    arr = np.asarray(color, dtype=np.uint8).copy()
    if chroma_key_color is None:
        return arr
    key = np.asarray(chroma_key_color, dtype=np.uint8)
    if np.array_equal(arr, key):
        arr[2] = np.uint8((int(arr[2]) + 1) % 256)
        if np.array_equal(arr, key):
            arr[1] = np.uint8((int(arr[1]) + 1) % 256)
    return arr


def _sanitize_rgb_array(colors: np.ndarray, chroma_key_color: tuple[int, int, int] | None) -> np.ndarray:
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


def _build_seed_points(count: int, width: int, height: int, rng: np.random.Generator) -> np.ndarray:
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


def _resolve_grid_dimensions(count: int, width: int, height: int, grid_columns: int = 0, grid_rows: int = 0) -> tuple[int, int, int]:
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


def _build_grid_points(
    count: int,
    width: int,
    height: int,
    overlap: float,
    grid_columns: int = 0,
    grid_rows: int = 0,
    grid_pattern: str = "rect",
) -> np.ndarray:
    cols, rows, count = _resolve_grid_dimensions(count, width, height, grid_columns, grid_rows)
    edge_x = 0.0 if cols <= 1 else 8.0 / max(width, 1)
    edge_y = 0.0 if rows <= 1 else 8.0 / max(height, 1)
    xs = np.linspace(edge_x, 1.0 - edge_x, cols, dtype=np.float32)
    ys = np.linspace(edge_y, 1.0 - edge_y, rows, dtype=np.float32)
    grid = np.array(np.meshgrid(xs, ys), dtype=np.float32).reshape(2, -1).T[:count]
    step_x = 0.0 if cols <= 1 else float(xs[1] - xs[0])
    step_y = 0.0 if rows <= 1 else float(ys[1] - ys[0])

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


def _cymatic_force(points: np.ndarray, width: int, height: int, mode: tuple[int, int], strength: float) -> np.ndarray:
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


def _frame_background(
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


def _prepare_layers(
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


def _estimate_grid_cell_span(points: np.ndarray, width: int, height: int) -> tuple[float, float]:
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


def _grid_cell_span_from_dimensions(width: int, height: int, cols: int, rows: int) -> tuple[float, float]:
    if cols <= 1:
        span_x = float(width - 16)
    else:
        span_x = float((width - 16) / max(cols - 1, 1))
    if rows <= 1:
        span_y = float(height - 16)
    else:
        span_y = float((height - 16) / max(rows - 1, 1))
    return max(span_x, 8.0), max(span_y, 8.0)


def _load_custom_elements(custom_element_paths: tuple[Path, ...]) -> list[pygame.Surface]:
    sprites: list[pygame.Surface] = []
    for custom_element_path in custom_element_paths:
        if not custom_element_path.exists():
            raise FileNotFoundError(f"Custom element file not found: {custom_element_path}")
        sprite = pygame.image.load(custom_element_path.as_posix()).convert()
        sprite.set_colorkey(CUSTOM_ELEMENT_KEY)
        sprites.append(sprite)
    return sprites


def _build_legacy_graphic_layers(config: RenderConfig) -> tuple[GraphicLayerConfig, ...]:
    layers: list[GraphicLayerConfig] = []
    for family in _resolve_graphic_cycle(config.graphic_cycle):
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


def _effective_graphic_layers(config: RenderConfig) -> tuple[GraphicLayerConfig, ...]:
    if config.graphic_layers:
        enabled = tuple(layer for layer in config.graphic_layers if layer.enabled and layer.family in GRAPHIC_TYPES)
        if enabled:
            return enabled
    return _build_legacy_graphic_layers(config)


def _build_layer_runtime(layer: GraphicLayerConfig, config: RenderConfig, seed: int) -> LayerRuntimeState:
    rng = np.random.default_rng(seed)
    is_grid_layout = layer.pattern_layout == "grid"
    if is_grid_layout:
        grid_cols, grid_rows, _ = _resolve_grid_dimensions(config.point_count, config.width, config.height, layer.grid_columns, layer.grid_rows)
        points = _build_grid_points(
            config.point_count,
            config.width,
            config.height,
            layer.tile_overlap,
            layer.grid_columns,
            layer.grid_rows,
            layer.grid_pattern,
        )
        grid_cell_span = _grid_cell_span_from_dimensions(config.width, config.height, grid_cols, grid_rows)
    else:
        grid_cols, grid_rows = 0, 0
        points = _build_seed_points(config.point_count, config.width, config.height, rng)
        grid_cell_span = None
    base_points = points.copy()
    velocity = np.zeros_like(points, dtype=np.float32)
    custom_elements = _load_custom_elements(layer.custom_element_paths)
    return LayerRuntimeState(
        layer=layer,
        points=points,
        base_points=base_points,
        velocity=velocity,
        rng=rng,
        previous_onset=0.0,
        style_a=_resolve_style(layer.style_a),
        style_b=_resolve_style(layer.style_b),
        custom_elements=custom_elements,
        inverted_custom_elements=[_invert_custom_surface(sprite) for sprite in custom_elements],
        custom_scale_cache={},
        custom_transform_cache={},
        voronoi_cache={},
        channel_surface=pygame.Surface((config.width, config.height), pygame.SRCALPHA, 32),
        is_grid_layout=is_grid_layout,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
        grid_cell_span=grid_cell_span,
    )


def _invert_custom_surface(surface: pygame.Surface) -> pygame.Surface:
    inverted = surface.copy()
    rgb = pygame.surfarray.array3d(inverted)
    key = np.asarray(CUSTOM_ELEMENT_KEY, dtype=np.uint8)
    mask = np.any(rgb != key[None, None, :], axis=2)
    rgb[mask] = 255 - rgb[mask]
    pygame.surfarray.blit_array(inverted, rgb)
    inverted.set_colorkey(CUSTOM_ELEMENT_KEY)
    return inverted


def _get_scaled_custom_surface(
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
    scaled = pygame.transform.scale(source_list[sprite_index], (width, height))
    if len(scale_cache) > 512:
        scale_cache.clear()
    scale_cache[key] = scaled
    return scaled


def _get_transformed_custom_surface(
    sprite_index: int,
    width: int,
    height: int,
    angle: float,
    inverted: bool,
    flip_xy: bool,
    custom_elements: list[pygame.Surface],
    inverted_custom_elements: list[pygame.Surface],
    scale_cache: dict[tuple[int, int, int, bool], pygame.Surface],
    transform_cache: dict[tuple[int, int, int, int, bool, bool], pygame.Surface],
    fast_preview: bool,
) -> pygame.Surface:
    scaled = _get_scaled_custom_surface(
        sprite_index,
        width,
        height,
        inverted,
        custom_elements,
        inverted_custom_elements,
        scale_cache,
    )
    if not fast_preview:
        transformed = pygame.transform.flip(scaled, True, True) if flip_xy else scaled
        return pygame.transform.rotate(transformed, angle)

    angle_bucket = int(round(angle / 12.0)) * 12
    key = (sprite_index, width, height, angle_bucket, inverted, flip_xy)
    cached = transform_cache.get(key)
    if cached is not None:
        return cached
    transformed = pygame.transform.flip(scaled, True, True) if flip_xy else scaled
    rotated = pygame.transform.rotate(transformed, float(angle_bucket))
    if len(transform_cache) > 1024:
        transform_cache.clear()
    transform_cache[key] = rotated
    return rotated


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
    border_color = _sanitize_rgb_triplet(style.border_color, chroma_key_color).tolist()
    for layer_idx, layer_points in enumerate(layers):
        layer_weight = 0.0 if count == 1 else layer_idx / (count - 1)
        phases = (
            np.arange(len(layer_points), dtype=np.float32) / max(len(layer_points), 1)
            + style_mix * 0.45
            + snapshot.brightness * 0.9
            + layer_idx * 0.11
        )
        colors = _sanitize_rgb_array(_build_palette(style, phases), chroma_key_color)
        radius = _circle_radius(style, snapshot, beat_pulse, layer_weight, surface.get_width())
        for idx, point in enumerate(layer_points):
            parity = idx % 2  # Alternate based on point index
            center = (int(point[0]), int(point[1]))
            fill = tuple(int(v) for v in colors[idx])
            if parity == 1 and "color" in cell_alternation:
                fill = tuple(255 - c for c in fill)  # Invert color
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
            parity = offset % 2  # Alternate based on strand offset
            jitter = np.empty_like(strand)
            phase = time_phase * 2.0 + offset * 0.7 + layer_idx * 0.5
            jitter[:, 0] = np.sin(np.arange(len(strand), dtype=np.float32) * 0.9 + phase) * jitter_amp
            jitter[:, 1] = np.cos(np.arange(len(strand), dtype=np.float32) * 0.8 - phase) * jitter_amp
            squiggle = np.clip(strand + jitter, [0, 0], [surface.get_width() - 1, surface.get_height() - 1]).astype(np.int32)
            color_phase = np.linspace(0.0, 1.0, len(squiggle), dtype=np.float32)
            color = _sanitize_rgb_triplet(
                _build_palette(style, color_phase + snapshot.harmonic_ratio + offset * 0.08)[0],
                chroma_key_color,
            )
            if parity == 1 and "color" in cell_alternation:
                color = tuple(255 - c for c in color)  # Invert color
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
        colors = _sanitize_rgb_array(_build_palette(style, phases), chroma_key_color)
        length = 24.0 + surface.get_width() * 0.04 * (0.4 + snapshot.contrast + beat_pulse)
        wobble = 0.8 + snapshot.percussive_flux * 1.4
        for idx, point in enumerate(layer_points):
            parity = idx % 2  # Alternate based on point index
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
                line_color = [255 - c for c in line_color]  # Invert color
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
    border_color = _sanitize_rgb_triplet(style.border_color, chroma_key_color).tolist()
    for layer_idx, layer_points in enumerate(layers):
        phases = np.linspace(0.0, 1.0, len(layer_points), dtype=np.float32) + style_mix * 0.4 + layer_idx * 0.12
        colors = _sanitize_rgb_array(_build_palette(style, phases + snapshot.brightness * 0.2), chroma_key_color)
        for idx, point in enumerate(layer_points):
            parity = idx % 2  # Alternate based on point index
            delta = point - center
            angle = math.atan2(float(delta[1]), float(delta[0])) + time_phase * 0.3
            if parity == 1 and "orientation" in cell_alternation:
                angle += math.pi  # Rotate 180 degrees
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
                fill_color = [255 - c for c in fill_color]  # Invert color
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
            cell_span_x, cell_span_y = _estimate_grid_cell_span(layer_points, surface.get_width(), surface.get_height())
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
            rotated = _get_transformed_custom_surface(
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
        regions, vertices = _voronoi_finite_polygons_2d(vor, radius=max(surface.get_size()) * 4.0)
        if voronoi_cache is not None:
            voronoi_cache[layer_idx] = (frame_idx, regions, vertices)

    phases = (
        np.arange(len(points), dtype=np.float32) / max(len(points), 1)
        + snapshot.brightness * 0.7
        + snapshot.contrast * 0.25
        + style_mix * 0.35
        + layer_idx * 0.13
    )
    colors = _sanitize_rgb_array(_build_palette(style, phases + snapshot.harmonic_ratio * 0.38), chroma_key_color)
    layer_weight = 0.0 if layer_count <= 1 else layer_idx / (layer_count - 1)
    scale = style.scale_bias + snapshot.bass * style.scale_response - layer_weight * overlap * 0.12
    echo_shift = _sanitize_rgb_triplet(style.glow_color, chroma_key_color).astype(np.float32) * (0.14 + snapshot.percussive_flux * 0.12)
    fill_mix = 0.78 + 0.22 * layer_weight
    border_color = _sanitize_rgb_triplet(style.border_color, chroma_key_color).tolist()
    glow_color = _sanitize_rgb_triplet(style.glow_color, chroma_key_color).tolist()

    for idx, region in enumerate(regions[: len(points)]):
        polygon = _clip_polygon(vertices[region], surface.get_width(), surface.get_height())
        if len(polygon) < 3:
            continue
        parity = idx % 2  # Alternate based on region index
        centroid = polygon.mean(axis=0, keepdims=True)
        polygon = centroid + (polygon - centroid) * scale
        if parity == 1 and "orientation" in cell_alternation:
            # Flip polygon around centroid
            polygon = centroid - (polygon - centroid)
        poly_int = polygon.astype(np.int32)

        fill = np.clip(colors[idx] * fill_mix + echo_shift * (1.0 - fill_mix), 0, 255).astype(np.uint8)
        if parity == 1 and "color" in cell_alternation:
            fill = 255 - fill  # Invert color
        fill = _sanitize_rgb_triplet(fill, chroma_key_color)
        if layer_idx > 0:
            echo_offset = np.array([layer_idx * overlap * 1.8, -layer_idx * overlap * 1.4], dtype=np.float64)
            shadow_poly = np.clip(polygon + echo_offset, [0, 0], [surface.get_width() - 1, surface.get_height() - 1]).astype(np.int32)
            pygame.draw.polygon(surface, glow_color, shadow_poly, 0)

        pygame.draw.polygon(surface, fill.tolist(), poly_int, 0)
        pygame.draw.polygon(surface, border_color, poly_int, style.outline_width)


def _draw_graphic_family(
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


def _surface_to_frame(surface: pygame.Surface) -> np.ndarray:
    rgb = pygame.surfarray.array3d(surface)
    return np.transpose(rgb, (1, 0, 2)).copy()


def _open_ffmpeg(output_path: Path, audio_path: Path, width: int, height: int, fps: int) -> subprocess.Popen[bytes]:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        audio_path.as_posix(),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        output_path.as_posix(),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _sample_snapshot(timeline: AudioFeatureTimeline, frame_idx: int, total_frames: int) -> AudioSnapshot:
    return AudioSnapshot(
        bass=timeline.sample("bass", frame_idx, total_frames),
        brightness=timeline.sample("brightness", frame_idx, total_frames),
        contrast=timeline.sample("contrast", frame_idx, total_frames),
        harmonic_ratio=timeline.sample("harmonic_ratio", frame_idx, total_frames),
        percussive_flux=timeline.sample("percussive_flux", frame_idx, total_frames),
        onset=timeline.sample("onset", frame_idx, total_frames),
        rms=timeline.sample("rms", frame_idx, total_frames),
    )


def _rebuild_bases(config: RenderConfig, reorg_mode: str, rng: np.random.Generator, points: np.ndarray, base_points: np.ndarray) -> np.ndarray:
    if reorg_mode == "split":
        direction = np.sign(points[:, 0] - config.width * 0.5).astype(np.float32)
        direction[direction == 0.0] = 1.0
        rebuilt = base_points.copy()
        rebuilt[:, 0] += direction * config.width * 0.08
        rebuilt[:, 1] += rng.normal(0.0, config.height * 0.03, size=len(rebuilt)).astype(np.float32)
        return np.clip(rebuilt, [8.0, 8.0], [config.width - 8.0, config.height - 8.0])
    return _build_seed_points(config.point_count, config.width, config.height, rng)


def _reorg_impulse(
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


def render_project(config: RenderConfig) -> None:
    if not config.audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {config.audio_path}")

    preview_mode = bool(config.preview or config.preview_only)
    mode_label = "live preview" if config.preview_only else ("preview render" if config.preview else "export render")
    _emit_status(f"Starting {mode_label} for {config.audio_path.name}.")
    _emit_status(
        f"Video setup: {config.width}x{config.height} at {config.fps} fps, {config.point_count} points, "
        f"{'cymatic on' if config.cymatic_mode else 'cymatic off'}."
    )
    _emit_status("Analyzing audio and building feature timeline...")
    timeline = analyze_audio(
        audio_path=config.audio_path,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        duration_limit=config.duration_limit,
    )
    total_frames = max(1, math.ceil(timeline.duration * config.fps))
    _emit_status(
        f"Audio analysis complete: {timeline.duration:.2f}s, about {total_frames} frames, estimated tempo {timeline.bpm:.1f} BPM."
    )

    pygame.init()
    flags = 0 if preview_mode else pygame.HIDDEN
    screen = pygame.display.set_mode((config.width, config.height), flags)
    surface = pygame.Surface((config.width, config.height))
    clock = pygame.time.Clock()
    frame_center = np.array([config.width * 0.5, config.height * 0.5], dtype=np.float32)

    graphic_layers = _effective_graphic_layers(config)
    runtimes = [_build_layer_runtime(layer, config, config.seed + idx * 997) for idx, layer in enumerate(graphic_layers)]
    _emit_status(f"Prepared {len(runtimes)} active channel(s) for rendering.")

    ffmpeg: subprocess.Popen[bytes] | None = None
    if not config.preview_only:
        _emit_status(f"Opening FFmpeg export pipeline for {config.output_path.name}...")
        ffmpeg = _open_ffmpeg(config.output_path, config.audio_path, config.width, config.height, config.fps)
        if ffmpeg.stdin is None:
            raise RuntimeError("Failed to open FFmpeg stdin.")
    else:
        _emit_status("Preview-only mode active: skipping FFmpeg export for a lighter live run.")

    try:
        for frame_idx in range(total_frames):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    _emit_status("Preview window closed by user.")
                    return

            if frame_idx == 0:
                _emit_status("Entering frame loop.")
            elif frame_idx % max(config.fps * 5, 1) == 0:
                elapsed_s = frame_idx / max(config.fps, 1)
                progress = (frame_idx + 1) / max(total_frames, 1) * 100.0
                _emit_status(f"Rendering frame {frame_idx + 1}/{total_frames} ({progress:.1f}%, t={elapsed_s:.1f}s).")

            snapshot = _sample_snapshot(timeline, frame_idx, total_frames)
            time_phase = frame_idx / max(config.fps, 1)
            beat_index = timeline.beat_index_at_time(time_phase)
            base_beat_pulse = timeline.beat_pulse_at_time(time_phase)
            if not runtimes:
                continue
            channel_plan = _resolve_channel_stack_plan(
                list(runtimes),
                beat_index,
                time_phase,
                config.stack_interaction,
                snapshot,
                base_beat_pulse,
            )

            first_runtime = runtimes[0]
            first_layer = first_runtime.layer
            first_style_mix = _compute_style_mix(frame_idx, total_frames, snapshot, first_layer.morph_rate)
            first_style = _activate_style(first_runtime.style_a, first_runtime.style_b, first_style_mix)
            _frame_background(surface, first_style, snapshot, first_style_mix, time_phase, config.chroma_key_color)

            for runtime, channel_gain in channel_plan:
                layer = runtime.layer
                style_mix = _compute_style_mix(frame_idx, total_frames, snapshot, layer.morph_rate)
                style = _activate_style(runtime.style_a, runtime.style_b, style_mix)
                response_gain = max(0.1, float(layer.response_gain))
                beat_pulse = np.clip(base_beat_pulse * response_gain, 0.0, 1.6)
                active_families = _resolve_track_families(layer, beat_index)

                onset_threshold = max(0.18, 0.62 / response_gain)
                onset_delta = max(0.03, 0.08 / response_gain)
                trigger = snapshot.onset > onset_threshold and (snapshot.onset - runtime.previous_onset) > onset_delta
                runtime.previous_onset = snapshot.onset

                points = runtime.points
                base_points = runtime.base_points
                velocity = runtime.velocity
                is_grid_layout = runtime.is_grid_layout
                geometry_rigidity = float(np.clip(layer.geometry_rigidity, 0.0, 1.0))
                layer_rigidity = float(np.clip(layer.layer_rigidity, 0.0, 1.0))

                drift = (base_points - points) * (0.010 + snapshot.harmonic_ratio * 0.026 * response_gain + layer.overlap * 0.008)
                velocity += drift

                if config.cymatic_mode:
                    cymatic_strength = (0.010 + snapshot.contrast * 0.022 + style_mix * 0.010) * response_gain
                    velocity += _cymatic_force(points, config.width, config.height, config.plate_mode, cymatic_strength)

                if trigger and not is_grid_layout:
                    layer_rng = runtime.rng
                    velocity += _reorg_impulse(config, layer.reorg_mode, layer_rng, points, snapshot)
                    base_points = _rebuild_bases(config, layer.reorg_mode, layer_rng, points, base_points)
                    runtime.base_points = base_points

                delta = points - frame_center
                swirl = np.empty_like(points, dtype=np.float32)
                swirl[:, 0] = -delta[:, 1]
                swirl[:, 1] = delta[:, 0]
                swirl /= np.maximum(np.linalg.norm(swirl, axis=1, keepdims=True), 1.0)

                drift_wave = np.empty_like(points, dtype=np.float32)
                drift_wave[:, 0] = np.sin(points[:, 1] * 0.012 + time_phase * 1.8)
                drift_wave[:, 1] = np.cos(points[:, 0] * 0.012 - time_phase * 1.5)

                if is_grid_layout:
                    grid_pull = np.clip(layer.grid_strength, 0.0, 1.0)
                    rigidity_pull = 0.10 + geometry_rigidity * 0.50 + grid_pull * 0.22 + layer.tile_overlap * 0.04
                    velocity += (base_points - points) * rigidity_pull
                    velocity += swirl * (0.004 + snapshot.brightness * 0.012 + style.swirl_bias * 0.08) * (1.0 - geometry_rigidity) * response_gain
                    velocity += drift_wave * (0.008 + layer.tile_overlap * 0.020 + snapshot.contrast * 0.015) * (1.0 - geometry_rigidity) * response_gain
                    velocity *= max(0.12, 0.58 - snapshot.rms * 0.04 - grid_pull * 0.16 - geometry_rigidity * 0.28)
                else:
                    velocity += swirl * (0.03 + snapshot.brightness * 0.12 + style.swirl_bias + snapshot.harmonic_ratio * 0.08) * response_gain
                    velocity += drift_wave * (layer.overlap * 0.10 + snapshot.contrast * 0.06) * response_gain
                    velocity *= max(0.25, 0.90 - snapshot.rms * 0.08)

                points += velocity
                if is_grid_layout:
                    snap_mix = geometry_rigidity
                    if snap_mix > 0.0:
                        points = base_points * snap_mix + points * (1.0 - snap_mix)
                    if geometry_rigidity >= 0.995:
                        points = base_points.copy()
                        velocity.fill(0.0)
                points[:, 0] = np.clip(points[:, 0], 8.0, config.width - 8.0)
                points[:, 1] = np.clip(points[:, 1], 8.0, config.height - 8.0)

                runtime.points = points
                runtime.velocity = velocity

                family_overlap = layer.overlap + beat_pulse * 0.18 + layer.tile_overlap * 0.10
                layers = _prepare_layers(
                    points,
                    style,
                    snapshot,
                    layer.layer_count,
                    family_overlap,
                    time_phase,
                    layer.pattern_layout,
                    layer_rigidity,
                )
                channel_surface = runtime.channel_surface
                channel_surface.fill((0, 0, 0, 0))
                for active_family in active_families:
                    _draw_graphic_family(
                        channel_surface,
                        active_family,
                        layers,
                        style,
                        snapshot,
                        style_mix,
                        beat_pulse,
                        family_overlap,
                        time_phase,
                        beat_index,
                        layer.beats_per_switch,
                        config.chroma_key_color,
                        runtime.custom_elements,
                        layer.pattern_layout,
                        geometry_rigidity,
                        layer.tile_overlap,
                        runtime.grid_cell_span,
                        runtime.grid_cols,
                        layer.cell_alternation,
                        runtime.inverted_custom_elements,
                        runtime.custom_scale_cache,
                        runtime.custom_transform_cache,
                        runtime.voronoi_cache,
                        frame_idx,
                        config.preview_only,
                    )
                surface.blit(
                    _apply_channel_composite(
                        channel_surface,
                        float(np.clip(layer.opacity * channel_gain, 0.0, 1.0)),
                        layer.transparent_colors,
                    ),
                    (0, 0),
                )

            if preview_mode:
                screen.blit(surface, (0, 0))
                pygame.display.flip()
                clock.tick(config.fps)

            if ffmpeg is not None and ffmpeg.stdin is not None:
                ffmpeg.stdin.write(_surface_to_frame(surface).tobytes())
    finally:
        code = 0
        if ffmpeg is not None:
            if ffmpeg.stdin is not None:
                ffmpeg.stdin.close()
            code = ffmpeg.wait()
        pygame.quit()
        if code != 0:
            raise RuntimeError(f"FFmpeg exited with status {code}.")
        _emit_status("Render session finished cleanly.")
