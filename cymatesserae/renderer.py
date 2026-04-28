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
    reorg_mode: str = "burst"
    graphic_cycle: tuple[str, ...] = ("voronoi", "circles", "scribbles", "lines", "geometrics")
    beats_per_switch: int = 4
    chroma_key_color: tuple[int, int, int] | None = None


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

GRAPHIC_TYPES = ("voronoi", "circles", "scribbles", "lines", "geometrics")


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
) -> list[np.ndarray]:
    layers: list[np.ndarray] = []
    center = np.array([points[:, 0].mean(), points[:, 1].mean()], dtype=np.float32)
    delta = points - center
    count = max(1, layer_count)
    for idx in range(count):
        layer_t = 0.0 if count == 1 else idx / (count - 1)
        offset_angle = time_phase * 0.045 + layer_t * math.tau * 0.6 + snapshot.brightness * 1.6
        offset_vec = np.array([math.cos(offset_angle), math.sin(offset_angle)], dtype=np.float32)
        offset_mag = style.layer_spread * overlap * (0.4 + layer_t) * (0.8 + snapshot.percussive_flux * 0.9)
        pulse = 1.0 + (layer_t - 0.5) * style.pulse_gain * snapshot.bass * 0.22
        shear = np.empty_like(points, dtype=np.float32)
        shear[:, 0] = delta[:, 0] + delta[:, 1] * 0.08 * overlap * (idx + 1)
        shear[:, 1] = delta[:, 1] - delta[:, 0] * 0.06 * overlap * (idx + 1)
        layer_points = center + shear * pulse
        layer_points += offset_vec * offset_mag * style.offset_gain
        layers.append(layer_points.astype(np.float32))
    return layers


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
            center = (int(point[0]), int(point[1]))
            fill = tuple(int(v) for v in colors[idx])
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
            pygame.draw.lines(surface, color.tolist(), False, squiggle.tolist(), 2 + (offset % 2))


def _draw_lines(
    surface: pygame.Surface,
    layers: list[np.ndarray],
    style: ActiveStyle,
    snapshot: AudioSnapshot,
    beat_pulse: float,
    time_phase: float,
    chroma_key_color: tuple[int, int, int] | None,
) -> None:
    direction_angle = time_phase * 0.9 + snapshot.brightness * math.tau + snapshot.contrast * 1.4
    base_direction = np.array([math.cos(direction_angle), math.sin(direction_angle)], dtype=np.float32)
    for layer_idx, layer_points in enumerate(layers):
        phases = np.linspace(0.0, 1.0, len(layer_points), dtype=np.float32) + layer_idx * 0.16 + snapshot.bass * 0.25
        colors = _sanitize_rgb_array(_build_palette(style, phases), chroma_key_color)
        length = 24.0 + surface.get_width() * 0.04 * (0.4 + snapshot.contrast + beat_pulse)
        wobble = 0.8 + snapshot.percussive_flux * 1.4
        for idx, point in enumerate(layer_points):
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
            pygame.draw.line(surface, colors[idx].tolist(), start.tolist(), end.tolist(), 1 + layer_idx)


def _draw_geometrics(
    surface: pygame.Surface,
    layers: list[np.ndarray],
    style: ActiveStyle,
    snapshot: AudioSnapshot,
    style_mix: float,
    beat_pulse: float,
    time_phase: float,
    chroma_key_color: tuple[int, int, int] | None,
) -> None:
    center = np.array([surface.get_width() * 0.5, surface.get_height() * 0.5], dtype=np.float32)
    border_color = _sanitize_rgb_triplet(style.border_color, chroma_key_color).tolist()
    for layer_idx, layer_points in enumerate(layers):
        phases = np.linspace(0.0, 1.0, len(layer_points), dtype=np.float32) + style_mix * 0.4 + layer_idx * 0.12
        colors = _sanitize_rgb_array(_build_palette(style, phases + snapshot.brightness * 0.2), chroma_key_color)
        for idx, point in enumerate(layer_points):
            delta = point - center
            angle = math.atan2(float(delta[1]), float(delta[0])) + time_phase * 0.3
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
            pygame.draw.polygon(surface, colors[idx].tolist(), tri.tolist(), 0)
            pygame.draw.polygon(surface, border_color, tri.tolist(), 1)


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
) -> None:
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
        centroid = polygon.mean(axis=0, keepdims=True)
        polygon = centroid + (polygon - centroid) * scale
        poly_int = polygon.astype(np.int32)

        fill = np.clip(colors[idx] * fill_mix + echo_shift * (1.0 - fill_mix), 0, 255).astype(np.uint8)
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
    chroma_key_color: tuple[int, int, int] | None,
) -> None:
    if family == "circles":
        _draw_circles(surface, layers, style, snapshot, style_mix, beat_pulse, chroma_key_color)
        return
    if family == "scribbles":
        _draw_scribbles(surface, layers, style, snapshot, beat_pulse, time_phase, chroma_key_color)
        return
    if family == "lines":
        _draw_lines(surface, layers, style, snapshot, beat_pulse, time_phase, chroma_key_color)
        return
    if family == "geometrics":
        _draw_geometrics(surface, layers, style, snapshot, style_mix, beat_pulse, time_phase, chroma_key_color)
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


def _rebuild_bases(config: RenderConfig, rng: np.random.Generator, points: np.ndarray, base_points: np.ndarray) -> np.ndarray:
    if config.reorg_mode == "split":
        direction = np.sign(points[:, 0] - config.width * 0.5).astype(np.float32)
        direction[direction == 0.0] = 1.0
        rebuilt = base_points.copy()
        rebuilt[:, 0] += direction * config.width * 0.08
        rebuilt[:, 1] += rng.normal(0.0, config.height * 0.03, size=len(rebuilt)).astype(np.float32)
        return np.clip(rebuilt, [8.0, 8.0], [config.width - 8.0, config.height - 8.0])
    return _build_seed_points(config.point_count, config.width, config.height, rng)


def _reorg_impulse(
    config: RenderConfig,
    rng: np.random.Generator,
    points: np.ndarray,
    snapshot: AudioSnapshot,
) -> np.ndarray:
    impulse = rng.normal(0.0, 1.0, size=points.shape).astype(np.float32)
    strength = 18.0 + 56.0 * snapshot.percussive_flux
    if config.reorg_mode == "swirl":
        center = np.array([config.width * 0.5, config.height * 0.5], dtype=np.float32)
        delta = points - center
        impulse[:, 0] = -delta[:, 1]
        impulse[:, 1] = delta[:, 0]
        impulse /= np.maximum(np.linalg.norm(impulse, axis=1, keepdims=True), 1.0)
        strength *= 1.35
    elif config.reorg_mode == "split":
        impulse[:, 0] = np.sign(points[:, 0] - config.width * 0.5)
        impulse[:, 1] = rng.normal(0.0, 0.25, size=len(points)).astype(np.float32)
        strength *= 1.20
    elif config.reorg_mode == "shockwave":
        center = np.array([config.width * 0.5, config.height * 0.5], dtype=np.float32)
        impulse = points - center
        impulse /= np.maximum(np.linalg.norm(impulse, axis=1, keepdims=True), 1.0)
        strength *= 1.45
    return impulse * strength


def render_project(config: RenderConfig) -> None:
    if not config.audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {config.audio_path}")

    timeline = analyze_audio(
        audio_path=config.audio_path,
        hop_length=config.hop_length,
        n_fft=config.n_fft,
        duration_limit=config.duration_limit,
    )
    total_frames = max(1, math.ceil(timeline.duration * config.fps))
    rng = np.random.default_rng(config.seed)
    points = _build_seed_points(config.point_count, config.width, config.height, rng)
    base_points = points.copy()
    velocity = np.zeros_like(points, dtype=np.float32)

    style_a = _resolve_style(config.style_a)
    style_b = _resolve_style(config.style_b)
    graphic_cycle = _resolve_graphic_cycle(config.graphic_cycle)
    beats_per_switch = max(1, config.beats_per_switch)

    pygame.init()
    flags = 0 if config.preview else pygame.HIDDEN
    screen = pygame.display.set_mode((config.width, config.height), flags)
    surface = pygame.Surface((config.width, config.height))
    clock = pygame.time.Clock()

    ffmpeg = _open_ffmpeg(config.output_path, config.audio_path, config.width, config.height, config.fps)
    if ffmpeg.stdin is None:
        raise RuntimeError("Failed to open FFmpeg stdin.")

    try:
        previous_onset = 0.0
        for frame_idx in range(total_frames):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            snapshot = _sample_snapshot(timeline, frame_idx, total_frames)
            style_mix = _compute_style_mix(frame_idx, total_frames, snapshot, config.morph_rate)
            style = _activate_style(style_a, style_b, style_mix)
            time_phase = frame_idx / max(config.fps, 1)
            beat_index = timeline.beat_index_at_time(time_phase)
            beat_pulse = timeline.beat_pulse_at_time(time_phase)
            graphic_index = (max(beat_index, 0) // beats_per_switch) % len(graphic_cycle)
            graphic_family = graphic_cycle[graphic_index]

            trigger = snapshot.onset > 0.62 and (snapshot.onset - previous_onset) > 0.08
            previous_onset = snapshot.onset

            drift = (base_points - points) * (0.010 + snapshot.harmonic_ratio * 0.026 + config.overlap * 0.008)
            velocity += drift

            if config.cymatic_mode:
                cymatic_strength = 0.010 + snapshot.contrast * 0.022 + style_mix * 0.010
                velocity += _cymatic_force(points, config.width, config.height, config.plate_mode, cymatic_strength)

            if trigger:
                velocity += _reorg_impulse(config, rng, points, snapshot)
                base_points = _rebuild_bases(config, rng, points, base_points)

            center = np.array([config.width * 0.5, config.height * 0.5], dtype=np.float32)
            delta = points - center
            swirl = np.empty_like(points, dtype=np.float32)
            swirl[:, 0] = -delta[:, 1]
            swirl[:, 1] = delta[:, 0]
            swirl /= np.maximum(np.linalg.norm(swirl, axis=1, keepdims=True), 1.0)

            drift_wave = np.empty_like(points, dtype=np.float32)
            drift_wave[:, 0] = np.sin(points[:, 1] * 0.012 + time_phase * 1.8)
            drift_wave[:, 1] = np.cos(points[:, 0] * 0.012 - time_phase * 1.5)

            velocity += swirl * (0.03 + snapshot.brightness * 0.12 + style.swirl_bias + snapshot.harmonic_ratio * 0.08)
            velocity += drift_wave * (config.overlap * 0.10 + snapshot.contrast * 0.06)
            velocity *= 0.90 - snapshot.rms * 0.08

            points += velocity
            points[:, 0] = np.clip(points[:, 0], 8.0, config.width - 8.0)
            points[:, 1] = np.clip(points[:, 1], 8.0, config.height - 8.0)

            family_overlap = config.overlap + beat_pulse * 0.18
            _frame_background(surface, style, snapshot, style_mix, time_phase, config.chroma_key_color)
            layers = _prepare_layers(points, style, snapshot, config.layer_count, family_overlap, time_phase)
            _draw_graphic_family(
                surface,
                graphic_family,
                layers,
                style,
                snapshot,
                style_mix,
                beat_pulse,
                family_overlap,
                time_phase,
                config.chroma_key_color,
            )

            if config.preview:
                screen.blit(surface, (0, 0))
                pygame.display.flip()
                clock.tick(config.fps)

            ffmpeg.stdin.write(_surface_to_frame(surface).tobytes())
    finally:
        ffmpeg.stdin.close()
        code = ffmpeg.wait()
        pygame.quit()
        if code != 0:
            raise RuntimeError(f"FFmpeg exited with status {code}.")
