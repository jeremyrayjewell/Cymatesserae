from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter


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


def sample_snapshot(timeline: AudioFeatureTimeline, frame_idx: int, total_frames: int) -> AudioSnapshot:
    return AudioSnapshot(
        bass=timeline.sample("bass", frame_idx, total_frames),
        brightness=timeline.sample("brightness", frame_idx, total_frames),
        contrast=timeline.sample("contrast", frame_idx, total_frames),
        harmonic_ratio=timeline.sample("harmonic_ratio", frame_idx, total_frames),
        percussive_flux=timeline.sample("percussive_flux", frame_idx, total_frames),
        onset=timeline.sample("onset", frame_idx, total_frames),
        rms=timeline.sample("rms", frame_idx, total_frames),
    )
