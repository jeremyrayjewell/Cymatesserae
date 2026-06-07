from __future__ import annotations

from cymatesserae.audio_analysis import analyze_audio


def test_analyze_audio_smoke(generated_wav) -> None:
    timeline = analyze_audio(generated_wav, hop_length=512, n_fft=2048, duration_limit=None)

    assert 0.9 <= timeline.duration <= 1.1
    assert timeline.sr == 44_100
    assert timeline.bpm > 0.0
    assert len(timeline.beat_times) >= 1

    for key in ("bass", "brightness", "contrast", "harmonic_ratio", "percussive", "percussive_flux", "onset", "rms"):
        values = timeline.features[key]
        assert values.ndim == 1
        assert len(values) >= 1
