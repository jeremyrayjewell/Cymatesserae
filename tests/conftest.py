from __future__ import annotations

import math
import wave
from pathlib import Path

import pytest


def write_test_wav(
    path: Path,
    *,
    duration_s: float = 1.0,
    sample_rate: int = 44_100,
    frequency: float = 440.0,
    amplitude: float = 0.35,
) -> Path:
    frame_count = max(1, int(sample_rate * duration_s))
    pcm = bytearray()
    for idx in range(frame_count):
        sample = amplitude * math.sin(2.0 * math.pi * frequency * (idx / sample_rate))
        value = max(-32767, min(32767, int(sample * 32767)))
        pcm.extend(int(value).to_bytes(2, byteorder="little", signed=True))

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(pcm))

    return path


@pytest.fixture
def generated_wav(tmp_path: Path) -> Path:
    return write_test_wav(tmp_path / "tone.wav")
