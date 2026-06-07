from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from cymatesserae.config import RenderConfig
from cymatesserae.renderer import render_project


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not available on PATH")
def test_renderer_export_smoke(monkeypatch, tmp_path: Path, generated_wav: Path) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    output_path = tmp_path / "smoke.mp4"

    config = RenderConfig(
        audio_path=generated_wav,
        output_path=output_path,
        width=160,
        height=90,
        fps=6,
        point_count=20,
        duration_limit=0.5,
        cymatic_mode=True,
    )

    render_project(config)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
