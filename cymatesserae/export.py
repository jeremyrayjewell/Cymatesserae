from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pygame


def surface_to_frame(surface: pygame.Surface) -> np.ndarray:
    rgb = pygame.surfarray.array3d(surface)
    return np.transpose(rgb, (1, 0, 2)).copy()


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg was not found on PATH. Install FFmpeg and make sure `ffmpeg` is available from the command line.")


def open_ffmpeg(output_path: Path, audio_path: Path, width: int, height: int, fps: int) -> subprocess.Popen[bytes]:
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
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg was not found on PATH. Install FFmpeg and make sure `ffmpeg` is available from the command line.") from exc
