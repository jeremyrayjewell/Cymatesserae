from __future__ import annotations

import argparse
import os
from pathlib import Path


APP_DIR_NAME = "Cymatesserae"


def parse_chroma_key_color(value: str) -> tuple[int, int, int]:
    text = normalize_hex_color_text(value)
    if not text:
        raise argparse.ArgumentTypeError("Chroma key color must be a 6-digit hex value like 00ff00 or #00ff00.")
    return tuple(int(text[idx : idx + 2], 16) for idx in (0, 2, 4))


def normalize_output_path(path: Path) -> Path:
    if path.suffix.lower() == ".mp4":
        return path
    if path.suffix:
        return path.with_suffix(".mp4")
    return path.with_name(f"{path.name}.mp4")


def normalize_mp4_path_text(path_text: str) -> str:
    text = path_text.strip()
    if not text:
        return "cymatesserae_output.mp4"
    return str(normalize_output_path(Path(text)))


def normalize_video_dimension(value: int) -> int:
    number = max(2, int(value))
    if number % 2 == 0:
        return number
    return number + 1


def normalize_hex_color_text(value: str) -> str:
    text = value.strip().lower()
    if not text:
        return ""
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        raise ValueError("Color must be a 6-digit hex value like 00ff00.")
    try:
        int(text, 16)
    except ValueError as exc:
        raise ValueError("Color must be a valid 6-digit hex value like 00ff00.") from exc
    return text


def get_app_state_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        return Path(base) / APP_DIR_NAME
    return Path.home() / ".cymatesserae"

