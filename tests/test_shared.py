from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from cymatesserae.shared import (
    normalize_hex_color_text,
    normalize_output_path,
    normalize_video_dimension,
    parse_chroma_key_color,
)


def test_normalize_output_path_preserves_mp4_suffix() -> None:
    assert normalize_output_path(Path("clip.mp4")) == Path("clip.mp4")


def test_normalize_output_path_replaces_other_suffix() -> None:
    assert normalize_output_path(Path("clip.mov")) == Path("clip.mp4")


def test_normalize_output_path_adds_mp4_when_missing() -> None:
    assert normalize_output_path(Path("clip")) == Path("clip.mp4")


def test_normalize_video_dimension_makes_values_even() -> None:
    assert normalize_video_dimension(1279) == 1280
    assert normalize_video_dimension(720) == 720
    assert normalize_video_dimension(1) == 2


def test_normalize_hex_color_text_accepts_hash_prefix() -> None:
    assert normalize_hex_color_text("#00FF00") == "00ff00"


def test_normalize_hex_color_text_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        normalize_hex_color_text("xyz")


def test_parse_chroma_key_color_parses_rgb_triplet() -> None:
    assert parse_chroma_key_color("#ff00aa") == (255, 0, 170)


def test_parse_chroma_key_color_raises_argument_error_for_blank() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_chroma_key_color("")
