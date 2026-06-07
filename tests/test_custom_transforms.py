from __future__ import annotations

import os

import pygame
import pytest

from cymatesserae.geometry import (
    CUSTOM_MAX_SURFACE_AREA_MULTIPLIER,
    CUSTOM_MAX_SURFACE_DIMENSION_MULTIPLIER,
    CUSTOM_SCALE_CACHE_LIMIT,
    CUSTOM_TRANSFORM_CACHE_LIMIT,
    clamp_custom_surface_size,
    get_transformed_custom_surface,
)


@pytest.fixture(scope="module", autouse=True)
def pygame_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.display.set_mode((1, 1))
    yield
    pygame.quit()


def _make_sprite() -> pygame.Surface:
    sprite = pygame.Surface((64, 64))
    sprite.fill((255, 255, 255))
    return sprite


def test_clamp_custom_surface_size_limits_large_requests() -> None:
    width, height = clamp_custom_surface_size(100_000, 80_000, 320, 180)

    assert width <= int(320 * CUSTOM_MAX_SURFACE_DIMENSION_MULTIPLIER)
    assert height <= int(180 * CUSTOM_MAX_SURFACE_DIMENSION_MULTIPLIER)
    assert width * height <= int(320 * 180 * CUSTOM_MAX_SURFACE_AREA_MULTIPLIER)


def test_repeated_custom_transforms_keep_caches_bounded() -> None:
    sprite = _make_sprite()
    inverted = sprite.copy()
    scale_cache: dict[tuple[int, int, int, bool], pygame.Surface] = {}
    transform_cache: dict[tuple[int, int, int, int, bool, bool], pygame.Surface] = {}

    for idx in range(CUSTOM_TRANSFORM_CACHE_LIMIT * 3):
        get_transformed_custom_surface(
            0,
            90 + (idx % 11) * 8,
            70 + (idx % 9) * 6,
            320,
            180,
            float(idx * 17),
            False,
            bool(idx % 2),
            [sprite],
            [inverted],
            scale_cache,
            transform_cache,
            True,
        )

    assert len(scale_cache) <= CUSTOM_SCALE_CACHE_LIMIT
    assert len(transform_cache) <= CUSTOM_TRANSFORM_CACHE_LIMIT


def test_custom_transform_falls_back_when_rotate_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    sprite = _make_sprite()
    inverted = sprite.copy()
    scale_cache: dict[tuple[int, int, int, bool], pygame.Surface] = {}
    transform_cache: dict[tuple[int, int, int, int, bool, bool], pygame.Surface] = {}

    def fail_rotate(_surface: pygame.Surface, _angle: float) -> pygame.Surface:
        raise pygame.error("Out of memory")

    monkeypatch.setattr(pygame.transform, "rotate", fail_rotate)

    result = get_transformed_custom_surface(
        0,
        96,
        96,
        320,
        180,
        45.0,
        False,
        False,
        [sprite],
        [inverted],
        scale_cache,
        transform_cache,
        True,
    )

    assert isinstance(result, pygame.Surface)
    assert result.get_width() > 0
    assert result.get_height() > 0
    assert transform_cache == {}
