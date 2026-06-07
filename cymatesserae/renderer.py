from __future__ import annotations

import math
import subprocess

import numpy as np
import pygame

from .audio_analysis import AudioFeatureTimeline, AudioSnapshot, analyze_audio, sample_snapshot
from .config import GraphicLayerConfig, LayerRuntimeState, RenderConfig
from .drawing import apply_channel_composite, draw_graphic_family
from .export import open_ffmpeg, require_ffmpeg, surface_to_frame
from .geometry import (
    GRAPHIC_TYPES,
    build_layer_runtime,
    cymatic_force,
    effective_graphic_layers,
    rebuild_bases,
    reorg_impulse,
    resolve_track_families,
    prepare_layers,
)
from .styles import ActiveStyle, STYLE_PRESETS, StylePreset, activate_style, compute_style_mix, frame_background


def _emit_status(message: str) -> None:
    print(f"[cymatesserae] {message}", flush=True)


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

    graphic_layers = effective_graphic_layers(config)
    runtimes = [build_layer_runtime(layer, config, config.seed + idx * 997) for idx, layer in enumerate(graphic_layers)]
    _emit_status(f"Prepared {len(runtimes)} active channel(s) for rendering.")

    ffmpeg: subprocess.Popen[bytes] | None = None
    if not config.preview_only:
        require_ffmpeg()
        _emit_status(f"Opening FFmpeg export pipeline for {config.output_path.name}...")
        ffmpeg = open_ffmpeg(config.output_path, config.audio_path, config.width, config.height, config.fps)
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

            snapshot = sample_snapshot(timeline, frame_idx, total_frames)
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
            first_style_mix = compute_style_mix(frame_idx, total_frames, snapshot, first_layer.morph_rate)
            first_style = activate_style(first_runtime.style_a, first_runtime.style_b, first_style_mix)
            frame_background(surface, first_style, snapshot, first_style_mix, time_phase, config.chroma_key_color)

            for runtime, channel_gain in channel_plan:
                layer = runtime.layer
                style_mix = compute_style_mix(frame_idx, total_frames, snapshot, layer.morph_rate)
                style = activate_style(runtime.style_a, runtime.style_b, style_mix)
                response_gain = max(0.1, float(layer.response_gain))
                beat_pulse = np.clip(base_beat_pulse * response_gain, 0.0, 1.6)
                active_families = resolve_track_families(layer, beat_index)

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
                    velocity += cymatic_force(points, config.width, config.height, config.plate_mode, cymatic_strength)

                if trigger and not is_grid_layout:
                    layer_rng = runtime.rng
                    velocity += reorg_impulse(config, layer.reorg_mode, layer_rng, points, snapshot)
                    base_points = rebuild_bases(config, layer.reorg_mode, layer_rng, points, base_points)
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
                layers = prepare_layers(
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
                    draw_graphic_family(
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
                    apply_channel_composite(
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
                try:
                    ffmpeg.stdin.write(surface_to_frame(surface).tobytes())
                except BrokenPipeError as exc:
                    stderr_text = ""
                    if ffmpeg.stderr is not None:
                        stderr_text = ffmpeg.stderr.read().decode("utf-8", errors="replace").strip()
                    detail = f"\n\nFFmpeg output:\n{stderr_text}" if stderr_text else ""
                    raise RuntimeError(f"FFmpeg stopped accepting video frames.{detail}") from exc
    finally:
        code = 0
        stderr_text = ""
        if ffmpeg is not None:
            if ffmpeg.stdin is not None:
                try:
                    ffmpeg.stdin.close()
                except BrokenPipeError:
                    pass
            _, stderr_bytes = ffmpeg.communicate()
            code = ffmpeg.returncode
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip() if stderr_bytes else ""
        pygame.quit()
        if code != 0:
            detail = f"\n\nFFmpeg output:\n{stderr_text}" if stderr_text else ""
            raise RuntimeError(f"FFmpeg exited with status {code}.{detail}")
        _emit_status("Render session finished cleanly.")


__all__ = [
    "ActiveStyle",
    "AudioFeatureTimeline",
    "AudioSnapshot",
    "GRAPHIC_TYPES",
    "GraphicLayerConfig",
    "LayerRuntimeState",
    "RenderConfig",
    "STYLE_PRESETS",
    "StylePreset",
    "analyze_audio",
    "open_ffmpeg",
    "render_project",
]
