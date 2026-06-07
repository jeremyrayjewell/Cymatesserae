from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .project_io import graphic_layer_from_dict, load_project_file
from .shared import normalize_output_path, normalize_video_dimension, parse_chroma_key_color

if TYPE_CHECKING:
    from .renderer import GraphicLayerConfig, RenderConfig


def load_graphic_layers(path: Path | None) -> tuple[GraphicLayerConfig, ...]:
    if path is None:
        return ()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(graphic_layer_from_dict(item, path.parent) for item in raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cymatesserae",
        description="Transform audio into cymatic Voronoi tessellations.",
    )
    parser.add_argument("audio", type=Path, nargs="?", help="Path to the input audio file.")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the desktop control panel instead of running directly from the CLI.",
        default=None,
    )
    parser.add_argument(
        "--project",
        type=Path,
        help="Optional JSON project file containing reproducible render and layer settings.",
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="MP4 output path.",
        default=None,
    )
    parser.add_argument("--width", type=int, default=None, help="Video width.")
    parser.add_argument("--height", type=int, default=None, help="Video height.")
    parser.add_argument("--fps", type=int, default=None, help="Output frames per second.")
    parser.add_argument(
        "--points",
        type=int,
        default=None,
        help="Number of Voronoi seed points.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview the animation in a window while rendering.",
        default=None,
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Open a lighter live preview window without exporting frames to FFmpeg.",
        default=None,
    )
    parser.add_argument(
        "--cymatic",
        action="store_true",
        help="Enable cymatic nodal-line attraction mode.",
        default=None,
    )
    parser.add_argument(
        "--plate-mode",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        default=None,
        help="Plate vibration mode pair for cymatic attraction.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=None,
        help="Hop length used for librosa feature extraction.",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=None,
        help="FFT window size used for the STFT.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Optional limit in seconds for quick renders.",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic tessellations.",
    )
    parser.add_argument(
        "--style",
        default=None,
        choices=["ceramic", "neon", "lava", "glass", "monolith"],
        help="Primary visual style preset.",
    )
    parser.add_argument(
        "--style-b",
        default=None,
        choices=["ceramic", "neon", "lava", "glass", "monolith"],
        help="Secondary visual style preset used for morphing.",
    )
    parser.add_argument(
        "--morph-rate",
        type=float,
        default=None,
        help="Rate at which the renderer drifts between style presets.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="Number of overlapping tessellation layers.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="Amount of spatial overlap and echo between layers.",
    )
    parser.add_argument(
        "--pattern-layout",
        default=None,
        choices=["flow", "grid"],
        help="Whether patterns move freely or stay more locked to a grid.",
    )
    parser.add_argument(
        "--grid-strength",
        type=float,
        default=None,
        help="How strongly grid layout resists drift and stays stationary.",
    )
    parser.add_argument(
        "--geometry-rigidity",
        type=float,
        default=None,
        help="How rigidly points lock to their grid positions in grid layout.",
    )
    parser.add_argument(
        "--layer-rigidity",
        type=float,
        default=None,
        help="How much overlapping layers resist shear, offset, and pulse in grid layout.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=None,
        help="How tightly grid and custom elements overlap when using stationary layouts.",
    )
    parser.add_argument(
        "--grid-columns",
        type=int,
        default=None,
        help="Optional explicit number of grid columns when using grid layout. Use 0 to auto-fit.",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=None,
        help="Optional explicit number of grid rows when using grid layout. Use 0 to auto-fit.",
    )
    parser.add_argument(
        "--grid-pattern",
        default=None,
        choices=["rect", "brick", "hex", "diamond"],
        help="Cell arrangement to use when pattern layout is grid.",
    )
    parser.add_argument(
        "--cell-alternation",
        default=None,
        choices=["none", "orientation", "color", "both"],
        help="Alternate inverse orientation and/or inverse color across grid cells.",
    )
    parser.add_argument(
        "--reorg-mode",
        default=None,
        choices=["burst", "swirl", "split", "shockwave"],
        help="How percussive hits reorganize tesserae points.",
    )
    parser.add_argument(
        "--graphic-cycle",
        default=None,
        help="Comma-separated list of graphic families to switch between on the beat.",
    )
    parser.add_argument(
        "--beats-per-switch",
        type=int,
        default=None,
        help="How many beats each graphic family stays active before switching.",
    )
    parser.add_argument(
        "--stack-interaction",
        default=None,
        choices=["none", "crossfade", "shuffle", "pulse", "duck", "spotlight"],
        help="How enabled channels interact while stacked together, including audio-responsive modes.",
    )
    parser.add_argument(
        "--chroma-key-color",
        type=parse_chroma_key_color,
        help="Optional solid background color for chroma keying, as RRGGBB or #RRGGBB.",
        default=None,
    )
    parser.add_argument(
        "--custom-element",
        type=Path,
        nargs="+",
        help="One or more painted sprite assets used by the custom graphic family.",
        default=None,
    )
    parser.add_argument(
        "--graphics-config",
        type=Path,
        help="Optional JSON file describing per-graphic layer settings from the GUI.",
        default=None,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.gui:
        from .gui import launch_gui

        launch_gui()
        return 0

    if args.project is not None:
        try:
            project_config = load_project_file(args.project)
        except FileNotFoundError:
            raise SystemExit(f"Could not load project file: file not found: {args.project}")
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Could not load project file: {exc}")
    else:
        project_config = None
    audio_arg = args.audio if args.audio is not None else (project_config.audio_path if project_config is not None else None)

    if audio_arg is None:
        raise SystemExit("Audio file is required unless you use --gui.")
    if audio_arg.name.lower() == "preview" and not audio_arg.exists() and not args.preview:
        raise SystemExit(
            "`preview` is a flag, not a subcommand.\n"
            "Use: python -m cymatesserae <audio-file> --preview [options]\n"
            "Or launch the GUI with: python -m cymatesserae --gui"
        )

    from .renderer import RenderConfig, render_project

    base = project_config or RenderConfig(audio_path=audio_arg, output_path=Path("cymatesserae_output.mp4"))
    preview_only = bool(args.preview_only) if args.preview_only is not None else base.preview_only
    preview = (bool(args.preview) if args.preview is not None else base.preview) or preview_only
    graphic_layers = load_graphic_layers(args.graphics_config) if args.graphics_config is not None else base.graphic_layers
    graphic_cycle = base.graphic_cycle
    if args.graphic_cycle is not None:
        graphic_cycle = tuple(part.strip() for part in args.graphic_cycle.split(",") if part.strip())

    config = RenderConfig(
        audio_path=audio_arg,
        output_path=normalize_output_path(args.output if args.output is not None else base.output_path),
        width=normalize_video_dimension(args.width if args.width is not None else base.width),
        height=normalize_video_dimension(args.height if args.height is not None else base.height),
        fps=args.fps if args.fps is not None else base.fps,
        point_count=args.points if args.points is not None else base.point_count,
        preview=preview,
        preview_only=preview_only,
        cymatic_mode=bool(args.cymatic) if args.cymatic is not None else base.cymatic_mode,
        plate_mode=tuple(args.plate_mode) if args.plate_mode is not None else base.plate_mode,
        hop_length=args.hop_length if args.hop_length is not None else base.hop_length,
        n_fft=args.n_fft if args.n_fft is not None else base.n_fft,
        duration_limit=args.duration if args.duration is not None else base.duration_limit,
        seed=args.seed if args.seed is not None else base.seed,
        style_a=args.style if args.style is not None else base.style_a,
        style_b=args.style_b if args.style_b is not None else base.style_b,
        morph_rate=args.morph_rate if args.morph_rate is not None else base.morph_rate,
        layer_count=args.layers if args.layers is not None else base.layer_count,
        overlap=args.overlap if args.overlap is not None else base.overlap,
        pattern_layout=args.pattern_layout if args.pattern_layout is not None else base.pattern_layout,
        grid_strength=args.grid_strength if args.grid_strength is not None else base.grid_strength,
        geometry_rigidity=args.geometry_rigidity if args.geometry_rigidity is not None else base.geometry_rigidity,
        layer_rigidity=args.layer_rigidity if args.layer_rigidity is not None else base.layer_rigidity,
        tile_overlap=args.tile_overlap if args.tile_overlap is not None else base.tile_overlap,
        grid_columns=args.grid_columns if args.grid_columns is not None else base.grid_columns,
        grid_rows=args.grid_rows if args.grid_rows is not None else base.grid_rows,
        grid_pattern=args.grid_pattern if args.grid_pattern is not None else base.grid_pattern,
        cell_alternation=args.cell_alternation if args.cell_alternation is not None else base.cell_alternation,
        reorg_mode=args.reorg_mode if args.reorg_mode is not None else base.reorg_mode,
        graphic_cycle=graphic_cycle,
        beats_per_switch=args.beats_per_switch if args.beats_per_switch is not None else base.beats_per_switch,
        stack_interaction=args.stack_interaction if args.stack_interaction is not None else base.stack_interaction,
        chroma_key_color=args.chroma_key_color if args.chroma_key_color is not None else base.chroma_key_color,
        custom_element_paths=tuple(args.custom_element) if args.custom_element is not None else base.custom_element_paths,
        graphic_layers=graphic_layers,
    )
    render_project(config)
    return 0
